"""Pins for task #163: the era probe must never enter torch.compile.

The probe is an auxiliary eval that runs INSIDE the trainer process, and the
trainer's ``self.model`` is a ``torch.compile`` ``OptimizedModule``. On
2026-08-05 (resumed trial d2003) the probe's row-cap shape was new to
inductor, which began a fresh cudagraph capture; CUBLAS threw
``CUBLAS_STATUS_INTERNAL_ERROR`` mid-capture
(``torch.AcceleratorError: operation failed due to a previous error during
capture`` / ``cudaErrorStreamCaptureInvalidated``) and the aborted capture left
``cudagraph_trees`` mid-recording, so every later probe in that process died
instantly with ``RuntimeError: beginAllocateToPool: already recording to
mempool_id``. Probes read nan for the rest of the session.

These tests are CPU-only and use a COUNTING BACKEND rather than asserting on
CUDA behaviour: the quantity that matters is "did the probe forward enter
dynamo at all", and a compile that never happens cannot capture. The negative
control is
:func:`test_the_decorator_alone_does_not_save_us` — it shows the counter DOES
fire on the compiled wrapper, so a zero above is a measurement and not a dead
instrument.
"""
from __future__ import annotations

import importlib
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.eval import era_probe as ep
from chess_anti_engine.eval import puzzles
from chess_anti_engine.eval.puzzles import Puzzle, PuzzleSuite
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.moves.encode import MODEL_POLICY_SIZE, compact_policy_index
from chess_anti_engine.tune import trainable_phases
from chess_anti_engine.tune.trainable_phases import _run_era_probes_if_due
from chess_anti_engine.tune.trial_config import TrialConfig
# One probe-set fixture for the repo: its `illegal_regret` default encodes a
# production fact (PR #315, finding 2) that a second copy would drift from.
from tests.test_era_forgetting_probe import N_LEGAL, PLANES, POLICY, _probe_arrays


class _CompileCounter:
    """A torch.compile backend that counts graphs handed to it."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, gm: Any, example_inputs: Any) -> Any:
        _ = example_inputs
        self.n += 1
        return gm.forward


class _TinyNet(torch.nn.Module):
    """Smallest module carrying the production output keys."""

    def __init__(self) -> None:
        super().__init__()
        self.pol = torch.nn.Linear(PLANES * 8 * 8, POLICY)
        self.val = torch.nn.Linear(PLANES * 8 * 8, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        return {"policy_own": self.pol(flat), "wdl": self.val(flat)}


def _probe_set(n: int = 6) -> ep.ProbeSet:
    rng = np.random.default_rng(0)
    arrays = _probe_arrays(
        n=n,
        regret=rng.random((n, N_LEGAL)).astype(np.float32),
        wdl_target=rng.integers(0, 3, size=n),
    )
    return ep.ProbeSet(
        label=ep.PROBE_ERA, path="<synthetic>", arrays=arrays,
        n_rows=n, n_policy_rows=n, digest="0" * 16, provenance={},
    )


def _tc(**kw: Any) -> TrialConfig:
    return TrialConfig.from_dict(
        {"era_probe_interval": 1, "era_probe_batch_size": 4, **kw})


# ---------------------------------------------------------------------------
# (a) the probe forward runs eager


def test_scoring_a_compiled_model_enters_dynamo_zero_times() -> None:
    """MUTATION: score through ``model`` instead of ``eager_module(model)``
    in ``score_probe_set`` — the counter then reads >= 1 and this goes RED.

    Counts COMPILES, not wall time: the wedge is caused by entering inductor
    at a new shape, and the only observation that proves it cannot happen is
    that no graph was ever handed to a backend. Three different batch shapes
    are forced (the probe's last chunk is short, which is exactly the new
    shape that tripped the live run) because a single shape could be served
    from an existing guard.
    """
    counter = _CompileCounter()
    net = _TinyNet()
    compiled = cast("torch.nn.Module", torch.compile(net, backend=counter))
    assert hasattr(compiled, "_orig_mod"), "torch.compile did not wrap the module"
    assert ep.eager_module(compiled) is net

    # n=6 with batch 4 gives chunks of 4 and 2 -> two distinct shapes.
    reading = ep.score_probe_set(compiled, _probe_set(6), device="cpu", batch_size=4)
    assert counter.n == 0, f"probe scoring compiled {counter.n} graph(s)"
    assert np.isfinite(reading.policy_eregret)
    assert np.isfinite(reading.value_err)
    assert reading.n_rows == 6


def test_the_eager_module_is_what_actually_executes() -> None:
    """Independent of the counter: patch a marker onto the ORIG module's
    forward and assert the probe's numbers came through it.

    The counter answers "did anything compile"; this answers "whose forward
    ran, and was it running under a tracer". Both are needed — a backend that
    silently fell back to eager would satisfy the counter alone.

    ``torch.compiler.is_compiling()`` is the discriminating part: a marker that
    merely records the shapes it saw is NOT mutation-sensitive here, because
    dynamo executes the patched forward while TRACING it and the shapes come
    out identical.
    """
    net = _TinyNet()
    compiled = cast(
        "torch.nn.Module", torch.compile(net, backend=_CompileCounter()))
    calls: list[tuple[int, bool]] = []
    orig_forward = net.forward

    def _marked(x: torch.Tensor) -> dict[str, torch.Tensor]:
        calls.append((int(x.shape[0]), bool(torch.compiler.is_compiling())))
        return orig_forward(x)

    net.forward = _marked
    try:
        ep.score_probe_set(compiled, _probe_set(6), device="cpu", batch_size=4)
    finally:
        net.forward = orig_forward
    assert calls == [(4, False), (2, False)], (
        f"eager forward saw {calls}, expected the probe chunks with no tracer")


def test_the_decorator_alone_does_not_save_us() -> None:
    """NEGATIVE CONTROL for the two tests above, and the reason the unwrap —
    not ``torch.compiler.disable`` — is the load-bearing part of the fix.

    ``OptimizedModule.__call__`` installs its own dynamo context, so calling
    it from a ``torch.compiler.disable``d frame still compiles. If this ever
    stops firing, the counter has gone dead and the zeros above mean nothing.
    """
    counter = _CompileCounter()
    net = _TinyNet()
    compiled = cast("torch.nn.Module", torch.compile(net, backend=counter))
    with torch.inference_mode():
        ep._probe_forward(compiled, torch.zeros(3, PLANES, 8, 8))
    assert counter.n >= 1, "the compile counter never fires — it is not an instrument"


def test_the_disable_wrapper_is_live_not_decorative() -> None:
    """``_probe_forward`` is built by CALLING ``torch.compiler.disable`` and
    re-annotating the result, which is exactly the shape a knob takes when it
    is accepted and then silently ignored. So drive it from a compiled frame
    and check the eager module ran OUTSIDE the tracer.

    MUTATION: replace the ``torch.compiler.disable(...)`` call with the bare
    function — the marker then records ``is_compiling() is True``.
    """
    net = _TinyNet()
    seen: list[bool] = []
    orig_forward = net.forward

    def _marked(x: torch.Tensor) -> dict[str, torch.Tensor]:
        seen.append(bool(torch.compiler.is_compiling()))
        return orig_forward(x)

    net.forward = _marked

    def _outer(x: torch.Tensor) -> torch.Tensor:
        return ep._probe_forward(net, x)["wdl"]

    try:
        compiled_outer = torch.compile(_outer, backend=_CompileCounter())
        compiled_outer(torch.zeros(2, PLANES, 8, 8))
    finally:
        net.forward = orig_forward
    assert seen == [False], f"the eager forward ran under a tracer: {seen}"


@pytest.mark.parametrize("depth", ["bare", "compiled", "averaged_compiled"])
def test_the_unwrap_reaches_an_uncompiled_module_at_every_nesting(depth: str) -> None:
    """``AveragedModel`` nests the compiled module (``module._orig_mod.*``),
    which is exactly the shape PR #267's ``removeprefix`` bug could not see.
    The unwrap is by ATTRIBUTE and peels one layer at a time, so it must land
    on a module with no ``_orig_mod`` at every depth — including none.
    """
    net = _TinyNet()
    model: torch.nn.Module = net
    if depth != "bare":
        model = cast(
            "torch.nn.Module", torch.compile(net, backend=_CompileCounter()))
    if depth == "averaged_compiled":
        model = torch.optim.swa_utils.AveragedModel(model)
    inner = ep.eager_module(model)
    assert not hasattr(inner, "_orig_mod"), f"{depth}: unwrap stopped on a wrapper"
    assert isinstance(inner, _TinyNet)
    if depth != "averaged_compiled":
        # AveragedModel deep-copies at init, so identity only holds when it is
        # not in the chain; the type check above is the assertion there.
        assert inner is net


# ---------------------------------------------------------------------------
# the same rule, one call site over: puzzle eval


class _PuzzleNet(torch.nn.Module):
    """Stub carrying both heads and the encoding attributes puzzle eval reads.

    ``encode_position_for_model`` refuses to guess the input encoding
    (rl_loop_audit M11), and ``OptimizedModule`` forwards attribute access to
    ``_orig_mod``, so declaring them here covers the compiled case too.
    """

    def __init__(self, *, move_idx: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.move_idx = move_idx
        self.input_history_encoding = "lc0_root_legacy_meta"
        self.input_extra_features = "v2_threats"

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(x.shape[0])
        policy = torch.zeros((n, MODEL_POLICY_SIZE), dtype=torch.float32)
        policy[:, self.move_idx] = 100.0
        return {"policy_own": policy, "wdl": torch.zeros((n, 3), dtype=torch.float32)}


@pytest.mark.parametrize(
    ("fn_name", "n_puzzles", "batch_size"),
    [("run_policy_sequence_eval", 3, 2), ("run_value_head_puzzle_eval", 1, 8)],
)
def test_puzzle_eval_forwards_run_eager_on_a_compiled_model(
    fn_name: str, n_puzzles: int, batch_size: int,
) -> None:
    """MUTATION: score through ``model`` instead of ``eager_module(model)`` in
    ``eval/puzzles.py`` — the counter then reads >= 1 and the marker records
    ``is_compiling() is True``.

    Same defect class as the era probe, one call site over: both functions
    chunk at ``batch_size`` with a SHORT FINAL CHUNK, which is exactly the new
    shape that sent the trainer's compiled model into a cudagraph capture. The
    parameters give each path at least two distinct shapes (3 boards at batch 2
    -> 2 then 1; 20 legal moves at batch 8 -> 8, 8, 4).

    ⚑ The marker asserts on ``torch.compiler.is_compiling()``, NOT on the
    shapes it saw: dynamo executes a patched forward while TRACING it, so a
    shape-only marker passes on the compiled path too and is not an
    instrument.
    """
    board = chess.Board()
    best = chess.Move.from_uci("e2e4")
    net = _PuzzleNet(move_idx=compact_policy_index(move_to_index(best, board)))
    counter = _CompileCounter()
    compiled = cast("torch.nn.Module", torch.compile(net, backend=counter))

    seen: list[bool] = []
    orig_forward = net.forward

    def _marked(x: torch.Tensor) -> dict[str, torch.Tensor]:
        seen.append(bool(torch.compiler.is_compiling()))
        return orig_forward(x)

    net.forward = _marked
    suite = PuzzleSuite(
        puzzles=[Puzzle(board=board.copy(), best_moves=[best]) for _ in range(n_puzzles)],
        name="epd",
    )
    try:
        result = getattr(puzzles, fn_name)(
            compiled, suite, device="cpu", batch_size=batch_size)
    finally:
        net.forward = orig_forward

    assert seen, "the puzzle eval never ran a forward — the test proves nothing"
    assert not any(seen), f"a puzzle forward ran under a tracer: {seen}"
    assert counter.n == 0, f"puzzle eval compiled {counter.n} graph(s)"
    assert result.total == n_puzzles


# ---------------------------------------------------------------------------
# (b) a raising probe is contained, and warns ONCE


def test_a_raising_probe_warns_once_and_does_not_propagate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: drop the ``try/except`` in ``_run_era_probes_if_due`` — the
    RuntimeError then escapes and takes the training iteration with it.

    Also pins the QUIET part: one warning line for the whole iteration naming
    every failed label and its exception type, and at most one traceback for
    the process. The live wedge printed ~60 traceback frames per iteration,
    every iteration, for the rest of the session.
    """
    trainable_phases._PROBE_TRACEBACK_PRINTED = False

    class _Boom(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            _ = x
            raise RuntimeError("beginAllocateToPool: already recording to mempool_id")

    probes = {ep.PROBE_ERA: _probe_set(4), ep.PROBE_INWINDOW: _probe_set(4)}
    boom = _Boom()
    out = _run_era_probes_if_due(
        boom, probes, tc=_tc(), device="cpu", iteration_zero_based=0)

    assert np.isnan(out["probe_era_policy_eregret"])
    assert np.isnan(out["probe_inwindow_policy_eregret"])
    assert out["probe_era_n"] == 0.0
    cap = capsys.readouterr()
    first = cap.out + cap.err
    assert first.count("[probe] WARNING") == 1, first
    assert "RuntimeError" in first, "the warning must name the exception type"
    assert ep.PROBE_ERA in first
    assert ep.PROBE_INWINDOW in first
    # Two sets failed; the traceback is printed for the FIRST one only.
    assert first.count("Traceback (most recent call last)") == 1, (
        "the first failure must still be diagnosable, exactly once")

    # The second iteration repeats the one-liner and NOT the traceback.
    _run_era_probes_if_due(
        boom, probes, tc=_tc(), device="cpu", iteration_zero_based=1)
    cap = capsys.readouterr()
    second = cap.out + cap.err
    assert second.count("[probe] WARNING") == 1, second
    assert "Traceback (most recent call last)" not in second, second
    # Leave the process-wide flag as the module defines it, so this test
    # cannot couple to any other through import order.
    trainable_phases._PROBE_TRACEBACK_PRINTED = False


def test_the_probe_failure_handler_does_not_wrap_the_training_step() -> None:
    """The containment must be scoped to the probe call and nothing else.

    A ``try/except Exception`` that grew to cover the training step would turn
    a real training crash into a warning — this repo's signature defect in its
    most expensive form. Read the source: the only call inside the guarded
    block is ``score_probe_set``.
    """
    src = importlib.import_module(
        "chess_anti_engine.tune.trainable_phases").__file__
    assert src is not None
    from pathlib import Path

    text = Path(src).read_text(encoding="utf-8")
    start = text.index("def _run_era_probes_if_due(")
    body = text[start:text.index("\ndef ", start + 1)]
    guarded = body[body.index("        try:"):body.index("        except Exception as exc:")]
    assert "score_probe_set(" in guarded
    assert guarded.count("(") - guarded.count("score_probe_set(") <= 3, guarded
    assert "trainer" not in guarded
    assert "train_step" not in guarded
