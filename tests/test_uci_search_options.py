"""Every UCI search option must reach the search, or say that it cannot.

The acceptance criterion for an option here is NOT "a config field got
assigned". It is: what observation proves this reached the search? So every
test below drives the option through the real ``parse_command`` ->
``Engine.dispatch`` -> ``setoption`` path and then observes either

  * a different search — the root visit distribution and/or bestmove of a real
    ``go``, run against a deterministic evaluator so a difference cannot be
    thread noise (``test_gumbel_shape_option_changes_the_search``); or
  * an explicit refusal / NO-EFFECT report when the option cannot act on the
    live search path (``test_..._reported_inert_...``).

The failure mode this file exists for is real and recent in this repo:
``c_puct`` and the ``fpu_*`` family were accepted, stored, and printed as
realized Gumbel search configuration while ``full_tree=True`` made the PUCT
descent they drive unreachable. An operator could tune them for a whole
tournament against a number that never moved.

The NULL CONTROL is ``test_a_cosmetic_option_changes_nothing``: an option that
genuinely should not touch the search must leave the signature byte-identical.
Without it, a "the search changed" assertion could be passing because the
harness is noisy rather than because the knob works.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import pathlib
import re
import threading
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    PY_ONLY_GUMBEL_KNOBS,
    GumbelConfig,
)
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.mcts.search_options import (
    OPTIONS_BY_NAME,
    SEARCH_OPTIONS,
    SEARCH_PATHS,
    branch_note,
    inert_reason,
    realized_rows,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.engine import Engine, EngineOptions, emit_handshake
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.walker_pool import WalkerPoolConfig

# A quiet middlegame-ish position with plenty of legal moves, so the root has
# more candidates than `topk` and the halving schedule really runs.
FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Enough sims that the DESCENT-site transforms (QVisitExp / QVisitFloor /
# QGlobalScale) have a tree deep enough to act on. Measured: at 64 nodes three
# of them are indistinguishable from baseline, which would have produced a
# false "inert" reading for a knob that works. Resolution before threshold.
NODES = 2048


class _DetEval:
    """Deterministic logits from the encoded planes — a fixed random projection.

    Deterministic is the whole point: it makes the classic Gumbel path's root
    visit distribution reproducible, so a difference between two runs is the
    knob and nothing else. ``test_the_harness_is_deterministic`` asserts it.
    """

    def __init__(self, planes: int) -> None:
        rs = np.random.default_rng(1234)
  # 1/sqrt(fan_in): without it the pre-tanh sum overflows float32 on a
  # 175-plane board and the logits go NaN, which the search then treats as a
  # position rather than as a broken evaluator.
        self._w = (
            rs.normal(size=(planes * 64, 16)) / np.sqrt(planes * 64.0)
        ).astype(np.float32)
        self._p = rs.normal(size=(16, POLICY_SIZE)).astype(np.float32)
        self._v = rs.normal(size=(16, 3)).astype(np.float32)

    def evaluate_encoded(self, x, relations=None):
        del relations
  # The leaf buffer is `np.empty`-allocated and its PAD rows carry stale
  # content the search discards; sanitising here keeps a NaN from that
  # padding out of the logits instead of letting it look like a position.
        a = np.nan_to_num(
            np.asarray(x, dtype=np.float32).reshape(np.shape(x)[0], -1),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        h = np.tanh(a @ self._w)
        return (h @ self._p).astype(np.float32), (h @ self._v).astype(np.float32)


def _make_engine(
    *,
    threads: int = 1,
    evaluator: _DetEval | None = None,
    cfg_over: dict[str, float] | None = None,
) -> Engine:
    planes = input_plane_count("v2_threats")
    worker = SearchWorker(
        _DetEval(planes) if evaluator is None else evaluator,
        device="cpu",
        gumbel_cfg=dataclasses.replace(
            GumbelConfig(
                simulations=256, add_noise=False, temperature=0.0,
                input_extra_features="v2_threats",
            ),
            **{**PLAY_SEARCH_DEFAULTS, **(cfg_over or {})},
        ),
        chunk_sims=256,
        n_walkers=threads,
        vloss_weight=3,
    )
    options = EngineOptions(threads=threads)
  # Copy worker -> options exactly as `_build_engine` does. Without it the
  # harness diverges from production in a way that MAKES ITS OWN FINDINGS:
  # EngineOptions defaults to the PLAY_PUCT values (c_puct 1.75, fpu 0.33,
  # cpuct_factor 3.89) while GumbelConfig's own defaults are 2.5 / 1.2 / 0.0,
  # so `_sync_cpuct_to_worker`'s three-field mirror would silently move c_puct
  # as a side effect of setting FpuReduction — an artifact of the rig, not of
  # the engine, which `__main__` cannot produce.
    for opt in SEARCH_OPTIONS:
        options.set_search_value(opt.field, worker.realized_search_values()[opt.field])
    return Engine(worker=worker, options=options)


def _setoption(engine: Engine, line: str) -> None:
    """Drive one option through the REAL command path, not the handler."""
    engine.dispatch(parse_command(line))


def _signature(engine: Engine) -> tuple[int, int, tuple[int, ...]]:
    """Run a real ``go``; return (best root action, tree size, root visits).

    Goes through ``position`` / ``go`` dispatch so the observation is of the
    search the engine would run in a game, not of a helper called by hand.

    The tree size is in here because the ROOT visit distribution alone cannot
    see the DESCENT-site transforms. Under the shipped play shape the root uses
    the log transform (``QVisitExpRoot=-1``), which dominates the root's
    sequential halving, so ``QVisitExp`` / ``QGlobalScale`` / ``QVisitFloor``
    change which leaves get expanded deeper in the tree while leaving the root
    counts identical at 2048 sims. Judged on root visits alone all three would
    have read INERT — a wrong verdict produced by the instrument, not the code.
    Node count sees them, and the null control still does not move it.

    No node-count literals here on purpose: an earlier revision quoted four,
    and two reviewers on two boxes could reproduce none of them. Anything that
    changes the evaluator, the plane count or the C tree's growth policy moves
    every one, so a number frozen in a docstring rots into a false claim about
    the instrument. Re-derive instead, in one command:

        PYTHONPATH=. python3 -c "import tests.test_uci_search_options as T; \\
          print(T._run()[1], [T._run((f'setoption name QVisitExp value {v}',))[1] \\
                              for v in (0, 0.5, 2)])"

    The property that must hold — node count moves while the root visit vector
    does not — is enforced by the `_SETOPTION_EFFECT_CELLS` table below and by
    `test_gumbel_shape_option_changes_the_search`, not by these numbers.
    """
    engine.dispatch(parse_command(f"position fen {FEN}"))
    engine.dispatch(parse_command(f"go nodes {NODES}"))
    engine._wait_for_search()
    worker = engine._worker
    if worker._tree is None or worker._root_id is None:
        return -1, 0, ()
    actions, counts = worker._tree.get_children_visits(worker._root_id)
    actions = np.asarray(actions)
    counts = np.asarray(counts)
    best = int(actions[int(np.argmax(counts))]) if counts.size else -1
    return best, int(worker._tree.node_count()), tuple(
        int(v) for v in counts[np.argsort(actions)]
    )


def _run(
    lines: tuple[str, ...] = (), *, threads: int = 1,
) -> tuple[int, int, tuple[int, ...]]:
    engine = _make_engine(threads=threads)
    try:
        for line in lines:
            _setoption(engine, line)
        return _signature(engine)
    finally:
        engine._worker.close()


# --- harness controls --------------------------------------------------------


def test_the_harness_is_deterministic() -> None:
    """Precondition for every 'the search changed' assertion below.

    If this fails, a knob test could pass on run-to-run noise. The walker pool
    genuinely IS nondeterministic, which is why the walker-mode assertions in
    this file are structural rather than behavioural.
    """
    assert _run() == _run()


def test_a_cosmetic_option_changes_nothing() -> None:
    """NULL CONTROL. MultiPV changes how many lines are REPORTED, not searched.

    An option that must not move the search, driven through the same path, at
    the same budget. It has to leave the signature byte-identical or every
    positive result in this file is suspect.
    """
    base = _run()
    assert _run(("setoption name MultiPV value 3",)) == base
    assert _run(("setoption name UCI_ShowWDL value true",)) == base


# --- the search-shape options really shape the search ------------------------

# (option, probe value, extra setoptions applied to BOTH arms).
#
# The context column is not a convenience. `QVisitExpRoot` chooses between a
# LOG and a LINEAR root value-transform, and at the shipped `CScaleRoot=7` both
# land far enough up the sigma(q) curve that the completed-Q term already
# dominates the prior — the root ranking is then identical either way and the
# knob is unobservable at every value tried (0, 0.5, 1, 2, 98, -10). Lowering
# the root scale into the regime where the transform still competes with the
# prior is what makes the observation possible. Recording that here rather than
# quietly dropping the option is the point: "no probe value moved it" would
# otherwise have read as "the plumbing is broken".
#
# `searchconfig` now REPORTS that inertness rather than printing [LIVE] over it
# — see `inert_reason`'s q_visit_exp_root arms and the calibration test
# `test_an_inert_root_exponent_verdict_implies_an_unobservable_search` below.
#
# `GumbelScale` is absent: it is nondeterministic by construction and has its
# own test.
_MOVES_THE_SEARCH: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("PolicyTemperature", "3.0", ()),
    ("CScale", "0.9", ()),
    ("CVisit", "5.0", ()),
    ("CScaleRoot", "0.5", ()),
    ("CVisitRoot", "10.0", ()),
    ("QVisitExp", "0.4", ()),
    ("QVisitExpRoot", "1.0", ("setoption name CScaleRoot value 0.05",)),
    ("QVisitFloor", "20.0", ()),
    ("QGlobalScale", "true", ()),
    ("HalvingDiv", "4", ()),
    ("Topk", "4", ()),
    ("ChunkSims", "64", ()),
    ("VLossWeight", "0", ()),
    ("MinibatchSize", "1", ()),
)


@pytest.mark.parametrize(
    ("name", "value", "context"), _MOVES_THE_SEARCH,
    ids=[row[0] for row in _MOVES_THE_SEARCH],
)
def test_gumbel_shape_option_changes_the_search(
    name: str, value: str, context: tuple[str, ...],
) -> None:
    """The deciding test: setoption -> a demonstrably different search.

    Fails if the option is dropped anywhere between `parse_command` and the C
    search — a missing dispatch entry, a value written to a copy of the config,
    or a knob the path does not read.
    """
    base = _run(context)
    got = _run((*context, f"setoption name {name} value {value}"))
    assert got != base, (
        f"{name}={value} left the search byte-identical: same best root move, "
        "same tree size, same root visit distribution. It did not reach the "
        "search."
    )


def test_root_noise_scale_reaches_the_search_and_zero_stays_deterministic() -> None:
    """`GumbelScale` had no reachable mechanism at all before this option.

    `SearchWorker._run_gumbel_chunk` hard-coded `add_noise=False`, so
    `GumbelConfig.gumbel_scale` could not be observed from the engine at any
    value. The zero half of this test is what proves the default is still the
    deterministic search every prior build ran.
    """
    base = _run()
    assert _run(("setoption name GumbelScale value 0.0",)) == base

    noisy = [_run(("setoption name GumbelScale value 1.5",)) for _ in range(4)]
    assert any(sig != base for sig in noisy), (
        "GumbelScale 1.5 never perturbed the root candidate set"
    )


def test_root_noise_is_mirrored_into_the_shared_config_for_rpg() -> None:
    """The second half of GumbelScale, which the classic-path test cannot see.

    Root-parallel Gumbel needs >= 2 devices, so its behaviour is not testable
    here. What IS testable is the mechanism: RPG reads `add_noise` /
    `gumbel_scale` off the SHARED GumbelConfig by reference at candidate-
    selection time, so `set_root_noise_scale` has to write them there too or
    the option is a silent null on that path — with no warning, because the
    registry declares it live on `rpg`.

    A mutation that drops the mirror survived the behavioural suite; this is
    the test that kills it. Both halves are asserted: that the worker writes
    the fields, and that RPG is what reads them (source-level, since a
    two-device search is unavailable).
    """
    import inspect

    from chess_anti_engine.uci import root_parallel_gumbel as rpg

    engine = _make_engine()
    try:
        _setoption(engine, "setoption name GumbelScale value 1.5")
        cfg = engine._worker._cfg
        assert cfg.add_noise is True
        assert cfg.gumbel_scale == pytest.approx(1.5)

        _setoption(engine, "setoption name GumbelScale value 0")
        assert engine._worker._cfg.add_noise is False
    finally:
        engine._worker.close()

    src = inspect.getsource(rpg)
    assert "self._gcfg.add_noise" in src, (
        "RPG no longer reads add_noise off the shared config — re-decide "
        "whether GumbelScale is still live on the rpg path in SEARCH_OPTIONS"
    )
    assert "self._gcfg.gumbel_scale" in src


# --- inert options are never silently accepted -------------------------------


@pytest.mark.parametrize("name", ["CPuct", "CPuctFactor", "CPuctBase", "FpuReduction"])
def test_puct_option_is_reported_inert_under_classic_gumbel(
    name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact defect this surface could have mass-produced.

    Two assertions, and both matter: the engine SAYS the value cannot act
    (an operator reading a match log finds out), and the search is in fact
    unchanged (so the message is true rather than defensive boilerplate).
    """
    base = _run()
    engine = _make_engine()
    try:
        _setoption(engine, f"setoption name {name} value 9.5")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out, f"{name} was accepted silently under Gumbel"
        assert "gumbel" in out
        assert _signature(engine) == base, (
            f"{name} is advertised as inert under Gumbel but moved the search"
        )
    finally:
        engine._worker.close()


@pytest.mark.parametrize(
    "name", ["CScale", "CVisit", "Topk", "PolicyTemperature", "HalvingDiv"],
)
def test_gumbel_option_is_reported_inert_on_the_walker_pool(
    name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Threads>1 is plain PUCT: every Gumbel shape knob stops applying.

    The report is asserted behaviourally-by-proxy rather than by a visit-count
    diff, because the walker pool races N threads onto a shared tree and its
    visit distribution is NOT reproducible — a diff there would be
    uninterpretable. `test_the_walker_pool_config_carries_no_gumbel_knob`
    supplies the structural half.
    """
    engine = _make_engine(threads=2)
    try:
        capsys.readouterr()
        _setoption(engine, f"setoption name {name} value 3")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out
        assert "walker" in out
    finally:
        engine._worker.close()


def test_the_walker_pool_config_carries_no_gumbel_knob() -> None:
    """Structural proof behind the walker-mode inertness claim.

    The pool cannot read what it was never given. Adding a Gumbel field to
    `WalkerPoolConfig` should fail this test and force a re-decision of the
    `live_in` sets in the registry.
    """
    fields = {f.name for f in dataclasses.fields(WalkerPoolConfig)}
    gumbel_only = {
        o.field for o in SEARCH_OPTIONS if "walker" not in o.live_in
    }
    assert not (fields & gumbel_only), sorted(fields & gumbel_only)


def test_minibatch_size_is_inert_off_the_classic_gumbel_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MinibatchSize shipped before this PR with no mode gate at all.

    It is `run_gumbel_root_many_c(target_batch=...)` — a C-path function
    argument — so on the walker pool it was accepted, stored, echoed back as
    "MinibatchSize set to N", and read by nothing.
    """
    engine = _make_engine(threads=2)
    try:
        capsys.readouterr()
        _setoption(engine, "setoption name MinibatchSize value 512")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out
    finally:
        engine._worker.close()


def test_minibatch_size_marks_the_captured_cudagraph_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MinibatchSize reshapes the leaf batch the cudagraph was captured at.

    `_apply_search_option` sets `_warmup_dirty` so the next idle `isready`
    re-captures BEFORE the clock starts. Nothing pinned it: dropping the flag
    passed the whole file, and the symptom would have been a cold capture paid
    mid-move on the first `go` after a config change — i.e. on the clock.

    Observed through the real command path and the real `isready` handler, so
    this fails on the flag AND on the re-warm that reads it.
    """
    engine = _make_engine()
    warmups: list[int] = []
    monkeypatch.setattr(engine, "warmup_search", lambda: warmups.append(1))
    try:
        _setoption(engine, "setoption name MinibatchSize value 128")
        assert engine._warmup_dirty is True
        engine.dispatch(parse_command("isready"))
        assert warmups == [1], "isready did not re-warm the reshaped search path"

  # NULL CONTROL: a shape knob the C reads per call needs no re-capture, so
  # the flag must not simply be set by every setoption.
        _setoption(engine, "setoption name QVisitExp value 0.5")
        assert engine._warmup_dirty is False
        engine.dispatch(parse_command("isready"))
        assert warmups == [1]
    finally:
        engine._worker.close()


def test_vloss_weight_reaches_the_walker_pool_on_the_shipped_default() -> None:
    """Threads=2 is the shipped default, and the walker pool copies
    `vloss_weight` into `WalkerPoolConfig` at CONSTRUCTION.

    So `SearchWorker.set_vloss_weight`'s pool rebuild is the part that makes the
    option real. `Engine._apply_search_option` also calls
    `_reinstall_configured_search_path()`, which LOOKS like a second cover — but
    on the default configuration (`search_parallel="pucv"`, multi-GPU off, not
    leaving Gumbel) that method falls through every branch and does nothing.
    Measured with the rebuild removed: the engine prints "VLossWeight set to 17"
    while the pool keeps descending on 3 and `realized_search_values()` reports
    3. Accepted, echoed, and ignored.
    """
    engine = _make_engine(threads=2)
    try:
        assert engine._worker.realized_search_path() == "walker"
        before = engine._worker.realized_search_values()["vloss_weight"]
        assert before != 17
        _setoption(engine, "setoption name VLossWeight value 17")
        assert engine._worker.realized_search_values()["vloss_weight"] == 17
        assert engine._worker._walker_pool is not None
        assert engine._worker._walker_pool._cfg.vloss_weight == 17
    finally:
        engine._worker.close()


def test_the_worker_setter_rebuilds_the_pool_without_the_engine() -> None:
    """`SearchWorker.set_vloss_weight`'s OWN contract, no Engine involved.

    The method is public and the pool rebuild is inside it; a caller that is
    not the UCI handler gets no `_reinstall_configured_search_path` at all.
    """
    engine = _make_engine(threads=2)
    worker = engine._worker
    try:
        assert worker._walker_pool is not None
        worker.set_vloss_weight(17)
        assert worker._walker_pool is not None
        assert worker._walker_pool._cfg.vloss_weight == 17
    finally:
        worker.close()


def test_searchconfig_reports_the_path_that_actually_ran(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UseVL=true on an evaluator without the slot API is accepted and ignored.

    The readback must report the path that WILL run (classic gumbel), not the
    one that was requested, or it becomes another way to be confidently wrong
    about the engine's configuration.
    """
    engine = _make_engine()
    try:
        _setoption(engine, "setoption name UseVL value true")
        assert engine._worker.realized_search_path() == "gumbel"
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        out = capsys.readouterr().out
        assert "searchconfig path=gumbel" in out
        assert re.search(r"searchconfig CScale = 0\.025 \[LIVE\]", out)
        assert re.search(r"searchconfig CPuct = [0-9.]+ \[INERT\]", out)
        # Every registered option gets a row: an omitted row is
        # indistinguishable from an unsupported build.
        for opt in SEARCH_OPTIONS:
            assert f"searchconfig {opt.name} = " in out
    finally:
        engine._worker.close()


def test_searchconfig_reports_a_value_the_setoption_actually_applied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Readback and search must agree, or the readback is decoration."""
    engine = _make_engine()
    try:
        _setoption(engine, "setoption name PolicyTemperature value 2.5")
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        assert "searchconfig PolicyTemperature = 2.5 [LIVE]" in capsys.readouterr().out
        assert engine._worker._cfg.policy_temp == pytest.approx(2.5)
    finally:
        engine._worker.close()


# --- parsing and ranges ------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expect"),
    [
        ("setoption name PolicyTemperature value 0.4", "out of range"),
        ("setoption name PolicyTemperature value 5.5", "out of range"),
        ("setoption name PolicyTemperature value abc", "not a number"),
        ("setoption name Topk value 1", "out of range"),
        ("setoption name HalvingDiv value 1.5", "not a integer"),
        ("setoption name QGlobalScale value yes", "expected true/false"),
    ],
)
def test_a_rejected_value_says_so_and_keeps_the_old_one(
    line: str, expect: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """UCI gives a GUI no way to see a rejected option, so silence here is
    indistinguishable from success. A bad `PolicyTemperature 0` would also
    divide the policy logits by zero."""
    engine = _make_engine()
    try:
        before = dict(engine._worker.realized_search_values())
        capsys.readouterr()
        _setoption(engine, line)
        assert expect in capsys.readouterr().out
        assert engine._worker.realized_search_values() == before
    finally:
        engine._worker.close()


def test_policy_temperature_accepts_the_documented_range_ends() -> None:
    engine = _make_engine()
    try:
        for value in ("0.5", "5.0"):
            _setoption(engine, f"setoption name PolicyTemperature value {value}")
            assert engine._worker._cfg.policy_temp == pytest.approx(float(value))
    finally:
        engine._worker.close()


# --- the surface cannot drift from the code ----------------------------------


def test_advertised_defaults_match_the_live_worker() -> None:
    """A handshake default that is not what the engine runs is the same lie.

    `_build_engine` copies the constructed worker's realized values back into
    EngineOptions for exactly this reason; this test fails if someone
    reintroduces a hand-typed default.
    """
    from chess_anti_engine.uci.__main__ import _build_engine

    planes = input_plane_count("v2_threats")
    options = EngineOptions()
    engine = _build_engine(
        evaluator=_DetEval(planes),
        primary_device="cpu",
        chunk_sims=777,
        topk=9,
        c_scale=0.077,
        policy_temp=1.5,
        n_walkers=1,
        vloss_weight=2,
        walker_gather=1,
        pucv_vloss_mode=0,
        max_batch=64,
        vl_gather=64,
        eval_cache_entries=0,
        use_multi_gpu_pucv=False,
        input_extra_features="v2_threats",
        options=options,
    )
    try:
        realized = engine._worker.realized_search_values()
        for opt in SEARCH_OPTIONS:
            assert options.search_value(opt.field) == realized[opt.field], (
                f"handshake would advertise {opt.name}="
                f"{options.search_value(opt.field)} while the worker runs "
                f"{realized[opt.field]}"
            )
        # ...and the CLI values really landed, not just agreed with each other.
        assert realized["chunk_sims"] == 777
        assert realized["topk"] == 9
        assert realized["policy_temp"] == pytest.approx(1.5)
    finally:
        engine._worker.close()


def _printed_handshake_defaults(options: EngineOptions) -> dict[str, str]:
    """`{option name: advertised default}` parsed out of the PRINTED handshake.

    The printed line is the artifact. `test_advertised_defaults_match_the_live_worker`
    above compares `EngineOptions.search_value` with the worker and never calls
    `emit_handshake`, so reverting `declaration()` to `opt.default` passed the
    entire file (reviewer measurement: 83/83). Everything below reads stdout.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        emit_handshake(options)
    out: dict[str, str] = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"option name (\S+) type \S+ default (\S+)", line)
        if m is not None:
            out[m.group(1)] = m.group(2)
    return out


def _fmt_expected(opt, value: float | int | bool) -> str:
    """How `declaration()` renders `value` — computed from the value, not from
    the registry, so this helper cannot agree with a `self.default` mutant."""
    if opt.kind == "check":
        return "true" if value else "false"
    if opt.kind == "spin":
        return str(int(value))
    return repr(float(value))


def test_the_printed_handshake_advertises_the_live_worker() -> None:
    """THE guard `SearchOption.declaration`'s docstring claims.

    Kills the `opt.declaration(opt.default)` mutant: the CLI values below are
    all off-default, so a registry-default handshake prints 0.025 / 32 / 1.0 /
    2048 for a worker running 0.077 / 9 / 1.5 / 777.
    """
    from chess_anti_engine.uci.__main__ import _build_engine

    planes = input_plane_count("v2_threats")
    options = EngineOptions()
    engine = _build_engine(
        evaluator=_DetEval(planes), primary_device="cpu",
        chunk_sims=777, topk=9, c_scale=0.077, policy_temp=1.5,
        halving_div=4, root_noise_scale=0.25, q_global_scale=True,
        n_walkers=1, vloss_weight=2, walker_gather=1, pucv_vloss_mode=0,
        max_batch=64, vl_gather=64, eval_cache_entries=0,
        use_multi_gpu_pucv=False, input_extra_features="v2_threats",
        options=options,
    )
    try:
        realized = engine._worker.realized_search_values()
        printed = _printed_handshake_defaults(engine._options)
        for opt in SEARCH_OPTIONS:
            assert opt.name in printed, f"{opt.name} is not in the handshake at all"
            assert printed[opt.name] == _fmt_expected(opt, realized[opt.field]), (
                f"the handshake advertises {opt.name}={printed[opt.name]} while "
                f"the worker runs {realized[opt.field]}"
            )
        # ...and the off-default values really are off-default, so the
        # comparison above had something to catch.
        assert printed["ChunkSims"] == "777"
        assert printed["Topk"] == "9"
        assert printed["PolicyTemperature"] == "1.5"
        assert printed["CScale"] == "0.077"
        assert printed["HalvingDiv"] == "4"
        assert printed["GumbelScale"] == "0.25"
        assert printed["QGlobalScale"] == "true"
    finally:
        engine._worker.close()


def test_the_first_uci_advertises_the_engine_that_is_about_to_be_built() -> None:
    """The FIRST handshake — the one a GUI sends before the model is loaded.

    `__main__` answers `uci` from `startup_options` and `continue`s BEFORE
    `engine_ready.wait()`, while `_build_engine`'s worker -> options copy runs
    on the build thread. So a handshake sourced from bare `EngineOptions`
    defaults advertised `--c-scale 0.077 --topk 9 --policy-temp 1.5
    --chunk-sims 777` as 0.025 / 32 / 1.0 / 2048 and only corrected itself on a
    second `uci` most GUIs never send. Measured before the fix: 4 option lines
    differed from the search that then ran.

    This drives the REAL parser and the REAL startup-options factory, then
    builds the engine from the same `args`, so a drift between the two
    (a new CLI flag, a renamed `dest`) fails here rather than in a match log.

    ⚑ THE TWO SIDES COME FROM TWO SOURCES. The advertised side is
    `_seed_search_options_from_args`, which reads `_SEARCH_OPTION_ARG`; the
    built side is `_engine_search_kwargs`, the kwargs `main()` itself hands
    `_build_engine`, which does not consult that map at all. An earlier
    revision built the engine side FROM `_SEARCH_OPTION_ARG` too, and was
    therefore a control conditioned on its own outcome: setting
    `_SEARCH_OPTION_ARG["cpuct_factor"] = None` (reviewer mutant `n2`) dropped
    the option from the expectation as well as from the seeding, both sides
    fell back to 3.89, and the mutant survived 193 tests while
    `--cpuct-factor 7.0` really did advertise 3.89 for an engine running 7.0.
    """
    from chess_anti_engine.uci.__main__ import (
        _build_engine,
        _build_parser,
        _engine_search_kwargs,
        _startup_engine_options,
    )

    args = _build_parser().parse_args([
        "--checkpoint", "unused-by-this-test",
        "--c-scale", "0.077", "--topk", "9", "--policy-temp", "1.5",
        "--chunk-sims", "777", "--halving-div", "4", "--gumbel-scale", "0.25",
        "--q-global-scale", "--vloss-weight", "5", "--c-puct", "3.25",
        "--fpu-reduction", "0.9", "--c-visit-root", "123.0",
        "--cpuct-factor", "7.0", "--cpuct-base", "12345.0",
        "--walkers", "2",
    ])
    startup_options = _startup_engine_options(
        args, search_parallel="pucv", restore_multi_gpu_pucv=False,
    )
    first_uci = _printed_handshake_defaults(startup_options)

    planes = input_plane_count("v2_threats")
  # PRODUCTION's own kwargs, not a re-derivation from the map under test.
    built_from = _engine_search_kwargs(args)
    engine = _build_engine(
        evaluator=_DetEval(planes), primary_device="cpu",
        n_walkers=int(args.walkers), walker_gather=1, pucv_vloss_mode=0,
        max_batch=64, vl_gather=64, eval_cache_entries=0,
        use_multi_gpu_pucv=False, input_extra_features="v2_threats",
        options=EngineOptions(),
        **built_from,
    )
    try:
        realized = engine._worker.realized_search_values()
        for opt in SEARCH_OPTIONS:
            assert first_uci[opt.name] == _fmt_expected(opt, realized[opt.field]), (
                f"the FIRST `uci` advertises {opt.name}={first_uci[opt.name]} "
                f"while the engine it is about to build runs "
                f"{realized[opt.field]}"
            )
        assert first_uci["CScale"] == "0.077"
        assert first_uci["Topk"] == "9"
        assert first_uci["ChunkSims"] == "777"
        assert first_uci["VLossWeight"] == "5"
      # `n2`'s field, off BOTH defaults (registry 3.89, `_build_engine`
      # signature 3.89) so demoting it to `None` in `_SEARCH_OPTION_ARG` is
      # visible rather than masked by an accidental agreement -- which is
      # exactly why `n1` (`c_puct`) died and `n2` did not.
        assert first_uci["CPuctFactor"] == "7.0"
        assert first_uci["CPuctBase"] == "12345.0"
    finally:
        engine._worker.close()


def test_the_startup_copy_is_taken_after_the_multi_gpu_pool_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_build_engine`'s worker -> options copy must be the LAST thing it does.

    `realized_search_values` answers the PUCT family off the INSTALLED descent
    object, so a copy taken before `install_multi_gpu_pucv` /
    `install_root_parallel_gumbel` reads the classic-Gumbel config and then
    advertises it for a search that runs on the pool's.

    The pool itself is stubbed — the property under test is the ORDER of the
    copy relative to the install, not the pool — and its `_cfg` carries values
    no other object in the build has, so agreement cannot be a coincidence.
    """
    from chess_anti_engine.uci.__main__ import _build_engine

    installed_cfg = SimpleNamespace(
        c_puct=8.125, cpuct_factor=6.5, cpuct_base=4321.0,
        fpu_reduction=-2.25, vloss_weight=11,
    )

    class _StubPool:
        _cfg = installed_cfg

        def close(self) -> None:
            return None

    def _fake_install(
        self: SearchWorker, evaluators_or_factories: object, **kwargs: object,
    ) -> None:
        del evaluators_or_factories, kwargs
        self._pucv_pool = _StubPool()  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr(SearchWorker, "install_multi_gpu_pucv", _fake_install)

    planes = input_plane_count("v2_threats")
    options = EngineOptions()
    engine = _build_engine(
        evaluator=_DetEval(planes), primary_device="cpu",
        chunk_sims=256, topk=9,
        c_puct=1.75, cpuct_factor=3.89, cpuct_base=38739.0, fpu_reduction=0.33,
        n_walkers=1, vloss_weight=2, walker_gather=1, pucv_vloss_mode=0,
        max_batch=64, vl_gather=64, eval_cache_entries=0,
        use_multi_gpu_pucv=True,
        rebuild_multi_gpu_pucv_factories=lambda _mb, _g: [object(), object()],
        input_extra_features="v2_threats",
        options=options,
    )
    try:
        assert engine._worker.realized_search_path() == "pucv_pool"
        printed = _printed_handshake_defaults(engine._options)
        assert printed["CPuct"] == "8.125", (
            "the handshake advertises the pre-install c_puct while the "
            "installed pool descends on 8.125"
        )
        assert printed["FpuReduction"] == "-2.25"
        assert printed["VLossWeight"] == "11"
    finally:
        engine._worker._pucv_pool = None
        engine._worker.close()


def test_every_registry_option_has_a_named_startup_source() -> None:
    """A new option must declare where its startup value comes from.

    Without this, adding a registry entry silently reverts the first handshake
    to advertising a default the engine is not running — the exact defect
    above, one option at a time.

    ⚑ `None` is an ESCAPE HATCH, so it needs its own guard. `assert dest is
    None or dest in dests` accepts `None` unconditionally, which makes
    "this option has no CLI flag" an unfalsifiable claim: assert it against
    the parser instead. `_SEARCH_OPTION_ARG["cpuct_factor"] = None` is a lie
    (`--cpuct-factor` exists) and the old form waved it through.
    """
    from chess_anti_engine.uci.__main__ import _SEARCH_OPTION_ARG, _build_parser

    dests = {a.dest for a in _build_parser()._actions}
    for opt in SEARCH_OPTIONS:
        assert opt.field in _SEARCH_OPTION_ARG, (
            f"{opt.name} has no entry in _SEARCH_OPTION_ARG"
        )
        dest = _SEARCH_OPTION_ARG[opt.field]
        if dest is None:
            assert opt.field not in dests, (
                f"{opt.name} is mapped to None ('no CLI flag') but the parser "
                f"defines --{opt.field.replace('_', '-')}. The first `uci` "
                "would advertise the registry default while the engine is "
                "built from the flag."
            )
            continue
        assert dest in dests, (
            f"{opt.name} is seeded from args.{dest}, which the CLI does not define"
        )


def test_seeding_refuses_a_registry_option_with_no_startup_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup guard must be able to FIRE, not just be written down.

    Its job is to turn "a new option quietly advertises the wrong default" into
    a loud failure at engine start, so the failure mode has to be reachable.
    """
    from chess_anti_engine.uci import __main__ as uci_main

    unmapped = dataclasses.replace(
        SEARCH_OPTIONS[0], name="NewlyAdded", field="field_nobody_mapped",
    )
    monkeypatch.setattr(uci_main, "SEARCH_OPTIONS", (*SEARCH_OPTIONS, unmapped))
    with pytest.raises(AssertionError, match="no startup source"):
        uci_main._seed_search_options_from_args(
            EngineOptions(), argparse.Namespace(),
        )


def test_every_registry_field_resolves_on_engine_options() -> None:
    options = EngineOptions()
    for opt in SEARCH_OPTIONS:
        options.search_value(opt.field)  # raises AttributeError on a typo


def test_every_registry_option_is_reachable_by_its_uci_name() -> None:
    """A registry entry with no dispatch route is an option that cannot be set.

    `_handle_setoption` lowercases the name and looks it up in OPTIONS_BY_NAME,
    so this is the property that makes every declared option settable.
    """
    for opt in SEARCH_OPTIONS:
        assert OPTIONS_BY_NAME[opt.name.lower()] is opt


def test_float_options_declare_a_range_and_int_options_declare_bounds() -> None:
    for opt in SEARCH_OPTIONS:
        if opt.kind in ("string", "spin"):
            assert opt.lo is not None, opt.name
            assert opt.hi is not None, opt.name
            assert opt.lo < opt.hi, opt.name
            assert opt.lo <= float(opt.default) <= opt.hi, opt.name


def test_policy_temp_is_reachable_from_the_play_search_surface() -> None:
    """It was a live positive control in the inert-knob audit and the ONE shape
    knob no play-path config could set. Absence is not "at its default"."""
    assert "policy_temp" in PLAY_SEARCH_DEFAULTS
    assert PLAY_SEARCH_DEFAULTS["policy_temp"] == 1.0
    assert "policy_temp" in {o.field for o in SEARCH_OPTIONS}


def test_the_uci_surface_never_exposes_the_training_time_target_temperature() -> None:
    """`policy_target_temp` is a TRAINING knob on the policy TARGET. It never
    runs during play, and an operator who found it in the UCI option list would
    reasonably conclude otherwise."""
    names = {o.name.lower() for o in SEARCH_OPTIONS}
    assert "policytargettemperature" not in names
    assert not any("target" in n for n in names)


# --- the C path cannot silently drop a Python-only knob ----------------------


def test_the_c_path_refuses_a_python_only_knob() -> None:
    """The OTHER way a knob returns a flawless null.

    `volatility_q_scale` is a real, implemented GumbelConfig field — on the
    PYTHON search. `run_gumbel_root_many_c` has no code for it, so a caller
    that set it and landed on the C path would get a complete, reproducible,
    wrong measurement. The guard is on the dispatch boundary, so it covers
    every caller at once rather than one CLI.
    """
    cfg = GumbelConfig(
        simulations=8, add_noise=False, temperature=0.0,
        input_extra_features="v2_threats", volatility_q_scale=0.5,
    )
    planes = input_plane_count("v2_threats")
    with pytest.raises(ValueError, match="volatility_q_scale"):
        run_gumbel_root_many_c(
            model=None, boards=[chess.Board(FEN)], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg, evaluator=_DetEval(planes),
        )


def test_the_python_only_denylist_names_fields_that_exist() -> None:
    fields = set(GumbelConfig.__dataclass_fields__)
    assert fields >= PY_ONLY_GUMBEL_KNOBS
    # ...and none of them is exposed over UCI, since the UCI engine is
    # C-path-only and would have to refuse them at every `go`.
    assert not (PY_ONLY_GUMBEL_KNOBS & {o.field for o in SEARCH_OPTIONS})


def test_a_default_config_still_enters_the_c_path() -> None:
    """Negative control for the guard: it must not refuse ordinary configs."""
    cfg = GumbelConfig(
        simulations=8, add_noise=False, temperature=0.0,
        input_extra_features="v2_threats",
    )
    planes = input_plane_count("v2_threats")
    out = run_gumbel_root_many_c(
        model=None, boards=[chess.Board(FEN)], device="cpu",
        rng=np.random.default_rng(0), cfg=cfg, evaluator=_DetEval(planes),
    )
    assert out[1]  # an action was chosen


# --- the inertness model itself ---------------------------------------------


def test_rpg_reports_the_descent_knobs_inert_while_the_root_knobs_are_set() -> None:
    """Root-parallel Gumbel runs Gumbel only at the ROOT; the intra-candidate
    descent is a PUCT chunker. So `CScale` is read there only as the root
    fallback — and at the shipped defaults (CScaleRoot=7) it is not read at
    all. A static per-path table would have called it live."""
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    c_scale = OPTIONS_BY_NAME["cscale"]
    assert inert_reason(c_scale, "rpg", values) is not None
    assert inert_reason(c_scale, "gumbel", values) is None

    values["c_scale_root"] = -1.0
    assert inert_reason(c_scale, "rpg", values) is None


def test_topk_is_live_on_both_gumbel_paths_and_dead_on_the_puct_ones() -> None:
    topk = OPTIONS_BY_NAME["topk"]
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    assert inert_reason(topk, "gumbel", values) is None
    assert inert_reason(topk, "rpg", values) is None
    for path in ("walker", "pucv", "pucv_pool"):
        assert inert_reason(topk, path, values) is not None


def test_the_puct_family_is_live_exactly_where_a_puct_descent_runs() -> None:
    """The PREDICATE half of the [LIVE] claim, and only that half.

    This asks `inert_reason` what it thinks and never asks the search, so on
    its own it is exactly the kind of self-certifying assertion this file
    exists to distrust: it passed while a `setoption` provably could not
    deliver any of these four on `walker` or `pucv`. The behavioural half is
    `test_the_puct_family_reaches_the_pucv_descent_through_a_setoption` and
    `test_a_puct_setoption_rebuilds_the_walker_pool_the_threads_read`.
    """
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    for name in ("cpuct", "cpuctfactor", "cpuctbase", "fpureduction"):
        opt = OPTIONS_BY_NAME[name]
        assert inert_reason(opt, "gumbel", values) is not None
        for path in ("walker", "pucv", "pucv_pool", "rpg"):
            assert inert_reason(opt, path, values) is None


# --- a [LIVE] claim must be true of a setoption, not only of the CLI ---------
#
# `searchconfig` certifies CPuct / CPuctFactor / CPuctBase / FpuReduction as
# [LIVE] on `walker` (the SHIPPED default -- `--walkers default=2`) and on
# `pucv`. Both of those descents read the family off a config they snapshot at
# CONSTRUCTION, so until `rebuild_puct_helpers` existed the certification was
# false in the direction this whole surface exists to prevent: the engine
# printed "FpuReduction set to 9.0" and `FpuReduction = 9.0 [LIVE]` while the
# live `PucvChunker._fpu_red` was still 1.2 and the search was byte-identical.
#
# Worse than dead: the dropped value LANDED LATER, on the next unrelated
# command that rebuilt (`VLossWeight`), so the tree moved 40% on a command that
# had nothing to do with it.


class _DetAsyncEval(_DetEval):
    """`_DetEval` plus the 2-slot inplace-async API `PucvChunker` requires.

    The pucv path is single-threaded (CPU/GPU overlap, not parallelism), so
    unlike the walker pool it IS reproducible -- which is what makes a real
    before/after search comparison possible on a PUCT descent at all.
    `test_the_pucv_harness_is_deterministic` is the precondition.
    """

    def __init__(self, planes: int, max_batch: int = 512) -> None:
        super().__init__(planes)
        self.n_slots = 2
        self._bufs = [
            np.zeros((max_batch, planes, 8, 8), dtype=np.float32) for _ in range(2)
        ]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:bsz]

    def evaluate_inplace_async(self, bsz: int, *, slot: int = 0):
        pol, wdl = self.evaluate_encoded(self._bufs[slot][:bsz])
        return pol, wdl, None


def _live_walker_threads() -> int:
    """Walker threads alive right now. `WalkerPool` names them `walker-N`."""
    return sum(
        1 for t in threading.enumerate()
        if t.is_alive() and t.name.startswith("walker-")
    )


def _make_pucv_engine(cfg_over: dict[str, float] | None = None) -> Engine:
    """Engine whose realized search path is `pucv`, asserted not assumed."""
    planes = input_plane_count("v2_threats")
    engine = _make_engine(
        threads=1, evaluator=_DetAsyncEval(planes), cfg_over=cfg_over,
    )
    _setoption(engine, "setoption name VLGather value 32")
    _setoption(engine, "setoption name UseVL value true")
    assert engine._worker.realized_search_path() == "pucv", (
        f"harness did not reach the pucv path: "
        f"{engine._worker.realized_search_path()}"
    )
    return engine


def _run_pucv(
    lines: tuple[str, ...] = (), cfg_over: dict[str, float] | None = None,
) -> tuple[int, int, tuple[int, ...]]:
    engine = _make_pucv_engine(cfg_over)
    try:
        for line in lines:
            _setoption(engine, line)
        return _signature(engine)
    finally:
        engine._worker.close()


def test_the_pucv_harness_is_deterministic() -> None:
    assert _run_pucv() == _run_pucv()


@pytest.mark.parametrize(
    ("field", "name", "value"),
    [
        ("fpu_reduction", "FpuReduction", "9.0"),
        ("fpu_reduction", "FpuReduction", "-9.0"),
        ("c_puct", "CPuct", "99.0"),
        ("c_puct", "CPuct", "0.0001"),
    ],
)
def test_the_puct_family_reaches_the_pucv_descent_through_a_setoption(
    field: str, name: str, value: str,
) -> None:
    """A `setoption` must produce the SAME search the CLI value produces.

    Three arms, and the order matters — resolution before threshold:

    1. NULL/base: the pucv search at the shipped value.
    2. POSITIVE CONTROL: the same value delivered at CONSTRUCTION. If this does
       not move the signature the ruler is blind to the knob and arm 3 proves
       nothing, so it is asserted rather than assumed.
    3. THE CLAIM: the value delivered through `parse_command` -> `dispatch` ->
       `setoption`, with no other command in between, must reproduce arm 2
       exactly. `!=` base is not enough — landing on a THIRD signature would
       mean the option reached the descent as some other number.
    """
    base = _run_pucv()
    built = _run_pucv(cfg_over={field: float(value)})
    assert built != base, (
        f"instrument is blind to {field}={value}: the construction-time value "
        f"gives the same signature as the default, so this test could not "
        f"distinguish a working setoption from a dropped one"
    )
    via_option = _run_pucv((f"setoption name {name} value {value}",))
    assert via_option == built, (
        f"{name} {value} is certified [LIVE] on pucv but the setoption did not "
        f"produce the search the same value produces at construction: "
        f"setoption={via_option} construction={built} base={base}"
    )


def test_the_cpuct_log_ramp_reaches_the_pucv_descent_when_the_ruler_can_see_it(
) -> None:
    """The other two family members, on a ruler with resolution for them.

    `CPuctFactor` and `CPuctBase` are unobservable at the SHIPPED
    `cpuct_base=38739` and 2048 nodes: the ramp is
    `factor·log((N+base+1)/base)` = `7·log(38996/38739)` ≈ **0.046**, far below
    what the tree can resolve, so both read `moved=0` on the signature and the
    parametrized test above could only cover `c_puct` and `fpu_reduction`.
    That is the instrument, not the code — compute the resolution before the
    threshold. At `cpuct_base=100` the same factor is worth ≈ **8.9** and both
    become visible, so all four family members are behaviourally verified
    rather than two plus a caveat.

    Both command orders, because `CPuctFactor` first leaves the ramp inert
    until `CPuctBase` arrives and the rebuild must happen on the second command
    too.
    """
    sensitive = {"cpuct_base": 100.0}
    base = _run_pucv(cfg_over=sensitive)
    built = _run_pucv(cfg_over={**sensitive, "cpuct_factor": 7.0})
    assert built != base, (
        f"instrument is blind to the log ramp even at cpuct_base=100: {built}"
    )
    for order in (
        ("setoption name CPuctBase value 100.0",
         "setoption name CPuctFactor value 7.0"),
        ("setoption name CPuctFactor value 7.0",
         "setoption name CPuctBase value 100.0"),
    ):
        assert _run_pucv(order) == built, (
            f"{order} did not produce the construction-time search {built}"
        )


def test_a_puct_setoption_lands_now_and_not_on_a_later_unrelated_command() -> None:
    """The latent half: a dropped value that arrives later is worse than dead.

    Before the fix, `FpuReduction 9.0` left the chunker at 1.2 and an unrelated
    `VLossWeight 4` three commands later silently installed it — the operator
    sees the knob do nothing, then sees the engine change its mind. So assert
    the live object the descent reads carries the value IMMEDIATELY, with no
    intervening command, and that a later unrelated rebuild changes nothing
    about it.
    """
    engine = _make_pucv_engine()
    try:
        _setoption(engine, "setoption name FpuReduction value 9.0")
        chunker = engine._worker._pucv
        assert chunker is not None
        assert chunker._fpu_red == pytest.approx(9.0), (
            f"PucvChunker still descends on fpu_reduction={chunker._fpu_red} "
            f"after the engine reported FpuReduction set to 9.0"
        )
        _setoption(engine, "setoption name VLossWeight value 4")
        later = engine._worker._pucv
        assert later is not None
        assert later._fpu_red == pytest.approx(9.0)
    finally:
        engine._worker.close()


def test_a_puct_setoption_rebuilds_the_walker_pool_the_threads_read() -> None:
    """Same claim on `walker`, the SHIPPED default (`--walkers default=2`).

    Structural rather than a signature diff, and deliberately so: the walker
    pool races N threads onto one tree, so its node count and visit vector are
    not reproducible and a diff there would be uninterpretable (same reason as
    `test_gumbel_option_is_reported_inert_on_the_walker_pool`). What IS exact
    is the object the descent reads — `walker_pool.py` pulls `c_puct`,
    `fpu_reduction`, `cpuct_factor` and `cpuct_base` off `self._cfg` inside the
    descent — so assert every one of the four certified-[LIVE] fields on that
    config, and that it is a genuinely rebuilt object rather than the stale one.
    """
    engine = _make_engine(threads=2)
    try:
        before_pool = engine._worker._walker_pool
        assert before_pool is not None
        assert engine._worker.realized_search_path() == "walker"
  # DELTA, not an absolute: `SearchWorker.close()` does not close the walker
  # pool (pre-existing; the threads are daemons and production has one worker
  # for the process lifetime), so earlier tests in this file leave their own
  # walkers parked and an absolute count would measure them.
        threads_before = _live_walker_threads()
        before = before_pool._cfg
        for line in (
            "setoption name FpuReduction value -9.0",
            "setoption name CPuct value 99.0",
            "setoption name CPuctFactor value 7.0",
            "setoption name CPuctBase value 12345.0",
        ):
            _setoption(engine, line)
        after = engine._worker._walker_pool
        assert after is not None
        cfg = after._cfg
        assert cfg is not before, (
            "the walker pool was never rebuilt, so its threads still descend "
            "on the configuration they were constructed with"
        )
        assert (
            cfg.c_puct, cfg.fpu_reduction, cfg.cpuct_factor, cfg.cpuct_base,
        ) == pytest.approx((99.0, -9.0, 7.0, 12345.0)), (
            f"walker descent config is {cfg} after four setoptions the engine "
            f"reported as reaching the live search"
        )
  # The rebuild must SHUT DOWN the pool it replaced. Nothing observed this, so
  # dropping the `close()` passed every test while leaking two walker threads
  # per setoption on the shipped default — the growth is invisible until a
  # tournament's worth of `setoption`s has run.
        assert before_pool._shutdown.is_set(), (
            "the replaced walker pool was never closed; its threads are still "
            "parked on the old config"
        )
        assert _live_walker_threads() == threads_before, (
            f"walker threads {threads_before} -> {_live_walker_threads()} "
            f"across four rebuilds of a 2-walker pool — the replaced pools are "
            f"leaking two threads per setoption"
        )
  # Stated as a gap, not papered over: `rpg` needs >= 2 devices with evaluator
  # factories, so neither this file nor any reviewer could drive it
  # behaviourally. Its rebuild path (`_reinstall_rpg_if_active`) and
  # `pucv_pool`'s (`_install_multi_gpu_pucv_pool`) are covered only by the
  # structural `test_realized_search_path_mirrors_the_dispatch_branch_order`
  # and by inspection.
    finally:
        engine._worker.close()


@pytest.mark.parametrize("path", ["walker", "pucv"])
def test_the_readback_witnesses_the_descent_and_not_the_request(
    path: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """`searchconfig` must be independent evidence, not a restatement of it.

    `realized_search_values` sourced the PUCT family from `self._cfg` — the
    field a setter writes — while every PUCT descent reads its own
    construction-time snapshot. So the readback printed `CPuct = 99.0 [LIVE]`
    identically whether or not `rebuild_puct_helpers` ran, and the artifact
    whose stated purpose is proving to a TCEC/cutechess operator that the
    engine used what the config said could not witness the one defect this
    change exists to fix. A readback that re-reads the request is not evidence.

    Driven through `set_gumbel_field` alone — the public setter this commit
    proves is insufficient on a PUCT path — so the divergent state is produced
    legitimately rather than by patching the engine.
    """
    engine = _make_engine(threads=2) if path == "walker" else _make_pucv_engine()
    try:
        worker = engine._worker
        assert worker.realized_search_path() == path
        descent_before = worker._puct_descent_values()

        worker.set_gumbel_field("c_puct", 99.0)
        worker.set_gumbel_field("fpu_reduction", -9.0)
        assert worker._cfg.c_puct == 99.0  # the request did land on the config

        values = worker.realized_search_values()
        assert values["c_puct"] == pytest.approx(descent_before["c_puct"]), (
            "the readback reports the value that was REQUESTED, not the one "
            "the descent will read — it cannot witness a dropped rebuild"
        )
        assert values["fpu_reduction"] == pytest.approx(
            descent_before["fpu_reduction"],
        )

        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        stale = capsys.readouterr().out
        assert "CPuct = 99.0" not in stale, stale

  # ...and once the rebuild really runs, the readback follows the descent.
        _setoption(engine, "setoption name CPuct value 99.0")
        _setoption(engine, "setoption name FpuReduction value -9.0")
        descent_after = worker._puct_descent_values()
        assert descent_after["c_puct"] == pytest.approx(99.0)
        healthy = worker.realized_search_values()
        for field, value in descent_after.items():
            assert healthy[field] == pytest.approx(value), field

        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        fixed = capsys.readouterr().out
        assert "CPuct = 99.0 [LIVE]" in fixed
        assert fixed != stale, (
            "searchconfig is byte-identical with and without the rebuild"
        )
    finally:
        engine._worker.close()


def test_every_live_row_on_a_puct_path_is_read_off_the_descent_object() -> None:
    """No `[LIVE]` row on `walker`/`pucv` may be sourced from the request.

    The registry is the moving part: adding an option with
    `live_in=_PUCT_PATHS` whose value the pool snapshots would silently
    reintroduce the F1 gap for that option. This fails when that happens,
    unless the new field is added to `_PUCT_DESCENT_FIELDS` too.
    """
    from chess_anti_engine.uci.search import _PUCT_DESCENT_FIELDS

    for path, factory in (("walker", lambda: _make_engine(threads=2)),
                          ("pucv", _make_pucv_engine)):
        engine = factory()
        try:
            worker = engine._worker
            assert worker.realized_search_path() == path
            values = worker.realized_search_values()
            live = {
                o.field for o in SEARCH_OPTIONS
                if inert_reason(o, path, values) is None
            }
  # `chunk_sims` / `minibatch_size` / `root_noise_scale` are worker fields the
  # search reads directly, so the worker IS their live object.
            worker_owned = {"chunk_sims", "minibatch_size", "root_noise_scale"}
            unwitnessed = live - set(_PUCT_DESCENT_FIELDS) - worker_owned
            assert not unwitnessed, (
                f"on path={path} these are certified [LIVE] but the readback "
                f"reads them off the shared GumbelConfig, which the descent "
                f"does not: {sorted(unwitnessed)}"
            )
        finally:
            engine._worker.close()


def test_a_mid_tree_puct_change_keeps_the_tree_and_is_a_hybrid() -> None:
    """Pins the one judgement call in `rebuild_puct_helpers`: no `reset_tree`.

    Adding `reset_tree()` there is a defensible choice and it passed every test
    in this file, so the decision was pinned by nothing — by this PR's own
    standard, a deliberate behavioural choice no mutant can fire on is not
    closed. Two assertions, and the second is the honest one:

    * the tree SURVIVES the setoption (node count unchanged by the command
      itself). This is what a `reset_tree()` would break.
    * the resulting search is therefore a HYBRID — arm C matches neither the
      old configuration nor the new one, because the visits already in the tree
      were bought by the old constant. The docstring states this as a
      trade-off; this asserts that it really is one rather than a caveat.
    """
    engine = _make_pucv_engine()
    try:
        _signature(engine)  # first go: builds the tree
        worker = engine._worker
        assert worker._tree is not None
        before_nodes = worker._tree.node_count()
        _setoption(engine, "setoption name CPuct value 99.0")
        assert worker._tree is not None, (
            "the tree was discarded by a CPuct setoption — a mid-game or "
            "mid-ponder `setoption` now throws away the search so far"
        )
        assert worker._tree.node_count() == before_nodes, (
            f"tree went {before_nodes} -> {worker._tree.node_count()} on a "
            f"CPuct setoption; rebuild_puct_helpers must not reset the tree"
        )
        hybrid = _signature(engine)
    finally:
        engine._worker.close()

  # Same two-`go` shape, so the comparison is against like tree budgets.
    def _two_gos(cfg_over: dict[str, float] | None = None):
        e = _make_pucv_engine(cfg_over)
        try:
            _signature(e)
            return _signature(e)
        finally:
            e._worker.close()

    old_regime = _two_gos()
    new_regime = _two_gos({"c_puct": 99.0})
    assert old_regime != new_regime, (
        "harness precondition: the two regimes must differ, or 'neither' is "
        "vacuous"
    )
    assert hybrid != old_regime, (
        f"a mid-tree CPuct change was fully ignored: {hybrid} == {old_regime}"
    )
    assert hybrid != new_regime, (
        f"a mid-tree CPuct change produced the pure new-regime search "
        f"{new_regime} — if this ever becomes true the tree is being reset, "
        f"and the docstring's trade-off no longer describes the code"
    )


# --- a NO EFFECT claim must be true of the TRANSITION, not of an arm --------

# (label, context applied to both arms, option, value A, value B).
#
# The bug this table exists for: `inert_reason` answered "are all values inside
# this arm equivalent?" while `Engine._set_search_option` used the answer to
# assert "this setoption had no effect". They diverge on any setoption that
# CROSSES an arm boundary, and the engine then printed `NO EFFECT on the live
# search` for a command that changed its own move. The first two rows are that
# case, in both directions -- the reverse one printed "Only crossing to >= 0 can
# change anything" having just crossed from >= 0.
#
# The old cell list was all within-arm, which is exactly why it passed.
_SETOPTION_EFFECT_CELLS: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    ("QVisitExpRoot CROSS-ARM at CScaleRoot=0.05, LOG -> POWER",
     ("setoption name CScaleRoot value 0.05",), "QVisitExpRoot", "-1.0", "1.0"),
    ("QVisitExpRoot CROSS-ARM at CScaleRoot=0.05, POWER -> LOG",
     ("setoption name CScaleRoot value 0.05",), "QVisitExpRoot", "1.0", "-1.0"),
    ("QVisitExpRoot CROSS-ARM at the shipped CScaleRoot=7",
     (), "QVisitExpRoot", "-1.0", "1.0"),
    ("QVisitExpRoot within the LOG arm", (), "QVisitExpRoot", "-1.0", "-10"),
    ("QVisitExpRoot within the POWER arm", (), "QVisitExpRoot", "0", "2"),
    ("QVisitExpRoot into the >=90 sentinel", (), "QVisitExpRoot", "98", "95"),
    # A genuinely unreachable knob: NO EFFECT here must stay true.
    ("CPuct under classic Gumbel", (), "CPuct", "1.75", "9.5"),
)


def _apply_and_observe(
    context: tuple[str, ...], option: str, value: str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, tuple[int, int, tuple[int, ...]]]:
    """Set one option through the real dispatch; return (its info string, search)."""
    engine = _make_engine()
    try:
        for line in context:
            _setoption(engine, line)
        capsys.readouterr()
        _setoption(engine, f"setoption name {option} value {value}")
        message = "".join(
            line for line in capsys.readouterr().out.splitlines(keepends=True)
            if option in line
        )
        return message, _signature(engine)
    finally:
        engine._worker.close()


@pytest.mark.parametrize(
    ("label", "context", "option", "value_a", "value_b"), _SETOPTION_EFFECT_CELLS,
    ids=[c[0] for c in _SETOPTION_EFFECT_CELLS],
)
def test_a_no_effect_report_implies_the_search_did_not_move(
    label: str, context: tuple[str, ...], option: str,
    value_a: str, value_b: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """THE deciding test, and it is on the CLAIM the engine actually makes.

    The previous version asserted `inert_reason(...) is not None => identical
    search`, which is a statement about a predicate. The operator never sees the
    predicate; they see `NO EFFECT on the live search`. So the implication that
    has to hold is:

        the engine printed NO EFFECT  =>  the search is byte-identical.

    Cross-arm cells are included precisely because the old list had none.
    """
    _, sig_a = _apply_and_observe(context, option, value_a, capsys)
    message, sig_b = _apply_and_observe(context, option, value_b, capsys)

    if "NO EFFECT" in message:
        assert sig_a == sig_b, (
            f"{label}: the engine reported NO EFFECT for "
            f"{option}={value_b} but the search MOVED: {sig_a[:2]} vs "
            f"{sig_b[:2]}. A knob that took effect and was reported as ignored "
            "is the same defect as one silently dropped, sign-flipped.\n"
            f"message: {message.strip()}"
        )


def test_the_reviewer_cross_arm_case_reports_honestly_and_moves_the_search(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Named regression for the exact reproduction that blocked this PR.

    `CScaleRoot 0.05`, `QVisitExpRoot -1.0 -> 1.0`: best root action 306 -> 553,
    tree 8037 -> 7764, and the engine used to call it NO EFFECT.
    """
    context = ("setoption name CScaleRoot value 0.05",)
    _, sig_log = _apply_and_observe(context, "QVisitExpRoot", "-1.0", capsys)
    message, sig_pow = _apply_and_observe(context, "QVisitExpRoot", "1.0", capsys)

    assert sig_log != sig_pow, (
        "harness precondition: this transition must move the search, otherwise "
        "the test cannot detect a false NO EFFECT"
    )
    assert "NO EFFECT" not in message
    assert "reaches the live search" in message


def test_the_shipped_root_exponent_is_branch_pinned_not_inert() -> None:
    """`QVisitExpRoot = -1.0` selects the shipped LOG root transform.

    Reporting it INERT, in the same column as `CPuct` (which genuinely cannot
    reach this path), told an operator proving their config that a load-bearing
    parameter was doing nothing.
    """
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    opt = OPTIONS_BY_NAME["qvisitexproot"]
    assert values["q_visit_exp_root"] == float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"])

    assert inert_reason(opt, "gumbel", values) is None
    note = branch_note(opt, "gumbel", values)
    assert note is not None
    assert "every value < 0 is the same search" in note

    rows = {name: (status, why) for name, _, status, why in realized_rows("gumbel", values)}
    assert rows["QVisitExpRoot"][0] == "BRANCH"
    assert rows["CPuct"][0] == "INERT"


def test_the_branch_note_is_exact_and_tracks_the_current_value() -> None:
    """Each arm is read off the C, so the note must follow the value it is about.

    Also a gate-can-fail check: an option with no branch structure must get no
    note on any path.
    """
    opt = OPTIONS_BY_NAME["qvisitexproot"]
    for value, expected in (
        (-1.0, "every value < 0"), (-10.0, "every value < 0"),
        (0.0, "the power branch"), (2.0, "the power branch"),
        (98.0, "every value in [90, 99]"),
    ):
        values = {o.field: o.default for o in SEARCH_OPTIONS}
        values["q_visit_exp_root"] = value
        for path in ("gumbel", "rpg"):
            note = branch_note(opt, path, values)
            assert note is not None, (path, value)
            assert expected in note, (path, value, note)

    values = {o.field: o.default for o in SEARCH_OPTIONS}
    for name in ("topk", "cscale", "policytemperature", "halvingdiv"):
        for path in SEARCH_PATHS:
            assert branch_note(OPTIONS_BY_NAME[name], path, values) is None


def test_searchconfig_counts_branch_pinned_apart_from_inert(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The readback's summary line must not fold the two together."""
    engine = _make_engine()
    try:
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        out = capsys.readouterr().out
    finally:
        engine._worker.close()

    assert "QVisitExpRoot = -1.0 [BRANCH]" in out
    # Value not pinned: the harness's realized CPuct is GumbelConfig's, not the
    # registry default. The STATUS is what this test is about.
    assert re.search(r"searchconfig CPuct = \S+ \[INERT\]", out)
    assert "1 branch-pinned" in out


def test_the_documented_searchconfig_summary_is_the_one_the_engine_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`docs/operations.md`'s example must be output, not a recollection of it.

    It shipped saying `15 live, 4 inert` — the pre-BRANCH format, with a count
    that no build produces. A documented readback an operator uses to prove
    their config is worth nothing if the doc and the engine disagree about how
    many parameters are even reaching the search, so pin the doc line to the
    live one rather than re-proofreading it.
    """
    engine = _make_engine()
    try:
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        out = capsys.readouterr().out
    finally:
        engine._worker.close()

    live = re.search(r"searchconfig (\d+ live, \d+ branch-pinned, \d+ inert) "
                     r"on path=gumbel", out)
    assert live is not None, out

    doc = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "operations.md"
    ).read_text(encoding="utf-8")
    documented = re.findall(
        r"searchconfig (\d+ live, \d+ branch-pinned, \d+ inert) on path=gumbel",
        doc,
    )
    assert documented, "docs/operations.md no longer shows a searchconfig summary"
    for shown in documented:
        assert shown == live.group(1), (
            f"docs/operations.md documents 'searchconfig {shown} on "
            f"path=gumbel'; the engine prints '{live.group(1)}'"
        )


# --- audit_targets --gumbel passthrough --------------------------------------


def test_audit_targets_gumbel_overrides_reach_the_play_profile() -> None:
    from scripts.audit_targets import build_search_profiles, parse_gumbel_overrides

    overrides = parse_gumbel_overrides(["policy_temp=2.2", "topk=8,halving_div=4"])
    profiles = build_search_profiles(
        {}, play_sims=64, play_topk=None, gumbel_overrides=overrides,
    )
    assert profiles["search"].overrides == overrides
    # ...and NOT the training rows, which describe the stored target.
    assert profiles["train"].overrides == ()
    assert profiles["train_fast"].overrides == ()

    both = build_search_profiles(
        {}, play_sims=64, play_topk=None, gumbel_overrides=overrides,
        override_training_rows=True,
    )
    assert both["train"].overrides == overrides


@pytest.mark.parametrize(
    ("spec", "expect"),
    [
        ("c_puct=3", "cannot affect a Gumbel search"),
        ("fpu_reduction=0.1", "cannot affect a Gumbel search"),
        ("not_a_field=1", "is not a GumbelConfig field"),
        ("topk", "expected k=v"),
        ("topk=abc", "is not a number"),
    ],
)
def test_audit_targets_refuses_an_override_that_would_measure_a_null(
    spec: str, expect: str,
) -> None:
    from scripts.audit_targets import parse_gumbel_overrides

    with pytest.raises(SystemExit, match=re.escape(expect)):
        parse_gumbel_overrides([spec])


def test_audit_targets_aborts_when_an_override_fails_to_reach_the_config() -> None:
    """THE dispatch guard, tested by making the plumbing fail on purpose.

    `_SearchProfile` carries a FIXED field list, so before this guard an
    override for a field outside it parsed, printed in the run header, and was
    dropped — a complete, reproducible, wrong audit. The guard compares against
    the config the runner is about to be handed.
    """
    from scripts.audit_targets import _assert_overrides_realized

    cfg = GumbelConfig(policy_temp=1.0)
    with pytest.raises(SystemExit, match="did not reach the search config"):
        _assert_overrides_realized(
            cfg, (("policy_temp", 2.2),), where="test",
        )
    # Negative control: a config that DID take the override passes.
    _assert_overrides_realized(
        GumbelConfig(policy_temp=2.2), (("policy_temp", 2.2),), where="test",
    )


def test_audit_targets_override_coercion_keeps_int_fields_int() -> None:
    """`--gumbel topk=8` must not land a float on an int field: the C
    `start_gumbel_sims` signature rejects it mid-search, which fails long after
    the checkpoint and Stockfish have been paid for."""
    from scripts.audit_targets import _coerce_override

    assert isinstance(_coerce_override(GumbelConfig().topk, 8.0), int)
    assert isinstance(_coerce_override(GumbelConfig().c_scale, 8), float)
    assert _coerce_override(GumbelConfig().q_global_scale, 1.0) is True


def test_realized_search_path_mirrors_the_dispatch_branch_order() -> None:
    """`realized_search_path` is only trustworthy if it reads the same state
    `_run_one_chunk` branches on, in the same order."""
    import inspect

    src = inspect.getsource(SearchWorker._run_one_chunk)
    order: list[str] = []
    for m in re.findall(r"self\.(_rpg_pool|_pucv_pool|_walker_pool|_pucv) is not None", src):
        # The `searchmoves` guard tests `_rpg_pool` a second time before the
        # branch chain; dedupe so this compares dispatch ORDER, not mentions.
        if m not in order:
            order.append(m)
    reported = re.findall(
        r"self\.(_rpg_pool|_pucv_pool|_walker_pool|_pucv) is not None",
        inspect.getsource(SearchWorker.realized_search_path),
    )
    assert reported == order[: len(reported)]


# --- the seeded handshake must stay inside its own advertised bounds ---------
#
# Seeding is what makes the first `uci` print the CLI value, so it is also
# what can make the first `uci` print a value the option surface itself would
# refuse. Measured before the range check:
#
#     --topk 1        -> option name Topk type spin default 1 min 2 max 256
#     --policy-temp 0 -> option name PolicyTemperature type string default 0.0
#
# The first is self-contradictory on its own line; the second advertises a
# temperature `policy_temp_active` says the search does not run.


def test_the_advertised_default_is_inside_the_advertised_range() -> None:
    """THE F2 anti-rot gate, read off the PRINTED line.

    Parses `min`/`max` back out of the spin declarations rather than trusting
    `opt.lo`/`opt.hi`, so a declaration that contradicts ITSELF fails here even
    if the registry and the seeding agree with each other.
    """
    from chess_anti_engine.uci.__main__ import _build_parser, _startup_engine_options

    args = _build_parser().parse_args(["--checkpoint", "unused"])
    options = _startup_engine_options(
        args, search_parallel="pucv", restore_multi_gpu_pucv=False,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        emit_handshake(options)

    by_name = {o.name: o for o in SEARCH_OPTIONS}
    seen: set[str] = set()
    for line in buf.getvalue().splitlines():
        m = re.match(
            r"option name (\S+) type spin default (\S+) min (\S+) max (\S+)", line,
        )
        if m is None or m.group(1) not in by_name:
            continue
        name, default, lo, hi = m.group(1), *(float(g) for g in m.groups()[1:])
        seen.add(name)
        assert lo <= default <= hi, (
            f"the handshake advertises {name} default {default} outside its "
            f"OWN advertised range [{lo}, {hi}]"
        )
    assert {"Topk", "ChunkSims", "HalvingDiv", "VLossWeight"} <= seen, (
        "the spin options were not found in the printed handshake at all — "
        "this gate would pass vacuously"
    )

    # The float ("string") options print no min/max, so check those against the
    # registry the handler enforces.
    printed = _printed_handshake_defaults(options)
    for opt in SEARCH_OPTIONS:
        if opt.kind != "string" or opt.lo is None or opt.hi is None:
            continue
        assert opt.lo <= float(printed[opt.name]) <= opt.hi, (
            f"the handshake advertises {opt.name}={printed[opt.name]}, outside "
            f"the [{opt.lo}, {opt.hi}] its own `setoption` handler enforces"
        )


@pytest.mark.parametrize(
    ("flag", "value", "option"),
    [
        ("--policy-temp", "0", "PolicyTemperature"),
        ("--policy-temp", "0.01", "PolicyTemperature"),
        ("--policy-temp", "1e300", "PolicyTemperature"),
        ("--topk", "1", "Topk"),
        ("--topk", "9999", "Topk"),
        ("--halving-div", "1", "HalvingDiv"),
        ("--vloss-weight", "-1", "VLossWeight"),
        ("--fpu-reduction", "-99", "FpuReduction"),
    ],
)
def test_an_out_of_range_cli_value_is_refused_at_startup(
    flag: str, value: str, option: str,
) -> None:
    """A startup value the engine's own `setoption` would refuse must not start.

    Refusing (rather than clamping) is the deliberate contract: a clamp would
    run 2 for a requested 1 and report 2, which is the accepted-then-ignored
    defect with a friendlier face.
    """
    from chess_anti_engine.uci.__main__ import _build_parser, _startup_engine_options

    args = _build_parser().parse_args(["--checkpoint", "unused", flag, value])
    with pytest.raises(SystemExit) as exc:
        _startup_engine_options(
            args, search_parallel="pucv", restore_multi_gpu_pucv=False,
        )
    assert option in str(exc.value)
    assert flag in str(exc.value)


def test_every_cli_default_is_inside_its_advertised_range() -> None:
    """NEGATIVE CONTROL: the refusal above cannot fire on a bare command line.

    A range check whose own defaults are out of band would make the engine
    refuse to start at all — so this is the observation that proves the guard
    is a guard and not a brick.
    """
    from chess_anti_engine.uci.__main__ import (
        _SEARCH_OPTION_ARG,
        _build_parser,
        _startup_engine_options,
    )

    args = _build_parser().parse_args(["--checkpoint", "unused"])
    _startup_engine_options(
        args, search_parallel="pucv", restore_multi_gpu_pucv=False,
    )  # must not raise
    for opt in SEARCH_OPTIONS:
        dest = _SEARCH_OPTION_ARG[opt.field]
        if dest is None or opt.lo is None or opt.hi is None:
            continue
        raw = getattr(args, dest)
        if isinstance(raw, bool):
            continue
        assert opt.lo <= float(raw) <= opt.hi, (
            f"the CLI default for {opt.name} ({raw}) is outside [{opt.lo}, "
            f"{opt.hi}], so the engine would refuse to start with no flags"
        )
