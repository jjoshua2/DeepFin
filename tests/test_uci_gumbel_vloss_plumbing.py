"""The UCI classic-Gumbel path must apply virtual loss like its four siblings.

Same defect class as ``tests/test_selfplay_gumbel_batching_plumbing.py``: a
value that is accepted and then silently ignored. ``SearchWorker`` takes
``vloss_weight`` (UCI ``--vloss-weight``, default 3) and hands it to the walker
pool, the multi-GPU pucv pool, the root-parallel Gumbel pool and the pucv
chunker — but ``_run_gumbel_chunk`` passed only ``target_batch``, so the one
path that actually runs Gumbel (``--walkers 1``) ran with ``vloss_weight=0``:
the pre-C17 duplicate-leaf search, where a later halving rep re-walks an
unchanged tree onto a leaf already in flight and back-propagates the same value
twice. The flag existed, the field existed, and neither reached the search.

Assertions run the real ``_run_gumbel_chunk`` and capture what the C entry point
was called with, rather than reading source text: an AST check cannot tell
``vloss_weight=self._vloss_weight`` from ``vloss_weight=0``.
"""
from __future__ import annotations

import ast
import inspect
from typing import Any, cast

import chess
import numpy as np

from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_TARGET_BATCH,
    PLAY_SEARCH_VLOSS_WEIGHT,
    GumbelConfig,
)
from chess_anti_engine.uci import search as search_mod
from chess_anti_engine.uci.search import SearchWorker

# Every place SearchWorker hands a search a virtual-loss weight. A new search
# backend added here without `vloss_weight` is the exact regression this file
# exists for, so the sweep is over the call names rather than line numbers.
_SEARCH_LAUNCH_CALLS = {
    "run_gumbel_root_many_c",
    "WalkerPoolConfig",
    "MultiGpuPucvConfig",
    "RootParallelGumbelConfig",
    "PucvChunker",
}


def _bare_worker(*, vloss_weight: int, minibatch: int = 0) -> SearchWorker:
    """A SearchWorker carrying only what ``_run_gumbel_chunk`` reads.

    ``__init__`` wants a real evaluator and a compiled-extension ABI check; the
    chunk method itself only touches these fields.
    """
    w = SearchWorker.__new__(SearchWorker)
    w._device = "cpu"
    w._rng = np.random.default_rng(0)
    w._cfg = GumbelConfig(simulations=4)
    # Never called: the C entry point is stubbed out in every test below.
    w._evaluator = cast(Any, object())
    w._root_pol_logits = None
    w._root_wdl_logits = None
    w._tree = None
    w._root_id = None
    w._last_gumbel_action_idx = None
    w._minibatch_size = int(minibatch)
    w._vloss_weight = int(vloss_weight)
    return w


def _capture_chunk_kwargs(monkeypatch: Any, worker: SearchWorker) -> dict[str, Any]:
    """Run one real ``_run_gumbel_chunk`` and return the C call's kwargs."""
    seen: dict[str, Any] = {}

    def fake_gumbel(**kwargs: Any):
        seen.update(kwargs)
        # (probs, actions, values, masks, tree, root_ids) — the six the caller unpacks.
        return ([None], [0], [0.0], [None], object(), [0])

    monkeypatch.setattr(search_mod, "run_gumbel_root_many_c", fake_gumbel)
    worker._run_gumbel_chunk(4, chess.Board(), None)
    assert seen, "the C gumbel entry point was never called"
    return seen


def test_the_gumbel_chunk_passes_virtual_loss_to_the_search(monkeypatch: Any) -> None:
    """The defect proper: the kwarg was absent, so the C default 0 applied."""
    kwargs = _capture_chunk_kwargs(monkeypatch, _bare_worker(vloss_weight=3))

    assert "vloss_weight" in kwargs, (
        "the UCI Gumbel path drops vloss_weight again -- every --walkers 1 game "
        "is running the pre-C17 duplicate-leaf search"
    )


def test_the_chunk_passes_the_workers_value_not_a_literal(monkeypatch: Any) -> None:
    """Guards the 'passed but wired to a constant' variant.

    ``vloss_weight=3`` written literally would satisfy the test above while
    leaving ``--vloss-weight`` dead -- which is how ``matrix_weight_decay``
    became decorative.
    """
    kwargs = _capture_chunk_kwargs(monkeypatch, _bare_worker(vloss_weight=7))

    assert kwargs["vloss_weight"] == 7
    assert kwargs["vloss_weight"] != PLAY_SEARCH_VLOSS_WEIGHT


def test_disabling_virtual_loss_still_reaches_the_search(monkeypatch: Any) -> None:
    """0 must survive too: it is the A/B control arm for the C17 measurement."""
    kwargs = _capture_chunk_kwargs(monkeypatch, _bare_worker(vloss_weight=0))

    assert kwargs["vloss_weight"] == 0


def test_the_minibatch_knob_still_reaches_the_search(monkeypatch: Any) -> None:
    """``target_batch`` was already wired; keep it wired next to the new kwarg."""
    kwargs = _capture_chunk_kwargs(monkeypatch, _bare_worker(vloss_weight=3, minibatch=64))

    assert kwargs["target_batch"] == 64


def test_every_search_backend_in_the_worker_is_handed_a_virtual_loss() -> None:
    """A future sixth backend must not repeat the omission.

    Source-level on purpose: this is a coverage claim over call SITES, which is
    not observable from any single search run.
    """
    tree = ast.parse(inspect.getsource(search_mod))
    missing: list[str] = []
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in _SEARCH_LAUNCH_CALLS:
            continue
        found.add(name)
        if not any(kw.arg == "vloss_weight" for kw in node.keywords):
            missing.append(f"{name} at line {node.lineno}")

    assert found == _SEARCH_LAUNCH_CALLS, (
        f"search backends moved or were renamed: expected {sorted(_SEARCH_LAUNCH_CALLS)}, "
        f"found {sorted(found)}. Re-point this sweep, do not delete it."
    )
    assert not missing, f"search launched without a virtual-loss weight: {missing}"


def test_the_engine_flag_and_the_worker_default_are_the_same_constant() -> None:
    """The UCI default and the arena's 'play' shape must not drift apart.

    ``scripts/arena_standard.py --search-shape play`` claims to reproduce the
    engine's search; it reads PLAY_SEARCH_VLOSS_WEIGHT, so the engine has to be
    reading it too, not a duplicated literal.

    The argparse half is checked structurally because the UCI parser is built
    inline inside ``main()`` and cannot be obtained without starting an engine.
    A literal ``default=3`` is an ``ast.Constant`` and fails here.
    """
    from chess_anti_engine.uci import __main__ as uci_main

    sig_default = inspect.signature(SearchWorker.__init__).parameters["vloss_weight"].default
    assert sig_default == PLAY_SEARCH_VLOSS_WEIGHT

    defaults: list[ast.expr] = []
    for node in ast.walk(ast.parse(inspect.getsource(uci_main.main))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "--vloss-weight":
            defaults += [kw.value for kw in node.keywords if kw.arg == "default"]

    assert len(defaults) == 1, "--vloss-weight is not declared exactly once"
    assert isinstance(defaults[0], ast.Name), (
        "--vloss-weight has a hard-coded default again; it must read "
        "PLAY_SEARCH_VLOSS_WEIGHT so the engine and the arena cannot diverge"
    )
    assert defaults[0].id == "PLAY_SEARCH_VLOSS_WEIGHT"


def test_the_play_constants_are_todays_engine_behaviour() -> None:
    """Introducing the constants must not have moved the numbers."""
    assert PLAY_SEARCH_VLOSS_WEIGHT == 3  # lc0 default, walker mode since 2026-04
    assert PLAY_SEARCH_TARGET_BATCH == 0  # UCI MinibatchSize default = GSS_GPU_BATCH
