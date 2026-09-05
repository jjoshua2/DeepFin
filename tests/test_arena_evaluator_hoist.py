"""The arena's evaluator must be hoisted, and it must ARRIVE at the search.

``scripts/arena_standard.py``'s matched_sims loops called
``selfplay/match.pick_moves_for_boards`` with no evaluator, so
``mcts/gumbel_c.run_gumbel_root_many_c`` built a THROWAWAY
``LocalModelEvaluator`` per call -- per side, per ply. Each instance lazily
creates its own CUDA stream; torch hands streams out of a fixed round-robin
pool of 32 per device and the caching allocator partitions its segments BY
STREAM, so a two-model arena cycles the whole pool in 16 plies and each stream
ends up holding a full forward's working set. Reserved VRAM inflates until the
card OOMs. The same throwaway also has no ``_max_batch``, which is the only
thing ``gumbel_c`` mins its leaf cap against, so the forward batch grew with
concurrency without bound.

The hazard in the FIX is the house defect: an ``evaluator=`` that
``pick_moves_for_boards`` accepts and then drops before the C entry point. That
version passes every test that only checks the arena still runs, and it would
leave the OOM exactly where it was. So the tests here assert IDENTITY at the
consumer -- the object the search entry point was called with must BE the object
the arena built -- and they assert it per SIDE, because one evaluator threaded
to both sides would satisfy a weaker check while playing the candidate's
weights for the reference.
"""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

import scripts.arena_standard as arena
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import supports_inplace_api
from chess_anti_engine.mcts.gumbel_c import leaf_buffer_rows
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.selfplay import match as match_mod
from scripts.arena_standard import (
    DEFAULT_EVAL_MAX_BATCH,
    SideSearch,
    arena_uncapped_leaf_rows,
    build_arena_evaluator,
    play_paired_games_matched_sims,
    play_paired_games_matched_sims_rolling,
)

_OPENING_FENS = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
)


def _opening_fens(n: int) -> list[str]:
    """``n`` distinct legal opening FENs, deterministically.

    TWO disjoint sources, not one list with a prefix: ``n <= 4`` returns the
    curated strings alone, so the small cases keep byte-identical files to the
    ones the older tests were written against, and any larger ``n`` returns a
    pure depth-2 enumeration from the start position. 400 of those are
    available and every (white move, black move) pair gives a different
    position, so no dedup can shrink the list below ``n``. Needed because a
    test that names ``--max-concurrent-games 128`` has to be able to SCHEDULE
    128 pairs: the arena bounds its leaf figure by the pool the schedule can
    fill, so four openings would silently make it an 8-game arena.
    """
    if n <= len(_OPENING_FENS):
        return list(_OPENING_FENS[:n])
    out: list[str] = []
    root = chess.Board()
    for m1 in root.legal_moves:
        after_white = root.copy(stack=False)
        after_white.push(m1)
        for m2 in after_white.legal_moves:
            after_black = after_white.copy(stack=False)
            after_black.push(m2)
            out.append(after_black.fen())
            if len(out) == n:
                return out
    raise AssertionError(f"only {len(out)} distinct openings available, need {n}")


def _openings_file(tmp_path: Path, n: int = 2) -> Path:
    # Named by count. A single shared name is a trap here: `_run` builds its
    # defaults (which call this with n=1) BEFORE applying overrides, so a test
    # asking for 2 openings would have its file truncated back to 1 by the
    # default it was trying to override.
    path = tmp_path / f"openings_{n}.fen"
    path.write_text("\n".join(_opening_fens(n)) + "\n")
    return path


class _StubEvaluator:
    """Satisfies the BatchEvaluator protocol; its IDENTITY is the assertion.

    ``evaluate_encoded`` is present because that is what makes it a
    ``BatchEvaluator`` rather than an ``Any`` smuggled past the type checker --
    and it RAISES, because every search entry point in these tests is
    monkeypatched, so a real call means the test wired up something other than
    what it claims to be testing. A stub that silently returned plausible
    tensors would hide that.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Named for protocol parity: gumbel_c calls `relations` by keyword.
        del x, relations
        raise AssertionError(
            f"stub evaluator {self.name} was actually called; these tests assert "
            "identity at a monkeypatched search entry point and never evaluate"
        )

    def __repr__(self) -> str:
        return f"<stub {self.name}>"


class _DummyModel(torch.nn.Module):
    """Two heads, which is all a Gumbel search reads."""

    def __init__(self, tag: str = "m") -> None:
        super().__init__()
        self.tag = tag
        self._inference_only = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = int(x.shape[0])
        return {
            "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.zeros((bs, 3), dtype=torch.float32),
        }


def _side(**kwargs: Any) -> SideSearch:
    base: dict[str, Any] = {
        "shape": "test", "source": "test", "gumbel": {},
        "vloss_weight": 0, "target_batch": 0,
    }
    base.update(kwargs)
    return SideSearch(**base)


def _fake_search_result(boards: Sequence[chess.Board]) -> tuple:
    """A stub search result whose chosen action is LEGAL on each board.

    ⚑ The action ids used to be a flat ``[0] * n``, which worked only while the
    arena applied them non-strictly and substituted the first legal move for an
    id that decoded to nothing. Both arena loops now pass ``strict=True`` --
    deliberately, because they are the deciding Elo instrument and silently
    playing an unrelated legal move turns an action-space regression into
    unattributable Elo loss -- so a stub returning id 0 raises
    ``ActionDecodeError`` before the assertion under test is ever reached.

    Encoded from each board's own first legal move rather than pinned to a
    constant, since the ids are position-dependent and this helper is called on
    boards several plies deep.
    """
    actions = [
        int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards
    ]
    n = len(boards)
    return (
        [np.zeros(POLICY_SIZE, dtype=np.float32)] * n,
        actions,
        [0.0] * n,
        [np.ones(POLICY_SIZE, dtype=bool)] * n,
    )


def _capture(monkeypatch: Any, attr: str) -> list[tuple[Any, dict[str, Any]]]:
    """Record (model, kwargs) for every call to one search entry point.

    The MODEL is captured alongside the kwargs because the per-side property is
    the one a shared evaluator would silently break.
    """
    seen: list[tuple[Any, dict[str, Any]]] = []

    def fake(model: Any, boards: list[chess.Board], **kwargs: Any):
        seen.append((model, kwargs))
        return _fake_search_result(boards)

    monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", True)
    monkeypatch.setattr(match_mod, attr, fake)
    return seen


# ---------------------------------------------------------------------------
# The kwarg arrives at the consumer, by identity
# ---------------------------------------------------------------------------


def test_the_evaluator_reaches_the_c_search_entry_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plumbing proper: accepted here must mean received THERE.

    An implementation that took ``evaluator`` and never forwarded it would
    still play a full arena and still OOM; only identity at the entry point
    separates the two.
    """
    seen = _capture(monkeypatch, "_run_gumbel_root_many_c")
    stub = _StubEvaluator("cand")
    match_mod.pick_moves_for_boards(
        _DummyModel().eval(), [chess.Board()],
        device="cpu", rng=np.random.default_rng(0),
        mcts_type="gumbel", mcts_simulations=2, temperature=0.0, c_puct=2.5,
        gumbel_add_noise=False, evaluator=stub,
    )
    assert seen, "the C search was never invoked"
    for _model, kwargs in seen:
        assert kwargs["evaluator"] is stub, (
            "the evaluator was accepted by pick_moves_for_boards and did not "
            "arrive at run_gumbel_root_many_c"
        )


def test_omitting_it_is_todays_behaviour_for_every_other_caller() -> None:
    """Default None => gumbel_c builds its own, exactly as before this change."""
    params = inspect.signature(match_mod.pick_moves_for_boards).parameters
    assert params["evaluator"].default is None
    for fn in (play_paired_games_matched_sims, play_paired_games_matched_sims_rolling):
        loop_params = inspect.signature(fn).parameters
        assert loop_params["evaluator_candidate"].default is None
        assert loop_params["evaluator_reference"].default is None


def test_omitting_it_lands_as_an_explicit_none_at_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not merely 'absent from kwargs': gumbel_c keys its fallback on None."""
    seen = _capture(monkeypatch, "_run_gumbel_root_many_c")
    match_mod.pick_moves_for_boards(
        _DummyModel().eval(), [chess.Board()],
        device="cpu", rng=np.random.default_rng(0),
        mcts_type="gumbel", mcts_simulations=2, temperature=0.0, c_puct=2.5,
        gumbel_add_noise=False,
    )
    assert seen
    for _model, kwargs in seen:
        assert kwargs["evaluator"] is None


def test_the_python_gumbel_fallback_is_handed_it_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must not be where a hoisted evaluator quietly disappears.

    Volatility search forces the Python path. ``run_gumbel_root_many`` accepts
    ``evaluator``, so dropping it there would be a silent per-call evaluator on
    exactly the path the C search cannot serve.
    """
    seen: list[dict[str, Any]] = []

    def fake(_model: Any, boards: list[chess.Board], **kwargs: Any):
        seen.append(kwargs)
        return _fake_search_result(boards)

    monkeypatch.setattr(match_mod, "run_gumbel_root_many", fake)
    stub = _StubEvaluator("py")
    match_mod.pick_moves_for_boards(
        _DummyModel().eval(), [chess.Board()],
        device="cpu", rng=np.random.default_rng(0),
        mcts_type="gumbel", mcts_simulations=2, temperature=0.0, c_puct=2.5,
        gumbel_add_noise=False,
        volatility_q_scale=0.5,  # forces the Python path
        evaluator=stub,
    )
    assert seen, "the Python gumbel search was never invoked"
    assert seen[0]["evaluator"] is stub


def test_the_puct_path_is_handed_it_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """mcts_type='puct' is a different entry point with the same kwarg."""
    seen: list[dict[str, Any]] = []

    def fake(_model: Any, boards: list[chess.Board], **kwargs: Any):
        seen.append(kwargs)
        return _fake_search_result(boards)

    monkeypatch.setattr(match_mod, "_HAS_C_TREE", False)
    monkeypatch.setattr(match_mod, "run_mcts_many", fake)
    stub = _StubEvaluator("puct")
    match_mod.pick_moves_for_boards(
        _DummyModel().eval(), [chess.Board()],
        device="cpu", rng=np.random.default_rng(0),
        mcts_type="puct", mcts_simulations=2, temperature=0.0, c_puct=2.5,
        gumbel_add_noise=False, evaluator=stub,
    )
    assert seen, "the PUCT search was never invoked"
    assert seen[0]["evaluator"] is stub


# ---------------------------------------------------------------------------
# Both arena loops, and the right evaluator on the right side
# ---------------------------------------------------------------------------


def _assert_sides_paired(
    seen: list[tuple[Any, dict[str, Any]]],
    *,
    cand_model: Any, ref_model: Any,
    cand_ev: Any, ref_ev: Any,
) -> None:
    by_model = {id(cand_model): cand_ev, id(ref_model): ref_ev}
    saw = set()
    for model, kwargs in seen:
        expected = by_model[id(model)]
        assert kwargs["evaluator"] is expected, (
            f"model {getattr(model, 'tag', model)} was searched with "
            f"{kwargs['evaluator']!r}, expected {expected!r}"
        )
        saw.add(id(model))
    assert saw == {id(cand_model), id(ref_model)}, (
        "only one side ever moved; a per-side assertion over one side is vacuous"
    )


@pytest.mark.parametrize("loop", ["chunked", "rolling"])
def test_both_arena_loops_hand_each_side_its_own_evaluator(
    monkeypatch: pytest.MonkeyPatch, loop: str,
) -> None:
    """The two call sites the fix has to cover, and the swap it must not make.

    Parametrised rather than written once: the chunked and rolling loops are
    SEPARATE call sites into ``pick_moves_for_boards``, and the defect this
    replaces was present at both. A fix applied to only one would leave every
    rolling arena -- which is the default, and what the ratchet runs -- exactly
    as broken as before.
    """
    seen = _capture(monkeypatch, "_run_gumbel_root_many_c")
    cand_model = _DummyModel("cand").eval()
    ref_model = _DummyModel("ref").eval()
    cand_ev = _StubEvaluator("cand")
    ref_ev = _StubEvaluator("ref")
    board = chess.Board()
    board.push_uci("e2e4")
    side = _side()
    common: dict[str, Any] = {
        "device": "cpu", "rng": np.random.default_rng(0),
        "sims_candidate": 2, "sims_reference": 2,
        "max_plies": 4, "temperature": 0.0, "gumbel_add_noise": False,
        "search_candidate": side, "search_reference": side,
        "evaluator_candidate": cand_ev, "evaluator_reference": ref_ev,
    }
    if loop == "chunked":
        play_paired_games_matched_sims(cand_model, ref_model, [board], **common)
    else:
        play_paired_games_matched_sims_rolling(
            cand_model, ref_model, [board], pool_size=2, **common,
        )
    assert seen, "the C search was never invoked"
    _assert_sides_paired(
        seen, cand_model=cand_model, ref_model=ref_model,
        cand_ev=cand_ev, ref_ev=ref_ev,
    )


# ---------------------------------------------------------------------------
# What the arena actually builds
# ---------------------------------------------------------------------------


def test_the_built_evaluator_is_what_the_search_reads_off_it() -> None:
    """Read back through the exact expressions mcts/gumbel_c.py uses.

    ``_max_batch`` is read by ``getattr(eval_impl, "_max_batch", <uncapped>)``
    and is the ONLY cap on the leaf batch; ``supports_inplace_api`` (not a bare
    hasattr) is what gates the pinned zero-copy path; ``n_slots >= 2`` is what
    keeps the 2-group eval pipeline on for calls of >= 64 boards. An evaluator
    that failed any of these would be hoisted and then quietly not used as one.
    """
    ev = build_arena_evaluator(
        _DummyModel().eval(), device="cpu", max_batch=8,
    )
    assert getattr(ev, "_max_batch", None) == 8
    assert supports_inplace_api(ev) is True
    assert int(getattr(ev, "n_slots")) >= 2
    # Deliberately dense float32 leaves: DirectGPUEvaluator defaults
    # legal_bf16=True, and turning it on would switch the non-pipelined leaf
    # transport to BF16 logits softmaxed in C -- a numerics change against
    # every arena already in the ledger. LocalModelEvaluator (today's throwaway)
    # has no evaluate_legal_bf16 at all, so today's arena runs dense.
    assert bool(getattr(ev, "supports_legal_bf16")) is False
    # Volatility search reads this off the evaluator; losing it would make
    # --volatility-* fail loudly instead of silently, but it must not be lost.
    assert hasattr(ev, "evaluate_encoded_with_volatility")


# ---------------------------------------------------------------------------
# run_arena: the ordering, the guard, and the wiring
# ---------------------------------------------------------------------------


def _stub_model_loader(monkeypatch: pytest.MonkeyPatch) -> list[_DummyModel]:
    built: list[_DummyModel] = []

    def _load(path: str, **_kw: Any) -> _DummyModel:
        m = _DummyModel(str(path)).eval()
        built.append(m)
        return m

    monkeypatch.setattr(
        "chess_anti_engine.uci.model_loader.load_model_from_checkpoint", _load,
    )
    return built


def _stub_play_loops(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _spy(_c: Any, _r: Any, openings: list[chess.Board], **kw: Any) -> list[float]:
        seen.append(kw)
        return [1.0] * len(openings)

    for name in (
        "play_paired_games_matched_sims",
        "play_paired_games_matched_sims_rolling",
    ):
        monkeypatch.setattr(arena, name, _spy)
    return seen


def _run(tmp_path: Path, **overrides: Any) -> dict:
    side = _side()
    # ⚑ The default SCHEDULE is sized from the concurrency the caller named.
    # run_arena bounds its leaf-buffer figure by `arena_pool_size` — the games
    # that can actually be in flight — so `_run(max_concurrent_games=128)` over
    # one opening is a 2-board arena asking for 512 rows, not a 128-board one
    # asking for 4096, and every assertion about the bound for that concurrency
    # would be measuring the wrong pool. A test that wants a SHORT schedule
    # against a large ceiling overrides `games`/`openings_fen` explicitly.
    mcg = int(overrides.get("max_concurrent_games", 2))
    kwargs: dict[str, Any] = {
        "candidate": "cand.pt", "reference": "ref.pt", "games": 2 * mcg,
        "openings_path": None, "openings_fen": _openings_file(tmp_path, mcg),
        "opening_plies": 4, "mode": "matched_sims",
        "sims_candidate": 2, "sims_reference": 2, "ms_per_move": 0,
        "max_plies": 4, "temperature": 0.0, "gumbel_add_noise": False,
        "device": "cpu", "seed": 1, "out_path": None,
        "compile_models": False, "rolling": True, "max_concurrent_games": 2,
        "search_candidate": side, "search_reference": side,
        "game_log_path": tmp_path / "games.jsonl",
    }
    kwargs.update(overrides)
    return arena.run_arena(**kwargs)


def test_inference_only_is_set_on_both_models_before_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Ordering, observed from inside torch.compile itself.

    ``_inference_only`` selects a two-head branch of ``ChessNet.forward``, so
    setting it AFTER ``torch.compile`` changes a guard on an already-traced
    graph rather than the graph. The only structurally honest place to observe
    the order is the compile call: it records what the flag WAS at the moment
    the model was handed over.
    """
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    at_compile: list[bool] = []

    def _fake_compile(model: Any, *_a: Any, **_kw: Any) -> Any:
        at_compile.append(bool(getattr(model, "_inference_only")))
        return model

    monkeypatch.setattr(torch, "compile", _fake_compile)
    _run(tmp_path, compile_models=True)
    assert at_compile == [True, True], (
        "both models must already be inference-only when they reach "
        f"torch.compile; saw {at_compile}"
    )


def test_inference_only_is_withheld_when_volatility_needs_the_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The guard, and it is not cosmetic.

    Volatility-aware search reads the ``volatility`` head through
    ``LocalModelEvaluator.evaluate_encoded_with_volatility``, which substitutes
    ZEROS when the key is absent instead of raising. The two-head
    ``_inference_only`` branch does not emit that key, so setting it here would
    leave every ``--volatility-*`` arena searching with vol=0 on every node and
    reporting the result as a volatility measurement.
    """
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    _run(tmp_path, volatility_candidate={"volatility_q_scale": 0.5, "volatility_fpu": 0.0})
    assert len(built) == 2
    assert [m._inference_only for m in built] == [False, False]


@pytest.mark.parametrize("rolling", [True, False])
def test_run_arena_hands_the_hoisted_evaluators_to_the_play_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rolling: bool,
) -> None:
    """The last leg: what run_arena built is what the loop is called with.

    Driven at device='cuda' because the hoist is CUDA-only by design; the
    builder is stubbed, so no GPU is touched. Without this the loops could
    accept the kwargs and run_arena could still pass None to both.
    """
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    made: list[Any] = []

    def _fake_build(model: Any, **_kw: Any) -> Any:
        ev = _StubEvaluator(getattr(model, "tag", "?"))
        made.append(ev)
        return ev

    monkeypatch.setattr(arena, "build_arena_evaluator", _fake_build)
    _run(tmp_path, device="cuda", rolling=rolling)
    assert len(made) == 2, "one evaluator per side, built once for the whole run"
    assert seen, "no play loop ran"
    for kw in seen:
        assert kw["evaluator_candidate"] is made[0]
        assert kw["evaluator_reference"] is made[1]
    assert made[0] is not made[1], "the two sides must not share one evaluator"


def test_a_cpu_arena_does_not_hoist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """CUDA-only by design: no stream pool off CUDA, and ~0.5G of host buffers
    at the default cap would be pure cost. The reason is PRINTED, not silent."""
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)

    def _fake_build(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("build_arena_evaluator must not run on a CPU arena")

    monkeypatch.setattr(arena, "build_arena_evaluator", _fake_build)
    _run(tmp_path, device="cpu")
    assert seen
    for kw in seen:
        assert kw["evaluator_candidate"] is None
        assert kw["evaluator_reference"] is None


def test_eval_max_batch_zero_restores_the_per_call_evaluators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The documented opt-out has to actually opt out, on CUDA too."""
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)

    def _fake_build(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("--eval-max-batch 0 must not build an evaluator")

    monkeypatch.setattr(arena, "build_arena_evaluator", _fake_build)
    _run(tmp_path, device="cuda", eval_max_batch=0)
    assert seen
    for kw in seen:
        assert kw["evaluator_candidate"] is None
        assert kw["evaluator_reference"] is None


def test_a_cap_below_the_pool_is_refused_before_anything_is_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The C root submit is one whole side of the pool and is NOT bucketed
    against the cap, so a too-small cap raises inside get_input_buffer mid-run.
    Refuse at launch instead -- after a checkpoint load and a multi-minute
    compile is the wrong place to learn it."""
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    with pytest.raises(SystemExit, match="eval-max-batch"):
        _run(tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=32)
    assert built == [], "refused after loading a checkpoint, not before"


def test_the_cli_flag_reaches_run_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flag the parser advertises and main drops is the defect in miniature."""
    seen: dict[str, Any] = {}

    def _fake_run_arena(**kwargs: Any) -> dict:
        seen.update(kwargs)
        return {"elo": 0.0, "elo_ci95": [0.0, 0.0], "pairs": 0}

    monkeypatch.setattr(arena, "run_arena", _fake_run_arena)
    monkeypatch.setattr("sys.argv", [
        "arena_standard.py",
        "--candidate", "a.pt", "--reference", "b.pt",
        "--games", "2", "--mode", "matched_sims",
        "--search-shape", "play", "--eval-max-batch", "777",
    ])
    arena.main()
    assert seen.get("eval_max_batch") == 777


# ---------------------------------------------------------------------------
# --eval-max-batch is a SEARCH-SHAPE knob below the uncapped leaf buffer
#
# gumbel_c mins its leaf buffer against the evaluator's `_max_batch`, and when
# that buffer fills `_mcts_tree.c` does NOT flush and retry -- it appends the
# leaf as a SOLVED_UNKNOWN pseudo-terminal carrying the ROOT's Q
# (`stored_append_terminal(s, path_buf, path_len, g->root_qs[bi],
# SOLVED_UNKNOWN)`). Leaves beyond the buffer are absorbed, not evaluated, so a
# cap below what the search asked for changes which move is played. Reviewer
# measured 128 vs 4096 on CPU: 57-75% of leaf evaluations gone, 53/64 boards
# picking a different action.
# ---------------------------------------------------------------------------


def _original_pipelined_rows(n_boards: int, topk: int) -> int:
    """The pre-extraction expression, transcribed from `git show` of the parent."""
    mid = n_boards // 2
    max_grp = max(mid, n_boards - mid)
    return max(512, max_grp * max(2, int(topk)) * 2)


def _original_single_rows(n_boards: int, topk: int) -> int:
    """`_max_leaves_per_rep * 2`, the pre-extraction expression."""
    return max(256, n_boards * max(2, int(topk))) * 2


@pytest.mark.parametrize("topk", [2, 8, 16, 32, 64])
@pytest.mark.parametrize("n_boards", [1, 2, 7, 63, 64, 65, 128, 256])
def test_the_extracted_leaf_buffer_helper_is_value_identical(
    n_boards: int, topk: int,
) -> None:
    """The extraction into gumbel_c.leaf_buffer_rows must be a pure move.

    gumbel_c is shared with production selfplay, so this is the one file the
    branch was allowed to touch and the only acceptable change to it is one
    that cannot move a number.
    """
    assert leaf_buffer_rows(n_boards, topk=topk, pipelined=True) == (
        _original_pipelined_rows(n_boards, topk)
    )
    assert leaf_buffer_rows(n_boards, topk=topk, pipelined=False) == (
        _original_single_rows(n_boards, topk)
    )


def test_the_two_regimes_are_not_ordered_in_n() -> None:
    """Why the arena must take the max over BOTH, not evaluate one at its top.

    The single-buffer path applies below 64 boards and at topk 32 asks for more
    rows at 63 boards than the pipelined path does at 64. A bound computed from
    the pipelined path alone would under-report, and the warning would not fire
    on a cap that really does shrink the search.
    """
    assert leaf_buffer_rows(63, topk=32, pipelined=False) == 4032
    assert leaf_buffer_rows(64, topk=32, pipelined=True) == 2048
    assert leaf_buffer_rows(63, topk=32, pipelined=False) > (
        leaf_buffer_rows(64, topk=32, pipelined=True)
    )


class _RecordingDirectEvaluator(DirectGPUEvaluator):
    """A real evaluator that records the row counts the search asks it for."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.buffer_requests: list[tuple[int, int]] = []

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        self.buffer_requests.append((int(bsz), int(slot)))
        return super().get_input_buffer(bsz, slot=slot)


class _PlanesModel(_DummyModel):
    input_extra_features = "v1"


@pytest.mark.parametrize(
    # 65 is not redundant with 64. The pipelined expression splits the boards
    # and takes the CEIL half (`max(mid, n_boards - mid)`); at an even count
    # that equals `n_boards // 2`, so an even-only case cannot tell the real
    # expression from a floor-half copy of it. A mutant that inlined exactly
    # that drift at the call site survived the 64-board case.
    ("n_boards", "pipelined"), [(8, False), (64, True), (65, True)],
)
def test_the_search_sizes_its_leaf_buffer_through_the_helper(
    n_boards: int, pipelined: bool,
) -> None:
    """Consumer-side, executing: the helper and the CALL SITE cannot drift.

    A value-parity test alone would still pass if gumbel_c stopped calling the
    helper and kept an inline copy. This runs the real C search and reads the
    row count it actually requests from the evaluator, so the helper is checked
    where it is used, not where it is defined.
    """
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    topk = 8
    cfg = GumbelConfig(
        simulations=4, temperature=0.0, add_noise=False, topk=topk,
        input_extra_features="v1", policy_encoding="lc0_1858",
    )
    model = _PlanesModel().eval()
    ev = _RecordingDirectEvaluator(
        model, device="cpu", max_batch=16384, n_slots=2, use_amp=False,
        legal_bf16=False,
    )
    boards = [chess.Board() for _ in range(n_boards)]
    run_gumbel_root_many_c(
        model, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=ev, allow_terminal_root_shortcuts=True,
    )
    expected = leaf_buffer_rows(n_boards, topk=topk, pipelined=pipelined)
    asked = [rows for rows, _slot in ev.buffer_requests]
    assert expected in asked, (
        f"the search asked for {asked}; the helper says the {'pipelined' if pipelined else 'single'} "
        f"leaf buffer is {expected} rows -- helper and call site have drifted"
    )
    # Non-vacuous: the root submit is a different, smaller request, so the
    # assertion above is not just matching the root call by coincidence.
    assert (n_boards, 0) in ev.buffer_requests
    assert expected != n_boards


def test_the_arena_bound_takes_the_max_over_the_reachable_board_counts() -> None:
    """What --eval-max-batch is compared against at launch."""
    side = _side(gumbel={"topk": 32})
    got = arena_uncapped_leaf_rows(max_concurrent_games=128, sides=(side, side))
    assert got == max(
        leaf_buffer_rows(63, topk=32, pipelined=False),
        leaf_buffer_rows(128, topk=32, pipelined=True),
    )
    assert got == 4096, "the play shape at mcg 128 asks for 4096 leaf rows"
    # Below 64 boards the pipelined path is unreachable, so only the single one
    # counts -- and a None side (matched_time) contributes nothing.
    assert arena_uncapped_leaf_rows(
        max_concurrent_games=8, sides=(side, None),
    ) == leaf_buffer_rows(8, topk=32, pipelined=False)


def test_the_two_sides_can_ask_for_different_widths() -> None:
    """--cand-gumbel topk=N is per side; the bound must cover the wider one."""
    narrow, wide = _side(gumbel={"topk": 2}), _side(gumbel={"topk": 64})
    both = arena_uncapped_leaf_rows(max_concurrent_games=128, sides=(narrow, wide))
    assert both == arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(wide, wide),
    )
    assert both > arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(narrow, narrow),
    )


def test_a_cap_that_shrinks_the_search_warns_loudly_and_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """Allow-but-warn, the way compile is handled: it runs, and it is loud."""
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=512,
        search_candidate=side, search_reference=side,
    )
    err = capsys.readouterr().err
    assert "512" in err, "the cap must be named"
    assert "4032" in err, "the uncapped leaf-buffer size must be named"
    assert "SOLVED_UNKNOWN" in err, "the mechanism must be stated, not just the fact"
    assert "NOT COMPARABLE" in err.upper()
    assert seen, "a capped arena must still PLAY -- this is warn, not refuse"


def test_a_cap_at_or_above_the_uncapped_size_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """The default must not cry wolf: at mcg 128 / topk 32 it is exactly 4096."""
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side,
    )
    # Keyed on THIS warning's own mechanism string: the stubbed play loop
    # returns scores without writing rows, so the unrelated game-log
    # disagreement warning is on stderr too and a bare "WARNING" check would
    # pass for the wrong reason.
    assert "SOLVED_UNKNOWN" not in capsys.readouterr().err


def test_the_refusal_recommends_the_uncapped_size_not_the_bare_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Naming --max-concurrent-games would swap a loud crash for a quiet one.

    That value is exactly where the ROOT submit stops raising and the LEAF cap
    binds hardest, so recommending it is recommending a silently shrunken
    search.
    """
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    side = _side(gumbel={"topk": 32})
    with pytest.raises(SystemExit) as excinfo:
        _run(
            tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=32,
            search_candidate=side, search_reference=side,
        )
    message = str(excinfo.value)
    assert "--eval-max-batch 4032" in message, "must name the uncapped size"
    assert "--eval-max-batch 0" in message
    assert "SHRINKS THE SEARCH" in message


def test_a_negative_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """0 is the documented opt-out; a negative is a typo, on every path."""
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    with pytest.raises(SystemExit, match="must be >= 0"):
        _run(tmp_path, device="cuda", eval_max_batch=-1)
    assert built == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "cpu"},
        {"device": "cuda", "volatility_candidate": {"volatility_q_scale": 0.5,
                                                    "volatility_fpu": 0.0}},
    ],
    ids=["cpu", "volatility"],
)
def test_a_path_that_builds_no_evaluator_is_not_refused_for_a_small_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict,
) -> None:
    """A cap that can never bind must not refuse a run it cannot affect.

    The evaluator is only built on matched_sims + CUDA + volatility-off, so on
    every other path --eval-max-batch is inert and refusing would be a knob
    rejecting a configuration it has no bearing on.
    """
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    _run(tmp_path, max_concurrent_games=64, eval_max_batch=8, **overrides)
    assert seen, "the arena must have run"


def test_matched_time_is_not_refused_for_a_small_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """matched_time plays through UCI subprocesses; nothing here is hoisted."""
    monkeypatch.setattr(
        arena, "play_paired_games_matched_time",
        lambda *_a, **_kw: [1.0],
    )
    record = arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=2,
        openings_path=None, openings_fen=_openings_file(tmp_path, 1),
        opening_plies=4, mode="matched_time",
        sims_candidate=0, sims_reference=0, ms_per_move=10,
        max_plies=4, temperature=0.0, gumbel_add_noise=False,
        device="cuda", seed=1, out_path=None,
        max_concurrent_games=64, eval_max_batch=8,
        game_log_path=tmp_path / "mt.jsonl",
    )
    assert record["eval_hoist"] == "n/a"


# ---------------------------------------------------------------------------
# --eval-max-batch 0 is documented as restoring the PRE-HOIST arena
#
# The help text promises reproduction, so everything this branch added has to
# be off under it -- not just the evaluator. _inference_only traces a different
# graph under torch.compile, and the cache frees are new behaviour too.
# ---------------------------------------------------------------------------


def test_zero_leaves_the_ten_head_forward_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A 2-head forward is not what a pre-hoist arena ran."""
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    _run(tmp_path, device="cuda", eval_max_batch=0)
    assert [m._inference_only for m in built] == [False, False]


def test_a_normal_run_still_sets_inference_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The control for the test above: without the opt-out it IS set."""
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    _run(tmp_path, device="cuda")
    assert [m._inference_only for m in built] == [True, True]


@pytest.mark.parametrize("eval_max_batch", [0, DEFAULT_EVAL_MAX_BATCH])
def test_zero_frees_no_cache_between_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, eval_max_batch: int,
) -> None:
    """The chunk-boundary free is new behaviour, so 0 must not do it either."""
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    calls: list[str] = []
    monkeypatch.setattr(arena, "_free_cached_vram", calls.append)
    _run(tmp_path, device="cuda", rolling=False, eval_max_batch=eval_max_batch)
    assert (calls == []) is (eval_max_batch == 0), (
        f"eval_max_batch={eval_max_batch} produced {calls}"
    )


@pytest.mark.parametrize("eval_max_batch", [0, DEFAULT_EVAL_MAX_BATCH])
def test_the_rolling_loop_is_told_whether_to_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, eval_max_batch: int,
) -> None:
    """run_arena's half of the contract: the switch is threaded, not assumed."""
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    _run(tmp_path, device="cuda", rolling=True, eval_max_batch=eval_max_batch)
    assert seen
    for kw in seen:
        assert kw["free_cached_vram"] is bool(eval_max_batch)


@pytest.mark.parametrize("free_cached_vram", [True, False])
def test_the_rolling_loop_honours_the_free_switch(
    monkeypatch: pytest.MonkeyPatch, free_cached_vram: bool,
) -> None:
    """The loop's half: a threaded flag that the loop ignored would look green
    on the test above and still free the cache under --eval-max-batch 0."""
    calls: list[str] = []
    monkeypatch.setattr(arena, "_free_cached_vram", calls.append)
    seen = _capture(monkeypatch, "_run_gumbel_root_many_c")
    board = chess.Board()
    board.push_uci("e2e4")
    side = _side()
    play_paired_games_matched_sims_rolling(
        _DummyModel("c").eval(), _DummyModel("r").eval(), [board],
        device="cuda", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2, max_plies=4,
        temperature=0.0, gumbel_add_noise=False,
        search_candidate=side, search_reference=side,
        pool_size=2, free_cached_vram=free_cached_vram,
    )
    assert seen, "the loop never played"
    assert bool(calls) is free_cached_vram


# ---------------------------------------------------------------------------
# The hoist state is RECORDED -- the way compile is
# ---------------------------------------------------------------------------


def test_the_tag_distinguishes_the_four_states() -> None:
    """'128<4096' is the form that has to survive into a log read years later:
    eval_max_batch alone cannot say whether the cap BOUND, because that depends
    on topk and concurrency."""
    assert arena.hoist_tag(4096, mode="matched_time", no_hoist=None,
                           uncapped_leaf_rows=4096) == "n/a"
    assert arena.hoist_tag(0, mode="matched_sims", no_hoist="--eval-max-batch 0",
                           uncapped_leaf_rows=0) == "off"
    assert arena.hoist_tag(4096, mode="matched_sims", no_hoist=None,
                           uncapped_leaf_rows=4096) == "4096"
    assert arena.hoist_tag(128, mode="matched_sims", no_hoist=None,
                           uncapped_leaf_rows=4096) == "128<4096"


def test_a_row_written_before_the_field_reads_as_unknown_not_off() -> None:
    """A default that answers for rows the field never covered is how a
    resumed splice stops being visible."""
    assert arena.row_hoist_tag({}) == "unknown"
    assert arena.row_hoist_tag({"eval_hoist": None}) == "unknown"
    assert arena.row_hoist_tag({"eval_hoist": "128<4096"}) == "128<4096"


def test_the_result_record_carries_the_hoist_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=512,
        search_candidate=side, search_reference=side,
    )
    assert record["eval_hoist"] == "512<4032"
    assert record["eval_max_batch"] == 512
    assert record["eval_leaf_cap_uncapped"] == 4032
    assert record["eval_leaf_cap_bound"] is True
    assert record["mixed_eval_hoist"] is False


def test_an_unbound_run_records_that_it_was_unbound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The control: the same fields must read differently when nothing bound."""
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=64,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side,
    )
    assert record["eval_hoist"] == str(DEFAULT_EVAL_MAX_BATCH)
    assert record["eval_leaf_cap_bound"] is False
    assert record["eval_leaf_cap_uncapped"] == 4032


class _SimulatedCrash(RuntimeError):
    pass


def _emitting_play_loops(
    monkeypatch: pytest.MonkeyPatch, *, crash_after: int | None = None,
) -> None:
    """A play loop that writes real game rows, and can die mid-schedule."""
    emitted = {"n": 0}

    def _spy(
        _c: Any, _r: Any, openings: list[chess.Board], *,
        pgn_sink: Any = None, pair_ids: Any = None, **_kw: Any,
    ) -> list[float]:
        ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
        scores: list[float] = []
        for k, opening in enumerate(openings):
            for half, a_is_white in ((0, True), (1, False)):
                if pgn_sink is not None:
                    pgn_sink(
                        pair_id=ids[k], half=half, a_is_white=a_is_white,
                        start_fen=opening.fen(), moves=(), result="1/2-1/2",
                        termination="rules", plies=10, duration_s=0.5,
                        chunk=None, loop="stub",
                    )
                emitted["n"] += 1
                if crash_after is not None and emitted["n"] >= crash_after:
                    raise _SimulatedCrash("simulated OOM")
            scores.append(1.0)
        return scores

    for name in (
        "play_paired_games_matched_sims",
        "play_paired_games_matched_sims_rolling",
    ):
        monkeypatch.setattr(arena, name, _spy)


def test_a_played_game_row_records_the_hoist_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from chess_anti_engine.utils.game_log import read_game_log

    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    log_path = tmp_path / "rows.games.jsonl"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=512,
        search_candidate=side, search_reference=side, game_log_path=log_path,
    )
    rows = read_game_log(log_path).games
    assert rows, "no game rows were written"
    assert {arena.row_hoist_tag(r) for r in rows} == {"512<4032"}


def test_a_resume_across_two_hoist_settings_reports_the_mix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The whole point of recording it: a spliced run must not read as one arena.

    eval_max_batch is deliberately OUTSIDE the resume fingerprint -- a pre-hoist
    log has to stay resumable under post-hoist code -- so the mix it permits is
    surfaced instead of refused, exactly as for compile.
    """
    log_path = tmp_path / "mix.games.jsonl"
    openings = _openings_file(tmp_path, 2)
    side = _side(gumbel={"topk": 32})
    common: dict[str, Any] = {
        "games": 4, "openings_fen": openings, "max_concurrent_games": 64,
        "search_candidate": side, "search_reference": side,
        "game_log_path": log_path, "device": "cuda",
    }

    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch, crash_after=2)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    with pytest.raises(_SimulatedCrash):
        _run(tmp_path, eval_max_batch=DEFAULT_EVAL_MAX_BATCH, **common)

    # Resume under the opt-out: same fingerprint, different search transport.
    _emitting_play_loops(monkeypatch)
    record = _run(tmp_path, eval_max_batch=0, resume=True, **common)
    assert record["mixed_eval_hoist"] is True
    assert record["eval_hoist_values"] == ["4096", "off"]
    assert record["eval_hoist"] == "off", "the tag is what THIS process played"


def test_a_resume_that_scores_nothing_new_mixes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Gated on pairs SCORED, not scheduled -- same rule as mixed_compile.

    A resume of an already-complete log plays no game, so flagging a mix there
    would report a splice that never happened.
    """
    log_path = tmp_path / "done.games.jsonl"
    openings = _openings_file(tmp_path, 2)
    side = _side(gumbel={"topk": 32})
    common: dict[str, Any] = {
        "games": 4, "openings_fen": openings, "max_concurrent_games": 64,
        "search_candidate": side, "search_reference": side,
        "game_log_path": log_path, "device": "cuda",
    }
    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    _run(tmp_path, eval_max_batch=DEFAULT_EVAL_MAX_BATCH, **common)

    record = _run(tmp_path, eval_max_batch=0, resume=True, **common)
    assert record["mixed_eval_hoist"] is False
    assert record["eval_hoist_values"] == ["4096"]


def test_the_hoist_is_not_in_the_resume_fingerprint() -> None:
    """A pre-hoist log must resume under post-hoist code; a fingerprint over
    eval_max_batch would refuse exactly the retry this is for."""
    settings = arena.arena_game_log_settings(
        mode="matched_sims", candidate="a", reference="b", games=4, seed=1,
        openings_path="o", openings_kind="fen", opening_plies=4,
        sims_candidate=2, sims_reference=2, ms_per_move=None, max_plies=4,
        temperature=0.0, gumbel_add_noise=False,
        search_candidate=None, search_reference=None,
        volatility_candidate=None, uci_args="", syzygy_path=None, tb_max_pieces=6,
    )
    assert not [k for k in settings if "eval" in k or "hoist" in k]


# ---------------------------------------------------------------------------
# The recorded leaf-cap fields are POST-LOAD EXACT, not the launch-time floor
#
# The launch check must run before the checkpoints load (its refusal has to
# beat a multi-minute compile), so it can only assume relations OFF. A model
# with `use_dynamic_relations` forces `_use_pipeline` False at every board
# count, so the single-buffer path runs at the real n: 8192 rows at mcg 128 /
# topk 32 where the floor said 4096. A cap in [4096, 8192) is then bound in
# fact while nothing warns and the record says bound=False.
# ---------------------------------------------------------------------------


def test_relations_raise_the_uncapped_size_above_the_relations_off_floor() -> None:
    side = _side(gumbel={"topk": 32})
    floor = arena_uncapped_leaf_rows(max_concurrent_games=128, sides=(side, side))
    exact = arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(side, side), relations=(True, True),
    )
    assert floor == 4096
    assert exact == 8192, "relations force the single-buffer path at the real n"
    # The blind window the record used to misreport.
    assert floor <= 6000 < exact


def test_relations_are_per_side() -> None:
    """Two different checkpoints; only one of them may have the flag."""
    side = _side(gumbel={"topk": 32})
    one_on = arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(side, side), relations=(False, True),
    )
    assert one_on == arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(side, side), relations=(True, True),
    )
    assert one_on > arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(side, side), relations=(False, False),
    )


def test_omitting_relations_is_the_relations_off_answer() -> None:
    """The floor is not a separate formula, just the all-off case."""
    for n in (1, 8, 63, 64, 128, 256):
        for topk in (2, 16, 32):
            side = _side(gumbel={"topk": topk})
            assert arena_uncapped_leaf_rows(
                max_concurrent_games=n, sides=(side, side),
            ) == arena_uncapped_leaf_rows(
                max_concurrent_games=n, sides=(side, side), relations=(False, False),
            )


class _RelationsModel(_DummyModel):
    """A checkpoint that may compute dynamic relations.

    Declared on the class, not poked onto an instance: `nn.Module.__setattr__`
    is typed for Tensors and Modules, so a bare `m.use_dynamic_relations = True`
    is a type error. Kept OFF `_DummyModel` so that
    `test_a_model_without_the_attribute_is_treated_as_relations_off` still has a
    model that genuinely lacks the attribute, the way a pre-relations checkpoint
    does.
    """

    use_dynamic_relations: bool = False


def _stub_model_loader_with_relations(
    monkeypatch: pytest.MonkeyPatch, *, relations: bool | Sequence[bool],
) -> list[_RelationsModel]:
    """``relations`` may be per side, in LOAD order: candidate then reference.

    Per side because "reads the flag" and "reads BOTH sides' flags" are
    different properties, and a fixture that sets it on both models cannot
    tell them apart — a run_arena that only ever looked at the candidate would
    pass every same-on-both test.
    """
    flags = ([bool(relations)] * 2 if isinstance(relations, bool)
             else [bool(r) for r in relations])
    built: list[_RelationsModel] = []

    def _load(path: str, **_kw: Any) -> _RelationsModel:
        m = _RelationsModel(str(path)).eval()
        m.use_dynamic_relations = flags[len(built) % len(flags)]
        built.append(m)
        return m

    monkeypatch.setattr(
        "chess_anti_engine.uci.model_loader.load_model_from_checkpoint", _load,
    )
    return built


def test_a_relations_model_flips_the_recorded_bound_in_the_blind_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """The defect: cap 4096 at mcg 128 / topk 32 passes the pre-load floor
    (4096 >= 4096, no warning) but IS bound once relations are known (< 8192)."""
    _stub_model_loader_with_relations(monkeypatch, relations=True)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side,
    )
    assert record["eval_leaf_cap_uncapped"] == 8192, "the record must be post-load exact"
    assert record["eval_leaf_cap_bound"] is True
    assert record["eval_hoist"] == "4096<8192"
    err = capsys.readouterr().err
    assert "SOLVED_UNKNOWN" in err, "the late warning must still fire"
    assert "after loading the checkpoints" in err, "and say why it is late"


def test_a_relations_off_model_records_exactly_the_pre_load_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """Parity: the re-derivation must not move the common case.

    Same invocation as the test above with the flag off — every recorded field
    has to agree with the launch-time floor, and nothing may warn.
    """
    _stub_model_loader_with_relations(monkeypatch, relations=False)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side,
    )
    floor = arena_uncapped_leaf_rows(
        max_concurrent_games=128, sides=(side, side),
    )
    assert record["eval_leaf_cap_uncapped"] == floor == 4096
    assert record["eval_leaf_cap_bound"] is False
    assert record["eval_hoist"] == "4096"
    assert "SOLVED_UNKNOWN" not in capsys.readouterr().err


def test_a_model_without_the_attribute_is_treated_as_relations_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Older checkpoints have no such attribute; absent must mean off, not crash."""
    _stub_model_loader(monkeypatch)  # plain _DummyModel: no use_dynamic_relations
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side,
    )
    assert record["eval_leaf_cap_uncapped"] == 4096
    assert record["eval_leaf_cap_bound"] is False


def test_the_corrected_tag_reaches_the_game_rows_not_just_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The record is written once at the end; rows are written DURING play.

    Re-deriving into the record alone would leave every row carrying the stale
    pre-load tag, and the rows are what a resume reads.
    """
    from chess_anti_engine.utils.game_log import read_game_log

    _stub_model_loader_with_relations(monkeypatch, relations=True)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    log_path = tmp_path / "relrows.games.jsonl"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side, game_log_path=log_path,
    )
    rows = read_game_log(log_path).games
    assert rows
    assert {arena.row_hoist_tag(r) for r in rows} == {"4096<8192"}


def test_a_relations_flag_on_either_side_alone_is_seen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Both sides' models must be read, not just the candidate's.

    The two sides are different checkpoints and the arena's own bound is the max
    over them, so a re-derivation that inspected only `model_candidate` would
    under-report exactly when the reference is the relations-on net — and every
    fixture that sets the flag on both models would still pass.
    """
    side = _side(gumbel={"topk": 32})
    for flags in ((True, False), (False, True)):
        with monkeypatch.context() as ctx:
            _stub_model_loader_with_relations(ctx, relations=flags)
            _stub_play_loops(ctx)
            ctx.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
            record = _run(
                tmp_path, device="cuda", max_concurrent_games=128,
                eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
                search_candidate=side, search_reference=side,
                game_log_path=tmp_path / f"side_{flags}.games.jsonl",
            )
        assert record["eval_leaf_cap_uncapped"] == 8192, f"missed relations on {flags}"
        assert record["eval_leaf_cap_bound"] is True


def test_a_relations_list_that_does_not_match_the_sides_is_refused() -> None:
    """The two sequences are zipped; a length mismatch would silently pair a
    side with the wrong model's flag rather than fail."""
    side = _side(gumbel={"topk": 32})
    with pytest.raises(ValueError, match="relations has 1 entries for 2 sides"):
        arena_uncapped_leaf_rows(
            max_concurrent_games=128, sides=(side, side), relations=(True,),
        )


# ---------------------------------------------------------------------------
# The bound is taken over the pool the SCHEDULE can fill, not the ceiling
#
# `--max-concurrent-games` is a ceiling. Both matched_sims loops feed from a
# queue of `2 * n_pairs` games, so a 2-game smoke arena at the default 128 has
# two boards alive and its search asks for 512 leaf rows, not 4096. Computing
# the bound from the ceiling made `--eval-max-batch 512` there print the "NOT
# COMPARABLE" warning and record `eval_leaf_cap_bound=true` about a search it
# never touched -- and refused smaller caps against a root submit carrying two
# boards. Wrong in the safe direction is still wrong.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mcg", "n_pairs", "expected"),
    [
        (128, 1, 2),      # the 2-game smoke arena: two boards, never 128
        (128, 2, 4),
        (128, 64, 128),   # exactly fills the ceiling
        (128, 1000, 128), # a long schedule cannot exceed it
        (1, 1000, 1),
        (128, 0, 1),      # a fully-resumed schedule still asks for >= 1
    ],
)
def test_the_pool_is_the_ceiling_capped_by_the_schedule(
    mcg: int, n_pairs: int, expected: int,
) -> None:
    assert arena.arena_pool_size(max_concurrent_games=mcg, n_pairs=n_pairs) == expected


def test_the_bound_follows_the_schedule_not_the_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """One cap, one ceiling, two schedule lengths — and they must READ apart.

    This is the whole change in one assertion. Both arms pass
    ``--max-concurrent-games 128`` and ``--eval-max-batch 512``; only the number
    of loaded opening pairs differs. The short arm can only ever put 2 boards in
    flight, so 512 is at the uncapped size and inert; the full arm reaches 128
    boards, wants 4096 rows, and 512 genuinely shrinks its search.
    """
    side = _side(gumbel={"topk": 32})
    records: dict[str, dict] = {}
    for label, n_pairs in (("short", 1), ("full", 128)):
        with monkeypatch.context() as ctx:
            _stub_model_loader(ctx)
            _stub_play_loops(ctx)
            ctx.setattr(
                arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"),
            )
            records[label] = _run(
                tmp_path, device="cuda", max_concurrent_games=128,
                eval_max_batch=512,
                search_candidate=side, search_reference=side,
                games=2 * n_pairs, openings_fen=_openings_file(tmp_path, n_pairs),
                game_log_path=tmp_path / f"pool_{label}.games.jsonl",
            )
    assert records["short"]["eval_leaf_cap_uncapped"] == 512
    assert records["short"]["eval_leaf_cap_bound"] is False
    assert records["short"]["eval_hoist"] == "512"
    assert records["full"]["eval_leaf_cap_uncapped"] == 4096
    assert records["full"]["eval_leaf_cap_bound"] is True
    assert records["full"]["eval_hoist"] == "512<4096"


def test_a_cap_inert_for_the_real_pool_does_not_cry_wolf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """The warning is the operator-facing half of the same defect.

    Keyed on this warning's own mechanism string: the stubbed play loop writes
    no rows, so the unrelated game-log disagreement warning is on stderr too and
    a bare "WARNING" check would pass for the wrong reason.
    """
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=128, eval_max_batch=512,
        search_candidate=side, search_reference=side,
        games=2, openings_fen=_openings_file(tmp_path, 1),
    )
    assert "SOLVED_UNKNOWN" not in capsys.readouterr().err


def test_the_root_submit_refusal_is_against_the_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A 2-board root submit cannot overflow a cap of 32, so nothing is refused.

    It is still BELOW the 512-row leaf buffer those two boards ask for, so the
    run is warned and recorded — allow-but-warn, not refuse.
    """
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128, eval_max_batch=32,
        search_candidate=side, search_reference=side,
        games=2, openings_fen=_openings_file(tmp_path, 1),
    )
    assert seen, "a cap the pool can serve must not refuse the run"
    assert record["eval_hoist"] == "32<512"


def test_the_refusal_still_fires_when_the_pool_is_really_that_wide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The control for the test above: a schedule that FILLS 128 still refuses,
    and the message quotes the pool it was measured against."""
    built = _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    side = _side(gumbel={"topk": 32})
    with pytest.raises(SystemExit) as excinfo:
        _run(
            tmp_path, device="cuda", max_concurrent_games=128, eval_max_batch=32,
            search_candidate=side, search_reference=side,
        )
    message = str(excinfo.value)
    assert "pool of 128 concurrent game(s)" in message
    assert "128 loaded opening pair(s)" in message
    assert "--eval-max-batch 4096" in message, "must name the uncapped size"
    assert built == [], "refused after loading a checkpoint, not before"


def test_a_short_fen_list_shrinks_the_pool_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """The case ``--games`` alone cannot see.

    ``load_fen_openings`` uses ALL rows when the file holds fewer than
    ``--games / 2``, so a 256-game invocation over a 2-row list is a 4-game
    arena. A bound derived from the requested game count would still say 4096.
    """
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128, eval_max_batch=512,
        search_candidate=side, search_reference=side,
        games=256, openings_fen=_openings_file(tmp_path, 2),
    )
    assert "usable rows < 128 requested pairs" in capsys.readouterr().out
    assert record["eval_leaf_cap_uncapped"] == 512, "4 boards, not 128"
    assert record["eval_leaf_cap_bound"] is False
    assert record["eval_hoist"] == "512"


def test_the_pool_bound_survives_the_post_load_re_derive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The re-derive must add relations WITHOUT restoring the ceiling.

    Re-deriving from ``max_concurrent_games`` there would quietly undo the
    launch-time bound for every relations run, and the record — which is
    written from the post-load figures — is where it would show.
    """
    _stub_model_loader_with_relations(monkeypatch, relations=True)
    _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    record = _run(
        tmp_path, device="cuda", max_concurrent_games=128, eval_max_batch=512,
        search_candidate=side, search_reference=side,
        games=2, openings_fen=_openings_file(tmp_path, 1),
    )
    # Relations force the single-buffer path, which at 2 boards is the same 512
    # rows the pipelined estimate gave -- the pool is what bounds it, not the flag.
    assert record["eval_leaf_cap_uncapped"] == 512
    assert record["eval_leaf_cap_bound"] is False
    assert record["eval_hoist"] == "512"


# ---------------------------------------------------------------------------
# The PGN carries the hoist state
#
# A pooled Ordo/BayesElo fit merges games from many runs. Two runs with the same
# engine names, ConfigHash, GitSha and SideSearch but different evaluator caps
# played DIFFERENT searches, and without a tag they are fitted as ONE player.
# ---------------------------------------------------------------------------


def _pgn_headers(path: Path) -> list[dict[str, str]]:
    import chess.pgn as pgn_mod

    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        while True:
            game = pgn_mod.read_game(fh)
            if game is None:
                return out
            out.append(dict(game.headers))


def test_every_pgn_game_carries_the_effective_hoist_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    pgn_out = tmp_path / "hoist.pgn"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=512,
        search_candidate=side, search_reference=side, pgn_out=pgn_out,
        game_log_path=tmp_path / "pgnhoist.games.jsonl",
    )
    headers = _pgn_headers(pgn_out)
    assert headers, "no games were written to the PGN"
    assert {h.get("EvaluatorHoist") for h in headers} == {"512<4032"}


def test_the_pgn_tag_is_the_post_load_value_not_the_launch_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Why the tag is per-game and not a base tag.

    ``ArenaPgnWriter`` is constructed before the checkpoints are loaded, so a
    base tag would freeze the relations-OFF floor: every game would be stamped
    "4096" for a search that was in fact bound against 8192 — a provenance field
    that exists, looks authoritative, and is wrong.
    """
    _stub_model_loader_with_relations(monkeypatch, relations=True)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    pgn_out = tmp_path / "relations.pgn"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
        search_candidate=side, search_reference=side, pgn_out=pgn_out,
        game_log_path=tmp_path / "relpgn.games.jsonl",
    )
    headers = _pgn_headers(pgn_out)
    assert headers
    assert {h.get("EvaluatorHoist") for h in headers} == {"4096<8192"}


def test_the_pgn_tag_agrees_with_the_game_log_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The two records of the same fact must not be able to drift apart."""
    from chess_anti_engine.utils.game_log import read_game_log

    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    pgn_out = tmp_path / "agree.pgn"
    log_path = tmp_path / "agree.games.jsonl"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=512,
        search_candidate=side, search_reference=side, pgn_out=pgn_out,
        game_log_path=log_path,
    )
    from_pgn = {h.get("EvaluatorHoist") for h in _pgn_headers(pgn_out)}
    from_log = {arena.row_hoist_tag(r) for r in read_game_log(log_path).games}
    assert from_pgn == from_log == {"512<4032"}


def test_a_pgn_written_with_the_hoist_off_says_so_rather_than_omitting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """ABSENCE has to keep meaning "pre-hoist code", so the opt-out still tags.

    ``--eval-max-batch 0`` restores the pre-hoist ARENA, not a pre-hoist BUILD;
    a file that dropped the tag there would be indistinguishable from one
    written before the tag existed.
    """
    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch)
    pgn_out = tmp_path / "off.pgn"
    side = _side(gumbel={"topk": 32})
    _run(
        tmp_path, device="cuda", max_concurrent_games=64, eval_max_batch=0,
        search_candidate=side, search_reference=side, pgn_out=pgn_out,
        game_log_path=tmp_path / "off.games.jsonl",
    )
    assert {h.get("EvaluatorHoist") for h in _pgn_headers(pgn_out)} == {"off"}


# ---------------------------------------------------------------------------
# The sims ladder can choose its own cap
#
# scripts/elo_vs_sims.py is a supported run_arena caller and had no
# --eval-max-batch at all, so every ladder was pinned to the default: no smaller
# cap on a constrained card, and no way to use the documented 0 escape hatch to
# reproduce a pre-hoist rung.
# ---------------------------------------------------------------------------


def _ladder_run_arena_kwargs(
    monkeypatch: pytest.MonkeyPatch, argv_extra: Sequence[str],
) -> list[dict[str, Any]]:
    import scripts.elo_vs_sims as ladder

    seen: list[dict[str, Any]] = []

    def _fake_run_arena(**kwargs: Any) -> dict:
        seen.append(kwargs)
        return {"elo": 0.0, "elo_ci95": [0.0, 0.0], "pairs": 0}

    monkeypatch.setattr(ladder, "run_arena", _fake_run_arena)
    monkeypatch.setattr("sys.argv", [
        "elo_vs_sims.py", "--checkpoint", "a.pt", "--sims", "32,64,128",
        "--games-per-rung", "2", "--search-shape", "play", *argv_extra,
    ])
    ladder.main()
    assert seen, "the ladder ran no rung"
    return seen


def test_the_ladder_forwards_its_cap_to_every_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag the parser advertises and the loop drops is the defect in miniature.

    Asserted on EVERY rung, not just the first: a ladder that forwarded the flag
    once and fell back to the default afterwards would compare rungs measured
    under two different searches.
    """
    seen = _ladder_run_arena_kwargs(monkeypatch, ["--eval-max-batch", "777"])
    assert len(seen) == 2, "three rungs make two adjacent-rung arenas"
    assert [kw["eval_max_batch"] for kw in seen] == [777, 777]


def test_the_ladder_default_is_the_arena_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding the flag must not move the ladder anyone is already running."""
    seen = _ladder_run_arena_kwargs(monkeypatch, [])
    assert [kw["eval_max_batch"] for kw in seen] == [DEFAULT_EVAL_MAX_BATCH] * 2


def test_the_ladder_can_reach_the_documented_zero_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is what reproduces a pre-hoist rung, and it must survive the forward:
    a truthiness test somewhere on the path would silently restore the default."""
    seen = _ladder_run_arena_kwargs(monkeypatch, ["--eval-max-batch", "0"])
    assert [kw["eval_max_batch"] for kw in seen] == [0, 0]


# ---------------------------------------------------------------------------
# Review follow-ups on the pool change
# ---------------------------------------------------------------------------


def test_the_record_banks_the_inputs_the_leaf_bound_was_derived_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A banked row has to be RE-DERIVABLE, not merely readable.

    `eval_leaf_cap_uncapped` and `eval_hoist` are now a function of the POOL,
    and the pool is a function of two things the row did not carry: the
    concurrency ceiling and the number of opening pairs actually loaded. At mcg
    128 / topk 32 the same flags give 4096 rows over a full schedule, 2560 over
    40 games and 512 over 2 — so without these fields a row states an answer
    whose inputs are gone. `games` cannot stand in for the pool either: a short
    --openings-fen list shrinks it below what --games asked for.
    """
    side = _side(gumbel={"topk": 32})
    for n_pairs, want_pool, want_rows in ((1, 2, 512), (20, 40, 2560), (128, 128, 4096)):
        with monkeypatch.context() as ctx:
            _stub_model_loader(ctx)
            _stub_play_loops(ctx)
            ctx.setattr(
                arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"),
            )
            record = _run(
                tmp_path, device="cuda", max_concurrent_games=128,
                eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
                search_candidate=side, search_reference=side,
                games=2 * n_pairs, openings_fen=_openings_file(tmp_path, n_pairs),
                game_log_path=tmp_path / f"rec_{n_pairs}.games.jsonl",
            )
        assert record["max_concurrent_games"] == 128
        assert record["arena_pool_size"] == want_pool
        assert record["eval_leaf_cap_uncapped"] == want_rows
        # The point of banking them: the answer follows from the row's own
        # fields, with no access to the invocation that wrote it.
        assert record["eval_leaf_cap_uncapped"] == arena_uncapped_leaf_rows(
            max_concurrent_games=record["arena_pool_size"], sides=(side, side),
        )


def test_the_pool_fields_are_banked_even_where_the_leaf_fields_are_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Unconditional, unlike the leaf-cap fields.

    Those go null off the hoisted path because they mean nothing there. The
    pool is a property of the SCHEDULE and is just as true for a CPU row, so
    making it conditional would produce a null that means "not applicable" on
    some rows and "written before the field existed" on others — a distinction
    no later reader could recover.
    """
    _stub_model_loader(monkeypatch)
    _stub_play_loops(monkeypatch)
    record = _run(
        tmp_path, device="cpu", max_concurrent_games=64,
        games=8, openings_fen=_openings_file(tmp_path, 4),
    )
    assert record["eval_leaf_cap_uncapped"] is None, "no hoist on a CPU arena"
    assert record["max_concurrent_games"] == 64
    assert record["arena_pool_size"] == 8


@pytest.mark.parametrize(
    ("eval_max_batch", "refuses"),
    [
        (127, True),    # one below the pool: the root submit would raise
        (128, False),   # EXACTLY the pool: servable, must run
        (129, False),
    ],
)
def test_the_refusal_boundary_sits_at_the_pool_size_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    eval_max_batch: int, refuses: bool,
) -> None:
    """`<` and `<=` differ on exactly one value, and it is a legal one.

    A cap equal to the pool serves the widest root submit the search can make,
    so refusing it would reject a configuration that works. The three rows
    bracket the boundary; the middle one is the whole test.
    """
    _stub_model_loader(monkeypatch)
    seen = _stub_play_loops(monkeypatch)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    side = _side(gumbel={"topk": 32})
    call = lambda: _run(  # noqa: E731 - one expression, used twice below
        tmp_path, device="cuda", max_concurrent_games=128,
        eval_max_batch=eval_max_batch,
        search_candidate=side, search_reference=side,
        game_log_path=tmp_path / f"bound_{eval_max_batch}.games.jsonl",
    )
    if refuses:
        with pytest.raises(SystemExit, match="pool of 128 concurrent game"):
            call()
        assert seen == [], "refused, so nothing may have played"
        return
    record = call()
    assert seen, "a cap the pool can serve must run"
    # Still below the 4096-row leaf buffer, so it is warned and recorded --
    # allow-but-warn, which is a different thing from being refused.
    assert record["eval_hoist"] == f"{eval_max_batch}<4096"


def test_the_rolling_banner_announces_the_pool_not_the_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """"keep 128 games active" over a 2-game arena is the same falsehood.

    Paired arms: one ceiling, two schedule lengths, and the banner must move
    with the schedule.
    """
    side = _side(gumbel={"topk": 32})
    banners: dict[int, str] = {}
    for n_pairs in (1, 128):
        with monkeypatch.context() as ctx:
            _stub_model_loader(ctx)
            _stub_play_loops(ctx)
            ctx.setattr(
                arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"),
            )
            _run(
                tmp_path, device="cuda", max_concurrent_games=128, rolling=True,
                eval_max_batch=DEFAULT_EVAL_MAX_BATCH,
                search_candidate=side, search_reference=side,
                games=2 * n_pairs, openings_fen=_openings_file(tmp_path, n_pairs),
                game_log_path=tmp_path / f"banner_{n_pairs}.games.jsonl",
            )
        banners[n_pairs] = capsys.readouterr().out
    assert "ROLLING pool: keep 2 games active" in banners[1]
    assert "keep 128 games active" not in banners[1]
    assert "ROLLING pool: keep 128 games active" in banners[128]


def test_the_rolling_banner_tracks_the_remainder_a_resume_still_has_to_play(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """The banner describes THIS invocation's loop, not the whole match.

    Deliberately a different quantity from the `arena_pool_size` in the RECORD,
    which is bounded by the full loaded schedule because it scopes fields that
    describe the match (resumed pairs included). A resume that has 3 of 4 pairs
    left can keep 6 games active, not 8 — and printing 8 would be a smaller
    version of the falsehood this change removes.
    """
    log_path = tmp_path / "banner_resume.games.jsonl"
    side = _side(gumbel={"topk": 32})
    common: dict[str, Any] = {
        "games": 8, "openings_fen": _openings_file(tmp_path, 4),
        "max_concurrent_games": 128, "rolling": True, "device": "cuda",
        "search_candidate": side, "search_reference": side,
        "game_log_path": log_path,
    }
    _stub_model_loader(monkeypatch)
    _emitting_play_loops(monkeypatch, crash_after=2)
    monkeypatch.setattr(arena, "build_arena_evaluator", lambda m, **_k: _StubEvaluator("x"))
    with pytest.raises(_SimulatedCrash):
        _run(tmp_path, **common)
    assert "ROLLING pool: keep 8 games active" in capsys.readouterr().out

    _emitting_play_loops(monkeypatch)
    record = _run(tmp_path, resume=True, **common)
    assert "ROLLING pool: keep 6 games active" in capsys.readouterr().out, (
        "one pair was kept from the log; the loop can only fill 6 of the 8"
    )
    # And the RECORD keeps the match-scoped figure, which is the whole reason
    # the two are computed separately.
    assert record["arena_pool_size"] == 8


def test_the_ladder_refuses_a_games_flag_it_cannot_honour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--games` rides in from the shared parser and elo_vs_sims never reads it.

    Every rung is sized by --games-per-rung, so `--games 40` used to be
    accepted and then ignored: 400 games a rung, silently. Forwarding it is not
    the fix -- it would have to mean "per rung" to be meaningful, and a flag
    meaning something different in each script is worse than one that is
    absent. The error has to NAME the flag that does work.
    """
    for spelling in (["--games", "40"], ["--games=40"]):
        with pytest.raises(SystemExit, match="--games-per-rung"):
            _ladder_run_arena_kwargs(monkeypatch, spelling)


def test_the_games_refusal_does_not_catch_games_per_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control, and the risky half of an argv scan.

    `--games-per-rung` starts with `--games`; a prefix test would refuse the
    one flag the script actually wants, and every test above passes it only by
    omission.
    """
    seen = _ladder_run_arena_kwargs(monkeypatch, ["--games-per-rung", "6"])
    assert [kw["games"] for kw in seen] == [6, 6]
