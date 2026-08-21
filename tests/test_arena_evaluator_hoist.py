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
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

import scripts.arena_standard as arena
from chess_anti_engine.inference_dispatcher import supports_inplace_api
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay import match as match_mod
from scripts.arena_standard import (
    SideSearch,
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


def _openings_file(tmp_path: Path, n: int = 2) -> Path:
    path = tmp_path / "openings.fen"
    path.write_text("\n".join(_OPENING_FENS[:n]) + "\n")
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
    base = {
        "shape": "test", "source": "test", "gumbel": {},
        "vloss_weight": 0, "target_batch": 0,
    }
    base.update(kwargs)
    return SideSearch(**base)  # pyright: ignore[reportArgumentType]


def _fake_search_result(n: int) -> tuple:
    return (
        [np.zeros(POLICY_SIZE, dtype=np.float32)] * n,
        [0] * n,
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
        return _fake_search_result(len(boards))

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
        return _fake_search_result(len(boards))

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
        return _fake_search_result(len(boards))

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
    kwargs: dict[str, Any] = {
        "candidate": "cand.pt", "reference": "ref.pt", "games": 2,
        "openings_path": None, "openings_fen": _openings_file(tmp_path, 1),
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
