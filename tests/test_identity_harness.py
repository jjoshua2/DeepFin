"""Tests for the identity harness (``scripts/match_vs_handicapped_sf.py``).

The harness exists to measure anti-SF exploitation at a FROZEN handicap, so the
two things worth pinning are (a) that the handicap it applies is production's
within-regret rule and not a lookalike, and (b) that a run is reproducible from
its seed. Everything here drives the real
``selfplay.stockfish_turn`` functions through the harness entry point; nothing
restates the selection rule, because a test that restated it would pass against
a rule that had drifted.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

import chess

from chess_anti_engine.moves.encode import legal_move_indices, uci_to_policy_index
from chess_anti_engine.selfplay.stockfish_turn import (
    _choose_curriculum_opponent_move,
    within_regret_candidates,
)
from chess_anti_engine.stockfish.uci import StockfishPV, StockfishResult
from scripts.match_vs_handicapped_sf import (
    REGRET_MAX,
    build_fingerprint,
    handicapped_sf_move,
    net_score_from_result,
    sf_move_rng,
    summarize,
)

# THE FIXTURE, stated once. Five MultiPV rows in rank order whose win fractions
# (w + 0.5*d — the units `_pv_wdl_score` produces with the arguments the
# production MOVE path passes) are 0.60 / 0.59 / 0.58 / 0.50 / 0.40. Against a
# regret band of 0.025 the first THREE qualify (gaps 0.00 / 0.01 / 0.02) and the
# last two do NOT (gaps 0.10 / 0.20). The gap to the first excluded move is 5x
# the band, so no float tolerance can move the boundary.
_INSIDE = ("e2e4", "d2d4", "g1f3")
_OUTSIDE = ("c2c4", "b1c3")
_SCORES = (0.60, 0.59, 0.58, 0.50, 0.40)
_BAND = 0.025


def _pv(uci: str, score: float) -> StockfishPV:
    """A MultiPV row whose ``w + 0.5*d`` equals *score* (draw mass fixed at 0.4)."""
    d = 0.4
    w = score - 0.5 * d
    return StockfishPV(
        move_uci=uci,
        wdl=np.array([w, d, 1.0 - w - d], dtype=np.float32),
        cp=round(score * 100),
        mate=None,
    )


def _result() -> StockfishResult:
    pvs = [
        _pv(uci, score)
        for uci, score in zip(_INSIDE + _OUTSIDE, _SCORES, strict=True)
    ]
    return StockfishResult(
        bestmove_uci=_INSIDE[0],
        wdl=pvs[0].wdl,
        pvs=pvs,
        cp=pvs[0].cp,
        mate=None,
        nodes=75000,
        depth=22,
    )


def _cands(board: chess.Board) -> tuple[list[int], list[float]]:
    """The (indices, scores) the harness feeds the selection rule."""
    from chess_anti_engine.selfplay.stockfish_turn import _collect_sf_pv_candidates

    legal = {int(x) for x in legal_move_indices(board)}
    return _collect_sf_pv_candidates(_result(), _turn=bool(board.turn), legal_set=legal)


def test_qualifying_set_is_exactly_the_three_moves_within_the_band() -> None:
    board = chess.Board()
    cand_idxs, cand_scores = _cands(board)
    assert len(cand_idxs) == 5, "all five hand-built PVs must survive as candidates"

    qualifying = within_regret_candidates(
        cand_indices=cand_idxs, cand_scores=cand_scores, regret_limit=_BAND,
    )
    expected = {uci_to_policy_index(u, True) for u in _INSIDE}
    assert set(qualifying) == expected
    assert len(qualifying) == 3
    for uci in _OUTSIDE:
        assert uci_to_policy_index(uci, True) not in qualifying


def test_selection_is_uniform_over_the_qualifying_set_and_never_leaves_it() -> None:
    """Across many seeds: only the 3 in-band moves appear, and roughly evenly.

    ⚑ THE MUTANT THIS EXISTS FOR: comparing the regret to the wrong side
    (``score - best <= limit`` instead of ``best - score <= limit``) makes EVERY
    candidate qualify, so the two out-of-band moves start appearing here. The
    uniformity half catches the complementary drift — a rule that kept the set
    but stopped drawing over it.
    """
    board = chess.Board()
    res = _result()
    counts: Counter[str] = Counter()
    trials = 2000
    for seed in range(trials):
        move, outcome = handicapped_sf_move(
            res, board, regret=_BAND, rng=sf_move_rng(seed=seed, game_index=0, ply=0),
        )
        counts[move.uci()] += 1
        assert outcome.qualifying == 3
        assert outcome.candidates == 5

    assert set(counts) == set(_INSIDE), (
        f"selection left the qualifying set: saw {sorted(counts)}, "
        f"expected {sorted(_INSIDE)}"
    )
    for uci in _INSIDE:
        share = counts[uci] / trials
        assert 0.28 < share < 0.39, f"{uci} share {share:.3f} is not ~1/3"


def test_regret_zero_admits_only_the_best_move() -> None:
    """The wiring calibration point: at regret 0 the qualifying set is ~1."""
    board = chess.Board()
    res = _result()
    for seed in range(50):
        move, outcome = handicapped_sf_move(
            res, board, regret=0.0, rng=sf_move_rng(seed=seed, game_index=0, ply=0),
        )
        assert outcome.qualifying == 1
        assert outcome.tied_at_best == 1
        assert outcome.admitted_by_regret == 0
        assert not outcome.saturated
        assert move.uci() == _INSIDE[0]
        assert outcome.rank == 1
        assert outcome.regret == pytest.approx(0.0, abs=1e-9)


def _saturated_result() -> StockfishResult:
    """Six MultiPV lines that all score IDENTICALLY (a dead-drawn position).

    The shape SF's native WDL actually produces once a position is decided —
    every line reads the same (W, D, L), so ``w + 0.5*d`` is constant across the
    whole list. Not invented for the test: this is what made a real ``--regret
    0.0`` smoke report a mean qualifying set of 4.39.
    """
    pvs = [_pv(u, 0.5) for u in ("e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "b1a3")]
    return StockfishResult(
        bestmove_uci="e2e4", wdl=pvs[0].wdl, pvs=pvs, cp=0, mate=None,
        nodes=75000, depth=30,
    )


def test_a_saturated_position_reports_no_handicap_despite_a_full_qualifying_set() -> None:
    """⚑ THE CONFOUND. Every line ties, so the band admits nothing extra.

    ``qualifying`` reads 6 at EVERY band including zero — which looks exactly
    like a leaked handicap — while ``admitted_by_regret`` is 0, which is the
    truth. A statistic that reported only the first would say the opponent was
    handicapped in positions where the rule provably could not act.
    """
    board = chess.Board()
    res = _saturated_result()
    for regret in (0.0, _BAND, 0.5):
        _move, outcome = handicapped_sf_move(
            res, board, regret=regret, rng=sf_move_rng(seed=3, game_index=0, ply=0),
        )
        assert outcome.candidates == 6
        assert outcome.qualifying == 6
        assert outcome.tied_at_best == 6
        assert outcome.admitted_by_regret == 0, (
            f"a saturated position cannot be handicapped, but regret {regret} "
            "was reported as admitting moves"
        )
        assert outcome.saturated is True
        assert outcome.regret == pytest.approx(0.0, abs=1e-9)


def test_admitted_by_regret_is_the_statistic_that_tracks_the_band() -> None:
    """It is 0 at band 0 and rises with the band; the raw set size need not."""
    board = chess.Board()
    res = _result()
    admitted = []
    for regret in (0.0, 0.015, 0.025, 0.15, 0.25):
        _move, outcome = handicapped_sf_move(
            res, board, regret=regret, rng=sf_move_rng(seed=5, game_index=0, ply=0),
        )
        admitted.append(outcome.admitted_by_regret)
    assert admitted == [0, 1, 2, 3, 4]


def test_a_wider_band_strictly_grows_the_qualifying_set() -> None:
    """Monotone in the knob, which is what makes the reported mean a reading."""
    board = chess.Board()
    cand_idxs, cand_scores = _cands(board)
    sizes = [
        len(within_regret_candidates(
            cand_indices=cand_idxs, cand_scores=cand_scores, regret_limit=limit,
        ))
        for limit in (0.0, 0.015, 0.025, 0.15, 0.25)
    ]
    assert sizes == [1, 2, 3, 4, 5]


def test_exact_ties_are_admitted_at_regret_zero() -> None:
    """A tie is inside every band including the empty one (``<=``, not ``<``)."""
    qualifying = within_regret_candidates(
        cand_indices=[10, 20, 30], cand_scores=[0.5, 0.5, 0.4], regret_limit=0.0,
    )
    assert sorted(qualifying) == [10, 20]


def test_empty_candidate_list_is_reported_as_unmeasured_not_as_a_set() -> None:
    """No legal PV => qualifying 0, and the move still has to be legal.

    A desynced engine returns lines that are illegal in the queried position;
    production falls back to a uniform legal move there. The harness must record
    that as UNMEASURED (0), never as a qualifying set of size zero or one, or the
    realized-handicap mean would quietly average in moves the handicap never
    touched.
    """
    board = chess.Board()
    empty = StockfishResult(
        bestmove_uci="e2e4", wdl=None, pvs=[], cp=None, mate=None,
    )
    move, outcome = handicapped_sf_move(
        empty, board, regret=_BAND, rng=sf_move_rng(seed=1, game_index=0, ply=0),
    )
    assert outcome.qualifying == 0
    assert outcome.candidates == 0
    assert outcome.rank is None
    assert move in board.legal_moves

    assert within_regret_candidates(
        cand_indices=[], cand_scores=[], regret_limit=_BAND,
    ) == []


def test_same_seed_reproduces_the_whole_move_sequence() -> None:
    """(seed, game, ply) keying: identical seed => identical sequence."""
    res = _result()

    def sequence(seed: int, game: int) -> list[str]:
        board = chess.Board()
        out: list[str] = []
        for _ in range(6):
            move, _outcome = handicapped_sf_move(
                res, board, regret=0.25,
                rng=sf_move_rng(
                    seed=seed, game_index=game, ply=len(board.move_stack),
                ),
            )
            out.append(move.uci())
            board.push(move)
            board.push(next(iter(board.legal_moves)))
        return out

    assert sequence(42, 0) == sequence(42, 0)
    assert sequence(42, 0) != sequence(43, 0), "a different seed must re-draw"
    assert sequence(42, 0) != sequence(42, 1), "a different game must re-draw"


def test_the_rng_key_is_order_independent() -> None:
    """A move's draw depends on (seed, game, ply) ONLY — not on play order.

    Games run interleaved and Stockfish futures complete out of order, so this
    is what makes a run reproducible at a different --max-concurrent-games.
    """
    a = sf_move_rng(seed=7, game_index=3, ply=11).integers(1_000_000)
    b = sf_move_rng(seed=7, game_index=3, ply=11).integers(1_000_000)
    assert a == b
    assert a != sf_move_rng(seed=7, game_index=3, ply=12).integers(1_000_000)
    assert a != sf_move_rng(seed=7, game_index=4, ply=11).integers(1_000_000)


def test_extraction_did_not_change_the_production_rng_stream() -> None:
    """``_choose_curriculum_opponent_move`` consumes rng exactly as before.

    ⚑ The regression the extraction could have introduced, and the reason the
    two degenerate branches stayed in the caller: routing the ``inf`` case
    through ``within_regret_candidates`` would return ``[best]`` and then draw
    from it, consuming one ``rng.integers`` call that the old code did not — and
    silently re-streaming every selfplay game that ever plays a refute ply.
    """
    legal = legal_move_indices(chess.Board())
    cand_idxs, cand_scores = _cands(chess.Board())

    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    chosen = _choose_curriculum_opponent_move(
        rng=rng, legal_indices=legal, cand_indices=cand_idxs,
        cand_scores=cand_scores, regret_limit=math.inf,
    )
    assert chosen == cand_idxs[0]
    assert rng.bit_generator.state == before, "the inf path must not draw"

    rng2 = np.random.default_rng(0)
    _choose_curriculum_opponent_move(
        rng=rng2, legal_indices=legal, cand_indices=cand_idxs,
        cand_scores=cand_scores, regret_limit=_BAND,
    )
    assert rng2.bit_generator.state != before, "the banded path must draw once"


def test_regret_guard_refuses_out_of_band_values() -> None:
    from scripts.match_vs_handicapped_sf import _validate

    class _Args:
        regret = 0.05
        games = 4
        sf_nodes = 5000
        sims = 8
        sf_multipv = 6
        sf_workers = 1
        stockfish = __file__  # exists; the guard only checks presence

    for bad in (-0.001, -1.0, REGRET_MAX + 0.001, 2.0):
        args = _Args()
        args.regret = bad
        with pytest.raises(SystemExit, match="outside"):
            _validate(args)  # pyright: ignore[reportArgumentType]

    _validate(_Args())  # pyright: ignore[reportArgumentType] - in band, must pass


def test_score_is_from_our_sides_point_of_view() -> None:
    assert net_score_from_result("1-0", net_is_white=True) == 1.0
    assert net_score_from_result("1-0", net_is_white=False) == 0.0
    assert net_score_from_result("0-1", net_is_white=True) == 0.0
    assert net_score_from_result("0-1", net_is_white=False) == 1.0
    assert net_score_from_result("1/2-1/2", net_is_white=True) == 0.5
    with pytest.raises(ValueError, match="unscoreable"):
        net_score_from_result("*", net_is_white=True)


def _outcome(game: int, score: float, quals: list[int]):
    from scripts.match_vs_handicapped_sf import GameOutcome, SfMoveOutcome

    return GameOutcome(
        game_index=game, pair_id=game // 2, half=game % 2,
        net_is_white=(game % 2 == 0),
        result={1.0: "1-0", 0.5: "1/2-1/2", 0.0: "0-1"}[score]
        if game % 2 == 0 else {1.0: "0-1", 0.5: "1/2-1/2", 0.0: "1-0"}[score],
        score=score, plies=40, termination="rules", start_fen=chess.STARTING_FEN,
        moves=(),
        sf_moves=[
            SfMoveOutcome(qualifying=q, candidates=6, tied_at_best=1, rank=1,
                          regret=0.0, nodes=75000, depth=20)
            for q in quals
        ],
    )


def test_summary_pairs_the_games_and_excludes_unmeasured_moves_from_the_mean() -> None:
    outcomes = [
        _outcome(0, 1.0, [3, 3, 0]),   # one no-PV move
        _outcome(1, 0.5, [1, 5]),
        _outcome(2, 0.0, [4]),
        _outcome(3, 1.0, [4]),
    ]
    s = summarize(outcomes, regret=0.05)
    assert s["pairs_complete"] == 2
    assert s["pairs_dropped"] == 0
    assert s["ci_unit"] == "pair (pentanomial)"
    assert s["score"] == pytest.approx(0.625)
    h = s["handicap"]
    assert h["sf_moves"] == 7
    assert h["no_pv_moves"] == 1
    assert h["measured_moves"] == 6
    # (3 + 3 + 1 + 5 + 4 + 4) / 6 — the 0 is NOT in the denominator.
    assert h["qualifying_set_size_mean"] == pytest.approx(20 / 6)
    # Every fixture move carries tied_at_best=1, so admitted = qualifying - 1.
    assert h["tied_at_best_mean"] == pytest.approx(1.0)
    assert h["admitted_by_regret_mean"] == pytest.approx(14 / 6)
    assert h["regret_bound_respected"] is True


def test_summary_flags_a_run_whose_played_move_broke_the_band() -> None:
    """The exact invariant, not a statistic: max played regret <= the band.

    If this ever reads False the harness did not run the rule its fingerprint
    claims, and every score in the file is void — so it is computed and stored
    rather than assumed.
    """
    from scripts.match_vs_handicapped_sf import GameOutcome, SfMoveOutcome

    def one(regret: float) -> GameOutcome:
        return GameOutcome(
            game_index=0, pair_id=0, half=0, net_is_white=True, result="1-0",
            score=1.0, plies=2, termination="rules", start_fen=chess.STARTING_FEN,
            moves=(),
            sf_moves=[SfMoveOutcome(
                qualifying=2, candidates=6, tied_at_best=1, rank=2,
                regret=regret, nodes=1, depth=1,
            )],
        )

    assert summarize([one(0.04)], regret=0.05)["handicap"][
        "regret_bound_respected"] is True
    assert summarize([one(0.05)], regret=0.05)["handicap"][
        "regret_bound_respected"] is True
    assert summarize([one(0.06)], regret=0.05)["handicap"][
        "regret_bound_respected"] is False


def test_summary_drops_an_orphan_half_instead_of_imputing_it() -> None:
    s = summarize(
        [_outcome(0, 1.0, [2]), _outcome(1, 0.0, [2]), _outcome(2, 1.0, [2])],
        regret=0.05,
    )
    assert s["pairs_complete"] == 1
    assert s["pairs_dropped"] == 1
    assert s["games"] == 3


def test_fingerprint_moves_with_every_opponent_defining_field() -> None:
    base = {
        "regret": 0.05, "sf_nodes": 75000, "sf_multipv": 6, "sims": 100,
        "seed": 42, "book": "b.pgn", "syzygy": "a:b", "sf_fresh_tt": True,
    }
    ref = build_fingerprint(base)
    assert build_fingerprint(dict(base)) == ref
    for key, other in (
        ("regret", 0.051), ("sf_nodes", 75001), ("sf_multipv", 7),
        ("sims", 101), ("seed", 43), ("book", "c.pgn"), ("syzygy", "a"),
        ("sf_fresh_tt", False),
    ):
        assert build_fingerprint({**base, key: other}) != ref, f"{key} not in it"


# ---------------------------------------------------------------------------
# The play loop, driven with a stub Stockfish pool (no GPU, no subprocess)
# ---------------------------------------------------------------------------


class _StubPool:
    """Records every kwarg ``play_chunk`` hands ``submit`` and answers locally."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def submit(self, fen: str, **kwargs: object):
        from concurrent.futures import Future

        self.calls.append({"fen": fen, **kwargs})
        board = chess.Board(fen)
        # Six real legal moves with a decreasing win fraction, so the handicap
        # has something to act on and the played move is always legal.
        moves = list(board.legal_moves)[:6]
        pvs = [_pv(m.uci(), 0.60 - 0.01 * k) for k, m in enumerate(moves)]
        fut: Future = Future()
        fut.set_result(StockfishResult(
            bestmove_uci=moves[0].uci(), wdl=pvs[0].wdl, pvs=pvs,
            cp=pvs[0].cp, mate=None, nodes=1234, depth=7,
        ))
        return fut


def _stub_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """Our side plays its first legal move — deterministic, no model needed."""
    from chess_anti_engine.moves.encode import move_to_index

    def fake(_model, boards, **_kwargs):
        return [move_to_index(next(iter(b.legal_moves)), b) for b in boards]

    monkeypatch.setattr(
        "chess_anti_engine.selfplay.match.pick_moves_for_boards", fake,
    )


def _run_chunk(monkeypatch: pytest.MonkeyPatch, *, openings=None, **over):
    from scripts.arena_standard import resolve_search_shape
    from scripts.match_vs_handicapped_sf import play_chunk

    _stub_net(monkeypatch)
    pool = _StubPool()
    boards = [chess.Board()] if openings is None else list(openings)
    kwargs = {
        "pair_ids": list(range(len(boards))),
        "pool": pool,
        "device": "cpu",
        "regret": 0.025,
        "sf_nodes": 4321,
        "sf_syzygy_path": "/nonexistent/tb",
        "sf_fresh_tt": True,
        "sims": 2,
        "seed": 7,
        "temperature": 0.0,
        "gumbel_add_noise": False,
        "side": resolve_search_shape("play"),
        "max_plies": 12,
        "evaluator": None,
        "tablebase": None,
        "tb_max_pieces": 6,
        "net_rng": np.random.default_rng(7),
    }
    kwargs.update(over)
    outcomes = play_chunk(None, boards, **kwargs)  # pyright: ignore[reportArgumentType]
    return outcomes, pool


def test_play_loop_hands_stockfish_the_frozen_settings_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE WIRING CHECK, read off the CONSUMER's call, not the CLI namespace.

    ``--sf-nodes``, ``--syzygy`` and the cold-TT decision are exactly the kind of
    value this codebase accepts and then silently drops. Asserting them on the
    argparse namespace would prove nothing; this asserts them on what
    ``pool.submit`` actually received, on every SF move of a real play loop.
    """
    outcomes, pool = _run_chunk(monkeypatch)
    assert pool.calls, "the loop never queried Stockfish"
    for call in pool.calls:
        assert call["nodes"] == 4321, "the frozen node budget did not reach SF"
        assert call["syzygy_path"] == "/nonexistent/tb"
        assert call["fresh"] is True, (
            "cold-TT was requested but the pool was asked for a warm search; "
            "the run would not be reproducible"
        )

    # ⚑ MEASURED, not assumed: with a WARM TT two identical 4-game invocations
    # returned different scores, because the pool hands a request to whichever
    # engine is free and each engine's TT carries what it searched before.
    _outcomes2, pool2 = _run_chunk(monkeypatch, sf_fresh_tt=False)
    assert all(c["fresh"] is False for c in pool2.calls)

    assert len(outcomes) == 2, "one opening must produce both colorings"
    assert {o.net_is_white for o in outcomes} == {True, False}
    assert {o.game_index for o in outcomes} == {0, 1}
    for o in outcomes:
        assert o.score in (0.0, 0.5, 1.0)
        assert o.sf_moves, "no SF move was recorded for this game"
        assert all(m.nodes == 1234 for m in o.sf_moves)


def test_the_two_colorings_of_a_pair_get_INDEPENDENT_sf_randomization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The pair is the sampling unit, so its halves must not share a stream.

    ``play_chunk`` keys each SF draw on the GLOBAL game id, so the net-white and
    net-black halves of one opening randomize independently. A call site that
    keyed on the pair (or on a constant) would leave the two halves drawing the
    SAME index at the same ply — the halves would then be correlated, the paired
    design's whole point would be gone, and the pentanomial CI would be wrong in
    a direction nothing else here would reveal. Survived an earlier mutant; this
    test is why it no longer does.

    With ``regret`` wide enough that all six stub candidates qualify, the played
    move's ``rank`` IS the raw draw, so comparing rank sequences compares streams
    directly rather than through the positions.
    """
    outcomes, _pool = _run_chunk(monkeypatch, regret=0.9, max_plies=24)
    by_game = {o.game_index: o for o in outcomes}
    assert set(by_game) == {0, 1}
    ranks = {g: [m.rank for m in o.sf_moves] for g, o in by_game.items()}
    assert all(o.sf_moves for o in outcomes)
    n = min(len(ranks[0]), len(ranks[1]))
    assert n >= 4, f"need overlapping SF plies to compare streams, got {n}"
    assert ranks[0][:n] != ranks[1][:n], (
        "both colorings of the pair drew the identical SF move-rank sequence — "
        f"the two halves share an RNG stream. ranks={ranks}"
    )


def test_two_pairs_playing_the_SAME_opening_randomize_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE SHARPER HALF, and it took a surviving mutant to find it.

    Keying on ``ply`` alone already separates the two COLORINGS of one opening,
    because SF moves on opposite parities there — so a call site that dropped the
    game id entirely still passed the sibling test above. It does NOT separate
    two different PAIRS: game 0 and game 2 are both net-white and move on the
    same parity, so under a constant key they draw the identical sequence and
    two independent samples become one.

    Both openings here are the standard start ON PURPOSE. Our stubbed side is
    deterministic and every stub candidate qualifies at this band, so the ONLY
    thing that can make the two games differ is the RNG key — which makes this an
    exact detector rather than a probabilistic one.
    """
    outcomes, _pool = _run_chunk(
        monkeypatch,
        openings=[chess.Board(), chess.Board()],
        regret=0.9,
        max_plies=24,
    )
    by_game = {o.game_index: o for o in outcomes}
    assert set(by_game) == {0, 1, 2, 3}
    a, b = by_game[0], by_game[2]  # both net-white, same opening, same parity
    assert a.net_is_white
    assert b.net_is_white
    assert a.pair_id != b.pair_id
    ranks_a = [m.rank for m in a.sf_moves]
    ranks_b = [m.rank for m in b.sf_moves]
    assert ranks_a
    assert ranks_b
    assert ranks_a != ranks_b, (
        "two different pairs played the identical game from the identical "
        "opening — the SF draw is not keyed on the game index, so the pairs are "
        f"perfectly correlated and the CI is fiction. ranks={ranks_a}"
    )


def test_play_loop_is_reproducible_from_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same seed, same everything — including every qualifying-set size."""
    a, _ = _run_chunk(monkeypatch)
    b, _ = _run_chunk(monkeypatch)
    assert [o.moves for o in a] == [o.moves for o in b]
    assert [[m.qualifying for m in o.sf_moves] for o in a] == [
        [m.qualifying for m in o.sf_moves] for o in b
    ]
    c, _ = _run_chunk(monkeypatch, seed=8)
    assert [o.moves for o in a] != [o.moves for o in c], "a new seed must re-draw"
