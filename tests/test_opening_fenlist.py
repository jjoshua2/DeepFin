"""Blind-spot FEN-list opening seeding (selfplay/opening.py)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay.finalize import _update_aggregate_stats
from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    _load_fen_list,
    resolve_slot_opening,
    sample_starting_board,
    seed_board_from_line,
)
from chess_anti_engine.selfplay.state import SelfplayState

FEN_BLACK = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FEN_WHITE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"


def _write_list(tmp_path: Path, lines: list[str]) -> str:
    p = tmp_path / "fens_test.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_load_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = _write_list(tmp_path, ["# header", "", FEN_BLACK, "  ", FEN_WHITE])
    assert _load_fen_list(path) == (FEN_BLACK, FEN_WHITE)


# Unusable seed lines are SKIPPED (with a warning), not raised per-line, so one
# bad line in a hand-edited file can't crash the worker fleet (Codex review). We
# raise only when NOTHING usable remains. Each bad type is paired with a good FEN
# to assert it is dropped but the good one survives.
_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate, terminal
_NO_KING = "8/8/8/8/8/8/8/4K3 w - - 0 1"  # illegal: black has no king
_FORCED = "r5k1/p1P3pp/3QN1q1/3P1pB1/8/5P2/P6P/2n1q1K1 w - - 0 31"  # 1 legal move
_FIFTY = "4k3/8/4K3/8/8/8/8/5R2 w - - 100 120"  # claimable 50-move draw


@pytest.mark.parametrize("bad", ["not a fen", _NO_KING, _FORCED, _FIFTY, _MATE])
def test_load_skips_unusable_lines_keeps_good(tmp_path: Path, bad: str) -> None:
    path = _write_list(tmp_path, [FEN_BLACK, bad, FEN_WHITE])
    assert _load_fen_list(path) == (FEN_BLACK, FEN_WHITE)


def test_load_tolerates_utf8_bom(tmp_path: Path) -> None:
    # A BOM-prefixed file (common from Windows editors) must not corrupt the
    # first line — the utf-8-sig read strips it.
    p = tmp_path / "fens_bom.txt"
    p.write_text("\n".join([FEN_BLACK, FEN_WHITE]) + "\n", encoding="utf-8-sig")
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
    assert _load_fen_list(str(p)) == (FEN_BLACK, FEN_WHITE)


def test_claim_draw_fen_is_the_reason_case() -> None:
    # Guards the specific CBoard-vs-python-chess mismatch: is_game_over() alone
    # (no claim) misses halfmove>=100, which CBoard treats as over.
    b = chess.Board(_FIFTY)
    assert not b.is_game_over()
    assert b.is_game_over(claim_draw=True)


def test_load_raises_when_nothing_usable(tmp_path: Path) -> None:
    # All-bad / all-comment / empty file must fail fast at load (warm-up), not
    # mid-run: _sample_fen_list has no empty guard, so a silent empty tuple would
    # crash the selfplay thread on first sample.
    for lines in ([_NO_KING, _FORCED], ["# only comments", ""], []):
        path = _write_list(tmp_path, lines) if lines else _write_list(tmp_path, ["# x"])
        with pytest.raises(ValueError, match="no usable seeds"):
            _load_fen_list(path)


def test_prob_one_always_samples_fenlist(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK, FEN_WHITE])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)
    rng = np.random.default_rng(0)
    fens = {FEN_BLACK, FEN_WHITE}
    for _ in range(20):
        start = sample_starting_board(rng=rng, cfg=cfg)
        assert start.source == "fenlist"
        assert start.board.fen() in fens


def test_prob_zero_or_unset_never_samples_fenlist(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK])
    rng = np.random.default_rng(0)
    for cfg in (
        OpeningConfig(opening_fen_list_path=path, opening_fen_prob=0.0),
        OpeningConfig(),  # default: feature off
    ):
        for _ in range(20):
            assert sample_starting_board(rng=rng, cfg=cfg).source != "fenlist"


def test_fenlist_preserves_side_to_move_and_castling(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_BLACK])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)
    board = sample_starting_board(rng=np.random.default_rng(0), cfg=cfg).board
    assert board.turn is chess.BLACK
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert board.fullmove_number == 3


def test_fenlist_takes_priority_over_book(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [FEN_WHITE])
    cfg = OpeningConfig(
        opening_fen_list_path=path,
        opening_fen_prob=1.0,
        # book path unset is the common case; random_start_plies exercises the
        # fallback branch that fenlist must preempt
        random_start_plies=2,
    )
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert sample_starting_board(rng=rng, cfg=cfg).source == "fenlist"


def test_tb_adjudication_defers_virgin_fenlist_slot(monkeypatch) -> None:
    """A TB-eligible FEN seed must survive adjudication until a ply is played."""
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.selfplay import manager as mgr

    fen = "8/8/8/4k3/8/3K4/4P3/8 w - - 0 1"  # KPvK: TB-eligible at ply 0
    board = chess.Board(fen)
    moved = chess.Board(fen)
    moved.push_uci("e2e4")  # net (white) moves
    moved.push_uci("e5e6")  # opponent replies -> 2 plies played, net has decided

    st = cast(Any, object.__new__(SelfplayState))
    st.batch_size = 3
    st.done_arr = np.zeros(3, dtype=np.int8)
    st.finalized_arr = np.zeros(3, dtype=np.int8)
    st.tb_result_arr = [None, None, None]
    st.tb_adj_roll_arr = np.ones(3, dtype=np.int8)
    st.pending_sf_moves = {}
    # slot 0: virgin fenlist (0 plies) -> deferred; slot 1: same position from a
    # book -> adjudicated (not protected); slot 2: fenlist after 2 plies (net has
    # moved) -> adjudicated. The guard defers while <2 plies are played so it
    # covers the opponent-replies-first seating too.
    st.cboards = [
        CBoard.from_board(board), CBoard.from_board(board), CBoard.from_board(moved),
    ]
    st.opening_source_arr = ["fenlist", "book1", "fenlist"]
    st.starting_boards = [board.copy(), board.copy(), board.copy()]
    st.starting_ply_arr = np.full(3, board.ply(), dtype=np.int32)
    st.tb_probe = SimpleNamespace(max_pieces=6)
    st.game = SimpleNamespace(syzygy_path="/nonexistent-tb-path")
    monkeypatch.setattr(mgr, "tb_adjudicate_result", lambda _b, _p: "1/2-1/2")

    assert mgr._tb_adjudicate_active_games(st) == 2
    assert list(st.done_arr) == [0, 1, 1]


def _agg_state(source: str, *, selfplay: bool, seed_fen: str | None = None) -> Any:
    """One-slot bare SelfplayState double for ``_update_aggregate_stats`` tests
    (pure-Python path, no C extension). ``seed_fen`` populates
    ``starting_boards`` — the seed-position retention the stm telemetry reads;
    None models states built without it (must degrade to no stm keys)."""
    st = cast(Any, object.__new__(SelfplayState))
    st.selfplay_arr = np.array([1 if selfplay else 0], dtype=np.int8)
    st.net_color_arr = np.array([1], dtype=np.int8)  # net plays white (curriculum)
    st.opening_source_arr = [source]
    st.starting_boards = [chess.Board(seed_fen)] if seed_fen is not None else None
    st.stats = SimpleNamespace(
        w=0, d=0, l=0, plies_win=0, plies_draw=0, plies_loss=0,
        selfplay_games=0, curriculum_games=0,
        selfplay_adjudicated_games=0, curriculum_adjudicated_games=0,
        total_draw_games=0, selfplay_draw_games=0, curriculum_draw_games=0,
        outcome_stats={},
    )
    return st


def _stm_keys(st: Any) -> list[str]:
    return [k for k in st.stats.outcome_stats if "_stm_" in k]


def test_fenlist_curriculum_excluded_from_pid_winrate() -> None:
    """A blind-spot seed is a systematic loss; it must not enter the PID winrate
    sample (state.stats.w/d/l) or it eases SF across the whole batch — but its
    outcome must still be visible in per-source telemetry (Codex review [1])."""
    # net (white) loses -> "0-1"
    book = _agg_state("book1", selfplay=False)
    _update_aggregate_stats(book, 0, result="0-1", was_adjudicated=False, game_plies=40)
    assert book.stats.l == 1  # a normal curriculum loss DOES feed the PID sample

    fen = _agg_state("fenlist", selfplay=False, seed_fen=FEN_WHITE)
    _update_aggregate_stats(fen, 0, result="0-1", was_adjudicated=False, game_plies=40)
    assert fen.stats.l == 0            # seed loss EXCLUDED from the PID sample
    assert fen.stats.plies_loss == 0   # and from plies accounting
    assert fen.stats.curriculum_games == 1  # still counted as a game played
    assert fen.stats.outcome_stats.get("curriculum_fenlist_l") == 1  # telemetry kept
    # stm telemetry is selfplay-only: a curriculum fenlist game must not add it
    # (the curriculum_fenlist_* keys above already cover that seat).
    assert _stm_keys(fen) == []


# ── Seed-STM outcome telemetry (selfplay_fenlist_stm_{w,d,l}) ─────────────────
# Seeds are deep-SF-confirmed LOST for the side to move at the seed position; in
# seeded SELFPLAY both seats are the net, so an stm escape (draw/win) means the
# game-outcome component of the WDL blend contradicts the deep-SF truth. The
# telemetry makes that visible per iteration; healthy training keeps stm_l
# dominant. FEN_WHITE has white to move at the seed, FEN_BLACK black.


@pytest.mark.parametrize(
    ("seed_fen", "result", "expected_key"),
    [
        (FEN_WHITE, "1-0", "selfplay_fenlist_stm_w"),
        (FEN_WHITE, "1/2-1/2", "selfplay_fenlist_stm_d"),
        (FEN_WHITE, "0-1", "selfplay_fenlist_stm_l"),
        (FEN_BLACK, "0-1", "selfplay_fenlist_stm_w"),
        (FEN_BLACK, "1/2-1/2", "selfplay_fenlist_stm_d"),
        (FEN_BLACK, "1-0", "selfplay_fenlist_stm_l"),
    ],
)
def test_fenlist_selfplay_stm_outcome_telemetry(
    seed_fen: str, result: str, expected_key: str,
) -> None:
    st = _agg_state("fenlist", selfplay=True, seed_fen=seed_fen)
    _update_aggregate_stats(st, 0, result=result, was_adjudicated=False, game_plies=40)
    assert st.stats.outcome_stats.get(expected_key) == 1
    assert _stm_keys(st) == [expected_key]  # exactly one stm key per game
    # Existing per-source selfplay keys are unchanged alongside the new ones.
    assert st.stats.outcome_stats.get("selfplay_fenlist_games") == 1
    if result == "1/2-1/2":
        assert st.stats.outcome_stats.get("selfplay_fenlist_draws") == 1
    # Telemetry only: the PID winrate sample stays untouched by selfplay games.
    assert (st.stats.w, st.stats.d, st.stats.l) == (0, 0, 0)


def test_non_fenlist_selfplay_records_no_stm_keys() -> None:
    st = _agg_state("book1", selfplay=True, seed_fen=FEN_WHITE)
    _update_aggregate_stats(st, 0, result="1-0", was_adjudicated=False, game_plies=40)
    assert st.stats.outcome_stats.get("selfplay_book1_games") == 1
    assert _stm_keys(st) == []


def test_fenlist_selfplay_stm_skipped_without_starting_boards() -> None:
    # starting_boards is Optional on SelfplayState; without the seed board the
    # stm seat is unknown — degrade to no stm keys rather than crash or guess.
    st = _agg_state("fenlist", selfplay=True, seed_fen=None)
    _update_aggregate_stats(st, 0, result="1-0", was_adjudicated=False, game_plies=40)
    assert st.stats.outcome_stats.get("selfplay_fenlist_games") == 1
    assert _stm_keys(st) == []


def test_production_seed_asset_loads() -> None:
    # 76 = panel v2 minus the 35 held-out v1 rows, minus 2 forced-move seeds
    # curated out. v1 stays a pure generalization yardstick.
    asset = Path(__file__).resolve().parents[1] / "data" / "blindspot_fens_v1.txt"
    if not asset.exists():
        pytest.skip("seed asset not present in this checkout")
    fens = _load_fen_list(str(asset))
    assert len(fens) == len(set(fens)) == 76


# ── Seed-history format: '<start_fen> | <uci moves>' ─────────────────────────

_NAJDORF = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"
_SEED_WITH_HISTORY = f"{chess.STARTING_FEN} | {_NAJDORF}"


def _terminal_fen(line: str) -> str:
    return seed_board_from_line(line).fen()


def test_plain_fen_line_has_empty_stack() -> None:
    b = seed_board_from_line(FEN_BLACK)
    assert b.fen() == FEN_BLACK
    assert len(b.move_stack) == 0


def test_history_line_replays_to_terminal_with_stack() -> None:
    b = seed_board_from_line(_SEED_WITH_HISTORY)
    assert len(b.move_stack) == 10
    # The terminal position is the seed the net actually faces.
    expected = chess.Board()
    for u in _NAJDORF.split():
        expected.push(chess.Move.from_uci(u))
    assert b.fen() == expected.fen()


def test_history_line_produces_real_history_planes() -> None:
    # The whole point: a populated move_stack must change the LC0 history planes
    # vs the same terminal position with an empty stack (which repeat-fills).
    from chess_anti_engine.encoding.encode import encode_position

    b_hist = seed_board_from_line(_SEED_WITH_HISTORY)
    b_bare = chess.Board(b_hist.fen())  # same position, no history
    assert b_hist.fen() == b_bare.fen()
    e_hist = encode_position(b_hist, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta",
                             input_extra_features="v2_threats")
    e_bare = encode_position(b_bare, add_features=True,
                             input_history_encoding="lc0_root_legacy_meta",
                             input_extra_features="v2_threats")
    assert (e_hist != e_bare).any()  # history planes differ -> real history used


def test_illegal_move_in_line_is_skipped(tmp_path: Path) -> None:
    bad = f"{chess.STARTING_FEN} | e2e4 e7e5 e4d5"  # e4d5 = pawn capture to an empty square
    path = _write_list(tmp_path, [FEN_BLACK, bad, FEN_WHITE])
    assert _load_fen_list(path) == (FEN_BLACK, FEN_WHITE)  # bad skipped, good kept


def test_inline_comment_is_stripped(tmp_path: Path) -> None:
    line = f"{_SEED_WITH_HISTORY}  # src=game.pgn round=3 ply=42"
    path = _write_list(tmp_path, [line])
    loaded = _load_fen_list(path)
    assert loaded == (_SEED_WITH_HISTORY,)  # comment gone, seed intact
    assert len(seed_board_from_line(loaded[0]).move_stack) == 10


def test_history_line_terminal_validated_not_start(tmp_path: Path) -> None:
    # A line whose START is fine but whose TERMINAL is a forced/near-mate spot
    # is judged on the terminal. Here the terminal has >=2 legal moves -> kept.
    path = _write_list(tmp_path, [_SEED_WITH_HISTORY])
    assert _load_fen_list(path) == (_SEED_WITH_HISTORY,)


def test_sample_returns_history_board(tmp_path: Path) -> None:
    path = _write_list(tmp_path, [_SEED_WITH_HISTORY])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)
    start = sample_starting_board(rng=np.random.default_rng(0), cfg=cfg)
    assert start.source == "fenlist"
    assert len(start.board.move_stack) == 10
    assert start.board.fen() == _terminal_fen(_SEED_WITH_HISTORY)


def test_allow_fenlist_gates_seeding(tmp_path: Path) -> None:
    """Fix A (opening_fen_selfplay_only): a curriculum slot passes allow_fenlist=False
    and NEVER draws a seed even at prob 1.0, so a high dose can't consume curriculum
    slots (and starve the PID). A selfplay slot (allow_fenlist=True) still seeds."""
    path = _write_list(tmp_path, [FEN_BLACK, FEN_WHITE])
    cfg = OpeningConfig(opening_fen_list_path=path, opening_fen_prob=1.0)  # always seed if allowed
    rng = np.random.default_rng(0)
    assert sample_starting_board(rng=rng, cfg=cfg, allow_fenlist=True).source == "fenlist"
    for _ in range(20):
        assert sample_starting_board(rng=rng, cfg=cfg, allow_fenlist=False).source != "fenlist"


# ── Server-doled seeding (opening_fen_dole_per_iter) ─────────────────────────
# resolve_slot_opening is the shared consumption logic used by both create() and
# recycle_slot(); testing it directly needs no C extension.


def _dole_cfg(path: str) -> OpeningConfig:
    # prob=1.0 so a regression that re-enabled the probabilistic draw in dole mode
    # would visibly seed (source "fenlist") and fail the suppression assertions.
    return OpeningConfig(
        opening_fen_list_path=path, opening_fen_dole_per_iter=1, opening_fen_prob=1.0,
    )


def test_dole_selfplay_slot_drains_queue_in_order(tmp_path: Path) -> None:
    cfg = _dole_cfg(_write_list(tmp_path, [FEN_BLACK, FEN_WHITE]))
    queue = [FEN_WHITE, FEN_BLACK]  # explicit order, distinct from the file order
    rng = np.random.default_rng(0)
    s1 = resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=queue)
    s2 = resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=queue)
    assert (s1.source, s1.board.fen()) == ("fenlist", chess.Board(FEN_WHITE).fen())
    assert (s2.source, s2.board.fen()) == ("fenlist", chess.Board(FEN_BLACK).fen())
    assert queue == []  # FIFO-drained
    # Queue empty → dole mode must NOT fall back to the probabilistic seed.
    s3 = resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=queue)
    assert s3.source != "fenlist"


def test_dole_curriculum_slot_never_drains_queue(tmp_path: Path) -> None:
    cfg = _dole_cfg(_write_list(tmp_path, [FEN_BLACK]))
    queue = [FEN_BLACK, FEN_WHITE]
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=False, fen_dole_queue=queue)
        assert s.source != "fenlist"
    assert queue == [FEN_BLACK, FEN_WHITE]  # untouched by curriculum slots


def test_dole_mode_disables_probability_seed(tmp_path: Path) -> None:
    # A selfplay slot with an EMPTY queue must not probability-seed while dole is on
    # (dole owns FEN seeding), even at opening_fen_prob=1.0.
    cfg = _dole_cfg(_write_list(tmp_path, [FEN_BLACK, FEN_WHITE]))
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=[])
        assert s.source != "fenlist"


def test_dole_n_repeats_the_list(tmp_path: Path) -> None:
    # The worker builds queue = list(seeds) * N; each seed then appears N times.
    seeds = _load_fen_list(_write_list(tmp_path, [FEN_BLACK, FEN_WHITE]))
    cfg = OpeningConfig(
        opening_fen_list_path="dummy", opening_fen_dole_per_iter=2, opening_fen_prob=1.0,
    )
    queue = list(seeds) * 2
    rng = np.random.default_rng(0)
    drawn = [
        resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=queue).board.fen()
        for _ in range(4)
    ]
    assert queue == []
    assert drawn.count(chess.Board(FEN_BLACK).fen()) == 2
    assert drawn.count(chess.Board(FEN_WHITE).fen()) == 2


def test_legacy_mode_defers_to_probability_path(tmp_path: Path) -> None:
    # dole off (default 0): resolve_slot_opening behaves like the pre-dole path —
    # the probabilistic draw runs and a passed queue is ignored.
    cfg = OpeningConfig(opening_fen_list_path=_write_list(tmp_path, [FEN_BLACK, FEN_WHITE]),
                        opening_fen_prob=1.0)  # opening_fen_dole_per_iter == 0
    rng = np.random.default_rng(0)
    assert resolve_slot_opening(
        rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=None,
    ).source == "fenlist"
    q = [FEN_BLACK]
    resolve_slot_opening(rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=q)
    assert q == [FEN_BLACK]  # legacy mode never touches the dole queue


def test_legacy_selfplay_only_gate_still_applies(tmp_path: Path) -> None:
    # dole off + opening_fen_selfplay_only: the #127 allow_fenlist gate still holds
    # (curriculum slots don't seed; selfplay slots do).
    cfg = OpeningConfig(
        opening_fen_list_path=_write_list(tmp_path, [FEN_BLACK, FEN_WHITE]),
        opening_fen_prob=1.0, opening_fen_selfplay_only=True,
    )
    rng = np.random.default_rng(0)
    assert resolve_slot_opening(
        rng=rng, cfg=cfg, is_selfplay=False, fen_dole_queue=None,
    ).source != "fenlist"
    assert resolve_slot_opening(
        rng=rng, cfg=cfg, is_selfplay=True, fen_dole_queue=None,
    ).source == "fenlist"


def test_dole_config_round_trips() -> None:
    # (c) The new key survives the config→TrialConfig→OpeningConfig plumbing.
    from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import (
        _SELFPLAY_KEYS,
        flatten_run_config_defaults,
    )

    assert "opening_fen_dole_per_iter" in _SELFPLAY_KEYS
    flat = flatten_run_config_defaults({"selfplay": {"opening_fen_dole_per_iter": 3}})
    assert flat["opening_fen_dole_per_iter"] == 3
    tc = TrialConfig.from_dict({"opening_fen_dole_per_iter": 3})
    assert tc.opening_fen_dole_per_iter == 3
    opening = _play_batch_kwargs(tc)["opening"]
    assert opening.opening_fen_dole_per_iter == 3


def test_backed_fenlist_source_detection(tmp_path: Path) -> None:
    """Lines whose comment carries dropped= are blame-backed decision-point
    seeds and must resolve to the fenlist_backed opening source; plain lines
    stay fenlist. Cache is path-keyed, so use a unique file per test."""
    from chess_anti_engine.selfplay.opening import _fenlist_source

    start = chess.Board().fen()
    plain = f"{start} | e2e4 e7e5"
    backed = f"{start} | e2e4"
    seeds = tmp_path / "seeds_backed_detect.txt"
    seeds.write_text(
        f"# header\n{plain}  # deep_sq=-1.0\n"
        f"{backed}  # deep_sq=-1.0 blame_k=2 blunder=d1h5 dropped=d1h5,b8c6\n",
        encoding="utf-8",
    )
    assert _fenlist_source(plain, str(seeds)) == "fenlist"
    assert _fenlist_source(backed, str(seeds)) == "fenlist_backed"
    assert _fenlist_source(backed, None) == "fenlist"  # no list -> default


@pytest.mark.parametrize(
    ("result", "expect"),
    [("1-0", "w"), ("1/2-1/2", "d"), ("0-1", "l")],
)
def test_backed_fenlist_selfplay_stm_keys_split(result: str, expect: str) -> None:
    """Backed decision-point seeds must aggregate under their OWN stm keys —
    their health semantics invert the normal fenlist ones (stm escape = the
    avoidance goal) — and must stay out of the PID winrate sample."""
    st = _agg_state("fenlist_backed", selfplay=True, seed_fen=chess.Board().fen())
    c = _update_aggregate_stats(st, 0, result=result, was_adjudicated=False, game_plies=30)
    assert c is not None
    assert st.stats.outcome_stats.get(f"selfplay_fenlist_backed_stm_{expect}") == 1
    assert not any(k.startswith("selfplay_fenlist_stm_") for k in st.stats.outcome_stats)
    assert (st.stats.w, st.stats.d, st.stats.l) == (0, 0, 0)


def test_backed_fenlist_curriculum_excluded_from_pid_winrate() -> None:
    """count_in_pid must exclude fenlist_backed curriculum games exactly like
    plain fenlist ones (startswith guard, not equality)."""
    st = _agg_state("fenlist_backed", selfplay=False, seed_fen=chess.Board().fen())
    _update_aggregate_stats(st, 0, result="0-1", was_adjudicated=False, game_plies=30)
    assert (st.stats.w, st.stats.d, st.stats.l) == (0, 0, 0)


def test_dole_seed_weights(tmp_path: Path) -> None:
    """weight=N comment markers multiply a seed's dole exposure; weight=0
    soft-retires it; absurd weights cap at _MAX_SEED_WEIGHT; unmarked seeds
    default to 1. Path-keyed cache -> unique file per test."""
    from chess_anti_engine.selfplay.opening import _MAX_SEED_WEIGHT, expand_dole_seeds

    start = chess.Board().fen()
    a = f"{start} | e2e4"
    b = f"{start} | d2d4"
    c = f"{start} | c2c4"
    d = f"{start} | g1f3"
    seeds_file = tmp_path / "seeds_weights.txt"
    seeds_file.write_text(
        f"{a}  # weight=3\n"
        f"{b}  # weight=0 retired-in-place\n"
        f"{c}  # weight=999\n"
        f"{d}  # plain seed, no marker\n",
        encoding="utf-8",
    )
    out = expand_dole_seeds([a, b, c, d], str(seeds_file))
    assert out.count(a) == 3
    assert out.count(b) == 0
    assert out.count(c) == _MAX_SEED_WEIGHT
    assert out.count(d) == 1
    # No list path -> identity (probabilistic path / defensive default).
    assert expand_dole_seeds([a, b], None) == [a, b]
