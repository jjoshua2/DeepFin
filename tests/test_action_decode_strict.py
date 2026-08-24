"""Evaluation must not launder an undecodable action id into a legal move.

The laundering had two layers. `apply_actions_to_boards` re-checked the decoded
move against `board.legal_moves` and substituted the first legal move if it
failed — but that guard could never fire, because every return path of
`index_to_move_fast` is already legal: the substitution happens one level down,
inside `moves/encode.py`, on two branches (no LUT entry for this side to move,
and a constructed move that is illegal with no legal move encoding back to the
id). So a broken action space produced arena games that were scored normally.

These tests pin both layers: the strict decoder raises, the resilient one
substitutes AND counts, the evaluation call sites ask for strict, and the count
reaches the metric stream an operator actually reads.
"""

from __future__ import annotations

import ast
from pathlib import Path
import types

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.moves import (
    ActionDecodeError,
    decode_fallback_count,
    drain_decode_fallback_count,
    index_to_move_fast,
    index_to_move_strict,
    move_to_index,
)
from chess_anti_engine.moves import encode
from chess_anti_engine.moves.encode import _INDEX_TO_MOVE_LUT
from chess_anti_engine.selfplay import match as match_mod
from chess_anti_engine.selfplay.match import apply_actions_to_boards, play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig

ROOT = Path(__file__).resolve().parent.parent

# From-square a1 (oriented) with plane 21: a queen ray that leaves the board, so
# the reverse LUT holds nothing for it — for either side to move, since the LUT
# is built in oriented coordinates. Undecodable in EVERY position.
NO_LUT_ENTRY_ID = 21
# a1a2 from White's side: the LUT does hold an entry, but the rook is blocked by
# its own pawn in the start position and no legal move there encodes back to it.
BLOCKED_ROOK_ID = 0

# The move python-chess yields first in the start position; what both resilient
# decoders substitute for the two ids above.
FIRST_LEGAL_STARTPOS = chess.Move.from_uci("g1h3")


def test_fixture_ids_really_are_the_two_failure_branches() -> None:
    """Pin the fixtures to the branches they stand for.

    A LUT change could quietly make either id decodable, and both tests below
    would then pass while measuring nothing.
    """
    board = chess.Board()
    for turn in (chess.WHITE, chess.BLACK):
        assert int(_INDEX_TO_MOVE_LUT[int(turn), NO_LUT_ENTRY_ID][0]) < 0

    entry = _INDEX_TO_MOVE_LUT[int(board.turn), BLOCKED_ROOK_ID]
    assert int(entry[0]) >= 0, "this id must HAVE a LUT entry — that is the point"
    assert chess.Move(int(entry[0]), int(entry[1])) == chess.Move.from_uci("a1a2")
    assert int(move_to_index(chess.Move.from_uci("a1a2"), board)) == BLOCKED_ROOK_ID
    assert chess.Move.from_uci("a1a2") not in board.legal_moves
    assert next(iter(board.legal_moves)) == FIRST_LEGAL_STARTPOS


@pytest.mark.parametrize(
    ("action_id", "detail"),
    [(NO_LUT_ENTRY_ID, "no LUT entry"), (BLOCKED_ROOK_ID, "a1a2")],
)
def test_strict_decode_raises_naming_the_id_and_the_position(
    action_id: int, detail: str,
) -> None:
    board = chess.Board()
    with pytest.raises(ActionDecodeError) as excinfo:
        index_to_move_strict(action_id, board)

    message = str(excinfo.value)
    assert str(action_id) in message, "the message must name the action id"
    assert board.fen() in message, "the message must carry the position"
    assert detail in message
    assert excinfo.value.index == action_id
    assert excinfo.value.fen == board.fen()
    assert excinfo.value.turn is True
    assert board.move_stack == [], "decoding must not mutate the board"


@pytest.mark.parametrize("action_id", [NO_LUT_ENTRY_ID, BLOCKED_ROOK_ID])
def test_fast_decode_substitutes_first_legal_and_counts_it(action_id: int) -> None:
    board = chess.Board()
    before = decode_fallback_count()

    assert index_to_move_fast(action_id, board) == FIRST_LEGAL_STARTPOS
    assert decode_fallback_count() == before + 1, (
        "the substitution must be counted — an uncounted one is invisible"
    )


def test_first_substitution_is_announced_on_stderr_then_only_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The count is for whoever asks; the one-shot line is for whoever does not.

    Worker stderr is captured into the run log (``stderr=subprocess.STDOUT`` in
    ``tune/distributed_runtime.py``, ``> "$LOG" 2>&1`` in ``scripts/train.sh``),
    so this line is what makes a live substitution visible without anyone
    thinking to look. stdout is the UCI protocol channel and must stay clean.
    """
    monkeypatch.setattr(encode, "_DECODE_FALLBACK_WARNED", False)
    board = chess.Board()

    index_to_move_fast(BLOCKED_ROOK_ID, board)
    first = capsys.readouterr()
    assert first.out == "", "stdout is the UCI protocol channel"
    assert str(BLOCKED_ROOK_ID) in first.err
    assert board.fen() in first.err
    assert "SUBSTITUTED" in first.err

    index_to_move_fast(BLOCKED_ROOK_ID, board)
    assert capsys.readouterr().err == "", "further occurrences are counted, not logged"


def test_no_legal_move_is_not_recorded_as_a_substitution() -> None:
    """A board with no legal moves has nothing to substitute — count nothing.

    The resilient decoder ends in `next(iter(board.legal_moves))`, which raises
    on an empty set. Counting and announcing "SUBSTITUTED ... kept playing"
    before that would bank a move that never happened.
    """
    stalemated = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert not list(stalemated.legal_moves), "fixture must have no legal move"
    before = decode_fallback_count()

    with pytest.raises(StopIteration):
        index_to_move_fast(NO_LUT_ENTRY_ID, stalemated)

    assert decode_fallback_count() == before, (
        "no substitute existed, so nothing was substituted"
    )


def test_apply_actions_strict_propagates_and_leaves_the_board_alone() -> None:
    boards = [chess.Board()]
    with pytest.raises(ActionDecodeError):
        apply_actions_to_boards(boards, [0], [BLOCKED_ROOK_ID], strict=True)
    assert boards[0].move_stack == []


def test_apply_actions_non_strict_substitutes_and_counts() -> None:
    boards = [chess.Board()]
    before = decode_fallback_count()

    apply_actions_to_boards(boards, [0], [BLOCKED_ROOK_ID], strict=False)

    assert boards[0].move_stack == [FIRST_LEGAL_STARTPOS]
    assert decode_fallback_count() == before + 1


def test_legal_moves_round_trip_identically_through_fast_and_strict() -> None:
    """Every legal move decodes back to itself, and neither decoder falls back."""
    rng = np.random.default_rng(20260823)
    before = decode_fallback_count()
    positions = 0
    moves_checked = 0

    for _game in range(40):
        board = chess.Board()
        for _ply in range(24):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            positions += 1
            for move in legal:
                index = int(move_to_index(move, board))
                assert index_to_move_strict(index, board) == move
                assert index_to_move_fast(index, board) == move
                moves_checked += 1
            board.push(legal[int(rng.integers(0, len(legal)))])

    assert positions >= 400, f"walk must cover a few hundred positions, got {positions}"
    assert moves_checked >= 10_000
    assert decode_fallback_count() == before, (
        "a legal move must never take the substitution path"
    )


def _legal_actions(boards: list[chess.Board]) -> list[int]:
    return [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards]


@pytest.mark.parametrize("a_plays_white", [True, False])
def test_play_match_batch_propagates_a_decode_failure(
    monkeypatch: pytest.MonkeyPatch, a_plays_white: bool,
) -> None:
    """The paired match is a gate/eval instrument, so it must not play on.

    ``max_plies=1`` deliberately: only the side to move decodes, so each
    parametrisation pins ONE of the two `apply_actions_to_boards` call sites.
    Two plies would let the other call site's raise cover for a lenient one.
    """
    def _undecodable(_model: object, sub_boards: list[chess.Board], **_kw: object) -> list[int]:
        return [NO_LUT_ENTRY_ID] * len(sub_boards)

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", _undecodable)

    with pytest.raises(ActionDecodeError):
        play_match_batch(
            None, None,  # pyright: ignore[reportArgumentType] - the search is stubbed
            device="cpu",
            rng=np.random.default_rng(0),
            games=1,
            max_plies=1,
            a_plays_white=[a_plays_white],
            mcts_type="gumbel",
            mcts_simulations=1,
            opening_cfg=OpeningConfig(random_start_plies=0),
        )


@pytest.mark.parametrize("rolling", [False, True])
def test_arena_call_sites_ask_for_strict(monkeypatch: pytest.MonkeyPatch, rolling: bool) -> None:
    """Both `arena_standard` play loops must decode strictly.

    Behavioural, not a source scan: the recorder sits where the arena actually
    calls, so flipping either call site to `strict=False` fails this.
    """
    import scripts.arena_standard as arena

    seen: list[bool] = []
    real_apply = match_mod.apply_actions_to_boards

    def _record(
        boards: list[chess.Board], idxs: list[int], actions: list[int], *, strict: bool,
    ) -> None:
        seen.append(bool(strict))
        real_apply(boards, idxs, actions, strict=strict)

    def _pick(_model: object, sub_boards: list[chess.Board], **_kw: object) -> list[int]:
        return _legal_actions(sub_boards)

    monkeypatch.setattr(match_mod, "apply_actions_to_boards", _record)
    monkeypatch.setattr(match_mod, "pick_moves_for_boards", _pick)

    side = arena.resolve_search_shape("training")
    common: dict[str, object] = {
        "device": "cpu",
        "rng": np.random.default_rng(0),
        "sims_candidate": 1,
        "sims_reference": 1,
        "max_plies": 4,
        "temperature": 0.1,
        "gumbel_add_noise": False,
        "search_candidate": side,
        "search_reference": side,
    }
    play = (
        arena.play_paired_games_matched_sims_rolling
        if rolling
        else arena.play_paired_games_matched_sims
    )
    play(None, None, [chess.Board()], **common)  # pyright: ignore[reportArgumentType]

    assert seen, "the arena never applied an action — the test measured nothing"
    assert all(seen), "an arena play loop decoded actions non-strictly"


def _apply_call_sites() -> list[tuple[str, int, ast.keyword | None]]:
    """Every `apply_actions_to_boards(...)` call in shipped code, repo-wide.

    Globbed, not listed: a hardcoded file list answers for the files it names
    and silently passes a third one. `tests/` is excluded because a test may
    legitimately exercise the resilient mode — the shipped tree may not.
    """
    found: list[tuple[str, int, ast.keyword | None]] = []
    for root in ("chess_anti_engine", "scripts"):
        for path in sorted((ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name != "apply_actions_to_boards":
                    continue
                relpath = str(path.relative_to(ROOT))
                keyword = next((k for k in node.keywords if k.arg == "strict"), None)
                found.append((relpath, node.lineno, keyword))
    return found


def test_every_shipped_call_site_passes_strict_true() -> None:
    """A NEW call site anywhere in shipped code must not inherit the resilient mode.

    `strict` is required, so omitting it is a TypeError at runtime; this catches
    the halves a TypeError cannot — `strict=False`, or a non-literal whose value
    the reader cannot see. The behavioural tests above remain the load-bearing
    layer; this only stops a third file from dodging them.
    """
    sites = _apply_call_sites()
    assert len(sites) >= 4, f"expected the known call sites, found {sites}"
    for relpath, lineno, keyword in sites:
        where = f"{relpath}:{lineno}"
        assert keyword is not None, f"{where}: call site omits strict="
        assert isinstance(keyword.value, ast.Constant), (
            f"{where}: strict= must be a literal so the mode is readable here"
        )
        assert keyword.value.value is True, f"{where}: shipped call site is not strict"


# ---------------------------------------------------------------------------
# The count has to reach somewhere an operator reads
# ---------------------------------------------------------------------------

def test_drain_reports_each_substitution_exactly_once() -> None:
    drain_decode_fallback_count()  # start from a known watermark
    board = chess.Board()

    index_to_move_fast(BLOCKED_ROOK_ID, board)
    index_to_move_fast(NO_LUT_ENTRY_ID, board)

    assert drain_decode_fallback_count() == 2
    assert drain_decode_fallback_count() == 0, "a drained substitution must not re-report"
    assert decode_fallback_count() > 0, "the cumulative reader stays cumulative"


def test_finalize_publishes_the_count_into_outcome_stats() -> None:
    """The publisher: selfplay finalize folds the drain into the game's stats.

    `_update_aggregate_stats` is the real function — the state is duck-typed to
    the attributes it reads, because building a whole SelfplayState would test
    the fixture rather than the wiring.
    """
    from chess_anti_engine.selfplay import finalize as finalize_mod
    from chess_anti_engine.selfplay.state import _StatsAcc

    drain_decode_fallback_count()
    board = chess.Board()
    state = types.SimpleNamespace(
        selfplay_arr=[True],
        opening_source_arr=["book"],
        starting_boards=[board],
        stats=_StatsAcc(),
        pending_outcome_stats={},
    )
    index_to_move_fast(BLOCKED_ROOK_ID, board)
    index_to_move_fast(NO_LUT_ENTRY_ID, board)

    counters = finalize_mod._update_aggregate_stats(
        state,  # pyright: ignore[reportArgumentType] - duck-typed on purpose
        0, result="1/2-1/2", was_adjudicated=False, game_plies=10,
    )

    assert counters.outcome_stats is not None
    assert counters.outcome_stats["action_decode_fallbacks"] == 2
    assert dict(state.stats.outcome_stats)["action_decode_fallbacks"] == 2

    # The next finalized game must not re-report the same two.
    again = finalize_mod._update_aggregate_stats(
        state,  # pyright: ignore[reportArgumentType] - duck-typed on purpose
        0, result="1/2-1/2", was_adjudicated=False, game_plies=10,
    )
    assert again.outcome_stats is not None
    assert "action_decode_fallbacks" not in again.outcome_stats


def test_count_survives_every_real_hop_to_the_operator_column() -> None:
    """The transport: worker buffer -> server -> ingest summary -> report column.

    Every hop is the SHIPPED function, not a re-implementation, because each one
    filters keys by its own regex and any of them could silently drop this one.
    """
    from chess_anti_engine.server import app as server_app
    from chess_anti_engine.tune import distributed_runtime as dr
    from chess_anti_engine.tune import trainable_phases as phases
    from chess_anti_engine.worker_buffer import _merge_outcome_stats as worker_merge
    from scripts.loop_health import parse_outcome_stats

    per_game = {"action_decode_fallbacks": 7}

    buffered: dict[str, int] = {}
    worker_merge(buffered, per_game)
    worker_merge(buffered, per_game)  # a second game in the same shard
    assert buffered["action_decode_fallbacks"] == 14

    server_acc: dict[str, int] = {}
    server_app._merge_outcome_stats(server_acc, {"outcome_stats": buffered}["outcome_stats"])
    assert server_acc["action_decode_fallbacks"] == 14

    metrics = dr._extract_shard_metrics({"outcome_stats": server_acc}, 1)
    assert metrics["outcome_stats"]["action_decode_fallbacks"] == 14

    summary: dict[str, dict[str, int]] = {"matching_outcome_stats": {}}
    dr._merge_outcome_stats(summary["matching_outcome_stats"], metrics["outcome_stats"])
    fields = phases._selfplay_diagnostic_fields_from_ingest(summary)
    assert fields["outcome_stats"]["action_decode_fallbacks"] == 14

    # The report column is one pipe-joined string; `loop_health` is what parses
    # it back for an operator, so close the loop on the reader, not the writer.
    column = "|".join(f"{k}={int(v)}" for k, v in sorted(fields["outcome_stats"].items()))
    assert parse_outcome_stats(column)["action_decode_fallbacks"] == 14


# ---------------------------------------------------------------------------
# The other measurement paths
# ---------------------------------------------------------------------------

class _NullNet(torch.nn.Module):
    """Stands in for a checkpoint; the search that consumes it is stubbed."""

    input_history_encoding = "lc0_root_legacy_meta"
    input_extra_features = "v2_threats"

    def forward(self, _x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise AssertionError("the search is stubbed; the net must not be called")


def test_puzzle_eval_raises_instead_of_scoring_a_substitute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The puzzle suite is a yardstick: a substitute must not be graded.

    Laundering here reports the substituted move's miss as a puzzle the net got
    wrong, so an action-space regression reads as a strength regression.
    """
    from chess_anti_engine.eval import puzzles as puzzles_mod

    board = chess.Board()
    suite = puzzles_mod.PuzzleSuite(
        puzzles=[puzzles_mod.Puzzle(board=board, best_moves=[chess.Move.from_uci("e2e4")])],
        name="unit",
    )

    def _undecodable(_model, boards, **_kw):
        return (None, [NO_LUT_ENTRY_ID] * len(boards))

    monkeypatch.setattr(puzzles_mod, "run_mcts_many", _undecodable)

    with pytest.raises(ActionDecodeError):
        puzzles_mod.run_puzzle_eval(
            _NullNet(), suite, device="cpu", mcts_simulations=1, batch_size=1,
        )


def test_arena_run_voids_the_run_and_writes_no_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A corrupted instrument must exit non-zero and bank nothing.

    Not a source check: `run_arena` is driven for real, with only the model load
    and the play loop stubbed, so the assertion is that no JSONL row exists —
    the thing a downstream pooled fit would otherwise pick up.
    """
    import scripts.arena_standard as arena

    fens = tmp_path / "openings.txt"
    fens.write_text(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "results.jsonl"
    board = chess.Board()
    failure = ActionDecodeError(NO_LUT_ENTRY_ID, board, "no LUT entry for this side to move")

    def _boom(*_args: object, **_kw: object) -> list[float]:
        raise failure

    monkeypatch.setattr(
        "chess_anti_engine.uci.model_loader.load_model_from_checkpoint",
        lambda *a, **k: _NullNet(),
    )
    monkeypatch.setattr(arena, "play_paired_games_matched_sims_rolling", _boom)
    side = arena.resolve_search_shape("training")

    with pytest.raises(SystemExit) as excinfo:
        arena.run_arena(
            candidate="cand.pt", reference="ref.pt", games=2,
            openings_path=None, opening_plies=0, openings_fen=fens,
            mode="matched_sims", sims_candidate=1, sims_reference=1,
            ms_per_move=0, max_plies=4, temperature=0.1, gumbel_add_noise=False,
            device="cpu", seed=0, out_path=out_path,
            search_candidate=side, search_reference=side,
            compile_models=False, rolling=True,
        )

    assert excinfo.value.code == 2, "a VOID run must exit non-zero"
    assert not out_path.exists(), "a corrupted arena must bank no result row"
    err = capsys.readouterr().err
    assert "VOID" in err
    assert str(NO_LUT_ENTRY_ID) in err
    assert board.fen() in err


def test_worker_arena_task_contains_the_raise() -> None:
    """A void measurement must not kill the worker process.

    `WorkerSession.run` has no other `except` and a `finally` that exits, so
    an uncaught ActionDecodeError from the arena task takes the whole worker
    down. Driven through the real `run` body against a duck-typed worker.
    """
    from chess_anti_engine.worker import WorkerSession

    calls: list[str] = []
    errors: list[str] = []

    def _run_arena(_manifest: dict, _task: dict) -> None:
        calls.append("arena")
        raise ActionDecodeError(NO_LUT_ENTRY_ID, chess.Board(), "no LUT entry")

    stub = types.SimpleNamespace(
        _install_shutdown_handlers=lambda: None,
        _shutdown_requested=False,
        _shutdown_signal=0,
        _poll_manifest=lambda: {"task": {"type": "arena"}},
        _sync_assets=lambda _m: None,
        _maybe_ingest_dole_flag=lambda _m: None,
        model_sha="deadbeef",
        _run_arena=_run_arena,
        _cleanup=lambda: None,
        args=types.SimpleNamespace(poll_seconds=0.0),
        log=types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda fmt, *a: errors.append(str(fmt) % a if a else str(fmt)),
        ),
    )

    def _stop_after_first(_m: dict) -> None:
        stub._shutdown_requested = True

    stub._maybe_ingest_dole_flag = _stop_after_first

    WorkerSession.run(stub)  # pyright: ignore[reportArgumentType] - duck-typed on purpose

    assert calls == ["arena"], "the arena task must have run once"
    assert errors, "the containment must log, not swallow silently"
    assert "VOID" in errors[0], f"expected a VOID error line, got {errors}"


def test_promotion_guard_keeps_a_non_pawn_off_the_round_trip_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pawn-on-from-square check must actually gate the LUT promotion.

    The LUT marks any move landing on rank 1/8 as a queen-promotion candidate,
    so without the check a ROOK reaching the back rank decodes as `a7a8q`. That
    is illegal, and the round-trip fallback would then find the plain rook move
    and return the SAME answer — which is why a result-only assertion cannot see
    this regression at all. What it can see is the fallback being entered:
    `move_to_index` is called once per legal move there, and not at all when the
    guard does its job.
    """
    board = chess.Board("4k3/R7/8/8/8/8/8/4K3 w - - 0 1")
    rook_move = chess.Move.from_uci("a7a8")
    index = int(move_to_index(rook_move, board))
    assert int(_INDEX_TO_MOVE_LUT[int(board.turn), index][2]) == chess.QUEEN, (
        "fixture must be a LUT entry that CARRIES a promotion candidate"
    )

    calls: list[chess.Move] = []
    real = encode.move_to_index

    def _counting(move: chess.Move, b: chess.Board) -> int:
        calls.append(move)
        return real(move, b)

    monkeypatch.setattr(encode, "move_to_index", _counting)

    assert index_to_move_fast(index, board) == rook_move
    assert calls == [], (
        "the decode fell through to the round-trip scan: the promotion guard "
        "let a queen promotion be constructed for a rook"
    )
