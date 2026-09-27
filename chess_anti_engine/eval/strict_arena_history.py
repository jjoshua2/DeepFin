"""CPU-only, byte-bound full-history verification for future matched-sims arenas.

The caller owns capture of immutable JSONL/PGN bytes, an independently reconstructed
opening schedule, expected settings/tags, and a separately opened strict tablebase.
This module never opens a model, book, tablebase path, or GPU device.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

import chess
import chess.pgn
import chess.syzygy

from chess_anti_engine.utils.game_log import settings_fingerprint

SYZYGY_PROTOCOL = "rule50-aware-root-leaf-and-adjudication-v1"
_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})
_MAX_INPUT_BYTES = 64 * 1024 * 1024


class HistoryRefusal(ValueError):
    """The archived game population cannot earn full-history verification."""


class WdlDtzProbe(Protocol):
    """An owned strict tablebase handle; opening and identity are caller gates."""

    def probe_wdl(self, board: chess.Board) -> int: ...

    def probe_dtz(self, board: chess.Board) -> int: ...


@dataclass(frozen=True)
class HistoryReceipt:
    jsonl_sha256: str
    pgn_sha256: str
    pairs: int
    games: int
    rules_games: int
    syzygy_games: int
    replayed_pairs: tuple[int, ...]
    pair_ids: tuple[int, ...]
    opening_root_fen: str
    opening_plies: int
    compile_tags: tuple[str, ...]
    hoist_tags: tuple[str, ...]


def _refuse(message: str) -> NoReturn:
    raise HistoryRefusal(message)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _refuse(f"{label} is not an integer")
    return value


def _json_object(blob: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(blob, parse_constant=lambda word: _refuse(f"{label}: {word}"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HistoryRefusal(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        _refuse(f"{label} is not a JSON object")
    return value


def _log_rows(blob: bytes, expected_settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not blob.endswith(b"\n"):
        _refuse("JSONL has an incomplete final line")
    try:
        lines = blob.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HistoryRefusal("JSONL is not UTF-8") from exc
    if len(lines) < 3 or any(not line for line in lines):
        _refuse("JSONL lacks a header and a complete pair, or has blank rows")
    header = _json_object(lines[0], "JSONL header")
    if (header.get("kind") != "header" or header.get("version") != 1
            or header.get("driver") != "arena_standard"):
        _refuse("JSONL header is not arena_standard version 1")
    if header.get("settings") != dict(expected_settings):
        _refuse("JSONL settings differ from independently expected settings")
    if header.get("fingerprint") != settings_fingerprint(expected_settings):
        _refuse("JSONL settings fingerprint differs")
    rows = [_json_object(line, f"JSONL line {i}") for i, line in enumerate(lines[1:], 2)]
    if any(row.get("kind") != "game" for row in rows):
        _refuse("JSONL has a non-game row after its header")
    return rows


def _pgn_games(blob: bytes, maximum: int) -> list[chess.pgn.Game]:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoryRefusal("PGN is not UTF-8") from exc
    if not text.strip():
        _refuse("PGN is empty")
    stream = io.StringIO(text)
    games: list[chess.pgn.Game] = []
    while stream.tell() < len(text):
        before = stream.tell()
        game = chess.pgn.read_game(stream)
        if game is None:
            if text[before:].strip():
                _refuse("PGN has an unparsed trailing fragment")
            break
        if game.errors:
            _refuse("PGN has parser errors")
        node: chess.pgn.GameNode = game
        while node.variations:
            if len(node.variations) != 1:
                _refuse("PGN has a branch absent from the arena writer contract")
            node = node.variations[0]
        games.append(game)
        if len(games) > maximum:
            _refuse("PGN game count exceeds bounded expected population")
    return games


def _key(pair: object, half: object, label: str) -> tuple[int, int]:
    pair_id = _integer(pair, f"{label} pair")
    part = _integer(half, f"{label} half")
    if pair_id < 0 or part not in (0, 1):
        _refuse(f"{label} pair/half is out of range")
    return pair_id, part


def _pgn_key(game: chess.pgn.Game) -> tuple[int, int]:
    try:
        pair = int(game.headers["PairId"])
        half = int(game.headers["PairHalf"])
    except (KeyError, ValueError) as exc:
        raise HistoryRefusal("PGN lacks valid PairId/PairHalf") from exc
    if str(pair) != game.headers["PairId"] or str(half) != game.headers["PairHalf"]:
        _refuse("PGN pair/half is not canonical decimal")
    return _key(pair, half, "PGN")


def _row_signature(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row.get("result"), row.get("termination"), row.get("plies"),
        row.get("start_fen"), row.get("opening_root_fen"),
        json.dumps(row.get("opening_uci"), sort_keys=True),
    )


def _replay_marker(game: chess.pgn.Game) -> bool:
    marker = game.headers.get("ResumeReplay")
    if marker not in (None, "1"):
        _refuse("PGN has an invalid ResumeReplay marker")
    return marker == "1"


def _attempt_agrees(row: Mapping[str, Any], game: chess.pgn.Game) -> bool:
    """Bind each archived attempt's visible row fields to its PGN, even if stale."""
    headers = game.headers
    try:
        pgn_opening = json.loads(headers["OpeningUCI"])
        pgn_plies = int(headers["Plies"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return False
    return (
        headers.get("OpeningRootFEN") == row.get("opening_root_fen")
        and pgn_opening == row.get("opening_uci")
        and game.board().fen() == row.get("start_fen")
        and headers.get("Result") == row.get("result")
        and headers.get("Termination") == row.get("termination")
        and pgn_plies == row.get("plies")
        and len(list(game.mainline_moves())) == pgn_plies
    )


def _resolve_attempts(
    rows: Sequence[dict[str, Any]], games: Sequence[chess.pgn.Game],
    wanted: set[tuple[int, int]],
    expected_openings: Mapping[int, chess.Board],
    expected_pgn_tags: Mapping[str, str],
    expected_seed: int,
    candidate_name: str,
    reference_name: str,
    candidate_search: str,
    reference_search: str,
) -> tuple[dict[tuple[int, int], dict[str, Any]],
           dict[tuple[int, int], chess.pgn.Game], tuple[int, ...]]:
    row_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    pgn_groups: dict[tuple[int, int], list[chess.pgn.Game]] = defaultdict(list)
    for row in rows:
        row_groups[_key(row.get("pair_id"), row.get("half"), "JSONL")].append(row)
    for game in games:
        pgn_groups[_pgn_key(game)].append(game)
    if set(row_groups) != wanted or set(pgn_groups) != wanted:
        _refuse("JSONL/PGN pair-half population differs from expected schedule")
    selected_rows: dict[tuple[int, int], dict[str, Any]] = {}
    selected_games: dict[tuple[int, int], chess.pgn.Game] = {}
    replayed: set[int] = set()
    for key in sorted(wanted):
        candidates = row_groups[key]
        archives = pgn_groups[key]
        if len(candidates) != len(archives):
            _refuse(f"pair-half {key} has unmatched JSONL/PGN attempts")
        if any(not _attempt_agrees(row, game) for row, game in zip(candidates, archives)):
            _refuse(f"pair-half {key} has unmatched attempt contents")
        opening = expected_openings[key[0]]
        root = opening.root().fen()
        stack = [move.uci() for move in opening.move_stack]
        for row, game in zip(candidates, archives):
            candidate_white = key[1] == 0
            if (row.get("opening_root_fen") != root or row.get("opening_uci") != stack
                    or row.get("opening_fen") != opening.fen()
                    or row.get("start_fen") != opening.fen()
                    or row.get("opening_index") != key[0]
                    or row.get("a_is_white") is not candidate_white
                    or row.get("seed") != expected_seed
                    or row.get("loop") not in ("chunked", "rolling")
                    or row.get("result") not in _RESULTS
                    or type(row.get("score_candidate")) is not float
                    or row.get("score_candidate") != _score(row["result"], candidate_white)
                    or not isinstance(row.get("compile"), str)
                    or not row["compile"]
                    or not isinstance(row.get("eval_hoist"), str)
                    or not row["eval_hoist"]
                    or game.headers.get("White") != (
                        candidate_name if candidate_white else reference_name
                    )
                    or game.headers.get("Black") != (
                        reference_name if candidate_white else candidate_name
                    )
                    or game.headers.get("WhiteSearch") != (
                        candidate_search if candidate_white else reference_search
                    )
                    or game.headers.get("BlackSearch") != (
                        reference_search if candidate_white else candidate_search
                    )
                    or game.headers.get("EvaluatorHoist") != row.get("eval_hoist")
                    or any(game.headers.get(tag) != value
                           for tag, value in expected_pgn_tags.items())):
                _refuse(f"pair-half {key} has a stale or selected source/opening mismatch")
        if len(candidates) > 2:
            _refuse(f"pair-half {key} has more than one replay")
        if len(candidates) == 2:
            if _row_signature(candidates[0]) == _row_signature(candidates[1]):
                _refuse(f"pair-half {key} has indistinguishable JSONL attempts")
            if [_replay_marker(game) for game in archives] != [False, True]:
                _refuse(f"pair-half {key} has ambiguous ResumeReplay order")
            replayed.add(key[0])
        elif _replay_marker(archives[0]):
            replayed.add(key[0])
        selected_rows[key] = candidates[-1]
        selected_games[key] = archives[-1]
    for pair in replayed:
        if not any(len(row_groups[(pair, half)]) == 2 for half in (0, 1)):
            _refuse(f"pair {pair} has a replay marker but no replaced attempt")
        if not all(_replay_marker(selected_games[(pair, half)]) for half in (0, 1)):
            _refuse(f"pair {pair} has only one marked replay half")
    return selected_rows, selected_games, tuple(sorted(replayed))


def _rule50_result(board: chess.Board, tablebase: WdlDtzProbe, max_pieces: int) -> str | None:
    """Pinned rule50_match_result semantics, including WDL+DTZ consistency checks."""
    natural = board.outcome(claim_draw=True)
    if natural is not None:
        return board.result(claim_draw=True)
    if chess.popcount(board.occupied) > max_pieces or board.castling_rights:
        return None
    try:
        wdl = int(tablebase.probe_wdl(board))
        dtz = int(tablebase.probe_dtz(board))
    except (KeyError, IndexError, chess.syzygy.MissingTableError) as exc:
        raise HistoryRefusal(f"missing eligible WDL/DTZ probe at {board.fen()}") from exc
    if wdl not in (-2, -1, 0, 1, 2):
        _refuse("Syzygy WDL is outside -2..2")
    if wdl == 0:
        if dtz != 0:
            _refuse("Syzygy WDL/DTZ draw mismatch")
        return "1/2-1/2"
    if (wdl > 0) != (dtz > 0) or dtz == 0:
        _refuse("Syzygy WDL/DTZ sign mismatch")
    if (abs(wdl) == 2 and abs(dtz) > 100) or (abs(wdl) == 1 and abs(dtz) <= 100):
        _refuse("Syzygy WDL/DTZ magnitude mismatch")
    if abs(wdl) == 1:
        return "1/2-1/2"
    if board.halfmove_clock != 0:
        return None
    white_wins = (wdl > 0) == (board.turn == chess.WHITE)
    return "1-0" if white_wins else "0-1"


def _score(result: str, candidate_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    return float((result == "1-0") == candidate_white)


def verify_strict_arena_history(
    jsonl_bytes: bytes, pgn_bytes: bytes, *,
    expected_settings: Mapping[str, Any],
    expected_openings: Mapping[int, chess.Board],
    expected_pgn_tags: Mapping[str, str],
    candidate_name: str,
    reference_name: str,
    candidate_search: str,
    reference_search: str,
    tablebase: WdlDtzProbe | None,
    expected_opening_root_fen: str = chess.STARTING_FEN,
    expected_opening_plies: int = 16,
    max_input_bytes: int = _MAX_INPUT_BYTES,
) -> HistoryReceipt:
    """Verify exactly the supplied completed pair population from two byte snapshots.

    ``expected_openings`` is independently regenerated and contains only scored
    pair IDs. The caller must authenticate source, final result, tablebase custody,
    and both immutable byte captures before this receipt can support admission.
    """
    if not 0 < max_input_bytes <= _MAX_INPUT_BYTES:
        _refuse("input cap exceeds 64 MiB")
    if not 0 < len(jsonl_bytes) <= max_input_bytes or not 0 < len(pgn_bytes) <= max_input_bytes:
        _refuse("JSONL or PGN exceeds the input cap")
    if not expected_openings or any(type(pair) is not int or pair < 0 for pair in expected_openings):
        _refuse("expected pair schedule is empty or invalid")
    scheduled_games = expected_settings.get("games")
    if (type(scheduled_games) is not int or scheduled_games < 2
            or scheduled_games % 2 or max(expected_openings) >= scheduled_games // 2):
        _refuse("expected pair IDs exceed the arena's scheduled games")
    if type(expected_opening_plies) is not int or not 0 <= expected_opening_plies <= 256:
        _refuse("expected opening ply count is invalid")
    try:
        canonical_root = chess.Board(expected_opening_root_fen).fen()
    except ValueError as exc:
        raise HistoryRefusal("expected opening root FEN is invalid") from exc
    if canonical_root != expected_opening_root_fen:
        _refuse("expected opening root FEN is not canonical")
    if expected_settings.get("mode") != "matched_sims":
        _refuse("only matched_sims can claim preserved opening history")
    if (expected_settings.get("syzygy_protocol") != SYZYGY_PROTOCOL
            or expected_settings.get("syzygy_max_pieces") != 6
            or not expected_settings.get("syzygy")):
        _refuse("strict six-man rule50 protocol is not declared")
    if tablebase is None:
        _refuse("owned six-man WDL+DTZ handle is required")
    required_tags = {"ConfigHash", "GitSha", "ArenaMode", "SyzygyProtocol", "SyzygyMaxPieces"}
    if not required_tags.issubset(expected_pgn_tags):
        _refuse("expected PGN source/protocol tags are incomplete")
    if any(not expected_pgn_tags[tag] for tag in required_tags):
        _refuse("expected PGN source/protocol tags are empty")
    if (expected_pgn_tags["ArenaMode"] != "matched_sims"
            or expected_pgn_tags["SyzygyProtocol"] != SYZYGY_PROTOCOL
            or expected_pgn_tags["SyzygyMaxPieces"] != "6"):
        _refuse("expected PGN protocol tags are inconsistent")
    rows = _log_rows(jsonl_bytes, expected_settings)
    wanted = {(pair, half) for pair in expected_openings for half in (0, 1)}
    games = _pgn_games(pgn_bytes, maximum=2 * len(wanted))
    selected_rows, selected_games, replayed = _resolve_attempts(
        rows, games, wanted, expected_openings, expected_pgn_tags,
        _integer(expected_settings.get("seed"), "expected seed"),
        candidate_name, reference_name, candidate_search, reference_search,
    )
    rules = syzygy = 0
    for (pair, half) in sorted(wanted):
        row = selected_rows[(pair, half)]
        game = selected_games[(pair, half)]
        headers = game.headers
        board_expected = expected_openings[pair]
        root_fen = board_expected.root().fen()
        opening_uci = [move.uci() for move in board_expected.move_stack]
        start_fen = board_expected.fen()
        if root_fen != expected_opening_root_fen or len(opening_uci) != expected_opening_plies:
            _refuse(f"pair {pair} differs from required opening root/ply contract")
        if (row.get("opening_index") != pair or row.get("a_is_white") is not (half == 0)
                or row.get("loop") not in ("chunked", "rolling")
                or row.get("seed") != expected_settings.get("seed")):
            _refuse(f"pair-half {(pair, half)} has inconsistent schedule/color/loop")
        if (row.get("opening_root_fen") != root_fen or row.get("opening_uci") != opening_uci
                or row.get("opening_fen") != start_fen or row.get("start_fen") != start_fen):
            _refuse(f"pair-half {(pair, half)} JSONL opening history differs")
        try:
            pgn_uci = json.loads(headers["OpeningUCI"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise HistoryRefusal(f"pair-half {(pair, half)} lacks PGN opening UCI") from exc
        if (headers.get("OpeningRootFEN") != root_fen or pgn_uci != opening_uci
                or game.board().fen() != start_fen):
            _refuse(f"pair-half {(pair, half)} PGN opening history differs")
        for tag, expected in expected_pgn_tags.items():
            if headers.get(tag) != expected:
                _refuse(f"pair-half {(pair, half)} PGN tag {tag} differs")
        cand_white = half == 0
        if (headers.get("White") != (candidate_name if cand_white else reference_name)
                or headers.get("Black") != (reference_name if cand_white else candidate_name)
                or headers.get("WhiteSearch") != (candidate_search if cand_white else reference_search)
                or headers.get("BlackSearch") != (reference_search if cand_white else candidate_search)
                or headers.get("EvaluatorHoist") != row.get("eval_hoist")):
            _refuse(f"pair-half {(pair, half)} PGN sides/search/hoist differ")
        result = row.get("result")
        termination = row.get("termination")
        plies = _integer(row.get("plies"), "JSONL plies")
        if (result not in _RESULTS or headers.get("Result") != result
                or headers.get("Termination") != termination
                or headers.get("Plies") != str(plies)
                or row.get("score_candidate") != _score(result, cand_white)):
            _refuse(f"pair-half {(pair, half)} PGN result/termination/plies differ")
        if termination not in ("rules", "syzygy"):
            _refuse(f"pair-half {(pair, half)} is unresolved or max-ply censored")
        replay = chess.Board(root_fen)
        try:
            for uci in opening_uci:
                move = chess.Move.from_uci(uci)
                if move not in replay.legal_moves:
                    _refuse(f"pair-half {(pair, half)} has illegal opening move")
                replay.push(move)
        except ValueError as exc:
            raise HistoryRefusal(f"pair-half {(pair, half)} has invalid opening UCI") from exc
        if replay.fen() != start_fen:
            _refuse(f"pair-half {(pair, half)} opening replay FEN differs")
        moves = list(game.mainline_moves())
        if len(moves) != plies:
            _refuse(f"pair-half {(pair, half)} PGN move count differs")
        for move in moves:
            if replay.outcome(claim_draw=True) is not None or _rule50_result(replay, tablebase, 6) is not None:
                _refuse(f"pair-half {(pair, half)} played after an adjudicable state")
            if move not in replay.legal_moves:
                _refuse(f"pair-half {(pair, half)} has illegal PGN move")
            replay.push(move)
        natural = replay.outcome(claim_draw=True)
        if termination == "rules":
            if natural is None or replay.result(claim_draw=True) != result:
                _refuse(f"pair-half {(pair, half)} natural result differs")
            rules += 1
        else:
            if natural is not None or _rule50_result(replay, tablebase, 6) != result:
                _refuse(f"pair-half {(pair, half)} Syzygy WDL+DTZ result differs")
            syzygy += 1
    return HistoryReceipt(
        jsonl_sha256=hashlib.sha256(jsonl_bytes).hexdigest(),
        pgn_sha256=hashlib.sha256(pgn_bytes).hexdigest(),
        pairs=len(expected_openings), games=len(wanted), rules_games=rules,
        syzygy_games=syzygy, replayed_pairs=replayed,
        pair_ids=tuple(sorted(expected_openings)),
        opening_root_fen=expected_opening_root_fen,
        opening_plies=expected_opening_plies,
        compile_tags=tuple(sorted({row["compile"] for row in selected_rows.values()})),
        hoist_tags=tuple(sorted({row["eval_hoist"] for row in selected_rows.values()})),
    )
