"""Small-bank scalar Stockfish value facts and D-lite main-WDL attachment.

Inputs must be independently authenticated replay winners, complete per-game
history proofs, and selected-neural targets. This module checks their joins;
it does not admit a corpus, launch a bulk label job, or change policy/auxiliaries.
The caller owns the engine and publishes/reads back the resulting files.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chess
import numpy as np

from chess_anti_engine.stockfish.uci import StockfishUCI, _parse_info_fields
from chess_anti_engine.stockfish.wdl import cp_to_wdl, mate_to_effective_cp
from scripts import gen_sf_rooted_corpus as corpus

POLICY_WIDTH = 1858
TARGET_BYTES = 2 * (POLICY_WIDTH + 3)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Hold(ValueError):
    """A missing or inconsistent label must never become a zero-cp fallback."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise Hold(why)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def uid_of(value: Any) -> tuple[str, str, str, int, int]:
    need(type(value) is list and len(value) == 5 and
         all(type(x) is str and bool(x) for x in value[:3]) and
         HEX64.fullmatch(value[0]) is not None and
         all(type(x) is int and x >= 0 for x in value[3:]),
         "source-qualified winner UID")
    return tuple(value)  # type: ignore[return-value]


def _sha_field(value: Any, name: str) -> str:
    need(type(value) is str and HEX64.fullmatch(value) is not None,
         f"{name} SHA-256")
    return value


@dataclass(frozen=True)
class PreparedRow:
    uid: tuple[str, str, str, int, int]
    input_digest: str
    source_proof_sha256: str
    full_history_sha256: str
    pov_white: bool
    rule50: int
    history: corpus.RowHistory

    def identity(self) -> dict[str, Any]:
        return {"uid": list(self.uid), "input_digest": self.input_digest,
                "source_proof_sha256": self.source_proof_sha256,
                "full_history_sha256": self.full_history_sha256,
                "pov_white": self.pov_white, "rule50": self.rule50}


def prepare_row(winner: dict[str, Any], proof: dict[str, Any],
                game: dict[str, Any]) -> PreparedRow:
    """Rebuild the row from a compact authenticated game chain, never a tensor.

    ``row_index`` is the zero-based game-row index. BT4 UIDs count the 16-ply
    opening and Ceres/SF UIDs count rows from the selected game root. The
    source replay must certify the proof; this function independently checks
    the winner/proof join and every move needed for the scalar UCI position.
    """
    uid = uid_of(winner.get("uid"))
    need(uid_of(proof.get("uid")) == uid, "winner/proof UID")
    input_digest = _sha_field(winner.get("input_digest"), "winner input")
    need(proof.get("input_digest") == input_digest,
         "winner/proof input digest")
    source_proof = _sha_field(proof.get("source_proof_sha256"), "game proof")
    need(game.get("source_proof_sha256") == source_proof,
         "game proof identity")
    need(game.get("uid_prefix") == list(uid[:4]), "game/row source identity")
    row_index = proof.get("row_index")
    offset = game.get("uid_ply_offset")
    need(type(row_index) is int and row_index >= 0 and
         type(offset) is int and offset in (0, 16) and
         uid[4] == row_index + offset, "game/row ply identity")
    opening = game.get("opening_uci")
    played = game.get("played_uci")
    need(type(opening) is list and type(played) is list and
         all(type(m) is str for m in opening + played) and
         row_index < len(played) and len(opening) == 16,
         "complete selected-game history")
    root_fen = game.get("root_start_fen")
    need(type(root_fen) is str and bool(root_fen), "history root FEN")
    try:
        board = chess.Board(root_fen)
        need(board.is_valid(), "valid game root")
        for token in opening + played[:row_index]:
            move = chess.Move.from_uci(token)
            need(move in board.legal_moves, "legal recorded history")
            board.push(move)
    except (ValueError, chess.InvalidMoveError) as error:
        raise Hold("malformed recorded history") from error
    need(board.legal_moves.count() > 0 and not board.is_game_over(claim_draw=True),
         "nonterminal scalar row")
    stack = opening + played[:row_index]
    stack_sha = digest(canonical(stack))
    need(proof.get("full_history_sha256") == stack_sha,
         "full history stack SHA")
    pov_white = proof.get("pov_white")
    rule50 = proof.get("rule50")
    need(type(pov_white) is bool and pov_white == bool(board.turn) and
         type(rule50) is int and rule50 == board.halfmove_clock,
         "side-to-move/rule50 orientation")
    if "fen" in proof:
        need(proof["fen"] == board.fen(), "source row FEN")
    context = winner.get("context")
    need(type(context) is list and len(context) == 5 and
         context[0] == stack_sha and context[2] == rule50,
         "winner retained history/rule50 context")
    history = corpus.history_for(board)
    need(history.fen == board.fen(), "search history FEN")
    return PreparedRow(uid, input_digest, source_proof, stack_sha,
                       pov_white, rule50, history)


def prepare_v6_winner(winner: dict[str, Any], game_line: bytes) -> PreparedRow:
    """Adapt the exact tri-source v6 compact history_chain proof line.

    The caller must independently authenticate the v6 replay receipt and
    proof-file SHA before supplying its lines. This binds each label to the
    exact proof line as well as the replayed winner UID/input/context.
    """
    need(type(game_line) is bytes and 0 < len(game_line) <= 2**20,
         "bounded v6 game proof line")
    try:
        report = json.loads(game_line)
    except (UnicodeError, ValueError) as error:
        raise Hold("v6 game proof JSON") from error
    need(type(report) is dict, "v6 game proof object")
    chain = report.get("history_chain")
    need(type(chain) is dict and
         chain.get("schema") == "tri_source_full_game_history_chain_v1" and
         type(chain.get("row_index")) is list and
         0 < len(chain["row_index"]) <= 512,
         "complete bounded v6 history chain")
    uid = uid_of(winner.get("uid"))
    source = winner.get("source")
    need(source in ("BT4-v9", "Ceres-v8", "SF-d6") and
         report.get("game_id") == uid[3] and
         str(report.get("root_id")) == uid[2] and
         report.get("source", source) == source,
         "v6 source/game/root join")
    matching = [entry for entry in chain["row_index"]
                if type(entry) is dict and entry.get("uid") == list(uid)]
    need(len(matching) == 1 and
         len({tuple(uid_of(entry.get("uid"))) for entry in chain["row_index"]})
         == len(chain["row_index"]), "unique v6 row proof")
    entry = matching[0]
    offset = 16 if source == "BT4-v9" else 0
    proof_sha = digest(game_line)
    normalized_game = {
        "uid_prefix": list(uid[:4]), "source_proof_sha256": proof_sha,
        "uid_ply_offset": offset, "root_start_fen": chain.get("root_start_fen"),
        "opening_uci": chain.get("opening_uci"),
        "played_uci": chain.get("played_uci"),
    }
    normalized_row = {
        "uid": entry.get("uid"), "input_digest": entry.get("input_digest"),
        "source_proof_sha256": proof_sha,
        "row_index": uid[4] - offset,
        "full_history_sha256": entry.get("history_stack_sha256"),
        "pov_white": entry.get("pov_white"), "rule50": entry.get("rule50"),
    }
    return prepare_row(winner, normalized_row, normalized_game)


class ScalarSearcher(Protocol):
    def new_game(self) -> None: ...
    def stream(self, history: corpus.RowHistory, *, depth: int,
               multipv: int) -> list[str]: ...


def _score(lines: list[str], depth: int, board: chess.Board) -> dict[str, Any]:
    parsed = corpus.parse_depth_blocks(lines, expected_lines=1)
    block, full = corpus.deepest_block_with_width(parsed.blocks, want=1)
    need(full and block.complete and block.depth == depth and
         len(block.lines) == 1 and parsed.re_emissions_disagreeing == 0,
         "complete unambiguous fixed-depth scalar score")
    for line in lines:
        fields = line.split()
        if not fields or fields[0] != "info" or "upperbound" in fields or "lowerbound" in fields:
            continue
        rank, nodes, found_depth, cp, mate, native, move = _parse_info_fields(fields)
        if found_depth != depth or rank not in (None, 1) or move is None:
            continue
        need((cp is None) != (mate is None) and
             move == block.lines[0].move and nodes == block.lines[0].nodes and
             move in {m.uci() for m in board.legal_moves},
             "raw scalar score/move differs")
        need(native is not None and len(native) == 3 and
             all(type(x) is int and x >= 0 for x in native) and sum(native) == 1000,
             "native UCI WDL unavailable/malformed")
        need(type(nodes) is int and nodes > 0 and
             math.isfinite(block.lines[0].effective_cp),
             "scalar nodes/effective CP unavailable")
        wdl = cp_to_wdl(cp, mate, slope=0.006, draw_width_cp=120.0).tolist()
        need(all(math.isfinite(float(x)) for x in wdl), "finite D calibration")
        return {"cp": cp, "mate": mate, "effective_cp": block.lines[0].effective_cp,
                "native_wdl_permille": list(native), "d_style_wdl": wdl,
                "move": move, "nodes": nodes}
    raise Hold("requested raw scalar score absent")


def label_one(row: PreparedRow, searcher: ScalarSearcher, *, depth: int,
              hash_mb: int = 8) -> dict[str, Any]:
    """One cold-TT, serialized UCI scalar value; owned engine is caller's."""
    need(type(depth) is int and depth in (6, 8, 10) and
         type(hash_mb) is int and hash_mb == 8,
         "fixed depth/Hash8 profile")
    board = chess.Board(row.history.root_fen)
    for token in row.history.uci:
        move = chess.Move.from_uci(token)
        need(move in board.legal_moves, "legal UCI window")
        board.push(move)
    need(board.fen() == row.history.fen and bool(board.turn) == row.pov_white
         and board.halfmove_clock == row.rule50,
         "UCI window orientation/context")
    searcher.new_game()
    lines = searcher.stream(row.history, depth=depth, multipv=1)
    result = {**row.identity(), "depth": depth, "hash_mb": hash_mb,
              "wdl_orientation": "side_to_move", "score": _score(lines, depth, board)}
    return result


def label_bank(rows: list[PreparedRow], *, depth: int, stockfish: Path,
               stockfish_sha256: str, syzygy_path: str,
               max_wall_seconds: int = 3600,
               engine_factory: Any = StockfishUCI,
               searcher_factory: Any = corpus.StaircaseSearcher) -> list[dict[str, Any]]:
    """Owned, bounded small-bank path. One engine, cold TT and readyok per row.

    Full physical work additionally needs authenticated input/receipt publication
    and independent readback. This function never infers history from an input
    tensor and returns no partial success if a score is absent or malformed.
    """
    need(type(depth) is int and depth in (6, 8, 10) and
         0 < len(rows) <= 512 and len({r.uid for r in rows}) == len(rows) and
         type(max_wall_seconds) is int and 0 < max_wall_seconds <= 3600,
         "bounded unique small-bank geometry")
    need(isinstance(stockfish, Path) and stockfish.is_file() and not stockfish.is_symlink() and
         0 < stockfish.stat().st_size <= 256 << 20 and
         _sha_field(stockfish_sha256, "engine binary") == file_digest(stockfish),
         "qualified Stockfish binary bytes")
    parts = syzygy_path.split(":")
    need(len(parts) == 2 and all(Path(p).is_dir() for p in parts),
         "two strict Syzygy directories")
    engine = engine_factory(str(stockfish), multipv=1, hash_mb=8, threads=1,
                            nice=19, syzygy_path=syzygy_path,
                            syzygy_50_move_rule=True, syzygy_probe_limit=6,
                            retain_syzygy_on_new_game=True, read_timeout_s=60)
    try:
        need(engine.syzygy_ready_after_requests is True and
             engine.retain_syzygy_option_sent is True,
             "strict Syzygy/retention UCI request unavailable")
        searcher = searcher_factory(engine=engine,
                                    staircase=corpus.parse_staircase("1:8"),
                                    cp_slope=0.006, cp_draw_width=120.0,
                                    search_timeout_s=15)
        deadline = time.monotonic() + max_wall_seconds
        result = []
        for row in rows:
            need(time.monotonic() < deadline, "cooperative small-bank wall cap")
            result.append(label_one(row, searcher, depth=depth))
        return result
    finally:
        engine.close()


def attach_main_wdl(row: PreparedRow, label: dict[str, Any],
                    selected: dict[str, Any], target: bytes,
                    legal_mask: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    """Replace only selected main WDL bytes; policy and auxiliaries stay neural.

    ``target`` is the independently authenticated selected target row: 1858
    float16 policy entries then 3 float16 side-to-move WDL entries. A caller
    attaches the returned target through the existing packed ``search_wdl``
    path. No ``sf_wdl`` or ``sf_policy_target`` auxiliary is populated.
    """
    need(type(target) is bytes and len(target) == TARGET_BYTES,
         "selected target bytes")
    need(label.get("uid") == list(row.uid) == selected.get("uid") and
         label.get("input_digest") == row.input_digest == selected.get("input_digest") and
         label.get("source_proof_sha256") == row.source_proof_sha256 and
         label.get("full_history_sha256") == row.full_history_sha256 and
         label.get("pov_white") is row.pov_white and
         selected.get("pov_white") is row.pov_white and
         label.get("rule50") == row.rule50 and
         label.get("wdl_orientation") == "side_to_move" and
         label.get("depth") in (6, 8, 10) and label.get("hash_mb") == 8,
         "label/selected winner identity and orientation")
    need(selected.get("target_sha256") == digest(target),
         "selected target byte pin")
    need(type(legal_mask) is np.ndarray and legal_mask.shape == (POLICY_WIDTH,) and
         legal_mask.dtype == np.uint8 and np.isin(legal_mask, (0, 1)).all() and
         bool(legal_mask.any()), "selected legal mask")
    policy = np.frombuffer(target[:POLICY_WIDTH * 2], dtype="<f2")
    chosen = np.frombuffer(target[POLICY_WIDTH * 2:], dtype="<f2")
    need(np.isfinite(policy).all() and np.isfinite(chosen).all() and
         np.all(policy >= 0) and np.all(chosen >= 0) and
         np.all(policy[legal_mask == 0] == 0) and
         abs(float(policy.sum(dtype=np.float32)) - 1) <= .005 and
         abs(float(chosen.sum(dtype=np.float32)) - 1) <= .005,
         "selected policy/value/mask")
    score = label.get("score")
    need(type(score) is dict, "missing scalar score")
    cp, mate = score.get("cp"), score.get("mate")
    need((type(cp) is int and mate is None) or
         (cp is None and type(mate) is int), "missing scalar CP/mate")
    effective = float(cp if cp is not None else mate_to_effective_cp(mate))
    native = score.get("native_wdl_permille")
    board = chess.Board(row.history.fen)
    need(type(score.get("nodes")) is int and score["nodes"] > 0 and
         type(score.get("move")) is str and
         score["move"] in {move.uci() for move in board.legal_moves} and
         type(score.get("effective_cp")) in (int, float) and
         float(score["effective_cp"]) == effective and
         type(native) is list and len(native) == 3 and
         all(type(x) is int and x >= 0 for x in native) and sum(native) == 1000,
         "complete raw scalar score")
    sf = cp_to_wdl(cp, mate, slope=0.006, draw_width_cp=120.0)
    observed = score.get("d_style_wdl")
    need(type(observed) is list and len(observed) == 3 and
         all(type(x) in (int, float) and math.isfinite(x) for x in observed) and
         np.allclose(np.asarray(observed, dtype=np.float32), sf, rtol=0, atol=1e-7),
         "historical D-calibrated scalar value")
    mixed = (sf.astype(np.float32) + 2 * chosen.astype(np.float32)) / 3
    need(np.isfinite(mixed).all() and abs(float(mixed.sum()) - 1) <= .005,
         "finite D-lite main WDL")
    candidate = target[:POLICY_WIDTH * 2] + mixed.astype("<f2").tobytes()
    need(candidate[:POLICY_WIDTH * 2] == target[:POLICY_WIDTH * 2],
         "chosen neural policy unchanged")
    return candidate, {"uid": list(row.uid), "input_digest": row.input_digest,
                       "selected_target_sha256": digest(target),
                       "candidate_target_sha256": digest(candidate),
                       "main_wdl_rule": "(SF_D_calibrated + 2*selected_neural)/3",
                       "policy_bytes_unchanged": True,
                       "sf_auxiliary_targets": False}
