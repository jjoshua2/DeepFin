"""Authenticate original row/TB facts; no targets, tablebase queries or model work."""
from __future__ import annotations

import hashlib
from typing import Any

from scripts import core_finite_learner as core

SCHEMA = "sf_p0_main_policy_eligibility_v1"
MIN_PIECES = 8


def marker_sets(actual: dict[str, Any], verified: dict[str, Any],
                eligible_raw_plies: set[int]) -> tuple[set[int], set[int]]:
    """Compare BOTH archived sets; missing is never interpreted as empty."""
    result = []
    for name in ("applied", "abstained"):
        raw = actual["tablebase_policy_" + name + "_plies"]
        proof = verified["policy_" + name + "_plies"]
        core.require(isinstance(raw, list) and isinstance(proof, list)
                     and all(type(x) is int and x >= 0 for x in raw)
                     and len(set(raw)) == len(raw) and raw == proof,
                     "original strict TB marker list/type/proof disagreement")
        result.append(set(raw))
    applied, abstained = result
    core.require(not applied & abstained and applied | abstained == eligible_raw_plies
                 and verified["policy_all_eligible_raw_records_reprobed"] is True
                 and verified["policy_censored_bypass"] is False
                 and type(verified["policy_eligible_raw_records"]) is int
                 and verified["policy_eligible_raw_records"] == len(eligible_raw_plies),
                 "original strict TB marker completeness/disjointness")
    return applied, abstained


def join_game(game: dict[str, Any], generation: dict[str, Any], rows: list[dict[str, Any]],
              cap: int) -> list[dict[str, Any]]:
    """Legal original history -> accepted carrier P0; inherited outcome, no TB probe."""
    import chess
    from chess_anti_engine.moves import index_to_move_strict, move_to_index_for_encoding
    from scripts import sf_banked_p0_census as audit

    provenance, validation = audit.game_documents(game, generation, cap)
    actual = provenance["actual"]
    _, wrapper = core.read_ref(game["producer_proof"], max_bytes=cap)
    _, outcome = core.read_ref(wrapper["evidence"]["outcome_proof"], max_bytes=cap)
    verified = outcome["verification"]
    core.require(verified["schema"] == "CORE_ACTUALLY_VERIFIED_OUTCOME_V1"
                 and verified["audit_sha256"] == audit.digest(actual), "original strict proof/source context")
    core.require(actual["start"]["is_selfplay"] is True, "original non-selfplay role")
    records = actual["records"]
    core.require(isinstance(records, list)
                 and all(type(r["ply_index"]) is int and r["ply_index"] >= 0
                         and type(r["move_offset"]) is int and r["move_offset"] >= 0
                         and type(r["has_policy"]) is bool and type(r["pov_color"]) is bool
                         and type(r["is_sf_refute_opp"]) is bool for r in records)
                 and len({r["ply_index"] for r in records}) == len(records)
                 and len({r["move_offset"] for r in records}) == len(records), "original duplicate raw record")
    by_offset = {r["move_offset"]: r for r in records}
    board = chess.Board(actual["start"]["root_fen"])
    opening = actual["start"]["opening_moves"]
    for uci in opening:
        board.push_uci(uci)
    core.require(board.fen() == actual["start"]["start_fen"] and board.ply() == actual["start"]["start_ply"],
                 "original opening identity")
    positions, played, eligible = {}, [], set()
    for offset, action in enumerate(actual["played_move_indices"]):
        core.require(type(action) is int, "original action must be an integer")
        rec = by_offset.get(offset)
        if rec is not None:
            core.require(type(rec["ply_index"]) is int and rec["ply_index"] == board.ply()
                         and type(rec["has_policy"]) is bool and rec["pov_color"] is board.turn,
                         "original raw record actual-ply/POV/mask")
            positions[offset] = (board.copy(stack=True), list(played))
            if len(board.piece_map()) <= 6 and not board.castling_rights:
                eligible.add(board.ply())
        move = index_to_move_strict(int(action), board)
        core.require(move in board.legal_moves, "original illegal history action")
        played.append(move.uci())
        board.push(move)
    core.require(len(positions) == len(records) and board.fen() == chess.Board(actual["final_fen"]).fen(),
                 "original record bounds/final history")
    applied, abstained = marker_sets(actual, verified, eligible)
    original_rows = validation["rows_detail"]
    core.require(len(rows) == len(original_rows) == game["rows"], "eligibility complete game rows")
    result = []
    for i, (row, original) in enumerate(zip(rows, original_rows, strict=True)):
        identity = row["input_identity"]
        rec = by_offset.get(identity["move_offset"])
        core.require(rec is not None and rec["ply_index"] == identity["ply"] == original["ply"]
                     and rec["move_offset"] == original["move_offset"] and rec["is_sf_refute_opp"] is False
                     and row["original_row_index"] == i
                     and row["physical_row_id"] == original["physical_row_id"] == game["physical_game_id"] + ":" + str(i),
                     "eligibility original physical row/ply join")
        position, prefix = positions[rec["move_offset"]]
        history = [actual["start"]["root_fen"], list(opening) + prefix]
        core.require(identity["fen"] == position.fen() and identity["STM_white"] is position.turn
                     and identity["halfmove_clock"] == position.halfmove_clock
                     and identity["fullmove_number"] == position.fullmove_number
                     and identity["history_sha256"] == original["history_sha256"] == audit.digest(history)
                     and identity["input_F16_sha256"] == original["stored_input_float16_sha256"],
                     "eligibility P0 fullhistory/input identity")
        legal = bytearray(1858)
        for move in position.legal_moves:
            legal[move_to_index_for_encoding(move, position, policy_encoding="lc0_1858")] = 1
        core.require(hashlib.sha256(legal).hexdigest() == row["legal_mask_sha256"], "eligibility P0 legal identity")
        pieces = len(position.piece_map())
        tb = identity["ply"] in applied | abstained
        core.require(not tb or (pieces <= 6 and not position.castling_rights), "contradictory original TB marker/material")
        result.append({"original_has_policy": rec["has_policy"], "piece_count": pieces,
                       "TB_applied": identity["ply"] in applied, "TB_abstained": identity["ply"] in abstained,
                       "structurally_eligible": rec["has_policy"] and pieces >= MIN_PIECES and not tb})
    return result
