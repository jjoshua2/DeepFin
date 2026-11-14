"""Read-only, fixed-manifest SF P0 evidence census; never issues training labels."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
from typing import Any
import uuid

REPO = Path(__file__).resolve().parents[1]
CP_MISSING = -32768
PAD = (-1, CP_MISSING, 0, -1, -1)
UNKNOWN = "UNKNOWN_NOT_PERSISTED"
# This auditor only decodes the ordinary numeric public Zarr producer layout.
COMPRESSORS = {"blosc", "zstd", "lz4", "gzip", "zlib", "bz2", "lzma"}
SF_FIELDS = {
    "sf_wdl": ("has_sf_wdl", "<f2", (3,)),
    "sf_policy_target": ("has_sf_policy", "<f2", (1858,)),
    "sf_legal_mask": ("has_sf_legal_mask", "|u1", (1858,)),
    "sf_multipv_raw": ("has_sf_multipv_raw", "<i2", (48, 5)),
    "sf_label_meta": ("has_sf_label_meta", "<i4", (6,)),
    "sf_move_index": ("has_sf_move", "<i4", ()),
    "sf_played_move_index": ("has_sf_played_move", "<i4", ()),
    "sf_p0_policy_target": ("has_sf_p0", "<f2", (1858,)),
    "sf_p0_regret": ("has_sf_p0_regret", "<f2", (1858,)),
}


def require(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def stamp(path: Path, *, readonly: bool = True) -> tuple[int, ...]:
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and not path.is_symlink(), "not a regular input: " + str(path))
    require(not readonly or not s.st_mode & 0o222, "writable input: " + str(path))
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def file_sha(path: Path, *, readonly: bool = True) -> str:
    before = stamp(path, readonly=readonly)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    require(stamp(path, readonly=readonly) == before, "input changed during read: " + str(path))
    return h.hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in pairs:
        require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def checked(ref: dict[str, Any], cap: int, *, readonly: bool = True) -> dict[str, Any]:
    require(set(ref) == {"path", "sha256"}, "invalid authenticated reference")
    p = Path(ref["path"])
    require(p.is_absolute() and stamp(p, readonly=readonly)[2] <= cap, "metadata path/cap")
    require(file_sha(p, readonly=readonly) == ref["sha256"], "metadata hash mismatch: " + str(p))
    value = json.loads(p.read_bytes(), object_pairs_hook=unique_object,
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    require(file_sha(p, readonly=readonly) == ref["sha256"], "metadata changed")
    require(isinstance(value, dict), "metadata object required")
    return value


def verify_inventory(shard: dict[str, Any], cap: int) -> dict[str, tuple[int, ...]]:
    root = Path(shard["path"])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink()
            and not root.stat().st_mode & 0o222, "readonly ordinary directory shard required")
    rows = shard["files"]
    require(isinstance(rows, list) and 0 < len(rows) <= cap, "inventory file count")
    names = [r["path"] for r in rows]
    require(names == sorted(set(names)), "duplicate/unsorted inventory")
    entries = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in entries), "symlink in shard")
    actual = sorted(str(p.relative_to(root)) for p in entries if not p.is_dir())
    require(actual == names, "inventory omitted or added file")
    result = {}
    for row in rows:
        name = row["path"]
        require(set(row) == {"path", "bytes", "sha256"} and isinstance(name, str)
                and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                and "\\" not in name, "unsafe inventory member")
        path = root / name
        require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent), "symlink inventory member")
        before = stamp(path)
        require(before[2] == row["bytes"] and file_sha(path) == row["sha256"], "shard inventory hash/size differs")
        result[name] = before
    return result


def open_arrays(shard: dict[str, Any], rows: int, decoded_cap: int) -> Any:
    """Validate codecs and padded allocation bounds BEFORE creating Zarr proxies."""
    import numpy as np
    import zarr

    root = Path(shard["path"])
    total = 0
    metadata = list(root.rglob(".zarray"))
    require(bool(metadata), "no Zarr arrays")
    for path in metadata:
        require(path.parent.parent == root and path.stat().st_size <= 65536, "nested/oversized array metadata")
        value = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
        require("dtype" in value, "missing dtype")
        dtype = np.dtype(value["dtype"])
        require(dtype.kind in "biuf" and not dtype.hasobject and not value.get("filters"), "unsafe dtype/filter")
        compressor = value.get("compressor")
        require(compressor is None or compressor.get("id") in COMPRESSORS, "unsafe compressor")
        shape, chunks = value["shape"], value["chunks"]
        require(isinstance(shape, list) and 1 <= len(shape) <= 4 and shape[0] == rows
                and all(type(n) is int and n > 0 for n in shape), "array shape/row count")
        require(len(chunks) == len(shape) and all(type(n) is int and n > 0 for n in chunks), "chunk shape")
        total += math.prod(((n + c - 1) // c) * c for n, c in zip(shape, chunks, strict=True)) * dtype.itemsize
        require(total <= decoded_cap, "per-game decoded allocation cap")
    return zarr.open_group(str(root), mode="r")


def field(arrays: Any, name: str, i: int) -> tuple[Any, list[str]]:
    """Absent/invalid label fields are recorded, never silently imputed."""
    import numpy as np

    flag, dtype, tail = SF_FIELDS[name]
    if flag not in arrays:
        return None, [name + ":missing_flag"]
    f = arrays[flag]
    if f.dtype != np.dtype("u1") or f.shape != (arrays["x"].shape[0],) or int(f[i]) not in (0, 1):
        return None, [name + ":invalid_flag"]
    if int(f[i]) == 0:
        return None, [name + ":absent"]
    if name not in arrays:
        return None, [name + ":flag_without_payload"]
    a = arrays[name]
    if a.dtype != np.dtype(dtype) or a.shape != (arrays["x"].shape[0], *tail):
        return None, [name + ":storage_dtype_or_shape"]
    value = np.asarray(a[i])
    if not np.isfinite(value).all():
        return None, [name + ":nonfinite"]
    return value, []


def native_wdl(w: int, d: int) -> list[int] | None:
    if (w, d) == (-1, -1):
        return None
    require(0 <= w <= 1000 and 0 <= d <= 1000 and w + d <= 1000, "malformed native WDL permille")
    return [w, d, 1000 - w - d]


def sparse_evidence(raw: Any, legal: set[int]) -> dict[str, Any]:
    rows, reasons = [], []
    padded = False
    seen: set[int] = set()
    for row in raw.tolist():
        move, cp, mate, w, d = map(int, row)
        if move == -1:
            if tuple(row) != PAD:
                reasons.append("multipv:malformed_padding")
            padded = True
            continue
        if padded:
            reasons.append("multipv:nonpadding_after_padding")
        if move in seen:
            reasons.append("multipv:duplicate_move")
        seen.add(move)
        if move not in legal:
            reasons.append("multipv:illegal_move")
        if cp != CP_MISSING and not -32000 <= cp <= 32000:
            reasons.append("multipv:invalid_cp")
        if not -127 <= mate <= 127:
            reasons.append("multipv:invalid_mate")
        try:
            native = native_wdl(w, d)
        except ValueError:
            native = None
            reasons.append("multipv:invalid_native_wdl")
        rows.append({"move": move, "cp": None if cp == CP_MISSING else cp, "mate": None if mate == 0 else mate,
                     "native_query_STM_WDL_permille": native,
                     "cp_at_clip_boundary": abs(cp) == 32000, "mate_at_clip_boundary": abs(mate) == 127})
    return {"candidates": rows, "support": len(seen & legal), "legal_count": len(legal),
            "full_legal_support": bool(legal) and seen == legal, "reasons": sorted(set(reasons)),
            "bounds": UNKNOWN, "full_PV": UNKNOWN, "per_candidate_effort": UNKNOWN,
            "returned_original_K": UNKNOWN, "truncation_beyond_48": UNKNOWN}


def probability(value: Any, legal: set[int]) -> bool:
    import numpy as np

    return bool((value >= 0).all() and abs(float(value.astype(np.float64).sum()) - 1) <= 0.002
                and not np.any(value[[j for j in range(len(value)) if j not in legal]]))


def label_evidence(arrays: Any, i: int, legal: set[int], game: dict[str, Any]) -> dict[str, Any]:
    """P1 data attached to THIS retained record, in explicitly distinct POVs."""
    import numpy as np
    from chess_anti_engine.stockfish.wdl import cp_to_wdl

    origin = sys.modules[cp_to_wdl.__module__].__file__
    if not isinstance(origin, str):
        raise ValueError("missing WDL source origin")
    require(Path(origin).resolve() == REPO / "chess_anti_engine/stockfish/wdl.py", "foreign WDL source")
    values, reasons = {}, []
    for name in ("sf_wdl", "sf_policy_target", "sf_legal_mask", "sf_multipv_raw", "sf_label_meta", "sf_move_index", "sf_played_move_index"):
        values[name], why = field(arrays, name, i)
        reasons.extend(why)
    policy, mask = values["sf_policy_target"], values["sf_legal_mask"]
    if mask is not None and (not np.isin(mask, [0, 1]).all() or set(np.flatnonzero(mask)) != legal):
        reasons.append("sf_legal_mask:query_legal_mismatch")
    if policy is not None and not probability(policy, legal):
        reasons.append("sf_policy_target:invalid_query_distribution")
    sparse = sparse_evidence(values["sf_multipv_raw"], legal) if values["sf_multipv_raw"] is not None else None
    if sparse is not None:
        reasons.extend(sparse["reasons"])
        if not sparse["candidates"] and policy is not None:
            reasons.append("policy_without_legal_PV:fallback_possible_not_proven")
    meta = values["sf_label_meta"]
    native, comparison = None, "UNAVAILABLE"
    if meta is not None:
        nodes, depth, cp, mate, w, d = map(int, meta)
        if nodes < -1 or depth < -1 or (cp != CP_MISSING and abs(cp) > 32000) or abs(mate) > 127:
            reasons.append("sf_label_meta:invalid_scalar")
        try:
            native = native_wdl(w, d)
        except ValueError:
            reasons.append("sf_label_meta:invalid_native_wdl")
        if sparse and sparse["candidates"]:
            best = sparse["candidates"][0]
            if best["cp"] != (None if cp == CP_MISSING else cp) or best["mate"] != (None if mate == 0 else mate):
                reasons.append("metadata_vs_PV1:score_disagreement")
            if best["native_query_STM_WDL_permille"] != native:
                reasons.append("metadata_vs_PV1:native_WDL_disagreement")
    wdl = values["sf_wdl"]
    if wdl is not None:
        if not (bool((wdl >= 0).all()) and abs(float(wdl.astype(np.float64).sum()) - 1) <= 0.002):
            reasons.append("sf_wdl:invalid_distribution")
        elif meta is not None:
            _, _, cp, mate, _, _ = map(int, meta)
            if game["sf_wdl_use_cp_logistic"] and (cp != CP_MISSING or mate != 0):
                comparison = "UNKNOWN_CLIPPED_SOURCE_SCORE" if abs(cp) == 32000 or abs(mate) == 127 else "CP_LOGISTIC_THEN_POV_FLIP_F16"
                if comparison != "UNKNOWN_CLIPPED_SOURCE_SCORE":
                    expected = cp_to_wdl(None if cp == CP_MISSING else cp, None if mate == 0 else mate,
                                         slope=game["sf_wdl_cp_slope"], draw_width_cp=game["sf_wdl_cp_draw_width"])[::-1]
                    if not np.array_equal(expected.astype(np.float16), wdl):
                        reasons.append("sf_wdl:cp_logistic_flip_inconsistent")
            elif native is not None:
                comparison = "NATIVE_PERMILLE_QUANTIZATION_AND_POV_FLIP_F16"
                # Native W/D were rounded to permille and the vector to F16.
                # W/D round by <=.0005 each, reconstructed L by <=.001;
                # F16 rounding adds <=.000245 per component. This is a
                # representation bound, not a corpus acceptance threshold.
                if not np.allclose(np.asarray(native[::-1]) / 1000, wdl, atol=0.0015, rtol=0):
                    reasons.append("sf_wdl:native_quantized_flip_inconsistent")
    moves = {}
    for name in ("sf_move_index", "sf_played_move_index"):
        v = values[name]
        moves[name] = None if v is None else int(v)
        if v is not None and int(v) not in legal:
            reasons.append(name + ":illegal_query_move")
    if (sparse is not None and moves["sf_move_index"] is not None
            and moves["sf_move_index"] not in {r["move"] for r in sparse["candidates"]}):
        reasons.append("bestmove_not_in_retained_PV:partial_or_fallback_UNKNOWN")
    invalid = [r for r in reasons if any(t in r for t in ("invalid", "mismatch", "malformed", "duplicate", "illegal", "inconsistent", "disagreement", "flag_without", "storage_dtype", "nonfinite", "nonpadding"))]
    return {"stored_sf_wdl_record_POV": None if wdl is None else wdl.tolist(),
            "next_P0_STM_vector_from_one_POV_flip": None if wdl is None else wdl[::-1].tolist(),
            "WDL_relation": comparison, "native_query_STM_WDL_permille": native,
            "record_meta_int32": None if meta is None else meta.tolist(), "multipv": sparse, **moves,
            "policy_present": policy is not None, "query_legal_mask_present": mask is not None,
            "source_reconstructed_context_only": True, "wire_query_history": UNKNOWN,
            "bound_flags": UNKNOWN, "actual_requested_nodes": UNKNOWN, "escalation": UNKNOWN,
            "fallback_provenance": UNKNOWN, "reasons": sorted(set(reasons)), "invalid_reasons": sorted(set(invalid))}


def explicit_p0(arrays: Any, i: int, legal: set[int], raw_predecessor: bool,
                game: dict[str, Any]) -> dict[str, Any]:
    policy, a = field(arrays, "sf_p0_policy_target", i)
    regret, b = field(arrays, "sf_p0_regret", i)
    reasons = a + b
    if (policy is not None or regret is not None) and not raw_predecessor:
        reasons.append("explicit_P0:no_actual_ply_raw_predecessor")
    if policy is not None and not probability(policy, legal):
        reasons.append("explicit_P0:invalid_policy")
    if regret is not None and not bool(((regret >= 0) & (regret <= 1)).all()):
        reasons.append("explicit_P0:invalid_regret")
    if policy is not None and not game["record_sf_p0_policy"]:
        reasons.append("explicit_P0:disabled_source_policy_flag")
    if regret is not None and not game["record_sf_p0_regret"]:
        reasons.append("explicit_P0:disabled_source_regret_flag")
    return {"policy_present": policy is not None, "regret_present": regret is not None,
            "policy_F16_sha256": None if policy is None else hashlib.sha256(policy.tobytes()).hexdigest(),
            "regret_F16_sha256": None if regret is None else hashlib.sha256(regret.tobytes()).hexdigest(),
            "regret_is_probability": False, "native_P0_WDL_supplied_by_these_fields": False,
            "source_shift_before_thinning": True,
            "regret_transform_reconstructed": False, "reasons": reasons}


def game_documents(item: dict[str, Any], generation: dict[str, Any], cap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = checked(item["producer_proof"], cap)
    serving = checked(generation["actor_serving"], cap)
    state = serving["pre_state_sha256"]
    require(serving["actor_sha256"] == generation["actor_sha256"] and len(state) == 64
            and state == serving["post_state_sha256"] == serving["checkpoint_model_state_sha256"], "accepted serving identity mismatch")
    require(wrapper["status"] == "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", "unqualified game wrapper")
    common = {k: generation[k] for k in ("run_uuid", "source_revision", "generation", "actor_sha256")}
    common.update({k: item[k] for k in ("physical_game_id", "game_id", "rows", "bank")})
    common["served_state_sha256"] = state
    require(common["bank"] == "train" and item["rows"] > 0, "non-TRAIN/empty game")
    require(all(wrapper.get(k) == v and type(wrapper.get(k)) is type(v) for k, v in common.items()), "wrapper identity mismatch")
    require(wrapper["resolved"] is True and wrapper["full_history_verified"] is True
            and wrapper["raw_targets_verified"] is True, "unqualified original data")
    require(digest(item["shard"]["files"]) == wrapper["shard_inventory_sha256"], "shard/wrapper inventory join")
    docs = {k: checked(v, cap) for k, v in wrapper["evidence"].items()}
    require({"outcome_proof", "provenance", "validation"} <= docs.keys(), "original evidence absent")
    for doc in docs.values():
        require(all(doc.get(k) == v and type(doc.get(k)) is type(v) for k, v in common.items()), "evidence identity mismatch")
    original, provenance, validation = (docs[k] for k in ("outcome_proof", "provenance", "validation"))
    require(original["schema"] == "CORE_PRODUCING_OUTCOME_PROOF_V1"
            and digest(original["verification"]) == original["proof_sha256"] == validation["verified_outcome_sha256"], "original proof digest mismatch")
    require(original["verification"]["resolved"] is True and original["verification"]["retained_rows"] == item["rows"]
            and original["verification"]["audit_sha256"] == digest(provenance["actual"]), "proof/provenance mismatch")
    expected_class = "STRICT_TB_ADJUDICATED" if item["label_class"] == "TB_ADJUDICATED" else "NATURAL"
    require(item["label_class"] in ("NATURAL", "TB_ADJUDICATED")
            and original["scientific_class"] == original["verification"]["scientific_class"] == expected_class, "outcome class mismatch")
    require(validation["full_history_verified"] is True and validation["usable_shard"] is True
            and validation["raw_targets_verified"] is True and len(validation["rows_detail"]) == item["rows"], "invalid row proof")
    require(original["verification"]["physical_game_id"] == item["physical_game_id"]
            and original["verification"]["game_id"] == item["game_id"]
            and original["verification"]["source_revision"] == generation["source_revision"]
            and original["verification"]["actor_sha256"] == generation["actor_sha256"], "proof payload identity mismatch")
    return provenance, validation


def inspect_game(item: dict[str, Any], generation: dict[str, Any], producer: dict[str, Any], limits: dict[str, int],
                 *, structure: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    import chess
    import numpy as np
    from chess_anti_engine.moves import index_to_move_strict, move_to_index_for_encoding

    origin = sys.modules[index_to_move_strict.__module__].__file__
    if not isinstance(origin, str):
        raise ValueError("missing move mapping origin")
    require(Path(origin).resolve() == REPO / "chess_anti_engine/moves/encode.py", "foreign move mapping source")
    provenance, validation = game_documents(item, generation, limits["metadata_bytes"])
    actual = provenance["actual"]
    require(provenance["planned"]["physical_game_id"] == item["physical_game_id"]
            and actual["game_id"] == item["game_id"] and actual["start"]["is_selfplay"] is True, "planned physical game/selfplay mismatch")
    records = {r["ply_index"]: r for r in actual["records"]}
    require(len(records) == len(actual["records"]), "duplicate actual ply")
    by_offset = {r["move_offset"]: r for r in actual["records"]}
    require(len(by_offset) == len(records) and all(type(k) is int and 0 <= k < len(actual["played_move_indices"])
                                                for k in by_offset), "duplicate/outside raw record offset")
    if structure is not None:
        structure.update(raw_record_count=len(records), raw_records=actual["records"],
                         raw_policy_record_count=sum(r["has_policy"] for r in records.values()),
                         played_plies=len(actual["played_move_indices"]),
                         provenance=checked(item["producer_proof"], limits["metadata_bytes"])["evidence"]["provenance"])
    rows = validation["rows_detail"]
    require(len({r["ply"] for r in rows}) == len(rows) and len({r["move_offset"] for r in rows}) == len(rows), "duplicate retained row")
    before = verify_inventory(item["shard"], limits["inventory_files"])
    group = open_arrays(item["shard"], item["rows"], limits["game_decoded_bytes"])
    attrs = group.attrs
    require(attrs.get("policy_encoding") == "lc0_1858" and attrs.get("input_history_encoding") == "lc0_root_legacy_meta"
            and attrs.get("history_rep_fix") is True, "shard encoding/rep-fix identity")
    n = item["rows"]
    required = {"x": ("<f2", (n, 175, 8, 8)), "legal_mask": ("|u1", (n, 1858)),
                "game_id": ("<i8", (n,)), "ply_index": ("<i4", (n,))}
    identity_flags = ("has_game_id", "has_ply_index", "has_policy", "has_legal_mask", "is_network_turn",
                      "has_is_network_turn", "is_selfplay", "has_is_selfplay")
    needed = {*required, *identity_flags, *SF_FIELDS, *(spec[0] for spec in SF_FIELDS.values())}
    # Bounded one-game materialization prevents re-decoding whole chunks for
    # every row. No arrays survive into the following game's iteration.
    arrays = {name: np.asarray(group[name][:]) for name in needed if name in group}
    for key, (dtype, shape) in required.items():
        require(key in arrays and arrays[key].dtype == np.dtype(dtype) and arrays[key].shape == shape, "identity array schema: " + key)
    for key in identity_flags:
        require(key in arrays and arrays[key].dtype == np.dtype("u1") and arrays[key].shape == (n,)
                and bool((arrays[key] == 1).all()), "identity/NET/selfplay flag: " + key)
    wanted = {r["move_offset"]: i for i, r in enumerate(rows)}
    board = chess.Board(actual["start"]["root_fen"])
    for uci in actual["start"]["opening_moves"]:
        board.push_uci(uci)
    require(board.fen() == chess.Board(actual["start"]["start_fen"]).fen()
            and board.ply() == actual["start"]["start_ply"], "opening full-history mismatch")
    played, contexts, queries = [], {}, {}
    for offset, action in enumerate(actual["played_move_indices"]):
        raw_record = by_offset.get(offset)
        if raw_record is not None:
            require(raw_record["ply_index"] == board.ply() and raw_record["pov_color"] is board.turn,
                    "structural raw-record ply/POV mismatch")
        move = index_to_move_strict(action, board)
        require(move in board.legal_moves, "illegal full-history action")
        if offset in wanted:
            i = wanted[offset]
            row = rows[i]
            rec = records.get(row["ply"])
            require(rec is not None and rec["move_offset"] == offset and rec["pov_color"] is board.turn
                    and rec["has_policy"] is True and rec["is_sf_refute_opp"] is False
                    and row["ply"] == board.ply() == int(arrays["ply_index"][i])
                    and int(arrays["game_id"][i]) == item["game_id"], "retained ply/POV/game join")
            require(row["physical_row_id"] == item["physical_game_id"] + ":" + str(i), "physical row identity")
            history = [actual["start"]["root_fen"], list(actual["start"]["opening_moves"]) + played]
            require(digest(history) == row["history_sha256"], "history fingerprint mismatch")
            x = np.asarray(arrays["x"][i])
            require(bool(np.isfinite(x).all()) and hashlib.sha256(x.astype("<f2").tobytes()).hexdigest()
                    == row["stored_input_float16_sha256"], "stored input/history identity mismatch")
            legal = {move_to_index_for_encoding(m, board, policy_encoding="lc0_1858") for m in board.legal_moves}
            require(np.array_equal(np.asarray(arrays["legal_mask"][i]), np.asarray([int(j in legal) for j in range(1858)], dtype=np.uint8)), "P0 legal mask mismatch")
            contexts[i] = {"ply": board.ply(), "move_offset": offset, "STM_white": board.turn,
                           "halfmove_clock": board.halfmove_clock, "fullmove_number": board.fullmove_number,
                           "fen": board.fen(), "history_sha256": row["history_sha256"], "legal": legal,
                           "input_F16_sha256": row["stored_input_float16_sha256"]}
        played.append(move.uci())
        board.push(move)
        if offset in wanted:
            i = wanted[offset]
            next_move = (index_to_move_strict(actual["played_move_indices"][offset + 1], board)
                         if offset + 1 < len(actual["played_move_indices"]) else None)
            queries[i] = {"ply": board.ply(), "STM_white": board.turn, "fen": board.fen(),
                          "actual_next_move_index": None if next_move is None else move_to_index_for_encoding(next_move, board),
                          "legal_count": board.legal_moves.count(),
                          "history_sha256": digest([actual["start"]["root_fen"], list(actual["start"]["opening_moves"]) + played]),
                          "legal": {move_to_index_for_encoding(m, board, policy_encoding="lc0_1858") for m in board.legal_moves}}
    require(len(contexts) == n and board.fen() == chess.Board(actual["final_fen"]).fen(), "final board/retained offset mismatch")
    labels = {i: label_evidence(arrays, i, queries[i]["legal"], producer["configs"]["game"]) for i in range(n)}
    by_ply = {r["ply"]: i for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        context = contexts[i]
        prev = by_ply.get(row["ply"] - 1)
        raw_prev = records.get(row["ply"] - 1)
        explicit = explicit_p0(arrays, i, context["legal"], raw_prev is not None and raw_prev["has_policy"],
                               producer["configs"]["game"])
        if prev is None:
            join = {"status": "UNAVAILABLE", "reason": "no_retained_actual_ply_predecessor", "predecessor_row": None}
        else:
            require(queries[prev]["history_sha256"] == context["history_sha256"]
                    and queries[prev]["fen"] == context["fen"] and queries[prev]["STM_white"] is context["STM_white"], "P1-to-P0 context join mismatch")
            label = labels[prev]
            any_payload = label["policy_present"] or label["stored_sf_wdl_record_POV"] is not None or label["multipv"] is not None
            status = "INVALID_REDUCED_LABEL" if label["invalid_reasons"] else (
                "SOURCE_CONTEXT_REDUCED_LABEL" if any_payload else "SOURCE_CONTEXT_NO_REDUCED_LABEL")
            join = {"status": status,
                    "predecessor_row": rows[prev]["physical_row_id"], "native_value_available": label["native_query_STM_WDL_permille"] is not None,
                    "stored_value_available": label["stored_sf_wdl_record_POV"] is not None,
                    "policy_available": label["policy_present"], "wire_authenticated": False}
        best = labels[i]["sf_move_index"]
        labels[i]["stored_best_equals_actual_next_move"] = (None if best is None or queries[i]["actual_next_move_index"] is None
                                                           else best == queries[i]["actual_next_move_index"])
        yield {"physical_game_id": item["physical_game_id"], "game_id": item["game_id"],
               "physical_row_id": row["physical_row_id"], "input_identity": {k: v for k, v in context.items() if k != "legal"},
               "P0_legal_count": len(context["legal"]), "raw_actual_ply_predecessor": raw_prev is not None,
               "explicit_stored_P0": explicit, "retained_predecessor_P1_join": join,
               "own_record_P1": labels[i], "own_P1_query_context": {k: v for k, v in queries[i].items() if k != "legal"}}
    require(before == verify_inventory(item["shard"], limits["inventory_files"]), "shard custody changed during audit")


def source_guard(source: dict[str, Any]) -> None:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, env=env, text=True).strip() == source["revision"], "auditor source revision")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, env=env, text=True), "dirty auditor source")
    required = {"scripts/sf_banked_p0_census.py", "chess_anti_engine/moves/encode.py",
                "chess_anti_engine/stockfish/wdl.py"}
    require(required <= source["files"].keys(), "missing consumed source file pins")
    for name, expected in source["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and file_sha(REPO / name, readonly=False) == expected, "auditor source file changed")


def runtime_guard(runtime: dict[str, Any]) -> None:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2", "CPU-only visibility/threads")
    require(sorted(os.sched_getaffinity(0)) == runtime["cpu_affinity"] and len(runtime["cpu_affinity"]) == 2, "CPU affinity")
    require(0 < runtime["inner_seconds"] < runtime["owner_seconds"] <= 900 and __debug__, "finite runtime cap")
    parent = Path("/proc", str(os.getppid()), "cmdline").read_bytes().decode().split("\0")
    require(file_sha(Path(runtime["owner"]["path"]), readonly=False) == runtime["owner"]["sha256"]
            and runtime["owner"]["path"] in parent and parent.count("--resource") == parent.count("--seconds") == 1
            and parent[parent.index("--resource") + 1] == "cpu"
            and parent[parent.index("--seconds") + 1] == str(runtime["owner_seconds"]), "not declared direct CPU owner")
    require(not any(k == "torch" or k.startswith(("torch.", "onnxruntime", "chess_anti_engine.train")) for k in sys.modules), "forbidden runtime import")


def generation_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = protocol["limits"]["metadata_bytes"]
    generation = checked(protocol["generation_manifest"], cap)
    expected = protocol["expected"]
    require(generation["schema"] == "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1"
            and all(generation[k] == expected[k] for k in ("run_uuid", "source_revision", "generation")), "generation identity/source")
    games = generation["games"]
    require(len(games) == expected["physical_games"] and sum(g["rows"] for g in games) == expected["rows"]
            and len({g["physical_game_id"] for g in games}) == len(games)
            and len({g["game_id"] for g in games}) == len(games), "fixed population counts/duplicate games")
    require(all(g["bank"] == "train" and g["rows"] > 0 for g in games), "HOLD/nontraining population")
    producer = checked(protocol["producer_protocol"], cap)
    require(producer["source"]["revision"] == expected["source_revision"] and producer["run_uuid"] == expected["run_uuid"]
            and producer["generation"] == expected["generation"] and producer["actor"]["sha256"] == generation["actor_sha256"], "producer/generation mismatch")
    require(producer["configs"]["game"]["policy_encoding"] == "lc0_1858"
            and producer["configs"]["game"]["history_rep_fix"] is True
            and producer["configs"]["game"]["selfplay_fraction"] == 1.0, "unsupported producer encoding/role")
    owner = protocol["producer_owner"]
    terminal = checked(owner["terminal"], cap, readonly=False)
    require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
            and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
            and terminal["command"] == owner["command"] and terminal["supervisor_sha256"] == owner["owner_sha256"] == producer["owner"]["sha256"]
            and terminal["seconds"] == producer["owner"]["seconds"] and terminal["resource"] == "gpu", "producer not closed/authenticated")
    require(bool(terminal["known_processes"]), "missing producer births")
    for pid, birth in terminal["known_processes"].items():
        path = Path("/proc", str(pid), "stat")
        require(not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != birth, "producer birth still live")
    command = owner["command"]
    require(command[-8:-4] == ["--protocol", protocol["producer_protocol"]["path"], "--protocol-sha256", protocol["producer_protocol"]["sha256"]], "closed command producer mismatch")
    require(command[-4] == "--admission" and command[-2] == "--admission-sha256", "closed command admission missing")
    admission = checked({"path": command[-3], "sha256": command[-1]}, cap)
    require(admission["status"] == "ROOT_ADMITS_CORE_GENERATION_V1"
            and admission["protocol_sha256"] == protocol["producer_protocol"]["sha256"]
            and admission["actor_sha256"] == generation["actor_sha256"], "closed producer identity admission mismatch")
    return generation, producer


def count_row(counts: Counter[str], row: dict[str, Any]) -> None:
    label = row["own_record_P1"]
    counts["rows"] += 1
    counts["join:" + row["retained_predecessor_P1_join"]["status"]] += 1
    counts["WDL_relation:" + label["WDL_relation"]] += 1
    for key in ("policy_present", "query_legal_mask_present"):
        counts[key + ":" + str(label[key])] += 1
    counts["native_meta_WDL:" + str(label["native_query_STM_WDL_permille"] is not None)] += 1
    counts["raw_predecessor:" + str(row["raw_actual_ply_predecessor"])] += 1
    for key in ("policy_present", "regret_present"):
        counts["explicit_P0:" + key + ":" + str(row["explicit_stored_P0"][key])] += 1
    for reason in label["reasons"]:
        counts["own_P1:" + reason] += 1
    for reason in row["explicit_stored_P0"]["reasons"]:
        counts["explicit_P0:" + reason] += 1
    if label["multipv"] is not None:
        sparse = label["multipv"]
        counts["multipv_support:" + str(sparse["support"])] += 1
        counts["query_legal_count:" + str(sparse["legal_count"])] += 1
        counts["multipv_full_support:" + str(sparse["full_legal_support"])] += 1
        for pv in sparse["candidates"]:
            counts["PV_native_WDL:" + str(pv["native_query_STM_WDL_permille"] is not None)] += 1
            counts["PV_cp_boundary:" + str(pv["cp_at_clip_boundary"])] += 1
            counts["PV_mate_boundary:" + str(pv["mate_at_clip_boundary"])] += 1
    if label["record_meta_int32"] is not None:
        meta = label["record_meta_int32"]
        counts["observed_record_nodes:" + str(meta[0])] += 1
        counts["observed_record_depth:" + str(meta[1])] += 1
        counts["meta_cp_boundary:" + str(abs(meta[2]) == 32000)] += 1
        counts["meta_mate_boundary:" + str(abs(meta[3]) == 127)] += 1
    counts["stored_best_equals_actual_next:" + str(label["stored_best_equals_actual_next_move"])] += 1
    counts["actual_ply:" + str(row["input_identity"]["ply"])] += 1
    counts["no_actual_next_played_move:" + str(row["own_P1_query_context"]["actual_next_move_index"] is None)] += 1
    counts["all_query_legal_counts:" + str(row["own_P1_query_context"]["legal_count"])] += 1


def census(protocol: dict[str, Any], *, final_guard: Callable[[], None] | None = None,
           protocol_ref: dict[str, str] | None = None) -> dict[str, Any]:
    require(protocol["schema"] == "SF_BANKED_P0_CENSUS_V1" and str(uuid.UUID(protocol["audit_uuid"])) == protocol["audit_uuid"], "invalid census protocol")
    require(set(protocol["limits"]) == {"metadata_bytes", "game_decoded_bytes", "inventory_files", "report_bytes"}
            and all(type(v) is int and v > 0 for v in protocol["limits"].values()), "positive explicit census limits required")
    generation, producer = generation_inputs(protocol)
    output = Path(protocol["output_root"])
    require(output.is_absolute() and output == output.resolve() and not output.exists()
            and not output.is_relative_to(REPO), "exclusive external output required")
    protected = [Path(producer["output_root"]), Path(protocol["generation_manifest"]["path"]).parent,
                 Path(protocol["producer_protocol"]["path"]).parent,
                 Path(protocol["producer_owner"]["terminal"]["path"]).parent]
    require(all(not output.is_relative_to(p.resolve()) and not p.resolve().is_relative_to(output)
                for p in protected), "output overlaps input")
    output.mkdir(parents=False)
    journal = output / "rows.jsonl"
    counts: Counter[str] = Counter()
    games = []
    n = 0
    try:
        with journal.open("x") as stream:
            for item in generation["games"]:
                before = n
                game_counts: Counter[str] = Counter()
                structure: dict[str, Any] = {}
                for row in inspect_game(item, generation, producer, protocol["limits"], structure=structure):
                    line = canonical(row).decode() + "\n"
                    stream.write(line)
                    require(stream.tell() <= protocol["limits"]["report_bytes"], "report byte cap")
                    n += 1
                    count_row(counts, row)
                    count_row(game_counts, row)
                require(n - before == item["rows"], "game output denominator mismatch")
                games.append({"physical_game_id": item["physical_game_id"], "rows": n - before,
                              "structure": structure,
                              "overlapping_reason_and_distribution_counts": dict(sorted(game_counts.items()))})
                stream.flush()
                os.fsync(stream.fileno())
        require(n == protocol["expected"]["rows"], "population output denominator mismatch")
        if final_guard is not None:
            final_guard()
        report = {"status": "COMPLETE_SF_BANKED_P0_EVIDENCE_CENSUS_DATA_ONLY", "audit_uuid": protocol["audit_uuid"],
                  "protocol": protocol_ref, "protocol_canonical_sha256": digest(protocol),
                  "auditor_source": protocol.get("auditor_source"), "producer_protocol": protocol["producer_protocol"],
                  "declared_stockfish_config_not_actual_effort": producer.get("stockfish"),
                  "generation_manifest": protocol["generation_manifest"], "games": games, "rows": n,
                  "physical_games": len(games), "overlapping_reason_counts": dict(sorted(counts.items())),
                  "per_row_evidence": {"path": str(journal), "sha256": file_sha(journal, readonly=False)},
                  "query_wire_bounds_effort": UNKNOWN, "training_admitted": False, "gain_claim": False,
                  "limitations": ["F16 input hashes join existing accepted producer encoding proofs; no fresh encoding/F32 equivalence.",
                                  "Reduced-score source-context evidence is not wire-authenticated query identity.",
                                  "Counts are descriptive; no minimum coverage, target transform or training recipe selected."]}
        with (output / "readout.json").open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        for path in output.iterdir():
            path.chmod(0o444)
        fd = os.open(output, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        output.chmod(0o555)
        return report
    except BaseException as exc:
        failure = output.with_name(output.name + ".failed.json")
        with failure.open("x") as f:
            json.dump({"status": "FAILED_SF_BANKED_P0_CENSUS_NO_ADMISSION", "rows_written": n,
                       "error": f"{type(exc).__name__}: {exc}"}, f, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    protocol_ref = {"path": str(args.protocol), "sha256": args.protocol_sha256}
    protocol = checked(protocol_ref, 4 << 20)
    source_guard(protocol["auditor_source"])
    runtime_guard(protocol["runtime"])
    sys.path.insert(0, str(REPO))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed CPU census deadline")))
    signal.alarm(protocol["runtime"]["inner_seconds"])
    try:
        def final_guard() -> None:
            require(checked(protocol_ref, 4 << 20) == protocol, "census protocol changed")
            source_guard(protocol["auditor_source"])
            runtime_guard(protocol["runtime"])

        census(protocol, final_guard=final_guard, protocol_ref=protocol_ref)
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
