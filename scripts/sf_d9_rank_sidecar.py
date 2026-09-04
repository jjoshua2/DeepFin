#!/usr/bin/env python3
"""Bank d9 Stockfish rank/gap observations beside a derived replay corpus.

The frozen NNUE-bootstrap corpus stores a deliberately cold d9 policy, which
does not retain the centipawn gaps needed to distinguish true near ties from
float16 probability ties.  This tool replays the original derivation's prefix,
drop rule, shard boundaries, and within-shard permutation, then writes the
top-ranked compact move indices and their gaps from the best d9 score.  Every
output shard is checked against the derived row's ``(game_id, ply_index)`` and
legal mask before it is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    FULL_TO_COMPACT_POLICY,
    uci_to_policy_index,
)
from chess_anti_engine.replay.shard import iter_shard_paths
from scripts import derive_corpus_targets as derive
from scripts.bt4_policy_dump import file_sha256


SCHEMA = 1
SUMMARY_NAME = "sf_d9_rank_sidecar_summary.json"
DERIVE_SUMMARY = "derive_targets_summary.json"
INDEX_FIELD = "sf_d9_rank_index"
GAP_FIELD = "sf_d9_gap_cp"
COUNT_FIELD = "sf_d9_rank_count"
INVALID_INDEX = np.iinfo(np.uint16).max
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


@dataclass(frozen=True)
class RankObservation:
    game_id: int
    ply: int
    indices: np.ndarray
    gaps_cp: np.ndarray
    count: int


def _sha_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    writing = path.with_name(f".{path.name}.{os.getpid()}.writing")
    try:
        writing.write_text(
            json.dumps(dict(value), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(writing, path)
    finally:
        writing.unlink(missing_ok=True)


def d9_lines(row: Mapping[str, Any]) -> list[list[Any]]:
    """Return the one complete full-width d9 block from a run03-shaped row."""
    matches = [
        block
        for phase in row.get("phases", [])
        for block in phase.get("per_depth", [])
        if int(block.get("depth", -1)) == 9 and bool(block.get("complete"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has {len(matches)} "
            "complete d9 blocks, expected exactly one",
        )
    lines = list(matches[0].get("lines", []))
    if not lines:
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has an empty d9 block",
        )
    return lines


def rank_observation(row: Mapping[str, Any], *, top_k: int) -> RankObservation:
    """Extract compact indices and best-minus-move effective-cp gaps."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(d9_lines(row), key=lambda line: int(line[0]))
    ranks = [int(line[0]) for line in ranked]
    if ranks != list(range(1, len(ranked) + 1)):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed d9 ranks",
        )
    scores = np.asarray([float(line[2]) for line in ranked], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("d9 effective-cp scores must be finite")
    gaps = float(scores[0]) - scores
    if bool(np.any(gaps < -1e-6)):
        raise ValueError("d9 rank order disagrees with effective-cp scores")

    indices = np.full((top_k,), INVALID_INDEX, dtype=np.uint16)
    gaps_cp = np.full((top_k,), np.inf, dtype=np.float32)
    turn = str(row.get("stm")) == "w"
    count = min(top_k, len(ranked))
    seen: set[int] = set()
    for offset, line in enumerate(ranked[:count]):
        move = str(line[1])
        full_index = uci_to_policy_index(move, turn)
        compact_index = (
            int(FULL_TO_COMPACT_POLICY[full_index]) if full_index >= 0 else -1
        )
        if not 0 <= compact_index < COMPACT_POLICY_SIZE:
            raise ValueError(f"d9 move {move!r} is not compact-policy encodable")
        if compact_index in seen:
            raise ValueError(f"d9 block repeats compact move index {compact_index}")
        seen.add(compact_index)
        indices[offset] = compact_index
        gaps_cp[offset] = max(0.0, float(gaps[offset]))
    return RankObservation(
        game_id=int(row["game_id"]),
        ply=int(row["ply"]),
        indices=indices,
        gaps_cp=gaps_cp,
        count=count,
    )


def _flush(
    *,
    observations: Sequence[RankObservation],
    order: np.ndarray,
    source_path: Path,
    destination: Path,
    top_k: int,
    source_summary_sha256: str,
    raw_config_sha256: str,
) -> dict[str, Any]:
    rows = len(observations)
    if order.shape != (rows,):
        raise ValueError("permutation length does not match observation count")
    game_ids = np.asarray([row.game_id for row in observations], dtype=np.int64)[order]
    plies = np.asarray([row.ply for row in observations], dtype=np.int32)[order]
    indices = np.stack([row.indices for row in observations], axis=0)[order]
    gaps_cp = np.stack([row.gaps_cp for row in observations], axis=0)[order]
    counts = np.asarray([row.count for row in observations], dtype=np.uint8)[order]

    source: Any = zarr.open_group(str(source_path), mode="r")
    expected_game_ids = np.asarray(source["game_id"][:], dtype=np.int64)
    expected_plies = np.asarray(source["ply_index"][:], dtype=np.int32)
    if not np.array_equal(game_ids, expected_game_ids) or not np.array_equal(
        plies,
        expected_plies,
    ):
        raise ValueError(
            f"{source_path}: replayed raw row identity does not match derived order",
        )
    if indices.shape != (rows, top_k) or gaps_cp.shape != (rows, top_k):
        raise ValueError("rank sidecar arrays have the wrong shape")
    valid = np.arange(top_k)[None, :] < counts[:, None]
    if bool(np.any(indices[~valid] != INVALID_INDEX)) or not bool(
        np.all(np.isinf(gaps_cp[~valid]))
    ):
        raise ValueError("rank padding is malformed")
    if bool(np.any(indices[valid] >= COMPACT_POLICY_SIZE)) or not np.isfinite(
        gaps_cp[valid]
    ).all():
        raise ValueError("valid rank observations are out of range or non-finite")
    if bool(np.any(gaps_cp[valid] < 0.0)):
        raise ValueError("rank gaps must be nonnegative")
    if bool(np.any(gaps_cp[:, 0] != 0.0)):
        raise ValueError("the first-ranked move must have zero cp gap")
    legal = np.asarray(source["legal_mask"][:]) != 0
    row_index = np.repeat(np.arange(rows), counts.astype(np.int64))
    compact_index = indices[valid].astype(np.int64)
    if not bool(np.all(legal[row_index, compact_index])):
        raise ValueError(f"{source_path}: ranked d9 sidecar names an illegal move")
    stored_policy = np.asarray(source["policy_target"][:], dtype=np.float32)
    legal_policy = np.where(legal, stored_policy, -np.inf)
    if not bool(
        np.all(
            legal_policy[np.arange(rows), indices[:, 0].astype(np.int64)]
            == np.max(legal_policy, axis=1)
        )
    ):
        raise ValueError(
            f"{source_path}: d9 rank-1 is not a stored-policy maximum",
        )

    chunk_rows = max(1, min(rows, 8192))
    group: Any = zarr.open_group(str(destination), mode="w")
    group.create_dataset(
        INDEX_FIELD,
        data=indices,
        chunks=(chunk_rows, top_k),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        GAP_FIELD,
        data=gaps_cp,
        chunks=(chunk_rows, top_k),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        COUNT_FIELD,
        data=counts,
        chunks=(chunk_rows,),
        compressor=_COMPRESSOR,
    )
    identity_sha = _sha_arrays(game_ids, plies)
    payload_sha = _sha_arrays(indices, gaps_cp, counts)
    group.attrs.update(
        {
            "sf_d9_rank_sidecar_schema": SCHEMA,
            "source_shard": source_path.name,
            "source_rows": rows,
            "source_row_identity_sha256": identity_sha,
            "source_derive_summary_sha256": source_summary_sha256,
            "raw_config_sha256": raw_config_sha256,
            "depth": 9,
            "top_k": top_k,
            "index_encoding": "lc0_1858",
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
            "payload_sha256": payload_sha,
        }
    )
    return {
        "path": destination.name,
        "rows": rows,
        "source_row_identity_sha256": identity_sha,
        "payload_sha256": payload_sha,
    }


def bank(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw).resolve()
    source_dir = Path(args.shards).resolve()
    out_dir = Path(args.out).resolve()
    limit = int(args.limit)
    top_k = int(args.top_k)
    seed = int(args.seed)
    expected_rows = int(args.expected_rows)
    expected_shards = int(args.expected_shards)
    expected_summary_sha = str(args.expected_source_summary_sha256)
    if limit <= 0 or top_k <= 0 or top_k > 255:
        raise SystemExit("--limit and --top-k must be positive; top-k must be <=255")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; rank sidecars are immutable")
    writing = out_dir.with_name(out_dir.name + ".writing")
    if writing.exists():
        raise SystemExit(f"stale partial output exists: {writing}")

    source_summary_path = source_dir / DERIVE_SUMMARY
    source_summary_sha = file_sha256(source_summary_path)
    if source_summary_sha != expected_summary_sha:
        raise SystemExit(
            f"source summary SHA-256 mismatch: {source_summary_sha} != "
            f"{expected_summary_sha}",
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    realized = source_summary.get("realized", {})
    expected_contract = {
        "limit_requested": limit,
        "seed": seed,
        "rows_per_shard": int(args.rows_per_shard),
    }
    bad_contract = {
        key: (source_summary.get(key), value)
        for key, value in expected_contract.items()
        if source_summary.get(key) != value
    }
    if int(realized.get("rows_written", -1)) != expected_rows:
        bad_contract["realized.rows_written"] = (
            realized.get("rows_written"),
            expected_rows,
        )
    if bad_contract:
        raise SystemExit(f"derived source contract mismatch: {bad_contract}")
    source_paths = iter_shard_paths(source_dir)
    if len(source_paths) != expected_shards:
        raise SystemExit(
            f"derived source has {len(source_paths)} shards, expected {expected_shards}",
        )

    record = derive.read_corpus_record(raw_dir)
    raw_config_sha = str(record.facts.get("config_sha256", ""))
    source_corpus = source_summary.get("corpus", {})
    if not isinstance(source_corpus, dict) or source_corpus.get(
        "config_sha256"
    ) != raw_config_sha:
        raise SystemExit("raw and derived source config identities differ")

    writing.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    pending: list[RankObservation] = []
    written: list[dict[str, Any]] = []
    raw_rows = 0
    dropped_no_result = 0
    started = time.time()
    try:
        for raw_path in record.shards:
            for row in derive.iter_corpus_rows(raw_path):
                if raw_rows >= limit:
                    break
                raw_rows += 1
                if row.get("result") is None:
                    dropped_no_result += 1
                    continue
                pending.append(rank_observation(row, top_k=top_k))
                if len(pending) == int(args.rows_per_shard):
                    index = len(written)
                    written.append(
                        _flush(
                            observations=pending,
                            order=rng.permutation(len(pending)),
                            source_path=source_paths[index],
                            destination=writing / source_paths[index].name,
                            top_k=top_k,
                            source_summary_sha256=source_summary_sha,
                            raw_config_sha256=raw_config_sha,
                        )
                    )
                    pending = []
                    elapsed = max(time.time() - started, 1e-9)
                    if len(written) % 16 == 0:
                        print(
                            f"[sf-d9-ranks] {len(written)}/{expected_shards} shards, "
                            f"{sum(item['rows'] for item in written)} rows, "
                            f"{raw_rows / elapsed:.1f} raw rows/s",
                            flush=True,
                        )
            if raw_rows >= limit:
                break
        if pending:
            index = len(written)
            written.append(
                _flush(
                    observations=pending,
                    order=rng.permutation(len(pending)),
                    source_path=source_paths[index],
                    destination=writing / source_paths[index].name,
                    top_k=top_k,
                    source_summary_sha256=source_summary_sha,
                    raw_config_sha256=raw_config_sha,
                )
            )
        rows = sum(int(item["rows"]) for item in written)
        expected_dropped = int(realized.get("rows_dropped_no_result", -1))
        if (
            raw_rows != limit
            or rows != expected_rows
            or len(written) != expected_shards
            or dropped_no_result != expected_dropped
        ):
            raise ValueError(
                "rank sidecar cardinality mismatch: "
                f"raw={raw_rows}/{limit}, rows={rows}/{expected_rows}, "
                f"shards={len(written)}/{expected_shards}, "
                f"drops={dropped_no_result}/{expected_dropped}",
            )
        summary = {
            "schema": SCHEMA,
            "kind": "sf_d9_rank_gap_sidecar",
            "raw_dir": str(raw_dir),
            "raw_config_sha256": raw_config_sha,
            "raw_limit": limit,
            "raw_rows_read": raw_rows,
            "rows_dropped_no_result": dropped_no_result,
            "source_dir": str(source_dir),
            "source_derive_summary_sha256": source_summary_sha,
            "rows": rows,
            "shards": len(written),
            "rows_per_shard": int(args.rows_per_shard),
            "seed": seed,
            "depth": 9,
            "top_k": top_k,
            "index_encoding": "lc0_1858",
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
            "outputs": written,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(writing / SUMMARY_NAME, summary)
        os.replace(writing, out_dir)
    except BaseException:
        print(
            f"[sf-d9-ranks] FAILED; preserving partial output at {writing}",
            file=sys.stderr,
        )
        raise
    print(
        f"[sf-d9-ranks] complete: {rows} rows / {len(written)} shards -> {out_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return bank(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
