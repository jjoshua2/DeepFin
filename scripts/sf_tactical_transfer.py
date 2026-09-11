#!/usr/bin/env python3
"""Strong, high-confidence Stockfish tactical mass transfer on stored B100 policy.

This is deliberately distinct from B100Tactical100. Ordinary positions are left
byte-identical unless Stockfish's best non-mate move beats the next-best non-mate
move by strictly more than 300 effective centipawns. On gated rows, probability
mass is transferred from the inferior set to the SF-best set instead of merely
multiplying bad moves downward.

Winning and losing mates are handled categorically. The producer reuses the
qualified original schema-3/raw-to-derived join and B100 parent checks from
``sf_policy_rewrite``. It changes only ``policy_target``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.stockfish.wdl import SF_CP_CLAMP_CP
from scripts import sf_policy_rewrite as base

SUMMARY = "bt4_sf_tactical_transfer_summary.json"
ALGORITHM = "stored-b100-sf-gapgt300-transfer50-mate75-v1"
GAP_CP = 300.0
ORDINARY_TRANSFER = 0.50
MATE_TRANSFER = 0.75


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _transfer_mass(
    probabilities: np.ndarray,
    donors: np.ndarray,
    recipients: np.ndarray,
    fraction: float,
) -> float:
    """Move a fraction of donor mass to recipients, preserving within-set odds."""
    require(
        probabilities.ndim == donors.ndim == recipients.ndim == 1
        and probabilities.shape == donors.shape == recipients.shape,
        "transfer masks differ",
    )
    require(
        0.0 <= fraction <= 1.0
        and bool(np.isfinite(probabilities).all())
        and bool((probabilities >= 0).all()),
        "invalid transfer state",
    )
    require(
        bool(recipients.any()) and not bool(np.any(donors & recipients)),
        "invalid transfer sets",
    )
    donor_mass = float(probabilities[donors].sum())
    amount = donor_mass * fraction
    if amount == 0.0:
        return 0.0

    probabilities[donors] *= 1.0 - fraction
    recipient_mass = float(probabilities[recipients].sum())
    if recipient_mass > 0.0:
        probabilities[recipients] += amount * probabilities[recipients] / recipient_mass
    else:
        probabilities[recipients] += amount / int(np.count_nonzero(recipients))
    return amount


def _best_nonmate_set(
    scores: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return exact best-score set, remaining eligible moves, and best/second gap."""
    require(
        scores.ndim == eligible.ndim == 1
        and scores.shape == eligible.shape
        and bool(eligible.any()),
        "invalid best-set inputs",
    )
    best_score = float(np.max(scores[eligible]))
    best = eligible & (scores == best_score)
    others = eligible & ~best
    gap = best_score - float(np.max(scores[others])) if bool(others.any()) else None
    return best, others, gap


def tactical_transfer_target(
    obs: base.Observation,
    stored: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply sparse strong SF correction to an actual stored B100 policy row."""
    require(
        stored.shape == (COMPACT_POLICY_SIZE,) and stored.dtype == np.float16,
        "B100 policy schema differs",
    )
    require(
        bool(np.isfinite(stored).all()) and bool((stored >= 0).all()),
        "invalid B100 policy",
    )
    require(
        obs.indices.ndim == 1
        and np.issubdtype(obs.indices.dtype, np.integer)
        and len(set(obs.indices.tolist())) == len(obs.indices)
        and bool(((obs.indices >= 0) & (obs.indices < COMPACT_POLICY_SIZE)).all()),
        "invalid tactical compact indices",
    )
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=bool)
    legal[obs.indices] = True
    require(not bool(np.any(stored[~legal] != 0)), "B100 illegal policy mass")
    require(
        abs(float(stored.astype(np.float64).sum()) - 1.0) <= 2**-10,
        "B100 policy mass differs",
    )

    scores = obs.scores
    require(
        scores.shape == obs.indices.shape
        and len(scores) > 0
        and bool(np.isfinite(scores).all()),
        "invalid tactical scores",
    )
    magnitude = np.abs(scores)
    mate_domain = magnitude > SF_CP_CLAMP_CP
    require(
        bool(np.isin(magnitude[mate_domain], base.MATE_SCORE_VALUES).all()),
        "effective score outside cp or shared mate domain",
    )

    wins = scores > SF_CP_CLAMP_CP
    losses = scores < -SF_CP_CLAMP_CP
    nonloss = ~losses
    initial = stored[obs.indices].astype(np.float64)
    initial /= initial.sum()
    ideal = initial.copy()

    category = "ordinary"
    ordinary_gate = False
    ordinary_gap_cp: float | None = None
    transferred_mate = 0.0
    transferred_ordinary = 0.0
    best = np.zeros(len(scores), dtype=bool)

    if bool(wins.any()):
        category = "winning_mate_available"
        best = wins.copy()
        transferred_mate = _transfer_mass(ideal, ~wins, wins, MATE_TRANSFER)
    elif bool(losses.all()):
        category = "all_forced_losses"
        best[:] = True
    else:
        best, ordinary_donors, ordinary_gap_cp = _best_nonmate_set(scores, nonloss)
        if bool(losses.any()):
            category = "losing_mate_alternatives"
            transferred_mate = _transfer_mass(ideal, losses, best, MATE_TRANSFER)
        if (
            ordinary_gap_cp is not None
            and ordinary_gap_cp > GAP_CP
            and bool(ordinary_donors.any())
        ):
            ordinary_gate = True
            transferred_ordinary = _transfer_mass(
                ideal,
                ordinary_donors,
                best,
                ORDINARY_TRANSFER,
            )

    require(
        bool(np.isfinite(ideal).all())
        and bool((ideal >= 0).all())
        and abs(float(ideal.sum()) - 1.0) <= 1e-12,
        "invalid ideal tactical transfer",
    )

    changed = not np.array_equal(ideal, initial)
    result = stored.copy() if not changed else np.zeros_like(stored)
    if changed:
        result[obs.indices] = ideal.astype(np.float32).astype(np.float16)
    require(
        bool(np.isfinite(result).all())
        and abs(float(result.astype(np.float64).sum()) - 1.0) <= 2**-10,
        "tactical stored policy mass differs",
    )

    normalized = result[obs.indices].astype(np.float64)
    normalized /= normalized.sum()
    initial_positive = initial > 0
    final_positive = result[obs.indices] > 0
    relative = np.zeros_like(initial)
    ideal_positive = ideal > 0
    relative[ideal_positive] = np.abs(
        normalized[ideal_positive] / ideal[ideal_positive] - 1.0
    )

    best_initial_mass = float(initial[best].sum()) if bool(best.any()) else 0.0
    best_ideal_mass = float(ideal[best].sum()) if bool(best.any()) else 0.0
    return result, {
        "category": category,
        "ordinary_gate": ordinary_gate,
        "ordinary_gap_cp": ordinary_gap_cp,
        "transferred_mate_mass": transferred_mate,
        "transferred_ordinary_mass": transferred_ordinary,
        "transferred_total_mass": transferred_mate + transferred_ordinary,
        "sf_best_base_mass": best_initial_mass,
        "sf_best_ideal_mass": best_ideal_mass,
        "support_gains": int(np.count_nonzero(~initial_positive & final_positive)),
        "support_losses": int(np.count_nonzero(initial_positive & ~final_positive)),
        "ideal_to_stored_relative_error_max": float(relative.max()),
        "ideal_to_stored_TV": float(np.abs(normalized - ideal).sum() / 2.0),
    }


def recipe_for_summary() -> dict[str, Any]:
    return {
        "ordinary_gate": "strict best-vs-second nonmate gap > 300 effective cp",
        "gap_cp": GAP_CP,
        "ordinary_transfer_fraction": ORDINARY_TRANSFER,
        "mate_transfer_fraction": MATE_TRANSFER,
        "winning_mate_handling": (
            "transfer 75% of non-winning-mate mass to winning-mate set"
        ),
        "losing_mate_handling": (
            "transfer 75% of losing-mate mass to best non-losing move set"
        ),
        "all_forced_loss_handling": "identity",
        "recipient_allocation": (
            "preserve B100 odds inside SF-best set; uniform if its base mass is zero"
        ),
        "cp_domain": [-SF_CP_CLAMP_CP, SF_CP_CLAMP_CP],
        "base": "normalized stored B100 float16 policy",
        "storage": "float64 mass transfer -> float32 -> float16; identity preserves bytes",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "raw",
        "source",
        "b100-source",
        "out",
        "expected-source-summary-sha256",
        "expected-b100-summary-sha256",
        "expected-b100-mix-sha256",
    ]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--minimum-free-gib", type=float, default=150.0)
    return p


def _b100_source(
    args: argparse.Namespace,
    summary: dict[str, Any],
    source: Path,
) -> tuple[Path, dict[Path, str]]:
    admitted = base.tactical_source(
        argparse.Namespace(
            tactical_bt4_source=args.b100_source,
            expected_bt4_summary_sha256=args.expected_b100_summary_sha256,
            expected_bt4_mix_sha256=args.expected_b100_mix_sha256,
            score_space="q",
            temperature=0.0005,
            expected_source_summary_sha256=args.expected_source_summary_sha256,
        ),
        summary,
        source,
    )
    require(admitted is not None, "B100 source admission failed")
    return admitted


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    require(
        math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0,
        "invalid disk reserve",
    )
    raw_dir = Path(args.raw).resolve()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    writing = out.with_name(out.name + ".writing")
    require(
        not out.exists() and not writing.exists(),
        "output or partial already exists",
    )
    require(
        all(
            out != root and root not in out.parents and out not in root.parents
            for root in (source, raw_dir)
        ),
        "output overlaps source",
    )

    source_summary_path = source / base.derive.SUMMARY_NAME
    summary_bytes = source_summary_path.read_bytes()
    require(
        hashlib.sha256(summary_bytes).hexdigest()
        == args.expected_source_summary_sha256,
        "source summary pin differs",
    )
    summary = json.loads(summary_bytes)
    record = base.derive.read_corpus_record(raw_dir)
    base.source_contract(summary, record)

    parent, parent_pins = _b100_source(args, summary, source)
    require(
        out != parent and parent not in out.parents and out not in parent.parents,
        "output overlaps B100 source",
    )
    specs = summary["shards"]
    expected_names = [spec["path"] for spec in specs]
    require(
        [p.name for p in sorted(source.glob("shard_*.zarr"))] == expected_names,
        "source shard membership differs",
    )
    require(
        [p.name for p in sorted(parent.glob("shard_*.zarr"))] == expected_names,
        "B100 shard membership differs",
    )

    manifest_path = raw_dir / "manifest.json"
    manifest_present = os.path.lexists(manifest_path)
    metadata = {
        raw_dir / "summary.json": base.file_sha256(raw_dir / "summary.json"),
        source_summary_path: args.expected_source_summary_sha256,
        **parent_pins,
    }
    if manifest_present:
        metadata[manifest_path] = base.file_sha256(manifest_path)
        manifest = base.derive.corpus.read_launch_manifest(raw_dir)
        require(
            all(
                manifest[key] == record.facts[key]
                for key in ("config_sha256", "row_schema", "staircase_parsed")
            ),
            "raw manifest/summary identity differs",
        )

    source_states = {
        source / spec["path"]: base.rank._storage_identity(source / spec["path"])
        for spec in specs
    }
    parent_states = {
        parent / spec["path"]: base.rank._storage_identity(parent / spec["path"])
        for spec in specs
    }

    from chess_anti_engine.stockfish import wdl

    producer_paths = [
        Path(__file__),
        Path(base.__file__),
        Path(base.derive.__file__),
        Path(base.rank.__file__),
    ]
    assert wdl.__file__ is not None
    producer_paths.append(Path(wdl.__file__))
    producer_hashes = {str(path): base.file_sha256(path) for path in producer_paths}

    writing.mkdir(parents=True)
    start = time.monotonic()
    raw_rows = dropped = rows_written = 0
    changed = ordinary_gated = 0
    support_gains = support_losses = 0
    max_mass_error = storage_relative_error = storage_tv = 0.0
    transferred_mate_sum = transferred_ordinary_sum = 0.0
    best_base_mass_sum = best_ideal_mass_sum = 0.0
    max_ordinary_gap = 0.0
    categories: dict[str, int] = {}
    outputs: list[dict[str, Any]] = []
    raw_proofs: dict[str, dict[str, Any]] = {}
    output_states: dict[Path, str] = {}
    last_by_worker: dict[int, tuple[int, int]] = {}
    keys = hashlib.sha256()
    pending: list[base.Observation] = []
    rng = np.random.Generator(np.random.PCG64(int(summary["seed"])))

    def guard() -> None:
        require(
            not (writing / "STOP").exists() and not (out.parent / "STOP").exists(),
            "STOP requested",
        )
        require(
            shutil.disk_usage(writing).free >= args.minimum_free_gib * 1024**3,
            "free disk reserve breached",
        )

    def flush() -> None:
        nonlocal rows_written, changed, ordinary_gated
        nonlocal support_gains, support_losses, max_mass_error
        nonlocal storage_relative_error, storage_tv
        nonlocal transferred_mate_sum, transferred_ordinary_sum
        nonlocal best_base_mass_sum, best_ideal_mass_sum, max_ordinary_gap

        guard()
        index = len(outputs)
        require(index < len(specs), "too many source rows")
        spec = specs[index]
        sf_path = source / spec["path"]
        parent_path = parent / spec["path"]
        dst = writing / spec["path"]
        require(len(pending) == spec["rows"], "source shard row count differs")

        order = rng.permutation(len(pending))
        aligned = [pending[int(i)] for i in order]
        sf_group: Any = zarr.open_group(str(sf_path), mode="r")
        b100_group: Any = zarr.open_group(str(parent_path), mode="r")
        require(
            frozenset(sf_group.array_keys()) == base.ARRAYS
            and frozenset(b100_group.array_keys()) == base.ARRAYS,
            "source array inventory differs",
        )
        base.shard_contract(dict(sf_group.attrs), summary, len(aligned))
        require(
            np.array_equal(sf_group["game_id"][:], [row.game for row in aligned])
            and np.array_equal(sf_group["ply_index"][:], [row.ply for row in aligned]),
            "raw/source shuffled identity differs",
        )

        battrs = dict(b100_group.attrs)
        require(
            {
                key: value
                for key, value in battrs.items()
                if not key.startswith("policy_target_mix_")
            }
            == dict(sf_group.attrs),
            "B100 source attrs differ",
        )
        require(
            battrs.get("policy_target_mix_kind") == "global"
            and battrs.get("policy_target_mix_alpha") == 1.0
            and battrs.get("policy_target_mix_bt4_temperature") == 0.5,
            "B100 source recipe attrs differ",
        )

        for column in base.ARRAYS:
            for array in (sf_group[column], b100_group[column]):
                for coordinates in itertools.product(
                    *(
                        range(math.ceil(n / c))
                        for n, c in zip(array.shape, array.chunks, strict=True)
                    )
                ):
                    require(
                        array._chunk_key(coordinates) in array.chunk_store,
                        "missing stored chunk",
                    )
            require(
                sf_group[column].shape == b100_group[column].shape
                and sf_group[column].dtype == b100_group[column].dtype,
                "B100 array schema differs",
            )

        sf_policy = np.asarray(sf_group["policy_target"][:])
        base_policy = np.asarray(b100_group["policy_target"][:])
        legal = np.asarray(sf_group["legal_mask"][:])
        require(
            sf_policy.dtype == base_policy.dtype == np.float16
            and sf_policy.shape == base_policy.shape == legal.shape,
            "stored policy schema differs",
        )

        new_policy = np.empty_like(base_policy)
        for row_index, obs in enumerate(aligned):
            mask = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
            mask[obs.indices] = 1
            require(
                np.array_equal(mask, legal[row_index]),
                "stored/raw legal support differs",
            )
            require(
                np.array_equal(base.target(obs, "q", 0.0005), sf_policy[row_index]),
                "original q-policy reconstruction differs",
            )
            new_policy[row_index], diagnostic = tactical_transfer_target(
                obs, base_policy[row_index]
            )
            category = str(diagnostic["category"])
            categories[category] = categories.get(category, 0) + 1
            ordinary_gated += int(bool(diagnostic["ordinary_gate"]))
            support_gains += int(diagnostic["support_gains"])
            support_losses += int(diagnostic["support_losses"])
            transferred_mate_sum += float(diagnostic["transferred_mate_mass"])
            transferred_ordinary_sum += float(
                diagnostic["transferred_ordinary_mass"]
            )
            best_base_mass_sum += float(diagnostic["sf_best_base_mass"])
            best_ideal_mass_sum += float(diagnostic["sf_best_ideal_mass"])
            gap = diagnostic["ordinary_gap_cp"]
            if gap is not None:
                max_ordinary_gap = max(max_ordinary_gap, float(gap))
            storage_relative_error = max(
                storage_relative_error,
                float(diagnostic["ideal_to_stored_relative_error_max"]),
            )
            storage_tv = max(storage_tv, float(diagnostic["ideal_to_stored_TV"]))
            keys.update(bytes.fromhex(obs.input_key))

        require(
            base.rank._storage_identity(sf_path) == source_states[sf_path]
            and base.rank._storage_identity(parent_path) == parent_states[parent_path],
            "source changed before copy",
        )
        copied = base.copy_shard(parent_path, dst)

        sf_files = {
            str(path.relative_to(sf_path)): path
            for path in sf_path.rglob("*")
            if path.is_file()
            and path.relative_to(sf_path).parts[0] != "policy_target"
            and str(path.relative_to(sf_path)) != ".zattrs"
        }
        require(
            set(sf_files) == set(copied) - {".zattrs"},
            "B100 nonpolicy file inventory differs",
        )
        require(
            all(base.file_sha256(path) == copied[rel] for rel, path in sf_files.items()),
            "B100 changed nonpolicy bytes",
        )

        dest: Any = zarr.open_group(str(dst), mode="a")
        dest["policy_target"][:] = new_policy
        require(
            np.array_equal(dest["policy_target"][:], new_policy),
            "policy readback differs",
        )
        for name in list(dest.attrs):
            if name.startswith("policy_target_mix_"):
                del dest.attrs[name]
        dest.attrs["policy_target_rewrite"] = {
            "algorithm": ALGORITHM,
            **recipe_for_summary(),
            "source_summary_sha256": args.expected_b100_summary_sha256,
            "sf_summary_sha256": args.expected_source_summary_sha256,
            "mutated_arrays": ["policy_target"],
        }

        for rel, digest in copied.items():
            if rel != ".zattrs":
                require(base.file_sha256(dst / rel) == digest, "nonpolicy copy changed")

        changed += int(np.count_nonzero(np.any(new_policy != base_policy, axis=1)))
        max_mass_error = max(
            max_mass_error,
            float(np.max(np.abs(new_policy.astype(np.float64).sum(axis=1) - 1.0))),
        )
        rows_written += len(aligned)
        outputs.append(
            {
                "path": spec["path"],
                "rows": len(aligned),
                "source_storage_identity": parent_states[parent_path],
                "original_sf_storage_identity": source_states[sf_path],
                "source_policy_sha256": base.rank._sha_arrays(base_policy),
                "policy_sha256": base.rank._sha_arrays(new_policy),
                "copied_file_hashes": copied,
            }
        )
        output_states[dst] = base.rank._storage_identity(dst)
        pending.clear()

    try:
        for path in record.shards:
            if raw_rows >= summary["limit_requested"]:
                break
            guard()
            require(not path.is_symlink() and path.is_file(), "nonregular raw storage")
            before = base.rank._file_identity(path)
            consumed = 0
            for raw in base.derive.iter_corpus_rows(path):
                if raw_rows >= summary["limit_requested"]:
                    break
                raw_rows += 1
                consumed += 1
                base.derive._check_row_identity(raw, str(record.facts["config_sha256"]))
                require(
                    all(
                        type(raw[key]) is int and raw[key] >= 0
                        for key in ("worker_id", "game_id", "ply")
                    ),
                    "invalid raw physical identity",
                )
                worker = int(raw["worker_id"])
                key = (int(raw["game_id"]), int(raw["ply"]))
                require(
                    worker not in last_by_worker or key > last_by_worker[worker],
                    "duplicate/nonmonotone original worker identity",
                )
                last_by_worker[worker] = key
                if raw.get("result") is None:
                    dropped += 1
                    continue
                anomalies = raw["phases"][0].get("anomalies", {})
                require(
                    type(anomalies.get("bound_lines")) is int
                    and anomalies["bound_lines"] >= 0,
                    "missing original bound-line accounting",
                )
                require(
                    all(
                        line[3] is None or (type(line[3]) is int and line[3] >= 0)
                        for line in base.rank.d9_lines(raw)
                        if len(line) == 4
                    ),
                    "invalid d9 nodes field",
                )
                pending.append(
                    base.observation(raw, str(record.facts["config_sha256"]))
                )
                if len(pending) == summary["rows_per_shard"]:
                    flush()
            raw_proofs[str(path)] = {
                "identity": before,
                "sha256": base.file_sha256(path),
                "rows_consumed": consumed,
            }
            require(
                base.rank._file_identity(path) == before,
                "raw file changed during read",
            )

        if pending:
            flush()
        require(
            raw_rows == summary["limit_requested"]
            and rows_written == summary["realized"]["rows_written"]
            and dropped == summary["realized"]["rows_dropped_no_result"]
            and len(outputs) == len(specs),
            "final source complement differs",
        )
        guard()
        for path, state in {**source_states, **parent_states}.items():
            require(base.rank._storage_identity(path) == state, "source changed before publication")
        for path, proof in raw_proofs.items():
            require(
                base.rank._file_identity(Path(path)) == proof["identity"],
                "raw source changed before publication",
            )
        require(
            os.path.lexists(manifest_path) == manifest_present,
            "raw manifest presence changed",
        )
        for path, digest in metadata.items():
            require(base.file_sha256(path) == digest, "source metadata changed")

        result = {
            "schema": 1,
            "status": "COMPLETE",
            "kind": "bt4_sf_tactical_policy_mass_transfer",
            "algorithm": ALGORITHM,
            "source_dir": str(parent),
            "sf_source_dir": str(source),
            "raw_dir": str(raw_dir),
            "raw_limit": raw_rows,
            "rows": rows_written,
            "shards": len(outputs),
            "rows_dropped_no_result": dropped,
            "changed_rows": changed,
            "ordinary_gated_rows": ordinary_gated,
            "categories": categories,
            "transferred_mate_mass_sum": transferred_mate_sum,
            "transferred_ordinary_mass_sum": transferred_ordinary_sum,
            "sf_best_base_mass_sum": best_base_mass_sum,
            "sf_best_ideal_mass_sum": best_ideal_mass_sum,
            "ordinary_gap_cp_max": max_ordinary_gap,
            "stored_support_gains": support_gains,
            "stored_support_losses": support_losses,
            "stored_mass_error_max": max_mass_error,
            "stored_relative_error_max": storage_relative_error,
            "stored_TV_error_max": storage_tv,
            "mutated_arrays": ["policy_target"],
            "nonpolicy_arrays_copied": 16,
            "producer_sha256": producer_hashes,
            "source_derive_summary_sha256": args.expected_b100_summary_sha256,
            "source_policy_summary_sha256": args.expected_b100_mix_sha256,
            "sf_derive_summary_sha256": args.expected_source_summary_sha256,
            "raw_manifest_present": manifest_present,
            "metadata_sha256": {str(path): digest for path, digest in metadata.items()},
            "raw_files": raw_proofs,
            "raw_history_keys_sha256_in_emitted_order": keys.hexdigest(),
            "recipe": recipe_for_summary(),
            "outputs": outputs,
            "elapsed_seconds": time.monotonic() - start,
            "history_lineage": (
                "Inherited original input_key_verified/source x; no fresh history re-encoding."
            ),
            "score_bounds": (
                "Original producer discards UCI upper/lower bounds; rows retain "
                "aggregate counts, not per-line flags."
            ),
            "limitations": (
                "Original historical control/provenance limitations unchanged; "
                "d9 tactical confidence is not d10/d12 stability; no inference, "
                "training, playing-strength or G10 transfer claim."
            ),
        }
        base.rank._atomic_json(writing / SUMMARY, result)
        derived = dict(summary)
        derived["policy_target_postprocess"] = {
            key: value for key, value in result.items() if key != "outputs"
        }
        base.rank._atomic_json(writing / base.derive.SUMMARY_NAME, derived)
        guard()
        for path, state in output_states.items():
            require(
                base.rank._storage_identity(path) == state,
                "output changed before publication",
            )
        os.replace(writing, out)
        return result
    except BaseException as exc:
        base.rank._atomic_json(
            writing / "failed.json",
            {
                "error": repr(exc),
                "partial_output_preserved": True,
                "rows_written": rows_written,
                "raw_rows_read": raw_rows,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    rewrite(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
