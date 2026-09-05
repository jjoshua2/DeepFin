#!/usr/bin/env python3
"""Bank a bounded paired-checkpoint policy-tail diagnostic, or reread a bank.

No selfplay, Stockfish, optimizer, model publication or production YAML writes.
Run from the repository root with PYTHONPATH=. (see the experiment record).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.policy_tail import TailCohort, compare_policies, freeze_cohort

SCHEMA_VERSION = 1
BANK_FIELDS = ("reference_logits", "candidate_logits", "target", "legal", "cohort", "row_ids", "group_ids")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(path: str) -> Path:
    p = Path(path).resolve()
    if p.is_dir():
        p /= "trainer.pt"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def primary_logits(output: Any) -> torch.Tensor:
    """The two documented primary-head aliases; never fall back to policy_sf."""
    if not isinstance(output, dict):
        raise ValueError("expected a model output dictionary")
    for key in ("policy_own", "policy"):
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError("missing primary policy_own/policy output")


def select_labeled_rows(batch: dict[str, Any]) -> np.ndarray:
    """Presence flags are authoritative; missing labels are not negative data."""
    n = len(batch["has_policy"])
    policy_flags = np.asarray(batch["has_policy"])
    legal_flags = np.asarray(batch.get("has_legal_mask", np.zeros(n)))
    if not np.isin(policy_flags, [0, 1]).all() or not np.isin(legal_flags, [0, 1]).all():
        raise ValueError("presence flags must contain only 0/1")
    has_policy = policy_flags != 0
    has_legal = legal_flags != 0
    if has_policy.shape != (n,) or has_legal.shape != (n,):
        raise ValueError("presence flags must have one value per row")
    rows = has_policy & has_legal
    if rows.any() and ("policy_target" not in batch or "legal_mask" not in batch):
        raise ValueError("presence flag claims a missing target/legal array")
    return rows


def validate_bank(bank: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported observation-bank schema")
    for key in BANK_FIELDS:
        if key not in bank:
            raise ValueError(f"observation bank is missing {key}")
    if any(bank[key].ndim < 1 for key in BANK_FIELDS):
        raise ValueError("bank arrays must have a row dimension")
    n = bank["reference_logits"].shape[0]
    if not n or any(bank[key].shape[0] != n for key in BANK_FIELDS):
        raise ValueError("bank arrays must share a nonempty row dimension")
    if bank["legal"].dtype != np.bool_ or bank["cohort"].dtype != np.bool_:
        raise ValueError("bank legal/cohort arrays must be boolean, not probabilities")
    for key in ("row_ids", "group_ids"):
        if bank[key].shape != (n,) or bank[key].dtype.kind not in "US":
            raise ValueError(f"{key} must be a string vector")
    if len(set(bank["row_ids"].tolist())) != n:
        raise ValueError("duplicate row identities in observation bank")


def read_bank(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        if "manifest_json" not in source:
            raise ValueError("observation bank lacks provenance manifest")
        manifest = json.loads(str(source["manifest_json"].item()))
        bank = {key: source[key] for key in source.files if key != "manifest_json"}
    validate_bank(bank, manifest)
    return bank, manifest


def summarize_bank(
    bank: dict[str, np.ndarray], manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_bank(bank, manifest)
    spec = TailCohort(**manifest["cohort_spec"])
    report, per_row = compare_policies(
        torch.from_numpy(bank["reference_logits"]),
        torch.from_numpy(bank["candidate_logits"]),
        torch.from_numpy(bank["target"]), torch.from_numpy(bank["legal"]),
        cohort=torch.from_numpy(bank["cohort"]), spec=spec,
    )
    report["provenance"] = manifest
    report["missing_group_ids"] = int(np.count_nonzero(bank["group_ids"] == ""))
    report["uncertainty"] = (
        "Paired point estimates only. Cluster retained per-row observations by verified "
        "source-qualified games before a statistical decision; shard/game keys alone "
        "do not prove independence across split games, duplicates, or seed families."
    )
    report["interpretation"] = (
        "Reference-rare, historical-search-target-promoted moves, not verified tactical "
        "rescues. T=1 legal policy, not the tempered/noisy search prior. IID sampling "
        "curves are not MCTS/Gumbel inclusion probabilities. No playing-strength verdict."
    )
    # Fail closed on non-finite summaries before writing misleading JSON.
    json.dumps(report, allow_nan=False)
    rows = {key: value.cpu().numpy() for key, value in per_row.items()}
    rows.update(row_ids=bank["row_ids"], group_ids=bank["group_ids"])
    return report, rows


def collect_bank(args: argparse.Namespace, spec: TailCohort) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # Keep report-only mode and numerical tests independent of native/Trainer imports.
    from chess_anti_engine.encoding import input_plane_count
    from chess_anti_engine.encoding.lc0 import uses_lc0_root_history
    from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
    from chess_anti_engine.train.trainer import select_input_history_arrays
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint, model_config_from_arch

    paths = [checkpoint_path(args.reference), checkpoint_path(args.candidate)]
    hashes = [sha256_file(p) for p in paths]
    configs = []
    archs = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("arch"), dict):
            raise ValueError("banking requires checkpoints with embedded arch metadata")
        archs.append(payload["arch"])
        configs.append(model_config_from_arch(payload["arch"]))
        del payload
    fields = ("policy_encoding", "input_history_encoding", "history_rep_fix", "input_extra_features")
    signature = {key: getattr(configs[0], key) for key in fields}
    if any(getattr(configs[1], key) != value for key, value in signature.items()):
        raise ValueError("checkpoint input/policy identities differ; use a separately validated conversion")
    if any(c.use_dynamic_relations or c.policy_dynamic_relations for c in configs):
        raise ValueError("dynamic-relation checkpoints are not supported by this initial diagnostic")
    if signature["policy_encoding"] != "lc0_1858":
        raise ValueError("this collector requires compact lc0_1858 checkpoints")
    models = [load_model_from_checkpoint(p, device=args.device, require_complete=True) for p in paths]
    parts: dict[str, list[np.ndarray]] = {key: [] for key in BANK_FIELDS if key != "cohort"}
    parts["x"] = []
    sources = []
    scanned = selected = missing_policy = missing_legal = 0
    replay_dir = Path(args.replay_dir).resolve()
    shards = sorted(iter_shard_paths(replay_dir))[:args.max_shards]
    for shard in shards:
        if selected >= args.max_positions:
            break
        # Shared guarded loader: Zarr stays lazy; NPZ is legacy/eager by contract.
        arrays, metadata = load_shard_arrays(shard, lazy=True)
        declared = str(np.asarray(arrays.get("_policy_encoding", "")).item())
        history = str(np.asarray(arrays.get("_input_history_encoding", "")).item())
        rep_fix = str(np.asarray(arrays.get("_history_rep_fix", "false")).item()).lower() == "true"
        if declared != signature["policy_encoding"] or history != signature["input_history_encoding"]:
            raise ValueError(f"{shard}: replay/checkpoint encoding mismatch or missing marker")
        if rep_fix != signature["history_rep_fix"]:
            raise ValueError(f"{shard}: repetition-fix identity mismatch")
        used = 0
        n = int(arrays["x"].shape[0])
        keys = (
            "x", "x_lc0_root", "has_x_lc0_root", "policy_target", "legal_mask",
            "has_policy", "has_legal_mask", "game_id", "has_game_id", "ply_index", "has_ply_index",
        )
        for start in range(0, n, args.batch_size):
            if selected >= args.max_positions:
                break
            stop = min(start + args.batch_size, n)
            batch = {key: np.asarray(arrays[key][start:stop]) for key in keys if key in arrays}
            scanned += stop - start
            missing_policy += int(np.count_nonzero(np.asarray(batch["has_policy"]) == 0))
            missing_legal += int(np.count_nonzero(np.asarray(batch.get("has_legal_mask", np.zeros(stop-start))) == 0))
            indices = np.flatnonzero(select_labeled_rows(batch))[:args.max_positions-selected]
            if not len(indices):
                continue
            batch = {key: value[indices] for key, value in batch.items()}
            for key in ("_input_history_encoding", "_history_rep_fix", "_policy_encoding"):
                if key in arrays:
                    batch[key] = np.asarray(arrays[key])
            if uses_lc0_root_history(signature["input_history_encoding"]):
                batch = select_input_history_arrays(
                    batch, input_history_encoding=signature["input_history_encoding"],
                    allow_lossy_legacy_remap=False,
                )
            x = np.asarray(batch["x"])
            if x.shape[1:] != (input_plane_count(configs[0].input_extra_features), 8, 8):
                raise ValueError("replay input shape differs from checkpoint; no implicit padding/upgrades")
            if not np.isfinite(x).all():
                raise ValueError("nonfinite input planes")
            raw_legal = np.asarray(batch["legal_mask"])
            if not np.isin(raw_legal, [0, 1]).all():
                raise ValueError("legal mask must contain only 0/1")
            legal = raw_legal.astype(bool)
            target = np.asarray(batch["policy_target"], dtype=np.float32)
            with torch.inference_mode():
                inputs = torch.as_tensor(x, device=args.device, dtype=torch.float32)
                for key, model in zip(("reference_logits", "candidate_logits"), models, strict=True):
                    logits = primary_logits(model(inputs)).float().cpu().numpy()
                    if logits.shape != target.shape:
                        raise ValueError("network/replay policy widths differ; no action-index reinterpretation")
                    parts[key].append(logits)
            parts["target"].append(target)
            parts["legal"].append(legal)
            parts["x"].append(x)
            parts["row_ids"].append(np.asarray([f"{shard.resolve()}#row={start+int(i)}" for i in indices]))
            has_game = np.asarray(batch.get("has_game_id", np.zeros(len(indices)))) != 0
            game = np.asarray(batch.get("game_id", np.zeros(len(indices), dtype=np.int64)))
            parts["group_ids"].append(np.asarray([
                f"{shard.resolve()}#game={int(game[i])}" if has_game[i] else "" for i in range(len(indices))
            ]))
            used += len(indices)
            selected += len(indices)
        sources.append({"path": str(shard.resolve()), "selected_rows": used, "metadata": metadata})
    if not selected:
        raise ValueError("no rows with both policy labels and legal masks in the bounded selection")
    if hashes != [sha256_file(p) for p in paths]:
        raise RuntimeError("checkpoint changed during collection; copy immutable snapshots first")
    bank = {key: np.concatenate(values) for key, values in parts.items()}
    bank["cohort"] = freeze_cohort(
        torch.from_numpy(bank["reference_logits"]), torch.from_numpy(bank["target"]),
        torch.from_numpy(bank["legal"]), spec,
    ).numpy()
    manifest = {
        "schema_version": SCHEMA_VERSION, "cohort_spec": asdict(spec),
        "reference": {"path": str(paths[0]), "sha256": hashes[0], "arch": archs[0]},
        "candidate": {"path": str(paths[1]), "sha256": hashes[1], "arch": archs[1]},
        "input_identity": signature, "head": "policy_own", "target_field": "policy_target",
        "policy_temperature": 1.0, "sources": sources,
        "selection": "sorted shard prefix, first eligible rows; not random or representative",
        "scanned_rows": scanned, "selected_rows": selected,
        "missing_policy_rows": missing_policy, "missing_legal_rows": missing_legal,
        "limits": {"max_positions": args.max_positions, "max_shards": args.max_shards},
        "runtime": {"torch": str(torch.__version__), "device": args.device, "threads": args.threads},
        "collector_sha256": sha256_file(Path(__file__)),
        "math_sha256": sha256_file(Path(inspect.getfile(freeze_cohort))),
        "loader_sha256": sha256_file(Path(inspect.getfile(load_model_from_checkpoint))),
        "shard_reader_sha256": sha256_file(Path(inspect.getfile(load_shard_arrays))),
    }
    validate_bank(bank, manifest)
    return bank, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", type=Path, help="reread a frozen observations.npz without model inference")
    source.add_argument("--replay-dir", help="frozen, quarantined replay; not a live rotating replay directory")
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-positions", type=int, default=2048)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--max-prior", type=float)
    parser.add_argument("--min-target", type=float)
    parser.add_argument("--min-boost", type=float)
    args = parser.parse_args(argv)
    if any(getattr(args, key) < 1 for key in ("threads", "batch_size", "max_positions", "max_shards")):
        parser.error("threads, batch-size, max-positions and max-shards must be positive")
    changes = {key: getattr(args, key) for key in ("max_prior", "min_target", "min_boost") if getattr(args, key) is not None}
    if args.bank and (changes or args.reference or args.candidate):
        parser.error("bank replay preserves its checkpoint identities/cohort; do not override them")
    if not args.bank and (not args.reference or not args.candidate):
        parser.error("--replay-dir requires --reference and --candidate")
    spec = TailCohort(**changes)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.bank:
        bank, manifest = read_bank(args.bank)
        bank_path = args.bank.resolve()
    else:
        bank, manifest = collect_bank(args, spec)
        bank_path = args.output_dir / "observations.npz"
        np.savez_compressed(bank_path, **bank, manifest_json=np.asarray(json.dumps(manifest, allow_nan=False)))
    report, rows = summarize_bank(bank, manifest)
    report["bank"] = {"path": str(bank_path), "sha256": sha256_file(bank_path)}
    np.savez_compressed(args.output_dir / "per_row.npz", **rows)
    # This is the completion marker. A failed/incomplete directory has no report.
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "rows", "rare_rows", "rare_actions", "target_ce_delta", "rare_log_mass_delta",
        "rare_action_drop_10x_fraction",
    )}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
