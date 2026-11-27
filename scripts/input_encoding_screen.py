"""Freeze recorded v2 inputs and measure feature reliance without training.

Run ``python -m scripts.input_encoding_screen --help``. The bank is a diagnostic
sample, not a certified holdout. Zeroing and spatial shuffling are inference
interventions, not estimates of the strength of a retrained smaller model.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from scripts.input_feature_diagnostics import (
    FEATURE_GROUPS,
    compare_outputs,
    perturb_inputs,
    summarize_metrics,
)

ENCODING = {
    "input_history_encoding": "lc0_root_legacy_meta",
    "input_extra_features": "v2_threats",
    "history_rep_fix": True,
    "policy_encoding": "lc0_1858",
}
DEFAULT_FAMILIES = (
    "king_safety", "pins", "pawn_structure", "mobility", "outposts_space",
    "threats_all", "extras_all",
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("wall budget reached at a batch/shard boundary")


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "script_sha256": {
            name: _sha(root / "scripts" / name)
            for name in ("input_encoding_screen.py", "input_feature_diagnostics.py")
        },
        "python": platform.python_version(), "numpy": np.__version__,
    }


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Publish a complete file exclusively; never replace a prior bank/readout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, allow_pickle=False, **arrays)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _take(value: Any, indices: np.ndarray) -> np.ndarray:
    if hasattr(value, "oindex"):
        return np.asarray(value.oindex[indices])
    return np.asarray(value[indices])


def _row_identity(source: str, shard: str, row: int, digest: str, legal: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps([source, shard, row, digest]).encode()
        + np.asarray(legal, dtype=np.uint8).tobytes()
    ).hexdigest()


def _source_manifest(path: Path) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(
        set(manifest) == {"schema_version", "sources"}
        and manifest["schema_version"] == 1,
        "expected schema_version=1 and sources",
    )
    entries: list[tuple[str, Path]] = []
    namespaces: set[str] = set()
    paths: set[Path] = set()
    for source in manifest["sources"]:
        _require(set(source) == {"source_id", "shards"}, "invalid source fields")
        name = source["source_id"]
        _require(
            isinstance(name, str) and bool(name.strip()) and name not in namespaces,
            "source_id must be a unique nonblank conversion/game-id namespace",
        )
        namespaces.add(name)
        _require(bool(source["shards"]), "source has no shards")
        for raw in source["shards"]:
            shard = Path(raw).expanduser()
            _require(shard.is_absolute(), "shard paths must be absolute")
            shard = shard.resolve(strict=True)
            _require(shard.suffix != ".npz", "use prepared Zarr shards; legacy NPZ readers decode eagerly before row caps")
            _require(shard not in paths, "duplicate/aliased source shard")
            paths.add(shard)
            entries.append((name, shard))
    _require(bool(entries), "manifest has no shards")
    return manifest, entries


def freeze_bank(
    manifest_path: Path, out: Path, *, max_rows: int = 4096,
    max_source_rows: int = 50_000, seed: int = 20261122, max_seconds: float = 300,
) -> dict[str, Any]:
    """Copy a uniform row sample of explicit prepared shards, retaining identity.

    Existing shard readers preserve packed-array handling and codec checks. We
    only materialize selected input/legal/game rows; teacher targets are unused.
    No history reconstruction, padding, remapping, or fallback is performed.
    """
    from chess_anti_engine.eval.lc0_control_rows import input_ids
    from chess_anti_engine.replay.shard import open_shard_arrays

    _require(max_rows > 0 and max_source_rows >= max_rows, "invalid row budget")
    _require(np.isfinite(max_seconds) and max_seconds > 0, "invalid wall budget")
    _require(not out.exists() and out.suffix == ".npz", "bank must be a new .npz path")
    deadline = time.monotonic() + max_seconds
    manifest_sha = _sha(manifest_path)
    manifest, entries = _source_manifest(manifest_path)
    _require(_sha(manifest_path) == manifest_sha, "source manifest changed while loading")
    _require(len(entries) <= 64, "at most 64 explicit shards per freeze")
    chunks: list[dict[str, np.ndarray]] = []
    sources: list[dict[str, Any]] = []
    with ExitStack() as stack:
        opened: list[tuple[str, Path, dict[str, Any], int]] = []
        total = 0
        for source, path in entries:
            _deadline(deadline)
            arrays, meta = stack.enter_context(open_shard_arrays(path, lazy=True))
            _require(
                meta.get("input_history_encoding") == ENCODING["input_history_encoding"]
                and meta.get("history_rep_fix") is True,
                f"{path}: requires recorded corrected lc0_root_legacy_meta inputs",
            )
            _require(
                str(np.asarray(arrays.get("_policy_encoding")).item()) == ENCODING["policy_encoding"],
                f"{path}: requires compact lc0_1858 policy encoding",
            )
            n = int(arrays["x"].shape[0])
            _require(arrays["x"].shape == (n, 175, 8, 8), f"{path}: requires 175 stored planes")
            for name, tail in (("legal_mask", (1858,)), ("game_id", ())):
                _require(name in arrays and arrays[name].shape == (n, *tail), f"{path}: missing {name}")
                flag = f"has_{name}"
                _require(flag in arrays and arrays[flag].shape == (n,), f"{path}: missing {flag}")
            total += n
            _require(total <= max_source_rows, "listed source rows exceed --max-source-rows")
            opened.append((source, path, arrays, n))
        _require(total > 0, "source population is empty")
        chosen = np.sort(np.random.default_rng(seed).choice(total, min(max_rows, total), replace=False))
        offset = 0
        for source, path, arrays, n in opened:
            _deadline(deadline)
            selected = chosen[(chosen >= offset) & (chosen < offset + n)] - offset
            offset += n
            sources.append({"source_id": source, "path": str(path), "population_rows": n,
                            "selected_rows": len(selected)})
            if not len(selected):
                continue
            for flag in ("has_legal_mask", "has_game_id"):
                _require(bool(np.all(_take(arrays[flag], selected) == 1)), f"{path}: selected rows lack {flag}")
            x = _take(arrays["x"], selected).astype(np.float32)
            legal = _take(arrays["legal_mask"], selected)
            _require(bool(np.isfinite(x).all()), f"{path}: nonfinite inputs")
            _require(bool(np.isin(legal, [0, 1]).all() and np.all(legal.sum(axis=1) > 0)), f"{path}: invalid legal masks")
            games = _take(arrays["game_id"], selected)
            _require(np.issubdtype(games.dtype, np.integer), f"{path}: noninteger game_id")
            # JSON tuples avoid collisions between namespace text and game numbers.
            clusters = [json.dumps([source, int(game)], separators=(",", ":")) for game in games]
            inputs = input_ids({"x": x})
            row_ids = [_row_identity(source, str(path), int(row), digest, mask)
                       for row, digest, mask in zip(selected, inputs, legal, strict=True)]
            chunks.append({"x": x, "legal": legal.astype(np.uint8),
                           "source_id": np.asarray([source] * len(selected)),
                           "shard": np.asarray([str(path)] * len(selected)),
                           "shard_row": selected.astype(np.int64), "game_id": games,
                           "cluster_key": np.asarray(clusters), "row_id": np.asarray(row_ids),
                           "input_id": np.asarray(inputs)})
    _deadline(deadline)
    _require(_sha(manifest_path) == manifest_sha, "source manifest changed during freeze")
    bank = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    metadata = {
        "schema_version": 1, "kind": "input_feature_diagnostic_bank",
        "encoding": ENCODING, "source_manifest": manifest,
        "source_manifest_sha256": manifest_sha, "sources": sources,
        "selection": "uniform rows without replacement from explicitly listed shards",
        "seed": seed, "max_rows": max_rows, "max_source_rows": max_source_rows,
        "rows": len(bank["x"]), "population_rows": total,
        "duplicate_input_rows": len(bank["x"]) - len(set(bank["input_id"].tolist())),
        "identity_note": "input_id hashes selected float32 x; row_id includes source/shard/row and legal mask",
        "exposure_status": "unknown; this freeze does not certify a held-out population",
        "code": _code_identity(),
    }
    _write_npz(out, **bank, metadata_json=np.asarray(json.dumps(metadata)))
    return {"bank": str(out.resolve()), "sha256": _sha(out), **metadata}


def load_bank(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from chess_anti_engine.eval.lc0_control_rows import input_ids

    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {name: np.array(source[name]) for name in source.files if name != "metadata_json"}
    _require(metadata.get("schema_version") == 1 and metadata.get("encoding") == ENCODING,
             "unsupported bank schema/encoding")
    n = len(arrays["x"])
    _require(n > 0 and metadata.get("rows") == n, "bank row count mismatch")
    for name in ("source_id", "shard", "shard_row", "game_id", "cluster_key", "row_id", "input_id"):
        _require(arrays[name].shape == (n,), f"bank {name} shape mismatch")
    _require(len(set(arrays["row_id"].tolist())) == n, "duplicate bank row identities")
    # Validate inputs and masks before a potentially expensive checkpoint load.
    perturb_inputs(arrays["x"][:1], family="extras_all", mode="identity",
                   row_ids=arrays["row_id"][:1].tolist(), seed=0)
    _require(arrays["x"].shape == (n, 175, 8, 8) and bool(np.isfinite(arrays["x"]).all()), "invalid bank inputs")
    _require(arrays["legal"].shape == (n, 1858)
             and bool(np.isin(arrays["legal"], [0, 1]).all())
             and bool(np.all(arrays["legal"].sum(axis=1) > 0)), "invalid bank legal masks")
    _require(all(str(k).strip() for k in arrays["cluster_key"]), "missing bank game clusters")
    digests = input_ids({"x": arrays["x"]})
    _require(np.array_equal(digests, arrays["input_id"]), "bank input content identity mismatch")
    expected_rows = [_row_identity(str(source), str(shard), int(row), digest, mask)
                     for source, shard, row, digest, mask in zip(
                         arrays["source_id"], arrays["shard"], arrays["shard_row"],
                         digests, arrays["legal"], strict=True)]
    _require(np.array_equal(expected_rows, arrays["row_id"]), "bank row content identity mismatch")
    expected_clusters = [json.dumps([str(source), int(game)], separators=(",", ":"))
                         for source, game in zip(arrays["source_id"], arrays["game_id"], strict=True)]
    _require(np.array_equal(expected_clusters, arrays["cluster_key"]), "bank game cluster identity mismatch")
    return arrays, metadata


def _predict(model: Any, bank: dict[str, np.ndarray], *, batch_size: int, device: str,
             deadline: float, seed: int, family: str = "extras_all",
             mode: str = "identity") -> dict[str, np.ndarray]:
    import torch

    pieces: dict[str, list[np.ndarray]] = {"policy": [], "wdl": []}
    with torch.inference_mode():
        for start in range(0, len(bank["x"]), batch_size):
            _deadline(deadline)
            end = start + batch_size
            x = perturb_inputs(bank["x"][start:end], family=family, mode=mode,
                               row_ids=bank["row_id"][start:end].tolist(), seed=seed)
            result = model(torch.from_numpy(x).to(device=device, dtype=torch.float32))
            policy = result["policy_own"] if "policy_own" in result else result["policy"]
            for name, tensor in (("policy", policy), ("wdl", result["wdl"])):
                pieces[name].append(tensor.detach().float().cpu().numpy())
    _deadline(deadline)
    return {name: np.concatenate(value) for name, value in pieces.items()}


def run_screen(
    bank_path: Path, checkpoint: Path, out_dir: Path, *, families: list[str],
    modes: list[str], batch_size: int = 64, bootstrap: int = 2000,
    seed: int = 20261122, device: str = "cpu", threads: int = 2,
    max_seconds: float = 900, gpu_mem_fraction: float | None = None,
) -> dict[str, Any]:
    import torch
    from chess_anti_engine.model import build_model
    from chess_anti_engine.uci.model_loader import model_config_from_arch

    _require(batch_size > 0 and threads > 0 and bootstrap >= 0, "invalid runtime budget")
    _require(np.isfinite(max_seconds) and max_seconds > 0, "invalid wall budget")
    _require(bool(families) and len(set(families)) == len(families)
             and set(families) <= set(FEATURE_GROUPS), "unknown/duplicate/empty families")
    _require(bool(modes) and len(set(modes)) == len(modes)
             and set(modes) <= {"zero", "spatial_shuffle"}, "unknown/duplicate/empty modes")
    target = torch.device(device)
    _require(target.type in {"cpu", "cuda"}, "device must be cpu or cuda[:index]")
    if target.type == "cuda":
        _require(gpu_mem_fraction is not None and 0 < gpu_mem_fraction <= 1,
                 "CUDA requires explicit --gpu-mem-fraction in (0,1]")
    else:
        _require(gpu_mem_fraction is None, "--gpu-mem-fraction applies only to CUDA")
    deadline = time.monotonic() + max_seconds
    bank_sha = _sha(bank_path)
    bank, metadata = load_bank(bank_path)
    _require(_sha(bank_path) == bank_sha, "bank changed while loading")
    checkpoint_sha = _sha(checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arch = checkpoint_data.get("arch", {})
    _require(all(arch.get(key) == value for key, value in ENCODING.items()),
             "checkpoint must embed the bank's complete input/history/policy identity")
    _require(not arch.get("use_dynamic_relations", False)
             and not arch.get("policy_dynamic_relations", False),
             "relation inputs are unsupported by this plane-only screen")
    raw_state = checkpoint_data["model"]
    _require(isinstance(raw_state, dict), "checkpoint model must be a state dictionary")
    # The serving loader's require_complete=True is strict only AFTER warm-start
    # migrations. In particular, 146 -> 175 zero-padding would manufacture zero
    # reliance on threats. Here the only permitted normalization is compile's
    # wrapper prefix; keys and tensor shapes must otherwise match exactly.
    state = {key.removeprefix("_orig_mod."): value for key, value in raw_state.items()}
    _require(len(state) == len(raw_state), "duplicate keys after compile-prefix removal")
    del checkpoint_data
    del raw_state
    _deadline(deadline)
    out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1, "status": "running", "interpretation": "inference reliance; not strength or retraining benefit",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha, "arch": arch},
        "bank": {"path": str(bank_path.resolve()), "sha256": bank_sha, "metadata": metadata},
        "runtime": {"device": device, "dtype": "float32", "threads": threads,
                    "batch_size": batch_size, "seed": seed, "bootstrap": bootstrap,
                    "max_seconds": max_seconds, "gpu_mem_fraction": gpu_mem_fraction,
                    "torch": torch.__version__},
        "families": families, "modes": modes, "code": _code_identity(), "cases": {},
    }
    _write_json(out_dir / "run.json", report)
    started = time.monotonic()
    try:
        torch.set_num_threads(threads)
        if target.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=target)
        model = build_model(model_config_from_arch(arch))
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval()
        _require(_sha(checkpoint) == checkpoint_sha, "checkpoint changed while loading")
        _require(all(getattr(model, key, None) == value for key, value in ENCODING.items()),
                 "loaded model encoding differs from bank")
        if hasattr(model, "_inference_only"):
            setattr(model, "_inference_only", True)
        report["unique_parameters"] = sum(p.numel() for p in model.parameters())
        options: dict[str, Any] = {"batch_size": batch_size, "device": device, "deadline": deadline, "seed": seed}
        reference = _predict(model, bank, **options)
        _write_npz(out_dir / "baseline.npz", **reference, row_id=bank["row_id"])
        cases = [("identity", "extras_all", "identity")]
        cases += [(f"{family}__{mode}", family, mode) for family in families for mode in modes]
        for name, family, mode in cases:
            _deadline(deadline)
            candidate = _predict(model, bank, family=family, mode=mode, **options)
            metrics = compare_outputs(reference, candidate, bank["legal"])
            if mode == "identity":
                _require(all(np.allclose(reference[key], candidate[key], rtol=1e-6, atol=1e-7)
                             for key in reference)
                         and not bool(np.any(metrics["top1_changed"])),
                         "identity repeat-forward control failed; stop before feature interpretation")
            summary = summarize_metrics(metrics, cluster_keys=bank["cluster_key"].tolist(),
                                        bootstrap=bootstrap, seed=seed)
            _deadline(deadline)
            _write_npz(out_dir / f"{name}.npz", **metrics, row_id=bank["row_id"], cluster_key=bank["cluster_key"])
            _write_json(out_dir / f"{name}.json", summary)
            report["cases"][name] = summary
            print(json.dumps({"completed": name, "elapsed_seconds": time.monotonic() - started}), flush=True)
        _deadline(deadline)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(out_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="bank selected recorded inputs with source/game identity")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--max-rows", type=int, default=4096)
    freeze.add_argument("--max-source-rows", type=int, default=50_000)
    freeze.add_argument("--max-seconds", type=float, default=300)
    freeze.add_argument("--seed", type=int, default=20261122)
    run = commands.add_parser("run", help="identity control, then feature dependence diagnostics")
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--families", nargs="+", choices=sorted(FEATURE_GROUPS), default=list(DEFAULT_FAMILIES))
    run.add_argument("--modes", nargs="+", choices=["zero", "spatial_shuffle"], default=["zero", "spatial_shuffle"])
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--bootstrap", type=int, default=2000)
    run.add_argument("--seed", type=int, default=20261122)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--device", default="cpu")
    run.add_argument("--gpu-mem-fraction", type=float)
    run.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_bank(args.manifest, args.out, max_rows=args.max_rows,
                             max_source_rows=args.max_source_rows, seed=args.seed,
                             max_seconds=args.max_seconds)
        print(json.dumps({key: result[key] for key in ("bank", "sha256", "rows", "population_rows")}))
    else:
        result = run_screen(args.bank, args.checkpoint, args.out_dir, families=args.families,
                            modes=args.modes, batch_size=args.batch_size, bootstrap=args.bootstrap,
                            seed=args.seed, device=args.device, threads=args.threads,
                            max_seconds=args.max_seconds, gpu_mem_fraction=args.gpu_mem_fraction)
        print(json.dumps({"status": result["status"], "report": str(args.out_dir / "report.json")}))


if __name__ == "__main__":
    main()
