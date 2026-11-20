"""Training arms on one collection: a collect stage that REFERENCES another run's.

An arm is an ordinary run dir whose generation-g collect, instead of playing
games, seals a manifest pointing at a SOURCE run's sealed generation-g collect.
Every reader of rows / games goes through ``common.resolve_collect_dir``, so the
arm trains on exactly the source's rows: the holdout split (a hash of the game
id), the probe set (the first trained generation's held-out rows) and the batch
plan (seeded by ``--seed`` and the generation) are then identical across arms
that share ``--seed``, while each arm pins its own train settings.

The reference is created only when the source collect is sealed, is not itself
a reference, still matches every sha256 its own manifest recorded, was played
by the arm's own generation g-1 model (the source's gen-0 model must also be
the arm's), used the arm's TRAIN openings, and was collected with the arm's
collect settings (execution-only fields excepted). The reference manifest
records the sha256 of the source manifest, every shard file, games.jsonl,
starts.txt and the source gen-0 model as ``source_*`` inputs, which
``lineage.verify_stage`` re-hashes like in-run inputs: a source that is redone,
unsealed or edited later refuses every stage built on the reference. The
source's config and stats are copied, so the capped-collect taint rules apply
through the reference unchanged. Nothing here writes to, moves or unseals the
source: ``--redo`` / ``--redo-downstream`` act on the arm's own reference dir.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chess_anti_engine.simple_rl.common import (
    COLLECT_REFERENCE, input_ref, read_manifest, sha256_file, sha256_tree, write_manifest,
)

# Collect settings that say how/where the source ran, not which rows it made
# (the source's own record of them is copied with its config). Every other
# setting must equal the source's: a differing value would be pinned in the
# arm's run_config yet have no effect on the rows it trains on.
EXECUTION_ONLY_SETTINGS = frozenset({"device", "cuda_graphs", "eval_max_batch", "seed", "dup_sketch_budget",
                                     "exact_repeat_vs_previous"})


def source_collect_dir(source_run_dir: Path, generation: int) -> Path:
    return Path(source_run_dir).resolve() / f"gen_{int(generation):04d}" / "collect"


def settings_diff(source_manifest: dict[str, Any], settings: Any) -> dict[str, Any]:
    have = dict(source_manifest.get("config", {}).get("settings", {}))
    want = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(settings).items()}
    return {k: {"source": have.get(k), "arm": v} for k, v in sorted(want.items())
            if k not in EXECUTION_ONLY_SETTINGS and have.get(k) != v}


def precheck_source(source_run_dir: Path, generation: int, settings: Any) -> dict[str, Any]:
    """Cheap refusals (no shard hashing), also run by the driver before pinning."""
    src = source_collect_dir(source_run_dir, generation)
    manifest = read_manifest(src)
    if manifest is None:
        raise ValueError(f"--collect-from: {src} is not sealed (no manifest.json); an arm references only a "
                         "sealed collect")
    if COLLECT_REFERENCE in manifest:
        raise ValueError(f"--collect-from: {src} is itself a reference collect; point the arm at the run "
                         f"that holds the rows ({manifest[COLLECT_REFERENCE]['source_run_dir']})")
    diff = settings_diff(manifest, settings)
    if diff:
        raise ValueError(f"--collect-from: this arm's collect settings differ from the source collect's {diff}; "
                         "they would be pinned without affecting the rows. Pass the source's collect flags")
    return manifest


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"--collect-from: {msg}")


def reference_collect(*, collect_dir: Path, generation: int, source_run_dir: Path, actor_checkpoint: Path,
                      gen0_model: Path, train_jsonl: Path, settings: Any) -> dict[str, Any]:
    """Seal ``collect_dir`` as a verified reference to the source run's
    generation-``generation`` collect. Returns the source's stats."""
    src_run = Path(source_run_dir).resolve()
    src = source_collect_dir(src_run, generation)
    collect_dir = Path(collect_dir)
    _require(src.resolve() != collect_dir.resolve(), "the source is this run")
    manifest = precheck_source(src_run, generation, settings)
    for name, ref in sorted(manifest.get("outputs", {}).items()):
        p = src / Path(ref["path"]).name  # recorded as given (maybe relative); the file is the source's own
        _require(p.exists() and sha256_tree(p) == ref["sha256"],
                 f"the source collect's {name} ({p}) no longer matches the sha256 its manifest sealed")
    src_gen0 = src_run / "gen_0000" / "model.pt"
    gen0_manifest = read_manifest(src_run / "gen_0000")
    _require(gen0_manifest is not None and src_gen0.is_file()
             and sha256_file(src_gen0) == gen0_manifest["outputs"]["model"]["sha256"],
             f"the source's generation-0 model {src_gen0} is not sealed or no longer matches its manifest")
    src_gen0_sha = sha256_file(src_gen0)
    _require(sha256_file(gen0_model) == src_gen0_sha,
             f"this arm's generation-0 model is not the source's ({src_gen0_sha[:12]}); start the arm from the "
             "same --init-checkpoint")
    actor_sha = sha256_file(actor_checkpoint)
    _require(manifest["inputs"]["checkpoint"]["sha256"] == actor_sha,
             f"the source collect was played by {manifest['inputs']['checkpoint']['sha256'][:12]}, not this arm's "
             f"generation {generation - 1} model ({actor_sha[:12]}): its rows would be off-policy for this arm")
    _require(manifest["inputs"]["train_openings"]["sha256"] == sha256_file(train_jsonl),
             f"the source collect used other TRAIN openings than {train_jsonl}")
    if collect_dir.exists():
        shutil.rmtree(collect_dir)  # an unsealed attempt; never the source (checked above)
    collect_dir.mkdir(parents=True)
    inputs: dict[str, Any] = {"checkpoint": input_ref(actor_checkpoint), "train_openings": input_ref(train_jsonl),
                              "source_collect_manifest": input_ref(src / "manifest.json"),
                              "source_gen0_model": input_ref(src_gen0),
                              "source_games": input_ref(src / "games.jsonl"),
                              "source_starts": input_ref(src / "starts.txt")}
    shards = sorted((src / "shards").iterdir())
    _require(bool(shards), f"{src}/shards is empty")
    for p in shards:
        inputs[f"source_shard_{p.name}"] = input_ref(p)
    ref = {"source_run_dir": str(src_run), "source_collect_dir": str(src),
           "source_manifest_sha256": inputs["source_collect_manifest"]["sha256"],
           "source_gen0_model_sha256": src_gen0_sha, "source_shard_files": len(shards)}
    write_manifest(collect_dir, stage="collect", config=manifest["config"], inputs=inputs,
                   outputs=manifest.get("outputs", {}), counts=manifest.get("counts", {}),
                   seeds=manifest.get("seeds", {}),
                   extra={"generation": int(generation), "stats": manifest.get("stats", {}),
                          "opening_cursor": manifest.get("opening_cursor"),
                          "opening_cursor_before": manifest.get("opening_cursor_before"),
                          COLLECT_REFERENCE: ref})
    return dict(manifest.get("stats", {}))
