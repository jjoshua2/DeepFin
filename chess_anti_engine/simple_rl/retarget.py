"""Offline retarget: a new sealed collect whose policy target is rebuilt from
the source collect's stored root search statistics.

Why offline. ``--no-target-q-rescale`` changes only the stored target, never
the search (``GumbelConfig.target_q_rescale`` is store-arm only), but a
collection plays at a positive move temperature and samples the played move
FROM the stored target, so two same-seed collections with and without it do
not play the same games. Rebuilding the target from the root statistics every
simple-RL collect records (``root_log_prior`` / ``root_child_q`` /
``root_child_visits`` / ``root_net_value``) gives the other target on the SAME
games.

What it writes. A full run dir, not a reference collect (``--collect-from``
refuses a reference of a reference): ``gen_0000`` copied from the source run
(model and manifest, so an arm's generation-0 check passes), and
``gen_<g>/collect`` with every shard array byte-identical to the source's
except ``policy_target``, the row metadata, games and starts copied, and a
manifest whose config, inputs, counts and seeds are the source's (the games
were played exactly as the source recorded) plus a ``retarget`` block: the
source run / collect / manifest sha256, the rebuild settings, and the row
accounting. Its shard digest therefore differs from the source's.

The sealed settings state the target (``target_q_rescale`` is the rebuilt
one), so an arm must pass the matching flag (``--no-target-q-rescale``) and its
pinned run_config says what it trains on; the ``retarget`` block records that
the games were played under the source's target and search config. Sealing
happens inside a ``.<name>.partial`` dir that is renamed into place, so a kill
leaves nothing a rerun refuses. Shard bytes are not idempotent across two
identical retargets (zarr chunk bytes of ``future_legal_mask`` differ; every
array's values are identical), so compare retargets by array values, not by
the shard digest.

Rows:
- a policy row with root statistics: target rebuilt (``rebuild_improved_policy``
  with the source's search config and the requested ``target_q_rescale``);
- a policy row without them whose stored target is one-hot (a solved root: the
  search returned an exact move and recorded no statistics) or that is a
  tablebase policy override: kept, and counted;
- any other policy row without root statistics: REFUSED (the target cannot be
  rebuilt and must not silently stay the other arm's target);
- a row with no policy target (playout-capped fast rows): untouched.
"""
from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.simple_rl.common import (
    COLLECT_REFERENCE,
    input_ref,
    read_manifest,
    sha256_file,
    sha256_tree,
    write_manifest,
)

RETARGET = "retarget"
# A stored policy target counts as one-hot (a solved root) at this mass on one move.
ONE_HOT_MIN = 0.999


def _gen(run_dir: Path, g: int) -> Path:
    return Path(run_dir) / f"gen_{int(g):04d}"


def gumbel_cfg_for(manifest: dict[str, Any], *, target_q_rescale: bool) -> Any:
    """The source collect's own search config, with the requested target knob."""
    from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config

    search = SearchConfig(**manifest["config"]["dataclasses"]["search"])
    search = replace(search, gumbel_target_q_rescale=bool(target_q_rescale))
    cfg = build_selfplay_gumbel_config(search=search, game=GameConfig(), simulations=search.simulations)
    if bool(cfg.target_q_rescale) != bool(target_q_rescale):
        raise RuntimeError(f"rebuild config realised target_q_rescale {cfg.target_q_rescale}, "
                           f"requested {target_q_rescale}")
    return cfg


def rebuild_rows(arrays: dict[str, np.ndarray], meta: dict[str, np.ndarray], cfg: Any) -> tuple[
        np.ndarray, dict[str, Any]]:
    """The rebuilt ``policy_target`` array (same dtype) and the row accounting."""
    from chess_anti_engine.mcts.gumbel_c import rebuild_improved_policy

    stored = arrays["policy_target"]
    n = int(stored.shape[0])
    policy = arrays["has_policy"].astype(bool)
    has_root = (arrays["has_root_log_prior"].astype(bool) if "has_root_log_prior" in arrays
                else np.zeros(n, dtype=bool))
    tb = np.asarray(meta.get("tb_policy_override", np.zeros(n, dtype=bool))).astype(bool)
    one_hot = stored.astype(np.float32).max(axis=1) >= ONE_HOT_MIN
    rebuild = policy & has_root & ~tb
    kept_solved = policy & ~has_root & ~tb & one_hot
    kept_tb = policy & tb
    refused = policy & ~has_root & ~tb & ~one_hot
    if refused.any():
        rows = np.flatnonzero(refused)
        raise ValueError(f"{int(refused.sum())} policy rows have no root statistics and are neither a solved root "
                         f"(one-hot target) nor a tablebase override (first rows {rows[:5].tolist()}); their "
                         "target cannot be rebuilt, so the retarget is refused")
    out = stored.copy()
    max_change = 0.0
    for i in np.flatnonzero(rebuild):
        support = arrays["root_child_visits"][i] >= 0
        new = rebuild_improved_policy(log_prior=arrays["root_log_prior"][i][support],
                                      completed_q=arrays["root_child_q"][i][support],
                                      visits=arrays["root_child_visits"][i][support],
                                      net_value=float(arrays["root_net_value"][i]), cfg=cfg)
        row = np.zeros(stored.shape[1], dtype=np.float64)
        row[support] = new
        max_change = max(max_change, float(np.abs(row - stored[i].astype(np.float64)).max()))
        out[i] = row.astype(stored.dtype)
    counts = {"rows": n, "policy_rows": int(policy.sum()), "rebuilt": int(rebuild.sum()),
              "kept_solved_root": int(kept_solved.sum()), "kept_tablebase_override": int(kept_tb.sum()),
              "no_policy_untouched": int((~policy).sum()), "max_abs_target_change": max_change}
    return out, counts


def _check_source(source_run_dir: Path, generation: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    src = _gen(source_run_dir, generation) / "collect"
    manifest = read_manifest(src)
    if manifest is None:
        raise ValueError(f"retarget: {src} is not sealed (no manifest.json)")
    if RETARGET in manifest:
        raise ValueError(f"retarget: {src} is itself a retarget of {manifest[RETARGET]['source_collect_dir']}; "
                         "rebuild from the collect that played the games")
    if COLLECT_REFERENCE in manifest:
        raise ValueError(f"retarget: {src} is a reference collect; retarget the run that holds the rows "
                         f"({manifest[COLLECT_REFERENCE]['source_run_dir']})")
    for name, ref in sorted(manifest.get("outputs", {}).items()):
        p = src / Path(ref["path"]).name
        if not p.exists() or sha256_tree(p) != ref["sha256"]:
            raise ValueError(f"retarget: the source collect's {name} ({p}) no longer matches its sealed sha256")
    gen0 = _gen(source_run_dir, 0)
    gen0_manifest = read_manifest(gen0)
    if gen0_manifest is None or sha256_file(gen0 / "model.pt") != gen0_manifest["outputs"]["model"]["sha256"]:
        raise ValueError(f"retarget: the source's generation-0 model in {gen0} is not sealed or no longer matches")
    return src, manifest, gen0_manifest


def retarget_collect(*, source_run_dir: Path, generation: int, out_run_dir: Path,
                     target_q_rescale: bool = False) -> dict[str, Any]:
    """Write ``out_run_dir`` (must not exist) as a retargeted copy of the source
    run's generation-``generation`` collect. Returns the ``retarget`` record."""
    from chess_anti_engine.replay.shard import load_shard_arrays, save_local_shard_arrays

    source_run_dir = Path(source_run_dir).resolve()
    out_run_dir = Path(out_run_dir)
    if out_run_dir.exists():
        raise ValueError(f"retarget: {out_run_dir} exists; the retargeted collect is a new artifact, never an "
                         "overwrite")
    if out_run_dir.resolve() == source_run_dir:
        raise ValueError("retarget: the output is the source")
    src, manifest, gen0_manifest = _check_source(source_run_dir, generation)
    settings_rescale = manifest["config"]["settings"].get("target_q_rescale")
    if settings_rescale is not None and bool(settings_rescale) == bool(target_q_rescale):
        raise ValueError(f"retarget: the source collect's stored target already has target_q_rescale "
                         f"{settings_rescale}; nothing to rebuild")
    cfg = gumbel_cfg_for(manifest, target_q_rescale=target_q_rescale)
    tmp = out_run_dir.with_name(f".{out_run_dir.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        out_gen0 = _gen(tmp, 0)
        out_gen0.mkdir(parents=True)
        shutil.copy2(_gen(source_run_dir, 0) / "model.pt", out_gen0 / "model.pt")
        out_collect = _gen(tmp, generation) / "collect"
        (out_collect / "shards").mkdir(parents=True)
        totals: dict[str, Any] = {}
        for shard in sorted((src / "shards").glob("shard_*.zarr")):
            arrays, shard_meta = load_shard_arrays(shard, lazy=False, validate=True)
            meta_path = shard.with_name(shard.name.replace(".zarr", ".meta.npz"))
            with np.load(meta_path) as z:
                meta = {k: z[k] for k in z.files}
            arrays["policy_target"], counts = rebuild_rows(arrays, meta, cfg)
            save_local_shard_arrays(out_collect / "shards" / shard.name, arrs=arrays, meta=shard_meta)
            shutil.copy2(meta_path, out_collect / "shards" / meta_path.name)
            for k, v in counts.items():
                totals[k] = max(totals.get(k, 0.0), v) if k.startswith("max_") else totals.get(k, 0) + v
        for p in sorted(src.iterdir()):
            if p.name not in ("manifest.json", "shards"):
                (shutil.copytree if p.is_dir() else shutil.copy2)(p, out_collect / p.name)
        final_collect = _gen(out_run_dir, generation) / "collect"
        record = {"source_run_dir": str(source_run_dir), "source_collect_dir": str(src),
                  "source_manifest_sha256": sha256_file(src / "manifest.json"),
                  "source_shards_sha256": manifest["outputs"]["shards"]["sha256"],
                  "rebuild": "chess_anti_engine.mcts.gumbel_c.rebuild_improved_policy",
                  "target_q_rescale": bool(target_q_rescale), "rows": totals,
                  # Lineage: the games (and every array but policy_target) are the source's, played under
                  # the source's target (sampled from it at positive move temperature).
                  "games_played_with_target_q_rescale": settings_rescale,
                  "games_played_search_config": "config.dataclasses.search (the source's, unchanged)"}

        def ref(path_in_tmp: Path, final: Path) -> dict[str, Any]:
            return {"path": str(final), "sha256": sha256_tree(path_in_tmp)}

        # Seal INSIDE the partial dir (paths recorded at their final location), then publish with one
        # rename: a kill at any point leaves only the partial dir, which the next attempt removes.
        write_manifest(out_gen0, stage=str(gen0_manifest.get("stage", "gen0")),
                       config=gen0_manifest.get("config", {}),
                       inputs={**gen0_manifest.get("inputs", {}),
                               "source_gen0_manifest": input_ref(_gen(source_run_dir, 0) / "manifest.json")},
                       outputs={"model": ref(out_gen0 / "model.pt", _gen(out_run_dir, 0) / "model.pt")},
                       counts=gen0_manifest.get("counts", {}), seeds=gen0_manifest.get("seeds", {}),
                       extra={RETARGET: {"copied_from": str(_gen(source_run_dir, 0))}})
        inputs = dict(manifest["inputs"])
        if "checkpoint" in inputs and int(generation) == 1 \
                and inputs["checkpoint"].get("sha256") == sha256_file(out_gen0 / "model.pt"):
            inputs["checkpoint"] = ref(out_gen0 / "model.pt", _gen(out_run_dir, 0) / "model.pt")
        inputs["source_collect_manifest"] = input_ref(src / "manifest.json")
        outputs = {name: ref(out_collect / Path(r["path"]).name, final_collect / Path(r["path"]).name)
                   for name, r in sorted(manifest.get("outputs", {}).items())}
        if outputs["shards"]["sha256"] == record["source_shards_sha256"]:
            raise RuntimeError("retarget: the rebuilt shards hash like the source's; no target changed")
        # The settings state the TARGET truthfully: an arm referencing this collect must pass the same
        # target flag (reference.settings_diff), so its pinned run_config describes what it trains on.
        config = copy.deepcopy(manifest["config"])
        config["settings"]["target_q_rescale"] = bool(target_q_rescale)
        extra = {k: v for k, v in manifest.items()
                 if k not in ("schema", "stage", "written_unix", "config", "inputs", "outputs", "counts", "seeds",
                              "git", "runtime")}
        write_manifest(out_collect, stage="collect", config=config, inputs=inputs, outputs=outputs,
                       counts=manifest.get("counts", {}), seeds=manifest.get("seeds", {}),
                       extra={**extra, RETARGET: record})
        tmp.rename(out_run_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return record


def retarget_summary(record: dict[str, Any]) -> str:
    return json.dumps({RETARGET: record}, sort_keys=True)
