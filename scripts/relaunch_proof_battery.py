#!/usr/bin/env python3
"""First-row deploy-proof battery for the reduced-SF relaunch bundle.

One PASS/FAIL line per bundle change, judged on the strongest available
observable — fresh-shard evidence where the change shapes records, launch
config + first result row where it does not. Run AFTER the restarted run has
ingested at least one new shard:

    PYTHONPATH=. python3 scripts/relaunch_proof_battery.py \
        --trial-dir runs/pbt2_small/tune/<trial> \
        --shard-dir <trial>/replay_shards \
        --server-url http://127.0.0.1:8763   # optional, for the #161 check

Expected values live in BUNDLE below — edit them together with the prereg, not
ad hoc. A config key that is accepted but never takes effect is this codebase's
signature defect (see CLAUDE.md), which is why every check here prefers the
DOWNSTREAM observable (what workers actually wrote) over the yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Bundle of record (mirror of the relaunch prereg; keep in sync) ────────────
BUNDLE: dict[str, float | None] = {
    "sf_multipv": 6,
    "sf_label_nodes_floor": 150_000,  # user-locked 2026-08-06 (drafts said 100k/150k)
    "sf_label_nodes_cap": 200_000,
    "selfplay_fraction": 0.8,
    # Deliberately ABSENT from the yaml: rides the GameConfig dataclass default,
    # which equals the desired value (stockfish_turn.py:554).
    "sf_fast_ply_node_scale": None,
    "gumbel_c_scale": 0.025,
    "gumbel_topk": 32,
    "mcts_simulations": 256,
    "gumbel_scale": 1.0,
    "gumbel_scale_after": 0.5,
    "train_views_per_position": 4.0,
}
def _num(key: str) -> float:
    """A BUNDLE value that must be numeric. `sf_fast_ply_node_scale` is
    deliberately None (rides the dataclass default), so the dict is
    `float | None` and every ARITHMETIC use has to narrow it first."""
    want = BUNDLE[key]
    if want is None:
        raise KeyError(f"BUNDLE[{key!r}] is None; it has no numeric comparison")
    return float(want)


OLD_WORKER_PASSWORD = "chess_worker_2026"  # must be DEAD after #161 rotation

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def newest_shards(shard_dir: Path, n: int = 3) -> list[Path]:
    paths = sorted(shard_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime)
    return paths[-n:]


def shard_checks(shard_dir: Path) -> None:
    import zarr

    paths = newest_shards(shard_dir)
    if not paths:
        check("shards-present", False, f"no .zarr shards under {shard_dir}")
        return
    ages = [(p.name, os.path.getmtime(p)) for p in paths]
    print(f"  using newest shards: {[a for a, _ in ages]}")

    pv_counts, label_nodes, sp_flags = [], [], []
    for p in paths:
        z = zarr.open(str(p), mode="r")
        has_raw = np.asarray(z["has_sf_multipv_raw"]).astype(bool)
        if has_raw.any():
            raw = np.asarray(z["sf_multipv_raw"][np.where(has_raw)[0][:200].tolist()])
            pv_counts.append((raw[:, :, 0] >= 0).sum(axis=1))
        has_meta = np.asarray(z["has_sf_label_meta"]).astype(bool)
        if has_meta.any():
            meta = np.asarray(z["sf_label_meta"][np.where(has_meta)[0][:500].tolist()])
            nodes = meta[:, 0]
            label_nodes.append(nodes[nodes > 0])
        sp_flags.append(np.asarray(z["is_selfplay"]).astype(bool))

    if pv_counts:
        pv = np.concatenate(pv_counts)
        p95 = float(np.percentile(pv, 95))
        check(
            "sf_multipv (shard evidence)",
            p95 <= _num("sf_multipv") + 0.5,
            f"non-pad PV rows p95={p95:.0f} median={np.median(pv):.0f} "
            f"(expect <= {BUNDLE['sf_multipv']}; 40 = old width still live)",
        )
    else:
        check("sf_multipv (shard evidence)", False, "no rows with sf_multipv_raw")

    if label_nodes:
        ln = np.concatenate(label_nodes)
        med = float(np.median(ln))
        lo = _num("sf_label_nodes_floor") * 0.9
        hi = _num("sf_label_nodes_cap") * 1.15  # SF overshoots a nodes stop slightly
        check(
            "sf_label_nodes floor+cap (shard evidence)",
            lo <= med <= hi,
            f"label nodes median={med:,.0f} p10={np.percentile(ln, 10):,.0f} "
            f"p90={np.percentile(ln, 90):,.0f} (expect median in [{lo:,.0f}, {hi:,.0f}]; "
            f"~700k = floor unchanged, way above cap = cap not wired)",
        )
    else:
        check("sf_label_nodes floor+cap (shard evidence)", False, "no sf_label_meta rows")

    sp = np.concatenate(sp_flags)
    check(
        "selfplay_fraction (shard evidence)",
        abs(float(sp.mean()) - _num("selfplay_fraction")) < 0.1,
        f"is_selfplay mean={sp.mean():.3f} over {sp.size} rows "
        f"(expect ~{BUNDLE['selfplay_fraction']}; NOTE small drain-transient bias "
        f"right after restart — re-run on later shards if borderline)",
    )


def config_checks(trial_dir: Path) -> None:
    params_path = trial_dir / "params.json"
    if not params_path.exists():
        check("params.json", False, f"missing {params_path}")
        return
    params = json.loads(params_path.read_text())
    # params.json is the LAUNCH config — necessary, not sufficient. The shard
    # checks above are the sufficiency half for record-shaping keys; for
    # search-shape keys (restart-keyed, consumed by workers at session start)
    # the launch config IS the session value as long as no live-yaml edit
    # happened since, which the first-row window guarantees.
    for key in (
        "gumbel_c_scale", "gumbel_topk", "mcts_simulations",
        "gumbel_scale", "gumbel_scale_after",
        "sf_fast_ply_node_scale", "sf_multipv",
        "sf_label_nodes_floor", "sf_label_nodes_cap", "selfplay_fraction",
    ):
        got = params.get(key)
        want = BUNDLE[key]
        if want is None:
            ok = got is None
            check(f"launch config {key}", ok,
                  f"got {got}, want ABSENT (dataclass default is the desired value)")
        else:
            ok = got is not None and abs(float(got) - float(want)) < 1e-9
            check(f"launch config {key}", ok, f"got {got}, want {want}")

    result_path = trial_dir / "result.json"
    if result_path.exists():
        last = None
        with result_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last:
            row = json.loads(last)
            steps = row.get("train_steps_used")
            views = row.get("train_views_actual")
            check(
                "first-row training telemetry",
                steps is not None,
                f"train_steps_used={steps} train_views_actual={views} "
                f"iter={row.get('training_iteration')} "
                f"(read train_steps_used, NOT views_actual, for step spikes)",
            )
    else:
        check("result.json", False, "no result rows yet — re-run after first iteration")


def credential_check(server_url: str) -> None:
    import base64
    import urllib.error
    import urllib.request

    # HTTP Basic against a real authenticated route (POST /v1/lease_trial,
    # app.py:2406). Anything but 401/403 fails the check: 2xx = burned password
    # still accepted; 404/422 = wrong route/shape, i.e. no auth verdict at all.
    token = base64.b64encode(f"josh:{OLD_WORKER_PASSWORD}".encode()).decode()
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/lease_trial",
        method="POST",
        headers={"Authorization": f"Basic {token}",
                 "Content-Type": "application/json"},
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            check("#161 old credential DEAD", False,
                  f"old password still accepted (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        check("#161 old credential DEAD", e.code in (401, 403),
              f"server answered HTTP {e.code} to the burned password "
              f"(401/403 = rejected; anything else = no auth verdict)")
    except OSError as e:
        check("#161 old credential DEAD", False, f"server unreachable: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--server-url", type=str, default=None)
    args = ap.parse_args()

    shard_checks(args.shard_dir)
    config_checks(args.trial_dir)
    if args.server_url:
        credential_check(args.server_url)
    else:
        print("[SKIP] #161 credential check (no --server-url)")

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
