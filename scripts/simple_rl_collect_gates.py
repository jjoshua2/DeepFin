#!/usr/bin/env python3
"""Search-sanity gates of one sealed simple-RL collect, as JSON.

    PYTHONPATH=. python scripts/simple_rl_collect_gates.py <run>/gen_0001/collect \
        [--sample-rows 20000] [--seed 0] [--train-target-temp 1.5 ...] [--thresholds gates.json] [--out x.json]

Row metrics are computed on a UNIFORM seeded sample over every row of the
collect (all shards); game metrics on every game. A reference collect (an
arm's, simple_rl.reference) reads its source's rows.

Sides are by parity of plies after the book: side 0 is the BOOK-END MOVER (the
side to move in the position the opening line ends in, the first position the
net searches), side 1 the reply side. Roots are "fresh" when the search started
from 0 visits (``root_start_visits == 0``) and "reused" when tree reuse carried
visits in.

- ``spearman_childq_prior``: per row, Spearman rank correlation between the
  visited children's completed Q and their prior (``root_log_prior``) over
  children with visits > 0 (rows with >= 3 such children); mean per side x
  root kind. A healthy search ranks its prior's favourites well on both sides;
  the capped-evaluator defect (PR #1428) drove the reply side to ~0.
- ``book_end_mover_score``: over all games, the book-end mover's score (win 1,
  draw 0.5, loss 0). Its colour is white iff ``start_ply`` (``Board.ply()`` of the
  book-end position) is even.
- ``target_entropy``, ``kl_target_prior`` (over the prior's support,
  ``root_child_visits >= 0``), ``mean_survivor_minus_node_q`` and
  ``repeat_rate_within``: the collect's stats.json fields under the same names
  and definitions, recomputed over EVERY row (``whole_collect_stats``).
- ``*_by_side``: target entropy, KL and survivor - node Q on the sample, overall
  and per side.
- ``retempered_target_entropy``: target entropy after
  ``losses.retemper_main_policy_target`` at each ``--train-target-temp``.
- ``rows_per_game``, ``draw_rate``, ``decisive_rate``; from the manifest stats
  ``buffer_full_flushes`` and ``eval_max_batch_realised`` vs
  ``leaf_rows_required`` (``eval_max_batch_equals_leaf_rows``), with
  ``collect.capped_collect_record``.

``--content-digest``: print only ``content_digest`` -- a sha256 over the decoded
arrays of every shard plus the row sidecars, and games.jsonl's sha256 -- the
identity check for two collects that should hold the same rows.

``--thresholds FILE``: JSON ``{"dotted.metric.path": {"min": x, "max": y}}`` or
``{"dotted.metric.path": {"equals": v}}`` (bools and numbers; False / None /
missing fail ``{"equals": true}``); an empty bound is a presence gate (a finite
number, or True). Any gate that fails (or whose metric is missing / NaN) makes
the exit code 3.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.simple_rl.collect import GAMES, capped_collect_record
from chess_anti_engine.simple_rl.common import read_manifest, resolve_collect_dir

EXIT_GATE = 3
ROW_KEYS = ("policy_target", "has_policy", "root_child_q", "root_child_visits", "root_log_prior",
            "has_root_log_prior", "root_survivor_q", "root_node_q", "root_start_visits", "root_survivor_index")
SIDES = {"0": "book_end_mover", "1": "reply"}


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as Spearman's rho needs."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / counts[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a, b = _rank(np.asarray(x, np.float64)), _rank(np.asarray(y, np.float64))
    a -= a.mean()
    b -= b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else math.nan


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)


def retempered_entropy(target: np.ndarray, temp: float) -> np.ndarray:
    """Entropy of the stored target after the loss's own train-time re-temper."""
    import torch

    from chess_anti_engine.train.losses import retemper_main_policy_target

    t = torch.from_numpy(np.asarray(target, np.float32))
    return entropy(retemper_main_policy_target(t, temp=float(temp)).double().numpy())


def sample_rows(collect_dir: Path, *, n: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """A uniform seeded sample of ``n`` rows over every shard (all rows if fewer),
    loaded one shard at a time. Returns (arrays, row meta, total rows)."""
    from chess_anti_engine.replay.shard import load_shard_arrays

    shards = _shards(collect_dir)
    metas = []
    for s in shards:
        with np.load(s.with_name(s.name.replace(".zarr", ".meta.npz"))) as z:
            metas.append({k: z[k] for k in ("game_id", "ply_index")})
    sizes = np.asarray([len(m["game_id"]) for m in metas])
    total = int(sizes.sum())
    pick = np.sort(np.random.default_rng(seed).choice(total, size=min(n, total), replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    parts: list[dict[str, np.ndarray]] = []
    meta_parts: list[dict[str, np.ndarray]] = []
    for i, s in enumerate(shards):
        local = pick[(pick >= starts[i]) & (pick < starts[i + 1])] - starts[i]
        if not local.size:
            continue
        arrays, _ = load_shard_arrays(s, lazy=False, validate=True)
        parts.append({k: np.asarray(arrays[k])[local] for k in ROW_KEYS if k in arrays})
        meta_parts.append({k: v[local] for k, v in metas[i].items()})
    keys = set.intersection(*(set(p) for p in parts))
    return ({k: np.concatenate([p[k] for p in parts]) for k in keys},
            {k: np.concatenate([m[k] for m in meta_parts]) for k in ("game_id", "ply_index")}, total)


def _shards(collect_dir: Path) -> list[Path]:
    shards = sorted((resolve_collect_dir(collect_dir) / "shards").glob("shard_*.zarr"))
    if not shards:
        raise SystemExit(f"no shards under {resolve_collect_dir(collect_dir)}")
    return shards


def _kl_target_prior(target: np.ndarray, log_prior: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """Per-row KL(target || prior) over the prior's support (visits >= 0), as collect.policy_stats."""
    support = visits >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(support, log_prior, -np.inf)
        lp = lp - lp.max(axis=1, keepdims=True)
        prior = np.where(support, np.exp(lp), 0.0)
        prior = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-300)
        t = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-30)
        return np.where(t > 0, t * (np.log(np.where(t > 0, t, 1.0)) - np.log(np.where(prior > 0, prior, 1e-300))),
                        0.0).sum(axis=1)


def whole_collect_stats(collect_dir: Path) -> dict[str, Any]:
    """The collect's own stats.json fields, recomputed over EVERY row (one shard at a
    time) under the same definitions: ``target_entropy`` (policy rows),
    ``kl_target_prior`` (policy rows with root statistics), ``mean_survivor_minus_node_q``
    (rows with root statistics, NaN-skipping) and ``repeat_rate_within`` (1 - distinct
    input hashes / rows, from the row sidecars)."""
    from chess_anti_engine.replay.shard import load_shard_arrays

    ent_sum = kl_sum = gap_sum = 0.0
    ent_n = kl_n = gap_n = 0
    hashes: list[np.ndarray] = []
    for s in _shards(collect_dir):
        with np.load(s.with_name(s.name.replace(".zarr", ".meta.npz"))) as z:
            hashes.append(np.asarray(z["x_hash"], dtype=np.uint64))
        a, _ = load_shard_arrays(s, lazy=False, validate=True)
        target = np.asarray(a["policy_target"], np.float64)
        n = len(target)
        policy = np.asarray(a.get("has_policy", np.ones(n)), dtype=bool)
        stats = np.asarray(a.get("has_root_log_prior", np.zeros(n)), dtype=bool)
        e = entropy(target)[policy]
        ent_sum, ent_n = ent_sum + float(e.sum()), ent_n + int(e.size)
        sel = stats & policy
        if sel.any():
            kl = _kl_target_prior(target[sel], np.asarray(a["root_log_prior"], np.float64)[sel],
                                  np.asarray(a["root_child_visits"])[sel])
            kl_sum, kl_n = kl_sum + float(kl.sum()), kl_n + int(kl.size)
        if stats.any() and "root_survivor_q" in a and "root_node_q" in a:
            gap = (np.asarray(a["root_survivor_q"], np.float64) - np.asarray(a["root_node_q"], np.float64))[stats]
            gap = gap[np.isfinite(gap)]
            gap_sum, gap_n = gap_sum + float(gap.sum()), gap_n + int(gap.size)
    h = np.concatenate(hashes)
    return {"target_entropy": ent_sum / ent_n if ent_n else None,
            "kl_target_prior": kl_sum / kl_n if kl_n else None,
            "mean_survivor_minus_node_q": gap_sum / gap_n if gap_n else None,
            "repeat_rate_within": 1.0 - len(np.unique(h)) / max(1, len(h))}


def content_digest(collect_dir: Path) -> dict[str, Any]:
    """Identity of a collect's DECODED content: sha256 over every shard's arrays
    (sorted keys, raw bytes) and row sidecar, in shard order, plus games.jsonl's
    sha256. Two collects that made the same rows agree here even though their
    shard FILES differ (shard metadata carries a wall-clock timestamp and the
    compressed chunk files are not byte-stable)."""
    import hashlib

    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.simple_rl.common import sha256_file

    h = hashlib.sha256()
    rows = 0
    shards = _shards(collect_dir)
    for s in shards:
        a, _ = load_shard_arrays(s, lazy=False, validate=True)
        for k in sorted(a):
            v = np.ascontiguousarray(np.asarray(a[k]))
            h.update(f"{s.name}/{k}/{v.dtype.str}/{v.shape}".encode())
            h.update(v.tobytes())
        rows += int(np.asarray(a["x"]).shape[0])
        with np.load(s.with_name(s.name.replace(".zarr", ".meta.npz"))) as z:
            for k in sorted(z.files):
                v = np.ascontiguousarray(z[k])
                h.update(f"{s.name}/meta/{k}/{v.dtype.str}/{v.shape}".encode())
                h.update(v.tobytes())
    rows_dir = resolve_collect_dir(collect_dir)
    return {"collect_dir": str(collect_dir), "content_sha256": h.hexdigest(), "shards": len(shards), "rows": rows,
            "games_sha256": sha256_file(rows_dir / GAMES)}


def _mean(x: np.ndarray) -> float | None:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def _by_side(values: np.ndarray, side: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": _mean(values[mask]), "rows": int(mask.sum())}
    for k, name in SIDES.items():
        out[name] = _mean(values[mask & (side == int(k))])
    return out


def game_metrics(games: list[dict[str, Any]], rows: int) -> dict[str, Any]:
    results = [g["result"] for g in games]
    n = len(games)
    score = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}
    mover = [score[g["result"]][0 if int(g["start_ply"]) % 2 == 0 else 1] for g in games if g["result"] in score]
    white = [score[r][0] for r in results if r in score]
    return {"games": n, "rows": rows, "rows_per_game": rows / max(1, n),
            "draw_rate": results.count("1/2-1/2") / max(1, n),
            "decisive_rate": (results.count("1-0") + results.count("0-1")) / max(1, n),
            "unscored_games": n - len(mover),
            "book_end_mover_score": float(np.mean(mover)) if mover else None,
            "book_end_mover_white_share": float(np.mean([int(g["start_ply"]) % 2 == 0 for g in games])) if n else None,
            "white_score": float(np.mean(white)) if white else None}


def row_metrics(a: dict[str, np.ndarray], meta: dict[str, np.ndarray], start_ply: dict[int, int],
                temps: list[float]) -> dict[str, Any]:
    n = len(meta["game_id"])
    side = np.asarray([(int(p) - start_ply[int(g)]) % 2 for g, p in zip(meta["game_id"], meta["ply_index"],
                                                                         strict=True)], dtype=np.int64)
    policy = np.asarray(a.get("has_policy", np.ones(n)), dtype=bool)
    stats = np.asarray(a.get("has_root_log_prior", np.zeros(n)), dtype=bool)
    visits = np.asarray(a["root_child_visits"])
    child_q = np.asarray(a["root_child_q"], np.float64)
    log_prior = np.asarray(a["root_log_prior"], np.float64)
    start_visits = np.asarray(a.get("root_start_visits", np.zeros(n)))
    target = np.asarray(a["policy_target"], np.float64)

    rho = np.full(n, np.nan)
    survivor_is_argmax_q = np.full(n, np.nan)
    for i in np.flatnonzero(stats & policy):
        vis = visits[i] > 0
        if vis.sum() >= 3:
            rho[i] = spearman(child_q[i][vis], log_prior[i][vis])
        if "root_survivor_index" in a and vis.any():
            survivor_is_argmax_q[i] = float(int(np.argmax(np.where(vis, child_q[i], -np.inf)))
                                            == int(a["root_survivor_index"][i]))
    cells: dict[str, Any] = {}
    for k, name in SIDES.items():
        for kind, m in (("fresh", start_visits == 0), ("reused", start_visits > 0)):
            sel = stats & policy & m & (side == int(k))
            cells[f"{name}_{kind}"] = {"rows": int(sel.sum()), "rows_with_3_visited": int(np.isfinite(rho[sel]).sum()),
                                       "spearman_childq_prior": _mean(rho[sel]),
                                       "survivor_is_argmax_q": _mean(survivor_is_argmax_q[sel])}
    ent = entropy(target)
    kl = _kl_target_prior(target, log_prior, visits)
    gap = np.asarray(a["root_survivor_q"], np.float64) - np.asarray(a["root_node_q"], np.float64)
    out: dict[str, Any] = {
        "sampled_rows": n, "policy_rows": int(policy.sum()), "root_stats_rows": int(stats.sum()),
        "reused_root_share": _mean((start_visits > 0)[stats].astype(float)),
        "spearman_childq_prior": cells,
        "target_entropy_by_side": _by_side(ent, side, policy),
        "kl_target_prior_by_side": _by_side(kl, side, policy & stats),
        "survivor_minus_node_q_by_side": _by_side(gap, side, stats),
    }
    if temps:
        out["retempered_target_entropy"] = {
            f"{t:g}": _by_side(retempered_entropy(target, t), side, policy) for t in temps}
    return out


def collect_gates(collect_dir: Path, *, sample: int, seed: int, temps: list[float]) -> dict[str, Any]:
    collect_dir = Path(collect_dir)
    manifest = read_manifest(collect_dir)
    if manifest is None:
        raise SystemExit(f"{collect_dir} is not a sealed collect (no manifest.json)")
    rows_dir = resolve_collect_dir(collect_dir)
    games = [json.loads(r) for r in (rows_dir / GAMES).read_text().splitlines() if r.strip()]
    start_ply = {int(g["game_id"]): int(g["start_ply"]) for g in games}
    arrays, meta, total = sample_rows(collect_dir, n=sample, seed=seed)
    stats = manifest.get("stats") or {}
    known = stats.get("eval_max_batch_realised") is not None and stats.get("leaf_rows_required") is not None
    capped = capped_collect_record(manifest)
    return {
        "collect_dir": str(collect_dir), "rows_dir": str(rows_dir), "sample_seed": seed,
        "sample_rows_requested": sample,
        **game_metrics(games, total),
        **row_metrics(arrays, meta, start_ply, temps),
        "buffer_full_flushes": stats.get("buffer_full_flushes"),
        "eval_max_batch_realised": stats.get("eval_max_batch_realised"),
        "leaf_rows_required": stats.get("leaf_rows_required"),
        "eval_cap_covers_leaf_rows": (None if not known
                                      else int(stats["eval_max_batch_realised"]) >= int(stats["leaf_rows_required"])),
        # G-C2: the realised cap EQUALS the leaf rows one search call asks for (the auto default)
        "eval_max_batch_equals_leaf_rows": (None if not known
                                            else int(stats["eval_max_batch_realised"]) == int(stats["leaf_rows_required"])),
        "capped_collect": capped,
        # stats.json names, over every row (G-C5 / G-C6 / G-C9 / G-C10)
        **whole_collect_stats(collect_dir),
    }


def lookup(report: dict[str, Any], dotted: str) -> Any:
    cur: Any = report
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _gate_pass(value: Any, bound: dict[str, Any]) -> bool:
    """One gate. ``equals``: the metric must be present and equal (a bool only to
    a bool, a number to a number; False / None / missing fail ``{"equals": True}``).
    ``min`` / ``max``: the metric must be a finite number in range. An empty
    bound is a presence gate: a finite number, or the bool True."""
    if "equals" in bound:
        want = bound["equals"]
        if isinstance(want, bool) or isinstance(value, bool):
            return isinstance(value, bool) and isinstance(want, bool) and value is want
        if isinstance(want, (int, float)) and isinstance(value, (int, float)):
            return math.isfinite(float(value)) and float(value) == float(want)
        return value is not None and value == want
    if isinstance(value, bool):
        return value is True and not ({"min", "max"} & set(bound))
    ok = isinstance(value, (int, float)) and math.isfinite(float(value))
    if ok and "min" in bound:
        ok = float(value) >= float(bound["min"])
    if ok and "max" in bound:
        ok = float(value) <= float(bound["max"])
    return bool(ok)


def check_gates(report: dict[str, Any], thresholds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for metric, bound in sorted(thresholds.items()):
        unknown = set(bound) - {"min", "max", "equals"}
        if unknown:
            raise SystemExit(f"thresholds: {metric} has unknown keys {sorted(unknown)} (min / max / equals)")
        if "equals" in bound and {"min", "max"} & set(bound):
            raise SystemExit(f"thresholds: {metric}: equals excludes min / max")
        value = lookup(report, metric)
        out.append({"metric": metric, "value": value, **bound, "pass": _gate_pass(value, bound)})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("collect_dir")
    p.add_argument("--sample-rows", type=int, default=20000, help="uniform row sample size (all rows if fewer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-target-temp", type=float, action="append", default=[],
                   help="also report the target entropy re-tempered at this train-time temperature (repeatable)")
    p.add_argument("--thresholds", default=None, help="JSON {metric.path: {min, max} | {equals}}; exit 3 when one fails")
    p.add_argument("--out", default=None, help="also write the JSON here")
    p.add_argument("--content-digest", action="store_true",
                   help="only print the decoded-content digest (all shard arrays + row sidecars) and games.jsonl "
                        "sha256: equal for two collects that made the same rows (e.g. graphs on vs off)")
    a = p.parse_args(argv)
    if a.content_digest:
        print(json.dumps(content_digest(Path(a.collect_dir)), indent=1, sort_keys=True))
        return 0
    if a.sample_rows < 1:
        raise SystemExit("--sample-rows must be >= 1")
    report = collect_gates(Path(a.collect_dir), sample=a.sample_rows, seed=a.seed, temps=list(a.train_target_temp))
    code = 0
    if a.thresholds:
        gates = check_gates(report, json.loads(Path(a.thresholds).read_text()))
        report["gates"] = gates
        report["gates_passed"] = all(g["pass"] for g in gates)
        code = 0 if report["gates_passed"] else EXIT_GATE
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
