#!/usr/bin/env python3
"""Deep-SF LABEL ruler for gen-0 shard cells (task #273, the 2-arm readout).

The existing rulers score a NET (``scripts/value_regret.py``,
``scripts/audit_targets.py``) against the frozen audit set. This one scores the
LABELS a generator wrote into shards, on the generator's OWN positions, against
unhandicapped deep Stockfish -- which is what the #273 prereg's G2 gate needs:
"native cell's label quality >= UCI@512's on the same ruler".

Procedure, identical for every cell (native or banked UCI anchor):

1. Enumerate every row of the cell: shards sorted by NAME, rows in file order.
2. Draw ``--positions`` row ordinals without replacement from one RNG seeded by
   ``--sample-seed``. Same seed, same procedure, every cell.
3. Rebuild each row's position from its stored input planes
   (``eval/audit.decode_board_from_planes``) -- side-to-move CANONICAL, i.e.
   white to move, mirrored when black was to move.
4. CROSS-CHECK the rebuild: the compact-1858 indices of the rebuilt board's
   legal moves must equal the row's own stored ``legal_mask``. A cell whose
   agreement is not ~100% is not being read on its own positions and its number
   is void. Reported, never assumed.
5. Deep-SF label the position: ``go nodes <--nodes>`` at ``MultiPV <--multipv>``,
   ALWAYS with a cleared transposition table (``fresh=True``), so a label is a
   pure function of (fen, nodes, multipv, binary) and cannot depend on the order
   the cell happened to sample. Labels are CACHED by position key and the cache
   is SHARED across every cell in one invocation, so a position two cells both
   contain is scored against the identical ruler row.
   NOTE: ``scripts/build_audit_set.py`` labels with a WARM TT (no ``fresh``), so
   numbers from this ruler are NOT comparable with frozen-audit-set numbers.
6. Score the row's stored ``policy_target`` against the deep labels with the
   shared ``eval/audit`` scorers (1000cp regret cap).

PRIMARY metric: ``top1_regret_cp`` -- the deep-SF regret of the move the stored
policy target ranks first. For a FIXED search tree it is invariant under any
monotone re-sharpening of the target, which is the first-order effect of the
pinned ``--nnue-cp-per-unit``; it does NOT fully neutralise that knob, because
the leaf map is non-linear and the tree averages over it, so the knob also moves
sequential halving's own decisions. Expected regret is reported but is NOT the
primary: it is linear in cp and therefore rewards a cp-softmax target by
construction (eval/audit.py::expected_blunder_rates documents the measurement).

⚑ The stored ``wdl_target`` is the GAME OUTCOME, not an arm-derived value label,
so this ruler scores POLICY labels. ``value_agrees`` is reported as a property of
the games, never as a reading of an arm's value quality.

CIs are cluster bootstraps over GAMES (``game_id``), never over rows: rows
inside one game are heavily correlated and a row bootstrap would report a
resolution the instrument does not have.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from chess_anti_engine.eval.audit import (
    AUDIT_REGRET_CAP_CP,
    CRITICALITY_BUCKET_NAMES,
    PHASE_NAMES,
    AuditPosition,
    criticality_bucket,
    criticality_gap,
    decode_board_from_planes,
    expected_and_top1_regret,
    expected_blunder_rates,
    move_regrets,
    phase_bucket,
    position_key,
)
from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY, uci_to_policy_index
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    is_tmp_shard_name,
    load_shard_arrays,
)
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from chess_anti_engine.utils.git_meta import git_sha

BLUNDER_TAUS: tuple[float, ...] = (100.0, 300.0)

# full-4672 index -> compact-1858 index. The shards store the compact vector, so
# a legal move's column is found by encoding it in full space and mapping down.
_FULL_TO_COMPACT: dict[int, int] = {
    int(full): compact for compact, full in enumerate(COMPACT_TO_FULL_POLICY)
}


def shard_paths(cell_dir: Path) -> list[Path]:
    """Every real shard of a cell, in NAME order (the sampling order)."""
    return sorted(
        p for p in cell_dir.glob(f"*{LOCAL_SHARD_SUFFIX}")
        if not is_tmp_shard_name(p.name)
    )


def _compact_indices(board) -> tuple[list[str], list[int]]:
    """(uci, compact-1858 index) for every legal move of a CANONICAL board."""
    ucis: list[str] = []
    idxs: list[int] = []
    for mv in board.legal_moves:
        uci = mv.uci()
        full = uci_to_policy_index(uci, True)  # canonical board: white to move
        compact = _FULL_TO_COMPACT.get(int(full), -1)
        if compact >= 0:
            ucis.append(uci)
            idxs.append(compact)
    return ucis, idxs


class DeepLabeler:
    """Deep-SF MultiPV labels, cached by position key, shared across cells."""

    def __init__(
        self, *, path: str, nodes: int, multipv: int, hash_mb: int, workers: int,
        nice: int,
    ) -> None:
        self.path, self.nodes, self.multipv = path, int(nodes), int(multipv)
        self.hash_mb, self.workers, self.nice = int(hash_mb), int(workers), int(nice)
        self._cache: dict[str, dict | None] = {}
        self._lock = threading.Lock()
        self.n_searched = 0
        self.n_cache_hits = 0

    def _label_one(self, eng: StockfishUCI, fen: str) -> dict | None:
        res = eng.search(fen, fresh=True)
        move_cp: dict[str, float] = {}
        for pv in res.pvs or []:
            if pv.mate:
                eff = float(mate_to_effective_cp(int(pv.mate)))
            elif pv.cp is not None:
                eff = float(pv.cp)
            else:
                continue
            move_cp.setdefault(str(pv.move_uci), eff)
        if not move_cp and res.bestmove_uci:
            if res.mate:
                move_cp[str(res.bestmove_uci)] = float(mate_to_effective_cp(int(res.mate)))
            elif res.cp is not None:
                move_cp[str(res.bestmove_uci)] = float(res.cp)
        if not move_cp:
            return None
        wdl = res.wdl if res.wdl is not None else [333.0, 334.0, 333.0]
        total = max(1e-9, sum(float(v) for v in wdl))
        return {
            "move_cp": move_cp,
            "wdl": [float(v) / total for v in wdl],
            "bestmove": res.bestmove_uci,
            "nodes": int(res.nodes or self.nodes),
            "depth": int(res.depth or 0),
            "n_pvs": len(res.pvs or []),
        }

    def label_all(self, keys_fens: list[tuple[str, str]]) -> None:
        """Populate the cache for every (key, fen) not already present."""
        todo: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, fen in keys_fens:
            if key in self._cache or key in seen:
                self.n_cache_hits += 1
                continue
            seen.add(key)
            todo.append((key, fen))
        if not todo:
            return
        chunks: list[list[tuple[str, str]]] = [[] for _ in range(self.workers)]
        for i, item in enumerate(todo):
            chunks[i % self.workers].append(item)
        t0 = time.time()
        done = [0]

        def run(chunk: list[tuple[str, str]]) -> None:
            eng = StockfishUCI(
                self.path, nodes=self.nodes, multipv=self.multipv,
                hash_mb=self.hash_mb, nice=self.nice,
            )
            try:
                for key, fen in chunk:
                    rec = self._label_one(eng, fen)
                    with self._lock:
                        self._cache[key] = rec
                        self.n_searched += 1
                        done[0] += 1
                        if done[0] % 500 == 0:
                            el = time.time() - t0
                            print(
                                f"  [sf] {done[0]}/{len(todo)} labelled "
                                f"({el:.0f}s, {done[0] / max(el, 1e-9):.2f} pos/s)",
                                flush=True,
                            )
            finally:
                eng.close()

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(run, chunks))

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)


def gather_cell_rows(
    cell_dir: Path, *, positions: int, sample_seed: int,
) -> tuple[list[dict], dict]:
    """Deterministic uniform row sample from a cell, decoded to boards."""
    paths = shard_paths(cell_dir)
    if not paths:
        raise SystemExit(f"no shards under {cell_dir}")
    counts: list[int] = []
    for p in paths:
        arrs, _ = load_shard_arrays(p, lazy=True)
        counts.append(int(arrs["x"].shape[0]))
    total = int(sum(counts))
    rng = np.random.default_rng(int(sample_seed))
    take = min(int(positions), total)
    picked = np.sort(rng.permutation(total)[:take])
    bounds = np.cumsum([0, *counts])

    rows: list[dict] = []
    stats = {
        "cell_rows_total": total,
        "cell_shards": len(paths),
        "sampled": int(take),
        "excluded_no_policy": 0,
        "excluded_undecodable": 0,
        "excluded_terminal": 0,
        "legal_mask_agree": 0,
        "legal_mask_disagree": 0,
    }
    for si, p in enumerate(paths):
        lo, hi = int(bounds[si]), int(bounds[si + 1])
        local = picked[(picked >= lo) & (picked < hi)] - lo
        if local.size == 0:
            continue
        arrs, _ = load_shard_arrays(p, lazy=True)
        hist = str(np.asarray(arrs["_input_history_encoding"]).reshape(-1)[0])
        xs = np.asarray(arrs["x"][:], dtype=np.float32)
        pol = np.asarray(arrs["policy_target"][:], dtype=np.float32)
        lmask = np.asarray(arrs["legal_mask"][:]).astype(bool)
        haspol = np.asarray(arrs["has_policy"][:]).astype(bool)
        wdl_t = np.asarray(arrs["wdl_target"][:]).astype(np.int64)
        gid = np.asarray(arrs["game_id"][:]).astype(np.int64)
        ply = np.asarray(arrs["ply_index"][:]).astype(np.int64)
        for j in local.tolist():
            j = int(j)
            if not haspol[j]:
                stats["excluded_no_policy"] += 1
                continue
            board = decode_board_from_planes(xs[j], input_history_encoding=hist)
            if board is None:
                stats["excluded_undecodable"] += 1
                continue
            if board.is_game_over():
                stats["excluded_terminal"] += 1
                continue
            ucis, idxs = _compact_indices(board)
            stored = set(np.flatnonzero(lmask[j]).tolist())
            if set(idxs) == stored:
                stats["legal_mask_agree"] += 1
            else:
                stats["legal_mask_disagree"] += 1
            fen = board.fen()
            rows.append({
                "key": position_key(board),
                "fen": fen,
                "ucis": ucis,
                "probs": pol[j][np.asarray(idxs, dtype=np.int64)].astype(np.float64),
                "wdl_target": int(wdl_t[j]),
                "game_id": int(gid[j]),
                "ply_index": int(ply[j]),
                "phase": phase_bucket(
                    sum(1 for c in fen.split(" ", 1)[0] if c.isalpha()),
                ),
                "shard": p.name,
            })
    return rows, stats


def score_cell(rows: list[dict], labeler: DeepLabeler) -> tuple[list[dict], dict]:
    """Per-row deep-SF scores for one cell's sampled rows."""
    out: list[dict] = []
    unscoreable = 0
    for r in rows:
        rec = labeler.get(r["key"])
        if rec is None:
            unscoreable += 1
            continue
        pos = AuditPosition(
            key=r["key"], fen=r["fen"], phase=int(r["phase"]), source=0,
            move_cp=rec["move_cp"], best_cp=max(rec["move_cp"].values()),
            deep_wdl=(rec["wdl"][0], rec["wdl"][1], rec["wdl"][2]),
            sf_nodes=int(rec["nodes"]), sf_depth=int(rec["depth"]),
        )
        regrets = move_regrets(pos, r["ucis"])
        probs = np.asarray(r["probs"], dtype=np.float64)
        exp_r, top1_r = expected_and_top1_regret(probs, regrets)
        blunders = expected_blunder_rates(probs, regrets, BLUNDER_TAUS)
        gap = criticality_gap(rec["move_cp"])
        # deep-SF WDL argmax in the shard's own 0=W/1=D/2=L convention
        deep_cls = int(np.argmax(np.asarray(rec["wdl"], dtype=np.float64)))
        out.append({
            "key": r["key"], "game_id": r["game_id"], "ply_index": r["ply_index"],
            "phase": int(r["phase"]), "shard": r["shard"],
            "top1_regret_cp": float(top1_r),
            "expected_regret_cp": float(exp_r),
            "blunder_100": float(blunders[0]),
            "blunder_300": float(blunders[1]),
            "top1_is_sf_best": float(
                r["ucis"][int(np.argmax(probs))] == str(rec["bestmove"]),
            ),
            "value_agrees": float(int(r["wdl_target"]) == deep_cls),
            "criticality": int(criticality_bucket(gap)),
            "n_legal": len(r["ucis"]),
            "sf_depth": int(rec["depth"]),
            "n_pvs": int(rec["n_pvs"]),
        })
    return out, {"unscoreable_positions": unscoreable}


def cluster_bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int,
) -> tuple[float, float]:
    """95% CI of the mean, resampling GAMES (clusters), not rows."""
    uniq = np.unique(groups)
    by_group = [values[groups == g] for g in uniq]
    sums = np.array([float(v.sum()) for v in by_group])
    cnts = np.array([float(v.size) for v in by_group])
    rng = np.random.default_rng(int(seed))
    n = len(uniq)
    if n < 2:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, n, size=(int(n_boot), n))
    boot = sums[draws].sum(axis=1) / np.maximum(cnts[draws].sum(axis=1), 1e-9)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def summarize(scored: list[dict], *, n_boot: int, seed: int) -> dict:
    if not scored:
        return {"n": 0}
    arr = {k: np.array([float(s[k]) for s in scored]) for k in (
        "top1_regret_cp", "expected_regret_cp", "blunder_100", "blunder_300",
        "top1_is_sf_best", "value_agrees", "sf_depth", "n_legal",
    )}
    groups = np.array([int(s["game_id"]) for s in scored])
    phases = np.array([int(s["phase"]) for s in scored])
    crit = np.array([int(s["criticality"]) for s in scored])
    out: dict = {
        "n": len(scored),
        "n_games": int(np.unique(groups).size),
        "n_unique_positions": len({s["key"] for s in scored}),
        "sf_depth_mean": float(arr["sf_depth"].mean()),
        "n_legal_mean": float(arr["n_legal"].mean()),
    }
    for name in (
        "top1_regret_cp", "expected_regret_cp", "blunder_100", "blunder_300",
        "top1_is_sf_best", "value_agrees",
    ):
        v = arr[name]
        lo, hi = cluster_bootstrap_ci(v, groups, n_boot=n_boot, seed=seed)
        out[name] = {
            "mean": float(v.mean()), "ci95": [lo, hi],
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
        }
    t1 = arr["top1_regret_cp"]
    out["top1_regret_cp"]["gt100_pct"] = float(100.0 * (t1 > 100).mean())
    out["top1_regret_cp"]["gt300_pct"] = float(100.0 * (t1 > 300).mean())
    out["by_phase"] = {
        PHASE_NAMES[ph]: {
            "n": int((phases == ph).sum()),
            "top1_regret_cp": float(t1[phases == ph].mean()),
        }
        for ph in range(3) if (phases == ph).any()
    }
    out["by_criticality"] = {
        CRITICALITY_BUCKET_NAMES[b]: {
            "n": int((crit == b).sum()),
            "top1_regret_cp": float(t1[crit == b].mean()),
        }
        for b in range(len(CRITICALITY_BUCKET_NAMES)) if (crit == b).any()
    }
    return out


def paired_delta(a: list[dict], b: list[dict], *, n_boot: int, seed: int) -> dict:
    """Unpaired difference of means A-B with a cluster-bootstrap CI on each."""
    va = np.array([float(s["top1_regret_cp"]) for s in a])
    vb = np.array([float(s["top1_regret_cp"]) for s in b])
    ga = np.array([int(s["game_id"]) for s in a])
    gb = np.array([int(s["game_id"]) for s in b])
    ua, ub = np.unique(ga), np.unique(gb)
    sa = np.array([float(va[ga == g].sum()) for g in ua])
    ca = np.array([float((ga == g).sum()) for g in ua])
    sb = np.array([float(vb[gb == g].sum()) for g in ub])
    cb = np.array([float((gb == g).sum()) for g in ub])
    rng = np.random.default_rng(int(seed))
    da = rng.integers(0, len(ua), size=(int(n_boot), len(ua)))
    db = rng.integers(0, len(ub), size=(int(n_boot), len(ub)))
    ma = sa[da].sum(axis=1) / np.maximum(ca[da].sum(axis=1), 1e-9)
    mb = sb[db].sum(axis=1) / np.maximum(cb[db].sum(axis=1), 1e-9)
    d = ma - mb
    return {
        "delta_mean": float(va.mean() - vb.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "se": float(d.std(ddof=1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--cell", action="append", required=True, metavar="LABEL=DIR",
        help="a cell to score; repeatable. All cells share ONE label cache.",
    )
    ap.add_argument("--positions", type=int, default=5000)
    ap.add_argument("--sample-seed", type=int, default=20260825)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--multipv", type=int, default=10)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--sf-workers", type=int, default=8)
    ap.add_argument("--sf-nice", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--dump-rows", type=Path, default=None)
    args = ap.parse_args()

    if args.multipv < 10:
        raise SystemExit("--multipv must be >= 10 (the project's audit standard)")
    if args.nodes < 1_000_000:
        raise SystemExit("--nodes must be >= 1,000,000 (the project's audit standard)")

    cells: list[tuple[str, Path]] = []
    for spec in args.cell:
        label, _, d = str(spec).partition("=")
        if not d:
            raise SystemExit(f"--cell wants LABEL=DIR, got {spec!r}")
        cells.append((label, Path(d)))

    ruler = {
        "nodes": int(args.nodes), "multipv": int(args.multipv),
        "hash_mb": int(args.hash_mb), "threads_per_engine": 1,
        "tt": "fresh (ucinewgame) before EVERY position",
        "syzygy": None,
        "stockfish": str(args.stockfish),
        "regret_cap_cp": float(AUDIT_REGRET_CAP_CP),
        "sample_seed": int(args.sample_seed),
        "positions_per_cell": int(args.positions),
        "primary_metric": "top1_regret_cp",
        "ruler_git_sha": git_sha(),
    }
    print(f"[ruler] {json.dumps(ruler)}", flush=True)

    labeler = DeepLabeler(
        path=args.stockfish, nodes=args.nodes, multipv=args.multipv,
        hash_mb=args.hash_mb, workers=args.sf_workers, nice=args.sf_nice,
    )

    gathered: dict[str, tuple[list[dict], dict]] = {}
    for label, d in cells:
        rows, stats = gather_cell_rows(
            d, positions=args.positions, sample_seed=args.sample_seed,
        )
        gathered[label] = (rows, stats)
        print(
            f"[cell {label}] {d}: {stats['sampled']} sampled of "
            f"{stats['cell_rows_total']} rows in {stats['cell_shards']} shards; "
            f"decoded {len(rows)}; legal-mask agree "
            f"{stats['legal_mask_agree']}/"
            f"{stats['legal_mask_agree'] + stats['legal_mask_disagree']}; "
            f"excluded no_policy={stats['excluded_no_policy']} "
            f"undecodable={stats['excluded_undecodable']} "
            f"terminal={stats['excluded_terminal']}",
            flush=True,
        )

    all_keys: list[tuple[str, str]] = []
    for label, _ in cells:
        all_keys.extend((r["key"], r["fen"]) for r in gathered[label][0])
    uniq = len({k for k, _ in all_keys})
    print(
        f"[sf] {len(all_keys)} row-positions, {uniq} unique -> labelling with "
        f"{args.sf_workers} workers at nodes={args.nodes} multipv={args.multipv}",
        flush=True,
    )
    t0 = time.time()
    labeler.label_all(all_keys)
    print(
        f"[sf] {labeler.n_searched} searched, {labeler.n_cache_hits} cache hits, "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )

    report: dict = {"ruler": ruler, "cells": {}}
    scored_by_cell: dict[str, list[dict]] = {}
    with contextlib.ExitStack() as stack:
        dump = (
            stack.enter_context(open(args.dump_rows, "w", encoding="utf-8"))
            if args.dump_rows else None
        )
        for label, d in cells:
            rows, stats = gathered[label]
            scored, sstats = score_cell(rows, labeler)
            scored_by_cell[label] = scored
            summary = summarize(
                scored, n_boot=args.bootstrap, seed=args.bootstrap_seed,
            )
            report["cells"][label] = {
                "dir": str(d), "sampling": stats, **sstats, "summary": summary,
            }
            if dump is not None:
                for s in scored:
                    dump.write(json.dumps({"cell": label, **s}) + "\n")
            m = summary.get("top1_regret_cp", {})
            print(
                f"\n=== {label} === n={summary.get('n')} games="
                f"{summary.get('n_games')} unique_pos="
                f"{summary.get('n_unique_positions')}\n"
                f"  PRIMARY top1_regret_cp mean {m.get('mean', float('nan')):.2f} "
                f"[{m.get('ci95', [float('nan')] * 2)[0]:.2f}, "
                f"{m.get('ci95', [float('nan')] * 2)[1]:.2f}] "
                f"med {m.get('median', float('nan')):.1f} "
                f"P90 {m.get('p90', float('nan')):.1f} "
                f">100cp {m.get('gt100_pct', float('nan')):.1f}% "
                f">300cp {m.get('gt300_pct', float('nan')):.1f}%\n"
                f"  expected_regret_cp "
                f"{summary['expected_regret_cp']['mean']:.2f}  "
                f"blunder@100 {summary['blunder_100']['mean']:.4f}  "
                f"blunder@300 {summary['blunder_300']['mean']:.4f}\n"
                f"  top1_is_sf_best {summary['top1_is_sf_best']['mean']:.4f}  "
                f"value_agrees {summary['value_agrees']['mean']:.4f}  "
                f"sf_depth_mean {summary['sf_depth_mean']:.1f}  "
                f"n_legal_mean {summary['n_legal_mean']:.1f}",
                flush=True,
            )

    labels = [label for label, _ in cells]
    report["deltas"] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            if not scored_by_cell[a] or not scored_by_cell[b]:
                continue
            dl = paired_delta(
                scored_by_cell[a], scored_by_cell[b],
                n_boot=args.bootstrap, seed=args.bootstrap_seed,
            )
            report["deltas"][f"{a}-minus-{b}"] = dl
            print(
                f"[delta] {a} - {b}: top1_regret_cp "
                f"{dl['delta_mean']:+.2f} [{dl['ci95'][0]:+.2f}, "
                f"{dl['ci95'][1]:+.2f}] (bootstrap SE {dl['se']:.2f})",
                flush=True,
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[out] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
