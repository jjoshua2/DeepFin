#!/usr/bin/env python3
"""Build the frozen deep-SF audit set (data/audit_set_v1.jsonl).

Two stages, both resumable:

1. SAMPLE: scan recent replay shards, reconstruct positions from the stored
   input planes (side-to-move canonical — see eval/audit.py), stratify by
   game phase (piece-count buckets, same thresholds as the probe/model) and
   by source (selfplay vs curriculum), dedup by position key, and write a
   manifest (<out>.manifest.jsonl). The manifest is created ONCE; reruns
   reuse it.

2. LABEL: for every manifest position not yet present in the output JSONL,
   run UNHANDICAPPED Stockfish at --nodes (>=1M) with --multipv (>=10) and
   append one record per position (flush per record, so a kill at any point
   loses at most the in-flight positions). CPU-only; run it niced next to
   training: it never touches the GPU.

The set is FROZEN after generation: the script refuses to run when the
output exists unless --resume is passed, and --resume only ever APPENDS
missing labels — existing records are never rewritten. A README.md with the
generation config + git SHA is written alongside, marked FROZEN once every
manifest row is labeled.

Usage::

    PYTHONPATH=. nice -n 15 python3 scripts/build_audit_set.py \
        --replay-dir runs/pbt2_small/<trial>/replay_shards \
        --stockfish /usr/games/stockfish --sf-workers 8 \
        [--resume]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from chess_anti_engine.eval.audit import (
    PHASE_NAMES,
    SOURCE_NAMES,
    decode_board_from_planes,
    phase_bucket,
    position_key,
)
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    is_tmp_shard_name,
    load_shard_arrays,
)
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.utils.git_meta import git_sha


def _iter_shard_paths(replay_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for d in replay_dirs:
        paths.extend(
            p for p in d.glob(f"*{LOCAL_SHARD_SUFFIX}") if not is_tmp_shard_name(p.name)
        )
    # Newest first: the audit should reflect the position distribution the
    # CURRENT model is producing, not the bootstrap era.
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _sample_manifest(
    replay_dirs: list[Path], *, positions: int, seed: int,
) -> list[dict]:
    """Stratified position sample: 3 phases x 2 sources, deduped."""
    rng = np.random.default_rng(seed)
    per_stratum = positions // 6
    strata: dict[tuple[int, int], list[dict]] = {
        (ph, src): [] for ph in range(3) for src in range(2)
    }
    seen: set[str] = set()
    paths = _iter_shard_paths(replay_dirs)
    if not paths:
        raise SystemExit(f"no shards found under {[str(d) for d in replay_dirs]}")

    # Oversample 3x per stratum so the final draw is a uniform subsample of
    # the visited pool rather than purely shard-order biased.
    cap = per_stratum * 3
    for path in paths:
        if all(len(v) >= cap for v in strata.values()):
            break
        arrs, _meta = load_shard_arrays(path, lazy=True)
        if "x" not in arrs:
            continue
        hist = "legacy"
        if "_input_history_encoding" in arrs:
            hist = str(np.asarray(arrs["_input_history_encoding"]).reshape(-1)[0])
        n = int(arrs["x"].shape[0])
        has_src = np.asarray(arrs.get("has_is_selfplay", np.zeros(n))).astype(bool)
        is_sp = np.asarray(arrs.get("is_selfplay", np.zeros(n))).astype(bool)
        # Read x in sequential chunks (zarr-friendly) but visit chunks and
        # rows-within-chunk in random order, so the subsample stays unbiased
        # without per-row random access thrashing the chunk cache.
        chunk_rows = 1024
        chunk_starts = rng.permutation(np.arange(0, n, chunk_rows))
        for start in chunk_starts:
            stop = min(n, int(start) + chunk_rows)
            xs_chunk = np.asarray(arrs["x"][int(start):stop], dtype=np.float32)
            for j in rng.permutation(stop - int(start)):
                i = int(start) + int(j)
                if not has_src[i]:
                    continue
                src = 0 if is_sp[i] else 1
                x = xs_chunk[int(j)]
                piece_count = int(round(float(x[:12].sum())))
                ph = phase_bucket(piece_count)
                if len(strata[(ph, src)]) >= cap:
                    continue
                board = decode_board_from_planes(x, input_history_encoding=hist)
                if board is None or board.is_game_over():
                    continue
                key = position_key(board)
                if key in seen:
                    continue
                seen.add(key)
                strata[(ph, src)].append({
                    "key": key, "fen": board.fen(), "phase": ph, "source": src,
                })

    manifest: list[dict] = []
    leftovers: list[dict] = []
    for (ph, src), pool in sorted(strata.items()):
        rng.shuffle(pool)  # type: ignore[arg-type]
        manifest.extend(pool[:per_stratum])
        leftovers.extend(pool[per_stratum:])
        got = min(len(pool), per_stratum)
        print(f"[sample] {PHASE_NAMES[ph]:>10}/{SOURCE_NAMES[src]:<10} "
              f"{got}/{per_stratum} (pool {len(pool)})")
    # Backfill strata shortfalls from other strata so the set hits the
    # requested size even when e.g. endgame/curriculum is rare.
    shortfall = positions - len(manifest)
    if shortfall > 0 and leftovers:
        rng.shuffle(leftovers)  # type: ignore[arg-type]
        manifest.extend(leftovers[:shortfall])
        print(f"[sample] backfilled {min(shortfall, len(leftovers))} from leftovers")
    return manifest


def _label_one(eng: StockfishUCI, row: dict, *, nodes: int) -> dict:
    res = eng.search(row["fen"], nodes=nodes)
    multipv = []
    for pv in res.pvs or []:
        multipv.append({
            "move": pv.move_uci,
            "cp": None if pv.cp is None else int(pv.cp),
            "mate": None if pv.mate is None else int(pv.mate),
            "wdl": None if pv.wdl is None else [float(v) for v in pv.wdl],
        })
    if not multipv and res.bestmove_uci:
        multipv.append({"move": res.bestmove_uci, "cp": int(res.cp or 0),
                        "mate": res.mate, "wdl": None})
    wdl = res.wdl if res.wdl is not None else [333.0, 334.0, 333.0]
    return {
        **row,
        "multipv": multipv,
        "wdl": [float(v) for v in wdl],
        "bestmove": res.bestmove_uci,
        "nodes": int(res.nodes or nodes),
        "depth": int(res.depth or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--replay-dir", type=Path, action="append", required=True,
                    help="replay shard directory (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--positions", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", type=str, required=True)
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--multipv", type=int, default=10)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--sf-workers", type=int, default=4,
                    help="parallel Stockfish processes (1 thread each)")
    ap.add_argument("--nice", type=int, default=15,
                    help="niceness for the Stockfish processes")
    ap.add_argument("--resume", action="store_true",
                    help="continue labeling an incomplete set (append-only)")
    args = ap.parse_args()

    if args.multipv < 10:
        raise SystemExit("--multipv must be >= 10 (audit needs reasonable-move coverage)")
    out: Path = args.out
    manifest_path = out.with_suffix(out.suffix + ".manifest.jsonl")
    readme_path = out.parent / (out.stem + "_README.md")

    if out.exists() and not args.resume:
        raise SystemExit(
            f"{out} already exists. The audit set is FROZEN after generation; "
            "pass --resume only to finish an incomplete labeling run, or pick "
            "a new version name (audit_set_v2...)."
        )
    if readme_path.exists() and "FROZEN: yes" in readme_path.read_text(encoding="utf-8"):
        raise SystemExit(f"{out} is marked FROZEN in {readme_path}; refusing to touch it.")

    out.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        print(f"[manifest] reusing {manifest_path} ({len(manifest)} positions)")
    else:
        manifest = _sample_manifest(
            list(args.replay_dir), positions=int(args.positions), seed=int(args.seed),
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        print(f"[manifest] wrote {manifest_path} ({len(manifest)} positions)")

    labeled: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labeled.add(str(json.loads(line)["key"]))
    todo = [row for row in manifest if row["key"] not in labeled]
    print(f"[label] {len(labeled)} done, {len(todo)} to go "
          f"(nodes={args.nodes}, multipv={args.multipv}, workers={args.sf_workers})")

    sha = git_sha()
    config_desc = (
        f"positions={args.positions} seed={args.seed} nodes={args.nodes} "
        f"multipv={args.multipv} hash_mb={args.hash_mb} stockfish={args.stockfish}"
    )

    def _write_readme(*, frozen: bool) -> None:
        readme_path.write_text(
            f"# Audit set {out.name}\n\n"
            f"Frozen deep-Stockfish audit positions (see docs/eval_protocol.md).\n\n"
            f"- generated by: scripts/build_audit_set.py @ git {sha}\n"
            f"- config: {config_desc}\n"
            f"- replay dirs: {[str(d) for d in args.replay_dir]}\n"
            f"- positions are SIDE-TO-MOVE CANONICAL (white to move; black-to-move\n"
            f"  originals stored as their color-mirrored twin)\n"
            f"- labeled: {len(labeled)}/{len(manifest)}\n"
            f"- FROZEN: {'yes' if frozen else 'no (labeling incomplete)'}\n\n"
            f"Do NOT regenerate or append to a frozen set; new sampling = new version.\n",
            encoding="utf-8",
        )

    if not todo:
        _write_readme(frozen=True)
        print(f"[label] complete; {readme_path} marks the set FROZEN")
        return

    workers = max(1, int(args.sf_workers))
    engines = [
        StockfishUCI(
            args.stockfish, nodes=int(args.nodes), multipv=int(args.multipv),
            hash_mb=int(args.hash_mb), nice=int(args.nice),
        )
        for _ in range(workers)
    ]
    t0 = time.time()
    done = 0
    progress_lock = threading.Lock()
    work = iter(todo)
    try:
        with open(out, "a", encoding="utf-8") as f:

            def run_worker(wi: int) -> None:
                nonlocal done
                eng = engines[wi]
                while True:
                    with progress_lock:
                        row = next(work, None)
                    if row is None:
                        return
                    rec = _label_one(eng, row, nodes=int(args.nodes))
                    with progress_lock:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        done += 1
                        labeled.add(rec["key"])
                        if done % 25 == 0 or done == len(todo):
                            rate = done / max(1e-9, time.time() - t0)
                            eta_h = (len(todo) - done) / max(1e-9, rate) / 3600.0
                            print(f"[label] {done}/{len(todo)} "
                                  f"({rate:.2f} pos/s, eta {eta_h:.1f}h)")
                            _write_readme(frozen=False)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for fut in [pool.submit(run_worker, wi) for wi in range(workers)]:
                    fut.result()
    finally:
        for eng in engines:
            eng.close()

    _write_readme(frozen=len(labeled) >= len(manifest))
    print(f"[label] finished {done} positions in {(time.time() - t0) / 3600.0:.2f}h")


if __name__ == "__main__":
    main()
