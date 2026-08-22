#!/usr/bin/env python3
"""Shallow-SF wide-MultiPV label pass over a banked replay corpus — CPU ONLY.

Produces the labels the target-surgery ladder (arms R / V / G) joins against.
One JSONL record per UNIQUE corpus position, keyed by
``chess_anti_engine.eval.rvg_surgery.position_fingerprints`` — the same
function ``scripts/retarget_retrain.py`` calls on the batches it samples, so the
join is exact by construction rather than by convention.

THE LABEL, and why each part of it is what it is (ledger 2026-08-22):

* **MultiPV 40** by default (configurable). The coverage curve put MPV6's
  reach at 60.9% of the training target's bad mass against MPV20's 95.3% at
  production's real label budget, at ~equal wall cost; 40 goes further and takes
  the reach question off the table — at MultiPV >= the legal-move count,
  "unlisted" essentially stops existing, so the veto reads MEASURED cp instead of
  inferring badness from ABSENCE. That inference is precisely the June PR #78
  fabrication trap the arms exist to avoid. (The [[sf_multipv_width_costs_7x]]
  figure is a FIXED-DEPTH cost and does not apply at a fixed node budget; the
  smoke measures 40-at-fixed-nodes, which was unmeasured before this rig.)
* **150,000 nodes**. That IS production's label budget
  (``max(min(base, sf_label_nodes_cap), sf_label_nodes_floor)``); the 75k people
  quote is ``sf_nodes``, the OPPONENT budget. Per-line strength beyond this buys
  ~nothing for membership (-1 pp at 2.7x the cost).
* **``fresh=True`` on every submit.** ``StockfishPool.submit`` defaults
  ``fresh=False`` and no production caller overrides it, so production labels run
  on a TT warmed by unrelated positions: two warm passes over the same 300
  positions agree on the 6-move set only 32.3% of the time, while fresh
  reproduces at 100.0%. This pass is reproducible; production's is not, and the
  banked ``(75000, 6)`` shallow rows in ``data/audit_set_v1.jsonl.shallow_sf.jsonl``
  must NOT be joined against for the same reason.
* **Syzygy on**, matching the production label path. With tablebases on, TB
  lines report ``cp ~ +/-20000``; ``rvg_surgery.fold_multipv`` treats those as
  real evaluations and clips the resulting regret at 1000 cp like any other.

⚑ THE POSITION IS RECONSTRUCTED FROM PLANES, SIDE-TO-MOVE CANONICAL. Shards
store no FEN. ``eval.audit.decode_board_from_planes`` rebuilds each row as a
WHITE-to-move board whose white pieces are the original side to move (the
colour-mirrored twin when black was to move). Chess is exactly symmetric under
that mirror, so the search, its move list and the policy indices those moves
encode to are all in the same frame the stored ``policy_target`` uses. The pass
ASSERTS that alignment against the shard's own stored legal mask rather than
assuming it, on a per-shard random sample (``--roundtrip-sample``, 200 by
default; a single mismatch aborts before any search runs).

Usage::

    PYTHONPATH=. python3 scripts/rvg_label_pass.py \\
        --config configs/pbt2_small.yaml \\
        --replay-dir <banked corpus>/replay_shards \\
        --out <out dir>/rvg_labels_mpv40_150k.jsonl \\
        --threads 16

Two modes:

* ``--mode label`` (default) runs the SF pass described above;
* ``--mode enumerate-rows`` emits the DISTINCT corpus rows a seeded sweep will
  actually draw, reusing the rig's own buffer constructor and draw call, so the
  label pass can be restricted (``--restrict-to``) to exactly those rows and the
  wall-clock projection is honest rather than "the whole corpus, probably".

Resumable: re-running with the same ``--out`` reads the sidecar ``.partial``
file and skips every position already banked. Writes only the two files it is
given; reads the repo and the corpus. No torch on the label path (the enumerate
path imports the trainer's kwargs helper); no CUDA either way.
"""
from __future__ import annotations

import os

# Belt and braces: nothing here touches a GPU, and a stray CUDA init while a
# live experiment owns the card is exactly the failure this rig must not cause.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.eval.audit import decode_board_from_planes
from chess_anti_engine.eval.rvg_surgery import (
    RVG_LABEL_SCHEMA_VERSION,
    UNFINGERPRINTABLE,
    MultiPvLine,
    board_fingerprint,
    fold_multipv,
    position_fingerprints,
)
from chess_anti_engine.moves.encode import (
    POLICY_ENCODING_LC0_1858,
    move_to_index_for_encoding,
)
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.train.trainer import trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


class LabelRecord(TypedDict):
    """One banked label line. Typed so ``n_pv`` and ``key`` are readable without
    a cast — a dict of ``object`` turns every summary read into one."""

    v: int
    key: str
    fen: str
    shard: str
    row: int
    game_id: int
    ply_index: int
    policy_encoding: str
    n_pv: int
    best_cp: float
    move_index: list[int]
    cp_eff: list[float]
    regret_cp: list[float]
    uci: list[str]
    nodes: int | None
    depth: int | None


#: Production's realized label budget and the width the coverage curve chose.
DEFAULT_NODES = 150_000
DEFAULT_MULTIPV = 40


@dataclass(frozen=True)
class LabelOptions:
    """The production label path's SF options, READ off the yaml.

    ``sf_nodes`` is carried even though nothing here uses it, so the banked
    header states in the artifact itself that this pass's node budget is the
    LABEL budget and not the opponent budget the two are routinely confused for.
    """

    stockfish_path: str
    sf_hash_mb: int
    sf_nice: int
    sf_multipv_production: int
    sf_nodes_opponent: int
    sf_label_nodes_floor: int
    sf_label_nodes_cap: int
    label_syzygy_path: str


def load_label_options(config_path: Path) -> LabelOptions:
    """Read the label-relevant SF options off a run config.

    ``_sf_syzygy_path_for_slot``: a label query takes the "normal" path, which is
    ``stockfish_syzygy_path or syzygy_path``. The ``syzygy_path``-only "full"
    path is reached exclusively by the non-adjudicated <=6-piece play-through
    tail, never by a label query.
    """
    flat = flatten_run_config_defaults(load_yaml_file(config_path))

    def as_int(key: str, default: int) -> int:
        v = flat.get(key)
        return int(v) if isinstance(v, (int, float, str)) and str(v) != "" else default

    def as_str(key: str) -> str:
        v = flat.get(key)
        return str(v) if isinstance(v, (str, int, float)) else ""

    return LabelOptions(
        stockfish_path=as_str("stockfish_path"),
        sf_hash_mb=as_int("sf_hash_mb", 16),
        sf_nice=as_int("sf_nice", 0),
        sf_multipv_production=as_int("sf_multipv", 1),
        sf_nodes_opponent=as_int("sf_nodes", 0),
        sf_label_nodes_floor=as_int("sf_label_nodes_floor", 0),
        sf_label_nodes_cap=as_int("sf_label_nodes_cap", 0),
        label_syzygy_path=as_str("stockfish_syzygy_path") or as_str("syzygy_path"),
    )


@dataclass
class CorpusRow:
    """One selected corpus row, already canonicalized."""

    key: bytes
    fen: str
    shard: str
    row: int
    game_id: int
    ply_index: int
    policy_encoding: str
    legal_idx: np.ndarray


@dataclass
class ScanResult:
    """What ``scan_corpus`` found, with every drop counted rather than filtered.

    ``predicted_matches`` is the number of SELECTED rows that a complete label
    file must join — the count the verification step asserts. It is
    ``selected - unfingerprintable - undecodable``, i.e. the rows that have a
    key AND a board; a search failure later subtracts from it explicitly, so the
    prediction is never quietly adjusted to whatever the run produced.
    """

    rows: list[CorpusRow]
    unique: dict[bytes, CorpusRow]
    selected: int
    unfingerprintable: int
    undecodable: int
    fingerprint_roundtrip_checked: int
    fingerprint_roundtrip_mismatch: int
    alignment_checked: int
    alignment_illegal: int

    @property
    def predicted_matches(self) -> int:
        return self.selected - self.unfingerprintable - self.undecodable


def _legal_indices(arrs: dict[str, np.ndarray], i: int) -> np.ndarray:
    mask = arrs.get("legal_mask")
    has = arrs.get("has_legal_mask")
    if mask is None or has is None or not bool(np.asarray(has)[i]):
        return np.zeros((0,), dtype=np.int64)
    return np.flatnonzero(np.asarray(mask[i]) > 0).astype(np.int64)


def scan_corpus(
    replay_dir: Path,
    *,
    row_start: int,
    row_end: int | None,
    limit: int,
    roundtrip_sample: int,
) -> ScanResult:
    """Fingerprint + canonicalize the selected corpus rows.

    Rows are addressed by GLOBAL index in ``iter_shard_paths`` order followed by
    within-shard order — the same deterministic order the rig's own
    ``_snapshot_shards`` produces, so ``--row-range`` names the same rows on
    every invocation. ``limit`` caps how many rows are SELECTED, applied after
    the range.

    ⚑ Deduplication is by KEY, first row wins, mirroring the coverage rig's
    first-listing-wins convention. A key is a POSITION, and a position has one
    label; the duplicate rows all join to it.
    """
    paths = iter_shard_paths(replay_dir)
    if not paths:
        raise SystemExit(f"no replay shards under {replay_dir}")

    rows: list[CorpusRow] = []
    unique: dict[bytes, CorpusRow] = {}
    selected = 0
    unfingerprintable = 0
    undecodable = 0
    rt_checked = 0
    rt_mismatch = 0
    align_checked = 0
    align_illegal = 0
    rng = np.random.default_rng(0)

    global_index = 0
    for path in paths:
        # Lazy FIRST, to read the row count without materializing ~9 MB of
        # planes: `--row-range` is how a multi-hour pass gets split across
        # invocations, and a scan that fully loaded every shard before the range
        # would charge each of those invocations the whole corpus's read.
        lazy, _ = load_shard_arrays(path, lazy=True)
        n = int(lazy["x"].shape[0])
        lo = max(0, row_start - global_index)
        hi = n if row_end is None else max(0, min(n, row_end - global_index))
        if limit > 0 and selected + (hi - lo) > limit:
            hi = lo + max(0, limit - selected)
        if lo >= hi:
            global_index += n
            if limit > 0 and selected >= limit:
                break
            if row_end is not None and global_index >= row_end:
                break
            continue

        arrs, _ = load_shard_arrays(path, lazy=False)
        x = np.asarray(arrs["x"])
        enc = str(np.asarray(arrs["_input_history_encoding"]).reshape(-1)[0])
        pol_enc = str(np.asarray(arrs["_policy_encoding"]).reshape(-1)[0])
        if pol_enc != POLICY_ENCODING_LC0_1858:
            # `move_to_index_for_encoding` ignores its `policy_encoding` argument
            # and always returns the COMPACT index, so an az_4672 corpus would be
            # labeled with silently wrong slots. Refuse rather than mis-index.
            raise SystemExit(
                f"{path.name} stores policy_encoding={pol_enc!r}; this pass emits "
                f"{POLICY_ENCODING_LC0_1858!r} indices only"
            )
        keys = position_fingerprints(x[lo:hi], input_history_encoding=enc)
        game_ids = np.asarray(arrs["game_id"]).astype(np.int64)
        plies = np.asarray(arrs["ply_index"]).astype(np.int64)
        # Sample the expensive board-vs-plane key check rather than running it on
        # every row: it is a PROPERTY of the derivation, so a random sample
        # falsifies it just as well and a full pass would double the scan cost.
        check_rows = set(
            rng.choice(hi - lo, size=min(roundtrip_sample, hi - lo), replace=False).tolist()
        ) if roundtrip_sample > 0 else set()

        for j in range(lo, hi):
            selected += 1
            key = keys[j - lo]
            if key == UNFINGERPRINTABLE:
                unfingerprintable += 1
                continue
            board = decode_board_from_planes(x[j], input_history_encoding=enc)
            if board is None:
                undecodable += 1
                continue
            if (j - lo) in check_rows:
                rt_checked += 1
                if board_fingerprint(board) != key:
                    rt_mismatch += 1
            row = CorpusRow(
                key=key, fen=board.fen(), shard=path.name, row=j,
                game_id=int(game_ids[j]), ply_index=int(plies[j]),
                policy_encoding=pol_enc, legal_idx=_legal_indices(arrs, j),
            )
            rows.append(row)
            unique.setdefault(key, row)
            if (j - lo) in check_rows and row.legal_idx.size:
                # FRAME PROOF: every move legal on the reconstructed board must
                # encode to an index the SHARD's own legal mask also set. If the
                # POV frame or the policy encoding were wrong this is ~0, and
                # every edit downstream would silently address the wrong slots.
                align_checked += 1
                stored = set(row.legal_idx.tolist())
                for mv in board.legal_moves:
                    a = move_to_index_for_encoding(mv, board, policy_encoding=pol_enc)
                    if a >= 0 and a not in stored:
                        align_illegal += 1

        global_index += n
        if limit > 0 and selected >= limit:
            break
        if row_end is not None and global_index >= row_end:
            break

    return ScanResult(
        rows=rows, unique=unique, selected=selected,
        unfingerprintable=unfingerprintable, undecodable=undecodable,
        fingerprint_roundtrip_checked=rt_checked,
        fingerprint_roundtrip_mismatch=rt_mismatch,
        alignment_checked=align_checked, alignment_illegal=align_illegal,
    )


def label_record(row: CorpusRow, res: object, policy_encoding: str) -> LabelRecord | None:
    """One banked label record, or ``None`` when the search scored nothing.

    ``move_index`` is in the SHARD's policy space (production: the compact
    ``lc0_1858``), pre-encoded here so the training-time wrapper never has to
    touch python-chess. A line whose move does not encode (should not happen on
    a legal search result) is dropped and counted by the caller through the
    ``n_pv`` histogram, which is why the histogram is a summary column and not
    an afterthought.
    """
    pvs = getattr(res, "pvs", None) or []
    folded = fold_multipv([
        MultiPvLine(
            move=str(pv.move_uci),
            cp=None if pv.cp is None else int(pv.cp),
            mate=None if pv.mate is None else int(pv.mate),
        )
        for pv in pvs
    ])
    if folded is None:
        return None
    board = chess.Board(row.fen)
    idx: list[int] = []
    cp: list[float] = []
    reg: list[float] = []
    ucis: list[str] = []
    for uci, c, r in zip(folded.moves, folded.cp_eff, folded.regret_cp, strict=True):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        a = move_to_index_for_encoding(mv, board, policy_encoding=policy_encoding)
        if a < 0:
            continue
        idx.append(int(a))
        # ⚑ EFFECTIVE CP IS THE STORED VALUE; regret rides along for readability
        # and the loader always RE-DERIVES it. Regret is relative to the set's
        # own best, so it cannot survive a layered overlay — see
        # `rvg_surgery.regrets_from_cp`.
        cp.append(round(float(c), 2))
        reg.append(round(float(r), 2))
        ucis.append(uci)
    if not idx:
        return None
    return {
        "v": RVG_LABEL_SCHEMA_VERSION,
        "key": row.key.hex(),
        "fen": row.fen,
        "shard": row.shard,
        "row": row.row,
        "game_id": row.game_id,
        "ply_index": row.ply_index,
        "policy_encoding": policy_encoding,
        "n_pv": len(idx),
        "best_cp": round(folded.best_cp, 2),
        "move_index": idx,
        "cp_eff": cp,
        "regret_cp": reg,
        "uci": ucis,
        "nodes": getattr(res, "nodes", None),
        "depth": getattr(res, "depth", None),
    }



def enumerate_drawn_rows(
    *,
    config_path: Path,
    replay_dir: Path,
    steps: int,
    batch_size: int,
    out_path: Path,
    progress_every: int,
) -> dict[str, object]:
    """Emit the DISTINCT corpus rows a seeded sweep will actually draw.

    ⚑ COST CONTROL, AND IT REUSES THE RIG'S OWN DRAW. The buffer comes from
    ``scripts/retarget_retrain.build_rig_replay_buffer`` — the same constructor
    ``_run_variant`` calls — and the rows come from ``sample_batch_arrays``, the
    same method the Trainer's ``_sample_batch_host`` calls, at the same batch
    size, the same number of times (``steps * accum_steps``), from the same
    seed. Nothing about the selection is re-derived here; re-typing the
    constructor would silently name a different row set, because every sampling
    kwarg (recency exponent, draw cap, wl ratio, refresh interval,
    ``deterministic_refresh``) changes WHICH rows come out.

    What this reproduces exactly, and what it does not:

    * reproduced: the pool (``deterministic_refresh=True`` makes the draw a
      function of the seed and the shard list), the per-call batch size, the
      call count, and the wdl-balanced selection inside each call;
    * NOT reproduced, deliberately: a transient-CUDA retry, which pulls
      replacement batches and advances the buffer's RNG. That path also
      DE-PAIRS the sweep, so ``_assert_draws_unchanged`` aborts it — a sweep
      this enumeration under-covers is a sweep that is void anyway;
    * irrelevant: mirror augmentation and the SF-target rebuild both run AFTER
      ``sample_batch_arrays`` returns, so they change the CONTENT of a batch and
      never which rows it holds.

    All arms of a ladder share one enumeration: they are seeded identically and
    the rig ABORTS if their shard pool or draw count differs.
    """
    import numpy as _np

    from scripts.retarget_retrain import build_rig_replay_buffer

    base_config = flatten_run_config_defaults(load_yaml_file(config_path))
    accum_steps = int(trainer_kwargs_from_config(dict(base_config)).get("accum_steps", 1) or 1)
    draws = int(steps) * accum_steps
    # The stored plane width IS the arch's width for any valid sweep: the rig
    # refuses a mismatch up front (`_assert_replay_planes_match`), so reading it
    # off the shards cannot disagree with a checkpoint the sweep would accept —
    # and it keeps this mode from needing a GPU checkpoint on the CPU path.
    paths = iter_shard_paths(replay_dir)
    if not paths:
        raise SystemExit(f"no replay shards under {replay_dir}")
    lazy, _ = load_shard_arrays(paths[0], lazy=True)
    target_planes = int(lazy["x"].shape[1])
    seed = int(base_config.get("seed", 0) or 0)

    print(f"[rvg-enum] seed={seed} steps={steps} accum_steps={accum_steps} "
          f"draws={draws} batch={batch_size} planes={target_planes} "
          f"shards={len(paths)}", flush=True)

    buf = build_rig_replay_buffer(
        config=dict(base_config), replay_dir=replay_dir,
        target_planes=target_planes, rng=_np.random.default_rng(seed),
    )
    seen: set[bytes] = set()
    rows_drawn = 0
    t0 = time.time()
    try:
        for i in range(draws):
            arrs = buf.sample_batch_arrays(int(batch_size))
            encs = np.asarray(arrs["_input_history_encoding"]).reshape(-1)
            uniq = np.unique(encs)
            for enc in uniq:
                sel = np.flatnonzero(encs == enc)
                seen.update(position_fingerprints(
                    np.asarray(arrs["x"])[sel], input_history_encoding=str(enc),
                ))
            rows_drawn += int(np.asarray(arrs["x"]).shape[0])
            if progress_every > 0 and (i + 1) % progress_every == 0:
                rate = (i + 1) / max(1e-9, time.time() - t0)
                print(f"[rvg-enum] {i + 1}/{draws} draws "
                      f"({rate:.2f} draws/s, {len(seen)} distinct rows, "
                      f"eta {(draws - i - 1) / max(1e-9, rate) / 60.0:.1f} min)",
                      flush=True)
    finally:
        buf.close()

    wall = time.time() - t0
    header: dict[str, object] = {
        "record": "provenance",
        "mode": "enumerate-rows",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(config_path),
        "replay_dir": str(replay_dir),
        "seed": seed,
        "steps": int(steps),
        "accum_steps": accum_steps,
        "draws": draws,
        "batch_size": int(batch_size),
        "rows_drawn_with_repeats": rows_drawn,
        "distinct_rows": len(seen),
        "distinct_fraction_of_draws": (len(seen) / rows_drawn) if rows_drawn else 0.0,
        "wall_seconds": round(wall, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for key in sorted(seen):
            fh.write(key.hex() + "\n")
    print(f"[rvg-enum] {rows_drawn} row draws -> {len(seen)} DISTINCT rows "
          f"in {wall / 60.0:.1f} min -> {out_path}", flush=True)
    return header


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=("label", "enumerate-rows"), default="label",
                    help="'label' runs the SF pass; 'enumerate-rows' emits the "
                         "DISTINCT corpus rows a seeded sweep will draw (cost "
                         "control: feed it back with --restrict-to)")
    ap.add_argument("--config", type=Path, required=True,
                    help="run config the SF options are read off (production yaml)")
    ap.add_argument("--replay-dir", type=Path, required=True,
                    help="banked corpus shard directory (a FROZEN copy, never the live window)")
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl")
    ap.add_argument("--threads", type=int, default=16,
                    help="independent 1-thread engines run in parallel (default 16)")
    ap.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    ap.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0: label at most this many corpus ROWS (smoke runs)")
    ap.add_argument("--row-range", default="",
                    help="START:END global corpus row indices (END exclusive, blank = all)")
    ap.add_argument("--roundtrip-sample", type=int, default=200,
                    help="rows per shard to check board-vs-plane key agreement and "
                         "policy-index alignment on (0 disables; a mismatch is FATAL)")
    ap.add_argument("--restrict-to", type=Path, default=None,
                    help="a key file from --mode enumerate-rows: label ONLY the "
                         "rows the seeded sweep will actually draw")
    ap.add_argument("--enum-steps", type=int, default=6000,
                    help="enumerate-rows: optimizer steps the sweep will run")
    ap.add_argument("--enum-batch-size", type=int, default=0,
                    help="enumerate-rows: batch size (0 = the config's)")
    ap.add_argument("--enum-progress-every", type=int, default=100,
                    help="enumerate-rows: draws between progress lines")
    ap.add_argument("--no-syzygy", action="store_true",
                    help="deviate from the production label path and run WITHOUT "
                         "tablebases (recorded in the header)")
    args = ap.parse_args()

    if args.mode == "enumerate-rows":
        base = flatten_run_config_defaults(load_yaml_file(args.config))
        enumerate_drawn_rows(
            config_path=args.config, replay_dir=args.replay_dir,
            steps=int(args.enum_steps),
            batch_size=int(args.enum_batch_size or base.get("batch_size", 512)),
            out_path=args.out, progress_every=int(args.enum_progress_every),
        )
        return

    opts = load_label_options(args.config)
    if not opts.stockfish_path:
        raise SystemExit(f"no stockfish_path in {args.config}")
    binary = str(Path(opts.stockfish_path).resolve())
    syzygy = None if args.no_syzygy else (opts.label_syzygy_path or None)

    row_start, row_end = 0, None
    if args.row_range:
        a, _, b = args.row_range.partition(":")
        if not _:
            raise SystemExit(f"--row-range must be START:END, got {args.row_range!r}")
        row_start = int(a) if a else 0
        row_end = int(b) if b else None

    if args.restrict_to is not None and int(args.limit) > 0:
        # `--limit` caps the SCAN and `--restrict-to` filters it, so together
        # they label "the first N rows that happen to also be enumerated" —
        # a subset nobody asked for, silently. `--row-range` is the way to
        # shard a restricted pass across invocations.
        raise SystemExit(
            "--limit and --restrict-to together label an arbitrary subset "
            "(the limit caps the scan BEFORE the restriction). Use --row-range "
            "to split a restricted pass across invocations."
        )

    t_scan = time.time()
    scan = scan_corpus(
        args.replay_dir, row_start=row_start, row_end=row_end,
        limit=int(args.limit), roundtrip_sample=int(args.roundtrip_sample),
    )
    scan_wall = time.time() - t_scan
    print(f"[rvg-label] scanned {scan.selected} rows in {scan_wall:.1f}s: "
          f"{len(scan.rows)} canonicalized, {len(scan.unique)} unique positions, "
          f"{scan.unfingerprintable} unfingerprintable, {scan.undecodable} undecodable",
          flush=True)
    print(f"[rvg-label] key round-trip (board vs planes): "
          f"{scan.fingerprint_roundtrip_checked - scan.fingerprint_roundtrip_mismatch}"
          f"/{scan.fingerprint_roundtrip_checked} agree | "
          f"policy-index alignment vs stored legal mask: "
          f"{scan.alignment_illegal} illegal of {scan.alignment_checked} rows checked",
          flush=True)
    # ⚑ FATAL, not a warning. Both checks answer "do the labels address the rows
    # the rig will edit"; a partial answer here becomes a silently mis-joined
    # sweep hours later, which is the failure this whole rig exists to avoid.
    if scan.fingerprint_roundtrip_mismatch or scan.alignment_illegal:
        raise SystemExit(
            f"ABORT: {scan.fingerprint_roundtrip_mismatch} key round-trip mismatches "
            f"and {scan.alignment_illegal} moves outside the stored legal mask. The "
            "labels would address different rows (or different policy slots) than "
            "the rig edits."
        )
    if args.restrict_to is not None:
        wanted: set[bytes] = set()
        with open(args.restrict_to, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("{"):
                    wanted.add(bytes.fromhex(line))
        before = len(scan.unique)
        # Restrict BOTH the unique set (what gets searched) and the row list
        # (what the join is predicted and asserted over), so the join check keeps
        # measuring the same population the pass actually labeled.
        scan.unique = {k: v for k, v in scan.unique.items() if k in wanted}
        scan.rows = [r for r in scan.rows if r.key in wanted]
        scan.selected = len(scan.rows)
        scan.unfingerprintable = 0
        scan.undecodable = 0
        print(f"[rvg-label] --restrict-to {args.restrict_to}: "
              f"{len(wanted)} enumerated rows, {before} -> {len(scan.unique)} "
              "unique positions to label", flush=True)

    if not scan.unique:
        raise SystemExit("no labelable rows selected")

    out_path: Path = args.out
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, LabelRecord] = {}
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[str(rec["key"])] = rec
        print(f"[rvg-label] resuming: {len(done)} positions already banked", flush=True)

    todo = [r for k, r in scan.unique.items() if k.hex() not in done]
    engines: list[StockfishUCI] = []
    failures: list[dict[str, str]] = []
    failed_keys: set[str] = set()
    unscoreable_keys: set[str] = set()
    unscoreable = 0
    lock = threading.Lock()
    work = iter(todo)
    n_done = 0
  # Set AFTER the engines are up (below). Spawning N Stockfish processes costs
  # seconds, and folding that into the rate makes a 200-row smoke read ~2x
  # slower than the throughput a multi-hour pass will actually run at — which is
  # exactly the number the full-pass projection is taken from.
    t0 = time.time()

    def new_engine() -> StockfishUCI:
        return StockfishUCI(
            binary, nodes=int(args.nodes), multipv=int(args.multipv),
            hash_mb=opts.sf_hash_mb, syzygy_path=syzygy, nice=opts.sf_nice,
        )

    engine_spawn_s = 0.0
    try:
        t_spawn = time.time()
        engines = [new_engine() for _ in range(max(1, int(args.threads)))]
        engine_spawn_s = time.time() - t_spawn
        t0 = time.time()
        print(f"[rvg-label] {len(engines)} engines up in {engine_spawn_s:.1f}s; "
              f"{len(todo)} positions to search", flush=True)
        # Append mode + flush per record: an interrupted pass loses at most the
        # record being written, and a resumed pass reuses everything before it.
        with open(partial_path, "a", encoding="utf-8") as fh:
            def run_worker(wi: int) -> None:
                nonlocal n_done, unscoreable
                while True:
                    with lock:
                        item = next(work, None)
                    if item is None:
                        return
                    rec: LabelRecord | None = None
                    last_err = ""
                    scored = True
                    # One retry, on a FRESH engine: a desynced engine is poisoned
                    # permanently, so retrying on the same process could only
                    # ever raise again.
                    for attempt in (0, 1):
                        try:
                            res = engines[wi].search(
                                item.fen, nodes=int(args.nodes), fresh=True,
                            )
                            rec = label_record(item, res, item.policy_encoding)
                            scored = rec is not None
                            break
                        except Exception as exc:
                            last_err = f"{type(exc).__name__}: {exc}"
                            with lock:
                                print(f"[rvg-label] attempt {attempt} failed on "
                                      f"{item.key.hex()[:12]}: {last_err}", flush=True)
                            try:
                                engines[wi].close()
                            except Exception:
                                pass
                            try:
                                engines[wi] = new_engine()
                            except Exception as exc2:
                                last_err = f"engine respawn failed: {exc2}"
                                break
                    with lock:
                        if rec is not None:
                            fh.write(json.dumps(rec) + "\n")
                            fh.flush()
                            done[item.key.hex()] = rec
                        elif scored:
                            failures.append({
                                "key": item.key.hex(), "fen": item.fen, "error": last_err,
                            })
                            failed_keys.add(item.key.hex())
                        else:
                            unscoreable += 1
                            unscoreable_keys.add(item.key.hex())
                        n_done += 1
                        if n_done % 100 == 0 or n_done == len(todo):
                            rate = n_done / max(1e-9, time.time() - t0)
                            eta = (len(todo) - n_done) / max(1e-9, rate)
                            print(f"[rvg-label] {n_done}/{len(todo)} "
                                  f"({rate:.2f} pos/s, eta {eta / 60.0:.1f} min)",
                                  flush=True)

            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                for fut in [pool.submit(run_worker, wi) for wi in range(len(engines))]:
                    fut.result()
    finally:
        for eng in engines:
            try:
                eng.close()
            except Exception:
                pass

    wall = time.time() - t0
    rate = (len(todo) / wall) if wall > 0 else 0.0
    hist = Counter(r["n_pv"] for r in done.values())

    # THE PREDICTION, FROM SCAN-SIDE BOOKKEEPING ONLY. Every canonicalized row
    # whose POSITION got a label must join, so the count is
    # |{row : row.key in unique_keys - failed - unscoreable}| — derived from the
    # unique-key ledger this pass maintained, never from the file it is about to
    # check. (An earlier draft computed `predicted` by subtracting the rows that
    # did not match, which made the assertion below arithmetically incapable of
    # failing: a gate that cannot fail.)
    expected_keys = {k.hex() for k in scan.unique} - failed_keys - unscoreable_keys
    predicted = sum(1 for r in scan.rows if r.key.hex() in expected_keys)

    # THE MEASUREMENT, FROM THE BYTES ON DISK. Read back off the append log
    # rather than out of `done`, so a hex/serialization bug or a resume that
    # mixed in another corpus's records shows up as a disagreement instead of
    # being confirmed by the same dict that produced it.
    banked_keys: set[str] = set()
    with open(partial_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                banked_keys.add(str(json.loads(line)["key"]))
    matched = sum(1 for r in scan.rows if r.key.hex() in banked_keys)

    header = {
        "record": "provenance",
        "schema_version": RVG_LABEL_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replay_dir": str(args.replay_dir),
        "config": str(args.config),
        "engine_path_resolved": binary,
        "nodes": int(args.nodes),
        "multipv": int(args.multipv),
        "threads_per_engine": 1,
        "workers": len(engines),
        "hash_mb": opts.sf_hash_mb,
        "nice": opts.sf_nice,
        "syzygy_path": syzygy or "(none)",
        "syzygy_matches_production_label_path": bool(
            syzygy and syzygy == opts.label_syzygy_path,
        ),
        "ucinewgame_per_position": True,
        "row_start": row_start,
        "row_end": row_end,
        "limit": int(args.limit),
        "rows_selected": scan.selected,
        "rows_canonicalized": len(scan.rows),
        "unique_positions": len(scan.unique),
        "rows_unfingerprintable": scan.unfingerprintable,
        "rows_undecodable": scan.undecodable,
        "key_roundtrip_checked": scan.fingerprint_roundtrip_checked,
        "key_roundtrip_mismatch": scan.fingerprint_roundtrip_mismatch,
        "alignment_rows_checked": scan.alignment_checked,
        "alignment_moves_outside_stored_mask": scan.alignment_illegal,
        "positions_labeled": len(done),
        "positions_unscoreable": unscoreable,
        "n_failures": len(failures),
        "failures": failures[:50],
        "join_rows_matched": matched,
        "join_rows_predicted": predicted,
        "n_pv_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "wall_seconds_this_invocation": round(wall, 1),
        "engine_spawn_seconds": round(engine_spawn_s, 1),
        "scan_seconds": round(scan_wall, 1),
        "positions_searched_this_invocation": len(todo),
        # Search throughput with engine spawn EXCLUDED — the number a full-pass
        # projection should be taken from.
        "positions_per_second_this_invocation": round(rate, 3),
        "production_reference": {
            "sf_multipv": opts.sf_multipv_production,
            "sf_nodes_opponent": opts.sf_nodes_opponent,
            "sf_label_nodes_floor": opts.sf_label_nodes_floor,
            "sf_label_nodes_cap": opts.sf_label_nodes_cap,
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for rec in done.values():
            fh.write(json.dumps(rec) + "\n")

    print(f"[rvg-label] wrote {out_path} — {len(done)} label records, "
          f"{len(failures)} failures, {unscoreable} unscoreable, "
          f"{wall / 60.0:.1f} min, {rate:.2f} pos/s", flush=True)
    print(f"[rvg-label] n_pv histogram: {dict(sorted(hist.items()))}", flush=True)
    print(f"[rvg-label] JOIN: {matched} of {scan.selected} selected corpus rows "
          f"match a label (predicted {predicted})", flush=True)
    if matched != predicted:
        raise SystemExit(
            f"FAIL: join matched {matched} corpus rows, predicted {predicted}. "
            "The key derived at label time does not address the rows the rig "
            "will edit — do not run the ladder against this file."
        )
    if not failures and not unscoreable:
        partial_path.unlink(missing_ok=True)
    print("[rvg-label] OK: join verified", flush=True)


if __name__ == "__main__":
    main()
