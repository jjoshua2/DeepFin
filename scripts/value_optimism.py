#!/usr/bin/env python3
"""Value-head optimism by Stockfish-eval stratum, on PRODUCTION replay rows.

Answers "does the value head know it is losing?" without letting the net choose
the sample. Rows come from the live replay shards; the stratum comes from a
Stockfish evaluation of the row's OWN position; and the blended training target
is scored in the same buckets as the head, which is what separates "the head
fails to fit its target" from "the target is itself optimistic".

Why this is not a flag on ``scripts/value_regret.py``:

- value_regret scores value RANKING (1-ply deep-SF regret of the move the value
  head would pick) and is the project's VALUE yardstick. This scores value
  LEVEL. Folding a level metric into the ranking yardstick would change what a
  historical "value_regret = N cp" means, which this repo has already been
  burned by once.
- value_regret reads the frozen audit set, which stores FENs. A FEN has no move
  stack, so 117 of the 175 production input planes are zero and ABSOLUTE cp
  bars measured there are compromised (docs/rl_loop_audit.md M10). This script
  feeds the net the stored ``x`` planes — the exact tensor training and selfplay
  used — so its absolute numbers are not subject to that defect.
- The audit set carries neither the game outcome nor the blended target, so it
  physically cannot answer the head-vs-target question.

``scripts/value_regret.py --sf-strata`` reports the RANKING axis in the same
buckets, on the frozen set. Read them together.

The P0 ruler, and why it needs no new Stockfish compute: selfplay records one
row per net turn, and the SF label attached to a row is the evaluation of the
position AFTER that row's move, from the mover-to-come's point of view. So for
two rows that are consecutive plies of one game, the EARLIER row's stored
``sf_label_meta`` eval is an evaluation of the LATER row's own position, already
in the later row's point of view — no flip, no re-query. That is the same
one-ply shift ``sf_p0_policy_target`` uses. Labels are the production ones
(~698k nodes, MultiPV 40, median depth ~12); they are shallower than the frozen
audit set's 1M-node MultiPV-10 labels, which is the price of scoring the exact
rows the target was built from.

Usage:

    PYTHONPATH=. nice -n 19 python3 scripts/value_optimism.py \
        --checkpoint <trainer.pt> --shards 120 --gpu-mem-fraction 0.12

    # negative control: the same run with the position <-> net-eval
    # association destroyed. Every bucket effect must vanish.
    PYTHONPATH=. nice -n 19 python3 scripts/value_optimism.py \
        --checkpoint <trainer.pt> --shards 120 --shuffle-control
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.encoding import model_encoding_kwargs
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.value_optimism import (
    CP_CLAMP,
    SF_EVAL_BUCKET_EDGES,
    SF_LABEL_ATTACHMENT_MIN,
    BucketStat,
    OptimismRows,
    bucket_names_for,
    bucket_net_score_spread,
    cp_to_expected_score,
    expected_score,
    score_buckets,
    sf_label_attachment_corr,
    tail_asymmetry,
)
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp_array
from chess_anti_engine.train.trainer import resolve_sf_target_params
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file
from scripts.trial_paths import default_run_dir, latest_trial_dir

# The blend in train/losses.py is a plain convex combination ONLY while these
# modifiers sit at their neutral values. Each of them would silently move the
# target this script reconstructs, so a non-neutral one is a hard stop rather
# than a footnote — the alternative is a number that quietly stops describing
# what production trains on.
_NEUTRAL_BLEND_KNOBS: dict[str, Any] = {
    "sf_wdl_temperature": 1.0,
    "sf_search_dampen_sf_low": 0.0,
    "sf_search_dampen_sf_high": 0.0,
    "use_adjusted_wdl_target": False,
}


def _shard_num(path: Path) -> int:
    m = re.search(r"shard_(\d+)\.zarr$", path.name)
    return int(m.group(1)) if m else -1


def _resolve_replay_dir(run_dir: Path, replay_dir: str | None) -> Path:
    if replay_dir:
        return Path(replay_dir)
    trial = latest_trial_dir(run_dir, required=True)
    d = run_dir / "replay" / trial.name / "replay_shards"
    if not d.is_dir():
        raise SystemExit(f"no replay shards under {d}")
    return d


def _realized_sf_wdl_frac(run_dir: Path) -> tuple[float, str]:
    """Realized ``sf_wdl_frac`` from progress.csv, NOT the yaml nominal.

    The yaml's 0.50 is decorative: the frac is resolved per iteration from the
    live ``wdl_regret`` and returns the ``sf_wdl_frac_floor`` whenever regret is
    below ``sf_wdl_floor_at_regret``, which it always is. The only artifact that
    holds the answer is the trial's own progress row (rl_loop_audit method rule
    12: a realized value comes from the stage that resolves it).
    """
    import csv

    trial = latest_trial_dir(run_dir, required=True)
    path = trial / "progress.csv"
    if not path.exists():
        raise SystemExit(f"no progress.csv at {path}; pass --sf-wdl-frac explicitly")
    last: dict[str, str] | None = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sf_wdl_frac"):
                last = row
    if last is None:
        raise SystemExit(f"{path} has no sf_wdl_frac column; pass --sf-wdl-frac explicitly")
    return float(last["sf_wdl_frac"]), f"{path} iter={last.get('training_iteration')}"


def _check_blend_knobs(flat: dict[str, Any]) -> None:
    bad = []
    for key, want in _NEUTRAL_BLEND_KNOBS.items():
        got = flat.get(key, want)
        neutral = (bool(got) == bool(want)) if isinstance(want, bool) else (float(got) == float(want))
        if not neutral:
            bad.append(f"{key}={got!r} (neutral: {want!r})")
    if bad:
        raise SystemExit(
            "the production WDL blend is no longer a plain convex combination: "
            + "; ".join(bad)
            + ". This script reconstructs the target and would report a number that "
              "is not what training sees. Update the reconstruction in "
              "scripts/value_optimism.py before trusting any output.",
        )


def _load_rows(
    shard_paths: list[Path], *, max_rows: int, slope: float, draw_width: float,
    history_encoding: str, history_rep_fix: bool, attachment_min: float,
) -> dict[str, np.ndarray]:
    """Collect P0-paired rows from the newest shards, newest first."""
    import zarr

    keys = (
        "game_id", "ply_index", "sf_label_meta", "has_sf_label_meta", "sf_wdl",
        "has_sf_wdl", "search_wdl", "has_search_wdl", "wdl_target", "x",
    )
    cols: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    n_seen = 0
    n_rows_total = 0
    n_skipped_encoding = 0
    rejected: list[tuple[str, float]] = []
    for path in shard_paths:
        z = zarr.open(str(path), mode="r")
        # Method rule 7: verify identity before believing a lopsided number. A
        # shard written under a different input encoding would feed the net a
        # differently-laid-out tensor and produce a value error that is really
        # an encoding mismatch.
        shard_hist = normalize_lc0_history_encoding(z.attrs.get("input_history_encoding"))
        if shard_hist != history_encoding or bool(z.attrs.get("history_rep_fix", False)) != history_rep_fix:
            n_skipped_encoding += 1
            continue
        try:
            gid = np.asarray(z["game_id"][:])
            ply = np.asarray(z["ply_index"][:])
            meta = np.asarray(z["sf_label_meta"][:])
            has_meta = np.asarray(z["has_sf_label_meta"][:]).astype(bool)
        except KeyError:
            continue
        n_rows_total += int(gid.shape[0])
        # INTEGRITY GATE, and it is not decorative: shards written after the
        # 2026-07-31 restart carry SF labels that are detached from their own
        # rows, so every number below them would be arithmetic over noise.
        attach = sf_label_attachment_corr(
            np.asarray(z["x"][:, 0:12]), np.asarray(z["sf_wdl"][:]).astype(np.float64),
            np.asarray(z["has_sf_wdl"][:]).astype(bool),
        )
        if not np.isfinite(attach) or attach < attachment_min:
            rejected.append((path.name, float(attach)))
            continue
        # Row i is P0-paired iff row i-1 is the immediately preceding ply of the
        # same game AND carries a usable eval. Both conditions are needed: the
        # ply-gap check is what makes the earlier row's label an evaluation of
        # THIS row's position, and without it the shift silently spans a gap of
        # several plies (only ~24% of recorded rows are consecutive, because
        # most plies are fast-ply and unrecorded).
        paired = np.zeros(gid.shape[0], dtype=bool)
        paired[1:] = (gid[1:] == gid[:-1]) & (ply[1:] == ply[:-1] + 1)
        paired[1:] &= has_meta[:-1]
        cp_prev = np.zeros(gid.shape[0], dtype=np.float64)
        cp_raw = meta[:, 2].astype(np.int64)
        mate_raw = meta[:, 3].astype(np.int64)
        eff = np.where(
            mate_raw != 0,
            mate_to_effective_cp_array(mate_raw),
            np.where(cp_raw == SF_CP_SENTINEL, np.nan, cp_raw.astype(np.float64)),
        )
        cp_prev[1:] = eff[:-1]
        paired &= np.isfinite(cp_prev)
        paired &= np.asarray(z["has_sf_wdl"][:]).astype(bool)
        paired &= np.asarray(z["has_search_wdl"][:]).astype(bool)
        idx = np.flatnonzero(paired)
        if idx.size == 0:
            continue
        if max_rows > 0 and n_seen + idx.size > max_rows:
            idx = idx[: max_rows - n_seen]
        cols["game_id"].append(gid[idx])
        cols["ply_index"].append(ply[idx])
        cols["sf_label_meta"].append(cp_prev[idx])
        cols["has_sf_label_meta"].append(np.ones(idx.size, dtype=bool))
        cols["sf_wdl"].append(np.asarray(z["sf_wdl"][:])[idx].astype(np.float64))
        cols["has_sf_wdl"].append(np.ones(idx.size, dtype=bool))
        cols["search_wdl"].append(np.asarray(z["search_wdl"][:])[idx].astype(np.float64))
        cols["has_search_wdl"].append(np.ones(idx.size, dtype=bool))
        cols["wdl_target"].append(np.asarray(z["wdl_target"][:])[idx].astype(np.int64))
        cols["x"].append(np.asarray(z["x"][:])[idx])
        n_seen += int(idx.size)
        if max_rows > 0 and n_seen >= max_rows:
            break
    if n_skipped_encoding:
        print(f"[value-optimism] skipped {n_skipped_encoding} shards whose input encoding "
              f"differs from the checkpoint's ({history_encoding}, rep_fix={history_rep_fix})")
    if rejected:
        worst = ", ".join(f"{name} ({corr:+.3f})" for name, corr in rejected[:5])
        print(f"[value-optimism] REJECTED {len(rejected)} shards whose SF labels are not "
              f"attached to their own rows (material~sf_wdl rank corr < {attachment_min}): "
              f"{worst}{' ...' if len(rejected) > 5 else ''}")
    if n_seen == 0:
        raise SystemExit(
            "no usable P0-paired rows in the selected shards"
            + (f" ({len(rejected)} shards failed the SF-label attachment gate — widen "
               "--shards to reach older, sound ones)" if rejected else ""),
        )
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    out["_rows_scanned"] = np.array([n_rows_total], dtype=np.int64)
    out["_shards_rejected"] = np.array([len(rejected)], dtype=np.int64)
    # The P0 ruler as an expected score, through the SAME cp-logistic the
    # production SF label is built with, so the net is compared against the map
    # its own target was written in.
    out["_sf_ruler_score"] = cp_to_expected_score(
        out["sf_label_meta"], slope=slope, draw_width_cp=draw_width,
    )
    return out


def _net_scores(
    x: np.ndarray, *, model: torch.nn.Module, device: str, batch_size: int,
) -> np.ndarray:
    ev = LocalModelEvaluator(model, device=device)
    out = np.empty((x.shape[0], 3), dtype=np.float64)
    for s in range(0, x.shape[0], batch_size):
        chunk = np.asarray(x[s:s + batch_size], dtype=np.float32)
        with torch.no_grad():
            _, wdl = ev.evaluate_encoded(chunk)
        wdl = np.asarray(wdl, dtype=np.float64)
        if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
            z = wdl - wdl.max(axis=1, keepdims=True)
            e = np.exp(z)
            wdl = e / e.sum(axis=1, keepdims=True)
        out[s:s + chunk.shape[0]] = wdl
    return out


def _print_table(title: str, stats: list[BucketStat]) -> None:
    print(f"\n=== {title} ===")
    print(
        "  bucket               |     n | games | sf_cp  | SF_rul| net   | target| out   "
        "| tgtSF | srch  | net-SFrul       | tgt-SFrul       | net-tgt         | net-SFrul cp",
    )
    print("  " + "-" * 164)
    for s in stats:
        print(
            f"  {s.name:20s} | {s.n:5d} | {s.n_games:5d} | {s.sf_cp_mean:6.0f} | "
            f"{s.sf_ruler_score:.3f} | {s.net_score:.3f} | {s.target_score:.3f} | "
            f"{s.outcome_score:.3f} | {s.target_sf_score:.3f} | {s.search_score:.3f} | "
            f"{s.net_minus_sf:+.3f} [{s.net_minus_sf_ci[0]:+.3f},{s.net_minus_sf_ci[1]:+.3f}] | "
            f"{s.target_minus_sf:+.3f} [{s.target_minus_sf_ci[0]:+.3f},{s.target_minus_sf_ci[1]:+.3f}] | "
            f"{s.net_minus_target:+.3f} [{s.net_minus_target_ci[0]:+.3f},{s.net_minus_target_ci[1]:+.3f}] | "
            f"{s.net_minus_sf_cp:+7.1f} [{s.net_minus_sf_cp_ci[0]:+.0f},{s.net_minus_sf_cp_ci[1]:+.0f}]",
        )
    print(
        "  All score columns are mean EXPECTED SCORE (W+0.5D) from the scored side's POV; "
        "brackets are 95% game-clustered bootstrap CIs.",
    )
    print(
        "  SF_rul = the stratifying ruler: SF's eval of the row's OWN position (P0), "
        "through the production cp-logistic.",
    )
    print(
        "  tgtSF  = the SF component the blend actually consumes, which is the eval "
        "AFTER the net's move (P1) — so SF_rul minus tgtSF is the net's own move cost, "
        "not a defect.",
    )
    for s in stats:
        print(
            f"  {s.name:20s} optimistic(net>SF_rul) {100 * s.optimistic_frac:5.1f}%  "
            f"net_cp {s.net_cp_mean:+7.1f}  tgt-sf cp {s.target_minus_sf_cp:+7.1f}  "
            f"cp-clamped {100 * s.cp_clamped_frac:4.1f}%  TB-range(<=7men) {100 * s.tb_range_frac:4.1f}%",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="trainer.pt or checkpoint dir")
    ap.add_argument("--config", default="configs/pbt2_small.yaml",
                    help="live config, read for the cp-logistic params and the blend guard")
    ap.add_argument("--run-dir", default=None, help="default: $TRAIN_WORK_DIR or runs/pbt2_small")
    ap.add_argument("--replay-dir", default=None, help="explicit replay_shards dir")
    ap.add_argument("--shards", type=int, default=120,
                    help="newest N shards to scan (0 = all). Newest-first, so this is a "
                         "recency window, not a biased prefix of a sorted file.")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = every paired row in the window")
    ap.add_argument("--min-pieces", type=int, default=0,
                    help="0 (default) SCORES ALL ROWS INCLUDING TABLEBASE RANGE. Unlike "
                         "scripts/value_regret.py, which excludes <=7-man positions because "
                         "the engine plays those from tablebase so the net's value never "
                         "decides a move there, this instrument audits the TRAINING TARGET "
                         "— and TB-range rows are trained on exactly like any other, so "
                         "excluding them would hide part of what the head learns. Pass 8 "
                         "for the play-relevant robustness split; the TB share per bucket "
                         "is printed either way.")
    ap.add_argument("--sf-wdl-frac", type=float, default=None,
                    help="default: the REALIZED value from progress.csv, not the yaml nominal")
    ap.add_argument("--search-wdl-frac", type=float, default=None,
                    help="default: train.search_wdl_frac from --config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu-mem-fraction", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000, help="game-clustered bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edges", default=None, metavar="CP,CP,...",
                    help="override the SF-eval bucket edges in cp. The DEFAULT edges are "
                         "the comparable ones — two runs with different edges are two "
                         "different instruments, so never put their rows in one table. Use "
                         "this to read a level matched to some external claim, then say "
                         "which edges produced the number.")
    ap.add_argument("--attachment-min", type=float, default=SF_LABEL_ATTACHMENT_MIN,
                    help="reject a shard whose material~sf_wdl rank correlation falls below "
                         "this — the ruler-free check that its SF labels sit on their own "
                         "rows. Lower it only to deliberately measure through a known-bad "
                         "window, and say so in the writeup.")
    ap.add_argument("--shuffle-control", action="store_true",
                    help="NEGATIVE CONTROL: permute the net's evaluations across rows. "
                         "Every bucket effect must collapse; if it does not, the scorer "
                         "is reporting structure it invented.")
    ap.add_argument("--dump-json", default=None, metavar="PATH")
    args = ap.parse_args()

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        dev_idx = (int(args.device.split(":", 1)[1]) if ":" in args.device
                   else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=dev_idx)
        print(f"[value-optimism] GPU memory capped at fraction {args.gpu_mem_fraction} on cuda:{dev_idx}")

    # Resolve the label params through production's own resolver on production's
    # own flattened config, so a yaml reorganisation cannot leave this script
    # reading a key from a section it moved out of.
    flat = flatten_run_config_defaults(load_yaml_file(args.config))
    _check_blend_knobs(flat)
    sf_params = resolve_sf_target_params(flat)
    if not bool(sf_params.sf_wdl_use_cp_logistic):
        raise SystemExit(
            "sf_wdl_use_cp_logistic is off; the cp<->score map this script inverts is "
            "not the one production builds its SF label with",
        )
    slope = float(sf_params.sf_wdl_cp_slope)
    draw_width = float(sf_params.sf_wdl_cp_draw_width)

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    if args.sf_wdl_frac is not None:
        sf_frac, sf_frac_src = float(args.sf_wdl_frac), "--sf-wdl-frac"
    else:
        sf_frac, sf_frac_src = _realized_sf_wdl_frac(run_dir)
    search_frac = float(
        args.search_wdl_frac if args.search_wdl_frac is not None
        else flat.get("search_wdl_frac", 0.0)
    )
    game_frac = max(0.0, 1.0 - sf_frac - search_frac)
    print(f"[value-optimism] cp-logistic slope={slope} draw_width={draw_width}")
    print(f"[value-optimism] blend game={game_frac:.2f} sf={sf_frac:.2f} search={search_frac:.2f} "
          f"(sf_wdl_frac realized from {sf_frac_src}; the yaml nominal is decorative)")

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    if bool(getattr(model, "use_dynamic_relations", False)):
        raise SystemExit(
            "checkpoint uses dynamic relations, which the replay shards do not store; "
            "scoring it here would silently evaluate a relation-less model",
        )
    enc = model_encoding_kwargs(model)
    ckpt_hist = normalize_lc0_history_encoding(enc.get("input_history_encoding"))
    ckpt_rep_fix = bool(getattr(model, "history_rep_fix", False))
    print(f"[value-optimism] checkpoint encoding {ckpt_hist} history_rep_fix={ckpt_rep_fix}")

    replay_dir = _resolve_replay_dir(run_dir, args.replay_dir)
    paths = sorted(replay_dir.glob("shard_*.zarr"), key=_shard_num, reverse=True)
    if args.shards > 0:
        paths = paths[: args.shards]
    print(f"[value-optimism] {len(paths)} shards from {replay_dir}")

    data = _load_rows(paths, max_rows=args.max_rows, slope=slope, draw_width=draw_width,
                      history_encoding=ckpt_hist, history_rep_fix=ckpt_rep_fix,
                      attachment_min=float(args.attachment_min))
    x = data["x"]
    n = int(x.shape[0])
    print(f"[value-optimism] {n} P0-paired rows out of {int(data['_rows_scanned'][0])} scanned "
          f"({100.0 * n / max(1, int(data['_rows_scanned'][0])):.1f}%)")

    sf_cp = data["sf_label_meta"]
    sf_wdl = data["sf_wdl"]
    search_wdl = data["search_wdl"]
    game_oh = np.zeros((n, 3), dtype=np.float64)
    game_oh[np.arange(n), data["wdl_target"]] = 1.0

    # Mirror losses.py: clamp negatives, renormalise, then convex-combine. Every
    # row here carries both optional components (the loader required it), so the
    # missing-component fallback to the raw game result never applies.
    def _norm(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, 0.0, None)
        return q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)

    sf_probs = _norm(sf_wdl)
    search_probs = _norm(search_wdl)
    target = game_frac * game_oh + sf_frac * sf_probs + search_frac * search_probs

    attach = sf_label_attachment_corr(x[:, 0:12], sf_wdl)
    print(f"[value-optimism] pooled SF-label attachment (material~sf_wdl rank corr) = {attach:+.3f}")

    net_wdl = _net_scores(x, model=model, device=args.device, batch_size=args.batch_size)
    piece_count = (np.asarray(x[:, 0:12], dtype=np.float32) > 0.5).sum(axis=(1, 2, 3))

    rows = OptimismRows(
        sf_cp=sf_cp,
        sf_ruler_score=data["_sf_ruler_score"],
        net_score=expected_score(net_wdl),
        target_score=expected_score(target),
        outcome_score=expected_score(game_oh),
        target_sf_score=expected_score(sf_probs),
        search_score=expected_score(search_probs),
        game_id=data["game_id"],
        piece_count=piece_count.astype(np.int64),
    )
    if args.min_pieces > 0:
        keep = rows.piece_count >= int(args.min_pieces)
        print(f"[value-optimism] --min-pieces {args.min_pieces}: dropped {int((~keep).sum())} rows")
        rows = rows.select(keep)
    if args.shuffle_control:
        rows = rows.with_shuffled_net(np.random.default_rng(args.seed + 1))
        print("[value-optimism] NEGATIVE CONTROL ACTIVE: net evaluations permuted across rows")

    edges = SF_EVAL_BUCKET_EDGES
    if args.edges:
        edges = tuple(sorted(float(v) for v in str(args.edges).split(",") if v.strip()))
        print(f"[value-optimism] NON-DEFAULT bucket edges {edges} — not comparable with "
              f"default-edge runs {SF_EVAL_BUCKET_EDGES}")
    stats = score_buckets(rows, slope=slope, draw_width_cp=draw_width,
                          n_boot=args.bootstrap, seed=args.seed, edges=edges)
    label = "NEGATIVE CONTROL (shuffled)" if args.shuffle_control else "value optimism by SF-eval bucket"
    _print_table(f"{label} @ {args.checkpoint}", stats)

    spread = bucket_net_score_spread(stats)
    asym = tail_asymmetry(stats, edges)
    print(f"\n  control statistic (max-min bucket mean net score): {spread:.4f}"
          "   [shuffled control must collapse this to ~0]")
    if asym is not None:
        names = bucket_names_for(edges)
        print(f"  tail asymmetry (net-SFrul in '{names[0]}' + net-SFrul in "
              f"'{names[-1]}'): {asym:+.4f}")
        print("    Symmetric tails = regression toward the mean off a noisy ruler, not a "
              "directional defect. Only this sum is evidence of real optimism.")
    print("\n  Absolute cp bars here are NOT subject to the FEN-only defect (M10): the net "
          "is fed the stored production x planes, history included.")

    if args.dump_json:
        payload = {
            "checkpoint": args.checkpoint,
            "replay_dir": str(replay_dir),
            "shards": len(paths),
            "rows": int(rows.sf_cp.shape[0]),
            "slope": slope, "draw_width_cp": draw_width, "cp_clamp": CP_CLAMP,
            "game_frac": game_frac, "sf_wdl_frac": sf_frac, "search_wdl_frac": search_frac,
            "sf_wdl_frac_source": sf_frac_src,
            "shuffle_control": bool(args.shuffle_control),
            "min_pieces": int(args.min_pieces),
            "control_statistic_net_score_spread": spread,
            "tail_asymmetry": asym,
            "buckets": [vars(s) for s in stats],
        }
        Path(args.dump_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[value-optimism] dump -> {args.dump_json}")


if __name__ == "__main__":
    main()
