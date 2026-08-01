#!/usr/bin/env python3
"""Value-head optimism by Stockfish-eval stratum, on PRODUCTION replay rows.

Answers "does the value head know it is losing?" without letting the net choose
the sample. Rows come from the live replay shards; the stratum comes from a
Stockfish evaluation of the row's OWN position; and the blended training target
is scored in the same buckets as the head, which is what separates "the head
fails to fit its target" from "the target is itself optimistic".

**⚑ THE HEAD/TARGET ARM IS THE SELFPLAY SUBSET, NOT "THE TRAINING
DISTRIBUTION".** It needs two rows that are consecutive plies of one game (see
the P0 ruler note below), and curriculum games never produce such a pair —
measured pairing rate: selfplay 24.2%, curriculum **0.00%**, while the window is
36.0% curriculum rows. So that arm is 100% selfplay BY CONSTRUCTION. This
matters far beyond sample size: **the PID-handicapped Stockfish only plays in
curriculum games**, so any claim about the handicap read off the head/target arm
is read off the one population where the mechanism cannot operate. The
OUTCOME-CALIBRATION arm below exists for exactly that reason: it needs no
pairing and no model, covers every labelled row, and is split by
``is_selfplay``. Read it before saying anything about the handicap.

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

The P0 ruler, and why it needs no new Stockfish compute: a row's SF label is the
evaluation of the position AFTER that row's move, from the mover-to-come's point
of view. So for two rows that are consecutive plies of one game, the EARLIER
row's stored ``sf_label_meta`` eval is an evaluation of the LATER row's own
position, already in the later row's point of view — no flip, no re-query. That
is the same one-ply shift ``sf_p0_policy_target`` uses. Labels are the
production ones (~698k nodes, MultiPV 40, median depth ~12); they are shallower
than the frozen audit set's 1M-node MultiPV-10 labels, which is the price of
scoring the exact rows the target was built from.

Reading the output:

- ``net-tgt`` is the PRIMARY axis. Both sides are measured on the same row and
  neither is computed from the bucket, so a head that fit its target perfectly
  would read zero in every bucket under any bucketing.
- ``net-SFrul`` is the ruler-relative axis and can in principle be biased by
  bucketing on a noisy ruler. Whether it actually is, is decided empirically by
  the ``out`` column: the realized outcome is an unbiased draw of the true
  value, so the printed perfect-head null says which way the bias runs.
- ``tail_asymmetry`` has a non-zero null under BOTH the shuffle control and a
  perfect head; both are printed. Never read it against zero.

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
    AxisReading,
    SF_EVAL_BUCKET_EDGES,
    SF_DESYNC_MAX,
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
    BucketStat,
    OptimismRows,
    bucket_names_for,
    bucket_net_score_spread,
    cp_to_expected_score,
    expected_score,
    outcome_calibration,
    perfect_head_tail_asymmetry,
    score_buckets,
    sf_bestmove_is_first_legal_rate,
    sf_label_attachment_corr,
    sf_multipv_missing_rate,
    tail_asymmetry,
    tail_asymmetry_ci,
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

# Knobs the trainer RESOLVES per iteration and logs, so progress.csv is the only
# artifact that holds the answer (rl_loop_audit method rule 12). `sf_wdl_frac` is
# the live case: the yaml says 0.50, the realized value is the 0.45 floor.
# `sf_wdl_temperature` is logged from `trainer.sf_wdl_temperature`
# (tune/trainable_report.py), so the same rule applies to it.
_REALIZED_FROM_PROGRESS: tuple[str, ...] = ("sf_wdl_frac", "sf_wdl_temperature")

# Knobs with NO realized column anywhere. Nothing re-resolves them per
# iteration, so the two artifacts that can disagree are the live yaml (current,
# and these are live-reloadable) and the trial's params.json (what it launched
# with). Neither is authoritative alone — params.json is stale after a live
# reload, the yaml is wrong after a resume restored the trial config — so this
# script REQUIRES THEM TO AGREE and refuses to guess when they do not. That is
# as far as the available artifacts allow the realized-value rule to be applied;
# the residual gap is that "they agree" is not the same as "the trainer used it".
_CROSSCHECK_YAML_VS_PARAMS: tuple[str, ...] = (
    "search_wdl_frac", "sf_search_dampen_sf_low", "sf_search_dampen_sf_high",
    "use_adjusted_wdl_target",
)


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


def _last_progress_row(run_dir: Path, key: str) -> tuple[dict[str, str], Path]:
    import csv

    trial = latest_trial_dir(run_dir, required=True)
    path = trial / "progress.csv"
    if not path.exists():
        raise SystemExit(f"no progress.csv at {path}; cannot read realized {key}")
    last: dict[str, str] | None = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get(key):
                last = row
    if last is None:
        raise SystemExit(
            f"{path} has no non-empty {key} column. It is in _REALIZED_FROM_PROGRESS "
            "because the trainer resolves it per iteration; if it stopped being logged, "
            "decide where the realized value now lives rather than falling back to the "
            "yaml nominal.",
        )
    return last, path


def _trial_params(run_dir: Path) -> tuple[dict[str, Any], Path]:
    trial = latest_trial_dir(run_dir, required=True)
    path = trial / "params.json"
    if not path.exists():
        raise SystemExit(f"no params.json at {path}; cannot cross-check the blend knobs")
    return json.loads(path.read_text(encoding="utf-8")), path


def resolve_blend_knobs(flat: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Every blend knob, from the artifact that actually holds its value.

    Presence is REQUIRED at every source. An absent key must never fall back to
    "the neutral value" — that is the `reco_diff misses absent keys` shape, where
    a comparison that cannot see a missing key reports agreement. Here it would
    let a knob that was removed from the config silently pass the neutrality
    gate below.
    """
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}

    row, prog_path = _last_progress_row(run_dir, _REALIZED_FROM_PROGRESS[0])
    for key in _REALIZED_FROM_PROGRESS:
        if not row.get(key):
            raise SystemExit(
                f"{prog_path} has no realized {key} on the row it logs "
                f"{_REALIZED_FROM_PROGRESS[0]}; refusing to substitute the yaml nominal",
            )
        resolved[key] = float(row[key])
        sources[key] = f"progress.csv iter={row.get('training_iteration')} (REALIZED)"

    params, params_path = _trial_params(run_dir)
    for key in _CROSSCHECK_YAML_VS_PARAMS:
        if key not in flat:
            raise SystemExit(f"{key} is absent from the config; it must be present to be checked")
        if key not in params:
            raise SystemExit(f"{key} is absent from {params_path}; cannot cross-check it")
        a, b = flat[key], params[key]
        same = (bool(a) == bool(b)) if isinstance(a, bool) else (float(a) == float(b))
        if not same:
            raise SystemExit(
                f"{key} disagrees: config says {a!r}, {params_path} says {b!r}. One of a "
                "live reload and a resume-restored trial config won and this script "
                "cannot tell which. Determine what the trainer used and pass it "
                "explicitly.",
            )
        resolved[key] = a
        sources[key] = "config == params.json (no realized column exists)"

    for key in sorted(resolved):
        print(f"[value-optimism] {key} = {resolved[key]!r}  <- {sources[key]}")
    return resolved


def _check_blend_knobs(resolved: dict[str, Any]) -> None:
    """The blend is a plain convex combination only while these are neutral."""
    bad = []
    for key, want in _NEUTRAL_BLEND_KNOBS.items():
        if key not in resolved:
            raise SystemExit(f"{key} was not resolved; the neutrality gate cannot run on it")
        got = resolved[key]
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
    desync_max: float, multipv_miss_max: float, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Collect P0-paired rows from the newest shards, newest first."""
    import zarr

    keys = (
        "game_id", "ply_index", "sf_label_meta", "has_sf_label_meta", "sf_wdl",
        "has_sf_wdl", "search_wdl", "has_search_wdl", "wdl_target", "x",
    )
    cols: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    # The all-rows arm: no pairing, no model, so it keeps the curriculum rows the
    # paired arm structurally cannot reach.
    allrows: dict[str, list[np.ndarray]] = {
        "ruler_cp": [], "outcome_score": [], "is_selfplay": [], "game_id": [],
    }
    n_seen = 0
    n_rows_total = 0
    n_skipped_encoding = 0
    scanned_selfplay = paired_selfplay = 0
    scanned_curriculum = paired_curriculum = 0
    rejected: list[tuple[str, str]] = []
    rejected_rows = 0
    desync_seen: list[float] = []
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
        # INTEGRITY GATE. Two enforced axes plus one reported diagnostic; each
        # enforced axis earns its place by a shard the other cannot catch.
        #
        #   attachment  : rank corr of material vs the shard's own SF label.
        #                 Catches TOTAL detachment — the label block landing on
        #                 the wrong ROWS, which leaves every per-row field
        #                 internally consistent and so can coexist with a
        #                 perfect MultiPV rate. On THIS trial it rejects no
        #                 shard the MultiPV axis misses (ATT-only is empty), so
        #                 it is kept on MECHANISM plus its own honest
        #                 separation — lowest accepted +0.4189 vs detached
        #                 ~0.00 (max +0.2497), a 0.169 gap the 0.25 line did not
        #                 create — and NOT on a reject count, which would
        #                 condemn it here. Redundancy against this trial's
        #                 failure modes is not redundancy against the next.
        #   multipv miss: share of labelled rows with no MultiPV block. Catches
        #                 PARTIAL corruption — a Stockfish UCI desync leaves an
        #                 eval whose candidate list did not survive. The
        #                 justification is the SHAPE of the sound distribution,
        #                 not the gap: 89.9% of accepted shards read EXACTLY
        #                 0.000000 (median 0.000000, p99 0.004603, max
        #                 0.008032), a hard floor at zero, so any material rate
        #                 is anomalous wherever the cut sits. The 0.002478 gap
        #                 to the first rejected shard (0.010511, 22.2x headroom
        #                 over the accepted p90 of 0.000450) is secondary and
        #                 partly circular — it exists only after removing the
        #                 122 shards this threshold rejects. Sensitivity:
        #                 0.008/0.009/0.01 all reject 122-123.
        #   desync rate : bestmove-is-first-legal. REPORTED, not enforced by
        #                 default. Sound max 0.1496 vs corrupt min 0.1505 — any
        #                 threshold sits in a 0.0009 gap between two adjacent
        #                 order statistics, which is the "a gate tuned on the
        #                 episode that produced it detects that episode" trap
        #                 this file's own method note warns about. At 0.15 it
        #                 leaked seven shards sitting inside runs of rejects,
        #                 and it catches nothing the other two miss.
        #
        # A non-"ok" status on an ENFORCED axis is a reject with its reason
        # named: "field missing", "too few rows" and "genuinely bad" must not
        # share one string, or a pre-schema shard is swallowed under it.
        n_shard_rows = int(gid.shape[0])
        has_sf_wdl_all = np.asarray(z["has_sf_wdl"][:]).astype(bool)
        attach = sf_label_attachment_corr(
            np.asarray(z["x"][:, 0:12]), np.asarray(z["sf_wdl"][:]).astype(np.float64),
            has_sf_wdl_all,
        )
        try:
            multipv = sf_multipv_missing_rate(
                np.asarray(z["has_sf_multipv_raw"][:]), has_sf_wdl_all,
            )
        except KeyError:
            multipv = AxisReading(float("nan"), "field_missing")
        try:
            desync = sf_bestmove_is_first_legal_rate(
                np.asarray(z["sf_move_index"][:]), np.asarray(z["sf_legal_mask"][:]),
                np.asarray(z["has_sf_move"][:]).astype(bool)
                & np.asarray(z["has_sf_legal_mask"][:]).astype(bool),
            )
        except KeyError:
            desync = AxisReading(float("nan"), "field_missing")

        verdict: str | None = None
        if not attach.usable:
            verdict = attach.describe("attachment")
        elif attach.value < attachment_min:
            verdict = f"attachment {attach.value:+.4f} < {attachment_min}"
        elif not multipv.usable:
            verdict = multipv.describe("multipv-miss")
        elif multipv.value > multipv_miss_max:
            verdict = f"multipv-miss {multipv.value:.6f} > {multipv_miss_max}"
        elif desync_max < 1.0 and not desync.usable:
            verdict = desync.describe("desync")
        elif desync_max < 1.0 and desync.value > desync_max:
            verdict = f"desync {desync.value:.4f} > {desync_max}"
        if verdict is not None:
            rejected.append((path.name, verdict))
            rejected_rows += n_shard_rows
            continue
        if desync.usable:
            desync_seen.append(desync.value)

        # Row i is P0-paired iff row i-1 is the immediately preceding ply of the
        # same game AND carries a usable eval. The ply-gap check is what makes
        # the earlier row's label an evaluation of THIS row's position; without
        # it the shift silently spans a gap of several plies.
        #
        # ⚑ IT IS ALSO A POPULATION FILTER, NOT JUST A SPARSITY ONE. Curriculum
        # rows pair at 0.00% — the net does not move on consecutive plies there —
        # so this arm is 100% selfplay, and the PID-handicapped opponent plays
        # ONLY in curriculum. The composition is printed below so this cannot be
        # overlooked; the all-rows arm is collected for the same reason.
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

        selfplay = np.asarray(z["is_selfplay"][:]).astype(bool)
        scanned_selfplay += int(selfplay.sum())
        scanned_curriculum += int((~selfplay).sum())
        paired_selfplay += int((paired & selfplay).sum())
        paired_curriculum += int((paired & ~selfplay).sum())

        # All-rows arm: the row's OWN label, which is the eval AFTER its move
        # from the opponent's POV, so the scored side's POV is the negation.
        wt_all = np.asarray(z["wdl_target"][:]).astype(np.int64)
        own_ok = has_meta & np.isfinite(eff)
        allrows["ruler_cp"].append(-eff[own_ok])
        allrows["outcome_score"].append(
            ((wt_all == 0) * 1.0 + (wt_all == 1) * 0.5)[own_ok],
        )
        allrows["is_selfplay"].append(selfplay[own_ok])
        allrows["game_id"].append(gid[own_ok])

        idx = np.flatnonzero(paired)
        if idx.size == 0:
            continue
        if max_rows > 0 and n_seen + idx.size > max_rows:
            # A PREFIX would be biased: rows are stored in game order, so the
            # first k rows of a shard over-weight its earliest games and their
            # opening plies. Subsample uniformly instead, then restore order.
            take = max_rows - n_seen
            pick = np.random.default_rng(seed + len(cols["game_id"])).choice(
                idx.size, size=take, replace=False,
            )
            idx = idx[np.sort(pick)]
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
        worst = ", ".join(f"{name} ({why})" for name, why in rejected[:6])
        print(f"[value-optimism] REJECTED {len(rejected)} shards ({rejected_rows} rows, "
              f"{100.0 * rejected_rows / max(1, n_rows_total):.1f}% of scanned) failing the "
              f"integrity gate (attachment >= {attachment_min}, multipv-miss <= "
              f"{multipv_miss_max}): {worst}{' ...' if len(rejected) > 6 else ''}")
    else:
        print("[value-optimism] integrity gate: 0 shards rejected "
              f"(attachment >= {attachment_min}, multipv-miss <= {multipv_miss_max})")
    if desync_seen:
        print(f"[value-optimism] DIAGNOSTIC (not enforced at --desync-max {desync_max}): "
              f"bestmove-is-first-legal rate over ACCEPTED shards — median "
              f"{float(np.median(desync_seen)):.4f}, max {max(desync_seen):.4f}. "
              "Sound baseline is ~0.08; a max far above that means a corruption mode "
              "the enforced axes are not seeing.")
    if n_seen == 0:
        raise SystemExit(
            "no usable P0-paired rows in the selected shards"
            + (f" ({len(rejected)} shards failed the integrity gate — widen --shards to "
               "reach older, sound ones)" if rejected else ""),
        )
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    for k, v in allrows.items():
        out[f"_all_{k}"] = np.concatenate(v) if v else np.zeros(0)
    out["_rows_scanned"] = np.array([n_rows_total], dtype=np.int64)
    out["_shards_rejected"] = np.array([len(rejected)], dtype=np.int64)
    out["_rows_rejected"] = np.array([rejected_rows], dtype=np.int64)
    out["_paired_selfplay"] = np.array([paired_selfplay], dtype=np.int64)
    out["_paired_curriculum"] = np.array([paired_curriculum], dtype=np.int64)

    scanned = max(1, scanned_selfplay + scanned_curriculum)
    paired_n = max(1, paired_selfplay + paired_curriculum)
    print(
        f"[value-optimism] window composition: selfplay {scanned_selfplay} "
        f"({100.0 * scanned_selfplay / scanned:.1f}%), curriculum {scanned_curriculum} "
        f"({100.0 * scanned_curriculum / scanned:.1f}%)",
    )
    print(
        f"[value-optimism] PAIRED-SET composition: selfplay {paired_selfplay} "
        f"({100.0 * paired_selfplay / paired_n:.1f}%, pairing rate "
        f"{100.0 * paired_selfplay / max(1, scanned_selfplay):.2f}%), curriculum "
        f"{paired_curriculum} ({100.0 * paired_curriculum / paired_n:.1f}%, pairing rate "
        f"{100.0 * paired_curriculum / max(1, scanned_curriculum):.2f}%)",
    )
    if paired_curriculum == 0 and scanned_curriculum > 0:
        print(
            "[value-optimism] ⚑ THE PAIRED SET IS 100% SELFPLAY. The ply-gap guard "
            "structurally excludes every curriculum row, and the PID-handicapped "
            "opponent plays ONLY in curriculum games — so NOTHING in the head/target "
            "table below can support or refute a claim about the handicap. Use the "
            "outcome-calibration arm for that.",
        )
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
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 (default) = every paired row in the window. When it truncates, "
                         "rows are subsampled UNIFORMLY within the shard using --seed, not "
                         "taken as a prefix: shard rows are stored in game order, so a "
                         "prefix over-weights the earliest games and their opening plies.")
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
                         "rows. Catches TOTAL detachment only. Lower it only to deliberately "
                         "measure through a known-bad window, and say so in the writeup.")
    ap.add_argument("--multipv-miss-max", type=float, default=SF_MULTIPV_MISS_MAX,
                    help="reject a shard where more than this share of SF-LABELLED rows "
                         "carries no MultiPV block — the sharp fingerprint of a Stockfish "
                         "UCI desync, where the label sits on the right row but answers a "
                         "DIFFERENT position. Sound shards max at 0.008032 (median exactly "
                         "0.000000 on 89.9% of accepted shards, p99 0.004603, max 0.008032) "
                         "— a hard floor at zero, so any material rate is anomalous wherever "
                         "the cut sits. Corrupt shards start at 0.010511. The attachment axis "
                         "does NOT catch this. Set to 1.0 to measure the contamination effect.")
    ap.add_argument("--desync-max", type=float, default=SF_DESYNC_MAX,
                    help="bestmove-is-first-legal rate. DIAGNOSTIC ONLY by default (1.0 = "
                         "never reject): its sound max is 0.1496 and its corrupt min 0.1505, "
                         "so every threshold sits in a 0.0009 gap between adjacent order "
                         "statistics of the same quantity, and 0.15 leaked seven shards "
                         "sitting inside runs of rejects. It also catches nothing the "
                         "enforced axes miss. The value is always printed. Set it below 1.0 "
                         "only with a stated reason.")
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
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    knobs = resolve_blend_knobs(flat, run_dir)
    _check_blend_knobs(knobs)
    sf_params = resolve_sf_target_params(flat)
    if not bool(sf_params.sf_wdl_use_cp_logistic):
        raise SystemExit(
            "sf_wdl_use_cp_logistic is off; the cp<->score map this script inverts is "
            "not the one production builds its SF label with",
        )
    slope = float(sf_params.sf_wdl_cp_slope)
    draw_width = float(sf_params.sf_wdl_cp_draw_width)

    sf_frac = float(args.sf_wdl_frac if args.sf_wdl_frac is not None else knobs["sf_wdl_frac"])
    sf_frac_src = "--sf-wdl-frac" if args.sf_wdl_frac is not None else "progress.csv (realized)"
    search_frac = float(
        args.search_wdl_frac if args.search_wdl_frac is not None else knobs["search_wdl_frac"]
    )
    # Mirror losses.py:443-447 exactly, including the over-unity branch. Today
    # the fracs sum to 0.65 so the branch is dead, but "currently unreachable" is
    # how a reconstruction silently stops matching production after a config
    # change — and the two differ in kind, not degree: production RENORMALISES
    # sf/search onto the simplex, while `1 - sf - search` clamped at zero would
    # leave an unnormalised target.
    sf_frac = max(0.0, sf_frac)
    search_frac = max(0.0, search_frac)
    blend_sum = sf_frac + search_frac
    if blend_sum > 1.0:
        sf_frac /= blend_sum
        search_frac /= blend_sum
        game_frac = 0.0
        print(f"[value-optimism] blend fracs summed to {blend_sum:.3f} > 1 and were "
              "renormalised, matching losses.py; the game-outcome share is now zero")
    else:
        game_frac = 1.0 - blend_sum
    print(f"[value-optimism] cp-logistic slope={slope} draw_width={draw_width}")
    print(f"[value-optimism] blend game={game_frac:.2f} sf={sf_frac:.2f} search={search_frac:.2f} "
          f"(sf_wdl_frac from {sf_frac_src}; the yaml nominal 0.50 is decorative)")

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
    print(
        "[value-optimism] tablebase-range rows are "
        + (f"EXCLUDED (--min-pieces {args.min_pieces})" if args.min_pieces > 0 else
           "INCLUDED (--min-pieces 0, the default). scripts/value_regret.py excludes "
           "<=7-man because tablebase decides those moves; this instrument audits the "
           "TRAINING TARGET, which trains on them. Pass --min-pieces 8 for the "
           "play-relevant split"),
    )

    data = _load_rows(paths, max_rows=args.max_rows, slope=slope, draw_width=draw_width,
                      history_encoding=ckpt_hist, history_rep_fix=ckpt_rep_fix,
                      attachment_min=float(args.attachment_min),
                      desync_max=float(args.desync_max),
                      multipv_miss_max=float(args.multipv_miss_max), seed=int(args.seed))
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

    attach_pooled = sf_label_attachment_corr(x[:, 0:12], sf_wdl)
    print("[value-optimism] pooled SF-label attachment (material~sf_wdl rank corr) = "
          + attach_pooled.describe("corr"))

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

    names = bucket_names_for(edges)
    spread = bucket_net_score_spread(stats)
    asym = tail_asymmetry(stats, edges)
    null = perfect_head_tail_asymmetry(stats, edges)
    asym_ci = tail_asymmetry_ci(rows, n_boot=args.bootstrap, seed=args.seed, edges=edges)

    print("\n  --- PRIMARY (artifact-free): net minus its OWN target ---")
    for s in stats:
        print(f"  {s.name:20s} net-tgt {s.net_minus_target:+.4f} "
              f"[{s.net_minus_target_ci[0]:+.4f},{s.net_minus_target_ci[1]:+.4f}]")
    tails = [s for s in stats if s.name in (names[0], names[-1])]
    if len(tails) == 2:
        print(f"  tail sum of net-tgt ('{names[0]}' + '{names[-1]}'): "
              f"{tails[0].net_minus_target + tails[1].net_minus_target:+.4f}   "
              "[near zero = the head is compressed symmetrically, i.e. no DIRECTIONAL "
              "losing-position defect; the per-bucket magnitudes are still real]")

    print(f"\n  control statistic (max-min bucket mean net score): {spread:.4f}"
          "   [shuffled control must collapse this to ~0]")
    if asym is not None:
        ci_txt = f" CI [{asym_ci[0]:+.4f},{asym_ci[1]:+.4f}]" if asym_ci else ""
        print(f"  tail asymmetry of net-SFrul, TAIL PAIR '{names[0]}' + '{names[-1]}'"
              f" (NOT the +-300 mirror pair unless those are the edges): {asym:+.4f}{ci_txt}")
        if null is not None:
            print(f"    perfect-head null for the SAME pair (outcome-SFrul): {null:+.4f}"
                  f"   -> excess {asym - null:+.4f}")
            print("    The null is the empirical test of whether bucketing biases this "
                  "axis. A null BELOW zero means the true value in the losing tail is "
                  "more extreme than the ruler, i.e. bucketing does NOT manufacture "
                  "optimism here and the head's compression is real.")
        print("    Its shuffle-control null is also non-zero (-0.0051 measured live); "
              "run --shuffle-control on this same sample before quoting a level.")
    print("\n  Absolute cp bars here are NOT subject to the FEN-only defect (M10): the net "
          "is fed the stored production x planes, history included.")

    # --- The arm that can see the PID handicap -------------------------------
    all_cp = data["_all_ruler_cp"]
    calib: dict[str, list] = {}
    if all_cp.size:
        print("\n  --- OUTCOME CALIBRATION (all labelled rows, no pairing, no model) ---")
        print("  Did games actually score better than the objective eval says? The "
              "handicap mechanism lives HERE, not in the table above.")
        print("  ⚑ 'ruler' in THIS block is the P1 eval (the row's OWN label, i.e. AFTER "
              "its move) — a DIFFERENT ruler from the P0 one the head/target table uses. "
              "That is what lets it cover unpaired rows; do not read the two side by side "
              "as one quantity.")
        print(f"  {'bucket':22s} {'population':11s} {'n':>7s} {'ruler':>7s} {'outcome':>8s} "
              f"{'out-ruler':>10s}")
        for pop, mask in (
            ("selfplay", data["_all_is_selfplay"].astype(bool)),
            ("curriculum", ~data["_all_is_selfplay"].astype(bool)),
        ):
            if not mask.any():
                continue
            rowsc = outcome_calibration(
                ruler_cp=all_cp[mask], outcome_score=data["_all_outcome_score"][mask],
                game_id=data["_all_game_id"][mask], slope=slope, draw_width_cp=draw_width,
                edges=edges, n_boot=args.bootstrap, seed=args.seed,
            )
            calib[pop] = rowsc
            for c in rowsc:
                print(f"  {c.name:22s} {pop:11s} {c.n:7d} {c.ruler_score:7.3f} "
                      f"{c.outcome_score:8.3f} {c.delta:+10.3f} "
                      f"[{c.ci[0]:+.3f},{c.ci[1]:+.3f}]")
        print("  A LARGE POSITIVE out-ruler in the losing buckets of the CURRICULUM row "
              "is the PID-handicapped opponent failing to convert — the mechanism "
              "train/losses.py:459-463 already documents. The selfplay row cannot show "
              "it: no handicapped opponent plays there.")

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
            "attachment_min": float(args.attachment_min),
            "desync_max": float(args.desync_max),
            "multipv_miss_max": float(args.multipv_miss_max),
            "shards_rejected": int(data["_shards_rejected"][0]),
            "rows_rejected": int(data["_rows_rejected"][0]),
            "control_statistic_net_score_spread": spread,
            "tail_asymmetry": asym,
            "tail_asymmetry_ci": asym_ci,
            "tail_asymmetry_pair": [names[0], names[-1]],
            "perfect_head_tail_asymmetry_null": null,
            # Derived from the MEASURED counts, never asserted: a hardcoded True
            # would keep claiming selfplay-only after the pairing logic changed.
            "paired_selfplay_rows": int(data["_paired_selfplay"][0]),
            "paired_curriculum_rows": int(data["_paired_curriculum"][0]),
            "paired_set_is_selfplay_only": bool(int(data["_paired_curriculum"][0]) == 0),
            "buckets": [vars(s) for s in stats],
            "outcome_calibration": {k: [vars(c) for c in v] for k, v in calib.items()},
        }
        Path(args.dump_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[value-optimism] dump -> {args.dump_json}")


if __name__ == "__main__":
    main()
