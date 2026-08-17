#!/usr/bin/env python3
"""Measure how much ``policy_soft_target`` differs from ``policy_target``.

How the two targets are constructed (selfplay/finalize.py:finalize_game):

- ``policy_target`` is ``rec.policy_probs`` — the MCTS visit/improved
  distribution at the move-selection temperature, produced on NETWORK turns
  (selfplay/network_turn.py); Syzygy DTZ overrides may replace it for
  TB-covered positions (``tb_policy_overrides``). It is then mapped to the
  configured policy encoding.
- ``policy_soft_target`` is ``apply_policy_temperature(eff_probs,
  soft_policy_temp)`` applied to the SAME post-override distribution —
  i.e. ``p^(1/T)`` renormalized (selfplay/temperature.py:apply_policy_temperature).
  ⚑ This line used to assert "T = 3.0 in production". The live yaml has said
  ``soft_policy_temp: 2.0`` for as long as anyone checked, so the claim was
  simply wrong — which is why T is no longer written down here at all. It is
  RECOVERED from the stored arrays by ``_recover_soft_policy_temp`` and
  compared against the live config at run time; see the ``[shape]`` line the
  probe prints above its table.

So the soft target is a deterministic retempering of the hard target, not an
independent signal. Construction does NOT differ between selfplay
(model-vs-model) and curriculum (SF-opponent) games: samples are only emitted
on network turns, and both game types flow through the same ``finalize_game``
path; the ``is_selfplay`` flag tags the game type. The only way the two
targets coincide exactly is when the visit distribution is (near-)one-hot —
``p^(1/T)`` fixes one-hot vectors for EVERY T — so divergence here measures how
often search output is still multi-modal.

Streams the most recent shards from a replay dir and reports, per source
type (selfplay vs curriculum) and game phase (piece-count thresholds 13/22,
matching the model's phase buckets):

  * KL(policy_target || policy_soft_target) and reverse KL
  * total-variation distance
  * fraction with TV < 0.01 ("effectively identical")
  * argmax agreement rate

Usage::

    PYTHONPATH=. python3 scripts/probe_policy_targets.py \\
        --replay-dir runs/pbt2_small/tune/<trial>/replay_shards \\
        --positions 200000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chess_anti_engine.utils.architecture import DEFAULT_PHASE_PIECE_THRESHOLDS
from chess_anti_engine.replay.shard import (
    iter_shard_paths,
    load_shard_arrays,
    shard_index,
)

_EPS = 1e-12
_IDENTICAL_TV = 0.01
# Imported, not re-typed. This was the FOURTH copy of the same two numbers
# (utils/architecture, eval/audit, train/losses, here); a probe that silently
# stopped agreeing with the trainer and the audit would be comparing buckets
# that no longer name the same positions.
_PHASE_THRESHOLDS = DEFAULT_PHASE_PIECE_THRESHOLDS
_PHASES = ("endgame", "middlegame", "opening")
_SOURCES = ("selfplay", "curriculum", "unknown")


def _phase_bucket(piece_counts: np.ndarray) -> np.ndarray:
    low, high = _PHASE_THRESHOLDS
    phase = np.ones(piece_counts.shape[0], dtype=np.int64)
    phase[piece_counts <= low] = 0
    phase[piece_counts > high] = 2
    return phase


def _row_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized per-row metrics for (n, P) float32 distributions."""
    p = np.maximum(p.astype(np.float64), 0.0)
    q = np.maximum(q.astype(np.float64), 0.0)
    p_sum = p.sum(axis=1, keepdims=True)
    q_sum = q.sum(axis=1, keepdims=True)
    valid = (p_sum[:, 0] > 0) & (q_sum[:, 0] > 0)
    p = p / np.maximum(p_sum, _EPS)
    q = q / np.maximum(q_sum, _EPS)

    tv = 0.5 * np.abs(p - q).sum(axis=1)
    kl_pq = np.where(p > 0, p * (np.log(p + _EPS) - np.log(q + _EPS)), 0.0).sum(axis=1)
    kl_qp = np.where(q > 0, q * (np.log(q + _EPS) - np.log(p + _EPS)), 0.0).sum(axis=1)
    argmax_agree = (p.argmax(axis=1) == q.argmax(axis=1)).astype(np.float64)
    return {
        "tv": tv, "kl_pq": kl_pq, "kl_qp": kl_qp,
        "argmax_agree": argmax_agree, "valid": valid,
    }


def _recover_soft_policy_temp(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row estimate of the T that produced ``q`` from ``p``, or NaN.

    ``q = p**(1/T) / Z`` implies, for any reference index r::

        log q_i - log q_r = (1/T) * (log p_i - log p_r)

    so with ``A_i = log p_i - log p_r`` and ``B_i = log q_i - log q_r`` the
    least-squares slope through the origin, ``sum(A*B) / sum(B*B)``, IS T. No
    config value enters the estimate — it is read out of the stored arrays.

    ⚑ This is the guard that a `soft_policy_temp` config read could not be.
    Reading the key proves the operator's config says 2.0; it says nothing
    about the temperature the SHARDS on disk were written at, and shards
    outlive config edits by the length of the replay window. Comparing the
    recovered T against the live key is a value read on both sides.

    One-hot rows carry no information (``p**(1/T)`` fixes them for every T) and
    return NaN, which is also why this probe's headline TV statistic is small
    whenever search output is peaked.
    """
    tau = 1e-6
    p64 = np.maximum(p.astype(np.float64), 0.0)
    q64 = np.maximum(q.astype(np.float64), 0.0)
    p64 /= np.maximum(p64.sum(axis=1, keepdims=True), _EPS)
    q64 /= np.maximum(q64.sum(axis=1, keepdims=True), _EPS)
    mask = (p64 > tau) & (q64 > tau)
    ref = p64.argmax(axis=1)
    rows = np.arange(p64.shape[0])
    log_p = np.where(mask, np.log(np.maximum(p64, _EPS)), 0.0)
    log_q = np.where(mask, np.log(np.maximum(q64, _EPS)), 0.0)
    a = np.where(mask, log_p - log_p[rows, ref][:, None], 0.0)
    b = np.where(mask, log_q - log_q[rows, ref][:, None], 0.0)
    num = (a * b).sum(axis=1)
    den = (b * b).sum(axis=1)
    out = np.full(p64.shape[0], np.nan, dtype=np.float64)
  # ⚑ THE FLOOR IS LOAD-BEARING, and in the OPPOSITE direction to what an
  # earlier revision of this comment claimed. That revision recorded
  # `den > 0.0` as an EQUIVALENT mutant on the argument that "the mask already
  # drops every entry below 1e-6, so any surviving entry contributes a LARGE
  # log-ratio". The argument is wrong about what `den` accumulates: `b` holds
  # log-RATIOS to the reference, not log-probabilities, so two entries that
  # both clear the mask and are nearly EQUAL contribute a near-zero `b` and
  # hence a tiny `den`.
  #
  # `policy_target` is stored float16 (`replay/buffer.py:199`), which makes
  # near-ties the common case rather than a contrived one: a 1-ulp fp16 tie at
  # p ~= 0.5 gives den = 2.385e-07, under the floor. Without it that row's
  # slope is num/den over two quantities that are both rounding noise, and the
  # estimate comes back T ~= 3.0 for a row written at T = 2.0.
  #
  # Behaviour is UNCHANGED — the floor was always right, only its rationale was
  # wrong — and the practical effect on the reported median is nil (4000-row
  # Monte-Carlo, fp32 and fp16 both median 2.0000), because such rows are a
  # small minority of a median. `test_fp16_near_tie_needs_the_den_floor` is the
  # killing test, so `den > 0.0` is no longer recorded as equivalent.
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def _check_soft_policy_temp(recovered: np.ndarray) -> str:
    """Compare the temperature the SHARDS were written at against the live yaml.

    Returns a report line. Degrades loudly (never crashes) when the live config
    is unreadable, because this probe is otherwise host-independent.

    The failing input: shards written at one `soft_policy_temp` read against a
    config carrying another — which is exactly the state this file's own
    docstring was in before 2026-08-16, claiming "T = 3.0 in production" while
    the live yaml said 2.0.
    """
    from chess_anti_engine.eval.production_shape import (
        CONFIG_ABSENT,
        LIVE_CONFIG_ENV,
        load_live_config_or_reason,
    )

    finite = recovered[np.isfinite(recovered)]
    if finite.size == 0:
        return (
            "[shape] soft_policy_temp: NOT CHECKED — every sampled row is "
            "(near-)one-hot, so no temperature is recoverable from the data."
        )
    med = float(np.median(finite))
    live, reason = load_live_config_or_reason()
    if live is None:
        return (
            f"[shape] soft_policy_temp: shards were written at T~={med:.4f} "
            f"({finite.size} informative rows). NOT compared against production "
          # ⚑ The reason, not "unset or unreadable": unset, missing file and
          # fails-to-flatten are three states with three different operator
          # actions, and the union of them sends the reader to the wrong one.
            f"— {reason}."
        )
  # ⚑ NO `.get(..., 2.0)` FALLBACK. This function exists because a hard-coded
  # 3.0 in the module docstring went stale; a hard-coded 2.0 here is the same
  # bug with a fresher number, and it would report "MATCHES" for a config that
  # does not mention `soft_policy_temp` at all. The `<absent>` sentinel from
  # `compare_config_values` is the existing way to say "the key is not there",
  # and absence is a NOT-CHECKED, never a pass.
    raw = live.flat.get("soft_policy_temp", CONFIG_ABSENT)
    if raw is CONFIG_ABSENT:
        return (
            f"[shape] soft_policy_temp: shards were written at T~={med:.4f} "
            f"({finite.size} informative rows). NOT CHECKED (key absent from "
            f"config {live.path}) — there is nothing to compare against, "
            f"so this is not a pass.\n{live.header()}"
        )
    want = float(raw)
    ok = abs(med - want) <= 0.05 * max(1.0, want)
    verdict = "MATCHES" if ok else "DOES NOT MATCH"
  # ⚑ `authoritative`, NOT `live is None`. `load_live_config_or_reason` returns
  # the IN-TREE fallback with `authoritative=False` when
  # $CHESS_ANTI_ENGINE_LIVE_CONFIG is unset — which is the default in every
  # worktree — so branching on None alone printed "live config says ... ->
  # MATCHES" about a file the resolver had already decided was not live. That
  # is the exact defect `audit_targets.py` fixed in this same change, left
  # standing in a sibling instrument. `value_regret.py` prints `live.header()`
  # (which carries the [NOT-LIVE] mark) and is the in-tree precedent.
    reference = "live config" if live.authoritative else (
        "NON-AUTHORITATIVE reference config"
    )
    line = (
        f"[shape] soft_policy_temp: shards were written at T~={med:.4f} over "
        f"{finite.size} informative rows; {reference} says {want} -> {verdict}."
        f"\n{live.header()}"
    )
    if not live.authoritative:
        line += (
            "\n[shape] ⚑ This is NOT a production check. The comparison above "
            "is against the in-tree config, which is stale by construction "
            f"outside the live working tree. Export ${LIVE_CONFIG_ENV} to name "
            "the live yaml if you meant to check production."
        )
    if not ok:
        line += (
            "\n[shape] ⚑ The stored soft targets do NOT come from the live "
            "temperature. Either the replay window predates a config edit — in "
            "which case the trainer is consuming targets built at the OLD T for "
            "another window's worth of iterations — or soft_policy_temp is not "
            "reaching selfplay at all. Do not read the divergence numbers below "
            "as a property of the live setting."
        )
    return line


def _collect(
    replay_dir: Path, positions: int, chunk_rows: int = 4096,
) -> dict[str, np.ndarray]:
    paths = sorted(iter_shard_paths(replay_dir), key=shard_index, reverse=True)
    if not paths:
        raise SystemExit(f"no shards found under {replay_dir}")

    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "tv", "kl_pq", "kl_qp", "argmax_agree", "source", "phase", "temp_hat",
        )
    }
    seen = 0
    used_shards = 0
    for path in paths:
        if seen >= positions:
            break
        arrs, _meta = load_shard_arrays(path, lazy=True)
        if "policy_soft_target" not in arrs or "has_policy_soft" not in arrs:
            continue
        has_soft = np.asarray(arrs["has_policy_soft"]).astype(bool)
        if not has_soft.any():
            continue
        used_shards += 1
        # .shape on the lazy zarr array is metadata-only; np.asarray here would
        # materialize the full x array just to read its length.
        n = int(arrs["x"].shape[0])
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            sel = has_soft[start:stop]
            if not sel.any():
                continue
            p = np.asarray(arrs["policy_target"][start:stop], dtype=np.float32)[sel]
            q = np.asarray(arrs["policy_soft_target"][start:stop], dtype=np.float32)[sel]
            x = np.asarray(arrs["x"][start:stop], dtype=np.float32)[sel]
            metrics = _row_metrics(p, q)
            valid = metrics["valid"]
            if not valid.any():
                continue

            piece_counts = np.rint(x[:, :12].sum(axis=(1, 2, 3)))
            phase = _phase_bucket(piece_counts)[valid]

            source = np.full(p.shape[0], 2, dtype=np.int64)  # unknown
            if "is_selfplay" in arrs and "has_is_selfplay" in arrs:
                has_src = np.asarray(arrs["has_is_selfplay"][start:stop]).astype(bool)[sel]
                is_sp = np.asarray(arrs["is_selfplay"][start:stop]).astype(bool)[sel]
                source[has_src & is_sp] = 0       # selfplay
                source[has_src & ~is_sp] = 1      # curriculum (SF opponent)
            source = source[valid]

            for key in ("tv", "kl_pq", "kl_qp", "argmax_agree"):
                cols[key].append(metrics[key][valid])
            cols["source"].append(source)
            cols["phase"].append(phase)
          # Recovered from the SAME rows the statistics are computed over, so
          # the temperature check cannot be about a different population than
          # the numbers it is vouching for.
            cols["temp_hat"].append(_recover_soft_policy_temp(p, q)[valid])
            seen += int(valid.sum())
            if seen >= positions:
                break

    if seen == 0:
        raise SystemExit(
            f"no samples with policy_soft_target in the newest shards of {replay_dir}"
        )
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out["_used_shards"] = np.array(used_shards)
    return out


def _summarize(mask: np.ndarray, data: dict[str, np.ndarray]) -> dict | None:
    n = int(mask.sum())
    if n == 0:
        return None
    tv = data["tv"][mask]
    return {
        "n": n,
        "tv_mean": float(tv.mean()),
        "tv_median": float(np.median(tv)),
        "tv_p90": float(np.quantile(tv, 0.90)),
        "kl_mean": float(data["kl_pq"][mask].mean()),
        "rev_kl_mean": float(data["kl_qp"][mask].mean()),
        "frac_identical_tv_lt_0.01": float((tv < _IDENTICAL_TV).mean()),
        "frac_tv_gt_0.1": float((tv > 0.1).mean()),
        "argmax_agreement": float(data["argmax_agree"][mask].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--replay-dir", type=Path, required=True,
                    help="replay shard dir (e.g. runs/pbt2_small/tune/<trial>/replay_shards)")
    ap.add_argument("--positions", type=int, default=200_000,
                    help="most-recent positions to stream (default: 200k)")
    ap.add_argument("--out", type=Path, default=Path("runs/probe_policy_targets.json"))
    args = ap.parse_args()

    data = _collect(args.replay_dir, int(args.positions))
    n_total = int(data["tv"].shape[0])

    groups: dict[str, dict] = {}
    overall = _summarize(np.ones(n_total, dtype=bool), data)
    assert overall is not None
    groups["overall"] = overall
    for si, source in enumerate(_SOURCES):
        m = data["source"] == si
        got = _summarize(m, data)
        if got is not None:
            groups[source] = got
        for pi, phase in enumerate(_PHASES):
            got = _summarize(m & (data["phase"] == pi), data)
            if got is not None:
                groups[f"{source}/{phase}"] = got

    header = (
        f"{'group':28s} {'n':>8s} {'TV mean':>8s} {'TV med':>8s} {'TV p90':>8s} "
        f"{'KL':>8s} {'revKL':>8s} {'TV<.01':>7s} {'TV>.1':>7s} {'argmax=':>8s}"
    )
    print(f"[probe] replay dir: {args.replay_dir}  "
          f"(streamed {n_total} samples from {int(data['_used_shards'])} newest shards)")
  # Printed BEFORE the table, because it decides whether the table is about
  # the live setting or about a superseded one.
    temp_line = _check_soft_policy_temp(data["temp_hat"])
    print(temp_line)
    print(header)
    print("-" * len(header))
    for name, g in groups.items():
        print(
            f"{name:28s} {g['n']:>8d} {g['tv_mean']:>8.4f} {g['tv_median']:>8.4f} "
            f"{g['tv_p90']:>8.4f} {g['kl_mean']:>8.4f} {g['rev_kl_mean']:>8.4f} "
            f"{g['frac_identical_tv_lt_0.01']:>7.1%} {g['frac_tv_gt_0.1']:>7.1%} "
            f"{g['argmax_agreement']:>8.1%}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "replay_dir": str(args.replay_dir),
        "positions": n_total,
        "shards_used": int(data["_used_shards"]),
        "identical_tv_threshold": _IDENTICAL_TV,
      # Banked, not just printed: a reader of this JSON months from now must be
      # able to tell which temperature the numbers describe without rerunning.
        "soft_policy_temp_check": temp_line,
        "soft_policy_temp_recovered_median": (
            float(np.median(data["temp_hat"][np.isfinite(data["temp_hat"])]))
            if np.isfinite(data["temp_hat"]).any() else None
        ),
        "groups": groups,
        "argv": sys.argv,
    }
    args.out.write_text(json.dumps(record, indent=2))
    print(f"\n[probe] JSON written to {args.out}")


if __name__ == "__main__":
    main()
