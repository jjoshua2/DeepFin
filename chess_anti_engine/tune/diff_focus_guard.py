"""Realized-regime guard for the difficulty-focus curriculum.

WHY THIS EXISTS. On 2026-08-09 the search-authority bundle (``gumbel_c_scale``
0.025 -> 0.1 plus ``gumbel_policy_temp: 1.5``) moved KL(prior || search target)
by ~7.9x on fresh games. ``difficulty = |q_delta| * q_weight + kl * pol_scale``
is unnormalized and feeds a FIXED clamp
(``keep_prob = clamp(difficulty * slope, min_keep, 1.0)``), so the clamp
saturated: ``diff_focus_keep_rate`` 0.80 -> 0.93 within five iterations and the
curriculum that chooses WHICH PLIES ARE RECORDED went inert. Every config key
was accepted and every value stayed in range, so nothing warned. Each of the
numbers below was already computed and logged every iteration by
``tune/trainable_report.py`` -- and nothing read them, so the change ran ~140
iterations in plain sight.

WHAT THIS GUARDS. The REALIZED regime, not the config. A config check could not
have caught this: no diff-focus key changed. The bands are calibrated on the
measured pre-bundle steady state (iters 560-735, n=176 iterations), and the
per-iteration spread there is tiny -- ``diff_focus_keep_rate`` sd 0.0063 on a
mean of 0.8029 -- so a single-iteration excursion outside these bands is many
sigma and needs no consecutive-iteration debouncing.

WHAT IS DELIBERATELY *NOT* GUARDED. ``replay_pmass_kl_share`` looks like the
sharpest instrument here (0.37 -> 0.86 across the incident) and it is NOT in the
band table, because it cannot survive the very fix it would be guarding.
``DiskReplayBuffer._append_shuffle_arrays`` samples on the STORED ``priority``
column -- computed in the worker at selfplay time with whatever
``diff_focus_pol_scale`` was live THEN -- while
``_accumulate_priority_mass`` decomposes that mass by applying TODAY's
``diff_focus_pol_scale`` to the stored ``priority_policy_kl`` column. The two
agree only while ``pol_scale`` is constant. Recalibrate ``pol_scale`` and the
share misreports every row already in the ~1.5M-position window until it turns
over (~133 iterations at ~11.3k ingested/iteration). Use
``replay_pmass_kl_raw_mean`` instead: it is the mean stored KL itself, carries no
config factor, and is therefore comparable across a recalibration.
"""

from __future__ import annotations

import logging
import math
from typing import Final

log = logging.getLogger(__name__)

ALARM_KEY: Final = "diff_focus_regime_alarm"
ALARM_DETAIL_KEY: Final = "diff_focus_regime_alarm_metrics"

# Iterations at the start of a trial during which the bands are not evaluated.
# The seeded/restored window is drawn into the shuffle buffer over the first few
# iterations, so append-side means are not yet the steady state. Kept small on
# purpose: the 2026-08-09 incident crossed the ``diff_focus_keep_rate`` band on
# its FOURTH post-restart iteration, so a longer warmup would have hidden the
# thing this guard is for.
WARMUP_ITERATIONS: Final = 3

# Below this many worker-side records an iteration is not a usable read (a
# short iteration, a drained pool, a paused fleet). Steady state is ~40k.
MIN_DIFF_FOCUS_RECORDS: Final = 2_000

# Below this many optimizer steps the gradient-side rate is not a usable read.
MIN_GRAD_NORM_SAMPLES: Final = 20

# (low, high) inclusive band, and the report key holding its denominator.
# ``None`` on a side means the guard is one-sided there.
#
# Calibration, measured over iters 560-735 (the pre-bundle steady state) and
# checked against iters 736-745 (the first ten iterations after the bundle):
#
#   metric                        pre-bundle mean +/- sd   736-745      band
#   diff_focus_keep_rate          0.8029 +/- 0.0063        -> 0.936     0.55-0.88
#   diff_focus_keep_limited_frac  0.3737 +/- 0.0085        -> 0.146     0.22-0.60
#   diff_focus_priority_mean      0.8937 (600-735 window)  -> 3.06      0.40-1.90
#   replay_priority_mean          1.0979 +/- 0.0195        -> 3.94      0.45-2.20
#   grad_hard_clip_rate           0.0000 +/- 0.0000        -> 0.0       0.00-0.30
#
# The first four all cross within four iterations of the bundle restart. The
# clip rate does NOT -- it stayed at exactly 0.0000 until iter 782, 46
# iterations later, when ``grad_norm_mean`` finally ramped past the fixed
# ``zclip_max_norm``. It is in the table as the slow confirming arm, not as a
# detector, and NOT as the damage mechanism: direction-only descent was already
# refuted as a generalization factor (task #112), so a high clip rate is the
# airbag reporting that it fired.
REGIME_BANDS: Final[dict[str, tuple[float | None, float | None, str, int]]] = {
    "diff_focus_keep_rate": (0.55, 0.88, "diff_focus_records", MIN_DIFF_FOCUS_RECORDS),
    "diff_focus_keep_limited_frac": (
        0.22, 0.60, "diff_focus_records", MIN_DIFF_FOCUS_RECORDS,
    ),
    "diff_focus_priority_mean": (0.40, 1.90, "diff_focus_records", MIN_DIFF_FOCUS_RECORDS),
    "replay_priority_mean": (0.45, 2.20, "replay_priority_n", 1),
    "grad_hard_clip_rate": (None, 0.30, "grad_norm_samples", MIN_GRAD_NORM_SAMPLES),
}


def evaluate_diff_focus_regime(
    report: dict,
    *,
    enabled: bool,
    trial_iterations_completed: int,
    report_iteration: int,
) -> dict[str, object]:
    """Check the realized diff-focus regime and return report columns to merge.

    ``trial_iterations_completed`` is the caller's own loop counter (0 on the
    first iteration THIS process runs), which is what ``WARMUP_ITERATIONS``
    means. It is deliberately not the reported ``training_iteration``: a salvage
    warm start restores the donor's global counter, so a warmup keyed to that
    would be permanently satisfied on the one restart it exists to cover.
    ``report_iteration`` is used only to label the log line.

    ``report`` is the assembled per-iteration report dict, read AFTER the
    priority-mass stats have been merged into it. Reading the same dict that is
    about to be reported is the point: the guard and the operator-facing number
    are then the same float by construction, rather than two derivations that
    can drift (a guard must share the criterion's instrument).

    Returns ``{ALARM_KEY: 0|1, ALARM_DETAIL_KEY: str}``. The detail column is
    pipe-separated, never comma-separated: ``progress.csv`` rows are parsed with
    naive ``awk -F','`` by ``scripts/monitor_pbt.sh``.

    This never raises and never stops the trial. A telemetry excursion is not a
    reason to take production down -- the failure it detects cost ~76 Elo over
    140 iterations, an unplanned outage costs more than the four hours it takes
    an operator to read a loud row.
    """
    if not enabled:
        # The curriculum being off is a legitimate configuration; keep_rate is
        # then pinned at 1.0 by construction and the bands are meaningless.
        return {ALARM_KEY: 0, ALARM_DETAIL_KEY: "disabled"}
    if int(trial_iterations_completed) < WARMUP_ITERATIONS:
        return {ALARM_KEY: 0, ALARM_DETAIL_KEY: "warmup"}

    breaches: list[str] = []
    for key, (low, high, denom_key, denom_min) in REGIME_BANDS.items():
        if key not in report:
            continue
        try:
            denom = float(report.get(denom_key, 0.0))
            value = float(report[key])
        except (TypeError, ValueError):
            continue
        # A metric whose denominator was empty reports 0.0, which is inside no
        # band worth alarming on and outside several. Skipping it is what keeps
        # this guard from crying wolf on every paused or short iteration -- and
        # is why the denominator, not just the value, is part of the table.
        if denom < denom_min or not math.isfinite(value):
            continue
        if low is not None and value < low:
            breaches.append(f"{key}={value:.4f}<{low:g}")
        elif high is not None and value > high:
            breaches.append(f"{key}={value:.4f}>{high:g}")

    if not breaches:
        return {ALARM_KEY: 0, ALARM_DETAIL_KEY: ""}

    detail = "|".join(breaches)
    log.error(
        "DIFF-FOCUS REGIME ALARM (iter %d): the realized curriculum has left the "
        "band it was calibrated in: %s. The 2026-08-09 precedent is a search "
        "change moving the scale of an unnormalized `difficulty` under a fixed "
        "clamp, which silently disabled ply selection for ~140 iterations. "
        "Check `diff_focus_pol_scale`/`diff_focus_slope` against the current "
        "KL scale (`replay_pmass_kl_raw_mean`) before assuming this is noise.",
        int(report_iteration), detail,
    )
    return {ALARM_KEY: 1, ALARM_DETAIL_KEY: detail}
