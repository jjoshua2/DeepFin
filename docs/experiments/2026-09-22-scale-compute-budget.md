# 500M generation and GPU capacity requirements

Date: 2026-09-22. Status: illustrative resource model, not a measured schedule.

The proposed equal-third SF/BT4/Ceres mix must fit generation, labeling and training
into the same machine budget. A generation-only month estimate is insufficient.
The mix is defined by accepted positions, not game counts. Production acceptance must
include required history checks and cross-source deduplication; the short SF screen
only checked identities within each arm. These calculations assume
500M newly generated positions and one training epoch; existing usable data would
reduce the work. Neither this model nor throughput establishes target quality.

## Assumptions and reproducibility

[Inputs](evidence/scale-compute-budget-20260922/inputs.json),
[calculator](evidence/scale-compute-budget-20260922/calculate.py), and
[output](evidence/scale-compute-budget-20260922/requirements.json) are banked together.
Run `python3 docs/experiments/evidence/scale-compute-budget-20260922/calculate.py`.
The 8.76 BT4-label GPU-days, 4.23 Ceres-label GPU-days, and 4.7 training GPU-days
are **provisional planning inputs from prior discussion**, not throughput newly
verified by this calculation. They must be replaced by representative sustained
measurements before a production commitment. Generation rates below are requirements,
not predictions. GPU-days mean occupied elapsed GPU time, not peak-utilization days.

## CPU requirement

Generating 166.7M accepted SF positions in 30 days needs 64.3 positions/s with
continuous access, or 128.6 positions/s at 50% availability. The completed
[175-second screen](https://github.com/jjoshua2/DeepFin/pull/837) measured 65.07/s
at d6 with four workers, including startup and excluding result-less games.
It offers little margin and does not establish sustained capacity. The separate
600-second four/eight-worker confirmation is running; its result is not assumed here.

## Shared GPU requirement

All 333.3M neural-generated positions also consume GPU time. Each position still
needs both teachers' labels under the proposed blend. If a generating teacher's
outputs are retained and exactly meet the eventual label contract, only the other
teacher needs to run again on that position. SF-generated positions need both.
Thus one-third of each teacher's offline labeling work can potentially be avoided.
This is conditional on policy/value, history, calibration, precision and provenance
compatibility; it is not implemented or qualified yet. It is not free generation:
the generating teacher's inference remains charged to generation.

| Generating-teacher outputs reused | GPU availability | Remaining label days | Generation days left after labels + one epoch | Required aggregate neural generation positions/s |
|---|---:|---:|---:|---:|
| No | 100% | 12.99 | 12.31 | 313.4 |
| No | 80% | 12.99 | 6.31 | 611.4 |
| Yes, contract qualified | 100% | 8.66 | 16.64 | 231.9 |
| Yes, contract qualified | 80% | 8.66 | 10.64 | 362.6 |

The aggregate rate is total accepted BT4+Ceres positions divided by their combined
GPU generation time. It is not the arithmetic mean of two teacher rates. For equal
accepted counts it is their harmonic mean, and includes game scheduling, rejected
positions, history preparation and output writes. Isolated sidecar inference
throughput cannot substitute for it.

These are necessary resource conditions, not a feasible dependency schedule: CPU and
GPU can overlap, but training needs ready labels, and final-stage tails can extend
the calendar. Evaluation, retries, storage conversion, idle gaps and other experiments
need additional headroom; 80% availability illustrates a reduced compute budget but
does not itself schedule those tasks. External-disk contention must be measured with
representative concurrent workloads. A full second epoch adds another assumed 4.7 GPU-days.

## Optimization priorities from the code audit

1. Qualify generation capacity and worker scaling using completed, accepted positions.
2. Make the previously qualified retained-tablebase engine option reachable from the
   generator, explicitly opt-in and rejected by engines that do not advertise it.
   Keeping mappings can avoid repeated setup while retaining per-game TT clears;
   the generator-level benefit is not yet measured.
3. Build a batched neural game generator that preserves reusable teacher outputs.
   Existing evaluation and sidecar components are ingredients, not a ready generator.
4. Test reduced SF search width as a separate data-generation intervention. Current
   `StaircaseSearcher.search_position` scores all legal moves and `play_game` samples
   their value vector via Gumbel noise. Root-only or top-k search changes exploration;
   it cannot be presented as an identical-output optimization.
5. Qualify the existing trainer host-batch overlap switch in a future controlled GPU
   slot. The small gather prototype saved only about 1–2 ms per batch; see
   [the negative-priority readout](https://github.com/jjoshua2/DeepFin/pull/835).

Source inspection also found cached MultiPV settings, persistent engines, cached
opening books and cached Python tablebase handles already in place. Removing the
per-game `ucinewgame` would alter an explicit search invariant. A duplicated terminal
WDL probe can be removed, but it saves one probe per adjudicated game and has no
measured material effect on total generation throughput.

No production process, queue entry, target mixture or runtime configuration is
changed by this record. The running E arm and its paired evaluation remain the
quality evidence for removing SF from the main value blend on existing data.

Validation: a separate reviewer reproduced the calculator output exactly, checked
the resource accounting and source claims, and approved this record. Local JSON,
relative-link and whitespace checks passed.
