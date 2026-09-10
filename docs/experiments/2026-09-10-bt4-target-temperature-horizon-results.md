# BT4 target temperature and horizon: completed comparison

Both two-epoch trajectories and all three registered matches completed. The
relative horizon effect and the final sharpening-placement comparison remain
unresolved. The matches used the frozen September 9 preregistration snapshot identified in
the compact readout. The linked [registration record](2026-09-09-bt4-target-temperature-horizon.md)
includes subsequent status updates and is unchanged by this report.
The [compact readout](evidence/bt4-bootstrap/target-temperature-horizon-completed.json)
binds the accepted results, checkpoints, source receipts and independent reviews.

Each recipe used training seed 0, unchanged SF value supervision, and two
uninterrupted passes over the same 18,910,484 distinct rows. Each completed
73,870 updates and 37,820,968 row presentations, with its own epoch-one and final
checkpoint. The difference was pure BT4 policy target temperature: T1 versus
T0.5. These are fresh matched trajectories, not continuations of historical B100.

Each match completed 256 games from the same 128 color-swapped opening pairs,
with 16-ply histories and 400 simulations per move. There were no match extensions.
Scores below are from the T1 candidate's perspective; Elo is descriptive and its
95% interval uses opening pairs as the uncertainty unit.

| Comparison | Search priors, T1 / T0.5 | Candidate score | Elo | Nominal paired 95% Elo interval |
| --- | --- | ---: | ---: | ---: |
| Epoch 1 | 1 / 1 | 45.703% | −29.93 | [−69.30, +8.68] |
| Epoch 2 | 1 / 1 | 51.367% | +9.50 | [−29.51, +48.76] |
| Final sharpening placement | 0.5 / 1 | 45.703% | −29.93 | [−64.82, +4.36] |

The primary preregistered contrast is the change in T1's relative score from
epoch one to epoch two, aligned by opening pair. It is **+5.664 percentage points**,
with a 95% percentile bootstrap interval of **[−2.734, +13.867] points**
(10,000 resamples, seed 20260909). The interval crosses zero, so the registered
direction is unresolved. The point estimate is compatible with relative T1
recovery; it does not establish that recovery or measure either arm's absolute
improvement. This is the paired contrast, not a subtraction of Elo intervals.

The final placement comparison also remains uncertain. Ideal fitting predicts
the same effective exponent for target T1/search prior 0.5 and target T0.5/search
prior 1; imperfect fitting, support loss and shared-trunk effects can break that
relationship. The result establishes neither equivalence nor an optimal
temperature. All intervals are nominal development intervals, without correction
for related comparisons or uncertainty across training seeds.

An implementation seed mismatch was repaired before any of these matches. The
original CPU preparation used seed 42 against a panel generated with 20260909
and stopped at panel equality. A separately reviewed amendment propagated
20260909 through sampling, settings, commands and readers while retaining the
original registered panel and all trained checkpoints. The failed attempt remains
part of the cost: 107.09 seconds of owned CPU preparation; its full outer elapsed
time was not captured. Corrected CPU preparation took 120.40 seconds overall,
including 115.09 seconds in its owned stage.

Recorded training charges were 25,180.22 seconds for T1 and 24,863.87 seconds for
T0.5 (about 6.99 and 6.91 hours). The three owned GPU match stages took
1,394.49, 1,391.50 and 1,557.57 seconds; their outer elapsed times, including lease
wait and surrounding work, were 1,416.64, 1,471.03 and 1,770.52 seconds. These are
stage costs, not a complete materialization-to-readout project total. All three
matches logged Dynamo recompile-limit warnings at shrinking tail batches.
Fully compiled coverage and eager execution fraction were not measured. The
complete fixed-simulation results support the comparisons above, but these costs
do not establish a controlled speedup or equal-wall-clock deployment ranking.

Both training receipts retain historical-control limitations: no held-out purity
receipt, configuration checked against a committed pin rather than live state,
and without-replacement game-epoch sampling that differs from older replacement
controls. One seed and reused development openings limit generalization. Repeating
the original corpus twice also does not answer the distinct-data scaling question.

The current decision is to retain the incumbent and avoid extending these matches
or starting a fine temperature grid. Selective SF tactical policy correction and
a bounded Ceres dual-value capability pilot are the next substantive directions;
scale transfer and fresh-seed confirmation remain separate work.

Bulk checkpoints and game banks remain outside Git. The compact readout lists
exact retrieval identities under the [evidence convention](README.md#new-experiments-and-follow-ups);
no public bulk download URL is registered in this snapshot. It contains curated
scientific metadata, not host paths or raw operational logs.
