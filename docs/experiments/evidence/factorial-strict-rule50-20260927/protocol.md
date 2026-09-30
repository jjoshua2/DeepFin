# Strict factorial58 four-edge readout, preregistered before D–B/C–A/B–A results

This is a CPU-only readout of four **new strict six-man, rule50-aware** checkpoint arenas. It does not add games or change training. D–C has already completed; D–B, C–A, and B–A were not read by the author when this analysis was defined. The historical tablebase-disabled arenas are excluded from every estimate.

The fixed inputs are the four plan SHA-256 values embedded in `readout.py`: D–C `b8d61333…`, D–B `66a6809c…`, C–A `fdf6ad03…`, B–A `a563a609…`. Each edge must have a strict PASS receipt bound to its plan, 256 independently verified games, 128 paired openings, 400 simulations per side, the registered seed and opening book, and hashes for game log, result and PGN. The readout recomputes per-game candidate scores from result and candidate color, then independently reconstructs each opening-pair score and the terminal receipt's paired statistic. An incomplete, changed, truncated or mismatched edge stops the four-edge readout. The existing arena verifier remains the proof of strict game-level rule50 replay; this script does not replay moves or probe Syzygy.

For edge X–Y, with candidate X and reference Y, opening-pair score `q_i` is the average candidate score in the two colors, in `{0, .25, .5, .75, 1}`. Its reported score is mean `q_i`, its paired standard error is the sample standard deviation of 128 `q_i` values divided by `sqrt(128)`, and its 95% score interval is mean ±1.96 SE. Elo and its interval are `400 log10(s/(1-s))` applied to the mean and score endpoints, matching the arena receipt. These are unadjusted descriptive 95% intervals for each fixed edge, conditional on these checkpoints and the opening design. They do not establish a strength claim for a broader training seed population.

The four fixed interpretations are:

| Edge | Conditional contrast |
| --- | --- |
| B–A | Mixed versus BT4-only policy at SF/BT4 value |
| D–C | Mixed versus BT4-only policy at equal-thirds value |
| C–A | Equal-thirds versus SF/BT4 value at BT4-only policy |
| D–B | Equal-thirds versus SF/BT4 value at mixed policy |

For a cross-edge summary, all four banks must have the same opening book SHA and exact opening FEN at every pair ID. If this alignment fails, report the individually verified edges and mark the joint summary unavailable. If it holds, define per-opening edge advantage `d_XY,i = q_XY,i - .5`. The **policy average** is `100 × mean_i[(d_BA,i+d_DC,i)/2]` percentage points; the **value average** uses C–A and D–B similarly. Report two separate heterogeneity contrasts: `100 × mean_i(d_DC,i-d_BA,i)` and `100 × mean_i(d_DB,i-d_CA,i)`. There is no unique additive factorial interaction on head-to-head score or Elo scales; the two heterogeneity estimates must not be forced to agree or treated as a single interaction coefficient.

Uncertainty for the four cross-edge contrasts uses 20,000 nonparametric percentile-bootstrap replicates, NumPy PCG64 seed `2026092401`, `np.quantile(..., [0.025,0.975], method='linear')`. Each replicate samples 128 pair IDs with replacement; **the same sampled IDs are applied jointly to every edge**, retaining covariance from shared openings. The point estimate uses all original pairs. This is an opening-pair resampling interval, not a multiple-comparison-adjusted familywise bound or a training-seed interval.

The original factorial training receipts' `valid_control=false` limitation remains. Neither a positive edge nor a positive average automatically selects a new policy, promotes seed122, proves an optimizer change, or measures arena throughput; CPU overlap can alter wall time. No post hoc threshold or extra games are part of this readout.

To validate the completed D–C edge only without writing a four-edge result:

```bash
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 readout.py --check-edge D_C
```

Once all four strict receipts are complete, run `readout.py --out <fresh-absolute-json>` on a CPU-only host. Save the script/protocol/test SHA pins before executing the four-edge readout; preserve the resulting JSON and its SHA. Do not overwrite a previous readout.
