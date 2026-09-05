# PRE-REGISTERED PREDICTIONS — written BEFORE tgt_probe.py was run (2026-08-15)

Instrument: `scratchpad/target_audit/tgt_probe.py`, 8 newest shards per era
(2000 rows each = 16,000 rows/era), 5 eras.

## Exact-count predictions
- rows read per era = 8 x 2000 = **16000** exactly.
- `frac_has_policy` = 0.55–0.75 (full-sim plies only; diff_focus drops some).
- `frac_has_sf_p0_regret` (of has_policy rows) = 0.15–0.45
  (selfplay slots only, needs two consecutive full plies; selfplay_fraction 0.8).
- `n_legal` mean ~ 30–35.
- `target_row_sum` ~ 1.0 (f16 storage, so 0.999–1.001).
- `target_mass_on_ILLEGAL` ~ 0 (< 1e-4).
- `support_nonzero` ≈ `n_legal` (the Gumbel improved policy is
  softmax(log prior + sigma(completed Q)) over ALL legal moves — it is DENSE by
  construction; only f16 underflow (< 6e-8) can zero an entry).

## Substantive predictions (current era E)
- `entropy_nats` mean **1.0–1.6**
- `eff_support_expH` mean **3–5**
- `top1_mass` mean **0.45–0.60**
- `top8_mass` mean **> 0.95**  (⚑ saturated — will be reported but not used as a verdict)
- `KL_prior_to_target_nats` mean **0.05–0.20**
- `KL_over_H` mean **0.05–0.15**
- `AGREE_target_argmax_eq_sf_best` **0.45–0.55**, against a chance baseline
  `CHANCE_mean_1_over_nlegal` ≈ **0.033**
- `mass_on_sf_best` mean **0.35–0.55**
- `mass_on_sf_multipv_covered_set` mean **> 0.9** (6 of ~32 moves, but the top-6
  are where the search's mass is)

## Negative controls — the reading that means "instrument works"
- C1 `CONTROL_shuffled_agreement` must fall to ≈ chance, **0.02–0.05**.
  If it does not fall, the agreement number is an artifact.
- C2 `CONTROL_mass_under_shuffled_legal_mask` must fall from ~1.0 to **< 0.10**.
  This proves the target and the mask are in the SAME index space and the same row.

## Drift
- Era E (sims 100 / topk 16 / c_scale 0.1) is **SHARPER** than era A
  (sims 256 / topk 32 / c_scale 0.025): lower entropy, higher top-1.
  ⚑ CONFOUNDED by search config — this is a search-config comparison, not a
  learning comparison.

---
# PRE-REGISTERED PREDICTIONS for tgt_probe2.py (written before running it)

- `KL_frac_NEGATIVE` > 0 (the C sum drops terms with mp <= 1e-12, so the stored
  value is a TRUNCATED KL and is not guaranteed non-negative). Predict 5–20%.
- Calibration: agreement with the shallow-SF best move RISES monotonically with
  the target's top-1 mass, but stays far below it. In the [0.95, 1.01) bin
  predict mean top-1 mass ~0.97 and agreement **0.55–0.70**, i.e. an
  overconfidence gap of ~0.30.
- `cp_regret_of_target_argmax_LISTED` mean **20–60 cp**; `frac_eq_0` ≈ the
  overall agreement rate (~0.42).
- `CONTROL_shuffled_agree` ≈ 0.002 in every confidence bin (flat) — proving the
  calibration slope is not an artifact of the binning.
- `xcheck_sf_multipv_raw.match_rate` > 0.90 on the rows where row i-1 really is
  the record whose P1 MultiPV built row i's `sf_p0_regret`.
