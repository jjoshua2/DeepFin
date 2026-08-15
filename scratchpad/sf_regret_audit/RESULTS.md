# sf_p0_regret fabrication audit — 2026-08-15

CPU-only. No GPU used. Training was stopped throughout; nothing was started.

## Q1. Is the fabrication still live? YES — but only in the SHARDS.

MEASURED:
- `configs/pbt2_small.yaml:160` → `sf_multipv: 6`. Flattened via
  `flatten_run_config_defaults` → `sf_multipv = 6`.
- `configs/pbt2_small.yaml:213` → `record_sf_p0_regret: true`; survives
  `TrialConfig.from_dict` → `tc.record_sf_p0_regret is True`.
- The fabrication site is `chess_anti_engine/selfplay/finalize.py:1281`
  `_build_sf_p0_regret_vector` (C mirror `_prepare_sf_multipv_c`,
  `encoding/_lc0_ext.c`): every index not in the MultiPV block is pre-filled
  with `default = (worst_covered_regret + 1) / 2`, then the ≤6 covered indices
  are overwritten. `SF_OWN_REGRET_CAP_CP = 1000.0` (finalize.py:71).

Measured on 40 recent shards of the LIVE replay buffer
(`runs/pbt2_small/replay/train_trial_5ce02_.../replay_shards`), 16,549 eligible
rows:

| quantity | value |
|---|---|
| eligible-row fraction (`has_sf_p0_regret`) | 0.2188 (predicted ~0.22 from progress.csv; MATCH) |
| legal moves per row, mean | 27.1 |
| covered moves K, on unambiguous rows | **exactly 6 on 14045/14045 rows** |
| K > 6 | 0 rows |
| fabricated fraction of legal moves | **0.7623** |
| fabricated default value d, mean | **0.5672 → 567 cp** |
| worst *real* covered regret, mean | 0.1344 → 134 cp |
| inflation d / worst-real | **4.22×** |
| mean *real* covered regret | 0.0745 → 75 cp |

The banked "~570cp for ~68% of legal moves, ~5× true" reproduces: 567cp,
76.2%, 4.22× vs worst-covered (7.6× vs mean-covered). The 68→76% difference is
row population — 10.8% of eligible rows have L ≤ 6 and carry NO fabrication at
all, and 4.4% are cap-saturated (d = 1.0) where covered and fabricated entries
are indistinguishable.

### Prediction discipline
Registered before running (`sfreg_measure.py` docstring): P1 eligible fraction
~0.22 — HIT (0.2188). P2 "K == min(6,L) on ≥95%" — MISSED at 84.9%. P3
"worst_covered == 2d−1" — MISSED at 92.4%. Both misses were an ESTIMATOR
degeneracy, not data: on rows where SF covered ALL legal moves (L ≤ 6) there is
no default entry, so `max(v)` is a real covered regret and K is unrecoverable.
Confirmed 1429/1429 (`sfreg_diag.py`, `sfreg_measure2.py` H1). Restricted to
unambiguous rows, K == 6 on 100.000% and `worst_covered == 2d−1` on 100.000%.

## Q2. Does it reach the loss? NO. It is INERT on the gradient.

Full chain (all file:line on the live tree):

```
shard field  sf_p0_regret                     replay/shard.py:176
  -> sample  ReplaySample.sf_p0_regret        replay/sample.py:83
  -> batch   sf_p0_regret_t                   replay/dataset.py:78, :262
  -> loss    sf_own_regret = (softmax(masked_base) * reg_vec).sum(-1)
                                              train/losses.py:885-890
  -> mean    m_sf_own_regret = masked_mean(...)  train/losses.py:1141
  -> total   + float(w_sf_own_regret) * m_sf_own_regret
                                              train/losses.py:1201
```

The weight:
- `configs/pbt2_small.yaml:476` → **`w_sf_own_regret: 0.0`**
- read at `train/trainer.py:1696` (`_f("w_sf_own_regret", 0.0)`), assigned
  `trainer.py:2215`, passed to `compute_loss` via `_loss_kwargs`
  (`trainer.py:2634`)
- and **re-pushed from the live yaml every iteration**: it is in
  `config_keys.TRAINER_WEIGHT_KEYS:11`, applied by
  `tune/trainable_config_ops.py:1330-1332` (`setattr(trainer, wk, float(...))`),
  called at `tune/trainable.py:1152` inside the iteration loop. So this is not a
  launch-time-only value that a resume could revert.

History: `w_sf_own_regret` was **0.7** from 2026-07-27 (`3b4ca4737`) and was set
to **0.0** on 2026-08-06 by `ed9de8ee9` — the same commit that set
`sf_multipv: 6`. The fabrication got worse and the consumer got switched off in
the same change, which is why nobody has been hit by it.

**Proved by EXECUTION, not grep** (`sfreg_reach.py`): real `compute_loss`, real
shard rows, perturb every fabricated entry to 0.0 and diff `total` and
`d(total)/d(logits)`.

```
perturbation changed 1176 entries of 118912
w_sf_own_regret=0.0:  |dtotal|=0            max|dgrad|=0            (BITWISE ZERO)
w_sf_own_regret=0.7:  |dtotal|=0.294384     max|dgrad|=0.00165634   ||dgrad||/||grad||=0.0643
```

Arm B is the mutation control: the harness CAN see the effect, so arm A's zero
is a real null and not a vacuous test. **At the live weight the fabricated label
contributes exactly zero gradient.**

## Q3. Gradient share — 0% actual; ~6.4% counterfactual.

Actual: 0.0%. Counterfactual at the pre-2026-08-06 weight 0.7, the fabricated
entries alone move the policy-logit gradient by 6.4% of its norm
(`||dgrad||/||grad||` above) — i.e. for the ~11 days that `w_sf_own_regret: 0.7`
was live at `sf_multipv: 6`, roughly 6% of the policy gradient was being driven
by invented numbers. That window is 2026-07-27 → 2026-08-06 and it is a *past*
contamination of the run's history, not a present one.

Share of the *term itself* attributable to fabricated entries, on live shards
(INFERRED, policy proxies): 0.552 under `policy_target` (MCTS visits), 0.744
under `policy_soft_target`.

## Q4. Is there a mask? YES for illegal moves; NO for uncovered-legal.

- Illegal indices ARE masked: `apply_policy_mask_to_logits(..., "legal_mask")`
  (`losses.py:812`, `era_probe.py:587`) drives their probability to exactly 0,
  so the fabricated pre-fill on illegal indices cannot contribute. VERIFIED
  firing: the `legal_mask` is present on 100% of eligible rows measured.
- Row-level mask: `has_sf_p0_regret` gates to the ~22% of rows that carry the
  field. VERIFIED firing (0.2188 measured, matches the live
  `has_sf_p0_regret_frac` column exactly).
- **There is NO mask distinguishing covered from uncovered LEGAL moves.** 76.2%
  of legal moves carry a fabricated value that the loss/probe treats as
  indistinguishable from SF's real evaluations. `train/target_builder.py:733`
  documents this as deliberate ("`sf_p0_regret` is deliberately NOT masked"),
  reasoning only about rebuild-safety, not about coverage.
- Stale comment: `finalize.py:69-70` still says "Legal moves absent from the
  MultiPV default to 1.0." The code has used the `(worst+1)/2` midpoint since
  the docstring at :1281 was written. Worth fixing.

## THE ACTUAL LIVE EXPOSURE: the era-forgetting probe

`w_sf_own_regret` is 0.0, but `sf_p0_regret_t` has a SECOND live consumer that
is not gated by that weight:

`chess_anti_engine/eval/era_probe.py:592` — `policy_eregret`, computed EVERY
iteration (`era_probe_interval: 1`, `configs/pbt2_small.yaml:827`) on both
frozen probe sets. Its own docstring (era_probe.py:26-29) justifies the ruler as
"the same quantity `train/losses.py` minimises as `sf_own_regret` under
`w_sf_own_regret`" — that justification has been false since 2026-08-06, because
that weight is 0.0. **The probe reads an axis training does not push on.**

This is the live forgetting-hinge ruler: `probe_era_policy_eregret`,
`probe_inwindow_policy_eregret`, `probe_gap_policy_eregret`, 218 iterations of
data on the current trial.

MEASURED decomposition (`sfreg_probe_share.py`, CPU forward of
`checkpoint_000218` over the full 2048-row frozen sets). The reproduction is
**bit-exact** against live progress.csv iter 218 — 0.069640 / 0.067193 /
0.002447 — so this splits precisely the number the run reports:

| | era | inwindow | gap (era − inwindow) |
|---|---|---|---|
| `policy_eregret` (live, reproduced) | 0.069640 | 0.067193 | **+0.002447** |
| fabricated component | 0.028745 (**41.3%**) | 0.029570 (**44.0%**) | −0.000825 |
| real-SF component | 0.040895 (58.7%) | 0.037623 (56.0%) | +0.003272 |
| net prob mass on uncovered moves | 0.0653 | 0.0732 | |

Three consequences:

1. **41–44% of the ruler's LEVEL is invented.** The config already warns "Only
   `probe_gap_*`'s TREND is interpretable, never its LEVEL"
   (pbt2_small.yaml:814), which is partial protection.
2. **The fabricated term is not common-mode across the two sets.** It differs by
   −0.000825, which is 25% of the real-SF gap of +0.003272 — it *suppresses* the
   measured forgetting gap by a quarter. The two sets differ in fabricated
   content (era 41.3% vs inwindow 44.0%), so the gap is NOT a clean difference of
   two identically-biased instruments.
3. **The ruler can improve with zero improvement in move quality.** Effective
   fabricated value per unit of uncovered mass is 0.028745/0.0653 = **0.440**.
   Moving 0.01 of probability off the uncovered tail onto *any* covered move —
   including SF's 6th-best — improves `policy_eregret` by up to 0.0044, which is
   **1.8× the entire measured era/inwindow gap**. Since the net is known to be
   sharpening ("we are sharp and wrong", 3.7× narrower than BT4), the probe has a
   standing tailwind that is pure label geometry.

## Adjacent risk checked and CLEARED

`w_sf_move: 0.05` was turned on 2026-08-15 (bt4heads bundle) and trains
`policy_sf` on MultiPV-6-derived targets. That path does NOT carry the same
defect: uncovered legal moves get `sf_policy_label_smooth: 0.01` — a bounded 1%
uniform spread, applied only when `has_uncovered`
(`selfplay/stockfish_turn.py:961-963`). Explicit, small, and parameterised, not
an invented 567cp assertion. Not a concern at this magnitude.

Storage: `sf_p0_regret` is 11KB of a 734KB shard (~1.5%) — zarr compresses the
76%-constant vector almost away. There is no cost argument for switching the
recording off, and switching it off would blind the era probe.

## Other findings

- `docs/model_heads.md`'s head/target/loss table has **no row for the `sf_p0`
  teacher terms** — neither `w_sf_own` (sf_p0_ce) nor `w_sf_own_regret`
  (sf_own_regret) appears, although both sit in `total` at `losses.py:1200-1201`.
  Two loss terms are undocumented in the file that is supposed to be the
  authority on them.
- The era probe metrics are **reporting-only**: `probe_era_*` /
  `probe_inwindow_*` / `probe_gap_*` are written to progress.csv and gate
  nothing (verified by grep across the package — the only non-test consumers are
  `era_probe.py` itself and `trainable_report.py`). They are a human-read ruler,
  which is how the forgetting-hinge conclusions were reached.

## Files

- `sfreg_measure.py` — registered predictions P1-P5, shard measurement
- `sfreg_diag.py` — explains the P2/P3 misses
- `sfreg_measure2.py` — corrected measurement, H1/H2/H3 all confirmed
- `sfreg_reach.py` — EXECUTION reachability proof with mutation control
- `sfreg_probe_share.py` / `sfreg_probe_full.txt` — bit-exact probe decomposition

## DECISIVE HISTORICAL FACT: fabrication and consumer never coexisted

`git log -L` on `sf_multipv` in `configs/pbt2_small.yaml`: it was **40** for the
entire window `w_sf_own_regret: 0.7` was live (2026-07-27 `3b4ca4737` →
2026-08-06 `ed9de8ee9`). The SAME commit `ed9de8ee9` did both:
`sf_multipv: 40 -> 6` AND `w_sf_own_regret: 0.7 -> 0.0`.

Counterfactual coverage, measured on the same 16,549 live rows:

| sf_multipv | rows with ANY uncovered legal move | mean uncovered fraction of legal moves |
|---|---|---|
| 6 (now) | 0.8922 | **0.6755** |
| 20 | 0.6986 | 0.2667 |
| 40 (during w=0.7) | 0.1512 | **0.0166** |

So while the weight was 0.7 the label was ~98% real, and when the label became
68% fabricated the weight went to 0. **The fabricated label has never driven a
meaningful share of the training gradient in production.**

This also fully reconciles the banked "68%": 0.6755 is the mean uncovered
fraction over ALL eligible rows; the 0.7623 above is the same quantity
restricted to rows where fabrication is unambiguously identifiable. Both are
correct measurements of different populations.
