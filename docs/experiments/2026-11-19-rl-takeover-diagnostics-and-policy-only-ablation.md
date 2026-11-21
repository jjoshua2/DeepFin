# RL takeover: diagnostics so far, and a same-data policy-only ablation (prospective)

**Status:** D1–D4 complete (D4 round 2, 2026-11-20: seat control closed, F12 arms
banked, recalibrated-value diagnostics); **P1 read: value loss implicated**, and the
seat control no longer qualifies it (see "P1 readout" and "D4 round 2"). P1 was
registered here before any P1 training update. Owner changed on 2026-11-19: the
RL proof moved from the Codex task to the Claude session; the week-one studies stay
closed and their artifacts untouched.

Naming: "SL base" is the supervised starting net (checkpoint sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`, called A39 in
the week-one records). F12 and Q50 are the two twelve-cycle loops of
[2026-11-18](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md) and
[2026-11-19](2026-11-19-fresh-current-policy-a39-q50-twelve-cycle-baseline.md).

## Diagnostics (no training)

**D1. Search gain, SL base on both sides.** `scripts/arena_standard.py`,
`matched_sims`, `--search-shape training --no-gumbel-noise`, tablebases on,
`--compile off`, seed 20261119, default 147k-line opening book. Results in
`~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`.

| Candidate sims vs reference sims | Games | Score | Elo (95% CI) |
|---|---:|---:|---|
| 64 vs 1 | 200 | 0.9925 | +849 (lower bound +691) |
| 256 vs 64 | 400 | 0.8275 | +272 (+243 to +305) |
| 800 vs 64 | 200 | 0.94 | +478 (+410 to +582) |

Search is far stronger than the raw net, so there is a large gain for a loop to
distil. These are same-weights search-budget contrasts, not external ratings.

**D2. Read-only pipeline audit of F12 cycle 1** (17,628 rows, 192 games; scripts
and logs copied to `~/chess-artifacts/operations/rl-claude-20261119/rl-audit/`). No data, storage or
training defect was found: stored inputs equal the play-path encoding in every
plane group (2,847 rows, max abs difference 0.0); policy targets are aligned and
legal; outcome labels have the right point of view; mirroring is consistent; only
the policy and WDL losses are non-zero; the harness reproduces the production
cycle-1 learner bit-exactly. Measured on 4,735 held-out cycle-2 rows, one cycle
(69 updates at 3e-6):

| Arm (same rows, same order) | Policy KL from SL base | mean abs value change (Q = W−L, in [−1, 1]) |
|---|---:|---:|
| Real F12 update | 0.039 | 0.075 |
| Policy only (`w_wdl` = 0) | 0.038 | 0.020 |
| Value only | 0.017 | 0.092 |
| Self-target (the SL base's own outputs as targets) | 0.00006 | 0.010 |
| Value labels shuffled | 0.043 | 0.34 |
| Zero loss | 0 | 0 |

*Corrected 2026-11-21 (PR #1452):* the policy-only KL was banked
(`rl-audit/check6_policyonly.json`: `KL(A39||model)_policy` 0.0384), and the
value-only (`check6_valueonly.json`: KL 0.0171, mean |ΔQ| 0.0916) and self-target
(`check6_selftarget.json`: KL 5.8e-5, |ΔQ| 0.0103) arms were banked but omitted.
The value-only arm moves the policy (KL 0.017) without a policy loss: the trunk
carries the value update into the policy, as P1 shows the reverse.

The policy-only arm fits the stored targets as well as the real update (held-out
CE 0.9368 vs 0.9374). A second order/mirror seed lands at KL 0.0008 from the first, so the update is a
consistent push from the data, not optimizer noise. Stored policy targets average
0.66 nats of entropy against 1.0–1.1 for the SL base; held-out policy entropy fell
1.117 → 1.015 in one cycle; fit to the targets did not improve after cycle 1
(held-out CE 0.985 base, 0.937 c1, 0.920 c6, 0.936 c12). The SL base's own value
target was 100% teacher-search value; the loops replaced it with 90% (F12) or 50%
(Q50) game outcome, about 192 outcomes per cycle. In the stored "search value" the draw probability is the net's raw root draw; only
win and loss mass move within what is left.

**D3 (result below). Target-sharpness calibration.** On about 3,000 cycle-1 positions,
candidate 64-simulation targets (stored form, temperature, `target_q_rescale` off,
`c_scale` and `c_visit` variants, per-move visit weighting, prior mixtures) are
scored against an 800-simulation search of the SL base and against the raw
policies of lc0 BT4 and a Ceres net, with game-clustered bootstrap intervals. The
same bank scores the SL base and F12 cycle 1/6/12 priors (entropy, agreement with
the deep search) to test whether the loop ratchets sharpness without earning it.

**D4 (part-result below). Head-swap matches.** Policy from one net and value from the other
(two full forwards per evaluation), against the SL base at 64 simulations, 800
games per arm on one fixed set of opening pairs: full fine-tuned net; fine-tuned
policy + base value; base policy + fine-tuned value; base + base control. Q50
cycle 12 first. This locates the lost strength in the policy or value output as it
ended up; it does not identify the cause, because the heads share a trunk.

## P1: same-data policy-only ablation

**Question.** With the F12 games, row order, dose, optimizer and learning rate
held fixed, does switching the value loss off remove the strength loss?

**Hypothesis.** The loss in F12 comes mostly from the value loss (noisy
game-outcome labels moving a value head trained on a stronger teacher, and
reaching the policy through the shared trunk). Policy-only training on the same
rows should then not be weaker than the SL base.

**Treatment.** Student starts from the SL base, fresh NAdamW as in F12. Twelve
passes, one per F12 cycle bank in cycle order (241,125 rows, 948 updates, batch
256, flat LR 3e-6, mirror 0.5, each row once), using the F12 learner path as
reproduced by the D2 harness, with `w_wdl` = 0 and every other setting unchanged.
The banks were generated by the F12 actors, not by this student, so this is an
off-policy ablation of the loss, not a new loop.

**Control.** The F12 cycle-12 checkpoint
(`3d3ca6b8ad1bb220ee197b060273bf0fadca8f49663c96e6e7ddd7ea69a19def`), trained on
the same rows in the same order with the value loss on.

**Gate before any arena.** P1's cycle 1 must reproduce the D2 policy-only arm
(held-out CE 0.9368 and mean abs value change 0.020, or bit-exact weights);
otherwise stop. Row order uses F12's compiled per-cycle orders and its original
mirror draws. This proves `w_wdl` = 0 reached the learner and nothing else moved.

**Deciding yardstick (one).** Paired match of the P1 student directly against
the F12 cycle-12 checkpoint, 1,600 games (800 opening pairs), fixed N, final read
only:

```bash
PYTHONPATH=. python -u scripts/arena_standard.py \
  --candidate <P1 student> --reference <F12 cycle 12> --sims 64 --games 1600 \
  --mode matched_sims --search-shape training --no-gumbel-noise \
  --syzygy "$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6" \
  --compile off --seed 2026111950 --label p1_vs_f12c12 \
  --pgn-out <out>/p1_vs_f12c12.pgn --out <out>/results.jsonl
```

Recorded but not deciding, with the same seed and 800 games each: P1 against the
SL base, and F12 cycle 12 against the SL base.

**Pre-committed reading** (Elo from the arena's paired interval):

- *Value loss implicated:* the head-to-head 95% lower bound is above 0 Elo.
- *Value loss not implicated (look at the policy targets):* the head-to-head 95%
  upper bound is below +20 Elo, which rules out recovering the full F12 deficit.
- Anything else: unresolved at this sample size; say so.

The two readings cannot both hold. At 1,600 games the standard error is about 6.7
Elo, giving about 84% power to detect a 20 Elo difference. The head-to-head score
is not the difference of the two scores against the base; transitivity is not
assumed.

A pass does not show a gain over the SL base and promotes nothing. Its use is to
decide whether the next real loop trains the value head at all, and on what.

**Also recorded (not deciding):** per-cycle held-out policy entropy and policy CE
to the stored targets, mean abs value change from the SL base, and after D3
lands, agreement with the 800-simulation reference.

**Cost and recovery.** About 15 GPU minutes of training, one 1,600-game arena and
two 800-game arenas (roughly two hours of GPU in all). Outputs under
`~/chess-artifacts/operations/rl-claude-20261119/p1-policy-only/`.
Nothing live is touched; the week-one banks are read-only inputs. Recovery is
deleting the output directory.

**Confounds.** The yardstick is not the instrument that measured F12's −20 Elo
(root noise off here against 0.75 there, the default 147k-line book against the
consumed DEV panel, a 300-ply cap against 450 additional plies), so that deficit is
not assumed to reproduce; the head-to-head design does not depend on it. All three
arenas run from one tree, and the config sha printed in each log (`8275f2b84d0e`
at registration) must match or the read is void. Before the read, count the
sampled arena openings that match any of the 192 F12 training starts or their
mirrors and record the count. With `w_wdl` = 0 the decoupled weight decay still
acts on the value head; at this learning rate its multiplier rounds to 1.0 in
float32 (measured in D2). Off-policy data (generated by value-trained actors); the same 192
openings recur across cycles; a reused control checkpoint from a different
runtime build (cp313, torch 2.14, source `e633212da`) loaded into the current
interpreter. D2 verified that checkpoint loads and that the harness reproduces
the original learner's cycle-1 update exactly.

## P1 readout (2026-11-19)

**Harness control.** With the value loss on, the twelve-cycle harness reproduced
every F12 learner checkpoint bit-exactly (cycles 1–12: model, optimizer, scheduler
and clip state all identical, max abs difference 0.0). **Gate passed:** P1 cycle 1
matches the D2 policy-only arm exactly (held-out CE 0.9368, mean abs value change
0.0197, outputs identical to the audit's). Arena opening overlap with the 192 F12
training starts or their mirrors: 0 of 800 pairs. All arena logs print config sha
`8275f2b84d0e`. P1 final checkpoint sha256
`b4025a75d926586614d77e08ef96d820f8c052bcb5790b82667ae1a1fd763624`.

**Deciding match** (P1 vs F12 cycle 12, 1,600 games, banked final): score 0.5609,
**+42.6 Elo, 95% CI +30.1 to +55.2**; pentanomial WW 88, WD 236, DD/WL 301, LD 133,
LL 42.

**Verdict by the pre-committed rule: value loss implicated** (lower bound above 0).
Same rows, same order, same optimizer: switching the WDL loss off is worth about 43
Elo head-to-head. The verdict is by rule. The caveat first written here was that its
robustness depended on the open seat-swap control in D4: identical weights had read
+22.6 (+0.2 to +45.3) for the candidate seat on another seed. **That control is now
closed (D4 round 2, 2026-11-20):** no candidate-seat offset was detected. Pooled
over 998 same-net pairs the candidate seat reads +6.4 (−4.2 to +17.1; −5.0 to +17.8
clustered by opening), and the +22.6 is consistent with one draw of the
move-sampling generator. Even if the pooled +6.4 is subtracted together with its
uncertainty, the P1 reading against F12 cycle 12 becomes +36.2 (+19.3 to +53.1),
still well above 0. So the P1 reading of +42.6 (+30.1 to +55.2) stands.

**Recorded, not deciding.** P1 vs SL base, 800 games: 0.4844, −10.9 Elo (−27.7 to
+5.9) — policy-only training on these rows did not beat the base. F12 cycle 12 vs SL
base on this seed was interrupted by a reboot with 278 games banked of 800 and is **unread**
(no rolling number is quoted; its game log ends in a NUL-filled tail that must be
repaired before any resume); D4's own 800-game run of the same pairing on seed
20261119 read −34.9 (−53.1 to −16.8), from a different code head (`df55413a6`) and
a resumed run. Not done: scoring the P1 checkpoints against the D3 800-simulation
bank, which the registration listed as recorded-only.

**Per-cycle, fixed held-out cycle-2 rows** (in-sample for the policy from cycle 2):

| Net | Policy entropy | CE to stored target | Policy KL from base | mean abs value change |
|---|---:|---:|---:|---:|
| SL base | 1.117 | 0.985 | 0 | 0 |
| P1 c1 / c6 / c12 | 1.013 / 0.824 / 0.784 | 0.937 / 0.917 / 0.932 | 0.038 / 0.215 / 0.299 | 0.020 / 0.048 / 0.066 |
| F12 c1 / c6 / c12 | 1.015 / 0.836 / 0.787 | 0.937 / 0.920 / 0.936 | 0.039 / 0.198 / 0.288 | 0.075 / 0.106 / 0.107 |

The policy sharpened near-identically with and without the value loss (KL from
base at cycle 12: 0.299 P1, 0.288 F12), and with the
value loss off the value output still drifted by 0.066 through the shared trunk.
So the policy target is not cleared: P1 shows the value loss costs strength, not
that the policy update gains any.

## D3 result (calibration; target quality only)

Bank and scripts: `~/chess-artifacts/operations/rl-claude-20261119/rl-audit/`
(3,080 positions from the 192 cycle-1 games; game-clustered bootstrap).

- Policy entropy (nats): lc0 BT4 1.49, Ceres net 1.53, SL base 1.12, F12 cycle
  1/6/12 1.03/0.84/0.80, stored 64-simulation target 0.69, 800-simulation target
  0.46. None of the tested settings puts a 64-simulation target in the BT4/Ceres band;
  temperature 2 (1.32) is closest below and temperature 3 (1.89) is above, so an
  untested temperature between them would fall inside.
- Against the 800-simulation search, F12 cycle 12's prior puts more mass on the deep
  survivor (0.512 → 0.580) with unchanged top-1 (0.667 → 0.667). In-sample caveat:
  these positions are from games cycles 1+ trained on.
- Value against the 800-simulation survivor Q (MSE): game outcome 0.272, SL base raw
  net 0.035 (F12 cycle 12 raw 0.037), stored 64-simulation search value 0.0145 (a
  fresh 64-simulation survivor Q 0.017). The root-node Q of a Gumbel search sits
  0.235 below the survivor Q and must not be used as a value target.
- Value landscape over legal moves: best-to-second gap 0.145 (SL base) → 0.119 (F12
  c12) → 0.108 (Q50 c12); mean |Q| 0.41 → 0.33 → 0.30; BT4 gap 0.13 and mean
  |Q| 0.40, Ceres 0.13 and 0.39, on the external-reference subset.
  Ranking against deep search did not improve.
- The best 256-simulation targets beat every transformed 64-simulation target on
  mass, top-1 and regret (for example cv0/cs0.5: mass 0.702, top-1 0.763, regret
  0.014); the default 256-simulation target does not on mass (0.674 against 0.677
  for the sharpest 64-simulation cell), and intervals overlap. `target_q_rescale` cannot be set from the self-play config
  (not mapped in `build_selfplay_gumbel_config`). Collection reuses the tree across
  plies, so stored "64-simulation" targets include inherited visits.

## D4 part-result (head swap vs SL base, 800 games each, seed 20261119)

| Arm | Score | Elo (95% CI) |
|---|---:|---|
| Q50 c12 full | 0.410 | −63.2 (−81.1 to −45.7) |
| Q50 policy + base value | 0.506 | +4.3 (−12.5 to +21.2) |
| base policy + Q50 value | 0.421 | −55.2 (−73.6 to −37.1) |
| F12 c12 full | 0.450 | −34.9 (−53.1 to −16.8) |

The Q50 deficit sits in the value output as it ended up; the arms are additive
within noise (paired B + C − A = +0.0175 ± 0.0211 in score). This locates the loss;
it does not identify the cause, because the heads share a trunk.

Controls: hybrid(base, base) reproduces the plain base-vs-base match game for game
(400/400); swapping win and loss in the value, or permuting the policy columns,
scores 0/40. The base-vs-base control read 0.5325 at 400 games (+22.6, CI +0.2 to
+45.3). The arena samples moves at temperature 0.1 from one seeded generator, so the
two games of a pair are different games (197 of 200 pairs) and identical weights do
not score exactly 0.500; a second, 20-pair same-net sample on the same openings went the other way
(0.375). That is read as sampling noise (sign test p = 0.045, one of several looks),
and the arms are read against 0. Not excluded without new games: a seat-dependent
path, testable by rerunning the control with the seats swapped. Both of those were
open here: the F12 arms B and C, and the seat-swap control. Both are closed in
"D4 round 2" below.

**Next.** A fresh loop (new driver, PR #1416) whose first arms keep the value head
on a search-value target or off, with the large opening book and per-generation
tripwires; its record is written before launch.

## D4 round 2 — seat control, F12 swap arms, recalibrated value (2026-11-20)

Same instrument as the part-result above: SL base as reference, 64 simulations,
`matched_sims`, `--search-shape training --no-gumbel-noise`, tablebases on,
`--compile off`, default opening book, move temperature 0.1, 300-ply cap. Seed
20261119 unless stated, so all 800-game arms share one set of 400 opening pairs.
Elo is from the arena's paired interval. Pooled and paired figures are per-game
score excess over pairs, with pair-clustered 95% intervals.

### Seat control (SL base in both seats)

| Sample | Orientation | Seed / openings | Games | Candidate-seat Elo (95% CI) |
|---|---|---|---:|---|
| D (round 1) | candidate hybrid(base, base), reference plain | 20261119 | 400 | +22.6 (+0.2 to +45.3) |
| Second seed | as D | 20261120 (other openings) | 400 | −3.5 (−28.4 to +21.4) |
| Same openings, new move RNG | plain vs plain | D's openings, move generator reseeded to 20261121 | 400 | +3.5 (−19.6 to +26.6) |
| Seats swapped | candidate plain, reference hybrid(base, base) | 20261119 | 800 | +4.8 (−12.2 to +21.8) |

**No candidate-seat offset was detected.** The pooled interval excludes offsets above
about +17 Elo (about +18 clustered by opening).

Forking path: the three extra samples were run because D looked large. D stays in
the headline pool, and the pool without D is a sensitivity reading only.

- By orientation, counting only the hybrid-vs-plain runs:
  - original (D plus the second seed): +9.6 (−7.2 to +26.4) over 400 pairs;
  - swapped: +4.8 (−12.2 to +21.8) over 400 pairs.
- The plain-vs-plain reseeded sample is the move-RNG control, not an orientation
  sample: +3.5 (−19.6 to +26.6).
- Pooled over all 998 distinct pairs:
  - +6.4 (−4.2 to +17.1) clustered by pair;
  - +6.4 (−5.0 to +17.8) clustered by opening (599 openings; three samples share D's
    openings).
- Sensitivity, pooled without D: +2.4 (−9.6 to +14.4).

**The +22.6 is consistent with one draw of the move-sampling generator.** Replaying D's own
openings with a fresh move stream gave +3.5. The swap and the reseeded run play D's
openings but are different games: 2 of 400 and 1 of 400 move sequences are
identical.

With identical weights in both seats, a swap alone cannot separate a seat-dependent
code path from sampling noise, because the path moves with the seat label. The
independent draws are what decide it.

Consequence for P1: the caveat in "P1 readout" is closed. Subtracting the pooled +6.4
with its uncertainty still leaves the head-to-head at +36.2 (+19.3 to +53.1).

### F12 swap arms, beside Q50 (800 games each)

| Arm | W/D/L | Elo (95% CI) | Mean plies | Draw rate |
|---|---|---|---:|---:|
| Q50 c12 full (round 1) | 181/294/325 | −63.2 (−81.1 to −45.7) | 99.7 | 0.367 |
| Q50 policy + base value (round 1) | 237/336/227 | +4.3 (−12.5 to +21.2) | 99.4 | 0.420 |
| base policy + Q50 value (round 1) | 176/322/302 | −55.2 (−73.6 to −37.1) | 94.9 | 0.403 |
| F12 c12 full (round 1) | 201/318/281 | −34.9 (−53.1 to −16.8) | 99.4 | 0.398 |
| F12 policy + base value | 209/346/245 | −15.7 (−33.8 to +2.5) | 98.6 | 0.432 |
| base policy + F12 value | 215/328/257 | −18.3 (−36.2 to −0.4) | 93.4 | 0.410 |

Additivity, as paired B + C − A in score:
- Q50: +0.0175 ± 0.0211.
- F12: +0.0013 ± 0.0237 (in Elo, B + C = −33.9 against A = −34.9).

Reading:
- **Q50's loss is nearly all in the value output.**
- **F12's loss splits about evenly** between policy and value. Neither half resolves
  on its own at 800 games.
- As before, this locates the loss in the outputs as they ended up. It does not
  identify the cause, because the heads share a trunk.

### Recalibrated value (diagnostic arms)

At every net evaluation, root and leaves, the fine-tuned net's WDL logits (W, D, L
order, side to move) are replaced by `logits / T + [0, b, 0]` before the search
softmaxes them.

Parameters are the "base_shape" fits from the offline recalibration study: the
minimum cross-entropy fit to the SL base's own WDL on the 3,080 D3 audit rows.

| Net | T | b |
|---|---:|---:|
| F12 c12 | 0.6353 | −0.5878 |
| Q50 c12 | 0.5445 | −0.6383 |

**Not pre-registered.**
- These arms were proposed after the offline recalibration study.
- Their parameters were fitted on the audit rows, not on arena games.
- No kill or success threshold was committed before they ran.

They are diagnostics and decide nothing. A post-hoc monotone recalibration is an
inference-time patch. It is not evidence that training on these targets is fine.

**Take-effect proof, run before any games were played:**
- On the 3,080 audit rows, through the arena's own evaluator, mean |Q| and mean
  P(draw) moved as follows:

  | Value | Raw \|Q\| / P(draw) | Recalibrated \|Q\| / P(draw) |
  |---|---|---|
  | Q50 | 0.298 / 0.515 | 0.364 / 0.513 |
  | F12 | 0.331 / 0.528 | 0.386 / 0.513 |
  | SL base, for comparison | 0.409 / 0.513 | — |

- The evaluator's WDL matched raw / T + [0, b, 0] to within 4e-6.
- The policy output was unchanged (max difference 0).
- Each arena log prints the parameters as held by the hybrid net itself.
- Identity control: T = 1, b = 0 reproduced the unrecalibrated base-policy + Q50-value
  arm game for game (40 of 40 move sequences identical).

| Arm (800 games) | Elo (95% CI) | Unrecalibrated counterpart | Paired gain, Elo-equivalent (95% CI) | Share of deficit recovered (bootstrap 95%) | Mean plies | Draw rate |
|---|---|---|---|---|---:|---:|
| R1 base policy + recal. Q50 value | −8.2 (−26.1 to +9.6) | −55.2 | +46.7 (+22.1 to +71.9) | 0.85 (0.50 to 1.18) | 94.3 | 0.404 |
| R2 base policy + recal. F12 value | −12.2 (−29.7 to +5.3) | −18.3 | +6.1 (−18.9 to +31.1) | not determined | 94.4 | 0.412 |
| R3 Q50 c12 full, recal. value | −14.3 (−32.6 to +3.9) | −63.2 | +48.5 (+25.3 to +72.2) | 0.77 (0.47 to 1.06) | 99.1 | 0.426 |
| R4 F12 c12 full, recal. value | −0.9 (−19.0 to +17.2) | −34.9 | +34.0 (+8.4 to +60.0) | 0.97 (0.37 to 1.56) | 99.6 | 0.417 |

Reading:
- **Q50:** most of the play deficit from the value output goes away when the value is
  put back on the SL base's confidence and draw shape. The offline study found Q50's
  value gap was mostly calibration, but not entirely:
  - its raw gap of +0.0070 splits into calibration +0.0041 and information +0.0030
    (0.0012 to 0.0044);
  - a child-level top-1 deficit of −0.031 (−0.053 to −0.009) remains that no
    monotone map touches.

  The R arms show that, on this net, a two-parameter map recovers most of the play
  deficit. They do not show that the value head lost no information.
- **F12, value only (R2):** the gain does not resolve. F12's value deficit is small,
  and the offline study found it was mostly information.
- **F12, full net (R4):** R4 recovers more than R2 and the policy share together
  would suggest. With these intervals the R4-vs-R2 contrast is unresolved and is not
  read as an interaction.
- No recalibrated arm is significantly below or above the SL base.

### Procedural notes

- **f12_B rerun from scratch.** The pre-reboot partial (429 games) had a NUL-filled
  tail in its game log. It was quarantined untouched and f12_B was rerun whole.
- **One f12_C pair replayed.** One f12_C game reached the 300-ply cap unresolved, and
  the strict-tablebase arena refuses to call that a draw, so the run stopped. The
  arena's `--resume` discarded that one-colour pair and replayed it in full: 1 pair
  of 400 (pair 352).
- **Two round-1 arms also replayed pairs on resume:**
  - F12 c12 full stopped at the same 300-ply refusal (pair 104). That left 22 pairs
    with one colouring played; all 22 were discarded and replayed, giving 822 game
    rows for 800 scored games.
  - Q50 c12 full stopped on a CUDA error and replayed 21 pairs (821 rows).
- **Size of the length selection.** Only the pair that hit the cap is selected on game
  length, so about one game per affected arm (F12 full and f12_C). The other replayed
  pairs were simply in flight when the run stopped, and their replays are fresh draws.
- **Scratch harness only; no repository code was changed.**
  - A driver loads a "hybrid" net, then calls `scripts/arena_standard.py`'s `main()`
    unchanged. The hybrid takes the policy from one checkpoint and the WDL from
    another, using two full forwards.
  - Spec keys add the WDL transform above.
  - An environment variable reseeds the arena's generator after the openings are
    sampled; this is the same-openings control.
  - Another environment variable sets the reference seat; this is the swap.
  - Every checkpoint is sha256-checked at load.
- **Command shape (every arm):**

  ```bash
  PYTHONPATH=. python -u <bank>/headswap_arena.py \
    --candidate <checkpoint or hybrid spec> --reference <SL base or hybrid spec> \
    --sims 64 --games 800 --mode matched_sims --search-shape training \
    --no-gumbel-noise \
    --syzygy "$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6" \
    --compile off --seed 20261119 --label <arm> \
    --pgn-out <bank>/pgn/<arm>.pgn --out <bank>/results/<arm>.jsonl
  ```

  The code head is detached main `df55413a6`; arena logs print config sha
  `8275f2b84d0e`.
- **Bank:** `<bank>` = `~/chess-artifacts/operations/rl-claude-20261119/headswap/`.
  It holds the specs, per-game logs, PGNs, result rows and all analysis scripts and
  output (`analysis_v2.*`, `table.*`, `check_recal*`). The README there lists every
  file and the harness changes.
- **Recalibration study:** `~/chess-artifacts/operations/rl-claude-20261119/value-recalibration/`.

### What this means for the next loop

All of the following are candidates for a future registered experiment, not
decisions.

- The dominant measured loss in Q50 is the value output's confidence and draw-shape
  drift (flatter |Q| than the SL base). Most of its play deficit is recoverable by a
  two-parameter map at inference time. Offline, an information part of +0.0030
  (0.0012 to 0.0044) and a child-level top-1 deficit of −0.031 (−0.053 to −0.009)
  remain that no monotone map touches.
- F12's loss is not concentrated in one output. The evidence on calibration
  conflicts:
  - R4 (full F12, recalibrated) recovers the full-net deficit: −0.9 (−19.0 to +17.2),
    share 0.97 (0.37 to 1.56).
  - R2 (value only) is unresolved at +6.1 (−18.9 to +31.1), and the offline study
    found F12's value gap mostly information.

  Whether calibration clears an F12-like loop is unresolved.
- Candidate: train value on the net's own self-play search value, or keep the value
  loss off, as P1's +42.6 head-to-head suggests. This is not the SL base's regime: the
  SL base's value target was a teacher's search value (see D2), a different target
  from the net's own 64-simulation search.
- Candidate: monitor mean |Q| and P(draw) on the fixed 3,080 audit rows every
  generation, as a tripwire against confidence drift.
- Candidate: apply or learn a calibration step for the value output, registered with
  its own threshold and an arena yardstick on fresh openings.
- The seat control is closed. At 400 games a 64-simulation arena's 95% interval is
  about ±22 Elo, so a single 400-game read needs that margin.
