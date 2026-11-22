# RL run 1 pilot: results (2026-11-21)

These are the results of the pilot registered in
[the games-not-rows amendment](2026-11-21-rl-run1-amendment-games-not-rows.md) (Step 1) and made executable by
[the launch record](2026-11-21-rl-run1-pilot-launch-record.md). The verdicts below are read by the pre-registered
rules, in the same session as the readout.

**Summary.**

| Match | Verdict |
|---|---|
| Match 3: S vs base | **damage**, −20.0 [−36.8, −3.3]; S not adoptable; no extension |
| Match 1: Zq vs base (the gate) | −10.0 [−23.2, +3.2]: the recipe did not reproduce the damage at this dose. Zq is not adoptable |
| Match 2 | skipped by rule |
| Match 4: P0 policy + base value vs base | no boundary crossed in three looks, so the reading is **unread**. The look-3 lower bound misses 0 by 0.1 Elo |

By the adoption order, no value target is adopted and the pilot **stops for the owner**.

Bank: `R` = `~/chess-artifacts/operations/rl-run1-pilot`, frozen read-only, with `R/SHA256SUMS` [`561009b7`]
(every entry verified OK on 11-21) and `R/README.md` [`30bc6942`]. Each file is cited with its sha256 prefix.
Every number here was read from the bank files named, or computed from them where marked "derived".

**Nets.** All three cells and every match ran at `$PIN` = `0c795fb7ab5a46e20848f6b0013de5a686fcb8ad`
(`R/PIN.txt` [`a3c5daea`]); every arena result row's `git_sha` equals it. The checkpoints:

| Net | sha256 prefix | File |
|---|---|---|
| SL base | `a39f900c` | `R/sl_base.pt` |
| S | `66ceb271` | `R/arm_S/gen_0001/train/model.pt` |
| Zq | `ad16d2f1` | `R/arm_Zq/gen_0001/train/model.pt` |
| P0 | `4503bd4d` | `R/arm_P0/gen_0001/train/model.pt` |

## 1. Verdicts by the registered rules

Common settings for every match:
- seat-balanced: run A has the arm as candidate, run B as reference, both on the same seed;
- 64 sims, `matched_sims`, search shape `training`, no Gumbel noise, compile off, move temperature 0.1;
- openings from `R/openings/eval.pgn` at 16 opening plies;
- `--max-plies 600 --on-max-plies abort`, with the production Syzygy pair;
- the pool clusters by opening.

Elo is the arm's, against the SL base.

### Match 3: S vs base (800 games, seed 2026112103)

Rule (amendment, match 3): "Damage iff UB95 < 0. 'S adoptable' iff LB95 > −10 (Step 2)."

Extension rule (launch record): "**If S's estimate is > 0 but LB95 ≤ −10**, S is extended with match 4's look
schedule."

| Quantity | Value |
|---|---|
| **Pooled** | **−20.0 [−36.8, −3.3]** (excess −0.0288, SE 0.0122) |
| Run A (S as candidate) | −9.6 [−35.0, +15.8] |
| Run B (S as reference) | −30.5 [−54.0, −7.2] |
| Seat offset (candidate seat) | +10.4 [−7.3, +28.2] |
| Clusters | 200 openings, 0 repeated within a run; A/B correlation −0.059 |
| S's W/D/L | 112/165/123 in run A, 95/175/130 in run B; 207/340/253 in total |

- **Verdict: damage** (UB95 −3.3 < 0).
- **Not adoptable:** LB95 −36.8 ≤ −10.
- **No extension:** the estimate is not > 0.

Sources: `R/arenas/s_vs_base/pool.json` [`db8beb4f`], `pool.log` [`ea015d26`], `results.jsonl` [`75f6b118`]. W/D/L
is from `s_vs_base_1_A.games.jsonl` [`1b3830af`] and `s_vs_base_1_B.games.jsonl` [`4d45c491`], scored from the
arm's side (derived).

### Match 1: the gate, Zq vs base (1,600 games, seed 2026112101)

Rule (amendment, match 1): "'Damage reproduced' iff UB95 < 0. If not, match 2 is skipped and the record says the
recipe did not reproduce the damage at this dose. That is NOT evidence the recipe is safe (see Step 2's adoption
rule)."

Adoption rule (amendment Step 2): "A target is adopted only if its match against the base has **LB95 > −10 Elo**."

| Quantity | Value |
|---|---|
| **Pooled** | **−10.0 [−23.2, +3.2]** (excess −0.0144, SE 0.0097) |
| Run A (Zq as candidate) | −24.4 [−42.3, −6.5] |
| Run B (Zq as reference) | +4.3 [−13.9, +22.6] |
| Seat offset (candidate seat) | −14.3 [−26.7, −2.0]. The interval excludes 0. The pooled estimate cancels a seat offset by construction |
| Clusters | 400 openings, 0 repeated within a run; A/B correlation +0.070 |
| Zq's W/D/L | 210/324/266 in run A, 241/328/231 in run B; 451/652/497 in total |

- **Verdict: "the recipe did not reproduce the damage at this dose"** (UB95 +3.2 ≥ 0). This is not evidence of safety.
- **Zq is NOT adoptable:** LB95 −23.2 ≤ −10.
- **Match 2** (the recalibration head-to-head) was **skipped by rule**.

Sources: `R/arenas/gate/pool.json` [`c3e4f5c7`], `pool.log` [`f3fe097f`], `results.jsonl` [`cfd42dc1`]. W/D/L is from
`gate_1_A.games.jsonl` [`c34b3ac4`] and `gate_1_B.games.jsonl` [`d0f49763`] (derived).

### Match 4: P0's policy grafted onto the base's value, vs base (looks of 1,600 games)

Rule (launch record, as in the amendment): "stop with 'the policy pays' iff est − z·SE > 0; stop with 'does not pay
+10' iff est + z·SE < +10; otherwise run the next look; after look 3 without a stop, the reading is 'unread'.
There is no other stopping point."

The boundaries are z = 3.471 / 2.454 / 2.004. Each look's pool combines looks 1..k and clusters an opening drawn in
several looks as one cluster.

**The graft seat.** The P0 seat is a hybrid: P0's policy with the SL base's value (`--candidate-value-from` /
`--reference-value-from` = `R/sl_base.pt`, recorded in each run header's `value_candidate.value_from`).

| Look | Seed | Cumulative games | Clusters (recurring across looks) | Pooled Elo [est ± z·SE] | z | Excess (SE) | Reading |
|---|---|---|---|---|---|---|---|
| 1 | 2026112141 | 1,600 | 400 (0) | +3.5 [−19.6, +26.5] | 3.471 | +0.0050 (0.0095) | continue (LB < 0; UB ≥ +10) |
| 2 | 2026112142 | 3,200 | 759 (41) | +7.4 [−4.0, +18.9] | 2.454 | +0.0107 (0.0067) | continue |
| 3 | 2026112143 | 4,800 | 1,087 (107) | **+7.7 [−0.1, +15.4]** | 2.004 | +0.0110 (0.0056) | **no stop → "unread"** |

- **Verdict: "unread".** At look 3 the lower bound est − z·SE is −0.078 Elo, so it misses 0 by 0.1 Elo. **This is
  not a pass:** the registered rule has no "nearly", and there is no fourth look.
- The upper bound +15.4 is above +10, so "does not pay +10" did not fire either.
- **Secondary only:** the ordinary 95% interval (z 1.96) at look 3 is +7.7 [+0.1, +15.2] (`elo_ci95` in the pool).
  It is not the registered boundary and decides nothing.

Per run, at look 3. The first interval is at the look's z = 2.004 (`elo_ci`); the 95% interval at z = 1.96
(`elo_ci95`) is the secondary:

| Run | Elo [z 2.004] | Elo [95%] |
|---|---|---|
| Arm as candidate (A) | +9.6 [−1.1, +20.4] | [−0.9, +20.2] |
| Arm as reference (B) | +5.7 [−5.2, +16.6] | [−5.0, +16.4] |
| Seat offset | +2.0 [−5.6, +9.6] | [−5.4, +9.4] |

Each look on its own (descriptive, derived): look 1 +3.5, look 2 +11.3, look 3 +6.5. Each is ≈ (Elo of the A row −
Elo of the B row) / 2 from `results.jsonl`, the arm's Elo in A and minus the candidate's Elo in B; for look 2,
(+9.56 + 13.03) / 2 = 11.3. The same values are in `pool_look3.json` under `looks[k].result.pooled.elo` (3.47, 11.30,
6.52). The `look_inputs` entries carry no result.
Arm's W/D/L per run:

| Look | Run A | Run B |
|---|---|---|
| 1 | 254/313/233 | 236/323/241 |
| 2 | 239/344/217 | 258/314/228 |
| 3 | 233/352/215 | 244/324/232 |

That totals 1,464/1,970/1,366 over 4,800 games (derived).

Sources:
- `R/arenas/graft/pool_look1.json` [`d0730fee`], `pool_look2.json` [`900486c4`], `pool_look3.json` [`6316e097`];
- their `.log` files [`df3e7c1e`, `7377ce02`, `f2d69d62`];
- `results.jsonl` [`e9391a58`];
- the six `graft_{1,2,3}_{A,B}.games.jsonl` (sha256 in `R/SHA256SUMS`).

### Adoption order (amendment Step 2)

1. "S if match 3 has LB95 > −10": no (−36.8).
2. "else Zq if match 1 has LB95 > −10": no (−23.2).
3. "else, if match 2 read shape, Zq with the recalibration refitted ...": match 2 was skipped, so no.
4. **"else stop for the owner."** This is where the pilot stands.

### Game integrity, all four matches

- **Unresolved games: 0** in all 7,200 games (S 800, gate 1,600, graft 4,800). Every result row has
  `max_plies_unresolved` 0, `max_plies_aborts_banked` 0, `resumed_pairs` 0 and `truncated` false, and no
  `*.unresolved.jsonl` sidecar exists.
- **Longest game per match:**

  | Match | Longest game |
  |---|---|
  | S | 269 plies (run B, rules, draw) |
  | gate | 365 plies (run A, rules, 1-0); the next is 316 (run B, Syzygy, 0-1) |
  | graft | 268 plies (look 2 run B, rules, draw) |

- **Terminations:** about half Syzygy adjudication and half game rules in every run. For example, gate A had 414
  Syzygy and 386 rules (derived from the game logs).

## 2. Post-hoc contrasts (NOT registered)

These contrasts were chosen after the results were seen. They decide nothing.

**Method.** Each contrast uses the pools' pooled excess and SE. The three matches have different seeds. Their opening
sets overlap: the graft's 1,087 opening clusters share 65 openings with S's 200 and 113 with the gate's 400 (computed
from the game logs with the pool's opening key, `opening_fen` + `opening_uci`).
- **Covariance is negligible (computed).** The cross-match covariance of the two estimates, from per-opening cluster
  values over the shared openings, is 2.1e-7 (graft, S) and 1.2e-7 (graft, gate). With it, the combined SEs are
  unchanged to four decimals (0.0134, 0.0112).
- SE = √(SE₁² + SE₂²) on the score scale;
- z = difference / SE;
- the Elo difference is the difference of the pooled point estimates;
- the Elo SE uses the delta method at each estimate.

| Contrast | Score difference (SE) | z | Elo difference (SE ≈) |
|---|---|---|---|
| graft (look 3) − S | +0.0398 (0.0134) | 2.96 | +27.7 (9.4) |
| graft (look 3) − Zq | +0.0254 (0.0112) | 2.27 | +17.7 (7.8) |

The z column is without the covariance term (2.959, 2.274); with it, 2.963 and 2.276.

**The policy side looks alike across the cells** (fit numbers, Section 3):
- held-out policy CE 0.9816 / 0.9825 / 0.9815 for S / Zq / P0;
- probe KL from the base 0.0788 / 0.0788 / 0.0795;
- top-1 agreement with the base 0.920 / 0.921 / 0.919.

Together with the contrasts, this supports a hypothesis: **the policy training is not what lost strength; the
value-head training is the candidate.** It is a hypothesis these numbers support, not a registered verdict:
- Two contrasts were chosen after the fact.
- The graft seat also differs in kind: it runs two forwards per evaluation, P0's net for the policy and the SL base's net for the value.
- P0's own value head was never played.
- **No played seat isolates the value head.** The contrast compares P0's policy net, on the base's value, against
  S's and Zq's WHOLE nets. S's and Zq's own policies were never played on the base's value. So the policy halves of
  S and Zq are not shown harmless by any game: "the value-head training is the candidate" rests on the policy fit
  numbers above, not on a played seat. The offline diagnosis below speaks to the value heads directly, but it is
  offline and 1-ply, not a played seat, so it addresses this gap only partly.
  *Update (11-21, [follow-up results](2026-11-21-rl-run1-followup-results.md)):* match 6b now isolates S's value head on the base's own policy, at −21.4 [−27.7, −15.2]. Zq's value head is still not isolated by a played seat.

### Offline value diagnosis (report-only; pointer)

Bank: `~/chess-artifacts/operations/rl-claude-20261119/pilot-value-diagnosis/`. Every entry in its `SHA256SUMS`
[`5a1bc230`] verifies OK. The files:
- `README.md` [`ae5e0866`], the current version, with Addendum 1;
- the pre-addendum text, frozen as `README.v1-20261121T1323.md` [`0bdac6a5`];
- `PREREG.md` [`691dd3a4`], sealed 13:06, before the first output at 13:08. There were no matches and no training.
The numbers below were read from `out/*.json`:

- **Recalibration fit** to the base's WDL on the 39,237 hold-out rows (416 games):

  | Net | (T, b) | Share of KL(base ‖ net) removed |
  |---|---|---|
  | S | (0.853, −0.120) | 0.24 [0.23, 0.25] |
  | Zq | (0.704, −0.227) | 0.60 [0.58, 0.62] |
  | P0's own head | (1.028, +0.042) | 0.02 |

  Sources: `out/recal_{S,Zq,P0}.json` [`6522a6fc` / `7e665c5e` / `9f21b655`] and `out/holdout_analysis.json`
  [`7ae11cf7`].
- **1-ply value regret on the frozen audit set** (FEN-only, so the history planes are empty; n 3,723; CIs by
  position), net minus base, in cp:

  | Net | Regret vs base |
  |---|---|
  | S | +6.8 [+4.6, +9.2] |
  | Zq | +2.3 [+0.6, +4.0] |
  | P0's own head | +1.9 [+0.2, +3.6] |

  (T, b) changes 1.0% / 1.8% / 0.05% of choices and none of the regret: after (T, b) the figures are +7.0 / +2.5 /
  +1.9 (`out/value_regret_recal.json` [`995d1165`]).
- **With history**, on 2,433 rl-audit rows, the order is the same, S +1.6, Zq +0.8, P0 +0.2, but no difference
  excludes zero (CIs by game; `out/value_regret_history.json` [`2bcd8d01`]).
- **Calibration to outcomes on hold-out.** Both trained heads are better calibrated than the base: squared error to
  z minus the base's is S −0.011 [−0.016, −0.006] and Zq −0.017 [−0.024, −0.009]
  (`out/supp_outcome_distance.json` [`f1529197`]).
- **The 64-sim search value is no more biased than the base:** overall abs(z − search) 0.359 vs abs(z − base) 0.373,
  and the two are about equal bin by bin (`holdout_analysis.json`, `target_calibration`).

- **Sibling ordering** (PREREG addendum 1, `PREREG.addendum1.md` [`2a07543c`]; sealed in `PREREG.sha256` [`79509dd1`]
  together with the script `scripts/sibling_order.py` [`4dd4ea80`] before the run; result `out/sibling_order.json`
  [`bb8e5b16`], `logs/sibling_order.log` [`c917a34b`]).
  - **Data:** 2,433 rl-audit rows with full history, from 190 games (counted here from `rl-audit/positions.npz`
    `game_id`), giving 81,807 sibling pairs scored by SF 8M.
  - **Primary bin:** "clear+", an SF gap of ≥ 50 cp with the pair not decided: 38,705 pairs. The base orders 0.8554
    of them correctly.
  - **Net minus base**, with game-clustered CIs:

    | Net | Raw | After (T, b) |
    |---|---|---|
    | S | −0.0096 [−0.0129, −0.0064] | −0.0093 |
    | Zq | −0.0007 [−0.0032, +0.0017] | — |
    | P0's own head | −0.0020 [−0.0040, +0.0001] | — |

  - **Where S's drop sits:** about −0.010 in every piece band from 8 to 32 men (8–12 −0.0101, 13–22 −0.0100,
    23–32 −0.0093). It is not concentrated where S's value moved most.
  - **Disagreement rows:** S's drop is no larger on rows where the stored search value and the base disagree most
    (−0.0089 on the top decile, 244 rows, vs −0.0097 on the rest). This decile is an ANALOG computed on the rl-audit
    rows; the pilot's hold-out decile could not be joined, because it has no deep-SF labels.

**Status, by the addendum's pre-registered rule.**
- **S: supports "ranking content broke".** The CI is entirely below 0, and (T, b) does not restore it.
- **Zq: not separated.** This does NOT explain Zq's −10; Zq's shape account (0.60 of its divergence is removable by
  (T, b)) stays open. *Update:* matches 7 and 8 are consistent with the shape account in play (match 8 is not a test of shape vs content) ([follow-up results](2026-11-21-rl-run1-followup-results.md)).
- **P0's own head: not separated.**

The mechanism, that the value loss changes what the head says (content) rather than its scale (shape), is therefore
an S-specific HYPOTHESIS, not a verdict. It is offline and 1-ply, not a played seat.

## 3. Training facts (fit numbers, not strength)

From `R/arm_{S,Zq,P0}/gen_0001/train/metrics.json` [`3eb8e85e` / `7f3c610d` / `18f4c79f`] and `manifest.json`
[`5121de42` / `b46d1c77` / `f53fcee5`].

**Shared by all three cells:**
- 1,412 optimizer updates;
- window and drawn rows 722,742, each presented exactly once (`scheduled_exact_once`, presentation histogram {1: 722,742});
- 39,237 hold-out rows, never presented;
- batch 512; SL mix 0; not trained on capped rows;
- realised base LRs [6e-5, 3e-6, 3e-6, 3e-6] (`realised_optimizer.base_lrs` and `realised_lr_param_groups`);
- optimizer `AuroraWithAuxAdam`, built split [48, 0, 142, 289] with `names_equal` true against the init checkpoint (the Aurora split matches);
- policy target temperature 1.0.

Held-out metrics are on the 39,237 hold-out rows: policy CE to the stored target, and value NLL to the game outcome
(`learn.heldout_metrics`). The probe has 2,048 rows and compares against the previous generation, which is the SL
base.

| Cell | Value recipe (realised `w_wdl`, search share) | Held-out policy CE, before → after | Held-out value NLL, before → after | Probe mean abs(dq) | Probe top-1 agreement with the base | Probe policy KL from the base | Updates/s, wall |
|---|---|---|---|---|---|---|---|
| S | 1.0, 1.0 | 1.0280 → 0.9816 | 0.7422 → 0.7271 | 0.0520 | 0.9199 | 0.0788 | 3.55, 398 s |
| Zq | 1.0, 0.5 | 1.0280 → 0.9825 | 0.7422 → 0.6884 | 0.0703 | 0.9214 | 0.0788 | 3.21, 440 s |
| P0 | 0.0, 0.0 | 1.0280 → 0.9815 | 0.7422 → 0.7511 | 0.0334 | 0.9189 | 0.0795 | 3.55, 398 s |

**How to read these.**
- Zq's value NLL falls most because half of its value target is the outcome that NLL scores. That is fit to the
  target, not strength: Zq lost −10.0 and S lost −20.0.
- P0 has `w_wdl` 0, but its value head still moved: mean abs(dq) 0.033, and its NLL rose to 0.7511. The trunk is
  shared. The graft did not play that head.
- The launch record's estimates were 16 min per cell and ≈ 1,406 updates. The measured walls are 398–440 s, and
  the updates are 1,412.

## 4. Collection facts

From `R/collect/gen_0001/collect/manifest.json` [`61fc049f`] (its `stats` block equals `stats.json` [`48e8239a`]),
`games.jsonl` [`0ac540bf`] and `R/collect_gates.json` [`fe21e943`].

| Fact | Value |
|---|---|
| Games / rows / shards | 8,000 / 761,979 / 39 (95.25 rows per game); 3 processes × 128 concurrency, 64 sims |
| Results (White / draw / Black) | 2,471 / 3,920 / 1,609; decisive 0.51 |
| Plies | mean 111.57 (96.24 after the book); longest 344; 0 games at the 450-ply cap; 0 unresolved |
| Tablebase-adjudicated share | 0.541 (4,328 games); mates 2,231 |
| Target entropy / prior entropy | 0.7069 / 1.1199 |
| KL(target ‖ prior) | 0.2982 |
| Top-1 agreement (target vs prior) | 0.7780 |
| Reused-root share | 0.6456 |
| Within-collection repeat rate | 0.000205; 8,000 unique openings |
| Per-process rates | 1,741.7 / 1,734.9 / 1,754.3 games/h (2,667 / 2,667 / 2,666 games; wall 5,512 / 5,534 / 5,471 s) |
| Whole collection | 8,000 games / 5,534 s (the slowest process) = **5,204 games/h** (derived). The per-process rates sum to 5,231. The launch record planned at 5,021 |
| Played | 06:41:32–08:13:46 −05:00, from the per-process `started_at` and wall (derived) |

The pre-training check passed: `R/post_collect_checks.sh` [`4917f83f`] writes `collect_gates.json` only after the check exits 0. Recounted here: `games.jsonl` has 8,000 lines, and `game_resumes` is absent from the manifest and its stats.

## 5. Deviations and corrections to the launch record

1. **The seal step was OOM-killed after every game had been played.**
   - Exit 137 at 08:26:46 (`R/collect.exit.first_attempt_137` [`86f57604`]); anon-rss 83.7 GB on a 98 GB box (PR #1470 body).
   - Fixed by #1470 (`0c795fb7a`), which streams `generation_stats` per shard.
   - The games are untouched by the failure and the rerun. `games.jsonl` is byte-equal in the pre-seal backup and the sealed collect [`0ac540bf` both], and so is `starts.txt` (`R/collect_backup_20261121_0827/starts.txt` and `R/collect/gen_0001/collect/starts.txt` [`948d4927`] both).
2. **The games were PLAYED at C0 `6c72b821b` and SEALED at `0c795fb7a`** by a seal-only rerun from 09:51:00 to 09:54:42 (`R/reseal.started` [`00e497ec`], `R/reseal.exit` [`dc8a16ea`], `R/reseal.log` [`3f1bf579`]: "keeping finished collect process(es) [0, 1, 2]").
   - **The manifest's `git.sha` is the sealing commit.** Nothing in the manifest records the playing commit.
   - The playing commit is evidenced only by the play log, which loads `_mcts_tree` from the `rl-run1-pilot-6c72b821b` worktree (`R/collect.log` [`3ebe8c39`]), and by `R/README.md`.
   - The seal-only rerun also left the manifest's aggregate timing meaningless: `games_per_hour` is null, `wall_seconds` is 2.3e-6 and `procs_resumed` is [0, 1, 2]. The per-process walls above are the playing run's.
3. **`collect.exit` was rewritten** to "0 <reseal time>" (`R/collect.exit` [`dc8a16ea`], identical to `reseal.exit`) so the post-collect check script would run. The originals are kept (`collect.exit.first_attempt_137`, `collect.exit.reseal`, `collect_backup_20261121_0827/`).
4. **`$PIN` = `0c795fb7a`.** An earlier pin, `db977e47f` (#1464's merge), was replaced before any training, because the seal fix had to merge first. All three prerequisites the launch record listed as open were merged before `$PIN`: #1459 `d268f1e4b`, #1462 `ac688fb3c`, #1463 `daa5d62fd`. `git diff --name-only C0 $PIN` is `R/c0_to_pin_diff.txt` [`08fc1a98`] (19 files, including `simple_rl/collect.py`), so the guards were mandatory.
5. **Guard (a): byte-equal.** All six files `guard_a_{S,Zq,P0}_{6c72b821b,0c795fb7a}.json` share one sha256, [`715fb5d1`] (722,742 rows each).
   - The digest is the same for all three cells, not only across commits. It covers the decoded rows after `prepare_targets` and before the cell's value blend, so it checks the shared input, not the cell-specific target.
6. **Guard (b): byte-equal**, with a correction to the record's text.
   - The files are `guard_b_{db977e47f,6c72b821b,0c795fb7a}_repfix0.json` [`7ccd71e0`] and `guard_b_{db977e47f,6c72b821b,0c795fb7a}_repfix1.json` [`634a5a32`] (each stamp holds for all three): 300 games, 24,961 positions.
   - **The correction.** The SL base's `input_history_encoding` is `lc0_root_legacy_meta` (collect manifest, `config.arch`). For it, `selfplay/network_turn.py:_batch_encoder_pair` picks `batch_encode_146_lc0_root_legacy_meta[_bf16]`. The record's guard text named only `batch_encode_146_lc0_root[_bf16]`.
   - The run digested all three encoder pairs, so the encoder actually used is covered.
7. **The Stockfish label cost** the launch record quotes (53–91 cores) is corrected by evidence-log I34: 53–81 cores at the floor of 150k nodes; 91 is floor-to-cap.
8. **The per-game move list now exists:** #1444 merged at `8f9f1ab83`. The launch record's "Gap: the move record" applies to this collection, which was played before it.
9. **Look 2 started 14 minutes after look 1 finished:** `graft_1.exit` 11:51:14, `graft_2.started` 12:05:18 (`R/graft_1.exit` [`e0d411c2`], `R/graft_2.started` [`e6cb324e`]). This was an operator gap. Each look is a fresh invocation on its own seed, so it has no effect on the results.
10. **Two gate games ran past the old default.** The gate had games of 365 and 316 plies, which the old `--max-plies 300` default would have aborted. The registered 600 handled both. Both games are past the banked maximum of 298 that the launch record's max-plies rule cites.

## 6. What is NOT concluded

- **Multi-generation behaviour.** This is one generation from the SL base.
- **Zq's safety.** The gate's failure to read damage is not safety; Zq's LB95 is −23.2.
- **A deployable policy net.** The graft is a hybrid seat (two forwards: P0's policy, the SL base's value), not a single deployable net.
- **P0 as a full net.** P0's own value head, which drifted (mean abs(dq) 0.033), was not played. *Closed by match 5* ([follow-up results](2026-11-21-rl-run1-followup-results.md)): P0 as a single net read +2.2 [−4.2, +8.6], "single-net policy-only step is adoptable".
- **That the policy pays.** Match 4 is unread. The near-miss is not a pass.
- **Recalibration.** Match 2 did not run, so the pilot says nothing about whether recalibration recovers S's or Zq's loss. *Update* ([follow-up results](2026-11-21-rl-run1-followup-results.md)):
  - match 7 (new registration, not match 2): Zq + recal vs Zq +21.4 [+15.1, +27.6], "the loss is shape";
  - match 6, S's value recalibrated vs raw: R = +5.34 [−3.48, +14.15], "unread";
  - match 8 (its own record): Zq + recal vs the SL base +9.6 [+3.3, +16.0], "adoptable" and "improves on the base". It nominates "Zq + per-generation recalibration" under its own record; the (T, b) was fitted post hoc on this one net.

  No pilot verdict changes.

## 7. Owner decisions now open

No recommendation is made here.

1. The adoption order has reached step 4, "stop for the owner". What comes after the pilot?
2. Match 4 is finished and unread. Should a new, separately registered graft match be commissioned, and to what size?
3. Should the recalibration head-to-head (match 2) be run anyway, outside the skip rule, as its own record? (Answered by match 7, a new registration; see [follow-up results](2026-11-21-rl-run1-followup-results.md).)
4. The launch record's owner rows L4 (S's role), L5 (collection B with cells K-blend and K-z2, matches K1–K3) and L6 (the parked anchor: the three-way blend plus an SF-opponent share, or the outcome-conditioned alternative) remain unfilled.
5. Which value target, if any, should any next generation train? None is adoptable by the registered rule.
6. Should a next generation train the policy only, with the value held at the SL base's? This is a design question, not a result.

Fact, not a decision: the three follow-up matches are pre-registered in
[the follow-up match record](2026-11-21-rl-run1-match5-p0-single-net.md) (PR #1473, merged `51b3d56f3`). They are
P0 as a single net (5); S's value on the base policy, raw and recalibrated (6a, 6b); and Zq recalibrated vs Zq (7).
They are played in the order 5, 7, 6b, 6a, at 6,400 games each. They launched on 2026-11-21 after the merge (output
root `~/chess-artifacts/operations/rl-run1-followup/`, launched 11-21 14:34; the directory kept growing, so it carries no stamp). Their results are in [follow-up results](2026-11-21-rl-run1-followup-results.md).

## Files

Launchers:
- `R/launch_collect.sh`, `R/reseal_collect.sh`, `R/post_collect_checks.sh`;
- `R/train_cell.sh`, `R/run_match.sh`, `R/match3.sh`;
- `R/zq_then_gate.sh`, `R/p0_then_graft1.sh`, `R/graft_look.sh`.

Timestamps (start → exit, −05:00):

| Step | Start | Exit |
|---|---|---|
| S training | 10:08:11 | 10:19:38 |
| match 3 | 10:20:18 | 10:33:01 |
| Zq training | 10:33:46 | 10:48:05 |
| gate | 10:48:05 | 11:10:49 |
| P0 training | 11:11:14 | 11:22:07 |
| look 1 | 11:22:07 | 11:51:14 |
| look 2 | 12:05:18 | 12:33:27 |
| look 3 | 12:33:49 | 13:01:32 |

These come from the `*.started` / `*.exit` files: S `ce1fd56e` / `3673f4b3`, match3 `4382cec4` / `026133db`, Zq
`e371a686` / `71d216a1`, gate `f32b097f` / `57240c85`, P0 `6b96d8a7` / `e413c5a8`, graft_1 `229c5b0c` / `e0d411c2`,
graft_2 `e6cb324e` / `11a79c3f`, graft_3 `e7b2114f` / `85b17afc`.
