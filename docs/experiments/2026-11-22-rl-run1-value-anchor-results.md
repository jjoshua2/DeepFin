# 2026-11-22: RL run 1 value-anchor screen: results

These are the results of the screen registered in
[the value-anchor screen record](2026-11-21-rl-run1-value-anchor-screen.md). Every verdict is
read by that record's rules, in the same session as the readout. The screen trained cells on
the pilot's frozen collection and played two matches. It adopts nothing and changes nothing live.

**Summary.**

| Target | Stage A (offline veto) | Stage B match vs the SL base | Verdict by the registered rule |
|---|---|---|---|
| S (value on search WDL only) | λ 4.1 VETOED, **λ 7.0 PASS**, λ 13.4 INERT | S + λ 7.0: **+5.8 [−0.5, +12.2]** | **"adoptable"**; not "damage" |
| Zq (half outcome, half search) | **λ 11.3 PASS**, λ 18.1 INERT, λ 33.4 INERT | Zq + λ 11.3: **+3.6 [−2.6, +9.8]** | **"adoptable"**; not "damage" |

- For both targets, "the anchor is NOT the fix" does not hold, by criteria (a) and (b).
- Neither result is a "direction" reading: r reads "none" for both matchable S cells.
- The record says plainly: "An adoptable reading does not by itself show that the anchor's
  direction, rather than smaller movement, is responsible."

**Bank.** `V` = `~/chess-artifacts/operations/rl-run1-value-anchor` (in the evidence log's prefix
table). Every number below was read from the files named, or computed from them where marked
"derived".

| File | sha256 prefix | Contents |
|---|---|---|
| `V/run_queue2.sh` | `b372796e` | the queue that ran match 8, then this record's command blocks |
| `V/queue2.log` | `72e07964` | stage markers: `anchor_main` 22:41:32, `anchor_cells` 01:17:36, `anchor_stage_b` 01:54:26, `done` 04:36:58 |
| `V/queue2.exit` | `973d31c2` | `0 2026-11-22T04:36:58-05:00` |
| `V/offline/control/CONTROL_PASS` | `8ab051b4` | step 0: "adapted instruments reproduce the banked outputs exactly" |
| `V/stage_a/train_check.py` | `5a2fa045` | the registered field checks and gate (i), matching the record's sha |
| `V/stage_a/stage_a_decide.py` | `91d3ef26` | the registered Stage A rules (v3), matching the record's sha |
| `V/train_check.json` | `d69638c1` | per-cell fields, gate (i) |
| `V/offline/cells/holdout_analysis.json` | `d5874f74` | hold-out KL to the base, share of gap closed |
| `V/offline/cells/sibling_order.json` | `761e669b` | clear-pair paired differences vs the base |
| `V/offline/cells/stage_a_decision.json` | `ea8ab67b` | statuses, r, readings, `stage_b` selection |
| `V/INCIDENT_gpu_overlap_20261121.md` | `061f763d` | the operator's note on a possible GPU overlap during training |

## Training: ten runs, gate (i)

The launch precondition ran inside the queue: the anchor-gradient test and `test_simple_rl_accum_steps.py` at the
pin, with `|| exit 1`. The queue went on to exit 0.

- **All ten runs exited 0** at the first attempt (`V/arm_*.exit`). There is no `.crash1` directory and no `VOID` or
  `TARGET_*_VOID` marker.
- **Every cell has `ok: true`**, with no field failures (`V/train_check.json` [`d69638c1`]):
  - 1,412 updates and 722,742 drawn rows;
  - `presented_sha256` = `63e227ac…` in all ten, the pilot's batch plan;
  - realised `w_wdl` as designed: 1.0 for the anchored and truncated cells, 0.1 / 0.03 for the dose cells, 0.0 for
    P0-anchor;
  - realised search share: 1.0 for S, 0.5 for Zq, 0.0 for P0;
  - for anchored cells, `wdl_coef` = λ, `policy_coef` = 0, `reference_forwards` = 1,412 and `rows` = 722,742;
  - for the truncated-S run, snapshots at 353 / 635 / 988.
- **Gate (i) passes for both targets.** The training-time `mean_wdl_kl` falls strictly with λ:

  | Target | λ | `mean_wdl_kl` |
  |---|---|---|
  | S | 4.1 / 7.0 / 13.4 | 0.001776 / 0.000995 / 0.000505 |
  | Zq | 11.3 / 18.1 / 33.4 | 0.000686 / 0.000420 / 0.000253 |
  | P0-anchor | 7.0 | 0.000343 |

- **Walls:** 428–563 s per run, 4,764 s in total (1.32 GPU-h, derived). The registered estimate was ≈ 1.2.
  `wall_seconds` of `arm_Zq_k11.3` and `arm_Zq_k18.1` is not comparable with the others' (see the incident below).

## Stage A: the offline veto

**Step 0 passed** (`CONTROL_PASS`). Also, `run_to_run_noise_KL` is 0.0 (`stage_a_decision.json`): the truncated-S run's
final model reproduces the pilot's S.
- Its hold-out statistics equal S's exactly, and `r_minus_S_r` is [0, 0, 0].
- **Its weights are bit-identical to the banked S** (derived: all 494 tensors are equal, loaded on CPU).
- The `.pt` files' sha256 differ: `4bdf686c…` vs S's `66ceb271…`.

**Rules, from the record:**
- **VETOED** iff the clear-pair difference vs the base has UB95 < 0.
- **INERT** iff `share_of_gap_closed` has LB95 ≤ 0, or a point estimate below 0.1 × the unanchored cell's. The
  thresholds are S 0.0293 and Zq 0.0346.
- **NOT ANCHORED** iff gate (ii) fails.
- Stage B takes the smallest-λ cell that is none of these.

Columns: clear-pair difference vs the base in percentage points of accuracy, 95% CI; share of the top-decile gap
closed; hold-out KL(base ‖ cell). All three come from `V/offline/cells/stage_a_decision.json` [`ea8ab67b`].

| Cell | Clear-pair diff, pt | Share of gap closed | KL(base ‖ cell) | Status |
|---|---|---|---|---|
| S, unanchored (pilot) | −0.96 [−1.29, −0.64] | 0.293 | 0.0167 | reference |
| S λ 4.1 | −0.32 [−0.54, −0.11] | 0.0645 | 0.00194 | **VETOED** |
| **S λ 7.0** | −0.17 [−0.37, +0.03] | 0.0442 | 0.00108 [0.00102, 0.00114] | **PASS → Stage B** |
| S λ 13.4 | −0.12 [−0.30, +0.05] | 0.0265 | 0.00054 | **INERT** (0.0265 < 0.0293) |
| Zq, unanchored (pilot) | −0.07 [−0.32, +0.17] | 0.346 | 0.0275 | reference |
| **Zq λ 11.3** | +0.06 [−0.10, +0.22] | 0.0363 [0.0340, 0.0384] | 0.00073 | **PASS → Stage B** |
| Zq λ 18.1 | +0.08 [−0.07, +0.24] | 0.0234 | 0.00043 | **INERT** |
| Zq λ 33.4 | −0.01 [−0.14, +0.12] | 0.0158 | 0.00025 | **INERT** |

- **Every anchored cell passes gate (ii)** (`anchored: true`): each KL's UB95 lies far below the unanchored lower
  bound (S 0.01573, Zq 0.02642), and each point estimate falls with λ.
- **Zq λ 11.3 passes the inert rule narrowly.** Its point estimate 0.0363 clears the 0.0346 threshold by 0.0017, and
  its lower bound, 0.0340, lies below the threshold. The rule reads the point estimate, so the cell passes.
- The S veto has power and fired at λ 4.1. The record says the veto has no power for Zq; for the Zq cells, Stage A
  decided only inert or not anchored.
- `stage_b` = {S: `S_k7.0`, Zq: `Zq_k11.3`}.

**Descriptive cells** (never matched; compared with S descriptively):

| Cell | Clear-pair diff, pt | Share closed | KL(base ‖ cell) |
|---|---|---|---|
| S dose, `--w-wdl` 0.1 | −0.72 [−1.01, −0.43] | 0.286 | 0.0169 |
| S dose, `--w-wdl` 0.03 | −0.62 [−0.90, −0.35] | 0.285 | 0.0171 |
| truncated S at 353 updates | −0.48 [−0.72, −0.23] | 0.270 | 0.0156 |
| truncated S at 635 | −0.66 [−0.92, −0.41] | 0.293 | 0.0169 |
| truncated S at 988 | −0.67 [−0.94, −0.42] | 0.298 | 0.0171 |
| truncated S, final (= S) | −0.96 [−1.29, −0.64] | 0.293 | 0.0167 |
| P0 + anchor λ 7.0 | +0.07 [−0.08, +0.23] | 0.0039 (inert) | 0.00036 (P0 unanchored 0.0072) |

- **Lowering the value-loss weight does not brake the head.** At `--w-wdl` 0.1 and 0.03 the head moves as far as
  unanchored S (KL ≈ 0.017, share ≈ 0.29). This is the registered scale-invariance prediction.
- **The truncated-S snapshots did not provide a "moved less" control.**
  - S had already closed 0.270 of the gap by update 353. The min model predicted 0.25 of the full displacement there,
    ≈ 0.073 (derived).
  - So every snapshot moved 4–7 times as far as the anchored S cells (0.044–0.065).

**r and the direction readings, as registered.**
- r = paired clear-pair difference / share of gap closed; S's r is −0.0327.
- For each non-inert anchored S cell, the matched snapshot is the one whose measured share is nearest its own. For both
  S λ 4.1 and S λ 7.0 that is `S_trunc_353`, at share 0.270 against the cells' 0.0645 and 0.0442.

| Cell | r | r − r_S [95%] | r − r_snapshot [95%] | `direction_reading` |
|---|---|---|---|---|
| S λ 4.1 | −0.0501 | −0.0173 [−0.0438, +0.0088] | −0.0324 [−0.0599, −0.0054] | **none** |
| S λ 7.0 | −0.0394 | −0.0067 [−0.0450, +0.0315] | −0.0217 [−0.0608, +0.0175] | **none** |

- **What "none" licenses:** by the record's definition, "not better than S's r". The anchored cells' ranking damage
  per unit of movement toward the search value is not shown to be smaller than unanchored S's. Their point estimates
  are more negative.
- **What it does not license:**
  - any claim that the anchor's direction is responsible for S + λ 7.0's result;
  - the "distance" reading: that also needs r − r_S's CI above 0, which it is not;
  - a claim that the anchor's direction is harmful: the CIs of r − r_S include 0.
- The snapshot match is also weak by construction: the snapshots moved several times farther than the cells, as noted
  above.
- r is "NOT COMPUTABLE" for the Zq cells and P0-anchor, and "NOT COMPUTED (inert)" for S λ 13.4.
- The dose and snapshot cells' r − r_S CIs lie above 0, for example `--w-wdl` 0.1 +0.0075 [+0.0017, +0.0137]. These
  are descriptive, with no reading.

The report-only items the record lists are in `V/offline/cells/` and are not used here: `KL_share_removed`, the (T, b)
fits, value regret and outcome distance.

## Stage B: matches by match 5's rules

**Design.**
- Plain seats, seat-balanced, 6,400 games (3,200 per run, 1,600 openings).
- `matched_sims` 64 v 64, `--max-plies 600 --on-max-plies abort`.
- PIN `0c795fb7a` in every result row.
- The launcher's sha256 `7c8fd065…` was checked by the queue.
- The result rows' `candidate_sha256` / `reference_sha256` are the arm's (`16d486d3…` for S λ 7.0, `42a407d5…` for
  Zq λ 11.3) and the SL base's (`a39f900c…`).

**Rules:**
- "damage" iff UB95 < 0;
- "adoptable" iff LB95 > −10;
- "unread" otherwise.

The tie-break applies only below SE ≈ 2.55.

| Match | Seed | Pooled Elo [95%] | Elo SE | Arm as candidate (A) | Arm as reference (B) | Seat offset | Arm W/D/L (A; B) | Unresolved, longest |
|---|---|---|---|---|---|---|---|---|
| S + λ 7.0 vs base | 2026112181 | **+5.81 [−0.53, +12.16]** | 3.24 | +6.9 [−1.6, +15.5] | +4.7 [−4.2, +13.6] | +1.1 [−4.9, +7.1] | 939/1,386/875; 957/1,329/914 | 0; 285 plies |
| Zq + λ 11.3 vs base | 2026112182 | **+3.58 [−2.62, +9.79]** | 3.17 | +7.6 [−1.1, +16.3] | −0.4 [−8.8, +8.0] | +4.0 [−1.8, +9.9] | 954/1,362/884; 912/1,372/916 | 0; 340 plies |

**Readings:**
- **S + λ 7.0: "adoptable"** (LB95 −0.53 > −10). Not "damage" (UB95 +12.16 ≥ 0).
- **Zq + λ 11.3: "adoptable"** (LB95 −2.62 > −10). Not "damage" (UB95 +9.79 ≥ 0).
- No tie-break: the SEs are 3.24 and 3.17.
- Zq's seat offset, +4.0, has an interval that includes 0.

**Sources:**
- S: `V/arenas/anchor_S_k7.0/pool.json` [`0722d72b`], `results.jsonl` [`e9fa81be`],
  `anchor_S_k7.0_1_A.games.jsonl` [`2484a27c`], `anchor_S_k7.0_1_B.games.jsonl` [`ffbde0e0`].
- Zq: `V/arenas/anchor_Zq_k11.3/pool.json` [`2c890633`], `results.jsonl` [`e094dd91`],
  `anchor_Zq_k11.3_1_A.games.jsonl` [`3096550e`], `anchor_Zq_k11.3_1_B.games.jsonl` [`4fb72a61`].
- W/D/L are derived from the game files. All four runs exited 0.
- Walls: 2,449 + 2,345 s and 2,442 + 2,476 s, 2.70 GPU-h in total.

**What each reading licenses** (the record's table, verbatim):
- **adoptable, S target:** "NOMINATES that value recipe (S plus λ) for a 2-generation run together with the policy,
  under its own record. Nothing is adopted here."
- **adoptable, Zq target:** "NOMINATES the anchored Zq recipe only TOGETHER with a "Zq + (T, b)" alternative in the
  nomination record. It supports no claim that "the anchor fixes content" for Zq: match 7 showed Zq's loss is largely
  shape, and part of an anchored Zq cell's gain may be the anchor removing that shape error."
- And: "An adoptable reading does not by itself show that the anchor's direction, rather than smaller movement, is
  responsible. That claim needs r, as registered above." Here r reads "none".

**"The anchor is NOT the fix" for a target:** (a) every anchored cell is vetoed, inert or not anchored, or the target
is void; OR (b) its matched cell reads "damage".
- S: (a) fails, because λ 7.0 passed. (b) fails, because the match is not "damage". **Not met.**
- Zq: (a) fails, because λ 11.3 passed. (b) fails. **Not met.**
- This removes one way the anchor could have been ruled out. It does not show that the anchor is the fix.

**Pre-declared contrast: report-only and "nearly uninformative" (the record's words).** It is the anchored estimate
minus the unanchored banked result, with SEs treating the matches as independent (derived):

| Target | Anchored | Unanchored (banked) | Difference | SE of the difference | 95% |
|---|---|---|---|---|---|
| S | +5.81 (SE 3.24) | −20.0 (SE 8.53; `R/arenas/s_vs_base/pool.json` [`db8beb4f`]) | **+25.8** | 9.12 (registered ≈ 9.1) | [+7.9, +43.7] |
| Zq | +3.58 (SE 3.17) | −10.0 (SE 6.74; `R/arenas/gate/pool.json` [`c3e4f5c7`]) | **+13.6** | 7.44 (registered ≈ 7.5) | [−1.0, +28.2] |

The matches use separate seeds and different openings; the unanchored S figure rests on 800 games. No head-to-head of
anchored vs unanchored was run, by design.

## Confounds and incidents

- **Possible GPU overlap during training, 23:51:14–23:58:18** (`V/INCIDENT_gpu_overlap_20261121.md` [`061f763d`]).
  - What: an agent's launcher guard test fell through to the real arena launcher five times and may have started 4-game
    smoke arenas.
  - Exposure: `arm_Zq_k11.3` (its last ≈ 77 s) and `arm_Zq_k18.1` (its start). No samples were taken, so whether any
    test arena reached the GPU is unknown.
  - Why the numbers stand: training runs a fixed 1,412 updates over a fixed batch plan, and no clock enters any
    decision. Both cells' `presented_sha256` and fields match. A grep of both logs for OOM, Traceback, CUDA error and
    RuntimeError finds nothing.
  - Only those two cells' `wall_seconds` is not comparable.
  - The first incident that night, beside match 8's seat A, is in the
    [follow-up results](2026-11-21-rl-run1-followup-results.md).
- **Multiplicity.** These are readings 8 and 9 of RL run 1's registered nomination tests, with no correction
  registered. The nine (evidence log, Table 1):
  1. the pilot gate, Zq vs base;
  2. pilot match 3, S vs base;
  3. pilot match 4, the graft;
  4. match 5, P0 as a single net;
  5. match 7, recalibrated Zq vs Zq;
  6. match 6, S's value on the base policy (two pools, one reading);
  7. match 8, recalibrated Zq vs base;
  8. S + λ 7.0 vs base;
  9. Zq + λ 11.3 vs base.

  The record states each match nominates falsely with probability ≈ 2.5% at the −10 boundary, ≈ 5% across these two.
- **Three nominations now stand, each from one generation, and they are never ranked by point estimate:**
  - recalibrated Zq, +9.6 [+3.3, +16.0] (match 8);
  - S + anchor λ 7.0, +5.8 [−0.5, +12.2];
  - Zq + anchor λ 11.3, +3.6 [−2.6, +9.8].

  They use different seeds and openings, and their intervals overlap.
- **One data-affecting change per cell.** Each anchored cell differs from its unanchored pilot cell only in `--kl-wdl`
  / `--kl-reference`, proved at the consumer by the identical `presented_sha256`.
- **Policy-induced drift.** P0 + anchor shows that the anchor also holds the value against trunk drift from the policy
  loss: KL 0.00036 against P0's 0.0072. No subtraction is made.

## What this means for the owner's questions (interpretation, not a verdict)

- **"KL penalty from last known best" on the value head now has a played result.**
  - With the anchor at the selected strengths, S-target value training no longer reads damage against the base:
    +5.8 [−0.5, +12.2], against the unanchored −20.0.
  - Zq-target training reads +3.6 [−2.6, +9.8], against −10.0.
  - Both are one generation, and neither is shown to work through the anchor's direction rather than through moving
    less.
- **"Step less unless sure":** the value-loss weight is not the lever. At 0.1 and 0.03 the head moved as far as at 1.0.
  The anchor is the lever: it changed how far the head moved, by a factor of about 15 in KL at S λ 7.0.

**Open owner decisions; nothing is recommended or launched here:**
- which nominated value recipe or recipes a second generation uses: recalibrated Zq, S + anchor, Zq + anchor, or a
  comparison among them;
- the generation-2 record as registered in PR #1491 freezes recalibrated Zq, so any change of recipe is a change to
  that record.

## Items this closes

- The value-anchor screen's Stage A and Stage B are read; its status is set to completed.
- The owner's 11-19 request for a KL penalty on the value: run and read. The policy KL (`--kl-policy`) was not part of
  this screen and remains unmeasured.
