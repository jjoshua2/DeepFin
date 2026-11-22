# RL run 1 follow-up matches 5, 7, 6b, 6a and 8: results (2026-11-21)

These are the results of the matches registered in
[the follow-up match record](2026-11-21-rl-run1-match5-p0-single-net.md) (PR #1473), read by its rules in the same
session as the readout, plus match 8 from [its own record](2026-11-21-rl-run1-match8-zq-recal-vs-base.md). They
follow [the pilot results](2026-11-21-rl-run1-pilot-results.md). All five adopt nothing and change nothing live.

**Summary.**

| Match | Result | Verdict by the registered rule |
|---|---|---|
| 5: P0 as a single net vs the SL base | +2.2 [−4.2, +8.6] | **"single-net policy-only step is adoptable"** |
| 7: Zq recalibrated vs Zq | +21.4 [+15.1, +27.6] | **"the loss is shape"** |
| 6b: S's raw value on the base policy vs the base | −21.4 [−27.7, −15.2] | reads damage; step 1 is passed |
| 6a: S's recalibrated value on the base policy vs the base | −16.1 [−22.3, −9.9] | — |
| Match 6, decision order | R = 6a − 6b = +5.34 [−3.48, +14.15] | **"unread"** (step 5: 6a reads damage, but R's UB95 ≥ +10, so partial recovery is not excluded) |
| 8: Zq recalibrated vs the SL base ([own record](2026-11-21-rl-run1-match8-zq-recal-vs-base.md)) | +9.6 [+3.3, +16.0] | **"adoptable" and "improves on the base"** ([below](#match-8-recalibrated-zq-vs-the-sl-base-seed-2026112191)) |

Each match was 6,400 games (3,200 per run, 1,600 openings), with 0 unresolved games.

**Bank.** `F` = `~/chess-artifacts/operations/rl-run1-followup`. Every file is cited with its sha256 prefix, and
every number was read from the bank files named or computed from them where marked "derived".

| File | sha256 prefix | Contents |
|---|---|---|
| `F/run_all.sh` | `05dca140` | the record's three command blocks, line for line, plus marker lines |
| `F/run_followup_match.sh` | `7c8fd065` | byte-identical to the record's launcher block |
| `F/check_seat.py` | `8f4ce9eb` | byte-identical to the record's block |
| `F/stage.log` | `8f8a233f` | match 5 14:34:46, match 7 15:52:52, match 6 17:16:50, done 21:13:57, all −05:00 |
| `F/chain.exit` | `73eff0c6` | "0 2026-11-21T21:13:57-05:00" |
| `F/run_all.log` | `6549cfa5` | 12 "seat readback OK" lines |

The 12 readback lines are 4 for match 7 and 8 for match 6, a smoke and a real run each, both runs. Each line names the
hybrid role, the value-from sha256 (`66ceb271beee` = S, or `-`) and the recal (T, b).

**Common settings** (every result row):
- `git_sha` = `0c795fb7a`, the pilot's `$PIN`;
- 64 sims, `matched_sims`, CUDA graphs, `--compile off`;
- max plies 600, abort;
- pools at z 1.96, clustered by opening.

## Match 5: P0 as a single net vs the SL base (seed 2026112151)

Rules (record, verbatim):
- **"damage"** iff UB95 < 0;
- **"single-net policy-only step is adoptable"** iff LB95 > −10;
- otherwise **"unread"**.
- Tie-break only if the measured clustered SE is below ≈ 2.55 Elo.

| Quantity | Value |
|---|---|
| **Pooled** | **+2.2 [−4.2, +8.6]** (excess +0.0031, SE 0.0047; Elo SE 3.26) |
| P0 as candidate (A) | +3.9 [−4.7, +12.5] |
| P0 as reference (B) | +0.4 [−8.4, +9.3] |
| Seat offset | +1.7 [−4.3, +7.7] |
| Clusters | 1,600, 0 repeated within a run; A/B correlation +0.063 |
| P0's W/D/L | A 949/1,338/913, B 955/1,294/951 (derived) |

- **Verdict: "single-net policy-only step is adoptable"** (LB95 −4.2 > −10).
- Not "damage": UB95 +8.6 ≥ 0.
- The tie-break does not apply: the measured SE is 3.26, above 2.55.

**Secondary (report-only).** The cost of P0's value drift is match 5 − graft = +2.17 − 7.66 = **−5.5**, SE ≈
√(3.26² + 3.86²) ≈ 5.1 (derived; the graft's 3.86 is its look-3 SE in Elo by the delta method). That SE treats the
matches as independent.

**Licenses** (record's table): "NOMINATES a second policy-only generation for the owner, under its own record,
confirmed on fresh games. Nothing more." That generation needs its own record and the owner's go-ahead.

Sources: `F/arenas/p0_single/pool.json` [`22fc7f97`], `pool.log` [`6a70d14e`], `results.jsonl` [`efdd98f0`],
`p0_single_1_A.games.jsonl` [`31828e55`], `p0_single_1_B.games.jsonl` [`cb8a7eb3`].

## Match 7: Zq recalibrated vs Zq (seed 2026112171)

Rule (record, verbatim): **"the loss is shape"** iff LB95 > 0, with the arm being the recalibrated seat; otherwise
**"unread"**.

| Quantity | Value |
|---|---|
| **Pooled** (arm = Zq + recal (0.7036189636786957, −0.22711530923843357)) | **+21.4 [+15.1, +27.6]** (excess +0.0307, SE 0.0046; Elo SE 3.18) |
| Arm as candidate (A) | +25.7 [+17.0, +34.4] |
| Arm as reference (B) | +17.1 [+8.5, +25.7] |
| Seat offset | +4.3 [−1.7, +10.3] |
| Clusters | 1,600; A/B correlation +0.038 |
| Arm's W/D/L | A 1,068/1,300/832, B 1,012/1,333/855 (derived) |

**Seat readback.**
- The result rows carry `candidate_value` (run A) and `reference_value` (run B) = {value_from: none, recal: [0.7036…, −0.2271…]}.
- The smoke and both real readbacks printed "seat readback OK".

- **Verdict: "the loss is shape"** (LB95 +15.1 > 0).
- **Report-only:** the recovered fraction is 21.4 / 10.0 = 2.14. The gate's interval puts the denominator near 0, so
  it is never read.

**Licenses** (record): "'shape' NOMINATES a recalibrated-Zq vs base adoption match, as a later and separate
registration." That registration is [match 8](2026-11-21-rl-run1-match8-zq-recal-vs-base.md). Match 7 says nothing
about Zq vs the base.

Sources: `F/arenas/zq_recal_h2h/pool.json` [`e1945cbd`], `pool.log` [`19789768`], `results.jsonl` [`914bee93`],
`zq_recal_h2h_1_A.games.jsonl` [`7879ff38`], `zq_recal_h2h_1_B.games.jsonl` [`3733414d`].

## Match 6: S's value on the base policy, raw (6b) and recalibrated (6a) (seed 2026112161, the same 1,600 openings)

In both pools the arm is the hybrid seat: the SL base's policy with S's value (`--*-value-from` S, sha256
`66ceb271…`); 6a adds `--*-value-recal 0.853159495428655,-0.12023510932922338`. The result rows carry these in
`candidate_value` (A) and `reference_value` (B).

| Quantity | 6b: raw graft | 6a: recal graft |
|---|---|---|
| **Pooled** | **−21.4 [−27.7, −15.2]** (excess −0.0308, SE 0.0046) | **−16.1 [−22.3, −9.9]** (excess −0.0231, SE 0.0046) |
| Arm as candidate (A) | −21.2 [−30.1, −12.4] | −14.4 [−23.1, −5.8] |
| Arm as reference (B) | −21.6 [−30.5, −12.8] | −17.7 [−26.5, −9.0] |
| Seat offset | +0.2 [−6.1, +6.5] | +1.6 [−4.4, +7.7] |
| A/B correlation | −0.014 | +0.029 |
| Arm's W/D/L (derived) | A 855/1,295/1,050, B 842/1,317/1,041 | A 876/1,315/1,009, B 865/1,307/1,028 |

**The recovery estimator, as registered.** R = full 6a estimate − full 6b estimate (Elo, each from its own
`pool.json`), with SE_R = √(SE_a² + SE_b²).
- **SE_a and SE_b in Elo (derived).** Each is the pool's score-scale `se` times the slope of the Elo map at the pool's
  own estimate: dElo/dx = 400 / (ln 10 · (0.5 + x)(0.5 − x)). That gives SE_a = 3.18 and SE_b = 3.18 (3.1803 and
  3.1789).
- **Cross-check.** Half the width of each pool's `elo_ci95`, divided by 1.96, gives the same values to two decimals
  (3.1807, 3.1793).
- **R = −16.08 − (−21.42) = +5.34, SE_R = 4.50, 95% [−3.48, +14.15]** (derived).

**Decision order** (record, verbatim conditions):
1. "6b's UB95 ≥ 0 → unread": no. 6b's UB95 is −15.2 < 0, so 6b reads damage and steps 2–5 apply.
2. "partial recovery": needs 6a's UB95 < 0 AND R's UB95 < +10 AND R's LB95 > 0 AND 6a's LB95 > −10. **No** (R's UB95 +14.15; R's LB95 −3.48; 6a's LB95 −22.3).
3. **"content"** iff 6a's UB95 < 0 AND R's UB95 < +10: 6a's UB95 −9.9 < 0 holds, but R's UB95 +14.15 ≥ +10. **No.**
4. **"shape"** iff R's LB95 > 0 AND 6a's LB95 > −10: R's LB95 −3.48 ≤ 0. **No.**
5. **Otherwise "unread".** The cell is "6a reads damage, but R's UB95 ≥ +10: partial recovery is not excluded."

- **Verdict: "unread".** It licenses nothing (record's table).
- **Planned vs observed.** The record's estimate, "not a rule input", was ≈ −28 for 6b; it read −21.4.

**Report-only (pre-allowed by the registering record as "a report-only secondary"), not a verdict: R paired by opening.** 6a and 6b share seed 2026112161 and all 1,600
openings, so R can also be estimated as the mean over openings of (6a's cluster value − 6b's cluster value), from the
game logs with the pool's own cluster definition.
- The clusters correlate only weakly across 6a and 6b (r = 0.026), so pairing barely tightens the interval:
  **R = +5.34, clustered SE 4.44, 95% [−3.36, +14.03]**. On the score scale this is +0.0077 ± 0.0064, z 1.20.
- Read by the registered rule on this interval, it would still be step 5, "unread": the UB95 +14.03 ≥ +10 and the
  LB95 −3.36 ≤ 0. **The verdict above is unchanged.**

**The offline prediction against what happened.** The record quotes the diagnosis's Addendum 1: recalibration does not
repair S's sibling ordering (−0.0096 → −0.0093), "so it should NOT recover S's loss". Recalibration recovered **+5.3
of 6b's −21.4**, with an interval of [−3.5, +14.2] that includes both 0 and +14. The prediction is neither confirmed
nor refuted at this size.

**Observation, not a verdict: 6b alone.** For the first time a played seat isolates S's value head on an unchanged
policy: the SL base's own policy with S's value lost −21.4 [−27.7, −15.2]. S as a whole net lost −20.0 [−36.8, −3.3] in
the pilot (match 3). So S's whole loss can sit in the value head, with no policy change needed. The caveats:
- This is one net and one generation.
- The two matches used different seeds and opening blocks.

Sources:
- 6b: `F/arenas/s_value_raw/pool.json` [`3534eac9`], `pool.log` [`c6ae3300`], `results.jsonl` [`c188a3da`], `s_value_raw_1_A.games.jsonl` [`facedfaf`], `s_value_raw_1_B.games.jsonl` [`d43b3624`].
- 6a: `F/arenas/s_value_recal/pool.json` [`0ff08c98`], `pool.log` [`89a62c5a`], `results.jsonl` [`eac3f27e`], `s_value_recal_1_A.games.jsonl` [`9a1f88ca`], `s_value_recal_1_B.games.jsonl` [`43e580b2`].

## Integrity, cost and smokes

**Unresolved games: 0 in all 25,600.** Every result row has `max_plies_unresolved` 0, `max_plies_aborts_banked` 0,
`resumed_pairs` 0 and `truncated` false, and no `*.unresolved.jsonl` sidecar exists. Each run holds 3,200 games
(derived from the game logs).

**Longest game:** 350 plies (match 5 run B, rules, draw). The longest game in each other match:

| Match | Longest game |
|---|---|
| 7 | 318 plies |
| 6b | 278 plies |
| 6a | 302 plies |

**Measured SE vs planned.** The record planned 3.37 Elo at 6,400 games. The measured Elo SEs are 3.26 (match 5), 3.18
(match 7), 3.18 (6b) and 3.18 (6a).

**Wall and GPU-h** (sum of the two runs' `duration_s`, derived):

| Match | Wall | GPU-h, measured | GPU-h, planned |
|---|---|---|---|
| 5 | 4,670 s | 1.30 | 1.52 |
| 7 | 4,981 s | 1.38 | 1.52 |
| 6b | 7,172 s | 1.99 | 1.84–1.92 |
| 6a | 6,876 s | 1.91 | 1.84–1.92 |
| **Total** | | **6.58** | ≈ 6.7–6.9 |

The whole chain ran from 14:34:46 to 21:13:57.

**Smokes** (4 games per run, same seed and seat arguments, under `F/smoke/arenas/`). Every smoke run exited 0, all at
`0c795fb7a` with 0 unresolved games, and every smoke pool accepted the seat values. Their Elo figures mean nothing at 8
games:

| Smoke | Pool |
|---|---|
| match 7 | `zq_recal_h2h/pool.json` [`45957050`] |
| 6b | `s_value_raw/pool.json` [`117890d3`] |
| 6a | `s_value_recal/pool.json` [`15977898`] |

`F/presmoke_unregistered/` (`README.txt` [`0c20fd86`]) is an unregistered dry run on seed 1, before launch, and has no
reading.

**Confounds.**
- These are post-hoc diagnostics on one pilot generation.
- **Incident during match 8's seat A:** three stray collects shared the GPU for ≈ 126 s, from 21:38:48 to 21:40:54. Wall time enters no decision, and the seat offset is +1.2 [−4.9, +7.4] (see match 8).
- Both recal fits were made after the pilot gate had been read: S at 13:08 and Zq at 13:09, on 11-21
  (`pilot-value-diagnosis/logs/fit_{S,Zq}.log`). The gate finished at 11:10.
- The matches use the same `eval.pgn` openings as the pilot's matches; the overlaps are listed in the registering
  record.

## Match 8: recalibrated Zq vs the SL base (seed 2026112191)

Registered in [its own record](2026-11-21-rl-run1-match8-zq-recal-vs-base.md), which match 7's "shape" licensed. The
arm is Zq's own file with the recalibration (0.7036189636786957, −0.22711530923843357) applied to its WDL output: one
forward, no value-from net. The opponent is the plain SL base.

Rules (match-8 record, verbatim). Read on `result.pooled.elo_ci95`, arm = the recalibrated seat, on the measured SE:

| Reading | Condition |
|---|---|
| damage | UB95 < 0 |
| adoptable | LB95 > −10 |
| unread | neither |
| **improves on the base** (pre-declared, stronger; reported alongside adoptable) | LB95 > 0 |

The tie-break applies only below an SE of ≈ 2.55.

| Quantity | Value |
|---|---|
| **Pooled** | **+9.6 [+3.3, +16.0]** (+9.61 [+3.27, +15.96]; excess +0.0138, SE 0.0047; Elo SE 3.24) |
| Arm as candidate (A) | +10.9 [+2.0, +19.7] |
| Arm as reference (B) | +8.4 [−0.5, +17.2] |
| Seat offset | +1.2 [−4.9, +7.4] |
| Clusters | 1,600, 0 repeated within a run; A/B correlation +0.031 |
| Arm's W/D/L | A 962/1,376/862, B 974/1,329/897 (derived) |
| Games | 6,400, 0 unresolved (longest 308 plies). Both runs exited 0 (21:58:09, 22:41:29). No OOM, Traceback or CUDA error in either run log |

**Seat readback.** The result rows carry `candidate_value` (A) and `reference_value` (B) = {value_from: none, recal:
[0.7036…, −0.2271…]}. Four "seat readback OK" lines (smoke and real, both seats, `value_from=-`, the same recal) are in
`~/chess-artifacts/operations/rl-run1-value-anchor/run_queue2.out` (its first four lines; it was 444 bytes at 22:41 and is a living file, so it carries no stamp). That file belongs to the
value-anchor queue and may grow. The smoke pool is `F/smoke/arenas/zq_recal_vs_base/pool.json` [`ec58ba16`]; both smoke runs exited 0 (21:14:20, 21:14:39).

- **Verdict: "adoptable"** (LB95 +3.3 > −10), **and "improves on the base"** (LB95 +3.3 > 0).
- Not "damage": UB95 +16.0 ≥ 0.
- No tie-break: the SE is 3.24.

**Licenses** (match-8 record): it "NOMINATES 'Zq + per-generation recalibration' as the value recipe for a
2-generation run with the policy, under its own record ... It adopts nothing here." Confirmation is a fresh generation
under its own record.

**Caveats, as plain as the result.**
- **The (T, b) fit is post hoc on this one net.** It was made at 13:09 on 11-21 (`~/chess-artifacts/operations/rl-claude-20261119/pilot-value-diagnosis/logs/fit_Zq.log` [`3a6d157e`], `~/chess-artifacts/operations/rl-claude-20261119/pilot-value-diagnosis/out/recal_Zq.json` [`7e665c5e`]), after the pilot gate had been read. It was fixed before matches 7 and 8 were registered and run, and was not refit.
- **What the fit minimised:** the cross-entropy of Zq's recalibrated WDL, softmax(logits / T + [0, b, 0]), to the SL base net's WDL. That ran on the 39,237 rows of the pilot collect's 416 hold-out games: CE 0.556724 → 0.540235 (fit log). The recalibration target is therefore the base net's value shape, not the game outcomes.
- **What a 2-parameter fit can and cannot do.** softmax(l / T + [0, b, 0]) changes confidence and draw mass. Through the draw bias it can also reorder W − L between positions somewhat. It cannot add information or copy the base head. Pulling Zq's shape toward the base can recover the shape loss that match 7 measured, but any "beats the base" has to come from Zq's own head.
- **One net and one generation.**
- **A family of nomination tests ran that night,** with no multiplicity correction registered: seven registered readings were made that day (pilot matches 1, 3 and 4; follow-up matches 5, 7, 6 and 8, with match 6's one reading resting on two pools), and the value-anchor screens are still to come. Each reading is at its own 95% interval; any "adoptable" or "shape" reading only nominates.

**The Zq chain across the night:**

| Match | Contrast | Games | Elo [95%] | Elo SE |
|---|---|---|---|---|
| pilot gate (match 1) | raw Zq vs base | 1,600 | −10.0 [−23.2, +3.2] | 6.74 |
| match 7 | recal Zq vs raw Zq | 6,400 | +21.4 [+15.1, +27.6] | 3.18 |
| match 8 | recal Zq vs base | 6,400 | +9.6 [+3.3, +16.0] | 3.24 |

The naive sum −10.0 + 21.4 = +11.4 is 1.8 Elo above match 8's +9.6. As independent matches, the difference has an SE of
√(6.74² + 3.18² + 3.24²) ≈ 8.1, so z ≈ 0.22: **the three are mutually consistent within their SEs** (derived). They
use different seeds and opening blocks, and Elo need not be transitive across opponent pairs.

**INCIDENT, during seat A (from the evidence, not a measurement).**
- **What happened.** On 2026-11-21 from 21:38:48 to 21:40:54 −05:00, about 126 s, three stray collect processes from an agent's launcher guard test shared the GPU with match 8's seat A, the arena process PID 2337486.
- **Window.** The first and last samples that list the stray PIDs are in `~/chess-artifacts/operations/rl-run1-root-full-width/stage0/INCIDENT_guardtest_20261121/logs/pids.txt` [`2e17c16c`].
- **Memory.** The peak card memory in `gpu.csv` [`2ba32b23`] is 30,406 MiB of 32,607, at sample 224 of 281. The file carries no timestamps; the coordinator's ≈ 21:40:49 is an estimate from the 500 ms sample index. The lowest sample, 10,971 MiB, is the match alone.
- **Cause.** The guard's busy check used `nvidia-smi --query-compute-apps`, which lists nothing under WSL2.
- **Handling.** The strays were killed by exact PID, and nothing they wrote was sealed. Evidence: `~/chess-artifacts/operations/rl-run1-root-full-width/stage0/INCIDENT_guardtest_20261121/` (`INCIDENT.txt` [`1f4e5fe7`], `logs/p1.log` [`6c1ab296`], `logs/p2.log` [`2788c568`], `logs/p3.log` [`c6b7cbf9`]).
- **Why the result stands without a rerun.** Wall time enters no decision:
  - the arena flags in `run_followup_match.sh` are `--mode matched_sims --sims 64 --max-plies 600 --on-max-plies abort`, with no clock;
  - both result rows have `ms_per_move` null and `max_seconds` null, 64 sims on both sides, `paused_seconds` 0, `timing_independent` true.
- **Why the moves could not change.** The strays were separate processes and could not enter the arena's batches. GPU contention changes timing, not the arena's sim count or batch membership. Any numeric dependence on batch composition is the ordinary nondeterminism present in every run, not a bias toward a seat.
- **Symptom check.** A broken seat would most visibly show as a large seat offset. Seat A read +10.9 and seat B +8.4; the offset is +1.2 with an interval [−4.9, +7.4] that includes 0.
- **In-window diagnostic** (derived, games treated as independent, no rule). Each game row carries `ts` (its end time) and `duration_s`. The 185 seat-A games in flight during the window scored 0.559 for the arm, against 0.513 for the other 3,015. The difference is +0.047 ± 0.028, where ± is ONE SE with games treated as independent (95%: ± 0.055; z ≈ 1.65). Clustering by opening (103 openings, at most 2 games each) gives about the same, SE 0.028 and z ≈ 1.67. **Not significant; not evidence for the arm.**

Sources: `F/arenas/zq_recal_vs_base/pool.json` [`3e407ba8`], `pool.log` [`e5176649`], `results.jsonl` [`3f2b62ad`],
`zq_recal_vs_base_1_A.games.jsonl` [`9e85b212`], `zq_recal_vs_base_1_B.games.jsonl` [`c9b737f7`]. The run walls were
2,597 s and 2,590 s (1.44 GPU-h).

## Pilot items these results close

- **The pilot's "P0's own value head was not played":** match 5 played P0 as a single net (+2.2 [−4.2, +8.6]).
- **The pilot's "match 2 did not run":** match 7 is a new registration, not match 2. It reads "the loss is shape" for
  Zq against Zq, and it changes no pilot verdict.
- **The pilot's "no played seat isolates the value head":** 6b does, for S.
