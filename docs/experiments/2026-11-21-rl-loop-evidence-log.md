# RL-loop evidence log (rolling index)

- **What this is.** One searchable index of every measurement banked for the RL-loop work since the
2026-11-19 takeover, plus the earlier loop runs it builds on. Each row names its bank file and the
record or PR that carries it. It indexes; it does not re-derive. Numbers are copied from the named file.
- **How to use it.** Grep here before commissioning any match, ruler, throughput run or audit. If a row
answers the question, read its bank file instead of re-running. Check "Registered but not yet read" so
in-flight work is not duplicated.
- **How to add a row.** One row per banked measurement, added in the same PR or session that banks it:
date, question, nets with sha prefixes, settings, N, result with interval, status, bank path + file,
record or PR. Mark numbers from unfinished work *interim*.
- **Status vocabulary.** *registered verdict* (pre-committed rule applied) · *diagnostic* (no rule;
decides nothing) · *screen* (selection instrument; needs confirmation) · *interim* (bank still being
written) · *in progress* · *not run* · *withdrawn* (claim retracted; see the last section) ·
*discarded* (bank exists but must not be used: corrupted, interrupted or superseded).

Abbreviations: `RC` = `~/chess-artifacts/operations/rl-claude-20261119`, `WK` =
`~/chess-artifacts/operations/rl-week-one-20261112-v1`. Nets: **SL base** sha256 `a39f900c…`
(`RC/a39.pt`); **P1 c12** `b4025a75…` (policy-only ablation, `RC/p1-policy-only/ckpt/policyonly/c12/learner.pt`);
**F12 c12** `3d3ca6b8…` and **Q50 c12** `bb157a84…` (`WK/fresh-current-policy-A39-{F12,Q50}-cycle-12-learn-results-v1/learner/learner.pt`).
"Default book" = the production 8-move book (147k lines, 16 plies); "EVAL" = `RC/loop-smoke-gpu/openings/eval.pgn`.
Unless a row says otherwise, arenas are `scripts/arena_standard.py --mode matched_sims --search-shape
training --no-gumbel-noise --compile off`, production Syzygy pair, move temperature 0.1, and Elo is the
candidate's (or the pooled arm's) with its 95% interval.

## Table 1 — Matches

| Date | Question | Candidate vs reference | Sims | Mode / temp / noise | Games | Openings + seed | Elo [95% CI] | Status | Bank + file | Record / PR |
|---|---|---|---|---|---:|---|---|---|---|---|
| 11-19 | SL base search gain | SL base vs SL base | 64 v 1 | T0.1, no noise, one orientation | 200 | default book, 20261119 | +848.7 [+690.7, undefined] | diagnostic; **veto-confounded** (see Table 4 I3) | `RC/search-gain/results.jsonl` (`a39_s64_v_s1`) | [takeover D1](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md), [search ceiling](2026-11-20-sl-base-search-ceiling.md), #1438 |
| 11-19 | SL base search gain | SL base vs SL base | 256 v 64 | T0.1, no noise, one orientation | 400 | default book, 20261119 | +272.4 [+243.1, +305.3] | diagnostic | `RC/search-gain/results.jsonl` (`a39_s256_v_s64`) | same |
| 11-19 | SL base search gain | SL base vs SL base | 800 v 64 | T0.1, no noise, one orientation | 200 | default book, 20261119 | +478.0 [+410.2, +582.3] | diagnostic | `RC/search-gain/results.jsonl` (`a39_s800_v_s64`) | same |
| 11-20 | SL base search gain, seat-balanced rerun | SL base (byte copy `a39_hisims_seat.pt`) vs SL base | 128 v 64 | T0.1, no noise | 284 of 800 (134 pairs), orientation A only | EVAL, 2026112070 | descriptive +151.5 [+121.6, +183.7] — not a result | **discarded** (stopped by owner; duplicate of search-gain) | `RC/search-ceiling/INCOMPLETE.json` | [search ceiling](2026-11-20-sl-base-search-ceiling.md) |
| 11-19 | P1: does turning the value loss off remove F12's loss? (deciding) | P1 c12 vs F12 c12 | 64 v 64 | T0.1, no noise | 1,600 | default book, 2026111950 (0/800 pairs overlap F12 starts) | **+42.6 [+30.1, +55.2]** | **registered verdict: value loss implicated** | `RC/p1-policy-only/arena/results.jsonl` (`p1_vs_f12c12`) | [takeover P1](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md), #1419 |
| 11-19 | P1 vs SL base (recorded, not deciding) | P1 c12 vs SL base | 64 v 64 | T0.1, no noise, one orientation | 800 | default book, 2026111950 | −10.9 [−27.7, +5.9] | diagnostic | same file (`p1_vs_sl`) | same |
| 11-19 | F12 c12 vs SL base on P1's seed | F12 c12 vs SL base | 64 v 64 | T0.1, no noise | 278 of 800 | default book, 2026111950 | not read | **discarded** (reboot; NUL-filled tail) | `RC/p1-policy-only/arena/f12c12_vs_sl.games.jsonl` | takeover record |
| 11-19 | D4 head swap: where is Q50's loss? | Q50 c12 full vs SL base | 64 v 64 | T0.1, no noise | 800 (21 pairs replayed on resume) | default book, 20261119 | −63.2 [−81.1, −45.7] | diagnostic | `RC/headswap/results/q50_A_full.jsonl`; `table.md` | takeover D4, #1419 |
| 11-19 | D4 | hybrid Q50 policy + base value vs SL base | 64 v 64 | same | 800 | same | +4.3 [−12.5, +21.2] | diagnostic | `.../q50_B_Pq50_Vbase.jsonl` | same |
| 11-19 | D4 | hybrid base policy + Q50 value vs SL base | 64 v 64 | same | 800 | same | −55.2 [−73.6, −37.1] | diagnostic | `.../q50_C_Pbase_Vq50.jsonl` | same |
| 11-19 | D4 | F12 c12 full vs SL base | 64 v 64 | same | 800 (22 pairs replayed) | same | −34.9 [−53.1, −16.8] | diagnostic | `.../f12_A_full.jsonl` | same |
| 11-20 | D4 round 2 | hybrid F12 policy + base value vs SL base | 64 v 64 | same | 800 (rerun whole after a NUL tail) | same | −15.7 [−33.8, +2.5] | diagnostic | `.../f12_B_Pf12_Vbase.jsonl` | [D4 round 2](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md#d4-round-2--seat-control-f12-swap-arms-recalibrated-value-2026-11-20), #1425 |
| 11-20 | D4 round 2 | hybrid base policy + F12 value vs SL base | 64 v 64 | same | 800 (1 pair replayed) | same | −18.3 [−36.2, −0.4] | diagnostic | `.../f12_C_Pbase_Vf12.jsonl` | same |
| 11-19 | Seat control D | hybrid(base, base) vs SL base; plain base vs base reproduced it game for game | 64 v 64 | same | 400 | default book, 20261119 | +22.6 [+0.2, +45.3] | diagnostic (read as RNG noise, see next rows) | `.../D_hybrid_bb_400.jsonl`, `D_plain_bb_400.jsonl` | same |
| 11-20 | Seat control, second seed | hybrid(base, base) vs SL base | 64 v 64 | same | 400 | default book, 20261120 | −3.5 [−28.4, +21.4] | diagnostic | `.../D_hybrid_bb_s20_400.jsonl` | same |
| 11-20 | Seat control, same openings new move RNG | SL base vs SL base | 64 v 64 | same; move RNG reseeded 20261121 | 400 | D's openings | +3.5 [−19.6, +26.6] | diagnostic | `.../D_plain_bb_ps21_400.jsonl` | same |
| 11-20 | Seat control, seats swapped | plain SL base vs hybrid(base, base) | 64 v 64 | same | 800 | default book, 20261119 | +4.8 [−12.2, +21.8] | diagnostic | `.../S_swap_bb_800.jsonl` | same |
| 11-20 | **Seat offset, pooled** | all same-net samples above | 64 v 64 | — | 998 pairs | 599 openings | **+6.4 [−4.2, +17.1]** (pair-clustered); +6.4 [−5.0, +17.8] opening-clustered; without D +2.4 [−9.6, +14.4] | diagnostic: no seat offset detected | `RC/headswap/analysis_v2.json`, `analysis_v2.log` | same |
| 11-19 | Harness mutant controls | win/loss swapped; policy columns permuted | 64 v 64 | same | 40 each | default book | 0/40 each | control passed | `.../mut_swapWL.jsonl`, `mut_permPol.jsonl` | takeover D4 |
| 11-20 | Recalibration identity control | T=1, b=0 recal vs unrecalibrated base-policy + Q50-value | 64 v 64 | same | 40 each | same | identical games (40/40), −61.4 both | control passed | `.../idR_recalIdent_40.jsonl`, `idR_plainC_40.jsonl` | D4 round 2 |
| 11-20 | R1: recalibrated Q50 value (base policy) | hybrid, value `logits/0.5445 + [0, −0.6383, 0]` vs SL base | 64 v 64 | same | 800 | default book, 20261119 | −8.2 [−26.1, +9.6]; paired gain over q50_C +46.7 [+22.1, +71.9]; share recovered 0.85 [0.50, 1.18] | diagnostic (not pre-registered) | `.../R1_Pbase_Vq50_recalBS.jsonl`, `analysis_v2.json` | D4 round 2, #1425 |
| 11-20 | R2: recalibrated F12 value (base policy) | T 0.6353, b −0.5878 | 64 v 64 | same | 800 | same | −12.2 [−29.7, +5.3]; paired +6.1 [−18.9, +31.1] | diagnostic | `.../R2_Pbase_Vf12_recalBS.jsonl` | same |
| 11-20 | R3: full Q50, recalibrated value | | 64 v 64 | same | 800 | same | −14.3 [−32.6, +3.9]; paired +48.5 [+25.3, +72.2]; share 0.77 [0.47, 1.06] | diagnostic | `.../R3_Pq50_Vq50_recalBS.jsonl` | same |
| 11-20 | R4: full F12, recalibrated value | | 64 v 64 | same | 800 | same | −0.9 [−19.0, +17.2]; paired +34.0 [+8.4, +60.0]; share 0.97 [0.37, 1.56] | diagnostic | `.../R4_Pf12_Vf12_recalBS.jsonl` | same |
| 11-20 | Loop nets vs SL base at 1 sim | P1 c12 vs SL base | 1 v 1 | T0.1, no noise, seat-balanced (A+B) | 1,600 (800 + 800) | EVAL, 2026112050 (400 pairs) | +240.8 [+218.3, +265.2] | diagnostic; **veto-confounded** | `RC/raw-net-1sim/results.jsonl` (`P1c12_cand`, `_ref`); pooled `RC/plan-review-1439/outputs/pool_all.txt` | [RL run 1 plan](2026-11-20-rl-run1-multi-generation-plan.md) Stage 0 |
| 11-20 | same | F12 c12 vs SL base | 1 v 1 | same | 1,600 | same | +211.7 [+190.3, +234.7] | diagnostic; veto-confounded | same (`F12c12_*`) | same |
| 11-20 | same | Q50 c12 vs SL base | 1 v 1 | same | 1,600 | same | +128.7 [+109.4, +148.8]; seat offset −9.3 [−22.4, +3.7] | diagnostic; veto-confounded | same (`Q50c12_*`) | same |
| 11-20 | P1 gain vs sims | P1 c12 vs SL base | 4 v 4 | same | 800 (400 + 400) | EVAL, 2026112050 (200 pairs) | +98.1 [+72.4, +124.8] | diagnostic | same (`P1c12_*_s4`) | same |
| 11-20 | same | P1 c12 vs SL base | 16 v 16 | same | 800 | same | +20.0 [−2.2, +42.3] | diagnostic | same (`_s16`) | same |
| 11-20 | same | P1 c12 vs SL base | 64 v 64 | same | 800 | same | −3.9 [−22.8, +15.0] | diagnostic (second 64-sim read of P1; see row "P1 vs SL base") | same (`_s64`) | same |
| 11-20 | Softened prior at search time | P1 c12 at `policy_temp` 1.5 vs SL base at 1.0 | 64 v 64 | same | 800 | same | +3.9 [−13.6, +21.5]; seat offset −17.0 [−33.6, −0.4] | diagnostic: no gain | same (`_s64_pt1.5`) | same; #1439 review note 22 |
| 11-20 | Softening control | SL base at 1.5 vs SL base at 1.0 | 64 v 64 | same | 0 | — | — | **not run** (start refused: output path collision; logs only) | `RC/raw-net-1sim/BASE_*_s64_pt1.5.log` | plan said "still running" |
| 11-20 | **T1 (H1: 1-sim gain is a veto artefact)** | P1 c12 vs SL base | 1 v 1 | **temperature 0**, no noise, seat-balanced | 1,600 | EVAL, 2026112050 (same pairs as the +240.8) | **+79.1 [+59.7, +98.9]**; seat offset +1.7 [−5.1, +8.6] | **registered verdict: H1 supported** (point ≤ +120, upper < +180) | `RC/search-compose/results/pool_t0s1_P1.json`, `results.jsonl` | `RC/search-compose/PREREG.md` + addendum 5 |
| 11-20 | Hybrid identity smoke | hybrid(base, base) vs SL base | 1 v 1 | temp 0 | 64 | EVAL, 2026112050 | 0.5 score, 0.0 | control passed | `RC/search-compose/results/results.jsonl` (`smoke_hyb_ident`) | PREREG |
| 11-20 | One fast path: old compiled arena vs new path (strength test) | SL base compiled (old `--compile auto`) vs SL base eager (new; graphs replay it bitwise) | 64 v 64 | T0.1, no noise | 400 | EVAL, 20261121 | candidate −13.0 [−34.9, +8.7], i.e. new − old +13.0 [−8.7, +34.9] | **registered verdict: PASS (within noise)** by the rule in the PR body | `RC/one-fast-path/strength_compile_vs_eager.jsonl` | #1433 |
| 11-20 | Arena path benches | SL base vs SL base: eager / graphs / after-default / compiled | 64 v 64 | T0.1 | 64 each; 256 for g256 | EVAL, 20261121 | eager, graphs and after-default identical games (+16.3 each); compiled +49.2; g256 graphs −5.4, compiled +6.8 | diagnostic (timing; identity) | `RC/one-fast-path/arena_bench_*.jsonl` | #1433 |
| 11-20 | Optimizer-start arms after one smoke train phase | arm `a_fresh_w0` vs SL base; arm `b500_fresh_w500` vs SL base | 64 v 64 | T0.1 | 200 each | EVAL, 20261120 | −217.4 [−263.9, −176.9]; −15.7 [−50.9, +19.3] | **discarded**: trained on corrupted capped-collect rows | `RC/loop-smoke-gpu/optimizer-start/arena/*_vs_base.jsonl` | #1427 (review: figure not verified) |
| 11-20 | Loop-exp1 dry-run arena pair | stand-in vs SL base | 64 | — | 4 pairs per run | — | pooled +0.0, unread | rehearsal only (NOT_A_CANDIDATE) | `RC/exp1-prelaunch-dryrun/arenas/arena_pair.log` | #1437 |
| 11-18 | F12: 12 cycles (outcome 0.9 / search 0.1), final screen | F12 c12 vs SL base | 64 v 64 | T0.1, root noise 0.75, 450-ply cap | 800 | consumed DEV panel, 2026111999 | 47.125% [44.51, 49.74]; Elo-equivalent −20.00 [−38.31, −1.80] | **registered verdict: no gain** | `WK/fresh-current-policy-A39-F12-final-readout-*` | [F12 record](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md) |
| 11-18 | F12 interim views | F12 c3 / c6 vs SL base | 64 v 64 | same | 400 each (first 200 pairs) | same | 49.875% [46.01, 53.74]; 49.875% [46.16, 53.59] | diagnostic (descriptive) | `WK/fresh-current-policy-A39-F12-M{3,6}-vs-A39-400DEV-*-results-v1` | same |
| 11-19 | Q50: 12 cycles (outcome 0.5 / search 0.5), final screen | Q50 c12 vs SL base | 64 v 64 | same as F12 | 800 | consumed DEV panel, 2026111999 | 42.3125% [39.7715, 44.8535]; Elo-equivalent −53.85 [−72.09, −35.89] | **registered verdict: no gain** | `WK/Q50-final-readout-*` | [Q50 record](2026-11-19-fresh-current-policy-a39-q50-twelve-cycle-baseline.md) |
| 11-11 | Epoch-two distributed RL, first screen | 40-step candidate vs donor (SL base) | — | see record | 256 | 128-pair screen | 0.5176 [0.4720, 0.5631] | screen (eligible) | see record (`~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/`) | [transition](2026-11-11-epoch2-rl-transition.md) |
| 11-12 | Same, recovered 872-step endpoint | candidate vs donor | — | see record | 256 | same | 0.4375 [0.3877, 0.4873] | screen (failed) | see record (`recovery-screen-actual-independent-review-v1/`) | same |
| 11-12 | Same, sole confirmation | 40-step candidate (sha `d9c1f3d3…`) vs donor | — | see record | 1,000 | 500 disjoint pairs | 0.5005 [0.4767, 0.5243] | registered verdict: inconclusive | transition evidence dirs | same |
| 11-12 | Cap-40 exposure, 320-update endpoint | candidate vs donor | — | see record | 256 | validated panel | 0.375 (192/512) | screen (regressed) | `~/chess-artifacts/operations/rl-cap40-eight-20261112-fe23025d` | [cap-40](2026-11-12-epoch2-rl-cap40.md) |

The 2026-11-11/12 Ray run directories under `~/chess-artifacts/operations/` hold no summary of their own; their results are in the two records above: `rl-cpu-wiring-20261111-48546328` (CPU two-iteration wiring run, no strength claim), `rl-cpu-wiring-seeds-20261111-cd94a394`, `rl-policy-control-20261111-91c06e8f` and `rl-policy-control-20261112-ac566990` (the policy-control runs; their health audits are cited in the transition record), `rl-dose-eight-20261112-f856b343` (larger dose, 856 finite updates, rejected), `rl-prev-quota-recovery-20261112-8634d3da` (the recovered 872-step endpoint), `rl-cap40-eight-20261112-fe23025d` (cap-40). `rl-transition-preflight-20261111-local-v1/assessment.json` records the preflight status `RL_RESUMPTION_PREFLIGHT_COMPLETE_NOT_LAUNCHED`.

Earlier one-off records (2026-11-13 to 11-18), each with its own instrument; numbers from
[the experiments index](README.md): G8 vs SL base 21.875% [19.0304, 24.7196] (core outcome
baseline); LOW512 vs HIGH512 57.03125% [52.0518, 62.0107]; A512 vs SL base 50.65% [48.348, 52.952]
over 1,000 games; native SF-WDL .2 pilot 50.1953% vs A512 (256 games) and 51.1719% vs SL base (512);
SF current-P0 main-policy pilot B/A 0.416 and B/SL base 0.4265; retained-search main-WDL pilot .453 vs A
and .4555 vs SL base; U2 49.75% vs F1 and 45.375% vs SL base; SL base vs eight-thread Stockfish 60+1:
0W/4D/4L. None established a gain.

## Table 2 — Offline measurements

| Date | What | Rows / set | Result | Status | Bank + file | Record / PR |
|---|---|---|---|---|---|---|
| 11-19 | D2 pipeline audit of F12 cycle 1 | 17,628 rows, 192 games | stored inputs = play-path encoding (2,847 rows, max abs diff 0.0); targets legal and aligned; outcome POV right; harness reproduces the production cycle-1 learner bit-exactly | diagnostic: no defect | `RC/rl-audit/check1.log`, `check1_rows.json` | takeover D2 |
| 11-19 | D2 one-cycle arms (69 updates, 3e-6), held-out cycle-2 rows | 4,735 | policy KL from base / mean abs ΔQ: real 0.0393 / 0.0747; real seed 2 0.0386 / 0.0719; policy-only 0.0384 / 0.0197 (CE 0.9368 vs 0.9374); value-only 0.0171 / 0.0916; value labels shuffled 0.0434 / 0.3414; self-target 0.0001 / 0.0103; zero loss 0 / 0 | diagnostic | `RC/rl-audit/check6_*.json` | takeover D2 (the record omits the value-only and self-target arms and calls policy-only KL "not separately reported") |
| 11-19 | D3 policy entropy (nats) | 3,080 audit rows | BT4 1.49, Ceres C3 1.53, SL base 1.12, F12 c1/c6/c12 1.03/0.84/0.80, stored 64-sim target 0.69, R800 target 0.46 | diagnostic | `RC/rl-audit/target_scores.json`; `RC/plan-review-1439/outputs/ent.txt` | takeover D3 |
| 11-19 | D3 value vs 800-sim survivor Q (MSE) | 3,080 | outcome 0.272; SL base raw 0.035; F12 c12 raw 0.037; stored 64-sim search value 0.0145; Gumbel root Q sits 0.235 below survivor Q | diagnostic | `RC/rl-audit/value_scores.json` | takeover D3 |
| 11-19 | D3 value landscape over legal moves | external-ref subset | best-to-second gap 0.145 (base) → 0.119 (F12 c12) → 0.108 (Q50 c12); mean abs Q 0.41 → 0.33 → 0.30; BT4 gap 0.13 | diagnostic | `RC/rl-audit/value_children*.npz`, `analyze_values.log` | takeover D3 |
| 11-19 | P1 / F12 per-cycle drift on fixed held-out rows | 4,735 cycle-2 rows | KL from base at c12: P1 0.299, F12 0.288; abs ΔQ P1 0.066, F12 0.107; entropy 1.117 → 0.784 (P1 c12) | diagnostic | `RC/p1-policy-only/logs/eval_per_cycle.log` / `.json` | takeover P1 readout; [search ceiling](2026-11-20-sl-base-search-ceiling.md) dose→drift table |
| 11-19 | Value recalibration (F12 c12 / Q50 c12 vs SL base) | 3,080 root rows; 1,050 positions / 27,289 children | raw MSE gap to survivor Q: F12 +0.0024, Q50 +0.0070; information part F12 +0.0027 [0.0010, 0.0045], Q50 +0.0030 [0.0012, 0.0044]; Q50 calibration part +0.0041; child top-1 deficit F12 −0.029, Q50 −0.031 untouched by any monotone map; base_shape fits F12 T 0.6353 b −0.5878, Q50 T 0.5445 b −0.6383 | diagnostic (in-sample; A39-guided reference) | `RC/value-recalibration/results.json`, `recalibration_params.json`, `README.md` | D4 round 2 |
| 11-20 | Recalibration take-effect proof | 3,080 audit rows via the arena evaluator | Q50 abs Q / P(draw) 0.298 / 0.515 → 0.364 / 0.513; F12 0.331 / 0.528 → 0.386 / 0.513; WDL matches to 4e-6 | control passed | `RC/headswap/check_recal_auditrows.log` | D4 round 2 |
| 11-19 | Duplicate positions under the simple-RL book | book 153,456 TRAIN lines; 4,608 banked games / 492,464 rows | book clean (0 transpositions); cross-cycle row repeats F12 17.5%, Q50 17.7%; model: planned schedule at G=20k → 21.3% dup rows at generation 200, ~25% at 1e9 rows (extrapolated) | diagnostic (model calibrated on M) | `RC/duplicate-audit/README.md`, `dup_model_cal2.json`, `divergence_audit.json` | — (bank only) |
| 11-19 | Twofold repetition footprint | F12 and Q50 cycles 1–12 | rows with a game-history-repeat child 3.90% (F12) / 3.94% (Q50); about 25% of them store q exactly 0 (~1% of all rows); 1.11% of plies play into a twofold; 31.2% of games contain one | diagnostic | `RC/twofold-audit/tables.md`, `targets_tables.md` | [twofold record](2026-11-19-twofold-repetition-as-proven-draw.md), #1420 |
| 11-19 | Winning-root draw pruning footprint | 497,027 banked plies | 3.1 stored targets and 1.2 played moves per 10,000 plies change | diagnostic | `RC/root-draw-pruning/scan3.log`, `*_cases.json` | [root-draw record](2026-11-19-root-draw-pruning-singleton-fix.md), #1410 |
| 11-20 | 1-sim veto rate (own Q(top child) < own V(root)) | 1,050 value_children positions | base 57.0%, F12 c06 56.9%, P1 c12 38.2%, F12 c12 39.7%, Q50 c12 58.8%; 1-sim move = R800 48.1% (base) vs raw argmax 69.1% | diagnostic | `RC/search-compose/onesim_veto.json` | plan Stage 0 |
| 11-20 | Same rule on real 1-sim games | 395 positions from P1-vs-base 1-sim games | veto base 43.0%, P1 31.4%; played = raw argmax 0.658 / 0.608; veto rule predicts played move 0.786 / 0.829 | diagnostic | `RC/search-compose/pgn_onesim_check.json` | PREREG addendum 2 |
| 11-20 | Veto decomposition of the 1-sim gain | 1,050 / 828 leak-free | veto pattern explains 0.82–1.11 of P1 c12's and F12 c12's 1-sim top-1 gain, 0.32–0.37 of their regret gain; Q50 0.26 on R800 | interim (plan's reading) | `RC/search-compose/veto_decomp.json` | plan Stage 0 |
| 11-20 | Approximate 4-sim root | 1,050 | agree with R800: base raw 0.691 / 1-sim 0.481 / 4-sim 0.660; P1 c12 0.686 / 0.576 / 0.657 | diagnostic | `RC/search-compose/foursim_A39.json`, `foursim_P1c12.json` | — |
| 11-20 | Raw-policy rulers, in-sample (all 3,080 rows) | 3,080 | top-1 R800 / BT4 / C3: base 0.6666 / 0.7091 / 0.6909; P1 c01 0.6740 / 0.7143 / 0.6974; P1 c12 0.6685 / 0.6896 / 0.6857; net flips +31 (c01), −20 (c12) | diagnostic | `RC/plan-review-1439/outputs/indep.txt`, `rank.txt`; `RC/search-compose/flip_by_cycle.json` | plan Stage 0; #1439 review |
| 11-20 | Raw-policy rulers, leak-free | 2,433 rows / 190 games | base 0.658 / 0.704 / 0.679; P1 c12 0.649 / 0.675 / 0.666; F12 c12 0.647 / 0.676 / 0.667; on P1 c12's changed rows BT4 says net right 30%, base 49% | interim → confirmed by A1 negative control row | `RC/search-compose/flip_by_cycle_v2.json`, `ruler_table.json` | plan Stage 0 |
| 11-20 | Leak of audit rows into later training | 3,080 audit rows vs cycles 2–12 games | 647 rows recur (99 of 740 flip rows, 548 of 2,340 agreeing), plies 16–55; all 192 cycle-1 starts reused in cycles 2–12 | diagnostic | `RC/plan-review-1439/outputs/leak.txt`, `ovl.txt`; `RC/search-compose/leak_mask.npz` | #1439 review finding 6 |
| 11-20 | Target pre-screen (max top-1 gain if every flip learned) | 3,080 | stored +8.4 pp (R800) / +0.7 (BT4) / +1.4 (C3); R256 +11.9 / −0.5 / −0.1; R64 no-rescale +6.0 / +3.7 / +4.4 | diagnostic | `RC/plan-review-1439/outputs/prescreen.txt` | plan Stage 0 |
| 11-20 | Calibration gap (mean p_top1 − top-1 vs R800) | 3,080 | base −0.055 → P1 c12 +0.041; BT4 raw −0.148, C3 raw −0.152 (leak-free) | diagnostic | `RC/plan-review-1439/outputs/calib.txt`; plan's leak-free table | plan Stage 0b |
| 11-20 | Low-budget targets (flip, precision vs R800) | 3,080 | stored flip 0.240 prec 0.542 (+8.4 pp); R64 0.217 / 0.512 (+6.5); R16 0.216 / 0.396 (+2.1); R256n 0.268 / 0.617 (+11.9); CPU R64 reproduces the banked R64 32/32 | diagnostic | `RC/search-compose/flip_precision_lowbudget.json`, `logs/lowbudget_cpu_check.log` | — |
| 11-21 | **Deep-SF 8M labels (MultiPV 10, cold TT, production TB)** | 3,080 (leak-free 2,433) | 0 timeouts; depth median 20, min 13; best move unchanged 1M → final 0.746. Leak-free regret (cp): BT4 raw 18.9, R800 19.6, C3 raw 19.7, R256n 19.9, stored 22.8, R64 no-rescale 23.2, R64 24.3, R16 28.8, R8 30.4, trained nets 37.8–41.0, SL base 44.6 | diagnostic (cross-check ruler) | `RC/search-compose/logs/sf_parse_final.txt`, `sf_scores.json`, `sf_deep8M.jsonl` | plan Stage 0b (the plan quotes an interim n=127 table; superseded) |
| 11-21 | Deep-SF paired net-vs-base regret | 2,387 leak-free rows / 187 games | P1 c12 −5.41 cp [−10.07, −1.53]; F12 c12 −4.34 [−8.90, −0.10]; Q50 c12 −1.28 [−7.52, +4.58]; stored target −21.44; BT4 raw / C3 raw / R800 agree with SF's best 0.576 / 0.574 / 0.576 | diagnostic | `RC/search-compose/ruler_consensus.json` | — |
| 11-21 | Ruler ordering check vs known 64-sim Elo | leak-free value_children rows | R800, BT4 raw and C3 raw: `correct: false`; SF8M ranks P1 c12 and F12 c12 above the base (agrees with T1, not with 64 sims) | diagnostic | `RC/search-compose/ruler_table.json` (`ordering`), `ruler_consensus.json` | PREREG addendum 5 |
| 11-20 | **A1 negative control (rl-run1 P5)** | 2,433 leak-free / 190 games | top-1 vs base: BT4 raw P1 c12 −0.0279 [−0.0411, −0.0148], F12 c12 −0.0275 [−0.0405, −0.0146], Q50 c12 −0.0321 [−0.0450, −0.0191]; C3 raw −0.0140 [−0.0270, −0.0009], −0.0127 [−0.0261, +0.0007], −0.0214 [−0.0351, −0.0077] | diagnostic | `RC/rl-run1-prereq/a1_negctrl_score.json` | #1448 (open) |
| 11-20 | **A1 positive control** (base's R64 move vs base raw argmax) | same | BT4 raw +0.0074 [−0.0117, +0.0265]; C3 raw +0.0181 [−0.0019, +0.0381]: lower bound not above 0 on either raw ruler | diagnostic | `RC/rl-run1-prereq/a1_poscontrol_score.json` | plan Stage 0b rule (3) |
| 11-21 | Value pre-screen vs deep SF (expected score MAE) | 2,433 leak-free | base 0.0657; F12 c12 −0.0115 [−0.0148, −0.0083]; Q50 c12 −0.0119 [−0.0159, −0.0078]; P1 c12 +0.0100 [+0.0068, +0.0132]; stored search WDL −0.0156; root Q R64 +0.0499; F12 training target +0.0497; outcome z +0.0616 | **interim** (written 00:46, report pending) | `RC/search-compose/value_prescreen.json` | — |
| 11-20 | Learner batch screen | 2,433 leak-free | only `smoke_cpu` (2 steps) banked; no arm read | **in progress / held** | `RC/search-compose/batch_screen/score.json` | PREREG addenda 4–6 |
| 11-20 | lc0 32-node cross-check of the searched-BT4 ruler | 32 rows | real lc0 (eigen) root visits; the CUDA build later matched 8/8 rows exactly | control | `RC/search-compose/lc0_crosscheck_n32.json`; `~/local_engines/lc0-0.32.1-cuda/smoke_gpu.json` | — |
| 11-20 | Loop smoke collects with the 512 leaf cap | 3 generations | reply-side Spearman(child Q, prior) ~0 vs +0.60 uncapped; reply-side targets vs cold re-search KL 2.4, top-1 0.53; 30.6% of leaf visits absorbed (CPU replay) | **discarded** (CORRUPTED_TARGETS) | `RC/loop-smoke-gpu/run/CORRUPTED_TARGETS.txt`, `optimizer-start/reconcile_cold.txt`, `mini/*.stats` | #1427, #1428, [buffer-full record](2026-11-20-gumbel-buffer-full-fallback.md) |
| 11-20 | Optimizer-start drift (46 updates on smoke rows) | 622 probe rows | probe mean abs ΔQ: a 0.295, b46 0.357, b500 0.039, c (SL Aurora state) 0.249, d (lr 3e-7) 0.081, f Aurora mlp_out 0.228, f Aurora default scope 0.424; g (F12 c1 rows, 69 updates) 0.018 | discarded except arm g (corrupted rows) | `RC/loop-smoke-gpu/optimizer-start/arms/*/result.json`, `compare_g.json`, `compare_banked.json` | — |
| 11-20 | Encoding check of a smoke collect | gen 1 | fp32 vs bf16 evaluator top-1 agree 0.969; vs stored prior 0.984; mirror inconsistencies 0/200 | control | `RC/loop-smoke-gpu/check_encoding_gen1.txt`, `verify_rows_gen1.txt` | — |
| 11-20 | Audit-row value floor (L2) | 3,080 | SL base abs Q 0.4087, P(draw) 0.5134; Q50 c12 0.2981, 0.5145 | control | `RC/exp1-prelaunch-dryrun/gates/L2_*.json` | #1437 |
| 11-21 | Collect gates on a clean 3-process collect | 512 games | rows/game 94.8; KL(target, prior) 0.298; reused-root share 0.649; survivor − node Q 0.209 | diagnostic | `RC/collect-throughput/evidence/logs/gates_k3.json` | #1441 (open) |
| 11-20 | A1 priors via CPU fp32 vs banked priors | 3,080 | SL base argmax agreement 99.58%, max abs p diff 0.018 | control | `RC/rl-run1-prereq/README.md` | #1448 (open) |

## Table 3 — Throughput and engine speed

| Date | Setting | Result | GPU state | Bank + file | Record / PR |
|---|---|---|---|---|---|
| 11-20 | CUDA graphs vs eager, self-play 64 sims, 192 games / 128 slots | 1,602 → 2,727 games/h (1.70×); collector shape (max_batch 512) 2,487 → 5,588 (2.25×); 256 sims 544 → 782 (1.44×); arena 64 v 64 1,650 → 3,418 (2.07×); digests equal | shared with a head-swap arena | PR body | #1414 |
| 11-20 | Raw forward ceiling, SL base, inputs on device | graph rows/s: 4.4k (16 rows), 9.0k (64), 12.1k (128), 12.7k (256), 11.0k (512), 10.7k (1,020); bitwise equal to eager | — | PR body | #1414 |
| 11-20 | Collect 128 games / 64 sims / conc 128 | eager 2,720 games/h; graphs 3,763; after default 3,760 / 3,613; graph cap 1024 3,778, 2048 3,787; rule-50 TB memo 3,704 (11% hit); all one digest | idle | `RC/one-fast-path/logs/bench_*.txt`, `bench_*.log` | #1433 |
| 11-20 | Arena wall, SL base vs itself, 64 sims | 64 games: eager 115.9 s, graphs 59.5 / 60.1 s, compiled 307.1 s; 256 games: graphs 198.5 s, compiled 304.0 s; steady 0.65 vs 0.76 s/game | idle | `RC/one-fast-path/logs/bench_*.txt` | #1433 |
| 11-20 | Collect 128 games, graphs vs no graphs | 2,510 vs 3,478 games/h, same 12,299 rows | idle | `RC/snapshots-arms-e2e-20261120/logs/collect_*.log` | #1430 |
| 11-20 | Dry-run collect (loop-exp1 step 1, 128 games) | 3,284 games/h, 12,868 rows | idle | `RC/exp1-prelaunch-dryrun/logs/gpu_collect.log` | [loop-exp1](2026-11-20-loop-exp1-one-generation-from-sl-base.md) |
| 11-20 | Single-process concurrency sweep, 64 sims | 1×128: 2,881 games/h, 5,282 fwd rows/s, 8.7 GB; 1×256: 3,372, 6,151, 12.2 GB; 1×512: 3,355, 6,047, 29.9 GB; 1×256 graph cap 2,048: 2,675 | idle (one aborted start on a shared GPU, set aside) | `RC/collect-throughput/logs/sweep.txt` | #1441 (open) |
| 11-20 | Multi-process, 64 sims | 2×128: 4,546 games/h, 16.2 GB; 3×128: 5,021, 9,262 rows/s, 21.9 GB; 2×256: 4,553, 22.6 GB; 3×128 (prof) 5,131 games/h, 491k training rows/h, 9,487 rows/s, 22.1 GB | idle | same | #1441 |
| 11-20 | Exchange rate by sims | 16 sims: 1×256 619k rows/h; 3×128 1.39M; 4×128 1.66M; 8×64 1.85M (9,690 rows/s, 22.9 GB); 6×96 1.86M (9,755 rows/s, 25.9 GB); 8×96 1.44M (31.0 GB, slower). 32 sims: 1×256 452k; 3×128 879k. 64 sims: 3×128 491k. 1×256 at 64 sims (prof) failed to parse | idle | same | plan Stage 1 table; #1441 |
| 11-21 | `--collect-procs` evidence on git `347aaf08c` | K=1: 3,037 games/h, 288k rows/h, 8.6 GB; K=3: 4,982 games/h, 472k rows/h, 22.9 GB; K=3 repeat started | idle (Stockfish load at K=1 start) | `RC/collect-throughput/evidence/logs/evidence.txt` | #1441 — **in progress** |
| 11-19/20 | 800-game 64-sim head-swap arena walls | 1,600–2,527 s per arm (q50_B 2,527 s) | partly shared | `RC/headswap/table.md` (wall column; resumed arms show last segment) | D4 |
| 11-21 | UCI pool: parent `f51c51f20` vs #1440 head | visits/s geometric mean ×1.130 (movetime 5000), ×1.161 (100k nodes), ×1.300 (20k nodes); endgames ×1.7–1.9; start position first search ×0.905 | idle, ABAB, INDEPENDENT | `RC/engine-speed-profile/idle/pvp_*.jsonl` (`scripts/analyze_pvp.py`) | #1440 review |
| 11-20 | UCI pool shared-GPU profile | 16.8% of forwarded rows terminal, 15.4% duplicates, 67.8% unique | shared | `RC/engine-speed-profile/runs/pool_*.jsonl` | #1440 body |
| 11-21 | Stockfish label prefix cache | 878 → 321 µs per label (2.7×), engine input byte-identical | — | PR body | #1442 |
| 11-21 | lc0 v0.32.1 CUDA build, BT4-it332 | go nodes 800: 5,895 nps (cuda-fp16), 2,119 (cuda); short cold searches, not a throughput figure | idle | `~/local_engines/lc0-0.32.1-cuda/smoke_gpu.json`, README | — |
| 11-21 | Ceres 0cb57d8, BT4 via ONNX | 256 → 800 nodes in 0.32 s (~1,700 nodes/s); first info after ~56 s; no nps claim | idle | `~/local_engines/ceres-0cb57d8/README.txt` | — |

## Table 4 — Infrastructure facts established

| # | Fact | Source |
|---|---|---|
| I1 | F12, P1 and Q50 trained with NAdamW (torch NAdam, decoupled wd), betas (0.9, 0.98), eps 1e-7, one flat LR 3e-6 on both groups (190 tensors wd 1e-4, 289 wd 0), batch 256, no clipping, fresh optimizer at cycle 1. The simple-RL driver defaults to the same | `RC/search-compose/PREREG.sha256.addendum` (addendum 4); plan Stage 0 |
| I2 | The SL base was trained with production's Aurora split: 48 matrices (scope `mlp_out`) on Aurora at LR 6e-5 (20 × 3e-6), momentum 0.95 Nesterov; 431 tensors on AdamW at 3e-6, betas (0.9, 0.95); batch 512; saved groups carry `aurora_pp_iterations` 3, `pp_beta` 0.25, `polar_steps` 8, `polar_express`, fp16; zclip on; LR decayed from a 3e-5 peak | plan Stage 0 and Stage 1 |
| I3 | **What the 1-sim Gumbel arena plays:** with one simulation and no root noise only the top-prior child is visited; the move is sampled at T0.1 from softmax(log prior + 5.1·minmax(completed Q)), unvisited completed Q = (V_root + Q_top)/2. The top-prior move is vetoed whenever Q(top child) < V_root. Default-temperature 1-sim matches are veto-confounded | PREREG background; #1438 review |
| I4 | At temperature 0 with one simulation the halving survivor is the top-prior move: the match is pure prior argmax | PREREG T1 |
| I5 | **Same-seed collections cannot share games.** At positive move temperature the played move is sampled from the stored target, so a flag that changes the target (`--no-target-q-rescale`) changes play from the first sampled ply. Use one collection and the offline retarget step. Also, `--collect-procs 3` rows are not the K=1 rows (different RNG streams) | plan Stage 1 "The collection"; #1441 body |
| I6 | The no-rescale target can be rebuilt offline from root statistics: reproduces the searched no-rescale target to 9.27e-8; rescale-on rebuild differs from the stored target by up to 7.4e-4 | plan Stage 1; `RC/rl-audit/target_scores.json` |
| I7 | **Usable VRAM is about 26 GB total.** 8×96 at 16 sims used 31.0 GB and ran slower than 6×96 at 25.9 GB; 1×512 at 64 sims peaked at 29.9 GB; #1441 refuses shapes above an 83% limit | `RC/collect-throughput/logs/sweep.txt`; #1441 body |
| I8 | **Position-key leak between cycles:** every cycle of F12/Q50 replays the same 192 starts; 647 of the 3,080 audit rows recur in cycle 2–12 games; cross-cycle row repeats 17.5% / 17.7%; ply-16 rows repeat 100% by construction | `RC/plan-review-1439/outputs/leak.txt`, `ovl.txt`; `RC/duplicate-audit/README.md` |
| I9 | Audit row sets: 3,080 positions from the 192 cycle-1 F12 games (`RC/rl-audit/positions.npz`); leak-free subset 2,433 rows / 190 games (`RC/search-compose/leak_mask.npz`); value_children 1,050 positions (828 leak-free) | rl-audit, search-compose |
| I10 | **lc0 v0.32.1 CUDA build** at `~/local_engines/lc0-0.32.1-cuda/lc0` (sha256 `d3230262…`), backends `cuda` (fp32) / `cuda-fp16`, sm_120 only. `lc0.config` beside the binary holds the production Syzygy pair (loads 510 WDL / 510 DTZ at the first search); `lc0_tb.sh` launches BT4-it332 (sha256 `e6ada9d6…`) with tablebases. 32-node root visits match the eigen cross-check 8/8 | `~/local_engines/lc0-0.32.1-cuda/README.txt` |
| I11 | Older lc0 builds, left untouched: `~/local_engines/lc0` (v0.32.1, eigen CPU backend) and `~/local_engines/lc0_033` | `~/local_engines/lc0-0.32.1-cuda/README.txt` |
| I12 | **Ceres 0cb57d8** at `~/local_engines/ceres-0cb57d8`: `source env.sh` (puts `/usr/lib/wsl/lib` first, else ErrorNoDevice), launcher `ceres_tb.sh`, settings `run/Ceres.json`. Default net C3-768-30-pre8-I8 (sha256 `44aa02c7…`) via ONNX Runtime CUDA fp16, with `SyzygyPath` = the production pair (510/510). BT4 on Ceres's native CUDA backend fails; BT4 via its ONNX export works. The C3 default config's own GPU smoke is still pending | `~/local_engines/ceres-0cb57d8/README.txt` |
| I13 | Leaf-buffer cap defect: a collect whose evaluator cap (old default 512) is below concurrency × topk backed starved leaves with the root's Q (wrong sign at odd depths). Fixed by #1427 (cap = leaf buffer; refuse capped rows) and #1428 (flush and resume, ABI 11). Reach: only the loop smoke's collects; 0 of 708,381 leaf evaluations bind at the banked arena settings | #1427 / #1428 reviews; `RC/loop-smoke-gpu/run/CORRUPTED_TARGETS.txt` |
| I14 | CUDA graphs replay eager forwards bitwise (measured); since #1433 they are the only CUDA collect path, and `--compile auto` no longer compiles where graphs can run (it compiled at games × sims ≥ 12,800 before). The compiled arena path is the numeric-changing one | #1414, #1433 |
| I15 | No candidate-seat offset in the arena: +6.4 [−4.2, +17.1] over 998 same-net pairs. At T0.1 from one seeded generator the two games of a pair differ (197 of 200 pairs), so identical weights do not score exactly 0.5; a 400-game 64-sim read is about ±22 Elo | D4 round 2 |
| I16 | Collection reuses the tree across plies: stored "64-sim" targets include inherited visits (reused-root share 0.55–0.65 in clean collects); `target_q_rescale` is not mapped in `build_selfplay_gumbel_config` | takeover D3; `gates_k3.json`; `RC/snapshots-arms-e2e-20261120/gates_src_graphs.json` |
| I17 | The Gumbel root-node Q sits 0.235 below the survivor Q and must not be a value target; in the stored "search value" the draw probability is the net's raw root draw | takeover D2/D3 |
| I18 | The P1 harness reproduces all twelve F12 learner checkpoints bit-exactly (model, optimizer, scheduler, clip state) | takeover P1 readout |
| I19 | Deep-SF 8M MultiPV 10: 1M nodes at MultiPV 10 reaches only depth 12–14 on these rows; 8M reaches median depth 20 | `RC/search-compose/README.md`; `logs/sf_parse_final.txt` |
| I20 | Real-net outputs depend on batch size at float level on any device (3 of 12 real-net CPU trees differed, best moves identical); `scripts/bench_selfplay_identity.py` is nondeterministic run to run | #1440 review; #1428 review |
| I21 | Opening split counts differ by salt: the loop's split on the production books gives 4,000 EVAL / 153,842 TRAIN (EVAL 3,725 + 275 by book); the duplicate audit's own split (salt `duplicate-audit`) gave 153,456 TRAIN | `RC/exp1-prelaunch-dryrun/openings/summary.json`; #1416 review; `RC/duplicate-audit/README.md` |
| I22 | Searched BT4/C3 via `search-compose/scripts/ext_search.py` (no TB) is superseded as a ruler by the real engines with tablebases (owner instruction) | PREREG addendum 6 |
| I23 | The `debug on` pool fields `evals= terminal= duplicate=` come from the last chunk of a search, not the whole search | #1440 review |
| I24 | A test cell that fails only on local Python 3.10 (`add_note`) passes on CI's 3.13 | #1428 review |

## Already answered: do not rerun

- **SL base search gain at 64 / 256 / 800 sims?** +848.7 (64 v 1, veto-confounded), +272.4 (256 v 64), +478.0 (800 v 64). Table 1 search-gain rows. A 128-v-64 rerun was started and stopped as a duplicate.
- **Search headroom over the raw prior (temperature-0 64 v 1)?** Not yet measured; `t0_head` is queued (below). Do not reuse the +849 for this.
- **Head-swap hybrids at 64 sims, F12 and Q50?** Q50: policy +4.3, value −55.2, full −63.2. F12: policy −15.7, value −18.3, full −34.9; additive within noise. Table 1 D4 rows.
- **Trained policy + base value at 64 sims (T2 / `h64_PpVb`)?** Answered three times at ~0 (Q50 +4.3, F12 −15.7, P1 −3.9 / −10.9); T2 withdrawn as duplicative.
- **Does value recalibration recover the play deficit?** Q50: mostly (R1 share 0.85, R3 0.77); F12 value-only unresolved (R2 +6.1), full net R4 share 0.97. Offline: F12 gap mostly information, Q50 mostly calibration. Table 1 R1–R4, Table 2 recalibration.
- **Is there a seat offset?** No: +6.4 [−4.2, +17.1]. Table 1 pooled seat row.
- **Value loss implicated in F12's loss?** Yes by rule: P1 +42.6 [+30.1, +55.2] over F12 c12.
- **Loop nets vs SL base at 1 / 4 / 16 / 64 sims (T0.1)?** P1 c12 +240.8 / +98.1 / +20.0 / −3.9; F12 c12 +211.7 (1 sim); Q50 c12 +128.7 (1 sim). Table 1 raw-net-1sim rows.
- **Temperature-0 1-sim P1 (T1)?** +79.1 [+59.7, +98.9], H1 supported. `t0_F12`, `t0_Q50` are queued, not run.
- **Softened prior at search time (policy_temp 1.5)?** No gain: +3.9 [−13.6, +21.5]. Its base-vs-base control never ran.
- **Collector exchange rate by sims?** 16 / 32 / 64 sims: 1.86M / 0.88M / 0.49M training rows per GPU-hour at the best multi-process shapes; usable VRAM ~26 GB. Table 3.
- **Is the CUDA-graphs path strength-neutral vs the old compiled arena?** PASS within noise, +13.0 [−8.7, +34.9]. Table 1.
- **Does the F12 cycle-1 pipeline have a data or training defect?** No (D2). Table 2.
- **Do the trained policies improve on independent raw-policy rulers out of sample?** No: after c01 they decline on BT4 raw and C3 raw (A1 negative control CIs exclude 0 on BT4 raw). Table 2.
- **Which raw-policy ruler orders base / P1 / F12 / Q50 like 64-sim matches?** None of R800, BT4 raw, C3 raw; deep SF disagrees with all three. Table 2 ordering row.
- **Duplicate-row rate of the simple-RL book?** ~15% by generation 50 and ~21% by 200 at 20k games/generation (model). Table 2.
- **Twofold and draw-pruning footprints?** Table 2.

## Registered but not yet read / queued

- **`search-compose` queue after addendum 7** (`RC/search-compose/PREREG.sha256.addendum`): `t0_head` (SL base 64 v 1 sim, both temperature 0, 400 games per orientation, seed 2026112050; registered in addendum 2, expectation below +849, no rule); batch screen `bs_identity`, `bs_accum`, `bs_N256`, `bs_A512`, `bs_A2048`, `bs_N512` (rule in addendum 6: P-noise supported if A2048 beats A512, or N512 beats N256, on changed-row precision at matched KL by ≥ 0.05 on both BT4 raw and C3 raw with the paired CI excluding 0); `uci_lc0` (lc0 CUDA fp32 + BT4 + TB, 256 nodes, leak-free rows) and `uci_lc0_notb`; `uci_ceres` (C3 + TB, 256 nodes) after its smoke; `t0_F12`, `t0_Q50` (exploratory, same design as T1). Added in addendum 5: `ext_ch` (BT4/C3 1-ply child values).
- **Parked in addendum 6 with launch conditions:** `rt_cs02` / `rt_cs005` and `rtc_*` (search re-tune); `r4` / `r16` (P1 at s vs base at 2s); `h1_*`; `h64_PbVp`; low-budget GPU searches; `ext_children`; every fresh-seed confirmation unless a rule calls for it.
- **In progress:** `collect-throughput` evidence `k3_repeat`; `value_prescreen.json` report; deep-SF consensus write-up.
- **RL run 1 Stage 0b** ([plan](2026-11-20-rl-run1-multi-generation-plan.md)): A2 audit set (~3,000 rows from EVAL lines 800–1,799), ruler validation rule (negative control, positive control), searched BT4/Ceres labels. Scorer PR #1448 open.
- **RL run 1 Stage 1** (DRAFT, awaiting owner approval; nothing launched): one SL-base collection of 1,040 games, seed 2026112300; cells a512 (stored target), b512 (no-rescale, rebuilt offline), c512 (train temp 1.5), each batch 512 on the Aurora split with fresh state, and a2048 (stored, effective batch 2048, LR 1.2e-5); each plays 2,000 seat-balanced 64-sim games vs the SL base on `s1_match` (EVAL lines 100–599), z 2.498, success = lower bound > 0. Prerequisite PRs open: #1441, #1445, #1446, #1447, #1448.
- **Parked in the plan:** distance to BT4 at 1 node; BT4 raw-policy target diagnostic (launches only if every Stage 1 cell fails); value-first cell.
- **Loop experiment 1** is superseded as a standalone experiment by RL run 1 (its commands and gates are reused).

## Withdrawn or superseded claims

| Claim | Where | Replaced by |
|---|---|---|
| "The learner is the binding constraint, not target quality"; "per sim, 64 beats 256" | earlier plan draft | withdrawn as R800-only; no target wins on every ruler (plan Stage 0) |
| +849 as the headroom for "half of that 800 Elo" | owner anchor, plan draft | veto-confounded (#1439 finding 1); `t0_head` queued |
| "Flips learned barely exceed rows broken" | plan draft | inverted: 172 learned vs 192 broken at P1 c12, net −20 (#1439 finding 3) |
| Net flips scored "vs the SL base's 800-sim reference" | plan draft README | the metric is agreement with the stored training target (#1439 finding 4) |
| "No-rescale is the only target that improves on every ruler" | plan draft | contradicted by deep SF (#1439 N6) |
| Interim deep-SF table (n = 127): R64 16.3 cp beats no-rescale 25.5; base 42.6, P1 c12 29.5, stored 25.2 | plan Stage 0 (still quoted there) | full labels, leak-free n = 2,433: R64 24.3, no-rescale 23.2, stored 22.8, base 44.6, P1 c12 38.6 (`sf_parse_final.txt`). The R64-over-no-rescale ordering does not survive; trained nets above the base does |
| `ruler_table.json` SF8M column (base 38.55 cp on "2,433" rows) | search-compose | written 23:28, before the 8M labelling finished (log ALL_DONE after 10,844 s from 21:30); use `sf_scores.json` / `sf_parse_final.txt` |
| Veto rate 43% (base) / 31% (P1) under PREREG "Background" labelled as the audit positions | `PREREG.md` | addendum 2: 57% / 38% on the 1,050 audit positions; 43% / 31% on 395 real-game positions |
| T2 (`h64_PpVb`) and its confirmation | PREREG | withdrawn before running as duplicative of the head-swap (addendum 7) |
| Base-vs-base +22.6 as a possible seat offset | takeover D4 part-result | closed: one RNG draw; pooled +6.4 [−4.2, +17.1] (D4 round 2) |
| D1 800 v 64 mid-run reading | takeover draft | banked final +478 (#1413 review) |
| P1 deciding match at 800 games vs the base | takeover draft | underpowered; replaced by the 1,600-game head-to-head vs F12 c12 (#1413 review) |
| "T1 not read" | plan header | T1 read: +79.1, H1 supported (addendum 5) |
| Learner-screen arms N1024, N4096, N4096L | addendum 4; plan Stage 0b | addendum 6 arms: N256, A512, A2048, N512 |
| "The local lc0 builds are CPU-only" | plan, parked BT4 match | CUDA build `~/local_engines/lc0-0.32.1-cuda` (built 11-20, GPU smoke 11-21) |
| Collect throughput 2,424–2,583 games/h (eager) and 5,180–6,139 (concurrency 256) | earlier loop-exp1 drafts | eager numbers obsolete after #1433; the 256 reads had the cap defect (loop-exp1 record) |
| Optimizer-start arm −217 Elo | `loop-smoke-gpu/optimizer-start` | trained on corrupted rows; not verified (#1427 review) |
| 128-v-64 search-ceiling descriptive +152 | `search-ceiling` | stopped, incomplete, not a result |
| "96% of ply-16 rows are cross-cycle repeats" (outside claim) | duplicate audit brief | 100% by construction; 96% fits plies 16–19 pooled (`duplicate-audit/README.md` §C) |
| #1440 counters described as "per search" | #1440 body | last chunk only (#1440 review) |
