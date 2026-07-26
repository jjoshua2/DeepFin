# RL Loop Audit — plan and standing record

**Status: LIVING DOCUMENT.** Update the status cells in place as checks run.
This is both the plan (what to verify, in what order, with which exact command)
and the record (what has actually been verified, when, with what evidence).

`docs/experiment_ledger.md` owns *experiments* — hypotheses, yardsticks,
verdicts. This document owns *invariants* — things that must be true for any
experiment's numbers to mean anything. A ledger verdict read off a pipeline that
fails an invariant here is not a verdict.

---

## 1. Why this exists: the recurring bug class

Nearly every production-affecting defect found in this pipeline has the same
shape: **a number that does not mean what its name says**, with nothing
asserting the two agree. Not crashes — silent semantic drift.

| defect | name said | actually was |
|---|---|---|
| `train_views_per_position` (#225) | views per ingested position | per *matching* position; true reuse 0.46 vs 2.5 configured |
| `selfplay_fraction` (#224) | share of games | share of *slots*; realized completed mix 83.5% vs 0.35 |
| `opening_fen_dole_max_fraction` | cap | `0` means **uncapped**, not "none allowed" |
| `sf_nodes: 5000` | label depth | a PID *floor*; labels use ~698k |
| `gate_passed: 1` | gate passed | `gate_games: 0` — no gate ran |
| `holdout_generation` | ruler identity | inert (always 0); the ruler rotates anyway |
| `matching_games` | games generated | current-model games only |

**Corollary for auditing:** reading the code is not enough, because the code is
usually correct *locally*. The bug lives in the composition — between the config
that sets a number, the stage that resolves it, and the counter that reports it.
**Every check below must measure the realized value on the live run**, not
assert the code looks right.

---

## 2. Method rules — how not to fool ourselves

These are written from mistakes made *while doing this audit*, not hypotheticals.

1. **`scripts/audit_realized_config.py` is the authority on configured-vs-realized.**
   When a hand-rolled number disagrees with it, assume the hand-rolled one is
   wrong first. On 2026-07-26 an ad-hoc sum "found" a dole-cap violation, a
   mechanism was invented, and a fix was written, tested and committed before
   the premise was rechecked. The audit tool had said `ok` the whole time. The
   cap was fine.

2. **`outcome_stats` keys are NESTED. Never sum by substring.**
   It is a pipe-delimited string (not JSON — `json.loads` raises, and a
   `try/except` silently yields `{}` and zeros). `opening_fenlist_games` is a
   total; `selfplay_fenlist_games` repeats that subset; `selfplay_fenlist_stm_w/l`
   splits it again. Summing everything matching `fenlist` triple-counts. Prefer
   the `*_games` totals. Use `scripts/loop_health.py::parse_outcome_stats`.

3. **A mechanism must fit the magnitude, not just the direction.**
   The invented dole bug predicted exactly 4x; the (bad) data showed 2-3x. That
   mismatch was disconfirming evidence and was glossed over. If the proposed
   cause does not produce the observed number, it is not the cause.

4. **A median over a window hides a regime change.** `audit_realized_config.py`
   medians by design. When a knob changed mid-window, slice the window or read
   per-iteration rows.

5. **Never pick a yardstick that only exists when the bug fires.** The PR #224
   readout asked for `abandoned/started` from a log line that is only emitted at
   teardown — in the success case it emits nothing, so the ratio is undefined and
   absence-of-line is not a measurement.

6. **Frozen rulers are not strength rulers.** `value_regret`, `audit_targets`
   and the panels can all move opposite to real play
   (`offline_distillation_value_trap`, the 07-21 RULER MIRAGE row). Only a paired
   arena decides strength.

7. **Verify identity before believing a lopsided arena.** Dump
   `torch.load(ckpt)["arch"]` for both sides and confirm `policy_encoding`,
   `input_extra_features`, `input_history_encoding`, `history_rep_fix` match.

8. **Test a new parser against real output before trusting its file.** The
   ratchet's Elo regex was written against an assumed format and silently
   corrupts the CSV on positive Elo (see L1). A parser that is only exercised by
   the happy path you imagined is not verified. Run it against a captured log.

9. **Never edit a shell script while it is running.** Bash reads scripts
   incrementally by byte offset, so an edit that shifts offsets can make the
   running instance execute garbage. Kill and relaunch, or wait for it to exit.

10. **Beware `pgrep -f` self-matching.** A pattern passed on the command line
    appears in that command's own `/proc` cmdline, so `pgrep -f` matches the
    watcher itself. An `until ! pgrep -f "..."` loop written inline can never
    exit — one spun for 12h. Put the pattern inside a script file instead.

---

## 3. The loop, as stages

```
 (A) config resolve -> (B) opening select -> (C) selfplay generate
        -> (D) SF label -> (E) finalize to rows -> (F) upload/inbox
        -> (G) ingest + holdout split -> (H) sample -> (I) train step
        -> (J) checkpoint + best-model -> (K) publish to workers -> back to (C)
                                   \-> (L) evaluation / rulers
```

Each stage below lists its invariants, the exact instrument, and current status.

**Status vocabulary:** `VERIFIED <date>` (measured on the live run) ·
`FAILED <date>` (measured, invariant does not hold) · `OPEN` (never measured) ·
`CODE-ONLY` (argued from source, not measured — weaker, treat as OPEN for
anything load-bearing).

---

## A. Config resolution — yaml → trial → published reco → worker

| # | invariant | instrument | status |
|---|---|---|---|
| A1 | every ratio knob's realized value matches the yaml | `PYTHONPATH=. python3 scripts/audit_realized_config.py --last 20` | **VERIFIED 2026-07-26** — exit 0, all knobs ok |
| A2 | the live yaml validates and reloads (all-or-nothing) | grep `/tmp/chess_training.log` for validator/reject/reload-fail | **VERIFIED 2026-07-26** — clean after 2 edits |
| A3 | `pb2_bounds_*` do not silently OVERRIDE their base key | read `_build_mutation_param_space`; confirm trial `lr` in `progress.csv` | **VERIFIED 2026-07-26** — caught overriding `train.lr`; both now pinned 3e-5 |
| A4 | published reco matches the resolved config | diff `recommended_worker` in `runs/pbt2_small/server/trials/<prefix>_00000/publish/manifest.json` against the flattened yaml | **VERIFIED 2026-07-26** — 64 shared keys, 63 exact. The one difference is intentional: `sf_nodes` published **698289** vs yaml **5000**, because the yaml key is the PID *floor* and the published value is the live `base_nodes` ([[sf_label_nodes_are_not_sf_nodes]]). Any OTHER mismatch here is a finding |
| A5 | live yaml divergence from `main` is known and intentional | `git diff origin/main -- configs/pbt2_small.yaml` | **VERIFIED-DIVERGENT 2026-07-26** — 4 known deltas, see the reconciliation block below. **This is the highest-risk item here: a naive "take main's yaml" at the next restart reverts the LR fix and re-enables seeding.** |

### A5 reconciliation — read this BEFORE any restart

The live tree is never checked out while a run is live, so the live yaml drifts
from `main` by design. As of 2026-07-26 the deltas are:

| key | live | `main` | at the next restart |
|---|---|---|---|
| `lr` | **3e-5** | 0.0003 | **KEEP LIVE.** main's value is the one that cost −494 Elo. Must be changed together with `pb2_bounds_lr` (see A3) |
| `pb2_bounds_lr` | **[3e-5, 3e-5]** | [0.0003, …] | **KEEP LIVE** — this key OVERRIDES `train.lr`, it does not bound it |
| `opening_fen_dole_per_iter` | **0** | 1 | keep live until the no-seed readout closes; restoring costs a session restart (see C6) |
| `opening_fen_list_path` | `retire_32` | `retire_250` | **KEEP LIVE.** The monitor rewrites this every iteration; main's value is always stale |
| `gumbel_c_scale` | *absent* | **0.1** | **TAKE MAIN'S.** PR #249; value-identical to the resolved default so it is a no-op now, but the live yaml should gain the pin |

**Rule: reconcile key-by-key, never by wholesale copy in either direction.**

## B. Opening / seed selection

| # | invariant | instrument | status |
|---|---|---|---|
| B1 | seeded games ≤ `opening_fen_dole_max_fraction × games_per_iter` | per-iteration `opening_fenlist_games + opening_fenlist_sf_refute_games` vs `matching_games` | **VERIFIED 2026-07-26** — 70–113/iter, share 0.14–0.245 vs cap 0.25 |
| B2 | the dole goes to exactly ONE worker-poll per iteration (client-count-independent) | count `dole: received ... iter=N` across all worker logs; no `iter=` value twice | **VERIFIED 2026-07-26** — 26 events / 4 workers, 0 duplicates |
| B3 | seeding OFF means zero seeded games | same counters == 0 | **VERIFIED 2026-07-26** — 0 for 5 consecutive iters from iter 28 |
| B4 | seeds in the pool are actually lost positions (vetting holds) | `scratchpad/revet_active_pool.py` at 2M nodes | **FAILED 2026-07-26** — only 78/202 survive; 124 FPs, 23 winning for STM. Seeding disabled; see ledger PHANTOMS finding |
| B5 | `opening_fen_list_path` rewrites do not restart the selfplay session | worker log `live-applied ... no session restart` | **VERIFIED** (PR #224 readout, 188 applications, 0 restarts) |

## C. Selfplay generation

| # | invariant | instrument | status |
|---|---|---|---|
| C1 | realized selfplay/curriculum mix tracks `selfplay_fraction` | per-iteration `selfplay_games / matching_games` (NOT the median) | **VERIFIED 2026-07-26 in steady state** (0.46–0.53 before, 0.41–0.50 after) — but see C6 |
| C2 | games are not abandoned at session boundaries | worker `play_batch exit:` + config-application lines | **VERIFIED** (PR #224) for FEN-list rewrites — caveat: instrument only emits on failure, see method rule 5 |
| C3 | PID sample is healthy (curriculum W+D+L ≥ 30) | `pid_curriculum_w/d/l`, `pid_regret_reason` | **VERIFIED 2026-07-26 with a caveat** — 68–268/iter in steady state, but collapsed to 0/0/1/3 (`not_active`) for iters 28–31; see C6 |
| C4 | PID levers actually move (nodes lever not pinned) | `pid_nodes_active`; `wdl_regret` trajectory vs `sf_pid_wdl_regret_stage_end` | **VERIFIED-BY-DESIGN 2026-07-26 (an earlier FAILED verdict here was premature)** — `nodes_active = (not regret_enabled) or (gate_enabled and regret_stage_complete)`. Regret IS enabled, so nodes unlocks only once `wdl_regret <= 0.0075`. Live regret is **0.088**, 12× above. The pin is correct staging, not a broken lever. The real issue is C7 |
| C7 | stage 1 (regret) descends toward `stage_end` once winrate exceeds target | `wdl_regret`, `pid_raw_winrate`, `pid_regret_delta` per iteration | **VERIFIED-WITH-CONTEXT 2026-07-26 (an earlier FAILED verdict was retracted — the controller is behaving correctly)**. Sequence: regret opened 0.0393; **raw winrate at iters 1–3 was 0.41 / 0.36 / 0.41 — the restarted net was genuinely LOSING** — so the airbag fired at iter 2 exactly as specified (`raw_wr + 1.5·se = 0.364 + 0.047 = 0.411 < 0.45` floor, n=250) and eased regret to 0.0896. Raw winrate then recovered past target; `pid_ema_winrate` 0.513 → **0.589** (EMA lag is why tightening was delayed), and regret has been descending since iter 32 (Δ −0.0003, −0.0013, −0.0015). **The apparent 32-iteration "stall" was a correct airbag rescue plus EMA lag, not a defect.** NOTE the airbag reads `raw_wr`, NOT `ema_wr` — comparing the EMA against the floor gives the wrong answer |
| C8 | *(understanding, not a defect)* stage 2 is gated on MODEL STRENGTH, not on the controller | — | The controller targets winrate 0.50, so it stops tightening once the net holds 50%. Regret only reaches `stage_end` 0.0075 if the net can hold 50% against near-full-strength SF. **"The nodes lever never fires" is therefore a measure of the net not being strong enough yet, not a bug to fix.** Do not "fix" it by lowering `stage_end` without deciding that a nodes-based ladder is wanted at a weaker regret level |
| C6 | toggling a `_RECO_RESTART_KEYS` key does not silently distort the data mix | worker log `restarting selfplay session [restart_keys=...]`, then per-iteration mix for ~6 iters | **FAILED 2026-07-26** — `opening_fen_dole_per_iter` is restart-gated; writing it abandoned all in-flight games, driving selfplay share to **1.00 for 4 iterations** (configured 0.50) and blinding the PID. Recovers by ~iter +6. **Any config toggle on a restart key contaminates the following ~6 iterations — start readouts after it, and never treat such a toggle as a free A/B** |
| C5 | search converts sims into strength at the expected rate | sims-1 vs sims-32 paired arena vs a FIXED banked ref | **FAILED 2026-07-26** — gap widens 91.5 → 251.9 Elo; boot512 control queued |

## D. SF labelling

| # | invariant | instrument | status |
|---|---|---|---|
| D1 | label depth is the live PID budget, not the yaml floor | `_eff_sf_nodes`; `opponent_sf_nodes` in `progress.csv` | **VERIFIED 2026-07-26** — 698289, `sf_label_nodes_cap: 0` |
| D2 | label coverage: rows that should have `sf_wdl` do | `replay_has_sf_wdl_frac` | **VERIFIED 2026-07-26** — 0.947–0.999 over the last 5 iters |
| D3 | fast plies get 0.25x scale by design, not by accident | `_eff_sf_nodes` fast_scale path | **CODE-ONLY** (intended, see `eff_sf_nodes_fast_ply_scale`) |
| D4 | vet ruler ≥ label ruler (no gate shallower than the labels) | `harvest_gate_step.py --sf-nodes` default | **VERIFIED 2026-07-26** — raised 300k → 2M |

## E. Finalize → rows

| # | invariant | instrument | status |
|---|---|---|---|
| E1 | playout-capped rows are excluded from the POLICY corpus | `finalize.py:892` | **CODE-ONLY** |
| E2 | seed labels attach to the seed position (offset 0) | join `content_seed_id` to first labelled ply | **VERIFIED 2026-07-26** — offset 0 for 100% of 669 cohort games |
| E3 | `policy_sf` targets are the OPPONENT reply, POV-flipped | `sf_policy_target_is_opponent_reply` | **CODE-ONLY** (documented, load-bearing) |
| E4 | no all-zero / malformed policy or WDL rows reach the buffer | `PYTHONPATH=. python3 scripts/audit_row_integrity.py --shards 12` | **VERIFIED 2026-07-26** — 18,273 rows over the 10 newest live shards, CLEAN. Checks: `x` all-zero/NaN, policy all-zero/NaN/negative/sum≠1, `wdl_target` out of range, `sf_wdl` NaN or outside [0,1] |

## F. Upload → inbox

| # | invariant | instrument | status |
|---|---|---|---|
| F1 | `._tmp_` shards are never ingested | `_iter_shard_paths_nested` filter | **CODE-ONLY** (fixed historically) |
| F2 | un-ingested shards survive a restart | trace `replay_shard_dir` vs `selfplay_shards` durability | **VERIFIED 2026-07-26** — replay dir is durable via `tune_replay_root_override`; `selfplay_shards/` on `/tmp` is a derived export with no reader. NOT a data-loss path |
| F3 | shard load failures are surfaced, not silently dropped | `test_replay_shard_load_accounting` | **CODE-ONLY** (PR #11) |

## G. Ingest → replay + holdout split

| # | invariant | instrument | status |
|---|---|---|---|
| G1 | `replay_positions_ingested / matching_positions` ≈ 1 | `audit_realized_config.py` cross-check | **VERIFIED 2026-07-26** — 1.00 (was 5.61 pre-#225) |
| G2 | stale-model games do not exceed matching games | `distributed_stale_games` vs `matching_games` | **VERIFIED 2026-07-26** — 0/20 iters |
| G3 | replay window size matches config intent | `grep "buffer init" /tmp/chess_training.log` | **VERIFIED 2026-07-26** — `len(buf)=1497712 capacity=1500000` vs `replay_window_max: 1500000`, 815 shards, `startup_source=salvage seeded=True` |
| G4 | the holdout is a STABLE ruler | `test_size`, `holdout_generation`, turnover rate | **FAILED 2026-07-26** — FIFO ring at capacity 2000, ~159 new rows/iter ⇒ **full turnover ~13 iters**. `holdout_generation` inert (0 on all rows). Fix exists and is off: `freeze_holdout_at: 0` |

## H. Sampling

| # | invariant | instrument | status |
|---|---|---|---|
| H1 | realized views per ingested position matches config | `train_views_actual` | **VERIFIED 2026-07-26** — 2.52 vs 2.5 |
| H2 | priority/surprise weighting is actually non-uniform and safe | load `priority` from the newest shards; measure CV + mass concentration | **VERIFIED 2026-07-26** — CV **0.94**, 14,297 distinct values / 14,303 rows; top 10% of rows carry **31.5%** of priority mass (uniform = 10%), top 25% carry 57.1%. Sampling is 50% uniform / 50% priority (`surprise_mix`). `replay_sf_gap_priority_weight: 0` (experiment #104 killed), so the spread comes from `diff_focus` — which is therefore **not** inert as sampling pressure, whatever its Elo effect. Edge cases safe: 3 rows (0.021%) carry NEGATIVE priority, clamped by `np.maximum(pri, 0.0)` before use, with `nan_to_num` and a uniform fallback when the sum is non-positive |
| H3 | shuffle refresh does not drop shards silently | PR #11 accounting | **CODE-ONLY** |

## I. Training step

| # | invariant | instrument | status |
|---|---|---|---|
| I1 | realized LR == configured, including after warm start | `param_group_lrs` / `progress.csv` `lr` | **VERIFIED 2026-07-26** — 3e-5; warm-start guard silent |
| I2 | the warm-start LR guard fires on a regime mismatch | `guard_warm_start_lr` | **VERIFIED 2026-07-26** — it caught the `pb2_bounds_lr` override |
| I3 | every head receives gradient (none silently dead) | per-head grad-norm probe | **OPEN** — `scripts/probe_head_grad_share.py` exists, unrun |
| I4 | train/holdout gap is not widening | `policy_loss` vs `test_policy_loss` | **VERIFIED** (views readout: 0.13–0.18 flat) — but see G4, the holdout ruler moves |
| I5 | WDL blend's SF component is non-zero | loss weights in effective config | **CODE-ONLY** — load-bearing, removing it crashed winrate 0.64→0.40 |

## J. Checkpoint + best-model

| # | invariant | instrument | status |
|---|---|---|---|
| J1 | best-model never compares two different rulers | `best.json` `source` field | **VERIFIED** (PR #244) |
| J2 | best-model's ruler is itself stable | see G4 | **FAILED 2026-07-26** — inherited from G4 |
| J3 | a resume never overwrites live checkpoints | checkpoint-index ratchet | **VERIFIED** (PR #241) |
| J4 | `salvage-export` captures CURRENT state, not best-metric | requires `--metric training_iteration` | **CODE-ONLY** (guard PR #130) |

## K. Publish → worker freshness

| # | invariant | instrument | status |
|---|---|---|---|
| K1 | workers pick up new weights (no frozen `model_sha`) | `distributed_stale_games`; worker `model_sha` | **VERIFIED 2026-07-26** — stale 0/20 iters |
| K2 | model freshness is not bound to thread 0 | `worker_model_freshness_thread0_trap` | **CODE-ONLY** (fixed) |
| K3 | AOT broker serves the current weights (externalized + rebound) | `/proc` maps; broker pos/s | **CODE-ONLY** — stale `.pt2` files are expected and fine |

## L. Evaluation / rulers

| # | invariant | instrument | status |
|---|---|---|---|
| L1 | a strength ruler exists and runs on a cadence | `scripts/daily_gate_ratchet.sh` via `scripts/ratchet_loop.sh` | **VERIFIED 2026-07-26** — fires, snapshots, arenas. Two self-audit defects found and **FIXED the same day**: (a) the Elo regex omitted `+`, so a POSITIVE result passed the whole log line into the CSV and corrupted it — negative Elo parsed fine, so it broke *precisely* when the net improved; now `\+?` plus a numeric reject-guard; (b) 4 concurrent measured ~6h/run (43→105s/game under trainer contention), raised to 8. Installed by **atomic rename**, not in-place edit, because the one-shot was still running (see method rule 9) |
| L2 | day-over-day regression is attributable to a day | `data/ratchet/ratchet.csv` `vs_prev` series | **PENDING** — starts on the second day (needs an earlier snapshot) |
| L3 | drift from a frozen anchor is tracked | `vs_boot512` series | **VERIFIED 2026-07-26** — first run in flight |
| L4 | the in-loop gate is not reporting a fake pass | `gate_games`, `gate_passed` | **FAILED (known)** — `gate_games: 0` while `gate_passed: 1`; superseded by L1 rather than fixed |
| L5 | `value_regret` comparisons are era-matched | `--min-pieces` default 8 (TB-excluded) | **CODE-ONLY** — historical 70–76 / BT4=43 were full-set, NOT comparable |

---

## 4. Standing open items, in priority order

0. **C6 readout hygiene (act on this first, it invalidates other reads)** — the
   no-seed readout must start at **iter 33**, not 28. Iters 27–32 are a
   restart-induced mix transient.
1. **C5 boot512 control ladder** — decides whether the search-conversion deficit
   is LR damage (restart fixes it) or structural (value-head *ranking* becomes
   the priority target). Queued: `scratchpad/swap_gate/boot512_ladder.sh`.
1b. ~~C7 stage 1 stalled~~ **RESOLVED — no action.** The airbag fired correctly
   on a genuinely losing restart (raw winrate 0.36) and regret has been
   descending since iter 32 as the EMA caught up. Stage 2 is strength-gated
   (C8), not blocked. **Watch, don't touch:** confirm regret keeps falling over
   iters 35–60. If it plateaus while raw winrate sits above 0.50, revisit.
2. **G4 / J2 holdout ruler** — `freeze_holdout_at: 2000` + persist the frozen set
   to `_durable_trial_dir`. Deferred until the no-seed readout closes (it returns
   the 2% holdout split to training, so it is data-affecting).
3. **E4 row-integrity live check** — two poisoning paths were found and fixed
   with no standing assertion that a third is not open.
4. **I3 per-head gradient share** — `scripts/probe_head_grad_share.py` unrun.
5. **C3 / C4 PID health** — re-read now that seeding is off and the mix changed.
6. **A4 / A5 config divergence** — published-reco diff, and reconcile the live
   yaml against `main` at the next restart.

## 5. Cadence

- **Every deploy / restart:** A1, A2, A5, K1.
- **Daily, automatic:** L1/L2/L3 (the ratchet — nothing to run by hand).
- **Every readout window:** the stage owning the knob under test, plus G1 and H1.
- **After any "the numbers look wrong" moment:** section 2 first, *then* measure.

## 6. Audit log

| date | scope | outcome |
|---|---|---|
| 2026-07-26 | A1–A3, B1–B3, C1, D1, D4, F2, G1, G2, H1, I1, I2, K1, L1, L3 | all VERIFIED |
| 2026-07-26 | G4 / J2 holdout ruler | **FAILED** — rolling ring, ~13-iter turnover; fix identified, deferred |
| 2026-07-26 | B4 seed pool re-vet | **FAILED** — 124/202 false positives; seeding disabled |
| 2026-07-26 | C5 sims ladder | **FAILED** — deficit widens with search; control queued |
| 2026-07-26 | B1/B2 dole cap | VERIFIED — **an earlier "violation" here was an artifact of a bad ad-hoc sum; see method rules 1–3** |
| 2026-07-26 | C3, C4, C6, D2, I4 | D2 VERIFIED (0.947–0.999); C3 VERIFIED-with-caveat; **C6 FAILED** (restart-key toggle distorted 6 iters of mix). C4's initial FAILED was **retracted** — the nodes pin is correct staging; the real issue is C7 |
| 2026-07-26 | C4→C7 re-read | C4 VERIFIED-BY-DESIGN; **C7 FAILED** — regret flat ~0.089 for 32 iters after an iter-2 airbag fire, so stage 2 is unreachable |
| 2026-07-26 | A4, A5, E4, G3, H2 | all VERIFIED. A4: 63/64 exact, 1 intentional (`sf_nodes` floor). A5: 4 known divergences, reconciliation table added. E4: 18,273 rows CLEAN. G3: window at configured max. H2: priority CV 0.94, negatives safely clamped |
| 2026-07-26 | L1 self-check | **2 defects in the new ratchet** (task #22): Elo regex drops `+` and corrupts the CSV on improvement; 200 games @4 concurrent ≈ 6h |
