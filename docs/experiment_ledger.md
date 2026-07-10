# Experiment Ledger

**Purpose.** One place that answers: what did we try, what was the verdict, on which
yardstick, and what is live-but-unread right now. Every entry records enough to be
understood a year later. Companion to `docs/eval_protocol.md` (which defines HOW we
decide; this file records WHAT we decided).

**Rules for entries** (the protocol that keeps this scientific):

1. Before an experiment goes live: write its hypothesis, the single yardstick that
   judges it, and the pre-committed kill/success threshold. Verdicts are decided by
   the pre-committed rule, not by post-hoc reading.
2. One data-affecting change per readout window when possible; when changes must
   overlap, list them in each other's *Confounds*.
3. A verdict is WORKED / FAILED / MIXED / UNREAD. "Deferred" is not a verdict.
4. When reverting, record what returning to known-good means (exact keys/values).
5. Before any big/data-affecting change: snapshot weights + optimizer + PID + replay
   window with `./scripts/train.sh salvage-export --top-n 1 --metric training_iteration
   --out data/salvage/<label>` (safe while training runs; ~2.3G per snapshot) and
   record the pool path in the entry. `--metric training_iteration` is REQUIRED for
   a current-state snapshot: the default metric (`opponent_strength`) selects the
   best-metric row, which may be an older checkpoint. Restore =
   `./scripts/train.sh salvage-restart <pool>`. Config keys alone are NOT a revert —
   the window keeps training on the changed data for ~a day.
6. **Offline outcome-gate for sampling/loss-weight/target knobs** (added 07-06,
   after #104): any knob that reshapes what/how the trainer samples or weights
   must first show ≥neutral broad value_regret on a 2-seed offline
   `retarget_retrain` arm vs a same-seed control before it gets a live window.
   The #104 dry-run checked plumbing (priority mass, ESS) but never asked "does
   one epoch of this damage value?" — the post-hoc screen answered that in a
   day, GPU-cheap, and would have saved the whole live window + recovery. Live
   windows are the scarce resource; screens are nearly free. (Extends the
   audit-first rule from training-target candidates to knobs.)
7. **Every experiment carries the arm/measurement that explains its own
   failure** (added 07-06): a dose ladder, a control arm, or a
   mechanism-specific secondary (like #104's gap_resolution) — decided at
   design time. It converts "failed" into "failed because X" (dose-flat ⇒
   wrong mechanism; on-target-improved-but-broad-regressed ⇒ capacity theft)
   and is impossible to reconstruct after the fact.
8. **Verify artifact provenance before recording a verdict** (added 07-06,
   after the signed-arm no-op): a result file proves a job RAN, not that it ran
   the intended code. Before a verdict lands in this ledger: check the
   producing log/timestamps chain, and for code-flag experiments confirm the
   executing tree actually contains the flag (`grep` it). The no-op "signed"
   arms recorded their override in the report json while silently ignoring it
   — only code inspection of the worktree caught it.
9. **Kill verdicts are state-conditional** (added 07-07): mid-run and
   offline-from-checkpoint tests measure "this change, applied to THIS run's
   co-adapted weights, on this horizon" — they carry a status-quo advantage
   (the checkpoint sits in the groove production config dug; short retrain
   horizons overweight re-adaptation cost vs slow-compounding benefit). The
   in-house proof this bias is real: v3 input-plane ADDS failed AND v2_lean
   REMOVALS failed (ablation-reliance ≠ training-value). So record kills as
   "dead for this run", not "dead as an idea"; ideas with strong external
   priors (e.g. KataGo/LC0 validated surprise weighting FROM SCRATCH) go on
   the from-scratch retest list for any future from-scratch run (105M /
   default.yaml). Escalation tools when a mid-run verdict feels
   state-confounded: from-scratch mini-net A/B (June 8-net ablation infra),
   or a 3× longer offline arm. NOTE the diagnostic asymmetry: a change whose
   TARGET improves while the broad distribution regresses (#104's signature)
   is real reallocation, not adjustment cost — that kill stands on mechanism,
   not just on level.

## Mechanics: changing a PBT-pinned key (e.g. `lr`) on a running trial

A plain YAML edit + restart **silently no-ops** for PB2-searched keys (the resume
path preserves them). The working procedure (used successfully twice, 2026-07-01
and 2026-07-02):

1. YAML: set the key AND its `pb2_bounds_<key>` (pin both bounds to the new value).
2. `./scripts/train.sh stop` (or pause at a boundary first via `pause.txt`).
3. Edit the resume-target experiment_state json — the **newest by FILENAME
   timestamp** under `$WORK_DIR/tune/` (mtime is useless: stop touches all of
   them) — setting the trial's `config.<key>` (+ bounds) with an atomic,
   parse-verified write. Back the file up first.
4. `./scripts/train.sh start` (auto-resumes). The harness hotpatches scheduler
   bounds from YAML; the JSON edit is what changes the trial's live value.
5. VERIFY on the first new result row (e.g. `peak_lr`) — this is the step that
   catches a silent no-op.

## Revert points (salvage pools)

| snapshot | state captured | restore |
|---|---|---|
| `data/salvage/prechange_20260702_ckpt479` | iter 479, 2026-07-02 evening: post-uncap, rung-1 live ~2 iters, PRE fast-ply-revert / PRE LR-decision / PRE #104. 686 shards | `./scripts/train.sh salvage-restart data/salvage/prechange_20260702_ckpt479` |
| `data/best_pools/` (various) | older best-regret pools, pre-June-17 run | pick a pool from `./scripts/train.sh best-list`, then `./scripts/train.sh salvage-restart data/best_pools/<pool>` |

## Yardsticks (canonical protocols)

| yardstick | command / protocol | known-good anchor |
|---|---|---|
| Policy strength | `PYTHONPATH=. python3 scripts/audit_targets.py --checkpoint <ckpt> --sims 32 --batch-size 64 --gpu-mem-fraction 0.15 --max-positions 2000 --seed 0 --sf-effort low` — net+search E[regret] and **raw top-1** (teacher-sensitive) | 49.6 / 51.5 cp @ckpt457 (2026-07-01) |
| Value ranking | `PYTHONPATH=. python3 scripts/value_regret.py --checkpoint <ckpt> --max-positions 2000 --batch-size 128 --gpu-mem-fraction 0.15` — 1-ply deep-SF regret on the **canonical "v1-2k" subset**: the deterministic first-2000 rows of the sorted audit file (endgame+middlegame strata, NO openings — a biased slice of the full 4000 but the SAME rows every run, and every anchor in this ledger uses it, incl. the BT4 probe). Do NOT compare v1-2k levels against full-set runs; re-anchor on the full set only at the next baseline bank, running both once to bridge. NOT Brier/ECE (fooled by calibration) | 72.4 cp @ckpt457; BT4 reference **43.0** (2026-07-02) |
| Blind-spot panel | `PYTHONPATH=. python3 scripts/blindspot_panel.py --checkpoint <ckpt>` — frozen 35 Cheese-loss collapse positions (`data/blindspot_panel_v1.jsonl`); counts positions the net reads "fine" where deep SF says lost | 21/35 blind @ckpt477-478 |
| Real-game strength | `PYTHONPATH=. python3 scripts/arena_standard.py` paired openings, or external engine match | vs full Cheese: 0 wins / 35 real losses (Jun 21, ckpt150-era) |
| Progress-over-time | low-sim cross-checkpoint arena (current vs pinned older ckpt) | not yet cadenced — see Queue |

Copy checkpoints OUT of the tune dir before long audits (Ray prunes live
checkpoints). All three GPU yardsticks are safe concurrent with training at the
listed batch / mem-fraction settings.

**Paired CIs are mandatory for A/B verdicts (2026-07-02).** Both audit scripts
take `--dump-per-position`; judge any two reads with `scripts/paired_compare.py`
(value dumps: defaults; audit dumps: `--join-key key --field cand.raw.top1`,
`cand.search.exp`, ...). Measured paired 95% CI half-widths @n=2000 (2026-07-02
retro-read): value top-1 ±8.7cp, audit raw E[regret] ±4.3cp, audit search
E[regret] ±6.6cp (includes Gumbel re-search noise), audit raw top-1 ±9.8cp
(72% exact ties, so real shifts in the non-tied tail still clear it). Fixed
±2cp thresholds sit BELOW every one of these — a verdict needs the CI to
exclude zero, not a point delta. Same-checkpoint re-runs also move the point
numbers (search E[regret] ~1.3cp, raw metrics ~0.7cp — CUDA jitter flips
near-tie argmaxes): never compare point numbers from different sessions' runs;
always re-dump and pair.

**Protocol gotchas** (each cost us a bad reading once):
- `audit_targets` MUST pass `--max-positions 2000` — the audit set grew to 4000
  positions including an easy opening bucket; uncapped runs are not comparable to
  the ledger numbers (caught 2026-07-02).
- Live dashboard signals (winrate, sf_move_acc, SF nodes) are flat **by design**
  (PID pins winrate); they never show learning.
- The PID winrate sample is survivorship-biased whenever SF-pool congestion changes
  (label-node changes do this): finished-game count moves, measured winrate moves
  without strength changing. Judge difficulty by regret + nodes + n_games together.
- Match PGNs must be checked for `Termination "time"` — one 19-loss match (c8192,
  Jun 21) was 100% clock forfeits, not chess.
- **The live yaml is part of the git working tree — a branch switch can silently
  revert live experiments.** 2026-07-02 incident: the live-config edits were
  committed on a PR branch; checking out main reverted the on-disk yaml, and the
  running trial (which re-reads it every iteration) re-capped labels, re-enabled
  fast-ply rows, and rolled back rung 1 for iters 484–486 before it was caught.
  Rule: while a run is live, do ALL branch work in `git worktree` checkouts —
  never `git checkout` in the run's tree — and merge any PR that touches the live
  yaml promptly so main matches the live intent.
- `record_fast_ply_value` propagates to selfplay workers LIVE (~30 min), it is
  NOT restart-gated (proven by the same incident: shards flipped to 100%
  has_policy after the revert restart, then back to 25% within ~30 min of the
  stale yaml appearing, no restart in between). Treat every `selfplay.*` recording
  knob as live unless proven otherwise.

## Verdicts: WORKED (in production)

| date | change | evidence |
|---|---|---|
| 2026-04-17 | multipv=20 + label_smooth 0.05 | broke the 0.32 winrate plateau |
| 2026-04-17 | learnable log_temp per policy head | fixed 1/√d logit squashing (~640-iter policy-loss plateau at 2.5 nats) |
| 2026-04-17 | zclip_max_norm 1.0→5.0 | 1.0 was clipping 100% of steps (raw median 2.2) |
| 2026-04-17 | SF component of the WDL blend confirmed load-bearing | removing it crashed winrate 0.64→0.40 in 4 iters (reverted 52ab9c0) |
| 2026-06-17 | v2_threats input planes (175) + 12th layer, live via offline-recovery salvage | offline win in both frozen windows; v2 uniquely makes search scale (+0.016 @256 sims vs v1 flat) |
| 2026-06-23/24 | sf_p0 policy teacher + regret-weighted SF teacher (PR #78) | **the June policy win**: net+search E[regret] 56.7 → 49.6 cp |
| 2026-07-02 | broker batch_wait 5ms + arrival-adaptive coalescing (PR #103) + selfplay_batch 384 | +88% inference pos/s (59→239 pos/batch), live-verified |
| 2026-06 | UCI/match search tuning: c_scale 0.025 (+301 Elo @8k sims), root-log value transform (PR #84) | match-play settings; RL-side effect unverified |

## Verdicts: FAILED / SHELVED

| date | change | evidence / lesson |
|---|---|---|
| 2026-06-07 | variable-width embed chain | +0.045 eval_loss, 18% slower — conclusive negative |
| 2026-06 | v3 feature adds AND v2_lean removals | both directions negative; v2 planes are a local optimum. Lesson: ablation-reliance ≠ training-value |
| 2026-06 | dynamic board-relation bias | offline ≡ threat planes; shelved |
| 2026-06 | targets pivot (categorical sf-blend, w_sf_move upweight, soft-policy knobs) | all negative/neutral; w_sf_move upweight made PLAY worse (sf_policy is the opponent-reply target, not a move teacher) |
| 2026-06 | value-knob sweeps (w_wdl, sf_wdl_frac 0.15, etc.) | all null; sf_wdl_frac 0.15 harmful — never starve the SF anchor |
| 2026-07-02 | fp8 PTQ inference | quality-dead at PTQ; needs bf16-vs-fp32 noise baseline first |
| 2026-06 | policy_temp>1 for play | helps puzzles, hurts real audit (55→88cp) — never tune search on puzzles alone |
| **2026-07-01→02** | **throughput triple, leg 1: `sf_label_nodes_cap`** | **two-stage kill.** 150k: policy 49.6→55.3, raw top-1 51.5→60.7 at ~60% refill → raised to 400k. 400k: full-refill read @ckpt478 52.2 (>2cp over baseline) and raw top-1 61.7 never recovered → **uncapped (0)** 2026-07-02. The sf_p0/regret policy teacher needs full ~700k-node labels |
| 2026-07-01→02 | throughput triple, leg 2: `record_fast_ply_value` (75% value-only rows) | **REVERTED 2026-07-02** (bundle restart, protected by ckpt479 pool). Evidence: raw policy top-1 stuck +10cp over baseline through the whole window (trunk dilution — value-only rows crowd the policy gradient) and value 72.4→76.6. Leg 3 (`train_views_per_position 2.5`) KEPT — proportional step budgeting, not implicated. Net throughput-triple verdict: quality-negative; only the views bookkeeping survives. **Statistical post-mortem (07-02 late, paired retro-CI 457→478):** raw top-1 +10.95 [+1.49, +21.18] CONFIRMED (endgame-driven, +17.9 [+6.8, +31.3]); search E[regret] +3.9 [−2.5, +10.6] and value +1.1 [−7.6, +9.9] NOT significant — raw top-1 was the only statistically load-bearing kill signal; the kill stands on it |
| 2026-07-01→02 | flat lower LR (0.0003 sqrt_release → 0.0001 flat, PB2-pinned) | **UNREADABLE — reverted 2026-07-02.** Its watch criterion ("must not slow the policy downtrend") was made unjudgeable by the throughput triple landing the same day. Reverted to 0.0003/sqrt_release in the same bundle restart so the recovery read targets the exact known-good recipe. Lesson: simultaneous launches destroy both readouts. (Mechanic for changing PB2-pinned params: see memory `flat_lower_lr_experiment_live` — yaml alone silently no-ops) |

## LIVE, UNREAD (as of 2026-07-02 evening — every open loop)

| change | live since | readout & rule |
|---|---|---|
| **Server-doled FEN seeding ACTIVATED — switch prob→dole** (`opening_fen_dole_per_iter: 1`, `opening_fen_prob` 0.05→0, live yaml; code = **PR #128** merged, restarted onto `54f4037` @iter 690 resuming ckpt686) | 2026-07-08, ~iter 690 | **Mechanism switch, not a new seed set** (same 114-seed v3 list). Replaces the probabilistic `opening_fen_prob` draw with the deterministic server dole: each iteration the server hands the WHOLE list to one worker poll (durable per-iter gate), played as **selfplay slots only** (`resolve_slot_opening` dole branch → PID-safe by construction; the prob path's curriculum seeding is what forced the 0.22→0.05 dial-down). Dose 1 = each of the 114 seeds played once/iter ≈ **114 selfplay seeded games/iter** (vs the prob path's ~35 mixed games/iter, ~3–27 of them selfplay). **Two coupled effects of the single switch (the confound):** (a) ~3–10× MORE selfplay seed volume = more direct value-label injection; (b) curriculum refutation games (net on the blundering seat, 698k-SF punishes on-board) → **0** — dole is selfplay-only, so the on-board-refutation signal that co-drove the WORKED verdict is dropped. Threaded-path delivery wired + verified (production is `distributed_worker_threaded`; #128 round-3 P1). Baseline @ckpt686 (== the FEN-seeding-WORKED trunk; ledger baseline ckpt683 value 74.3, v1 16/35); banked revert trunk `data/salvage/pre_dole_ckpt686_20260708`. PRIMARY: held-out panel v1 BLIND severity holds/improves vs 16/35 (monitor `scratchpad/live_read/monitor/monitor.log`). SECONDARY: 68 vetted seeds resolve (become AWARE) via `scripts/blindspot_resolution.py`. GUARDRAIL/KILL: `value_regret` (`--max-positions 2000`) paired CI vs the ckpt686 baseline dump worse by >2cp (CI excludes 0 worse side) → REVERT. WATCH: PID sample — dole is selfplay-only so `sf_pid` games should STAY healthy (the whole point vs prob); if it still drops <30 or the airbag trips, that's an unexpected finding. **KILL ACTION (live, instant): `opening_fen_dole_per_iter: 0` + `opening_fen_prob: 0.05` → restores the known-good prob mechanism.** Readout ~1 day (~iter 725). Supersedes the prob-based row below (that mechanism is now off). |
| **Harvested blind-spot FEN seeding — v1→v2 SCALE-UP, dose 0.02→0.22** (`opening_fen_list_path`→`data/blindspot_fens_v2.txt`, `opening_fen_prob` 0.22, live yaml) | 2026-07-08, ~iter 683 | Scales the WORKED FEN lever (v1 verdict in the WORKED table): v2 = v1's 76 seeds + **68 deep-SF-VETTED** live-harvested blind spots. Gate: `scripts/blindspot_deepsf_gate.py` on 157 banked severe seeds → 4M nodes + 6-man syzygy TB (`data/syzygy_3-4-5-6`), keep deep-LOST → **68 pass (44%), 87 deep-FINE FPs dropped (56%)**. The gate is LOAD-BEARING: calibration (`scratchpad/harvest_fp/`, deep SF converged 4M=8M=16M, TB-invariant) showed the raw severe band is **~70% false positives** (deep SF agrees with the NET, not the ~700k in-loop label) — feeding the raw `.severe` file would re-label correct positions "lost" via the shallow in-loop SF and UN-TRAIN the net. 0 overlap with held-out panel v1 (generalization metric stays clean); 0 dup with v1. Dose 0.22 targets ~1 game/seed/iter (665 g/iter ÷ 144 seeds) — **11× the validated 2%**, poison-safe ONLY because deep-vetted. Baseline @ckpt~683 (value 74.3, v1 16/35, from the monitor). REVERT TRUNK banked: `data/salvage/pre_v2feed_ckpt678_20260708` (ckpt678 trainer+pid+rng — NOTE: `salvage-export --metric training_iteration` misfired to iter 449 with ~40 stale experiment_state files present, so this is a direct checkpoint copy; nuclear revert = dose→0.02 + hand-built pool from this trunk via the offline-recovery playbook). PRIMARY: held-out panel v1 BLIND severity holds/improves (monitor `scratchpad/live_read/monitor/monitor.log`). SECONDARY: the 68 vetted resolve (become AWARE) via `scripts/blindspot_resolution.py`. GUARDRAIL/KILL: `value_regret` paired CI vs the ckpt683 baseline worse by >2cp → revert dose to 0.02. WATCH: seeded curriculum games are excluded from the PID sample → a 0.22 dose could starve it and trip the airbag (known failure mode `curriculum_starvation`) — if `sf_pid` games <30 or the airbag fires, dial the dose down (`scratchpad/harvest_fp/feed_watch.log`). Dose is live-reloadable — revert instantly. Harvester code (full-game save, one-per-game cap, `ply=` stamp) = **PR #125**. Confound: overlaps the v1 seeds' continued exposure (v1 largely learned already, panel healed 23→16) — v1's isolated verdict is banked, so this reads as the incremental harvested-seed effect. |
| **Return-to-known-good bundle restart** (fast-ply revert + LR revert + #104 code, knobs off) | 2026-07-02 evening | **THE active readout**: over the next window refill (~1-1.5 days), policy net+search / raw top-1 vs **49.6 / 51.5** and value vs **76.6→toward 72.4** (protocol: `--max-positions 2000`). Recovery ⇒ throughput-era damage was config, not permanent; no recovery ⇒ window legacy or trunk state → consider salvage-restart from a clean pool. LR revert applied via the PBT-pinned mechanic (above) and VERIFIED: peak_lr=0.0003 on iter 482. Confound: rung-1 fracs (below) remain live by design. **Contamination (07-02 ~21:17–23:5x, iters 484–486): the branch-switch incident (see gotchas) reverted the yaml — fast-ply rows back ON (~150k value-only rows in those shards), labels re-capped 400k, blend back to 0.35/0.35 — restored 23:09; fast-ply-off re-propagation VERIFIED (shards 23:57–00:02 back to 100% has_policy). Recovery-read clock effectively restarts at iter ~487; interpret ckpt510-era reads with this in the window.** **Pre-committed recovery rule (added before the readout):** each read dumps per-position and pairs vs the banked ckpt457 dumps (`scratchpad/policy_ci/`). RECOVERED = raw top-1 paired CI vs ckpt457 includes 0 AND endgame-value paired CI includes 0 → bank fresh baseline dumps, proceed to #104 activation. NOT RECOVERED = raw top-1 CI still excludes 0 on the worse side after TWO window refills → stop and rebase: hand-build a salvage pool from the banked ckpt457 trainer + the then-clean replay window (offline-recovery playbook, memory `v2_threats_12layer_live_offline_recovery`) and salvage-restart. Do NOT rebase onto `prechange_20260702_ckpt479` — that pool carries the damaged trunk. **REFILL CLOCK RECALIBRATED (07-03 AM, live telemetry): window = 1.41M rows (not the 3M cap — still growing), turnover ~16.1k rows/iter at ~39 min/iter → ONE refill ≈ 87 iters ≈ 2.4 days (the "~1-1.5 days" assumed fast-ply-era 4× ingest). The fast-ply era ingested ~1.3-1.45M rows ⇒ the window was ~fully contaminated at the iter-487 restart, and ~75% of those rows are value-only — policy training stays diluted until they flush. INTERIM READ @ckpt500 (iter ~501, ~16% refilled; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt500.jsonl`): raw top-1 vs ckpt457 −11.2 [−21.8, −1.7] (endgame −20.9 [−36.4, −7.7]; middlegame +8.0 [−0.6, +19.6] — borderline BETTER); value vs ckpt457 −9.5 [−17.9, −1.4] (endgame −11.5); search E[regret] vs ckpt457 −0.7 NS (search still masks the raw damage); vs ckpt478 everything NS (raw top-1 −0.3, value −2.4) — no movement yet, consistent with 16%. Blind-spot panel 19/35 vs 21/35 baseline. NOT RECOVERED, expected at this refill depth — next read ~50% refill (07-04 AM), full-refill verdict read ~07-05, two-refill rebase deadline ~07-07. **TIMELINE SUPERSEDED by the window cut (row below): contamination flush ≈ 36 iters ≈ 24h → verdict read ~07-04, rebase deadline ~07-05.** TAIL ADDENDUM (same dumps, `scripts/tail_stats.py`): the mean is tail-dominated (median value regret 9–11cp vs mean ~70–80); >300cp blowups 4.2%→5.1% (457→500); paired flips 457→500 = 48 new vs 29 fixed (net +19) — the loop mints new tail blowups faster than it fixes old ones. Add the tail line + flip counts to every future read (secondary, descriptive).** |
| Value-blend rung 1: `search_wdl_frac` 0.35→0.20, `sf_wdl_frac_floor` 0.35→0.45 | iter 477 (07-02) | value_regret per ckpt vs the **76.6 pre-anchor** (ckpt478), judged at the FULL-refill read (post-bundle window). Pre-committed rule: SUCCESS = ≤70.4 (2cp below the 72.4 known-good — evidence rung 1 adds value beyond the bundle recovery); KEEP-BUT-UNREAD = 70.4–75.0 (recovered; bundle confound absorbs credit, rung 1 stays as principled default); KILL = >75.0 at full refill → revert pair 0.35/0.35 (live keys). **AMENDED 2026-07-02 late, BEFORE the readout: the ±2–5cp point bands sit inside the value yardstick's measured ±8.7cp paired noise floor — verdict now runs on `paired_compare` vs the ckpt478 dump (`scratchpad/paired_ci_smoke/dump_ckpt478.jsonl`): KILL = CI excludes 0 on the worse side; SUCCESS = CI excludes 0 on the better side; otherwise KEEP-BUT-UNREAD. Point bands demoted to descriptive. Also: iters 484–486 trained on the old blend (incident above) — rung-1 exposure restarts ~487** |
| PID `sf_pid_regret_tighten_streak_gain` 0.5→1.0 (temporary) | iter ~478 (07-02) | controller mode, not an experiment: restore 0.5 when EMA winrate ≈0.52. Expect the winrate sample to shift again post-uncap (congestion bias). 07-03 AM: EMA 0.565 falling ~0.005/iter — restore due later today |
| **`replay_window_max` 3M→800k, deepened same morning to a 250k ONE-ITERATION cut** (TEMP, live yaml) | iter ~502-503 (07-03 AM, user-approved; deepened on the user's "start now" push) | Ops lever, not an experiment: shrink evicts oldest rows. At 800k the flush was still ~24h (580k contaminated rows retained); the 250k deep cut evicts ALL contaminated rows at the next iteration boundary (newest ~236k rows are post-incident clean) → training ~100% clean immediately. Sequence: 250k lands (watcher) → flip cap to 800k + `replay_window_growth_frac` 0.10→1.0 (clean regrowth ~16k/iter, ~34 iters back to 800k) → at recovery banking, cap→1.5M (revisit growth_frac then). Verified live-reloadable (`tc` rebuilt per iteration; shrink via `buf.enforce_window()`; replay keys absent from topology/construction-bound lists). Rationale for waiting on #104/rebalance anyway: activating a training-distribution change inside the healing window confounds BOTH the rebase gate and the knob's own verdict (false-credit risk) — with the deep cut the gate shrinks to ~this evening, so nothing meaningful is lost. Recovery verdict read: earliest meaningful ~18-25 clean-exposure iters (~this evening); rule-quality read 07-04 AM. Cost accepted: one day of small-window recency bias (safe: views-targeting fixes per-row reuse at 2.5 regardless of window size — window size only sets batch recency/diversity, and 250k ≈ salvage-restart scale); refill-based verdicts get less exposure per refill. Note: per-row reuse invariance means the cut costs training QUALITY little while maximizing clean-fraction — strictly better healing. **EXECUTED + VERIFIED (07-03): cut landed iter 504 (1.41M→248,410 rows, all clean); cap flipped back to 800k + growth_frac 1.0 same hour; regrowth confirmed iter 506+ (~13.5k rows/iter avg — selfplay-heavy iters ~15-16k alternate with curriculum-heavy ~5k); winrate EMA steady ~0.56 through the surgery. INTERIM READ @ckpt510 (6 clean iters post-cut; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt510.jsonl`): BOTH pre-committed recovery criteria formally MET — raw top-1 vs ckpt457 −4.1 [−15.4, +8.0] includes 0 (was −11.2 SIG at ckpt500), endgame value vs ckpt457 −1.9 [−10.2, +6.1] includes 0 (was −20.9 SIG). Direct healing evidence vs ckpt500: endgame value +9.6 [+0.1, +19.1] SIGNIFICANT in 10 iters; value >300cp tail flips 500→510 = 22 new vs 43 FIXED (net −21) — first read where the loop fixes tail blowups faster than it mints them; vs ckpt457 value tail now breakeven (30/32, was 48/29). Panel v1 BLIND 18/35 (best yet: 21@477, 19@500); panel v2 55/113 (flat vs 54@500). Still short of full: endgame raw top-1 point −10.4 vs 457 (NS), policy tail net +8 vs 457. Trajectory steep (+6.8cp value in 10 iters) ⇒ NOT steady state — hold #104 for the confirming read @~iter 524 (watcher armed) per protocol; treat that as the banking read if it confirms. **VERDICT READ @ckpt524 (07-04 00:30, 19 clean iters; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt524.jsonl`): RECOVERED — raw top-1 vs ckpt457 −1.7 [−12.5, +10.0] and endgame value +1.3 [−10.3, +13.6], both include 0 with near-zero points; value 72.15 ≈ the 72.4 known-good; trajectory converged (510→524 deltas tiny — transient over). Rung-1 rule: 72.15 lands in KEEP-BUT-UNREAD (70.4–75.0) — recovered, rung-1's own credit inseparable from the bundle (expected). Panels: v1 18/35 (best ever), v2 47/113 (54→47 during recovery, no #104 — clean-window healing reaches the tail). Residuals (noted, not rule violations): policy >300cp tail net +10 vs 457; search E[regret] −5.8 [−12.5, +0.4] borderline NS. BANKED: ckpt524 = the new baseline; cap 800k→1.5M; #104 ACTIVATED (row below). Throughput-triple recovery arc CLOSED.** |
| Blind-spot panel v2 — **BUILT, FROZEN 07-03** (`data/blindspot_panel_v2.jsonl`, 113 rows) | 07-03 AM | Mined the June 15-16 iter520 full-Cheese + June-1 rofChade losses with the v1 method (300k-node SF, first decisive collapse, time-forfeits excluded; session `blindspot_mine_v2.py`): 86 losses profiled → 78 new rows + v1's 35, deduped to 113 (74 iter520-Cheese / 35 ckpt150-Cheese / 4 rofChade). **ckpt500 v2 baseline: BLIND 54/113 (47.8%), AWARE 43/113** — rate consistent with v1's 19/35; binomial CI half-width ~9% vs ~16%. v1 stays THE frozen panel for #104's pre-committed rule (≤16/35 / ≥20/35); v2 pre-committed secondary thresholds vs the 54/113 baseline: SUCCESS ≤ 44/113, KILL ≥ 60/113 (≈ ±2σ binomial). **LOSS-MODE ANALYSIS (user confound challenge answered — "is the newer tune flakier?" NO):** same-opponent cross-net comparison: iter520 (OLD run) vs Cheese = 100% decisive-collapse, 81% one-error dominance, 86% mate-scale — at least as collapse-y as ckpt150 (NEW run: 100%/57%/77%). g465 vs rofChade = 33% collapse, median ZERO ≥150cp drops (genuine grinds, median 32 plies). Collapse mode tracks OPPONENT (Cheese ~3000, balanced-until-blunder) not tune; grind is the expected mode vs the stronger rofChade (~3300). **Open cell: current net has NEVER played rofChade (only g465 did) — queue a ~20-game match at the next training pause (GPU-bound)** **SEARCH-vs-VALUE DECOMPOSITION @ckpt510 (07-03 evening; `scratchpad/policy_ci/blindspot_search_probe.py`, dump `searchprobe_v2_ckpt510_s200.jsonl`): at fen_before with production PLAY-shape Gumbel @200 sims, the net STILL picks the recorded losing move on 23/113 (search) vs 25/113 (raw argmax) — ~7× the ~3% random-match base rate, and search provides ZERO net rescue (balanced-before subset sf_before≥−50, n=37: raw 7/37, search 7/37). Value-awareness does NOT prevent the choice: AWARE rows get picked at the same ~20-24% rate as BLIND rows (specimen: d3d7, sf −11→−2992, net_after −0.49 AWARE, root_q −0.24, search plays it anyway) — Q-propagation/one-ply incoherence, not just after-position misread. Implications: (1) selfplay-sims search inherits the blindness → the loop keeps generating+reinforcing these lines (mechanism behind 'loop does not self-correct tail errors'); (2) #104 gap-priority targets the 56 value-BLIND rows but NOT the aware-but-picks mode — the improved-policy target itself is wrong there; the lever for that mode is blind-spot FEN seeding (games FROM fen_before so deep outcomes refute the move). Caveats: 200 sims ≪ match sims (match-level rescue untested — the queued rofChade/Cheese match is the test); panel move ≠ only losing move (rescue rates are upper bounds). Panel `move` field is SAN, not UCI — parse before comparing (first probe run scored 0/113 on exactly this). **SHAPE + SIMS SWEEP (same evening, logs `searchprobe_v2_ckpt510_{selfplay,play}_s*.log`): picks-losing-move = raw 25/113; selfplay shape @256 (c_scale 0.1 linear, topk 16) 19/113; play shape @256 (c_scale 0.025 root-log, topk 32) 22/113; play @3200 22/113 — 16× search rescues ZERO net positions. Composition flips with depth: @3200 BLIND-row picks RISE 7→11/56 while AWARE-row picks fall 14→9/46 — deep search converges to the value head's beliefs, fixing aware-mode errors and ENTRENCHING value-blind ones. The ~20% floor tracks the value head exactly ⇒ match-depth search is NOT protective; the play shape's lower value-trust is WORSE on AWARE rows @256 (14/46 vs raw 9/46 — down-weights the head where it was right). VERDICTS: (a) do NOT flip selfplay search to the play shape for tail reasons — no offline signal, and the tail is a data problem; (b) the blind-spot fix must come from training data (FEN seeding, #104), not search knobs; (c) play shape stays for matches (its +301 Elo is on-distribution). NEXT BUILD: FEN-seeding feature — `selfplay/opening.py` only supports startpos move-sequence books; needs a flag-gated FEN-list book type (defaults off) so curriculum-vs-SF games start from panel fen_befores and SF refutes the losing moves on the board** |
| Blind-spot FEN seeding — **BUILT (PR #108), flags default-off** | 07-03 evening | Starts selfplay/curriculum games FROM historically misplayed positions (the probe sweep's verdict: search can't rescue these at any depth — only data can). Feature: `opening_fen_list_path`/`opening_fen_prob`/`opening_fen_net_side_to_move` (FEN-list opening source, net forced to the blundering seat, full server→worker distribution mirroring the books, `fenlist` opening-source telemetry). **Seed asset = 78 FENs (panel v2 MINUS v1): panel v1's 35 rows are DELIBERATELY HELD OUT so v1 measures GENERALIZATION (transfer to unseen collapse positions) while seeded-subset v2 rows measure direct learning.** ACTIVATION (pre-committed, do NOT activate in the same window as #104): (1) requires restart onto merged code BEFORE adding yaml keys (strict validator); natural slot = the graceful restart after #104's window; (2) starting dose `opening_fen_prob: 0.02` (~[games/iter]×0.02 seeded games/iter; each of 78 positions revisited every few iters — temperature + SF-regret variety diversifies traces); (3) readouts after ~1 day: PRIMARY = held-out panel v1 BLIND (generalization; success = clear drop from its pre-activation count, kill = rise), SECONDARY = v2 seeded-subset BLIND (direct learning — should move first; if v2-seeded moves and v1 doesn't, the fix memorizes without generalizing → reconsider dose/diversity), GUARDRAIL = value_regret paired CI vs pre-activation ckpt (±2cp) + fenlist game outcomes in stats; (4) fresh-match confirmation (Cheese) only after panels move. **REVIEW ROUND (07-04, xhigh workflow, Codex out of credits): 10 findings fixed (round-3 commit) — headline was [1] PID contamination (seeds forced the net onto the losing seat → dragged curriculum winrate down → eased SF across the whole batch; now excluded from the PID sample like selfplay games, telemetry kept). Seed asset curated 78→76 (2 forced-move FENs dropped). ACTIVATION CHECKLIST NOW REQUIRES A REINSTALL: the min_worker_version=0.0.2 stale-worker gate only fires after `pip install -e .` regenerates the installed metadata AND the worker wheel is rebuilt at 0.0.2 — a bare `train.sh restart` does NEITHER, so without the reinstall old (0.0.1) workers silently mix non-seeded shards into the run. So activation = (a) merge #108, (b) `pip install -e .` + rebuild/publish the 0.0.2 worker wheel, (c) restart onto the code, (d) add the yaml keys. FINDING [0] MEASURED (`scratchpad/policy_ci/history_impact_probe.py`, 112/113 rows reconstructed from source PGNs): FEN seeds fabricate LC0 history planes (empty move-stack → repeat-fill), but the net's OUTPUT is borderline-benign across the two encodings — top-1 move agrees 86%, median value shift 8.6cp (P90 31). Usable for a first experiment; if seeding shows promise, store each seed's ~8 preceding moves and replay them (real history, eliminates the ~14% move-flip) as the follow-up.** **ACTIVATED 07-06 (~iter 613): full checklist executed — #108 merged, `pip install -e .` + egg_info to 0.0.2, 0.0.2 worker wheel, restart onto merged main (61e5707, second restart 18:04 after #113/#115 landed + per-game telemetry #115), yaml keys added (`opening_fen_prob: 0.02`, 84-FEN asset `data/blindspot_fens_v1.txt` — 84 lines at HEAD, VERIFIED 07-06: 0 overlap with panel v1 fen_befores (held-out design intact), 76/84 exact-match v2 fen_befores, 8 near-variants from the same mining pass; the "76" cited earlier was wrong — net_side_to_move on). DELIVERY VERIFIED (iters 613–619 outcome_stats): 8–22 fenlist games START per iter (~2% of games, as dosed) and BOTH paths deliver — selfplay seeds 1–11/iter (SF value labels attach → direct value-head injection) and curriculum fenlist games FINISH despite 698k-node SF (net from the blundering seat: 7W/23D/11L over 7 iters — the refutation data). The 07-05 starvation worry did not materialize: curriculum games finish in bursts on alternating iterations (game length ≈ iteration length beat), so no SF-budget fix needed at this dose. Baseline = ckpt609 (v1 23/35, v2 70/113, value 75.86). Readout ~iter 650 per the pre-committed protocol above.** **VERDICT: WORKED (readout landed @ckpt649, held/strengthened through ckpt669 via the monitor).** PRIMARY held-out panel v1 BLIND 23→19→18→**16/35**, severity paired +0.16→+0.20→**+0.26 SIG better** vs ckpt609 (generalization to positions never seeded). GUARDRAIL met: value_regret 72–74cp, paired vs ckpt609 NS throughout (+3.88/+3.71/+1.56, all CIs include 0 — no regression). Follow-up #1 (real-history seeds) shipped as the harvester `fen \| moves` format; follow-up #2 (broader mining at scale) = the v2 scale-up row at the top of this table (2026-07-08). |
| PR #104 gap-priority sampling — ACTIVATED 07-04 ~00:45, **KILLED 07-05 @ckpt559 readout — every kill threshold crossed** (`replay_sf_gap_priority_weight` 30→0, live yaml) | iter ~526 (activation) → ~577 (revert) | **Activation baselines @ckpt524** (recovery banked the same read): panel v1 18/35, panel v2 47/113, value 72.15, raw top-1 52.5; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt524.jsonl`. RULES: v1 (frozen, primary, unchanged): SUCCESS ≤16/35, KILL ≥20/35. v2 secondary — RE-ANCHORED PRE-ACTIVATION (documented, not silent: the original ≤44/≥60 was set vs 54/113 @ckpt500, but recovery healed the panel to 47 before #104 ever ran; thresholds re-derived vs 47 with the same ±2σ logic): SUCCESS ≤40/113, KILL ≥57/113. GUARDRAIL: paired value CI vs ckpt524 worse by >2cp = kill. SECONDARY READOUT: gap_resolution rate (top-decile-gap rows resolving; surprise-bakeoff finding was they do NOT resolve at baseline). Read after ~1 day (~35 iters). Do NOT start the rebalance rung or FEN seeding in this window. (gap_resolution measured with `PYTHONPATH=. python3 scripts/gap_resolution.py --checkpoint-old <pre> --checkpoint-new <post>` — repro'd the 466→478 finding: top decile +0.0006 vs bottom half +0.0077.) **PRE-ACTIVATION DRY-RUN (07-03, 16.5k rows / 25 newest clean shards): recording works (`has_priority_sf_search_gap` on 97.7% of rows; gap mean 0.132, p90 0.32, p99 0.75 value-units). At w=30 the gap term is 31.7% of priority mass; sampling gets BROADER not narrower (ESS 0.502→0.608, top-1% mass 5.9→4.6% — the flat-ish gap term dilutes the heavy-tailed base surprise); corr(base, gap)=+0.07 → orthogonal targeting; top-decile-GAP rows' priority-half mass 12.4%→20.3% (~1.6×; overall ~1.35× after the uniform sampling half). Verdict: safe to activate, moderate re-targeting, no starvation. w=60 adds nothing (ESS/top-mass identical) — 30 is the right first rung** **VERDICT READ @ckpt559 (07-05; ~21.6h/34 iters exposure at the read, ~51 iters total before the revert landed; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt559.jsonl`, banked ckpt `scratchpad/recovery_read_ckpt559`): KILLED — every pre-committed kill threshold crossed. Panel v1 18→24/35 (kill ≥20); panel v2 47→83/113 (kill ≥57; worst point ever recorded); value_regret 72.15→85.96, paired Δ −13.81 [−22.42, −5.83] SIG, endgame-driven (−17.0 [−29.3, −6.0]) — the >2cp guardrail blown ~7×. Policy raw top-1 Δ −4.97 [−14.17, +2.31] NS ⇒ a pure VALUE regression. SECONDARY (gap_resolution 524→559, 10,718 rows / 30 newest shards): the mechanism WORKED on its own target — top-decile-gap verr 0.489→0.422 (Δ +0.0666 positive, exactly what the rule asked for) — while the bottom half WORSENED (−0.0088); with ALL-replay delta +0.0038 (fine on its own training distribution) against −13.8cp on the frozen audit set, the signature is distribution shift / capacity theft: at ~36% of priority mass (live pmass_gap_share 0.35–0.38 ≈ dry-run 0.317) the trunk reallocated toward the gap rows at the broad distribution's expense. (Alternative reading not excluded: gap rows are where search and the SF label disagree — oversampling trains hard toward SF labels precisely on contested positions, i.e., label quality, not just capacity. Either way w=30 is net-negative.) CONFOUNDS DEFEATED: window regrowth (658k→913k across the window) ran under the SAME growth_frac-1.0 retain-all regime as the IMPROVING 500→524 segment; the iter-544 restart artifact is winrate-telemetry-only; ckpt524 was not a lucky baseline (arc 500/510/524 = 79.0/72.2/72.1 value, 19/18/18 v1, 55/47 v2 — ckpt559 is the worst point on all three). REVERTED to 0 (live yaml, ~iter 577) — the pre-committed kill action; ckpt524 REMAINS the canonical baseline. LESSON (2nd instance after the w_sf_move upweight): reweighting existing rows toward hard/disagreement rows at high mass is net-negative at 46M scale even when the target rows measurably resolve; a low-w retry (w≈5, ~8% mass) is a legitimate future experiment but is DOMINATED by FEN seeding (#108) for the next window — seeding adds NEW states instead of reweighting old ones. RECOVERY WATCH: sampling weights reverted with the data itself clean, so the head should re-fit fast; read panel v1 + value_regret at ~iter 610–620 (~1 day) — significant retrace toward ~72 ⇒ continue; still significantly worse vs ckpt524 ⇒ salvage-restart from the banked ckpt524 (`scratchpad/recovery_read_ckpt524`, offline-recovery playbook). **RECOVERY READ @ckpt609 (07-06; dumps `scratchpad/live_read/vdump_ckpt609.jsonl`): rule says CONTINUE — value 85.96→75.86, paired vs ckpt524 −3.72 [−13.88, +6.02] NS overall (was −13.81 SIG at the kill read) ⇒ not "still significantly worse", no salvage. Slices are NOT symmetric: endgame still −13.70 [−27.26, −1.56] SIG worse vs 524, middlegame +16.18 [+1.75, +33.14] SIG better; panels v1 23/35, v2 70/113 (healing from 83, still above the 47 bank). The residual endgame/tail damage is exactly the blind-spot mode → handed to the FEN-seeding window with ckpt609 as its pre-activation baseline (banked: value 75.86, v1 23/35, v2 70/113).** |
| sf_p0 + regret teacher weights | June | proven (see WORKED); leave alone |

## Analysis findings (offline, no live change)

**Native C LTO build experiment — FAILED (2026-07-09).** Hypothesis:
link-time optimization can improve the production BF16 CBoard encoder without
meaningfully slowing the frequent in-place extension rebuild. ONE deciding
yardstick: `PYTHONPATH=. python3 scripts/bench_native_build_flags.py --modes
native native-lto --rounds 3 --samples 7`. The script alternates build order,
forces every rebuild, compares exact float32/BF16 output hashes, and reports
median build time and encoder throughput. Pre-committed SUCCESS: native+LTO
BF16 throughput is at least 1% above native, all output hashes match, and its
median forced-build time is no more than 25% or 3 seconds above native
(whichever allowance is larger). Otherwise FAILED; float32 throughput is a
secondary diagnostic only. This changes no training data or live process, so
no salvage snapshot is required. An exploratory pre-registration probe showed
roughly +1.4% BF16 and +4% float32 throughput; it is explicitly not the verdict.
**VERDICT: FAILED by the pre-committed production-BF16 rule.** The independent
registered run produced native+LTO/native ratios of **0.951 BF16** and 1.025
float32; exact hashes matched. Median forced-build ratio was 0.883, so build
time was not a problem, but the deciding BF16 path regressed about 4.9% and
showed high run variance. Do not enable LTO. The run also exposed a separate
native-build correctness bug: standalone `_features_ext` took an AVX2 path
whose plane helpers existed only in `_cboard_impl.h`, yielding an undefined
symbol at import. That issue is fixed separately; it does not change the LTO
verdict.

**Native C LTO long-window replication — WORKED (2026-07-09).** The first
registered LTO experiment retains its FAILED verdict, but its individual
20-iteration timing windows were only about 20-30ms and showed extreme
external-load variance (native BF16 155k/180k/181k pos/s; LTO
231k/118k/172k). Hypothesis: longer, CPU-pinned measurements can resolve
whether the exploratory small LTO gain was real. ONE deciding yardstick:
`PYTHONPATH=. python3 scripts/bench_native_build_flags.py --modes native
native-lto --rounds 5 --samples 7 --iterations 200 --cpu 15`. Pre-committed
SUCCESS remains: native+LTO median BF16 throughput >=1.01x native, exact output
hashes match, and median forced-build time is within the larger of 25% or 3s
of native. Otherwise FAILED. Build order alternates; each reported sample now
times about 10x more work on a fixed CPU. No live/training change or snapshot.
**VERDICT: WORKED.** Native+LTO/native ratios were **1.082 BF16** (the
production path), 0.944 float32 (secondary), and 1.059 forced-build time;
exact hashes matched across all ten builds. Median build time rose only
7.94s -> 8.41s (+0.47s). This clears both pre-committed gates, so
`CAE_EXT_LTO=1` is enabled as an explicit option and recommended together with
`CAE_EXT_NATIVE=1` for local production BF16 builds. It is not a portable-build
default because float32 regressed and LTO toolchain support varies.

**Native C PGO experiment — FAILED (2026-07-10).** Hypothesis: a profile
trained on the production-native mix (BF16/float encoders, move generation and
board mutation, standalone features, fused per-ply processing, and MCTS
selection/backprop/WDL conversion) lets GCC improve hot branch layout and
inlining beyond native+LTO. ONE deciding yardstick: `PYTHONPATH=. python3
scripts/bench_native_build_flags.py --modes native-lto native-lto-pgo
--rounds 3 --samples 7 --iterations 200 --cpu 15`. The PGO mode performs a
clean instrumented build, runs `scripts/train_native_pgo.py`, rebuilds from the
collected profile, then executes the same CPU-pinned benchmark as the baseline.
Pre-committed SUCCESS: the geometric mean of BF16 encoding, legal movegen, and
WDL conversion throughput is >=1.02x native+LTO; no deciding component is below
0.98x; exact float32/BF16 output hashes match; and the PGO build/import completes
without missing-profile warnings. Otherwise FAILED. Float32 encoding is a
secondary diagnostic because production uses BF16. No live/training data change
or salvage snapshot.
**VERDICT: FAILED.** Native+LTO+PGO/native+LTO ratios: production geometric
mean **0.977**, BF16 0.959, WDL conversion 0.967, float32 0.969, movegen
1.004. Exact hashes matched, every extension produced and consumed profile
data, and the strict use build had no missing-profile warning, so this is a
performance failure rather than broken plumbing. Do not expose PGO as a
production build option for this profile. The reproducible scripts remain for
future experiments with materially different real-workload profiles; any such
retry needs a new pre-registered ledger entry.

**Gap-priority offline dose screen — the abs() family is DEAD; TRUE signed
test in flight (07-05→07-07; `scripts/retarget_retrain.py` arms from ckpt524
over 200 frozen shards, 2 seeds; artifacts `scratchpad/dose_screen/`).**
Value_regret paired vs the same-seed w0 arm: w5 −8.60 [−17.55, +0.17] /
−4.05 [−13.30, +4.90]; w30 −8.22 [−16.90, +0.24] / −6.47 [−16.05, +2.80];
plus two more w30-class replicates (see provenance note) −10.37
[−20.07, −0.95] **SIG** / −5.46 [−14.97, +3.67]. Panels v1: w0 23, w30-class
20/21/22/27 — no arm helps the blind spots. Per-arm CIs mostly borderline-NS,
but **6/6 arms damage in the same direction** (screen validity: the w30 arms
reproduce the live #104 damage direction, as the queue rule required).
**PROVENANCE CORRECTION (07-06 late, code-inspection verified):** the arms
named `w30sgn_*` are NOT signed — the overnight driver ran from the
wt-dosescreen worktree, which contains ZERO occurrences of
`sf_gap_priority_signed`; the override was recorded in retarget_report.json
(`arms/retarget_report_noop_w30sgn.json`) but silently ignored (TrialConfig
drops unknown keys). They are two extra w30-abs replicates — which is why
w30sgn≈w30 (−2.14/+1.00 NS). The 07-06 12:53 "fixed" driver then deadlocked
on a gate waiting for `vb_lessSF_s1.pt` (never built — its upstream driver was
killed) and died at the 18:02 restart having trained NOTHING. **The signed
variant (PR #113, `max(search_sig − sf_sig, 0)`) has never been tested.**
VERDICTS: (1) the low-w retry is DEAD — damage is ~flat in dose (w5 ≈ w30),
i.e. wrong mechanism, not wrong dose; (2) 3rd reweighting failure for the abs
family (after w_sf_move upweight, #104 live) ⇒ reweighting existing rows
keeps losing at this scale — new-data levers (FEN seeding) favored;
(3) signed verdict IN (07-07, true arms `w30sgn2_s0/s1`, provenance
CONFIRMED: main has the flag — grep 4/1/3 in disk_buffer/retarget/trial_config
— override registered in retarget_report, 4500 steps eager). **KILLED / no
better than control.** Paired vs w0: −10.00 [−18.24, −1.86] **SIG** (s0) /
−4.95 [−13.56, +3.76] NS (s1) — both point-negative, damages value like every
other arm; panels v1 23/23 both seeds (no help). Paired vs w30-abs: −1.78
[−9.13, +5.70] / +1.52 [−7.11, +10.32], both NS ⇒ **dropping the pessimistic
half of the boost changes nothing** — the directionality fix is inert. This
was the ONE untested member; the whole gap-priority reweighting family is now
CONCLUSIVELY dead (the no-op "signed" arms had accidentally previewed the same
≈w30 result). Signed-gap (#113) stays merged but OFF permanently. Reweighting
existing rows = closed chapter (3 distinct mechanisms × the improved variant,
all net-negative at 34.7M).**
ADDENDUM (07-06, same screen run): the banked **value-blend arms** read out
too — vb_moreSF (sf_wdl_frac 0.65/search 0.10) −6.20 [−15.16, +2.68] / −3.91
[−12.94, +4.97] vs w0; vb_lessSF (0.40/0.30) −3.04 [−12.71, +6.17] (s0 only;
the s1 twin was never trained — its driver was killed — and is queued in the
07-06 overnight batch). All NS, all point-negative ⇒ **no offline support for
moving the WDL blend in either direction from the live rung-1 (0.45 SF / 0.20
search)** — ladder rungs 2–4 stay parked; don't spend a live window on them
while FEN seeding is reading. Caveat on screen power: w0_s0 vs w0_s1 differ by
~7cp (seed noise), so this screen detects only large effects — "no win
detected", not "harm proven". ALSO IN THE OVERNIGHT BATCH: `vc_hi_s0/s1`
(w_categorical 0.30→1.00 — distributional-value emphasis → shared trunk →
wdl RANKING, the known deficit; the 07-05 queued screen deadlocked with the
rest of the chain). Read = paired vs w0; a win is live-reloadable (no
restart). **VERDICT (07-07 13:30): KILLED.** Paired vs w0: s0 −11.21
[−19.50, −3.13] **SIG** (middlegame-driven, −20.9 SIG), s1 −4.93 [−14.06,
+4.06] NS — both point-negative (vregret means 84.6/85.1 vs w0 73.4/80.2);
panels v1 25/35 and 27/35, worse than control. Upweighting the categorical
head steals trunk capacity like every other loss-mix move; the ranking
deficit does not respond to loss reweighting. `w_categorical` stays 0.30.
CHAINED BEHIND THE BATCH (07-06, process-liveness gate — never
artifact gates): **uniform-sampling ablation `unif_s0/s1`** — the always-on
KataGo surprise-priority sampling (hardcoded `surprise_mix=0.5`, KL term =
99.9% of shaped priority mass live) has NEVER been causally tested, and every
reweighting mechanism we HAVE tested lost. Arms train with surprise_mix=0.0
(pure uniform) via a session harness that monkeypatches the constant — it is
deliberately NOT a config key today; **if this ablation reads out interesting,
the reproducible follow-up is to promote `surprise_mix` to a real config/UCI
knob** (which also makes the experiment re-runnable from a clean checkout).
Provenance (ledger rule 8): the harness stamps a `[uniform-ablation]` marker in
its train log and the driver hard-aborts the evals if it's absent, so a no-op
patch can't masquerade as a result. Read = paired vs same-seed w0: ≈w0 ⇒ dead
machinery (simplify); worse ⇒ first causal validation of surprise weighting;
better ⇒ base-layer reweighting tax found (free value → promote the knob).

**Sidecar retro-analysis — the offline yardstick has a seed-noise floor too
(2026-07-03; all 256 `results.jsonl` under `runs/`, scanner + master CSV in
`scratchpad/{scan,analyze}_sidecars.py`, `scratchpad/sidecar_master.csv`).**
The May arch campaign ran its top config (`11l_splitqkv_mlp_out`) at 4 seeds:
eval_loss 6.301–6.377 (**σ≈0.034**), policy 2.404–2.449 (**σ≈0.019**); five
other repeat groups agree (2-seed eval ranges 0.005–0.038). Same-seed
variant-vs-baseline comparisons are partially paired (shared data order), so
treat the floor as an upper bound — but every June feature verdict sits AT or
BELOW it: v2_threats −0.039 eval/−0.015 policy, v2_lean +0.022, v3 adds
+0.003..+0.036, smolgen sizes ±0.015. Consequences, per claim:
(1) *v2_threats helps* — SOLID, not because of the sidecar delta but because it
replicated on a 2nd window (−0.028) AND has three orthogonal corroborations
(8-net ablation: attacks_typed +0.067..0.078; prescreen relevance; unique
search-scaling +0.016 @256 sims where v1 is flat). The 512-game arena
(+4.8 Elo [−22.7,+32.2]) never resolved it either way.
(2) *v3 adds hurt* — OVERSTATED. The sidecar deltas are noise-level; the solid
claim is "v3 adds ≈ 0, and the prescreen explains why" (xray/checks redundancy
0.82–0.91 = already derived from v2; see/passers relevance ≈ 0). The puzzle
bench at 256 sims (n=3000, v2 0.601 vs v3 0.570–0.583, 2–3σ binomial) is the
strongest negative — no v3 net inherits v2's search scaling.
(3) *v2_lean removal hurts* — UNPROVEN: +0.0217 is ~0.6× the 4-seed eval RANGE,
single run, no repeat. The defensible claim is "removal shows no gain, and the
14% speed win wasn't worth an unresolved-sign quality risk". KEEP v2 stands,
on decision-theory rather than on the point delta.
Cross-campaign, the only ABOVE-floor offline results ever measured: optimizer
family (aurora beats soap/cosmos/adamw by 0.14–0.23), smolgen off (+0.48 —
most load-bearing module in the net), variable-width chain (+0.048 kill), ffn
budgeted-shrink (−0.035, internally replicated by two shapes), soft-ablation
policy (+0.019/+0.019 on two windows — replicated, real, and now explained by
the 41% trunk-gradient share). Everything else (~40 arch variants, zclip grid,
relbasis/arc/smolscale families, heads16, layers12, bt4embed, deepnorm,
az4672-vs-lc1858) was within noise at the 1-epoch budget — decisive negatives
about where NOT to spend future sidecar compute. GOTCHA baked into the CSV:
runs that change loss composition (soft_ablation, cat_blend_*, sfmove_*,
exp_cat_*) have INCOMPARABLE eval_loss — judge them on the unchanged
per-component columns only. PROTOCOL going forward: an offline sidecar verdict
needs (a) |Δeval| > 0.07 (2× floor), or (b) a 2-seed repeat, or (c) an
orthogonal corroboration (ablation / prescreen / puzzle-bench / audit) — same
discipline as the live paired-CI rule.**

**Surprise-signal bake-off (2026-07-02, n=10,612 full rows, 30 recent shards,
outcomes scored with ckpt478).** Spearman / top-decile-lift of each candidate
sampling signal vs realized per-row error:

| signal | policy CE | value err vs SF | value err vs z |
|---|---|---|---|
| kl (diff-focus policy term) | **+0.43 / 1.37** | -0.03 / 1.05 | -0.09 / 0.76 |
| qd (diff-focus q-surprise) | -0.12 / 0.88 | +0.48 / 2.33 | +0.39 / 1.31 |
| gap (SF-vs-search value gap) | -0.05 / 1.04 | **+0.58 / 2.63** | +0.39 / 1.29 |
| spd (SF-policy-diff vs p0 teacher) | +0.35 / 1.22 | -0.10 / 1.52 | -0.14 / 1.24 |
| td2 (hindsight TD, t vs t+2) | -0.05 / 1.03 | +0.39 / 1.74 | +0.36 / 1.29 |
| zgap (outcome surprise) | -0.06 / 0.95 | +0.37 / 1.24 | +0.93* / 2.25 |

*zgap~verr_z is near-tautological (both measure distance to z).

Takeaways: (1) signals split into two orthogonal families — policy-error
predictors (kl, spd) and value-error predictors (gap > qd > td2) — no signal
predicts both, so the additive priority design is right and per-family weights
should follow this table. (2) `gap` is the best value signal → supports the
#104 activation exactly as planned. (3) qd is NOT dead weight (+0.48; earlier
"self-referential ≈ blind" claim was too strong) — keep it. (4) spd is a real
second policy signal (partially redundant with kl, corr 0.36) — candidate
add-on AFTER gap-priority reads. (5) td2 is dominated by gap on full rows
(corr 0.48, weaker) — not worth new plumbing; possible fast-row upgrade only.

**Resolution study (same rows, ckpt466 → ckpt478):** bottom-half-gap rows
improved (verr_sf 0.087 → 0.080) but **top-decile-gap rows did NOT resolve
(0.383 → 0.385, slightly worse) despite being in the training window** — the
microscopic twin of the 21/35 blind-spot persistence. Consequence: gap-priority
activation gets a SECONDARY readout — top-decile-gap resolution rate must turn
positive under 4-6x sampling; if it stays ~0, emphasis isn't the bottleneck →
capacity/target (rung 4). Caveats: the 466→478 window had the weak 150k/400k
teacher and pre-rung-1 blend (SF share of the value target was small), and
~30% of top-gap rows may be positions where SF itself is wrong (fortress-type).

**Value-head construction/target screens (2026-07-03 early).** Two offline
probes of "is the value head badly constructed / is the blend target the
ranking bottleneck":
(1) *Co-trained three-head ranking probe* (wdl vs sf_eval vs categorical on the
same trunk, ckpt457/466/478, v1-2k): sf_eval (pure SF target, identical
architecture, 10× LESS loss weight) ranks ~4cp better than wdl at 457/478,
flips at 466 (the 150k-label-cap era — pure-SF distillation tracks teacher
depth hard). Pooled paired CI −2.31 [−6.52, +2.01] — directional, NOT
significant. Categorical (mostly-z) consistently worst.
(2) *Frozen-trunk equal-budget head retrain* (fresh ValueHeads on ckpt478
trunk, 60k rows / 3000 steps each; scratchpad `frozen_head_screen.py`):
blend_prod 85.7 < sf_prod 89.6 ≈ sf_wide(256) 90.1 < blend_noecho 91.3. At
truly equal budget the BLEND trains the better-ranking head; wide ≈ prod says
head-local capacity is NOT binding. LIMITATION: all fresh heads land ≥10cp
worse than the co-trained heads (60k rows underfits — val CE overfits by step
2k), so absolute levels are floor-limited; relative reads only. Also
blend_noecho reaches the LOWEST val CE yet ranks WORST — CE and ranking
dissociate again.
**Combined verdict: no evidence for head mis-construction, and no
offline-demonstrable blend-target win — re-confirms the June targets-pivot
lesson from a new angle. The ranking deficit points at trunk scale /
co-training volume / gradient share, not head arch or target recipe. Next
cheap diagnostic: per-head trunk-gradient share (is value starved by the 4
policy heads?); cheap live lever if starved: w_wdl rebalance rung.**

**Trunk-gradient share diagnostic (2026-07-03 early; scratchpad
`grad_share_diag.py`, ckpt479 pool weights+window, production loss kwargs from
trial params.json + live yaml, 8 batches × 256).** Per-component weighted
trunk-gradient L2 shares: policy_ce 54.9%, **soft_policy_ce 41.1%** (w_soft=1.0
— an auxiliary at main-head weight), sf_own 11.6%, **blended_wdl 10.7%**,
categorical 4.6%, sf_own_regret 3.3%, sf_eval 1.2%, rest <1%. Policy-group vs
value-group ratio **5.9×**; policy-vs-value cosine +0.04 (orthogonal — pure
budget imbalance, not interference). The search-consumed value head trains the
trunk with ~1/9 of the budget. GOTCHAS baked into the script: the yaml alone
builds a WRONG 34M model (v2_threats + w_sf_own* + non-uniform ffn are
trial-injected — build from the checkpoint's `arch` payload, strict load,
weights from trial `params.json`); the tolerant loader accepted the wrong model
silently and produced garbage shares (64% categorical) on the first run.
Also: 46.2M checkpoint state = 34.7M trainable + 11.5M per-layer Smolgen
gen_weight buffer mirrors (shared weights).
**Queued rung — "gradient rebalance" (#104 killed 07-05; this rung now queues behind FEN seeding — never two value levers in one window):**
`w_soft` 1.0→0.5 + `w_wdl` 1.0→1.5 (both live-tunable) → policy:value ratio
~6:1 → ~2.3:1. Yardsticks: value_regret paired CI vs pre-rung dump (primary,
must improve); audit policy paired CI (guardrail, no significant regression);
one window. Rationale: captures the gradient-budget benefit of a value-neck
topology change with zero restart; the neck stays justified only if this rung
moves value ranking and then plateaus.

**Paired-CI retro-read (2026-07-02 late; PR #107 tooling).** The kill-window
point verdicts re-judged with paired bootstrap CIs on the frozen v1-2k
positions (delta = A−B, negative = A better; regret metrics, lower better):

| A vs B | metric | paired delta [95% CI] | verdict |
|---|---|---|---|
| ckpt478 vs ckpt457 | audit raw top-1 | **+10.95 [+1.49, +21.18]** | REAL regression; endgame-driven (+17.9 [+6.8, +31.3]) |
| ckpt478 vs ckpt457 | audit search E[regret] | +3.91 [−2.52, +10.62] | not significant |
| ckpt478 vs ckpt457 | audit raw E[regret] | +2.12 [−2.02, +6.48] | not significant |
| ckpt478 vs ckpt457 | value top-1 regret | +7.09 [−0.88, +15.37] | NS overall; **endgame +9.39 [+0.17, +19.00] significant** |
| ckpt478 vs ckpt466 | value top-1 regret | +1.08 [−7.57, +9.91] | not significant |

Reading: the throughput-window regression is real and **endgame-concentrated**
(raw policy top-1 and endgame value both clear their CIs; means alone hid the
phase structure). The headline "value 72.4→76.6" was partly an anchor artifact:
ckpt457 re-reads 69.5 (not 72.4) with the same protocol — stored point numbers
drift ~1–3cp across sessions. Consequence: verdicts pair fresh dumps, never
compare stored point numbers. Dumps banked: `scratchpad/policy_ci/*.jsonl`,
`scratchpad/paired_ci_smoke/*.jsonl` (ckpt457/466/478, value + audit).

Scripts: resolution readout is checked in (`scripts/gap_resolution.py`); the full six-signal bake-off lives in the session scratchpad `signal_predictiveness.py` (+ results JSON) — promote it too if it becomes a recurring read. Paired-CI tooling is checked in (`scripts/paired_compare.py`, PR #107).

## Open bets, validated but not built

- **Per-row SF-frac ramp by disagreement** (2026-07-02): shard evidence says when SF
  and our search disagree, SF is right 61%→70% monotonically in gap size — and the
  arbiter (selfplay z) is biased *against* SF, so that's a floor. Design: per-row
  `sf_frac` ramp in losses.py (both fields already in the batch). One knob, default off.
- **Blind-spot mining → curriculum seeds**: start games from harvested
  net-vs-deep-SF sign-flip positions. Needs a seed-FEN pool feature in selfplay.
- **Tail metrics in value_regret** (P95, sign-flip rate): the mean hides exactly the
  errors that lose games (Cheese autopsy: 80% of losses = one collapse; the net still
  reads 21/35 of those positions as fine 300+ iters later — the loop does not
  self-correct off-distribution value errors). BUILT: `scripts/tail_stats.py`
  (P90, >100cp, >300cp, paired >300cp flip counts); run it on every value dump pair.
- **DONE 07-07 — continuous panel severity** (`scripts/blindspot_panel.py --dump-per-position`
  + `paired_compare.py`): the binary BLIND count (net_after > −0.2 over 35/113 rows) is
  binomial-noisy — 23→19 read as "within noise". The additive dump writes per-row
  `{fen, value=net_after}` (higher=blinder; paired_compare's "A better=lower value"
  convention holds) so two checkpoints get a paired-CI severity delta. FIRST USE (ckpt609
  FEN-baseline → live ck639, 30 iters): **held-out panel v1 severity +0.16 [+0.08, +0.23]
  SIG better, v2 +0.13 [+0.07, +0.19] SIG** — where the binary count looked flat. The
  broad audit tail was FLAT over the same window (75.9→76.5, `scripts/tail_stats.py` net −4
  blowups), so the panel-only improvement is a TARGETED-intervention signature (FEN hits
  collapse positions), not broad #104 recovery (which would lift the audit too). CAVEAT: single
  checkpoint per side, 30-iter window, confound not fully excluded — but the panel/audit
  divergence is the tell. A session watcher (untracked, session-specific paths) polls this
  every ~10 checkpoints during the FEN window so the ~650 readout carries the paired
  severity automatically; the reproducible primitive is `blindspot_panel.py --dump-per-position`
  + `paired_compare.py`.
- **Cadenced cross-checkpoint arena + loop-health invariant monitor**: nothing runs on
  a schedule today; every regression so far was found by ad-hoc suspicion.
- **Scale-up spec — 512×14 h16 (2026-07-07, sizing study `scratchpad/arch_sizing/`).**
  Motivation: repeated capacity-theft signatures at current scale + local knob space
  exhausted (ledger rule 9 discussion). Ranked by MEASURED forward FLOPs (CPU
  profiler through the real forward; params are a bad speed proxy — per-layer
  Smolgen is 38% of params but only ~12% of FLOPs, LC0-consistent):
  prod 384×12 = 34.7M trainable / 2216M FLOPs/pos; **512×14 h16 = 55.7M / 4274M
  (×1.93)**; 640×14 h20 = 75.6M / 6479M (×2.92); 768×15 h24 = 103.7M / 9516M
  (×4.30). **SIZING FRAME CORRECTED 07-07 (user): do NOT hold a FLOPs budget or
  force width/depth balance — the objective is capacity + GPU util, and the only
  real cost is RL-loop throughput (slower forward → fewer selfplay games/day →
  less data/wall-clock).** Two corrections to the first pass: (1) WIDTH buys util
  — util = arithmetic intensity = matmul SIZE, so a wider net saturates the 5090
  better at the same batch; we sit at ~52% util in-game (12k of 23k pos/s @ batch
  ~400) so there's headroom width fills, while depth only stacks same-size matmuls
  (capacity, not util). So width-leaning is CORRECT when util is a goal (the
  earlier "rebalance to 480×16" reversed this). (2) The util gain SOFTENS the
  throughput cost — a ×2-FLOPs net can be only ~×1.4-1.6 wall-clock if util climbs
  toward peak — so FLOPs is an OUTPUT of the microbench, not a budget to hold.
  METHOD: microbench measures **pos/s AND achieved util%** across a real range —
  512×16 (63M), 640×15 (~80M), up to the existing `default.yaml` 768×15 h24 (103M,
  purpose-built as the future-larger-model) — and pick the LARGEST whose measured
  wall-clock the loop can stomach, leaning WIDER for util, depth kept ≥15 (LC0-like)
  for the compositional value-blindspot goal. Warm-start (distill-bootstrap) also
  shortens training, buying back more of the cost. Reference grid
  (`scratchpad/arch_sizing/`, head_dim 32, middle-wide FFN preserved): 512×14 36.6M...
  512×16 63M/×2.2, 640×14 75.6M/×2.9, 768×15 103.7M/×4.3. Keeps head_dim 32 and
  ALL input/output/smolgen choices identical — only trunk width+depth+FFN-schedule
  move. Final size = microbench (throughput+util) call, leaning big.
  **WIDTH MUST BE A MULTIPLE OF 128 (tensor-core tile alignment).** Blackwell GEMM
  kernels tile in ~128-wide blocks: 512=4×128 fills every tile, 480=3.75×128 leaves
  a padded partial tile (~tile-quantization waste + worse occupancy) — so the
  earlier 480/544 candidates were GPU-inefficient and are DROPPED. Tile-aligned
  ladder = 384 (prod) → 512 → 640 → 768 (`default.yaml`); util rises monotonically
  along it (aligned tiles + bigger matmuls), so the ceiling is throughput, not an
  efficiency cliff. head_dim stays 32 (512→16 heads) for step 1; head_dim 64 is a
  later knob. **512×16 h16 is the leading candidate: tile-aligned width +
  proportional depth (12→16, toward LC0/`default.yaml` 15L).**
  **FFN SCHEDULE (corrected 07-07): PRESERVE the production wider-in-the-middle
  profile — do NOT flatten to uniform.** Production 12L `ffn_mult_by_layer` peaks
  ~1.92 mid/late (mean 1.65); the sizing study's first pass used a uniform 1.6
  (`ffn_mult_by_layer=None`), which both drops the middle bump AND under-shoots
  the mean. Restore it by interpolating the 12L profile onto **the FINAL SELECTED
  depth** (`np.interp(linspace(0,1,num_layers), linspace(0,1,12), prod)`) — the
  schedule length MUST equal `num_layers` or `normalize_ffn_mult_by_layer` rejects
  it, so a 16-layer 512×16 pick needs a 16-entry schedule, NOT the 14-entry one
  (mean 1.65 / peak ~1.90 either way). Costs only ~+0.6% params / +1% FLOPs vs
  uniform, so the scale-up stays truly single-variable (width+depth up, FFN SHAPE
  held). Interp is the default starting schedule; re-searching the profile at the
  final depth is a later refinement, not a launch blocker.
  PARAM-COUNT
  GOTCHA (fixed in notes): production is 34.7M TRAINABLE — the "46M" figure
  counted state_dict, which double-counts the BT4-style weight-TIED smolgen gen
  matrices (12 names → 1 tensor) — use named_parameters for counts. GATES:
  (1) FEN readout + fresh Cheese block first; (2) compiled GPU microbench of
  the SELECTED ladder candidates at a training pause — 512×16 primary, with
  640×15 / 768×15 as comparison points (FLOPs→wallclock validation must run on
  the size we'd actually launch); (3) warm start via
  offline distill-bootstrap on the banked window (bootstrap infra +
  offline-recovery playbook), NOT net2net width surgery; (4) carries the rule-9
  from-scratch retest list (surprise_mix, gap-priority class, per-layer-vs-shared
  smolgen mini A/B). Throughput clawback lever if needed: TensorRT/fp8 on the
  bigger net (fp8 PTQ was quality-dead at 34.7M; larger nets quantize better).

## Decision queue (updated 2026-07-06 after the FEN activation; in order)

1. ~~Fast-ply revert~~ / ~~LR decision~~ / ~~#104 merge~~ — DONE via the bundle
   restart (see LIVE table). ~~Recovery read~~ — RECOVERED @ckpt524, banked.
2. ~~Activate the gap boost~~ — **DONE and KILLED (07-04→07-05)**; the offline
   dose screen RAN 07-05→06 and killed the abs() family — low-w retry DEAD
   (damage flat in dose). **Signed variant (#113): TESTED + DEAD (07-07)** —
   true arms ≈w30-abs (−1.8/+1.5 NS), not better than w0 (−10.0 SIG/−4.95 NS),
   panels 23/23 no help. Gap-priority reweighting family CONCLUSIVELY closed;
   #113 merged but OFF permanently.
3. ~~Post-kill recovery read~~ — **DONE @ckpt609 (07-06): CONTINUE per the
   pre-committed rule** (overall value NS vs ckpt524; endgame slice still SIG
   worse — that residue is the FEN window's problem). No salvage.
4. ~~FEN-seeding activation~~ — **ACTIVE since ~iter 613 (07-06)**, delivery
   verified via per-iter `fenlist` outcome_stats (see LIVE table row).
   **NEXT READ ~iter 650 (~1 day): PRIMARY held-out panel v1 BLIND vs 23/35
   @ckpt609; SECONDARY v2 seeded-subset; GUARDRAIL value paired CI ±2cp vs
   ckpt609.** If v1 flat but v2-seeded moves: raise dose / add stored-history
   seeds before judging the idea. PRE-COMMITTED ROLLBACK TRIGGER (07-07):
   ckpt524 is still the strongest banked point (current ~3-4cp worse NS
   overall, endgame −13.7 SIG @609). IF the ~650 FEN readout is a kill AND
   the following monitor read still shows the endgame slice SIG-worse vs
   ckpt524, rebase: salvage-restart warm-starting from the banked ckpt524
   state (`scratchpad/recovery_read_ckpt524/checkpoint_000523/` — trainer +
   pid + rng; current clean window carries over), FEN seeding kept on.
   Otherwise CONTINUE stands; don't re-litigate without this trigger firing.
4b. At the activation pause: Cheese-only match block (~20-40 games; rofChade
   DROPPED 07-05, user call — g465 already showed the grind mode and no
   decision changes on the answer; rofChade returns as a graduation match
   after we take games off Cheese). PRECONDITION (time-forfeit fix, the c8192
   match's 19 forfeits): (a) MERGE PR #116 — batch-time-variance clock margin
   (`ClockBatchMarginSigmas`, default 2σ; a chunk can't be interrupted mid-flight
   so the search now reserves ~2σ of measured chunk time before the deadline);
   (b) confirm the engine WARM-UP fires (already wired: `warmup_search` at
   startup + on `isready` after option changes — pays the ~3-4s cudagraph
   capture off-clock so move-1 doesn't forfeit); (c) **`chunk_sims` sweep**
   (`scripts/bench_uci_engine.py` sweeps it; `scripts/validate_time_mgmt.py`
   for Elo/forfeit) — the un-interruptible unit is ONE chunk (default
   `--chunk-sims 2048` ≈ 120-170ms in-game), and chunk size is DECOUPLED from
   GPU batch (`target_batch`/MinibatchSize = GSS_GPU_BATCH 1024 default drives
   util), so lowering chunk_sims toward ~1024 shrinks the overrun + the reserve
   at ~no util cost — pick the smallest chunk that holds nps AND move agreement.
   Run the block on the DEDICATED GPU (never concurrent — stable batch times
   keep the margin estimate valid). Fresh current-net losses = panel v3 mining
   material.
4b2. **LR-PLATEAU PROBE QUEUED (07-07, runs BEFORE 4c via gate ordering —
   ~10 GPU-h).** USER HYPOTHESIS: the plateau isn't capacity, it's the
   never-decayed flat LR 0.0003 (the 07-02 flat-lower-LR live test was
   confounded → UNREADABLE; this is the clean offline version). Arms from
   the frozen iter-647 checkpoint+window, identical 4500-step cold-optimizer
   budget: `lr3e4_s0` control (0.0003) / `lr1e4_s0` (3× drop) / `lr3e5_s0`
   (10× drop). Driver `scratchpad/scaleup/run_lr_probe.sh` (detached, gated
   on the sidecar chain; provenance check asserts the lr override registered
   in retarget_report.json — rule 8). **PRE-COMMITTED READ: paired
   value_regret vs the CONTROL arm.** A drop arm SIG better on broad value ⇒
   plateau is LR-BOUND ⇒ (a) plan a live LR drop right after the FEN window
   closes (PBT-pin mechanic: yaml + newest experiment_state edit + restart),
   (b) the bootstrap's phase-2 LR drops become load-bearing. Both drops ≈
   control ⇒ capacity story stands, bootstrap is the lever. Single-seed
   caveat: a win inside the ~7cp seed floor needs an s1 repeat before any
   live action. **VERDICT (read 07-08, recorded 07-08): NOT LR-BOUND — both
   drop arms NS vs control by the pre-committed paired rule** (lr1e4 −0.59
   [−9.12, +8.24]; lr3e5 +1.98 [−6.04, +10.35]; logs
   `scratchpad/scaleup/paired_lr{1e4,3e5}_s0_vs_control.log`). Neither drop
   arm SIG better ⇒ no live LR drop; capacity story stands and the 512×16
   bootstrap (4c) is the lever. Point values (75.9/76.5/73.9) all inside the
   seed floor — no s1 repeat warranted since there is no win to confirm.
4c. **512×16 BOOTSTRAP LAUNCHED (07-07, gated behind the sidecar chain +
   the 4b2 LR probe — user-authorized auto-start when the GPU slot frees).** Fresh-init 63.08M
   net (embed 512 / 16L / h16 / prod FFN profile interpolated to 16 entries;
   verified ×1.82 trainable vs 34.7M; SAME inputs/outputs — 175 planes,
   lc0_1858) trained offline on the frozen iter-647 window
   (`data/salvage/scaleup_512x16_window_20260707`, 808 shards; the pool is
   also the rule-2 revert point and the eventual swap vehicle). HYPOTHESIS:
   the reweighting graveyard's capacity-theft signature means 34.7M is
   capacity-bound; a fresh larger approximator on the same data reaches
   parity and unlocks headroom. Mechanics: `offline_replay_epoch.py`
   fresh-init (no --init-checkpoint; width can't migrate), candidates
   aurora_mlp_out, batch 256 eager, live-follow static-pool mode
   (credit-cap 0), 60k-step cap, eval/500 save/1000, hard GPU memcap 0.30 via
   `scratchpad/scaleup/bootstrap_memcap_wrapper.py` (offline_replay_epoch has
   no --gpu-mem-fraction; the cap makes the BOOTSTRAP die on overflow, never
   the live trainer). Driver `scratchpad/scaleup/run_bootstrap_512x16.sh`
   (detached; log `runs/scaleup_512x16_bootstrap/train.log`). LR PLAN
   (07-07): phase 1 runs the config schedule (0.0003, sqrt_release ≈ flat);
   at eval_loss plateau, PHASE 2 = classic step drops — re-invoke with
   `--init-checkpoint <best-so-far> --lr 0.0001`, then `--lr 0.00003`,
   stopping each phase at its plateau (offline stationary data wants decay;
   reactive plateau-triggered drops, NOT a pre-specified step count, because
   we don't know the plateau step in advance and the eval curve is cheap).
   STOP each phase at eval_loss plateau (playbook: static window overfits
   past it). **SWAP GATE
   (pre-committed): the live restart happens ONLY at audit parity or better —
   value_regret AND audit_targets (2000 pos, paired CIs) within +2cp of the
   then-current live net, panels not worse. Below parity = extend data/steps
   or kill the size; never swap on hope. Sequencing note: this front-runs the
   microbench gate (spec section) — acceptable because the bootstrap is sunk
   compute, not a live commitment; the microbench + Cheese block still gate
   the SWAP at the restart pause.** Carries the rule-9 from-scratch retest
   list at the new scale. **LAUNCH INCIDENT (07-08): the gated auto-start
   fired 10:26 and died in 23s — `--config configs/pbt2_small.yaml` does NOT
   carry `input_extra_features` (it lives in the Ray trial state, padded out
   of the yaml), so offline_replay_epoch built the net at the v1/146-plane
   default and refused every 175-plane shard ("refusing to truncate"), then
   starved the sampler; the driver's `exit=$?` was masked to 0 by `$(date)`
   in the same echo, so the crash was silent. FIXED + RELAUNCHED 19:49
   (driver now passes `--input-extra-features v2_threats` and captures rc
   before the date echo): verified 175 planes, 0 shard skips, training at
   ~146 samples/s eager (~0.57 steps/s ⇒ 60k cap ≈ 29h; plateau watch on
   report.jsonl per the stop rule). Crash log banked:
   `runs/scaleup_512x16_bootstrap/train.log.crashed_v1planes_20260708`.**
5. After FEN seeding reads out: gradient rebalance rung (`w_soft` 1.0→0.5 +
   `w_wdl` 1.0→1.5) OR teacher-distillation offline screen — pick by what the
   seeding readout says about the tail. Never two value levers in one window.
6. Build: tail metrics in value_regret; SF-frac ramp PR (per-row SF weight
   scaled by disagreement — premise validated 61%→70% monotone); arena cadence
   + loop-health invariant monitor.
7. Restore `sf_pid_regret_tighten_streak_gain: 0.5` when EMA winrate ≈0.52.
