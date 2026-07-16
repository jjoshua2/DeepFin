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
| `data/salvage/pre_aot_deploy_20260714` | 2026-07-14, pre-AOT-broker deploy (trial 4c17c ~iter 74, wedge-recovered). NOTE: AOT is an inference-path perf change — weights/optimizer/replay are UNCHANGED, so the real revert is just blanking `distributed_inference_aot_dir` + restart; this pool is belt-and-suspenders. | `./scripts/train.sh salvage-restart data/salvage/pre_aot_deploy_20260714` |
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
| **AOTInductor packages in the broker forward** (`distributed_inference_aot_dir: data/aot_models_512`, live yaml; code = feat/aot-broker-integ AOT runtime files landed on the live branch, off-by-default flag) | 2026-07-14, trial 4c17c restart (~iter 74) | **Hypothesis:** pre-compiled max-autotune AOT packages replace the reduce-overhead compiled forward for the 99.04%-of-forwards buckets ≤1190 (per `scratchpad/bucket_hist.json`: 90.9% pad to 128), giving a faster forward at zero quality change. **Full compiled ladder built (14 packages, 128–4096)** so batches >1190 (which by POSITIONS carry most forward compute — large summed-slot totals — even though 90.9% of forward CALLS pad to 128) also use AOT instead of eager; only totals >4096 (~0% per bucket_hist) fall back to the reduce-overhead eager path. **This is a PERF change, not a training-target change** — packages verified numerically ==eager (pol-prob Δ ~0.002, wdl-prob Δ ~0.03 across all 11 buckets, top-move vs fp32 0.977 == eager's 0.984; `scripts/build_aot_packages.py --verify-only`). Measured forward A/B on the quiet GPU (`scratchpad/aot_ab.py`, both inference-only + cudagraph): AOT ~5–6% faster than reduce-overhead median (128:+3.6% 512:+6.2% 1024:+6.7%; one bucket 170 −2.3%). **Expectation is ~0 end-to-end** — the forward is a small slice of the broker loop (sync+scatter ~88ms dominates a batch) and training is ingest-bound; this is a "try it, confirm no regression" deploy, not a throughput bet. **YARDSTICK (deciding):** games/h vs the ~641 games/h baseline (`scratchpad/live_read/monitor/monitor.log` + broker.out pos/s), read after ~1 day. **KILL (any):** (1) VRAM OOM or broker/worker crash on startup (11 AOT cudagraphs + the reduce-overhead fallback + trainer coexist — the main risk; watch `nvidia-smi` + broker.out right after restart); (2) games/h regresses vs baseline; (3) `value_regret --max-positions 2000` paired CI vs the pre-deploy ckpt worse (would be surprising given ==eager, but the guardrail against a silent AOT-path bug). **REVERT (instant):** blank `distributed_inference_aot_dir` (empty) + restart → back to pure reduce-overhead; weights/optimizer/replay untouched (no salvage-restart needed). Revert pool `data/salvage/pre_aot_deploy_20260714` banked as belt-and-suspenders. **VERIFY AOT ENGAGED after restart:** broker.out logs `AOT packages loaded from data/aot_models_512: buckets=[...]`; if absent → flag not read (fell back to eager, benign but not the experiment). Confound: none — single change, packages ==eager. Note for future: AOT's real payoff is the match/UCI path (forward-bound, instant-start), not training; the worker-direct AOTEvaluator path is wired but off. **VERDICT: KILLED for training 2026-07-15 (reverted, `distributed_inference_aot_dir: ""`) — NOT on the yardstick (never got a clean readout) but on a fatal side effect: AOT flipped the sporadic WSL dxg host-bridge wedge into a CLOCKWORK ~100-min wedge/recover loop. Evidence: pre-AOT wedges were sporadic (07-14: 11:08, 12:24, then a 6.5h HEALTHY stretch to 19:00); post-AOT (07-15) every single ~100-min window wedged — 9 watchdog recoveries, zero healthy stretches, iter advanced only 74→75 in ~15h (~90% downtime). Mechanism: loading + capturing/replaying 14 AOT cudagraph packages at broker startup is heavy dxg-bridge traffic that reliably trips the flaky bridge (same class as the max-autotune compile-hang variant, memory `wsl2-gpu-vmbus-wedge-signature`). Packages kept in `data/aot_models_512` (numerically ==eager, all verified) for the match/UCI path and a RETRY. RETRY CONDITION: a freshly-rebooted / confirmed-stable bridge (the wedge is host-level; AOT only aggravates it) — re-set the key + restart, watch the recover_stall cadence for the first ~2h. CONFIRM THE REVERT: reduce-overhead should restore multi-hour healthy stretches; if it STILL wedges every ~100 min, the host bridge itself has degraded → reboot needed (not AOT).** |
| **Server-doled FEN seeding ACTIVATED — switch prob→dole** (`opening_fen_dole_per_iter: 1`, `opening_fen_prob` 0.05→0, live yaml; code = **PR #128** merged, restarted onto `54f4037` @iter 690 resuming ckpt686) | 2026-07-08, ~iter 690 | **Mechanism switch, not a new seed set** (same 115-seed v3 list). Replaces the probabilistic `opening_fen_prob` draw with the deterministic server dole: each iteration the server hands the WHOLE list to one worker poll (durable per-iter gate), played as **selfplay slots only** (`resolve_slot_opening` dole branch → PID-safe by construction; the prob path's curriculum seeding is what forced the 0.22→0.05 dial-down). Dose 1 = each of the 115 seeds played once/iter ≈ **115 selfplay seeded games/iter** (vs the prob path's ~35 mixed games/iter, ~3–27 of them selfplay). **Two coupled effects of the single switch (the confound):** (a) ~3–10× MORE selfplay seed volume = more direct value-label injection; (b) curriculum refutation games (net on the blundering seat, 698k-SF punishes on-board) → **0** — dole is selfplay-only, so the on-board-refutation signal that co-drove the WORKED verdict is dropped. Threaded-path delivery wired + verified (production is `distributed_worker_threaded`; #128 round-3 P1). Baseline @ckpt686 (== the FEN-seeding-WORKED trunk; ledger baseline ckpt683 value 74.3, v1 16/35); banked revert trunk `data/salvage/pre_dole_ckpt686_20260708`. PRIMARY: held-out panel v1 BLIND severity holds/improves vs 16/35 (monitor `scratchpad/live_read/monitor/monitor.log`). SECONDARY: 68 vetted seeds resolve (become AWARE) via `scripts/blindspot_resolution.py`. GUARDRAIL/KILL: `value_regret` (`--max-positions 2000`) paired CI vs the ckpt686 baseline dump worse by >2cp (CI excludes 0 worse side) → REVERT. WATCH: PID sample — dole is selfplay-only so `sf_pid` games should STAY healthy (the whole point vs prob); if it still drops <30 or the airbag trips, that's an unexpected finding. **KILL ACTION (live, instant): `opening_fen_dole_per_iter: 0` + `opening_fen_prob: 0.05` → restores the known-good prob mechanism.** Readout ~1 day (~iter 725). Supersedes the prob-based row below (that mechanism is now off). **LIVE PATH UPDATE (2026-07-09):** `opening_fen_list_path` now points at auto-retired `data/blindspot_fens_retire_720.txt` (85 FENs; 16 AWARE seeds dropped from retire_700 by `blindspot_retire_step.py`). Mechanism unchanged (dole still dose 1 over the live list); activation baselines remain ckpt686 / 115-seed v3. |
| **Harvested blind-spot FEN seeding — v1→v2 SCALE-UP, dose 0.02→0.22** (`opening_fen_list_path`→`data/blindspot_fens_v2.txt`, `opening_fen_prob` 0.22, live yaml) | 2026-07-08, ~iter 683 | Scales the WORKED FEN lever (v1 verdict in the WORKED table): v2 = v1's 76 seeds + **68 deep-SF-VETTED** live-harvested blind spots. Gate: `scripts/blindspot_deepsf_gate.py` on 157 banked severe seeds → 4M nodes + 6-man syzygy TB (`data/syzygy_3-4-5-6`), keep deep-LOST → **68 pass (44%), 87 deep-FINE FPs dropped (56%)**. The gate is LOAD-BEARING: calibration (`scratchpad/harvest_fp/`, deep SF converged 4M=8M=16M, TB-invariant) showed the raw severe band is **~70% false positives** (deep SF agrees with the NET, not the ~700k in-loop label) — feeding the raw `.severe` file would re-label correct positions "lost" via the shallow in-loop SF and UN-TRAIN the net. 0 overlap with held-out panel v1 (generalization metric stays clean); 0 dup with v1. Dose 0.22 targets ~1 game/seed/iter (665 g/iter ÷ 144 seeds) — **11× the validated 2%**, poison-safe ONLY because deep-vetted. Baseline @ckpt~683 (value 74.3, v1 16/35, from the monitor). REVERT TRUNK banked: `data/salvage/pre_v2feed_ckpt678_20260708` (ckpt678 trainer+pid+rng — NOTE: `salvage-export --metric training_iteration` misfired to iter 449 with ~40 stale experiment_state files present, so this is a direct checkpoint copy; nuclear revert = dose→0.02 + hand-built pool from this trunk via the offline-recovery playbook). PRIMARY: held-out panel v1 BLIND severity holds/improves (monitor `scratchpad/live_read/monitor/monitor.log`). SECONDARY: the 68 vetted resolve (become AWARE) via `scripts/blindspot_resolution.py`. GUARDRAIL/KILL: `value_regret` paired CI vs the ckpt683 baseline worse by >2cp → revert dose to 0.02. WATCH: seeded curriculum games are excluded from the PID sample → a 0.22 dose could starve it and trip the airbag (known failure mode `curriculum_starvation`) — if `sf_pid` games <30 or the airbag fires, dial the dose down (`scratchpad/harvest_fp/feed_watch.log`). Dose is live-reloadable — revert instantly. Harvester code (full-game save, one-per-game cap, `ply=` stamp) = **PR #125**. Confound: overlaps the v1 seeds' continued exposure (v1 largely learned already, panel healed 23→16) — v1's isolated verdict is banked, so this reads as the incremental harvested-seed effect. |
| **Return-to-known-good bundle restart** (fast-ply revert + LR revert + #104 code, knobs off) | 2026-07-02 evening | **THE active readout**: over the next window refill (~1-1.5 days), policy net+search / raw top-1 vs **49.6 / 51.5** and value vs **76.6→toward 72.4** (protocol: `--max-positions 2000`). Recovery ⇒ throughput-era damage was config, not permanent; no recovery ⇒ window legacy or trunk state → consider salvage-restart from a clean pool. LR revert applied via the PBT-pinned mechanic (above) and VERIFIED: peak_lr=0.0003 on iter 482. Confound: rung-1 fracs (below) remain live by design. **Contamination (07-02 ~21:17–23:5x, iters 484–486): the branch-switch incident (see gotchas) reverted the yaml — fast-ply rows back ON (~150k value-only rows in those shards), labels re-capped 400k, blend back to 0.35/0.35 — restored 23:09; fast-ply-off re-propagation VERIFIED (shards 23:57–00:02 back to 100% has_policy). Recovery-read clock effectively restarts at iter ~487; interpret ckpt510-era reads with this in the window.** **Pre-committed recovery rule (added before the readout):** each read dumps per-position and pairs vs the banked ckpt457 dumps (`scratchpad/policy_ci/`). RECOVERED = raw top-1 paired CI vs ckpt457 includes 0 AND endgame-value paired CI includes 0 → bank fresh baseline dumps, proceed to #104 activation. NOT RECOVERED = raw top-1 CI still excludes 0 on the worse side after TWO window refills → stop and rebase: hand-build a salvage pool from the banked ckpt457 trainer + the then-clean replay window (offline-recovery playbook, memory `v2_threats_12layer_live_offline_recovery`) and salvage-restart. Do NOT rebase onto `prechange_20260702_ckpt479` — that pool carries the damaged trunk. **REFILL CLOCK RECALIBRATED (07-03 AM, live telemetry): window = 1.41M rows (not the 3M cap — still growing), turnover ~16.1k rows/iter at ~39 min/iter → ONE refill ≈ 87 iters ≈ 2.4 days (the "~1-1.5 days" assumed fast-ply-era 4× ingest). The fast-ply era ingested ~1.3-1.45M rows ⇒ the window was ~fully contaminated at the iter-487 restart, and ~75% of those rows are value-only — policy training stays diluted until they flush. INTERIM READ @ckpt500 (iter ~501, ~16% refilled; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt500.jsonl`): raw top-1 vs ckpt457 −11.2 [−21.8, −1.7] (endgame −20.9 [−36.4, −7.7]; middlegame +8.0 [−0.6, +19.6] — borderline BETTER); value vs ckpt457 −9.5 [−17.9, −1.4] (endgame −11.5); search E[regret] vs ckpt457 −0.7 NS (search still masks the raw damage); vs ckpt478 everything NS (raw top-1 −0.3, value −2.4) — no movement yet, consistent with 16%. Blind-spot panel 19/35 vs 21/35 baseline. NOT RECOVERED, expected at this refill depth — next read ~50% refill (07-04 AM), full-refill verdict read ~07-05, two-refill rebase deadline ~07-07. **TIMELINE SUPERSEDED by the window cut (row below): contamination flush ≈ 36 iters ≈ 24h → verdict read ~07-04, rebase deadline ~07-05.** TAIL ADDENDUM (same dumps, `scripts/tail_stats.py`): the mean is tail-dominated (median value regret 9–11cp vs mean ~70–80); >300cp blowups 4.2%→5.1% (457→500); paired flips 457→500 = 48 new vs 29 fixed (net +19) — the loop mints new tail blowups faster than it fixes old ones. Add the tail line + flip counts to every future read (secondary, descriptive).** |
| Value-blend rung 1: `search_wdl_frac` 0.35→0.20, `sf_wdl_frac_floor` 0.35→0.45 | iter 477 (07-02) | value_regret per ckpt vs the **76.6 pre-anchor** (ckpt478), judged at the FULL-refill read (post-bundle window). Pre-committed rule: SUCCESS = ≤70.4 (2cp below the 72.4 known-good — evidence rung 1 adds value beyond the bundle recovery); KEEP-BUT-UNREAD = 70.4–75.0 (recovered; bundle confound absorbs credit, rung 1 stays as principled default); KILL = >75.0 at full refill → revert pair 0.35/0.35 (live keys). **AMENDED 2026-07-02 late, BEFORE the readout: the ±2–5cp point bands sit inside the value yardstick's measured ±8.7cp paired noise floor — verdict now runs on `paired_compare` vs the ckpt478 dump (`scratchpad/paired_ci_smoke/dump_ckpt478.jsonl`): KILL = CI excludes 0 on the worse side; SUCCESS = CI excludes 0 on the better side; otherwise KEEP-BUT-UNREAD. Point bands demoted to descriptive. Also: iters 484–486 trained on the old blend (incident above) — rung-1 exposure restarts ~487** |
| PID `sf_pid_regret_tighten_streak_gain` 0.5→1.0 (temporary) | iter ~478 (07-02) | controller mode, not an experiment: restore 0.5 when EMA winrate ≈0.52. Expect the winrate sample to shift again post-uncap (congestion bias). 07-03 AM: EMA 0.565 falling ~0.005/iter — restore due later today |
| **`replay_window_max` 3M→800k, deepened same morning to a 250k ONE-ITERATION cut** (TEMP, live yaml) | iter ~502-503 (07-03 AM, user-approved; deepened on the user's "start now" push) | Ops lever, not an experiment: shrink evicts oldest rows. At 800k the flush was still ~24h (580k contaminated rows retained); the 250k deep cut evicts ALL contaminated rows at the next iteration boundary (newest ~236k rows are post-incident clean) → training ~100% clean immediately. Sequence: 250k lands (watcher) → flip cap to 800k + `replay_window_growth_frac` 0.10→1.0 (clean regrowth ~16k/iter, ~34 iters back to 800k) → at recovery banking, cap→1.5M (revisit growth_frac then). Verified live-reloadable (`tc` rebuilt per iteration; shrink via `buf.enforce_window()`; replay keys absent from topology/construction-bound lists). Rationale for waiting on #104/rebalance anyway: activating a training-distribution change inside the healing window confounds BOTH the rebase gate and the knob's own verdict (false-credit risk) — with the deep cut the gate shrinks to ~this evening, so nothing meaningful is lost. Recovery verdict read: earliest meaningful ~18-25 clean-exposure iters (~this evening); rule-quality read 07-04 AM. Cost accepted: one day of small-window recency bias (safe: views-targeting fixes per-row reuse at 2.5 regardless of window size — window size only sets batch recency/diversity, and 250k ≈ salvage-restart scale); refill-based verdicts get less exposure per refill. Note: per-row reuse invariance means the cut costs training QUALITY little while maximizing clean-fraction — strictly better healing. **EXECUTED + VERIFIED (07-03): cut landed iter 504 (1.41M→248,410 rows, all clean); cap flipped back to 800k + growth_frac 1.0 same hour; regrowth confirmed iter 506+ (~13.5k rows/iter avg — selfplay-heavy iters ~15-16k alternate with curriculum-heavy ~5k); winrate EMA steady ~0.56 through the surgery. INTERIM READ @ckpt510 (6 clean iters post-cut; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt510.jsonl`): BOTH pre-committed recovery criteria formally MET — raw top-1 vs ckpt457 −4.1 [−15.4, +8.0] includes 0 (was −11.2 SIG at ckpt500), endgame value vs ckpt457 −1.9 [−10.2, +6.1] includes 0 (was −20.9 SIG). Direct healing evidence vs ckpt500: endgame value +9.6 [+0.1, +19.1] SIGNIFICANT in 10 iters; value >300cp tail flips 500→510 = 22 new vs 43 FIXED (net −21) — first read where the loop fixes tail blowups faster than it mints them; vs ckpt457 value tail now breakeven (30/32, was 48/29). Panel v1 BLIND 18/35 (best yet: 21@477, 19@500); panel v2 55/113 (flat vs 54@500). Still short of full: endgame raw top-1 point −10.4 vs 457 (NS), policy tail net +8 vs 457. Trajectory steep (+6.8cp value in 10 iters) ⇒ NOT steady state — hold #104 for the confirming read @~iter 524 (watcher armed) per protocol; treat that as the banking read if it confirms. **VERDICT READ @ckpt524 (07-04 00:30, 19 clean iters; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt524.jsonl`): RECOVERED — raw top-1 vs ckpt457 −1.7 [−12.5, +10.0] and endgame value +1.3 [−10.3, +13.6], both include 0 with near-zero points; value 72.15 ≈ the 72.4 known-good; trajectory converged (510→524 deltas tiny — transient over). Rung-1 rule: 72.15 lands in KEEP-BUT-UNREAD (70.4–75.0) — recovered, rung-1's own credit inseparable from the bundle (expected). Panels: v1 18/35 (best ever), v2 47/113 (54→47 during recovery, no #104 — clean-window healing reaches the tail). Residuals (noted, not rule violations): policy >300cp tail net +10 vs 457; search E[regret] −5.8 [−12.5, +0.4] borderline NS. BANKED: ckpt524 = the new baseline; cap 800k→1.5M; #104 ACTIVATED (row below). Throughput-triple recovery arc CLOSED.** |
| Blind-spot panel v2 — **BUILT, FROZEN 07-03** (`data/blindspot_panel_v2.jsonl`, 113 rows) | 07-03 AM | Mined the June 15-16 iter520 full-Cheese + June-1 rofChade losses with the v1 method (300k-node SF, first decisive collapse, time-forfeits excluded; session `blindspot_mine_v2.py`): 86 losses profiled → 78 new rows + v1's 35, deduped to 113 (74 iter520-Cheese / 35 ckpt150-Cheese / 4 rofChade). **ckpt500 v2 baseline: BLIND 54/113 (47.8%), AWARE 43/113** — rate consistent with v1's 19/35; binomial CI half-width ~9% vs ~16%. v1 stays THE frozen panel for #104's pre-committed rule (≤16/35 / ≥20/35); v2 pre-committed secondary thresholds vs the 54/113 baseline: SUCCESS ≤ 44/113, KILL ≥ 60/113 (≈ ±2σ binomial). **LOSS-MODE ANALYSIS (user confound challenge answered — "is the newer tune flakier?" NO):** same-opponent cross-net comparison: iter520 (OLD run) vs Cheese = 100% decisive-collapse, 81% one-error dominance, 86% mate-scale — at least as collapse-y as ckpt150 (NEW run: 100%/57%/77%). g465 vs rofChade = 33% collapse, median ZERO ≥150cp drops (genuine grinds, median 32 plies). Collapse mode tracks OPPONENT (Cheese ~3000, balanced-until-blunder) not tune; grind is the expected mode vs the stronger rofChade (~3300). **Open cell: current net has NEVER played rofChade (only g465 did) — queue a ~20-game match at the next training pause (GPU-bound)** **SEARCH-vs-VALUE DECOMPOSITION @ckpt510 (07-03 evening; `scratchpad/policy_ci/blindspot_search_probe.py`, dump `searchprobe_v2_ckpt510_s200.jsonl`): at fen_before with production PLAY-shape Gumbel @200 sims, the net STILL picks the recorded losing move on 23/113 (search) vs 25/113 (raw argmax) — ~7× the ~3% random-match base rate, and search provides ZERO net rescue (balanced-before subset sf_before≥−50, n=37: raw 7/37, search 7/37). Value-awareness does NOT prevent the choice: AWARE rows get picked at the same ~20-24% rate as BLIND rows (specimen: d3d7, sf −11→−2992, net_after −0.49 AWARE, root_q −0.24, search plays it anyway) — Q-propagation/one-ply incoherence, not just after-position misread. Implications: (1) selfplay-sims search inherits the blindness → the loop keeps generating+reinforcing these lines (mechanism behind 'loop does not self-correct tail errors'); (2) #104 gap-priority targets the 56 value-BLIND rows but NOT the aware-but-picks mode — the improved-policy target itself is wrong there; the lever for that mode is blind-spot FEN seeding (games FROM fen_before so deep outcomes refute the move). Caveats: 200 sims ≪ match sims (match-level rescue untested — the queued rofChade/Cheese match is the test); panel move ≠ only losing move (rescue rates are upper bounds). Panel `move` field is SAN, not UCI — parse before comparing (first probe run scored 0/113 on exactly this). **SHAPE + SIMS SWEEP (same evening, logs `searchprobe_v2_ckpt510_{selfplay,play}_s*.log`): picks-losing-move = raw 25/113; selfplay shape @256 (c_scale 0.1 linear, topk 16) 19/113; play shape @256 (c_scale 0.025 root-log, topk 32) 22/113; play @3200 22/113 — 16× search rescues ZERO net positions. Composition flips with depth: @3200 BLIND-row picks RISE 7→11/56 while AWARE-row picks fall 14→9/46 — deep search converges to the value head's beliefs, fixing aware-mode errors and ENTRENCHING value-blind ones. The ~20% floor tracks the value head exactly ⇒ match-depth search is NOT protective; the play shape's lower value-trust is WORSE on AWARE rows @256 (14/46 vs raw 9/46 — down-weights the head where it was right). VERDICTS: (a) do NOT flip selfplay search to the play shape for tail reasons — no offline signal, and the tail is a data problem; (b) the blind-spot fix must come from training data (FEN seeding, #104), not search knobs; (c) play shape stays for matches (its +301 Elo is on-distribution). NEXT BUILD: FEN-seeding feature — `selfplay/opening.py` only supports startpos move-sequence books; needs a flag-gated FEN-list book type (defaults off) so curriculum-vs-SF games start from panel fen_befores and SF refutes the losing moves on the board** |
| Blind-spot FEN seeding — **BUILT (PR #108), flags default-off** | 07-03 evening | Starts selfplay/curriculum games FROM historically misplayed positions (the probe sweep's verdict: search can't rescue these at any depth — only data can). Feature: `opening_fen_list_path`/`opening_fen_prob`/`opening_fen_net_side_to_move` (FEN-list opening source, net forced to the blundering seat, full server→worker distribution mirroring the books, `fenlist` opening-source telemetry). **Seed asset = 78 FENs (panel v2 MINUS v1): panel v1's 35 rows are DELIBERATELY HELD OUT so v1 measures GENERALIZATION (transfer to unseen collapse positions) while seeded-subset v2 rows measure direct learning.** ACTIVATION (pre-committed, do NOT activate in the same window as #104): (1) requires restart onto merged code BEFORE adding yaml keys (strict validator); natural slot = the graceful restart after #104's window; (2) starting dose `opening_fen_prob: 0.02` (~[games/iter]×0.02 seeded games/iter; each of 78 positions revisited every few iters — temperature + SF-regret variety diversifies traces); (3) readouts after ~1 day: PRIMARY = held-out panel v1 BLIND (generalization; success = clear drop from its pre-activation count, kill = rise), SECONDARY = v2 seeded-subset BLIND (direct learning — should move first; if v2-seeded moves and v1 doesn't, the fix memorizes without generalizing → reconsider dose/diversity), GUARDRAIL = value_regret paired CI vs pre-activation ckpt (±2cp) + fenlist game outcomes in stats; (4) fresh-match confirmation (Cheese) only after panels move. **REVIEW ROUND (07-04, xhigh workflow, Codex out of credits): 10 findings fixed (round-3 commit) — headline was [1] PID contamination (seeds forced the net onto the losing seat → dragged curriculum winrate down → eased SF across the whole batch; now excluded from the PID sample like selfplay games, telemetry kept). Seed asset curated 78→76 (2 forced-move FENs dropped). ACTIVATION CHECKLIST NOW REQUIRES A REINSTALL: the min_worker_version=0.0.2 stale-worker gate only fires after `pip install -e .` regenerates the installed metadata AND the worker wheel is rebuilt at 0.0.2 — a bare `train.sh restart` does NEITHER, so without the reinstall old (0.0.1) workers silently mix non-seeded shards into the run. So activation = (a) merge #108, (b) `pip install -e .` + rebuild/publish the 0.0.2 worker wheel, (c) restart onto the code, (d) add the yaml keys. FINDING [0] MEASURED (`scratchpad/policy_ci/history_impact_probe.py`, 112/113 rows reconstructed from source PGNs): FEN seeds fabricate LC0 history planes (empty move-stack → repeat-fill), but the net's OUTPUT is borderline-benign across the two encodings — top-1 move agrees 86%, median value shift 8.6cp (P90 31). Usable for a first experiment; if seeding shows promise, store each seed's ~8 preceding moves and replay them (real history, eliminates the ~14% move-flip) as the follow-up.** **ACTIVATED 07-06 (~iter 613): full checklist executed — #108 merged, `pip install -e .` + egg_info to 0.0.2, 0.0.2 worker wheel, restart onto merged main (61e5707, second restart 18:04 after #113/#115 landed + per-game telemetry #115), yaml keys added (`opening_fen_prob: 0.02`, 84-FEN asset `data/blindspot_fens_v1.txt` — 84 lines at HEAD, VERIFIED 07-06: 0 overlap with panel v1 fen_befores (held-out design intact), 76/84 exact-match v2 fen_befores, 8 near-variants from the same mining pass; the "76" cited earlier was wrong — net_side_to_move on). DELIVERY VERIFIED (iters 613–619 outcome_stats): 8–22 fenlist games START per iter (~2% of games, as dosed) and BOTH paths deliver — selfplay seeds 1–11/iter (SF value labels attach → direct value-head injection) and curriculum fenlist games FINISH despite 698k-node SF (net from the blundering seat: 7W/23D/11L over 7 iters — the refutation data). The 07-05 starvation worry did not materialize: curriculum games finish in bursts on alternating iterations (game length ≈ iteration length beat), so no SF-budget fix needed at this dose. Baseline = ckpt609 (v1 23/35, v2 70/113, value 75.86). Readout ~iter 650 per the pre-committed protocol above.** **VERDICT: WORKED (readout landed @ckpt649, held/strengthened through ckpt669 via the monitor).** PRIMARY held-out panel v1 BLIND 23→19→18→**16/35**, severity paired +0.16→+0.20→**+0.26 SIG better** vs ckpt609 (generalization to positions never seeded). GUARDRAIL met: value_regret 72–74cp, paired vs ckpt609 NS throughout (+3.88/+3.71/+1.56, all CIs include 0 — no regression). Follow-up #1 (real-history seeds) shipped as the harvester `fen \| moves` format; follow-up #2 (broader mining at scale) = the v2 scale-up row at the top of this table (2026-07-08). |
| PR #104 gap-priority sampling — ACTIVATED 07-04 ~00:45, **KILLED 07-05 @ckpt559 readout — every kill threshold crossed** (`replay_sf_gap_priority_weight` 30→0, live yaml) | iter ~526 (activation) → ~577 (revert) | **Activation baselines @ckpt524** (recovery banked the same read): panel v1 18/35, panel v2 47/113, value 72.15, raw top-1 52.5; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt524.jsonl`. RULES: v1 (frozen, primary, unchanged): SUCCESS ≤16/35, KILL ≥20/35. v2 secondary — RE-ANCHORED PRE-ACTIVATION (documented, not silent: the original ≤44/≥60 was set vs 54/113 @ckpt500, but recovery healed the panel to 47 before #104 ever ran; thresholds re-derived vs 47 with the same ±2σ logic): SUCCESS ≤40/113, KILL ≥57/113. GUARDRAIL: paired value CI vs ckpt524 worse by >2cp = kill. SECONDARY READOUT: gap_resolution rate (top-decile-gap rows resolving; surprise-bakeoff finding was they do NOT resolve at baseline). Read after ~1 day (~35 iters). Do NOT start the rebalance rung or FEN seeding in this window. (gap_resolution measured with `PYTHONPATH=. python3 scripts/gap_resolution.py --checkpoint-old <pre> --checkpoint-new <post>` — repro'd the 466→478 finding: top decile +0.0006 vs bottom half +0.0077.) **PRE-ACTIVATION DRY-RUN (07-03, 16.5k rows / 25 newest clean shards): recording works (`has_priority_sf_search_gap` on 97.7% of rows; gap mean 0.132, p90 0.32, p99 0.75 value-units). At w=30 the gap term is 31.7% of priority mass; sampling gets BROADER not narrower (ESS 0.502→0.608, top-1% mass 5.9→4.6% — the flat-ish gap term dilutes the heavy-tailed base surprise); corr(base, gap)=+0.07 → orthogonal targeting; top-decile-GAP rows' priority-half mass 12.4%→20.3% (~1.6×; overall ~1.35× after the uniform sampling half). Verdict: safe to activate, moderate re-targeting, no starvation. w=60 adds nothing (ESS/top-mass identical) — 30 is the right first rung** **VERDICT READ @ckpt559 (07-05; ~21.6h/34 iters exposure at the read, ~51 iters total before the revert landed; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt559.jsonl`, banked ckpt `scratchpad/recovery_read_ckpt559`): KILLED — every pre-committed kill threshold crossed. Panel v1 18→24/35 (kill ≥20); panel v2 47→83/113 (kill ≥57; worst point ever recorded); value_regret 72.15→85.96, paired Δ −13.81 [−22.42, −5.83] SIG, endgame-driven (−17.0 [−29.3, −6.0]) — the >2cp guardrail blown ~7×. Policy raw top-1 Δ −4.97 [−14.17, +2.31] NS ⇒ a pure VALUE regression. SECONDARY (gap_resolution 524→559, 10,718 rows / 30 newest shards): the mechanism WORKED on its own target — top-decile-gap verr 0.489→0.422 (Δ +0.0666 positive, exactly what the rule asked for) — while the bottom half WORSENED (−0.0088); with ALL-replay delta +0.0038 (fine on its own training distribution) against −13.8cp on the frozen audit set, the signature is distribution shift / capacity theft: at ~36% of priority mass (live pmass_gap_share 0.35–0.38 ≈ dry-run 0.317) the trunk reallocated toward the gap rows at the broad distribution's expense. (Alternative reading not excluded: gap rows are where search and the SF label disagree — oversampling trains hard toward SF labels precisely on contested positions, i.e., label quality, not just capacity. Either way w=30 is net-negative.) CONFOUNDS DEFEATED: window regrowth (658k→913k across the window) ran under the SAME growth_frac-1.0 retain-all regime as the IMPROVING 500→524 segment; the iter-544 restart artifact is winrate-telemetry-only; ckpt524 was not a lucky baseline (arc 500/510/524 = 79.0/72.2/72.1 value, 19/18/18 v1, 55/47 v2 — ckpt559 is the worst point on all three). REVERTED to 0 (live yaml, ~iter 577) — the pre-committed kill action; ckpt524 REMAINS the canonical baseline. LESSON (2nd instance after the w_sf_move upweight): reweighting existing rows toward hard/disagreement rows at high mass is net-negative at 46M scale even when the target rows measurably resolve; a low-w retry (w≈5, ~8% mass) is a legitimate future experiment but is DOMINATED by FEN seeding (#108) for the next window — seeding adds NEW states instead of reweighting old ones. RECOVERY WATCH: sampling weights reverted with the data itself clean, so the head should re-fit fast; read panel v1 + value_regret at ~iter 610–620 (~1 day) — significant retrace toward ~72 ⇒ continue; still significantly worse vs ckpt524 ⇒ salvage-restart from the banked ckpt524 (`scratchpad/recovery_read_ckpt524`, offline-recovery playbook). **RECOVERY READ @ckpt609 (07-06; dumps `scratchpad/live_read/vdump_ckpt609.jsonl`): rule says CONTINUE — value 85.96→75.86, paired vs ckpt524 −3.72 [−13.88, +6.02] NS overall (was −13.81 SIG at the kill read) ⇒ not "still significantly worse", no salvage. Slices are NOT symmetric: endgame still −13.70 [−27.26, −1.56] SIG worse vs 524, middlegame +16.18 [+1.75, +33.14] SIG better; panels v1 23/35, v2 70/113 (healing from 83, still above the 47 bank). The residual endgame/tail damage is exactly the blind-spot mode → handed to the FEN-seeding window with ckpt609 as its pre-activation baseline (banked: value 75.86, v1 23/35, v2 70/113).** |
| sf_p0 + regret teacher weights | June | proven (see WORKED); leave alone |
| **Blind-spot FEN pool += 64 cheese-match-mined seeds** (`opening_fen_list_path`→`data/blindspot_fens_retire_732_cheese64.txt`, retire_732's 80 + 64 new = 144; live yaml, restart-gated at time of activation — see the live-reload mechanism entry directly below, which removes this gate going forward) | 2026-07-10, restart after iter 742/ckpt736 | Same lever as the WORKED v1 batch and the v1→v2 scale-up row above — adds a THIRD tranche, this one mined from the cheese_20260709_2357_checkpoint_000722 match (60 games, 48 losses) via the extended `scripts/mine_blindspot_seeds.py` (this session): 48 first-decisive-collapse seeds (existing method, one per loss) + 16 value-mismatch seeds (NEW criterion — deep-SF expected score vs DeepFin's own self-reported move-log eval, both converted to a common [-1,1] expected-score scale via `cp_to_wdl`/the algebraic inverse of `q_to_cp` so the comparison isn't a raw-cp scale artifact; gap ≥0.5, only emitted when ≥8 plies from that game's collapse ply so both signals get captured without near-duplicate seeds). All 64 deep-SF-vetted at mining time (300k nodes + 6-man TB, same standard as the v2 68-seed harvest). 0 overlap with panel v1/v2 (excluded via `--holdout`) and 0 overlap with v1/v2/v3/retire_722 (excluded via `--existing`); verified 0 placement-key dupes within the merged 144-seed file and all 144 lines parse via the live loader's own reject predicate. Baseline @ckpt732 (the last monitor read pre-merge, `scratchpad/live_read/monitor/monitor.log`): v1 BLIND 13/35 (severity paired +0.24 SIG vs ckpt609), v2 BLIND 43/113, value_regret 73.5cp (paired vs ckpt609 +2.41 [-5.76, +10.37] NS). PRIMARY: held-out panel v1 BLIND count/severity holds or improves vs 13/35 (these 64 seeds are not panel members — same generalization logic as v1/v2). GUARDRAIL/KILL: `value_regret` paired CI vs the ckpt732 baseline worse by >2cp (CI excludes 0 on the worse side) → revert. WATCH: pool nearly doubles (80→144) so doled selfplay-seeded games/iter roughly doubles too at `opening_fen_dole_per_iter: 1` — dole is selfplay-only by construction (PID-safe, established), but if `sf_pid` games <30 or the airbag trips, that's an unexpected finding, dial dose down. Readout via the existing monitor_fen.sh cadence (next scheduled read once checkpoint reaches 742, no new tooling). REVERT: `opening_fen_list_path` → `data/blindspot_fens_retire_732.txt` + restart — no salvage snapshot needed (pure opening-seed-pool change, doesn't touch weights/optimizer/replay). |
| **`opening_fen_list_path` live-reload mechanism** (server code, no training-target change) | 2026-07-10 | Fixes the gap this session's cheese64 addition (row above) worked around by restarting: `opening_fen_list_path` was one of `_LAUNCH_FIXED_ASSET_PATH_KEYS` (`trainable_config_ops.py`) — captured once at server-process launch, silently inert to any later yaml edit (the strict live-reload validator only *warns*, never errors, so this was easy to miss). Extended the SAME manifest/publish-dir pattern already proven for `model.pt`/stockfish binaries: `_publish_distributed_trial_state` (`distributed_runtime.py`) now copies the current source file into the trial's publish dir under a fixed name every iteration and advertises a freshly-computed sha (unconditional copy + direct `sha256_file`, not the mtime-keyed `_sha256_cached` used for the launch-fixed book paths — this asset is expected to change often, and mtime-tie risk on a small frequently-edited file was judged not worth the coarseness); a new trial-scoped route `GET /v1/trials/{trial_id}/opening_fen_list` (`server/app.py`) serves it via `_artifact_from_publish`, mirroring the existing stockfish route; worker download path updated to the trial-scoped endpoint (`worker.py`) for consistency with the model-download fallback pattern. Removed `opening_fen_list_path` from `_LAUNCH_FIXED_ASSET_PATH_KEYS` and the now-dead `--opening-fen-list-path` CLI plumbing (`run_server.py`, `harness.py`). Net effect: from this deploy forward, a yaml `opening_fen_list_path` change (new tranche, auto-retirement, a full swap) takes effect on the NEXT manifest publish — no restart. Coverage: `tests/test_distributed_selfplay_backpressure.py::test_opening_fen_list_path_swap_takes_effect_without_restart` (path change, same running app instance) and `::test_opening_fen_list_path_inplace_edit_detected` (in-place content edit at the same path, exercises the freshness fix). Full suite green pre-merge. No kill threshold — this is infra, not a training-target hypothesis; the FIRST real-world proof is the next time the pool changes without a restart, watch for it in the monitor_fen.sh cadence. **PROVEN 2026-07-11: the reinstate512 list swap (138→141) republished to `opening_fen_list_live.txt` (141 seeds verified) on the first manifest publish after the yaml edit — the mechanism works.** |
| **512-net seed feed += 54 mined seeds (v4 pruned + harvest_v5 cheese)** (`opening_fen_list_path`→`data/blindspot_fens_fed_v4v5_ck58.txt`, retire_58's 101 + 26 v4 + 28 harvest_v5 = 155, deduped by FEN; live yaml, live-reload — no restart needed for the seed part, but co-activated with the 07-13 broker-self-abort restart) | 2026-07-13, ~iter 64/ckpt63 (trial 4c17c) | Same lever as the WORKED v1/v2/cheese64 tranches — the FIRST substantial seed addition on the 512 net since the swap (retirement has only SHRUNK the pool 141→101 with `refed=0`; nothing has been added, and the 61 already-mined seeds sat unused in scratchpad). 26 = v4 pruned (KDEF×17 + gap-cluster CONV_UP×9, per `scratchpad/motifs/motif_report.md`, endgame cluster dropped as panel-resolved); 28 = harvest_v5 mined from the 20-game ckpt722 cheese block (`scratchpad/harvest_v5/mined_cheese512_20260712.txt`) incl. the value-mismatch band (sf vs own-eval gap ≥0.5) that directly targets the **flat value head** (this readout's motivation: policy panels improving 82→70/113 v2 BLIND ck48→58 but value regret flat 85.8→86.3cp). Merge validated: 0 parse failures, 0 FEN dupes, all 155 replay via `seed_board_from_line`. Carried forward automatically — `blindspot_retire_step` reads this file as its pool next read (ckpt 68). Baseline @ckpt58 (`scratchpad/live_read/monitor/monitor.log`): v1 BLIND 24/35, v2 BLIND 70/113, value_regret 86.3cp (vs_boot512 paired −9.36 [−20.25,+2.37] NS). PRIMARY: held-out panel v1/v2 BLIND holds or improves vs 24/35, 70/113 (these seeds are non-panel — generalization logic). VALUE YARDSTICK: `value_regret --max-positions 2000` paired CI vs the ckpt58 dump (`scratchpad/live_read/monitor/vdump_58.jsonl`) — success = improves (CI excludes 0 on the better side); the value-mismatch seeds are the bet that value moves. GUARDRAIL/KILL: value_regret paired worse by >2cp (CI excludes 0 on the worse side) → REVERT. WATCH: pool 101→155 so doled selfplay-seeded games/iter rises ~50% at dose 1 — dole is selfplay-only (PID-safe); if `sf_pid` games <30 or airbag trips, dial dose down. Readout via existing monitor_fen.sh cadence (ckpt 68+). REVERT (live, instant): `opening_fen_list_path` → `data/blindspot_fens_retire_58.txt` (pure seed-pool change, no weights/optimizer/replay touched, no salvage snapshot needed). Confound: co-activated with the broker-self-abort restart (code-only, no training-target effect) and the speedup-bundle readout window — value/panel movement is attributable to the seeds; throughput is the bundle. |

## Analysis findings (offline, no live change)

**Selfplay Stockfish-wait overlap experiment -- UNREAD / DEFAULT-OFF
(2026-07-11).** Live 512x16 worker telemetry attributed roughly 2700-3200%
cumulative thread time to the combined curriculum-SF finish phase versus only
38-60% to network work: many selfplay threads exhaust their runnable slots and
block for a high-node Stockfish opponent move. Hypothesis: continuous-mode
slot oversubscription keeps independent selfplay/network work runnable while
curriculum slots wait, increasing completed games/hour without changing any
search, Stockfish, PID, target, or per-game semantics. Code ships with
`slot_oversubscribe: 1.0` (exact current behavior); activation is a separate
restart-gated config change. ONE deciding yardstick: compare four consecutive
steady 60-second worker telemetry windows at 1.0 against four at 2.0 on the
same 512x16 topology and model, using aggregate `complete_gps` as primary and
the new `sf_block_starved` phase share as mechanism check. Pre-committed
SUCCESS: candidate median completed games/s >=1.15x baseline AND median
`sf_block_starved` share falls >=50%. KILL: completed games/s <1.05x, any
per-game synthetic parity failure, worker RSS grows by >4 GiB, or broker
request p95 grows >20%. Otherwise MIXED and revert to 1.0. Strength guards are
unchanged by construction; record SF node/regret/PID values as provenance.
Revert is `slot_oversubscribe: 1.0` plus restart. No salvage snapshot is needed
because scheduling changes game concurrency only and the revert is immediate;
do not overlap activation with another data/config experiment.
Activation watch item (review 2026-07-11): the factor multiplies ALL slots
including curriculum, so concurrent SF opponent queries scale ~linearly with
it — watch SF pool queue depth / curriculum pending depth alongside
`sf_block_starved`; deepened SF queueing is the plausible route to the broker
p95 / RSS kill criteria. Multiply-all is deliberate: oversubscribing only
selfplay-classified slots would shift the effective selfplay/curriculum data
mix and turn this scheduling-only change into a data-mix change.

**UCI abort root-snapshot reuse experiment -- UNREAD (2026-07-13).** The
match time-manager calls `_filtered_root_visits` in `_root_visit_lead` and then
again in `_move_is_decided` on every ordinary abort check (and a third time on
the non-leading-survivor branch), crossing the native boundary and allocating
duplicate NumPy arrays despite the helper's once-per-chunk contract. Hypothesis:
fetch one filtered root snapshot in `_abort_ready` and pass it through lead,
forced-move, visit-gap, and stability decisions. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_uci_abort_snapshot.py
--children 32 --iterations 200000 --rounds 9`. SUCCESS: candidate/reference
median time <=0.80, exact abort decisions and stability state over forced,
leading, trailing, filtered, and provable-bank cases, exactly one root snapshot
per check, focused UCI time/search tests and lint pass. KILL: ratio >0.95 or any
decision/state mismatch; otherwise MIXED and retain only if the final code is a
net simplification. Match scheduling only; no training/data/config change or
salvage.
**VERDICT: FAILED and reverted.** The exact nine-round alternating yardstick
measured 8.366475194s with repeated snapshots and 8.284874931s with one shared
snapshot, ratio **0.990247**: only 0.98% faster, beyond the pre-committed 0.95
kill threshold. Native snapshot calls did fall exactly 399,998 -> 200,000 and
abort decisions matched, confirming the duplicate read but showing it is not a
material cost. The optional snapshot plumbing was not a net simplification, so
remove the runtime and harness changes; retain existing match behavior.

**UCI discarded chunk-result elision experiment -- UNREAD (2026-07-11).**
Match search calls `run_gumbel_root_many_c` once per search chunk, but
`SearchWorker._run_gumbel_chunk` consumes only the selected action, value,
tree, and root id. The generic RL-facing function still constructs an improved
full policy and legal mask on every chunk; at 8,192 nodes and 512-node chunks
that discarded Python/NumPy work repeats 16 times per move. Hypothesis: an
explicit `return_policy=False` match path that computes only the deterministic
best action and its child value improves single-game search throughput without
changing search state or decisions. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_uci_chunk_results.py --nodes
8192 --chunk-sims 512 --rounds 7 --repeats 5`. The harness alternates generic
and lean paths on identical positions with a deterministic evaluator and fresh
trees. Pre-committed SUCCESS: lean/generic median simulations/s >=1.03x, and
every paired run has identical selected-action, value, root-child-visit, and
tree-state hashes. Otherwise FAILED and retain the generic result construction.
Focused Gumbel-C and UCI search tests plus repo lint must pass. This changes
match result packaging only; RL/selfplay keeps the default full-policy return.
No training/data change and no salvage snapshot.
**VERDICT: FAILED and reverted.** The exact seven-round alternating yardstick
measured generic result construction at 35,638 simulations/s and the lean path
at 36,701 simulations/s, a **1.029830x** ratio versus the precommitted
1.030000x requirement. Every run retained exact action/value/root-visit/tree
state hash `0f95b11ec58a0604`, and focused tests plus lint passed. The discarded
policy/mask packaging is measurable, but this isolated gain missed the gate by
0.017 percentage points; do not carry the extra API/runtime branch alone.

**Zero-floor Aurora telemetry sampling experiment -- UNREAD (2026-07-12).**
Production uses `aurora_uw_floor: 0`, but every optimizer step still computes
FP32 weight/update Frobenius norms for every Aurora matrix, builds ratio/scale
lists, runs quantiles, and synchronizes the resulting telemetry to the host.
Those values do not influence an update when the floor is zero; only the final
iteration report and periodic TensorBoard samples consume them. Live iters
32--36 attribute 33--35% of trainer wall time to `optimizer_step_time_s`.
Hypothesis: collecting update/weight telemetry only on TensorBoard-reporting
steps and the final requested step removes this semantically dead work while
preserving exact optimizer/model state and fresh final metrics. ONE deciding
yardstick: `PYTHONPATH=. taskset -c 15 python3 scripts/bench_aurora_stats.py
--rounds 7 --steps 10 --matrices 8 --size 256`. The harness alternates
always-on reference and sampled telemetry on identical deterministic CPU
matrices/gradients, timing complete Aurora steps. Pre-committed SUCCESS:
sampled/reference median optimizer throughput >=1.03x, every parameter and
momentum-buffer hash is exact after every round, sampled final UW statistics
equal the reference, and Aurora/trainer/SODA focused tests plus repo lint pass.
Otherwise revert and retain per-step telemetry. When `aurora_uw_floor > 0`,
per-matrix ratios remain mandatory for update scaling; the optimization may
skip only summary collection between reporting steps. No target/data/config
meaning changes and no salvage snapshot.
**VERDICT: FAILED and reverted.** The exact alternating yardstick measured
1.465272s for always-on telemetry and 1.519525s for sampled telemetry, only
**0.964296x** reference throughput versus the 1.03x gate. Every round retained
exact parameter/momentum hash
`b6745aa3faab08b70a476de8b7fe730a6b0253d6f64dd3d92bc877c0ec0515b3`,
and final UW statistics matched exactly. On this pinned CPU mechanism test the
norm/quantile work was hidden under Aurora's matrix iteration and external
variance; do not keep the extra telemetry control flow without a GPU result
that clears a separately registered gate.

**Aurora polar addmm fusion experiment -- UNREAD (2026-07-12).** Production
Aurora (`polar_express`, 8 steps, FP16 on CUDA) currently evaluates each
quintic step as `a*x + (b*XXt + c*(XXt@XXt))@x`, launching separate
pointwise scale/add kernels around both GEMMs for every selected matrix.
`torch.addmm` expresses the same polynomial while allowing each scale/add to
fuse into its GEMM epilogue. Hypothesis: two `addmm` calls per polar step
materially reduce full polar-factor time without a meaningful numerical
change. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_aurora_addmm.py --rounds 7 --repeats 5 --matrices 8 --rows 256
--cols 512 --steps 8`. The harness alternates the existing expression and
candidate on identical deterministic FP32 CPU matrices (the implementation's
CPU fallback), timing the full 8-step polar transform. Pre-committed SUCCESS:
candidate/reference median throughput >=1.05x, every result is finite, maximum
absolute error <=2e-5 and maximum relative error <=2e-4, a full Aurora-step
parity test passes at `rtol=2e-4, atol=2e-5`, and Aurora/trainer tests plus repo
lint pass. Otherwise revert. This is an algebraically equivalent optimizer
implementation change; no target/data/config meaning and no salvage snapshot.
**VERDICT: FAILED and reverted.** The exact seven-round alternating yardstick
measured 33.641 reference matrices/s and 41.251 candidate matrices/s, a
**1.226199x** throughput ratio that cleared the speed gate. Maximum absolute
error was only `6.85453415e-07`, and the full Aurora optimizer-step parity test
passed at the registered combined tolerance, but raw maximum relative error
was `0.738535106` around near-zero outputs versus the pre-committed `2e-4`
limit. The candidate therefore failed the complete acceptance rule. Do not
promote this algebraic fusion without a newly registered evaluation whose
numerical criterion explicitly handles near-zero values and CUDA FP16 behavior.

**Aurora polar addmm CUDA replication -- UNREAD (2026-07-12).** The first
fusion gate found a real 1.226x CPU gain and sub-micro absolute differences,
but correctly failed its pre-committed raw relative-error limit because values
arbitrarily close to zero make that metric unbounded. Hypothesis: the same
algebraic `addmm` fusion also improves production CUDA FP16 Polar Express, with
differences bounded at the scale of FP16 rounding and no meaningful optimizer
update drift. ONE deciding yardstick has two required arms: rerun the registered
CPU command, then run `PYTHONPATH=. python3 scripts/bench_aurora_addmm.py
--device cuda --rounds 9 --repeats 10 --matrices 16 --rows 512 --cols 896
--steps 8`. Both arms alternate order and synchronize their device. SUCCESS:
candidate/reference median throughput >=1.05x in both arms; CPU maximum
absolute error <=2e-5 and `torch.testing.assert_close` passes at `rtol=2e-4,
atol=2e-5`; CUDA maximum absolute error <=2e-3, normalized L2 error <=5e-4,
and cosine similarity >=0.999999; full production-shaped Aurora update parity
passes at `rtol=5e-3, atol=2e-3`; focused tests and lint pass. Otherwise revert.
The CUDA arm allocates only small eager tensors and does not compile or change
the live trainer. This remains algebraically equivalent implementation work;
no target/data/config meaning and no salvage snapshot.
**VERDICT: FAILED and reverted.** The corrected CPU arm measured 36.162
reference and 41.279 candidate matrices/s (**1.141526x**), maximum absolute
error `6.85453415e-07`, normalized L2 `1.92288167e-06`, cosine 1.0, and passed
full-update parity. The production-shaped CUDA FP16 arm measured 299.546 vs
398.649 matrices/s (**1.330842x**) and passed maximum absolute error
(`0.000793457031`) plus full-update parity, but normalized L2 error was
`0.0029939115` versus the `5e-4` gate and minimum cosine was `0.99999547`
versus `0.999999`. The full two-epilogue fusion is fast but fails the complete
numerical rule; test its two independent fusion sites separately rather than
accepting larger accumulated FP16 rounding drift.

**Aurora partial addmm fusion dose ladder -- UNREAD (2026-07-12).** Full
two-epilogue fusion is 1.33x faster on CUDA but accumulates too much FP16 polar
drift. Hypothesis: fusing only the inner polynomial (`b*XXt+c*XXt@XXt`) or only
the outer update (`a*x+polynomial@x`) retains a useful kernel-launch reduction
and one candidate stays inside the prior strict numerical limits. ONE deciding
yardstick: extend the same alternating CPU and production-shaped CUDA commands
to compare `inner`, `outer`, and `full` against `reference`. SUCCESS: choose
the fastest partial candidate with >=1.05x median throughput on both CPU and
CUDA, CPU max absolute <=2e-5 and assert-close at `rtol=2e-4, atol=2e-5`, CUDA
max absolute <=2e-3, normalized L2 <=5e-4, cosine >=0.999999, full Aurora
update parity at `rtol=5e-3, atol=2e-3`, focused tests, and lint. Otherwise
FAILED and keep the original expression. This is offline eager math only; no
live trainer mutation, target/data/config change, or salvage snapshot.
**VERDICT: FAILED; no runtime change.** On CPU, inner-only fusion measured
1.064536x and passed every numerical limit, while outer-only measured only
1.043054x and missed the speed gate. On production-shaped CUDA FP16, inner
measured **1.207337x** and outer **1.129477x**, but their normalized L2 errors
were `0.00294561288` and `0.00297925738`, and cosine minima were
`0.999995589` and `0.99999547`; both miss the same `5e-4` and `0.999999`
limits that rejected full fusion. Maximum absolute errors stayed below 0.001
and full-update parity passed, but neither partial candidate satisfies the
complete rule. Remove the dose-ladder harness and retain the original
operation ordering.

**Pending-slot direct-membership simplification -- UNREAD (2026-07-12).** Live
worker telemetry attributes roughly 20--30% of active selfplay orchestration
thread-time to classification. Every scheduler pass with outstanding SF work
copies `pending_sf_moves` into a temporary set, then filters three short index
lists; the dictionary already provides O(1) key membership. Hypothesis: test
membership directly against the dictionary, removing one allocation/copy per
pass with identical results. ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_pending_slot_filter.py --batch-size 32 --pending 24 --rounds 9
--iterations 1000000`, alternating set-copy and direct-dict filtering with
production-sized deterministic groups. SUCCESS: direct/reference median
throughput >=1.10x, exact outputs for randomized pending subsets, focused
selfplay/manager/continuous tests, and lint pass. Otherwise revert. This is a
small Python scheduling simplification; no live activation, data/config
meaning, or salvage snapshot.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
397,463 set-copy filters/s and 460,872 direct-dictionary filters/s, a
**1.159535x** mechanism speedup. One thousand randomized pending subsets
retained exact group outputs. Validation passed 65/65 focused continuous,
threaded, state, timeout, and distributed-backpressure tests after a clean
GCC15 native+LTO extension build; ruff, basedpyright, and vulture are clean.
Keep the allocation-free direct membership checks. This improves only the
classification/filter slice, not whole-worker throughput by 15.9%.

**C classifier direct-list return experiment -- UNREAD (2026-07-12).** The C
selfplay classifier releases the GIL for board scans but then allocates three
NumPy arrays, copies temporary index buffers into them, returns to Python, and
Python immediately allocates three lists and boxes every index through
`.tolist()`. Hypothesis: an optional direct-list result mode preserves the same
GIL-free scan while removing the transient arrays, copies, and Python
conversions. ONE deciding yardstick: after a clean production extension build,
run `PYTHONPATH=. taskset -c 15 python3 scripts/bench_classify_return.py
--batch-size 32 --rounds 9 --iterations 200000`, alternating the current array
return plus `.tolist()` against direct lists on identical CBoards and companion
arrays. SUCCESS: direct/reference median throughput >=1.10x, exact partitions
and `done` mutations across randomized board/flag states, C-classifier/state/
continuous/threaded tests pass, and C warnings plus repo lint are clean.
Otherwise revert. The existing array return remains the default public API;
only internal `SelfplayState` opts into lists. No live activation, data/config
meaning, or salvage snapshot.
**VERDICT: FAILED and reverted.** The exact nine-round alternating yardstick
measured 33,447 classifications/s for the existing NumPy-plus-`.tolist()` path
and 35,807 classifications/s for direct C lists, only **1.070557x** versus the
1.10 gate. One thousand randomized flag states retained exact partitions and
identical `done` mutation, and the GCC15 native+LTO build was warning-clean.
The board scan and GIL transition dominate enough that a 7.1% mechanism gain
does not justify another native API mode; retain the existing return contract.

**Curriculum-only pending-slot filtering experiment -- UNREAD (2026-07-12).**
Pending move futures are created only from `cur_opp_idxs`, and the board does
not advance until that future is removed and applied; therefore every pending
slot remains in the curriculum-opponent partition. The scheduler nevertheless
rebuilds all three net/selfplay/curriculum lists to exclude pending keys on
every pass. Hypothesis: filtering only `cur_opp_idxs` removes two unnecessary
list allocations and scans while preserving the lifecycle invariant. ONE
deciding yardstick: `taskset -c 15 python3
scripts/bench_pending_group_filter.py --batch-size 32 --pending 24 --rounds 9
--iterations 1000000`, alternating three-group and curriculum-only filtering
with randomized partitions whose pending keys are a subset of curriculum.
SUCCESS: candidate/reference median throughput >=1.10x, exact outputs for
1,000 randomized invariant-respecting cases, focused curriculum/continuous/
threaded selfplay tests, and lint pass. Otherwise revert. This is a scheduling
simplification only; no live activation, data/config meaning, or salvage.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
463,425 three-group filters/s and 812,741 curriculum-only filters/s, a
**1.753772x** speedup for the affected scheduler slice. One thousand randomized
partitions with pending keys constrained to the curriculum group retained exact
outputs. The lifecycle invariant is structural: only curriculum indices are
submitted, and their board cannot advance into another partition until the
future is removed and applied. Validation passed 86/86 curriculum-label,
continuous, threaded, state, timeout, and backpressure tests after a clean
GCC15 native+LTO rebuild; lint is clean. Keep the two-list allocation removal;
this is not a claim of 75.4% whole-worker throughput.

**Ready-only Stockfish future scan experiment -- UNREAD (2026-07-12).** Each
nonblocking curriculum poll currently materializes `list(pending.items())`
before checking readiness, then allocates a second list for completed results.
Live workers usually hold roughly 24 pending futures and complete zero or one
per scheduler pass. Hypothesis: collect only ready `(index, future)` pairs while
the dictionary remains unchanged, then remove those pairs in a second phase;
this retains mutation safety while avoiding allocation proportional to all
pending work. ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_pending_future_scan.py --pending 24 --ready 1 --rounds 9
--iterations 1000000`, alternating snapshot-all and ready-only collection.
SUCCESS: candidate/reference median throughput >=1.15x, exact ready/result/
remaining state across 1,000 randomized cases, focused curriculum/continuous
selfplay tests, and lint pass. Otherwise revert. Scheduling-only; no live
activation, data/config meaning, or salvage.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
250,980 snapshot-all scans/s and 313,877 ready-only scans/s, a **1.250606x**
speedup for pending-future collection. One thousand randomized readiness maps
retained exact completed results and remaining dictionary order/state.
Validation passed 74/74 curriculum-label, continuous, threaded, timeout, and
backpressure tests after a clean GCC15 native+LTO extension build; ruff,
basedpyright, and vulture are clean. Keep the ready-only collection; this
improves the polling slice rather than whole-worker throughput by 25.1%.

**Zero-ready async-label poll fast path -- UNREAD (2026-07-12).** Async SF
label polling rebuilds `pending_sf_labels` into a fresh list on every scheduler
pass even when no future is ready; live timing shows this poll runs continuously
while label work is outstanding. Hypothesis: a readiness precheck returns the
unchanged list immediately in the common zero-ready case, avoiding the list
allocation and appends; when any result is ready, the existing resolution path
runs unchanged. ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_pending_label_poll.py --pending 128 --rounds 9 --iterations
100000`, alternating rebuild-all and zero-ready-fast-path polls. SUCCESS:
candidate/reference median throughput >=1.25x for zero-ready polls, exact
pending/completed outcomes across 1,000 randomized readiness patterns, focused
SF-label/continuous/threaded tests, and lint pass. Otherwise revert. Scheduling
only; no live activation, target/config meaning, or salvage.
**VERDICT: FAILED and reverted.** The exact nine-round alternating yardstick
measured 52,153 rebuild-all polls/s and 58,394 zero-ready fast-path polls/s,
only **1.119667x** versus the 1.25 gate. One thousand randomized readiness
patterns retained exact pending/completed outcomes, but the modest zero-ready
gain does not justify an additional scan on polls that do contain ready work.
Keep the single-pass rebuild logic.

**Starved Stockfish wait pre-scan experiment -- UNREAD (2026-07-12).** The
manager first performs a nonblocking ready-only poll, then, when every runnable
slot is still awaiting Stockfish, `finish_pending_curriculum_moves(block=True)`
scans every future with `done()` before constructing the same tuple for
`wait(FIRST_COMPLETED)`. Calling `wait` directly is race-safe: it returns
immediately if a future completed after the preceding poll. Hypothesis:
removing the redundant condition-lock scan reduces starved scheduling overhead
without changing timeout or completion behavior. ONE deciding yardstick:
`taskset -c 15 python3 scripts/bench_stockfish_wait_prescan.py --pending 24
--rounds 9 --iterations 10000`, alternating pre-scan+wait and direct wait with
zero-ready futures, plus a ready-future arm. SUCCESS: direct/reference median
throughput >=1.10x for zero-ready, >=0.90x for one-ready, identical done/not-
done sets, focused curriculum/continuous tests, and lint pass. Otherwise
revert. Scheduling-only; no live/config/data change or salvage.
**VERDICT: FAILED and reverted.** The exact alternating yardstick measured
direct wait at **1.330696x** pre-scan throughput with zero ready futures, but
only **0.600713x** when one future was ready, far below the 0.90 race-case
guardrail. Ready-index outcomes matched exactly. Keep the pre-scan: avoiding a
`wait` call after completion is materially valuable when a future resolves
between the preceding nonblocking poll and the starved blocking path.

**CUDA final-step Aurora telemetry experiment -- UNREAD (2026-07-12).** The
earlier CPU telemetry-sampling arm was noisy/slower, but production runs Aurora
on CUDA and attributes roughly one third of trainer wall time to optimizer
steps. With `aurora_uw_floor: 0`, per-matrix FP32 weight/update norms and UW
quantiles cannot affect parameters; only `train_steps()`'s final metrics consume
`last_uw_stats`. Hypothesis: collect UW telemetry on the final optimizer step
of each train window (and always when the floor is positive), preserving fresh
reported statistics while removing unnecessary CUDA reductions and host
synchronization from preceding steps. ONE deciding yardstick: `PYTHONPATH=.
python3 scripts/bench_aurora_stats_cuda.py --rounds 7 --steps 5 --matrices 8
--rows 512 --cols 896`, alternating always-on and final-only telemetry across
complete production-config Aurora steps. SUCCESS: final-only/reference median
throughput >=1.03x, exact parameter and momentum hashes after every round,
identical final UW statistics, positive-floor behavior unchanged, focused
Aurora/trainer/SODA tests, and lint pass. Otherwise revert. This changes
telemetry scheduling only, not updates, targets, config, or replay; no salvage.
**VERDICT: WORKED.** The exact seven-round CUDA yardstick measured 0.886610s
for five always-on steps and 0.829691s for final-only telemetry, a
**1.068602x** complete-optimizer throughput gain. Every round retained exact
parameter/momentum hash
`b20bd8e4aaabab15427b3c72a09b5e626d77946ded65030d69e7ac0210aa79e0`,
and final UW statistics were identical. Positive-floor tests confirm telemetry
and scaling remain mandatory even when collection is disabled; SODA forwards
the control, and trainer coverage confirms only the final train-window step
collects. Validation: 51/51 Aurora/SODA/trainer tests plus clean ruff,
basedpyright, and vulture. Keep final-step collection; the live run activates
it only after restart onto this code.

**Compiled Aurora Polar Express experiment -- UNREAD (2026-07-12).** Eager
`addmm` fusion was fast but rejected for accumulated FP16 drift. The original
Polar Express expression still launches several pointwise kernels around each
GEMM. Hypothesis: `torch.compile` with dynamic matrix width fuses graph-level
pointwise work while retaining the source arithmetic closely enough to satisfy
the strict numerical gate. ONE deciding yardstick: `PYTHONPATH=.
TORCHINDUCTOR_CACHE_DIR=/tmp/cae-aurora-inductor python3
scripts/bench_aurora_compile.py --rounds 7 --repeats 10 --matrices 8 --rows
512 --steps 8`, alternating eager and compiled transforms across production
FFN widths after warmup. SUCCESS: compiled/eager median throughput >=1.10x,
initial compile <=120s, no more than two unique graph compilations across the
width ladder, max absolute error <=2e-3, normalized L2 <=5e-4, cosine
>=0.999999, full Aurora update parity at `rtol=5e-3, atol=2e-3`, focused tests,
and lint. Otherwise revert. This is offline eager/compiled optimizer math only;
no live activation, target/config/data change, or salvage.
**VERDICT: FAILED and reverted.** With a fresh Inductor cache, compile warmup
took 38.903s and steady compiled throughput was **4.431306x** eager (1178.146
vs 265.869 matrices/s). However maximum normalized L2 error was
`0.00341379573` and minimum cosine `0.999994278`, failing the `5e-4` and
`0.999999` numerical gates despite maximum absolute error below 0.001. The
first measured compiled round also fell to 29.298 matrices/s as additional
production widths compiled, so the two-graph dynamic-width condition was not
met. Do not use Inductor-rewritten Polar Express; pursue exact eager-kernel
CUDA graph replay instead.

**CUDA-graph Aurora Polar Express experiment -- UNREAD (2026-07-12).** The
Inductor arm proved launch overhead is large but altered FP16 arithmetic.
Hypothesis: capture the unchanged eager Polar Express kernels in one CUDA graph
per fixed production matrix shape, copy each update into a static input, and
replay them with bitwise-identical output while eliminating repeated Python and
launch overhead. ONE deciding yardstick: `PYTHONPATH=. python3
scripts/bench_aurora_cuda_graph.py --rounds 7 --repeats 10 --matrices 8 --rows
512 --steps 8`, alternating eager and graph replay across production FFN
widths after capture. SUCCESS: graph/eager median throughput >=1.20x including
static-input copy, exact bitwise output and full Aurora-update parity, total
captured static tensors/workspaces <=2 GiB for the tested width ladder, capture
time <=120s, focused tests, and lint. Otherwise revert. Offline optimizer math
only; no live activation, target/config/data change, or salvage.
**VERDICT: WORKED.** The exact seven-round polar yardstick measured 254.141
eager vs 2,969.265 graph matrices/s, an **11.683548x** replay speedup including
the static-input copy. Eight unique production widths captured in 2.812s and
allocated 330,530,816 bytes (315 MiB), well below the 2 GiB gate. Polar outputs
and full three-pass Aurora updates were bitwise exact. A supplemental complete
optimizer benchmark across the same eight widths measured 0.872756s eager vs
0.224910s graphed for five steps, a **3.880460x** throughput gain with exact
parameter/momentum hash
`250b75c4182bff4c550f97f1bbe8bd7051f9579a55e31fda796747ca800866de`.
Validation passed 52/52 Aurora/SODA/trainer tests including local CUDA graph
cache reuse and explicit eager fallback; lint is clean. Keep the per-optimizer,
per-shape graph cache. First use pays capture once; checkpoint state is
unchanged and the live run activates only after restart.

**Coalesced Aurora finite-check experiment -- UNREAD (2026-07-12).** CUDA
graph replay removes most polar launch overhead, leaving `_aurora_update`'s
Python truth test on `torch.isfinite(out).all()` as one host synchronization
per optimized matrix. Hypothesis: retain each computed update, enqueue all
finite reductions, synchronize once per Aurora parameter group, and apply
parameters only after the combined check passes. This removes N-1 syncs and
also prevents applying earlier valid updates before a later matrix fails. ONE
deciding yardstick: `PYTHONPATH=. python3
scripts/bench_aurora_finite_checks.py --rounds 7 --steps 5 --matrices 8 --rows
512`, alternating per-matrix and coalesced checks on graphed production-config
optimizer steps. SUCCESS: coalesced/reference median throughput >=1.10x,
exact parameter/momentum hashes, injected nonfinite update raises before any
parameter add, focused Aurora/SODA/trainer tests, and lint pass. Otherwise
revert. Optimizer validation scheduling only; no target/config/data change or
salvage.
**VERDICT: WORKED.** The exact seven-round complete-optimizer yardstick
measured 0.152192s with per-matrix checks and 0.134966s with one group check,
a **1.127629x** throughput gain on top of CUDA graph replay. Every round
retained exact parameter/momentum hash
`250b75c4182bff4c550f97f1bbe8bd7051f9579a55e31fda796747ca800866de`.
Failure injection confirms a nonfinite later matrix raises before any pending
parameter update is applied; valid CPU and CUDA paths remain exact. Focused
Aurora tests and lint pass. Keep coalesced checks; retained update tensors add
temporary memory roughly equal to the Aurora parameter group, acceptable on
the 32 GiB production GPU.

**Whole-update Aurora CUDA graph experiment -- UNREAD (2026-07-12).** Polar
graph replay is exact and fast, but each production update still crosses Python
between three polar passes for row scaling and reductions. Hypothesis: capture
the complete `_aurora_update(check_finite=False)` per fixed parameter shape,
then perform the already-coalesced group finite check outside the graph. ONE
deciding yardstick: `PYTHONPATH=. python3
scripts/bench_aurora_update_graph.py --rounds 7 --repeats 10 --matrices 8
--rows 512`, alternating current polar-only graph updates and whole-update
graphs across production FFN widths. SUCCESS: whole/polar-only median
throughput >=1.50x, bitwise exact outputs, capture <=120s and <=2 GiB for eight
widths, complete optimizer exact-state confirmation, focused tests, and lint.
Otherwise revert. Optimizer execution only; no target/config/data change or
salvage.
**VERDICT: WORKED.** The reproducible seven-round comparison measured 595.744
polar-only vs 948.292 whole-update updates/s, a **1.591778x** gain with bitwise
exact outputs. Eight captures took 1.346s and allocated 297,308,160 bytes
(283.5 MiB), below both gates. After integration, the complete five-step
optimizer benchmark measured 0.770955s eager vs 0.065889s whole-update graph,
an **11.700884x** throughput gain with exact parameter/momentum hash
`250b75c4182bff4c550f97f1bbe8bd7051f9579a55e31fda796747ca800866de`.
Captures are keyed per parameter plus algorithm settings, so repeated 512x512
weights cannot alias one static output while group updates are retained.
Validation passed 55/55 Aurora/SODA/trainer tests including repeated-shape CUDA
state parity and the config-driven eager fallback; lint is clean. Keep
whole-update capture and the explicit eager fallback. The direct eight-width
capture allocated 283.5 MiB; production uses parameter-specific outputs, so its
total scales with the selected parameter count. A supplemental 48-parameter
production-count run completed without OOM at 1.233610s eager vs 0.101139s
graphed (**12.197198x**) and exact state hash
`95370418d6dfd4d6382b3474db465398a1ccbd861444715a0209f18edde6e9bd`;
this corrects the earlier 32-matrix estimate (`mlp_out` selects both FFN
matrices plus attention output in each of 16 blocks).

**Aurora AdamW foreach fallback experiment -- UNREAD (2026-07-12).** Whole-
update graphs make Aurora matrices cheap, but the same optimizer still updates
all auxiliary, QKV, embedding, norm, bias, and head parameters through a Python
loop with separate AdamW kernels. Hypothesis: gather each fallback group while
retaining the existing integer `step` and moment state entries, then use
`torch._foreach_*` for decay, moments, denominator, and parameter updates.
ONE deciding yardstick: `PYTHONPATH=. python3
scripts/bench_aurora_adam_foreach.py --rounds 7 --steps 10 --matrices 16
--width 512`, alternating loop and foreach fallback updates on production-
shaped QKV matrices plus vectors. SUCCESS: foreach/loop median throughput
>=1.10x, bitwise exact parameters/moments/steps each round, checkpoint
round-trip compatibility, focused Aurora/trainer/SODA tests, and lint pass.
Otherwise revert. Optimizer execution only; no target/config/data change or
salvage.
**VERDICT: FAILED and reverted.** The first exact-state comparison diverged:
loop hash `b36f0f32d5bf228580fb6062af9683aec71ad8235277a97fe9ebc0a80a2aeed6`
versus foreach hash
`a598a53f25538d6d6dbe37608d9893e9d3b061e51be9507ec670e1ca59107dcd`.
Maximum parameter difference was one float32 ULP (`1.1920928955078125e-07`),
but the pre-committed requirement was bitwise checkpoint/state continuity.
Do not replace the fallback arithmetic with foreach kernels; retain the loop.

**Threaded-dispatcher singleton legal-pack experiment -- UNREAD
(2026-07-12).** `ThreadedDispatcher._submit_batch` always runs two
`np.concatenate` calls for compact legal indices/counts, even when the drain
contains one request. That copies arrays for UCI/single-request match searches
and low-concurrency worker drains while input rows already require pinned-slot
copying. Hypothesis: alias already-int32 singleton legal arrays and retain
concatenation for true multi-request batches. ONE deciding yardstick:
`taskset -c 15 python3 scripts/bench_threaded_singleton_legal_pack.py --batch
512 --legal-per 32 --rounds 9 --iterations 500000`. SUCCESS: candidate/reference
median packaging throughput >=5x, exact arrays, singleton outputs share memory,
focused threaded-dispatcher/UCI/Gumbel tests, and lint pass. Otherwise revert.
Inference packaging only; no target/config/data change or salvage.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
160,898 copy packs/s and 1,707,451 alias packs/s, a **10.611997x** mechanism
speedup while avoiding 67,584 bytes per 512-position request. Every array
matched exactly and singleton outputs shared memory with their sources.
Validation passed all 108 focused threaded-dispatcher, coalescer, GPU
dispatcher, Gumbel, root-parallel, UCI engine, and searchmoves tests after a
clean GCC15 native+LTO extension build; lint is clean. Keep the singleton alias
path; multi-request drains retain concatenation and bucket padding still owns
its required counts array.
**Threaded-dispatcher compact legal-length reuse experiment -- UNREAD
(2026-07-12).** Compact dispatch currently reduces every `legal_counts` array
once to size the submitted policy output and again per request while scattering,
although the corresponding one-dimensional `legal_flat.size` is already the
exact policy length. The inner evaluator still validates
`sum(legal_counts) == legal_flat.size` before launching inference. Hypothesis:
reuse the validated flat lengths for output sizing and slice offsets, removing
NumPy reductions from every compact submit/scatter without changing validation
or layout. ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_threaded_legal_length.py --batch 512 --legal-per 32 --requests 16
--rounds 9 --iterations 500000`. SUCCESS: candidate/reference median mechanism
throughput >=3x, exact totals and randomized per-request policy slice boundaries,
focused threaded-dispatcher/UCI/Gumbel tests, and lint pass. Otherwise revert.
Inference bookkeeping only; no target/config/data change or salvage.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
33.111050s for repeated count reductions versus 1.404287s for flat-length
reuse, a **23.578549x** mechanism speedup. The checksum was exactly
16,329,000,000 and all 16 randomized per-request cumulative policy boundaries
matched. A clean GCC 15 native+LTO build and all 109 focused threaded-
dispatcher, coalescer, GPU-dispatcher, Gumbel, root-parallel, UCI engine, and
searchmoves tests pass, including the new concurrent compact-request slice
regression; ruff, basedpyright, and vulture are clean. Keep the length reuse.

**Batch-coalescer compact scatter-length reuse experiment -- UNREAD
(2026-07-12).** The compiled UCI/match-play `BatchCoalescingDispatcher`
validates `sum(legal_counts) == legal_flat.size` before enqueue, but its
submitter thread reduces each request's counts again to advance compact-policy
slice offsets. Hypothesis: reuse each request's validated flat length during
scatter, removing a NumPy reduction from every compact match request with
identical slices. ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_threaded_legal_length.py --batch 512 --legal-per 32 --requests 16
--rounds 9 --iterations 100000`; its per-request scatter arm is the same
bookkeeping operation. SUCCESS: candidate/reference median mechanism throughput
>=3x, exact randomized slice boundaries/checksum, focused coalescer/UCI tests,
and lint pass. Otherwise revert. Match-inference bookkeeping only; no
target/config/data change or salvage.
**VERDICT: WORKED.** Nine alternating rounds measured 5.465108s for repeated
count reductions versus 0.243004s for flat-length reuse, a **22.489759x**
mechanism speedup, with exact randomized slice boundaries and checksum
3,265,800,000. All 40 focused coalescer/UCI engine/searchmoves tests pass;
ruff, basedpyright, and vulture are clean. Keep the flat-length scatter path.

**SlotBroker compact legal-length reuse experiment -- UNREAD (2026-07-12).**
The production shared-GPU broker must reduce counts once while validating
untrusted slot metadata, but then reduces them again while building compact
offsets and a third time in the dense-policy fallback gather. Live worker
telemetry shows about 79% compact legal requests. Hypothesis: retain the
trust-boundary reduction and use each copied flat array's exact length for the
two post-validation bookkeeping passes. ONE deciding yardstick: `taskset -c 15
python3 scripts/bench_threaded_legal_length.py --batch 512 --legal-per 32
--requests 16 --rounds 9 --iterations 100000`; the request-partition arm is the
same repeated offset operation. SUCCESS: candidate/reference median mechanism
throughput >=3x, exact randomized boundaries/checksum, focused SlotBroker and
shared-broker tests, and lint pass. Otherwise revert. Inference bookkeeping
only; validation, model outputs, targets, and data remain unchanged; no salvage.
**VERDICT: WORKED.** Nine alternating rounds measured 5.615790s for redundant
count reductions versus 0.243942s for validated flat-length reuse, a
**23.021000x** mechanism speedup, with exact randomized boundaries and checksum
3,265,800,000. All 31 focused SlotBroker, GPU-dispatcher, and multi-GPU
dispatcher tests pass; ruff, basedpyright, and vulture are clean. Keep both
post-validation flat-length reuse sites.

**SlotBroker compact metadata view experiment -- UNREAD (2026-07-12).** The
broker owns every REQUEST slot until it publishes RESPONSE, but currently copies
its compact counts/indices out of shared memory, then concatenates and converts
those snapshots again before GPU transfer. Live batches average roughly 1.5-2.3
slots and about 79% of requests are compact. Hypothesis: retain read-only shared-
memory views for the broker-owned request lifetime and only allocate the required
combined int64 GPU metadata, aliasing that conversion for singleton batches.
ONE deciding yardstick: `taskset -c 15 python3
scripts/bench_slot_broker_legal_metadata.py --rows-per-slot 32 --legal-per 32
--slots 1 2 4 --rounds 9 --iterations 20000`. SUCCESS: candidate/reference
median throughput >=1.15x for the deciding two-slot arm, no 1/4-slot arm slower
than 1.05x, exact counts/indices/rows/offsets, focused SlotBroker/shared-memory
tests, and lint pass. Otherwise revert. Metadata ownership/copy scheduling only;
validation and inference outputs remain unchanged, with no live activation or
salvage.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
reference/candidate medians of 0.467347/0.259311s for one slot (**1.802265x**),
0.651944/0.553719s for the deciding two-slot arm (**1.177392x**), and
1.032357/0.882983s for four slots (**1.169169x**). Counts, indices, expanded
rows, and offsets matched exactly. A clean GCC 15 native+LTO build and all 31
focused SlotBroker, GPU-dispatcher, and multi-GPU tests pass; ruff,
basedpyright, and vulture are clean. Keep the request-lifetime metadata views.

**SlotBroker pinned legal-metadata transfer experiment -- UNREAD
(2026-07-12).** The compact broker sends int64 legal indices and expanded row
IDs to CUDA from ordinary NumPy allocations; `torch.as_tensor(...,
device=cuda)` therefore performs synchronous pageable transfers before every
legal forward. Hypothesis: reusable pinned host tensors, filled from the
prepared arrays and transferred nonblocking, reduce complete CPU-copy plus H2D
latency at live compact request sizes. ONE deciding yardstick: `PYTHONPATH=.
python3 scripts/bench_slot_broker_pinned_metadata.py --legal 2048 8192 --rounds
9 --iterations 2000`, alternating current pageable transfers and pinned staging
with a synchronization per broker-shaped pair. SUCCESS: pinned/reference median
time <=0.90x at both sizes, exact GPU values, and full-capacity static pinned
buffers <=128 MiB; focused broker tests/lint must also pass. KILL: either size
>1.00x or any parity failure; otherwise MIXED and do not ship without a second
gate. Offline transfer scheduling only; no live activation, target/data change,
or salvage.
**VERDICT: WORKED.** Nine alternating rounds measured pageable/pinned medians
of 0.653344/0.339819s at 2,048 legal entries (ratio **0.520122**) and
0.830034/0.343559s at 8,192 entries (ratio **0.413909**), including the CPU
staging copy and a synchronization after every broker-shaped transfer pair.
GPU values matched exactly. At the production 19,040-position capacity, the
two `int64` staging tensors consume 77,987,840 bytes (74.4 MiB), below the
128 MiB gate. A clean GCC 15 native+LTO build and all 32 focused SlotBroker,
GPU-dispatcher, and multi-GPU tests pass, including exact CPU/CUDA padded-row
integration; ruff, basedpyright, and vulture are clean. Keep CUDA-only pinned
staging; the CPU path retains its direct NumPy-backed tensors.

**SlotBroker dense-fallback metadata reuse experiment -- UNREAD
(2026-07-12).** With compiled legal-row forward still gate-off, production
computes the dense policy and gathers legal logits afterward. The broker already
builds combined legal row/index arrays before choosing that path, but the fallback
then repeats every row expansion and concatenation and transfers the duplicate
pageable arrays to CUDA. Hypothesis: reuse the validated combined arrays and the
new pinned staging buffers, removing duplicate CPU construction on the currently
active compact path. ONE deciding yardstick: `PYTHONPATH=. python3
scripts/bench_slot_broker_fallback_metadata.py --slots 2 --rows-per-slot 32
--legal-per 32 --rounds 9 --iterations 1000`. SUCCESS: candidate/reference
median complete rebuild+transfer time <=0.50x, exact GPU rows/columns, focused
broker CPU/CUDA fallback tests and lint pass. Otherwise revert this extension;
the independently successful legal-row staging result remains valid. Inference
bookkeeping only; dense model arithmetic and outputs remain unchanged, with no
live activation or salvage.
**VERDICT: WORKED.** Nine alternating rounds measured 0.427115s for duplicate
row expansion/concatenation plus pageable transfer versus 0.179773s for reuse
plus pinned transfer, ratio **0.420900**, on 2,032 compact entries across two
live-shaped slots. GPU rows and columns matched exactly. All 33 focused
SlotBroker, GPU-dispatcher, and multi-GPU tests pass, including exact CPU/CUDA
dense-fallback and padded-row cases; ruff, basedpyright, and vulture are clean.
Keep reuse of the already-built combined metadata on the active dense fallback.

**SlotBroker direct compact-policy gather experiment -- FAILED and reverted
(2026-07-12).** Production compiled legal forward remains gate-off, so the
broker expands each 1,858-wide policy to 4,672 columns before gathering legal
logits. ONE deciding yardstick was nine alternating rounds of a 64-real/128-
padded-row, 32-legal-move BF16 CUDA benchmark; pre-committed SUCCESS required
candidate/reference <=0.50 with bitwise parity. A raw full-to-compact GPU remap
was exact for real legal indices but measured only 0.651867 and did not preserve
the existing `-1e9` result for in-range legacy padding indices. A separately
pre-registered modest-gain replication (<=0.80) measured 0.689158 for that
incomplete shortcut; the semantics-preserving mask/clamp/gather version was
**2.002723x** as slow as reference and a sentinel-column version was
**1.206282x**. A final pre-registered CPU-remap+pinned-gather arm preserved the
old fallback for invalid metadata and was bitwise exact, but measured 0.841692,
short of its <=0.80 gate while adding another metadata list and branch. Keep the
current expansion/gather. The remaining high-upside route is the registered
compiled legal-forward benchmark after live Ray GPU trees stop; one-off harnesses
were removed.

**Native classifier result-boundary profile -- UNREAD (2026-07-12).** Live
workers attribute roughly 10-30% cumulative thread time to classification. The
C fast path allocates three NumPy index arrays, then Python immediately calls
`.tolist()` on each. Hypothesis: list conversion is a material lower bound on
the removable C/Python result-boundary cost and justifies a direct-list native
ABI. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_classify_result_boundary.py --games 384 --rounds 9 --iterations
20000`, alternating raw native arrays and the production three-list conversion
on identical nonterminal boards. SUCCESS screen: list path >=1.15x raw-array
time and conversion delta >=10us per 384-game call, exact group-count checksum;
then implement and separately benchmark a direct-list candidate. Otherwise
FAILED and retain the current ABI. Offline CPU boundary profile only; no live,
target, config, or data change.
**VERDICT: FAILED; no ABI change.** Nine alternating rounds measured 6.327216s
for raw native index arrays and 6.353953s with the production three `.tolist()`
conversions, only **1.004226x** total and 1.337us conversion delta per 384-game
call. Both are far below the 1.15x and 10us gates, with exact checksum
7,680,000. Result containers are not the live classification bottleneck; retain
the NumPy ABI and look inside terminal detection instead. The one-off harness
was removed.

**Native classifier redundant terminal-check experiment -- UNREAD
(2026-07-12).** All three selfplay `CBoard.push_index` paths immediately set
`done_arr` after forced-network, searched-network, and curriculum-Stockfish
moves, while opening resolution rejects terminal starts. Nevertheless the C
classifier reruns full `cboard_is_game_over` on every live slot every scheduling
pass. Hypothesis: treat `done_arr` as the authoritative terminal invariant in
classification and retain only relative `max_plies` timeout detection. ONE
deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_classify_terminal_check.py --games 384 --rounds 9 --iterations
20000`, alternating the same native function with terminal recheck on/off over
identical nonterminal boards. SUCCESS: unchecked/reference throughput >=1.50x,
exact partitions/timeouts, and focused tests prove terminal done propagation for
all three push paths plus broad selfplay/native tests and lint. KILL: throughput
<1.20x or any invariant gap; otherwise MIXED and do not ship. Offline scheduling
implementation only; no target/data/config/live change or salvage.
**MECHANISM VERDICT: WORKED.** Nine alternating rounds measured 6.165308s with
redundant terminal rechecks and 0.095058s with authoritative `done_arr`, a
**64.858054x** classifier speedup with exact partition checksum 7,680,000.
Forced-network and curriculum-terminal push tests now pin their immediate
`done_arr` updates; native searched-network terminal propagation is covered by
`batch_process_ply` checkmate parity, and the Python searched-network path keeps
its direct post-push check. The C ABI test pins checked-default versus explicit
authoritative-done behavior. All 95 focused state/FEN/network/native tests plus
the 18-test CPU selfplay-to-training end-to-end smoke pass after a clean GCC 15
native+LTO build; ruff, basedpyright, and vulture are clean.
**FINAL VERDICT: WORKED.** Selfplay uses authoritative done mode; the public C
ABI retains its checked default for conservative external callers. This is a
mechanism result, not a 64x whole-selfplay claim; live phase telemetry after the
next natural restart will quantify the end-to-end reduction.
**PRE-ACTIVATION LIVE BASELINE:** the latest 20 steady phase windows from each
of four workers (80 total, 2026-07-12 23:50) have pooled median cumulative
`classify=21.6%` of wall time (IQR 19.625-26.65%); per-worker medians are
21.1/21.85/21.6/22.65%. The post-restart mechanism read is the same pooled
statistic after warmup; games/hour remains the outcome metric and SF wait is a
large confound, so do not infer a 21.6% throughput gain directly.

**Stockfish pool head-of-line scheduling experiment -- UNREAD (2026-07-12).**
The pool binds each request round-robin to an engine before submitting it to one
shared thread executor. With mixed-duration move and label searches, an executor
thread can therefore block on a busy engine's UCI lock while another engine is
idle, delaying unrelated work; live steady-state telemetry attributes roughly
2300% cumulative thread time to `sf_block_starved`. Hypothesis: one single-thread
executor per engine preserves round-robin engine ownership while queueing each
request behind only that engine, eliminating cross-engine head-of-line blocking.
ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_stockfish_pool_scheduling.py --workers 8 --requests 256 --rounds
9`. SUCCESS: candidate/reference median completion time <=0.80, exact result
checksum, deterministic concurrency coverage proves an available engine can run
while a different engine has queued work, focused Stockfish/selfplay tests and
lint pass. KILL: ratio >0.95 or any ordering/lifecycle regression; otherwise
MIXED and do not ship. Offline scheduler mechanism only; no live/config/data or
training-target change, and activation waits for a natural restart.
**MECHANISM VERDICT: WORKED.** Nine alternating rounds measured 0.011361731s
shared-executor median latency versus 0.000702322s with per-engine queues, a
**0.061815 ratio (16.18x faster)** for an available engine's probe under one
unrelated busy-engine queue, with identical checksum 2,429. The deterministic
pool regression test pins that engine 1 completes its request while engine 0's
second request remains queued and also verifies orderly shutdown. This removes
cross-engine head-of-line blocking; the post-natural-restart read must use
`sf_block_starved` and games/hour because the synthetic mechanism ratio is not a
whole-selfplay throughput claim.
**PRE-ACTIVATION LIVE BASELINE:** the latest 20 steady phase windows from each
of the four active workers (80 total, 2026-07-13 00:44) have pooled median
`sf_block_starved=1853.1%` cumulative thread time. Repeat the same active-trial
four-worker/latest-20 calculation after restart; games/hour is the outcome.

**Label-blocked finalization overlap experiment -- UNREAD (2026-07-13).** Live
steady-state workers report roughly 300-1000% cumulative thread time in
`finalize` while 32 selfplay threads share four Stockfish processes. A completed
selfplay game currently calls the blocking label flush immediately, stalling its
whole 24-slot state even when other slots are runnable. Hypothesis: defer only
done games that still own incomplete async label futures, keep scheduling the
other slots, and wait on the first label future with the existing 50ms control
deadline only when no runnable work remains. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_finalize_label_overlap.py
--label-delay-ms 20 --work-items 20 --work-ms 1 --rounds 9`. SUCCESS: candidate
median elapsed time <=0.65x blocking reference, identical completed-work
checksum, deterministic tests prove runnable work proceeds before the label and
idle states block rather than spin, focused selfplay/finalization tests and lint
pass. KILL: ratio >0.90, any dropped/unlabeled game, changed sample contents, or
pause/stop responsiveness beyond the existing 50ms bound; otherwise MIXED and
do not ship. Scheduling only; no target/config/data semantic change or salvage,
and activation waits for a natural restart.
**MECHANISM VERDICT: WORKED.** Nine alternating rounds measured 0.045351887s
for blocking label-then-work scheduling versus 0.023162679s when useful work
overlapped the same label, a **0.510732 ratio (1.96x mechanism speedup)** with
identical checksum 227. Continuous sessions now skip only a done game whose own
records still have pending label entries; a later poll finalizes and recycles it.
When nothing can advance, the scheduler waits for first SF completion with the
existing 50ms bound instead of spinning. Finite batches deliberately retain
blocking finalization so idle waits cannot consume their safety-step budget. A
true continuous-session stop force-drains already-done games before exit, so
deferral cannot turn completed games into teardown abandonment. A
clean GCC 15 native+LTO build, all 53 focused label/escalation/continuous-
selfplay tests, the 18-test CPU selfplay-to-replay-to-training end-to-end smoke,
and ruff/basedpyright/vulture pass. Keep the overlap; the next natural restart
must judge whole throughput from finalize share and games/hour.
**PRE-ACTIVATION LIVE BASELINE:** the same 80 active-worker windows have pooled
median `finalize=484.0%` cumulative thread time. Repeat the same windowed
calculation after restart; games/hour remains the outcome metric.

**Stockfish pool node-weighted queue-balance experiment -- UNREAD
(2026-07-13).** Per-engine FIFO queues remove cross-engine blocking, but request
ownership remains round-robin while production fast-ply searches use roughly
one quarter of the full-label/current-opponent node budget. With four engines,
clustering full searches on one queue extends the whole completion wave.
Hypothesis: assign each request to the engine with the least outstanding node
budget, using rotating ties, and decrement the estimate on future completion.
ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_stockfish_pool_balance.py --workers 4 --requests 96 --sequences
10000 --seed 20260713`. SUCCESS: weighted/round-robin aggregate makespan <=0.95,
no sequence regresses by >5%, exact request/result checksum, concurrent-submit
tests preserve accounting, focused Stockfish/selfplay tests and lint pass.
KILL: ratio >0.99, submit overhead >25us/request, or any leaked/negative pending
budget; otherwise MIXED and retain only for a net-simple implementation. Offline
scheduler only; engine count, node budgets, results, data, and live config stay
unchanged, with activation at a natural restart.
**VERDICT: MIXED and reverted.** Across 10,000 deterministic 96-request
production-ratio sequences, weighted assignment reduced aggregate modeled
makespan from 486,843 to 434,429 (**0.892339 ratio, 10.8% better**) and cost only
2.028us/request versus 0.269us round-robin. However, the worst individual
sequence regressed **1.060606x**, exceeding the pre-committed 1.05 tail guard.
The exact cost checksum was 1,683,141. Pending-node locks and future callbacks
are not a net simplification, so the MIXED retain clause does not apply: keep
the simple per-engine FIFO round-robin scheduler from the preceding experiment.

**Singleton coalescer zero-copy experiment -- UNREAD (2026-07-11).** Compiled
single-game UCI uses `BatchCoalescingDispatcher` to keep torch.compile/CUDA
graph work on one submitter thread, but each drain unconditionally executes
`np.concatenate` for encoded inputs and compact legal arrays even when the
batch contains one request. That copies roughly 22 MB for a 512x175 FP32 leaf
batch (11 MB for native BF16) while providing no coalescing. Hypothesis: aliasing
the original arrays for singleton drains, while retaining concatenation for
two or more requests, materially removes CPU packing cost with identical
inputs/results. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_coalescer_singleton_pack.py --batch 512 --planes 175 --legal-per
32 --iterations 500 --rounds 7`. The harness alternates old copy packing and
singleton alias packing on fixed production-shaped BF16/legal arrays. Pre-
committed SUCCESS: optimized/reference median packing throughput >=5.0x, exact
array contents match, the optimized objects share memory with the inputs, and
coalescer/dispatcher/Gumbel/UCI focused tests plus repo lint pass. Otherwise
FAILED and retain unconditional concatenation. No training/data change and no
salvage snapshot.
**VERDICT: WORKED.** The exact seven-round alternating yardstick measured 819
singleton packs/s for unconditional concatenation and 9,252,576 packs/s for
aliasing, a **11,295.8x packing-mechanism ratio**. Each production-shaped BF16
compact-policy request avoids copying 11,536,384 bytes; FP32 inputs avoid about
twice that. Every packed array matched exactly and all optimized outputs shared
memory with their source. This is intentionally a mechanism result, not a claim
that whole-engine throughput rises by the same ratio: GPU inference and C tree
work remain. Keep the simple singleton fast path; multi-request drains retain
the original concatenation/coalescing behavior.

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

**Python-to-native candidate profile -- UNREAD (2026-07-10).** Hypothesis:
despite the existing CBoard/MCTSTree fast paths, at least one production-shaped
selfplay or replay operation still spends enough wall time in Python-owned
work to justify a focused C-extension conversion. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/profile_python_native_candidates.py
--boards 384 --simulations 256 --replay-rows 5000 --batch-size 512 --repeats 7`.
The benchmark uses deterministic production-width v2 inputs, a zero-cost
evaluator to expose the CPU/search upper bound, production-sized Gumbel batches,
and a chunked sparse-policy replay buffer; it reports medians and cProfile
attribution. Pre-committed SUCCESS for a conversion candidate: a Python-owned
phase (excluding model inference, Stockfish, filesystem I/O, and NumPy/C calls)
accounts for at least 10% of its measured workload and at least 10 ms per
production-shaped call. Otherwise the profile is FAILED and no Python-to-C
conversion is recommended from these paths. Correctness hashes must be stable
across repeats. No live/training change or salvage snapshot.
**PROVENANCE CORRECTION BEFORE VERDICT:** the first invocation's fake evaluator
did not advertise the production legal-BF16 interface, so leaf logits took the
dense compact-to-full fallback and falsely attributed 820 ms to policy widening;
the replay hasher also omitted dict contents. That run is invalid, not a verdict.
The harness now implements `evaluate_legal_bf16` and hashes sorted dict fields;
thresholds and command are unchanged for the clean rerun.
**VERDICT: FAILED.** With the legal-BF16 production path active, Gumbel search
spent 2.7% project-Python self time (32.7 ms upper bound in a 1.211 s CPU-only
384-board search); its measured final-policy phase was 35 ms, while compact
policy widening fell from the invalid run's 820 ms to 0.9 ms. Replay sampling
took 4.61 ms total per 512-row batch (3.47 ms project-Python upper bound).
Both exact hashes were stable. Neither path clears the 10% + 10 ms gate, so do
not convert the remaining Gumbel glue or replay sampler to C.

**Per-game replay-finalization native profile -- UNREAD (2026-07-10).** The
search/replay screen above does not cover the worker's Python loop that turns a
completed game's `_NetRecord` rows into replay samples. Hypothesis: production-
shaped finalization of a 64-record selfplay game is a material remaining Python
cost. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
64 --repeats 7`. Pre-committed SUCCESS: median finalization is at least 10 ms,
project-Python self time is at least 50%, and its wall time is at least 3% of
the prior screen's CPU-only per-game search proxy (`64 * 1.210975s / 384 =
201.8 ms`). Otherwise FAILED. Stable hashes required; no live/training change
or salvage snapshot.
**VERDICT: WORKED.** Median finalization was 13.326 ms for 64 records, 76.6%
project-Python self time, and 6.60% of the CPU-only per-game search proxy, with
a stable exact hash. The largest single block was `hlgauss_target` (4.54 ms),
followed by SF-P0 regret construction (3.10 ms). This clears the gate, but the
HL-Gauss cost is repeated construction of only three outcome kernels under the
production default blend, so test a semantic-preserving cache before adding C.

**Ternary HL-Gauss cache experiment -- UNREAD (2026-07-10).** Hypothesis:
caching the immutable `-1/0/+1` HL-Gauss kernels and returning independent
copies removes most of the largest finalization hotspot without a new native
ABI. ONE deciding yardstick: the same pinned finalization command above,
`PYTHONPATH=. taskset -c 15 python3 scripts/profile_python_native_candidates.py
--only-finalize --finalize-records 64 --repeats 7`. Baseline is 13.326 ms and
hash `6137f66d62c8e111`. Pre-committed SUCCESS: median <=10.661 ms (>=20%
faster), exact hash unchanged, a regression test proves returned arrays do not
alias, and the target/finalization test suites pass. Otherwise FAILED and
revert the cache. No live/training change or salvage snapshot.
**VERDICT: FAILED and reverted.** Exact hash stayed `6137f66d62c8e111` and 34
focused tests passed, but median finalization improved only 13.326 -> 11.852 ms
(11.1%), short of the pre-committed 20% gate. The cache and its test were
removed; this result does not justify retaining stateful machinery.

**Ternary HL-Gauss cache modest-gain acceptance rerun -- UNREAD
(2026-07-10).** User decision after the first gate: small, low-complexity real
speedups are worth retaining rather than treating 20% as a universal floor.
Hypothesis: a bounded cache of the immutable `-1/0/+1` outcome kernels, with an
independent writable copy returned to every replay sample, reproduces the prior
11.1% finalization gain without aliasing or semantic change. ONE deciding
yardstick remains `PYTHONPATH=. taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
64 --repeats 7`. Baseline remains 13.326 ms and hash `6137f66d62c8e111`.
Pre-committed SUCCESS: median <=12.660 ms (>=5% faster), exact hash unchanged,
an aliasing regression test passes, and the focused finalization/categorical
tests pass. Otherwise FAILED and revert again. This changes CPU construction
cost only, not targets or training data meaning; no salvage snapshot.
**VERDICT: FAILED by the absolute-time rule.** The exact run preserved the
hash and passed correctness tests but measured 38.978 ms, not <=12.660 ms.
Every unrelated Python block was simultaneously ~3x slower (for example
SF-P0 regret 3.10 -> 8.57 ms), showing that core 15 was contended by the live
workload and the old absolute baseline was not comparable. This is still a
failed yardstick, not a cache win. The candidate is retained only through the
immediately following pre-registered paired decision and will be reverted if
that paired gate fails.

**Ternary HL-Gauss cache paired acceptance -- UNREAD (2026-07-10).**
Hypothesis: alternating cache-off/cache-on measurements on the same pinned core
will remove the live-load/frequency confound and reproduce a useful relative
gain. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
64 --repeats 3 --compare-finalize-cache --paired-rounds 5`. Each paired round
alternates order and compares identical hashes. Pre-committed SUCCESS: median
cache-on throughput is >=1.05x cache-off, all hashes are identical/stable, the
independent-copy regression passes, and focused finalization/categorical tests
pass. Otherwise FAILED and revert the cache. No training/data semantic change
or salvage snapshot.
**VERDICT: SUCCESS.** The exact alternating paired command measured uncached
33.654 ms vs cached 29.487 ms, a **1.141308x (14.1%) finalization speedup**,
with every run producing the identical stable hash `6137f66d62c8e111`.
The independent-writable-copy regression and 34 focused finalization /
categorical tests pass. Keep the bounded 24-entry ternary-kernel cache. Based
on the clean profile's 6.6% CPU-only finalization share, this is roughly a 0.9%
end-to-end CPU-only selfplay gain by itself; live phase telemetry now decides
whether a fused SF-regret/remap helper is worth pursuing for the larger next
step.

**Fused native SF-finalization experiment -- UNREAD (2026-07-10).** Live
post-restart phase telemetry shows replay finalization accumulating roughly
10-50 thread-seconds per wall-second during completed-game waves; the GIL makes
small per-game Python loops amplify across 32 selfplay threads. Hypothesis: one
native helper that validates each raw MultiPV matrix once, pads/remaps its move
column, and constructs the dense SF-P0 regret vector while releasing the GIL
removes the remaining dominant Python finalization blocks without changing
targets. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
64 --repeats 3 --compare-native-finalize --paired-rounds 5`. Each round
alternates the Python reference and native path on identical data.
Pre-committed SUCCESS: native median throughput >=1.15x Python, every exact
finalization hash matches and is stable, randomized parity covers full/compact
policy encodings plus cp/mate/sentinel/padded rows, and focused native /
finalization / sparse-label tests pass. Otherwise FAILED and revert the native
helper. This is a CPU implementation change only; no training-data meaning or
live config changes, so no salvage snapshot.
**VERDICT: SUCCESS.** The exact alternating paired yardstick measured Python
23.761 ms vs native 14.066 ms per 64-record game, a **1.689246x (68.9%)
finalization speedup**, with the identical stable hash `6137f66d62c8e111` in
every run. Randomized full/compact parity, mate/cp/sentinel/padding cases, an
8-thread native stress, and 59 focused tests pass; the strict native+LTO build
also passes `-Werror`. Keep the fused helper. Because its compute loops release
the GIL, live completed-game waves should gain more than the scalar timing alone
suggests; measure that separately rather than inferring a thread-scaling number.
Two exact-command replications after input-hardening/refactoring measured
**1.878196x** and **2.436058x**, both with the same exact hash. External load
moves the absolute times, but all three alternating reads clear the 1.15x gate;
use the conservative 1.689x first read as the headline until live telemetry.

**Curriculum regret-suffix finalization experiment -- UNREAD (2026-07-10).**
The selfplay-shaped finalization benchmark returns early from future opponent-
regret construction and therefore did not measure curriculum games. Hypothesis:
precomputing integer ply indices and replacing five repeated Python horizon
walks per record with C-level `bisect` plus `sum` materially reduces curriculum
finalization while preserving the original ascending floating-point addition
order. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
64 --repeats 3 --compare-curriculum-finalize --paired-rounds 5`. Each round
alternates the original reference and optimized path on identical curriculum-
shaped records. Pre-committed SUCCESS: optimized median throughput is >=1.05x
reference, every exact finalization hash is identical and stable, randomized
parity covers missing/non-finite/negative regrets and irregular increasing ply
indices, and focused finalization/adjusted-WDL tests pass. Otherwise FAILED and
revert the optimization. This changes CPU construction only; no target or live
config semantics and no salvage snapshot.
**VERDICT: FAILED and reverted.** The exact alternating yardstick measured the
reference at 13.319 ms and the `bisect`/`sum` version at 17.635 ms, only
**0.755268x** reference throughput (24.5% slower), while every run retained the
identical stable hash `cf16b2dbed33e358`. The C-level search did not offset
building five list slices and invoking five sums per record. Focused tests and
lint passed before the timing, but the performance gate failed, so none of the
runtime, benchmark, or test changes were retained.

**Stockfish host-native PGO build experiment -- UNREAD (2026-07-10).** The
production engine is the official `Stockfish dev-20260420-ed651aab` generic
`x86-64-bmi2` binary, while the host is an AMD Ryzen 9 5950X with AVX2/BMI2.
Hypothesis: rebuilding the identical source release with Stockfish's supported
`make -j profile-build ARCH=native` path produces a useful node-throughput gain
without changing deterministic search results. Build in an isolated `/tmp`
copy; do not replace the live binary during this experiment. ONE deciding
yardstick: `PYTHONPATH=. python3 scripts/bench_stockfish_build.py --baseline
/home/josh/local_stockfish/extract/usr/games/stockfish --candidate
/tmp/stockfish-native/src/stockfish --rounds 5 --cpu 15 --hash-mb 16 --threads
1 --depth 13`. The harness warms both binaries, alternates order, and uses
Stockfish's 51-position bench. Pre-committed SUCCESS: candidate median nodes/s
is >=1.03x baseline, every best-move/node-count semantic hash is identical, and
the candidate reports the same Stockfish source identity. Otherwise FAILED and
keep the official BMI2 binary. A successful result still requires a production-
shaped fixed-node MultiPV confirmation before any live path change. No training
or data change and no salvage snapshot.
**VERDICT: FAILED; keep the official BMI2 binary.** The exact alternating
yardstick measured the official baseline at 1,312,114 nodes/s and the local
host-native PGO+LTO candidate at 1,167,073 nodes/s, only **0.889460x** baseline
throughput (11.1% slower). Both binaries identified as
`Stockfish dev-20260420-ed651aab`; all five rounds produced the identical
best-move/node semantic hash `433cd809ad16ef93` and node count 2,723,949. The
source/architecture premise was valid but local GCC 11 code generation lost to
the official release build, so do not replace the engine and do not run the
production MultiPV confirmation for this failed candidate.

**Stockfish Clang PGO build experiment -- UNREAD (2026-07-10).** GCC 11
host-native PGO was 11.1% slower than the official binary, but Clang 14 is now
installed and Stockfish provides a distinct supported `COMP=clang` PGO+LTO
recipe. Hypothesis: Clang's code generation closes the release-binary gap and
produces a useful same-source speedup. Rebuild the isolated source with
`make -j profile-build ARCH=native COMP=clang GIT_SHA=ed651aab
GIT_DATE=20260420`; do not touch the live binary. ONE deciding yardstick is the
same alternating `scripts/bench_stockfish_build.py` command and settings as the
preceding experiment. Pre-committed SUCCESS: candidate median nodes/s >=1.03x
the official baseline, exact best-move/node-count hashes match across all
rounds, and source identity matches. Otherwise FAILED and keep the official
binary. A win still needs production MultiPV confirmation before deployment.
No training/data change and no salvage snapshot.
**VERDICT: FAILED at the toolchain gate; no candidate was deployed or timed.**
Clang 14 compiled and linked the instrumented binary after redirecting the
Windows `TMP`/`TEMP` variables to writable `/tmp`, and its PGO training bench
completed with the exact 2,723,949-node workload. The required profile-use
stage then failed because matching `llvm-profdata` is not installed (the
user's package installation could not fetch it). A non-PGO Clang binary would
not test the registered hypothesis or compare fairly with the release build,
so this arm ends here and the official BMI2 binary remains production.

**GCC 15.3 Stockfish PGO experiment -- UNREAD (2026-07-10).** GCC 11's
same-source PGO build was 11.1% slower than the official engine, so the open
question is compiler generation rather than architecture flags. Hypothesis:
an isolated, upstream-source GCC 15.3 toolchain closes that gap and may beat
the official `dev-20260420-ed651aab` BMI2 binary on this Zen 3 host. Build only
C/C++ under `/tmp/gcc-15.3-install` (no system compiler replacement), then
rebuild the identical Stockfish source with `make -j profile-build ARCH=native
COMP=gcc COMPCXX=/tmp/gcc-15.3-install/bin/g++ GIT_SHA=ed651aab
GIT_DATE=20260420`. ONE deciding yardstick: the same five-round alternating
`scripts/bench_stockfish_build.py` command/settings as the prior compiler
experiments. Pre-committed SUCCESS: candidate median nodes/s >=1.03x official,
all best-move/node hashes are identical, and source identity matches. Otherwise
FAILED and keep the official binary. A winner still needs a fixed-node MultiPV
confirmation before deployment. No training/data change or salvage snapshot.
**VERDICT: FAILED; keep the official Stockfish binary.** The exact five-round
alternating yardstick measured 541,110 nodes/s for the official binary and
539,502 nodes/s for the GCC 15.3 host-native PGO+LTO candidate, a **0.997028x**
ratio versus the required 1.03x. Every run retained node count 2,723,949 and
semantic hash `433cd809ad16ef93`; both binaries report
`dev-20260420-ed651aab`. GCC 15 closed GCC 11's 11.1% deficit but did not beat
the release toolchain, so no MultiPV confirmation or deployment is warranted.

**GCC 15.3 project-extension experiment -- UNREAD (2026-07-10).** Hypothesis:
GCC 15.3 improves the production BF16 LC0 encoder beyond the current GCC 11
`-march=native -flto` build without changing any encoded output. ONE deciding
yardstick: extend the existing controlled compiler harness to alternate
`gcc11-native-lto` and `gcc15-native-lto`, then run `PYTHONPATH=. python3
scripts/bench_native_build_flags.py --modes gcc11-native-lto
gcc15-native-lto --rounds 5 --samples 7 --iterations 200 --cpu 15`. Pre-
committed SUCCESS: GCC15/GCC11 median BF16 throughput >=1.01x, exact hashes
match for every extension, the full focused native test set passes under the
GCC15 build, and no measured component is below 0.98x. Otherwise FAILED and
rebuild production with GCC 11 native+LTO. This is a compiler-only change with
no data semantics or salvage snapshot.
**VERDICT: SUCCESS.** The exact five-round alternating yardstick produced
GCC15/GCC11 ratios of **1.0272 BF16**, 1.0277 F32, 1.0640 move generation,
and 1.0761 WDL conversion; production geometric mean **1.0556x**. Every round
matched the exact F32/BF16 hashes, no component fell below 0.98x, and 138
focused native tests passed (1 skipped). Keep GCC 15.3 native+LTO for local
production extension builds. Median forced-build time increased 24.9s ->
40.4s, an acceptable ~15.6s edit/rebuild cost for the measured runtime gain.
**DEPLOYMENT HARDENING (2026-07-12):** `scripts/build_production_extensions.py`
now makes that winning recipe the explicit local-production rebuild path:
validated `~/.local/gcc-15.3/bin/gcc` (or `CAE_GCC15_CC`) + native + LTO.
The stale-extension startup check points to this wrapper, preventing a routine
post-edit rebuild from silently replacing the winning binaries with the
portable GCC11 recipe. Portable package/wheel builds remain unchanged.

**Native BF16 fallback/match-input experiment -- UNREAD (2026-07-11).** The
live distributed RL path was re-verified after this entry was drafted: it uses
`SlotInferenceClient` and `SlotBroker`, which already transport native BF16
input bits. The remaining widening is limited to the worker-local threaded
fallback and match-style direct evaluator paths, where `DirectGPUEvaluator` is
constructed with `input_bf16=False`. Hypothesis: wiring its existing pinned
BF16 input path removes widening and halves H2D traffic in those paths without
changing model outputs, because inference autocast already executes the first
projection in BF16. ONE deciding yardstick: run
`scripts/bench_worker_legal_policy.py` on the same current checkpoint with
`--policy-path dense --input-path widened` versus `--policy-path dense
--input-path native-bf16`, batch 680, max-autotune, three warmups and 20 timed
iterations, alternating order for three rounds. Pre-committed SUCCESS:
native/widened median positions/s >=1.03x, exact output hashes match every
round, and focused dispatcher/broker/MCTS tests pass. Otherwise FAILED and
leave fallback/match input widening unchanged. This candidate does not claim an
RL broker gain. Offline inference-only; no data change or salvage snapshot. Run
this before the legal-policy experiment so each A/B changes only one path.

**Compiled legal-policy inference experiment -- UNREAD (2026-07-11).**
Production workers already transport only legal policy indices and BF16 input,
but a compiled model defaults `CAE_COMPILED_LEGAL_POLICY` off and therefore
executes the dense 1,858-logit policy head before gathering legal logits.
Hypothesis: compiling the checkpoint's `forward_legal_policy` graph removes
enough policy-head work to improve worker inference throughput without changing
BF16 outputs. ONE deciding yardstick: run `scripts/bench_worker_legal_policy.py`
on the same current checkpoint with `--path dense` and `--path legal`, batch 680,
32 legal moves/position, `--input-path widened`, max-autotune, three warmups and 20 timed iterations,
alternating path order for three rounds after the duplicate live Ray trees are
stopped. Pre-committed SUCCESS: legal/dense median positions/s >=1.03x, every
round's exact output hash matches between paths, and the focused transformer,
dispatcher, broker, and native MCTS tests pass. Otherwise FAILED and leave the
compiled legal gate off. This is an offline inference-only experiment; no data
or training change and no salvage snapshot.

**Stockfish MultiPV final-line parsing experiment -- UNREAD (2026-07-12).**
Production uses MultiPV 40. `StockfishUCI.search` currently scans every token
list several times, allocates a normalized NumPy WDL vector, and constructs a
`StockfishPV` for every intermediate-depth info line, even though the next
depth overwrites that PV entry. Hypothesis: one token scan per line plus raw
latest-PV retention, normalizing/materializing only the final line for each PV,
reduces worker GIL/CPU cost without changing UCI results. ONE deciding
yardstick: `PYTHONPATH=. python3 scripts/bench_stockfish_info_parser.py
--multipv 40 --depths 30 --rounds 9 --iterations 250`. The benchmark
alternates the exact old parse/update loop with the candidate on identical
production-shaped streams. Pre-committed SUCCESS: candidate/reference median
streams/s >=1.50x, exact result hashes match in every round, and focused UCI,
Stockfish-label, sparse-MultiPV, and end-to-end smoke tests plus repo lint pass.
Otherwise revert. This changes parsing/materialization only: engine commands,
node budgets, final PVs, WDL/CP/mate values, search decisions, and targets stay
identical. No live/training change and no salvage snapshot.
**VERDICT: WORKED.** The exact nine-round alternating yardstick measured
56.893 reference streams/s and 104.282 candidate streams/s, a **1.83295x**
parser speedup. All rounds retained exact result hash
`f457ed0c5c4d1c71dc368830c9899e68ec65ad649dfd2d5eb5a23ff31f2485e1`.
On the representative 1,200-line stream this removes about 7.99 ms of Python
parsing/materialization per query. This is deliberately a parser/GIL result,
not a whole-game throughput claim; Stockfish node search remains dominant.
The non-primary MultiPV-1 guard also improved 1.58656x (2,188.56 -> 3,472.28
30-line streams/s, exact hash), so the accumulator does not trade away the
single-PV path used by smokes and some analysis tools.

**UCI compact-BF16 transport experiment -- UNREAD (2026-07-11).** Match play's
compiled evaluator is wrapped by `ThreadSafeGPUDispatcher` and
`BatchCoalescingDispatcher`; neither forwards the legal-policy BF16 API or the
native-BF16 capability, so single-game Gumbel computes/transports dense FP32
policy outputs and FP32 inputs even though the underlying evaluator and C tree
support compact legal BF16. Hypothesis: forwarding and coalescing that API,
with native BF16 pinned inputs enabled under `CAE_UCI_COMPACT_BF16=1`, improves
match-search throughput without changing selected moves. ONE deciding
yardstick: alternate three rounds of `CAE_UCI_COMPACT_BF16=0` and `=1` running
`PYTHONPATH=. python3 scripts/bench_uci_engine.py --checkpoint <current-copy>
--device cuda --nodes 8192 --repeats 3 --chunk-sims 512 --topk 32 --max-batch
1024 --compile-mode max-autotune`. Pre-committed SUCCESS: candidate/control
median aggregate sims/s >=1.05x, every per-position bestmove agrees, no search
times out, compile counters show no new graph breaks/recompiles, and focused
dispatcher/Gumbel/UCI tests pass. Otherwise FAILED and keep the env default
off. Match-inference implementation only; no training/data change or salvage
snapshot.

**Broker recycled-response coalescing experiment -- UNREAD (2026-07-11).**
Live post-recovery telemetry shows only ~34-55 positions and 1.3-2.1 of 16
slots per batch, typically 1.5-3.2k pos/s, despite a 20 ms hard cap and 2 ms
adaptive-idle window. `_gather_more_within_window` returns immediately when
all non-ready slots are `RESPONSE`, incorrectly assuming they cannot be
consumed and become a new `REQUEST` during the window. Hypothesis: allowing
response slots to recycle until the adaptive idle deadline restores intended
coalescing and materially raises broker throughput. ONE deciding yardstick at
the next natural restart: compare ten consecutive steady-state `[broker]`
windows against the current-session baseline using the same production config;
pre-committed SUCCESS is median positions/batch >=1.20x and median pos/s >=1.15x,
with client roundtrip p95 not worse by >10%, exact inference parity tests, and a
deterministic response-to-request transition test proving recycled slots join
the pending batch. Otherwise revert the gather condition. No model/data
semantic change and no salvage snapshot; restart-gated broker implementation.

**Vectorized finalization-target experiment -- UNREAD (2026-07-10).**
Hypothesis: vectorizing the fixed-width MultiPV move remap and SF-P0 regret
construction removes the repeated per-row policy-encoding/string work that is
now the largest avoidable finalization cost, without a C ABI. ONE deciding
yardstick: the same pinned finalization command above. Baseline remains 13.326
ms and hash `6137f66d62c8e111`. Pre-committed SUCCESS: median <=10.661 ms
(>=20% faster), exact hash unchanged, randomized parity tests cover compact and
full policy encodings plus cp/mate/sentinel rows, and focused finalization /
sparse-label tests pass. Otherwise FAILED and revert. No live/training change
or salvage snapshot.
**VERDICT: FAILED and reverted.** The exact hash stayed unchanged and 46
focused tests passed, but median finalization regressed 13.326 -> 14.049 ms
(5.4% slower). At only 40 MultiPV rows, temporary NumPy arrays cost more than
the original scalar loops. The remaining credible conversion is a *fused* C
finalization helper (HL-Gauss + SF-P0 regret + compact-policy remaps), not a
piecemeal NumPy rewrite. Based on the profile, eliminating 80-90% of those
blocks would roughly halve 13.3 ms finalization and save at most ~3-4% of the
CPU-only selfplay proxy before GPU/Stockfish time; validate against workers'
existing `selfplay phase stats: ... finalize=` telemetry before implementing a
new ABI.

**GIL-delay telemetry overhead experiment -- SUCCESS (2026-07-10).**
Hypothesis: one daemon probe sleeping outside the GIL at 10 ms cadence can
measure delayed interpreter reacquisition without materially slowing worker or
match-play Python. The delay is explicitly an upper bound (GIL wait + OS wake
latency), not a claimed exact GIL percentage. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_gil_probe.py --seconds 5
--rounds 5 --interval-ms 10`. The script alternates identical deterministic
CPU-bound rounds with the probe off/on. Pre-committed SUCCESS: probed median
throughput is >=0.995x control, workload hashes match, and the probe records at
least 50 samples/s. Otherwise FAILED and do not enable it by default in
workers; UCI opt-in profiling may still retain it. No training/data change or
salvage snapshot.
**VERDICT: SUCCESS.** On the exact pre-registered command, probed throughput
was 1.004184x control (2,841,983 vs 2,830,140 iterations/s), the deterministic
hash was stable, and the probe collected 64.3 samples/s. This clears all three
gates. The 10 ms probe may therefore ship enabled in distributed workers and
opt-in for UCI match/search profiling. Interpret its delay distribution only
alongside the existing phase/thread wall-time metrics: it is an upper bound on
GIL contention plus scheduler/timer latency, not a GIL-held percentage.
CPU-only end-to-end UCI smoke (`checkpoint_000722`, five positions, 16 nodes,
`--no-compile --gil-profile`) also passed and aggregated 802 samples: mean
0.265 ms, worst-search p95 <=2 ms, p99 <=5 ms, max 4.51 ms. This validates the
UCI plumbing only; it is not a production throughput or walker-count verdict.

**CPython 3.14t worker/match compatibility experiment -- FAILED
(2026-07-10).** Hypothesis: the lite worker/UCI dependency graph can run under
the latest CPython 3.14 free-threaded build without any imported native module
silently re-enabling the GIL. This is a side-by-side `/tmp` environment only;
it does not replace the live trainer's Python. ONE deciding yardstick after
building the project extensions in that environment:
`PYTHONPATH=. /tmp/cae-py314t/bin/python
scripts/check_free_threading_compat.py`. Pre-committed SUCCESS: NumPy, PyTorch,
python-chess, PyYAML, requests, zarr, and all three project C extensions import;
`sys._is_gil_enabled()` remains false after every import; and the focused C
extension tests pass. Otherwise FAILED: do not upgrade production Python, name
each blocker, and treat porting it as a prerequisite to any no-GIL throughput
benchmark. Training/Ray compatibility is a separate diagnostic via
`--include-training`, not part of this worker/match gate. No training/data
change or salvage snapshot.
**VERDICT: FAILED; do not upgrade production Python.** The exact yardstick ran
under uv-managed CPython 3.14.2t. NumPy 2.5.1, PyTorch 2.12.1+cu130,
python-chess, PyYAML, and requests imported with the GIL still disabled. zarr
2.18.7 then imported `numcodecs.blosc`, which emitted CPython's unsupported
extension warning and globally enabled the GIL. The project constraint
`zarr<3` resolved numcodecs 0.15.1; no 3.14t wheel exists, and its source build
took 5m16s but still did not declare free-thread safety. Our three single-phase
C extensions likewise have neither a `Py_mod_gil` slot nor
`PyUnstable_Module_SetGIL(..., Py_MOD_GIL_NOT_USED)`, so they are additional
blockers pending a real shared-state/refcount audit (merely adding the marker
would be unsafe). The training path is blocked earlier still: Ray 2.53.0 has no
CPython 3.14 wheel, and current Ray 2.56.0 publishes regular `cp314` Linux
wheels but no `cp314t` wheel. PyTorch is not the blocker. Reconsider as a
side-by-side worker/UCI experiment only after (1) porting the project C
extensions, (2) replacing/upgrading the zarr-2/numcodecs path with a verified
free-threaded build, and (3) keeping Ray training on its supported interpreter
until a `cp314t` release exists.

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

- **SF-label escalation on net-vs-label disagreement — BUILT (PR #135), flags
  default-off, NOT LAUNCHED** (2026-07-09). HYPOTHESIS: the ~700k-node sf_wdl
  label is wrong in 70-81% of high net-vs-label-gap cases (deep-SF audits of
  the harvest band side with the NET), and those wrong labels feed the
  load-bearing WDL blend daily; re-querying exactly those plies at 3M nodes
  (cold TT) fixes label noise at the source. Mechanism: `stockfish.
  sf_label_escalate_q_gap` (0=off) / `_nodes` / `_max_per_game`; original
  label preserved as `sf_wdl_original` so the harvester still sees the
  original gap; telemetry `sf_label_escalated{,_moved}`. ACTIVATION (a later,
  ledgered decision — NOT in the dole readout window): rung 1 = q_gap 0.8 /
  3M / max 2 via `configs/exp_label_escalation.yaml`. DECIDING YARDSTICK:
  value_regret paired CI vs the pre-activation banked dump (2000 pos).
  PRE-COMMITTED KILL: selfplay positions/s drops >10% vs pre-change OR value
  paired CI worse (excludes 0). SUCCESS: escalated-moved rate >30% with the
  value guardrail clean → consider rung 2 (q_gap 0.6). Confound guard: do
  not activate in the same window as any other data-affecting change.
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
   **SAMPLER INCIDENT (07-09 13:49): the mixed-batch run (cont_0709_1100,
   mixed sampler + LR 1e-4) died at step ~4950 — "failed to sample a live
   replay batch after repeated shard reloads". Root cause: pool shards have
   heterogeneous key sets (29 pre-sf_played-era shards lack 6 sf_played_*
   fields); once an old-schema shard entered the 16-entry LRU next to new
   ones, the cross-shard merge KeyError'd on EVERY draw, and the retry
   handler evicted the innocent DRAWN shard while the mismatched entry
   stayed cached — 8 strikes, dead. FIX (PR #136, merged): mix only
   schema-identical cache entries (no silent field drops; lone-schema draw
   degrades to a single-shard batch), verified against the real pool.
   RELAUNCHED 14:29 as cont_0709_1429_lr0.0001 from trainer_best_eval.pt
   (step 4500, eval 4.726 — best of the crashed run; ~450 steps lost).
   Probe note: the 12:11 probe read 87.9cp overall (down from the unmixed
   run's 101.4) at only step ~400 of the mixed run — early but directionally
   the single-shard-correlation diagnosis holding.**
   **STALE-BAR PLATEAU STOP (07-09 ~15:13): cont_1429 self-stopped at step
   1500, reason eval_plateau — but the inherited bar (4.726 @4500 via
   trainer_best_eval.json) is a lucky-outlier eval: the SAME weights
   re-evaluated at 4.786, neighbors 4.888-4.900, and the pool composition
   shifts under the feeder, so no new state could beat the bar, get probed,
   or survive the best-restore. 15:12 probe: 87.8cp, +12.09 [+2.76,+21.55]
   vs live-710 — mixed sampler arrested the damage (peak 101.4) but the
   probe kept scoring the same frozen best state. FIX: bar json moved aside
   (trainer_best_eval.json.stale_bar_20260709, reversible) so the bar
   re-seeds from the new run's own first eval; relaunched with
   PLATEAU_EVALS=6, same LR 1e-4 + mixed sampler, init unchanged
   (trainer_best_eval.pt = the step-4500 weights). Next probe ~18:11 scores
   genuinely mixed-trained weights for the first time. Lesson: an on-pool
   eval bar carried across runs on a MOVING pool is not a valid plateau
   criterion — the frozen-ruler probe stays the only decision signal
   ([[offline-distillation-value-trap]] corollary).**
   **DESIGN AMENDMENT (07-09, supersedes the phased-LR plan): probe-driven +
   fed data.** (a) Found + fixed a second launch-design bug: live-follow with
   reuse=3 on a frozen pool spends ~3 epochs of credit then sleeps FOREVER
   (never reaches the 60k cap) — PR #133 (idle-exit, plateau stop,
   best-eval checkpoint; two Codex P1s fixed in `e24bc96`: LR-override
   captured warmup-floor rates ⇒ any --lr phase would run at ~1e-5, and the
   shard feeder could race the sampler's permanent position cache with
   partially-linked zarrs). (b) DATA: live shards are now continuously
   hardlink-fed (30-min loop, atomic-rename) into a DEDICATED pool
   `data/scaleup_pool_512x16` (seeded from the salvage window's 808 + 331
   shards fed on 07-08). CONFOUND (explicit): the fed pool retires the
   original frozen-iter-647-window condition — no longer a clean "same data
   as the 34.7M net" capacity test; the 512 trains on a superset including
   dole-seeded blind-spot games. Accepted deliberately: the swap gate is
   audit parity vs the CURRENT live net (fresher data serves it) and parity,
   not hypothesis purity, is the deliverable; frozen-window purity remains
   recoverable from the untouched salvage pool if the capacity question
   ever needs a clean re-test. The salvage pool is a frozen revert point
   and is NOT fed anymore; the 331 shards fed into it before this amendment are
   listed in `scratchpad/scaleup/fed_into_salvage_pool.txt` (mtime-dated
   07-08; originals 07-04) and get REMOVED when the current phase1 process
   exits (they're hardlink-preserved in the dedicated pool). (c) DECISION
   SIGNAL: parity probe every 3h (`scratchpad/scaleup/parity_probe.sh` →
   `parity_probe.log`): value_regret 2000 pos memcapped on the bootstrap's
   best ckpt, PAIRED vs the newest live monitor vdump. Levers chosen from
   the probe trajectory when a run stops (far+closing → `continue` = more
   data at same LR; close+flat → `continue 1e-4` then `3e-5`; plateaued far
   short → kill). NO auto-chained LR phases. The pre-committed SWAP GATE is
   unchanged (audit parity: value + audit_targets + panels, paired CIs,
   +2cp). **FIRST PROBE (07-09 00:04, step 7800, ~4h of training): boot
   83.6cp overall, paired vs live ckpt690 +12.30 [−1.09, +24.36] — within
   noise of parity on the VALUE yardstick after 4 hours.** (Policy/panel
   parity unmeasured so far — value alone does not clear the swap gate.)
   **PHASE1 ARC CLOSED (07-09 ~08:10, credit-exhausted at ~24.7k steps):
   probe trajectory 83.6 → 96.1 → 88.9cp (paired vs live +12.3 → +24.9 SIG
   → +16.1 SIG) = value OSCILLATING sideways, not closing on the ~74 gate,
   while pool-eval kept improving (eval_wdl also oscillated on-pool) —
   ruler-vs-pool divergence caught ONLY by the probes. Policy losses
   improved monotonically throughout (policy parity still unmeasured).
   Boundary executed: phase1 killed at the stall; salvage pool restored to
   its pristine 808 (the 331 fed shards moved reversibly to
   seeds/slot_000/fed_overlay_quarantine_20260709/, still hardlinked in the
   dedicated pool); CONTINUATION launched 08:13 (fixed driver `continue`:
   warm-start from the ~24k-step trainer.pt, reuse=1 on the fed pool @2.26M
   positions, inherited flat LR, best-eval tracking + plateau stop live) —
   natural ~5h window, then the next probe-driven lever decision. Snapshot
   trail: scratchpad/scaleup/trainer_snap_step14100.pt + per-probe snaps.**
5. After FEN seeding reads out: gradient rebalance rung (`w_soft` 1.0→0.5 +
   `w_wdl` 1.0→1.5) OR teacher-distillation offline screen — pick by what the
   seeding readout says about the tail. Never two value levers in one window.
6. Build: tail metrics in value_regret; SF-frac ramp PR (per-row SF weight
   scaled by disagreement — premise validated 61%→70% monotone); arena cadence
   + loop-health invariant monitor.
7. Restore `sf_pid_regret_tighten_streak_gain: 0.5` when EMA winrate ≈0.52.

**512x16 bootstrap: on-pool eval methodology fix + step-triggered parity probe
(2026-07-10, tooling not a hypothesis test — no kill threshold needed).**
Diagnosed two compounding bugs while reading a plateau on `cont_0710_1114_lr0.00003`:
(a) `offline_replay_epoch.py`'s live-follow eval (`_load_eval_arrs`) built its
2048-position eval set from `iter_shard_paths(args.replay_dir)` sorted ascending
with no shard cap — since eval_positions=2048 fills from 1-2 shards, this was
the SAME ~2048 positions (the pool's oldest surviving shards) re-evaluated
every 500 steps for the run's entire life, not a rolling/representative
sample. `best_eval_loss` set early (step 2000) was never re-beaten afterward —
the model drifting off a frozen 2048-position snapshot, not a real capability
plateau (code already anticipated this: "static windows overfit past the eval
minimum" comment at the plateau-stop site). (b) `parity_probe.sh` (the
trusted frozen-ruler signal) read its `steps=` field from a hardcoded
`train.log` that the phased driver (`run_bootstrap_512x16.sh`) writes but the
current ad-hoc `continue`-launch workflow does NOT — that file went stale
2026-07-10 03:49 and every probe since (06:11/09:11/12:12, all "steps=400")
was silently reading dead state, independent of the overnight pause.
FIX (both tooling-only, no training-target change):
1. `feed_bootstrap_shards.py`: deterministic 1-in-40 (2.5%, by source shard
   index mod 40 — stable regardless of scan order) quarantine split to a new
   `--holdout-dir` (`data/scaleup_pool_512x16/holdout_shards`), never fed
   into `--boot-dir`. Smoke-tested (100 fake shards → 97/3 split, idempotent
   re-run, incremental-feed correctness) before touching the live feeder.
2. `offline_replay_epoch.py`: new `--eval-replay-dir` (+ independent
   `--eval-max-shards`/`--eval-newest-shards`, decoupled from the existing
   training-shard `--max-shards`/`--newest-shards`) lets live-follow eval
   point at the quarantined holdout instead of the training pool. Default
   (unset) preserves exact legacy behavior for every other caller of this
   script (the offline multi-candidate sweep path was deliberately left
   untouched — out of scope, don't want to silently change sweep results).
3. `parity_probe.sh`: new `step-loop` mode reads `steps=` from the newest
   `cont_*.log` by mtime (not the stale `train.log`) and fires a probe every
   `STEP_INTERVAL` (default 1500) steps of real progress, plus immediately on
   any run-stop event (`live_follow_stop`/`live_follow_done`/`candidate_done`/
   `run_boundary`) — closes the gap between a run plateauing and the next
   lever decision having fresh evidence, instead of waiting up to the old
   fixed 3h timer. Measured probe cost ~46-60s wall-clock, GPU-mem-isolated
   from the trainer (0.15 vs the trainer's 0.30 `--mem-fraction`) — no
   visible throughput dip in the one window directly measured; at 1500-step
   cadence (~33-50min at current ~130 samples/s) that's ~2-3% worst-case
   overhead. Old fixed-interval `loop` mode kept for other callers.
   Deliberately did NOT fold the probe into the in-loop stop/LR-drop
   decision (the other option discussed) — would require running
   value_regret as a subprocess from inside the training loop and would make
   the stop decision relative to a MOVING target (delta vs whichever live
   vdump exists at that instant), reintroducing the same moving-bar noise
   trap as the STALE-BAR PLATEAU incident above. Kept parity fully separate,
   deliberately, per that lesson.
Swapped the live watcher from `loop` to `step-loop` (pid 32251 killed, new
pid 118961) — first real read (2026-07-10 13:44, steps=4950 finally correct,
was always reading the dead train.log's frozen "400" before): **boot 75.8cp,
paired vs live +9.81 [+1.32, +18.71]** — meaningfully closer to parity than
the stale 88.0cp/+22.02 reading that persisted through the entire overnight
pause; the lr=3e-5 leg may be working, first *real* evidence either way.
CAVEAT: `--eval-replay-dir`/holdout-quarantine only takes effect on the NEXT
`continue` launch — `cont_0710_1114_lr0.00003` is still running against the
OLD (stale-oldest-shards) eval set until it stops or is relaunched; its
on-pool `eval_plateau` bookkeeping should still be read with that caveat
until then.
UPDATE (13:59): relaunched right after the step-5500 checkpoint (minimal lost
compute) as `cont_0710_1359_lr0.00003_holdout`, same lr=3e-5/warm-start,
`--eval-replay-dir` added. **TRUE HOLDOUT BASELINE measured (14:11) to
resolve the transitional-bar caveat**: snapshotted the exact pre-relaunch
warm-start weights (`trainer_best_eval.pt` at that moment, saved as
`scratchpad/scaleup/baseline_check/pre_holdout_snapshot.pt`) and ran a
1-step (lr=3e-5, negligible drift), eval-only pass against the NEW holdout
set in an isolated `--out-dir` (never touches the live run's checkpoints):
**eval_loss = 4.75008** — confirms the caveat was real: the live run's own
seeded plateau bar (4.6379, from the OLD stale-shard eval) is NOT the right
number to compare against; 4.75008 is the correct apples-to-apples "before"
for judging whether `cont_0710_1359_lr0.00003_holdout`'s own evals (first
one due ~step 500, ~14:15) represent real improvement. Gotcha hit twice
retrying this: `--mem-fraction 0.10` and `0.25` both OOM'd on this 512x16
model's forward pass (8GB wasn't enough even for a single step) — 0.30 (the
live run's own value) is the correct floor for this architecture; don't
retry below it.
**FIRST REAL EVAL under the new methodology (step 500, ~14:15): eval_loss
4.68409 vs the 4.75008 baseline = -0.066 (-1.39%), genuine improvement.** But
the run's own plateau bookkeeping logged it as a `live_plateau_tick`
(streak 1/6), not a new best — still comparing against the stale seeded bar
(4.6379) from the OLD methodology, which this doesn't beat. `trainer.pt`
(saved every 500 steps regardless of plateau status) reflects the improving
weights; `trainer_best_eval.pt` does not update until/unless a later eval
beats 4.6379 specifically — and the probe prefers `trainer_best_eval.pt`
when present, so probe readings will lag the real trajectory until that
resolves. FIX (same playbook as the 07-09 stale-bar incident, already
anticipated in the code's own comment at the seed site — "a stale baseline
errs conservative, which is the right direction"): moved
`trainer_best_eval.json` aside to `trainer_best_eval.json.stale_bar_baseline_swap_20260710`
(reversible; `.pt` left in place — the WEIGHTS are still legitimate, only
the recorded loss value was incomparable). Scope: only affects a FUTURE
`continue` launch (self-seeds from its own first eval instead) — does
NOT change `cont_0710_1359_lr0.00003_holdout`'s already-in-memory bar; that
run keeps ticking plateau against 4.6379 until it stops or beats it. Manual
tracking of the raw eval_loss trajectory vs the 4.75008 baseline is the
trustworthy read until then.
**CONTAMINATION BUG found + fixed (14:2x): 4 of 21 holdout shards
(031120/031160/031200/031240) were ALSO present in the training pool** —
`feed_bootstrap_shards.py`'s `_is_holdout()` was a pure function of source
index, so a shard already fed to `--boot-dir` (all of them, pre-fix — the
old code fed everything with no split) could still be routed to
`--holdout-dir` later if its index happened to land on a holdout slot,
since the holdout dedup check only looked at `holdout_names`, never
`boot_names`. FIX: `_is_holdout()` now returns False if the shard is
already in `boot_names`, regardless of index — holdout membership is
"never trained on," not just "right index." Contaminated shards moved to
scratchpad for forensics; holdout now 17/17 clean (verified zero overlap).
**Our two eval numbers (4.75008 baseline, 4.68409 @step500) are UNAFFECTED**
— `_eval_shard_paths` fills to `--eval-positions` in ascending order and
only needed shard_030480(+030520), neither contaminated; pure luck of
which shards sorted first, not a property of the fix. Separately flagged
by the user: the on-pool holdout, even clean, is still a 2.5% slice of the
SAME self-play stream the model trains on — same-distribution, not an
independent benchmark. Reordering which shards get read
(`--eval-newest-shards`) doesn't change that. Framing going forward: this
holdout is a cheap same-distribution plateau/memorization-detection signal
only; it does NOT replace the parity probe (deep-SF-anchored, genuinely
out-of-training-loop) as the trusted quality signal. Not building a fully
independent holdout corpus — redundant with what the probe already is.
**`cont_0710_1359_lr0.00003_holdout` ran to its `eval_plateau` stop (step
7000, 17:31): on-pool eval_loss best 4.588688 (from the 4.75008 baseline —
genuine -3.4%), but the parity probe (`scripts/paired_compare.py`, sign
convention: delta = A-B, negative = A better) showed boot vs live ckpt722
flat at **+7 to +10cp WORSE** the whole afternoon (SIG in 5/6 readings, final
+9.29 [+0.76, +18.51]) — the on-pool plateau does not track the metric that
actually matters, so stopping on it was actively counterproductive here.
FIX (2026-07-10 21:20): relaunched as `cont_0710_2120_lr0.00003_noplateau`
with `--live-plateau-evals 0` (disables the stop entirely — confirmed via
its own `live_follow_ready` log line: `"plateau_evals": 0`) and
`--live-max-steps 0` ("run until stopped" per the flag's own help text),
warm-started from the plateaued `trainer_best_eval.pt`, same lr=3e-5,
`--replay-dir`/`--eval-replay-dir` pointed at the live `feed_bootstrap_shards.py`
pool (`data/scaleup_pool_512x16/{replay_shards,holdout_shards}`, fed every
30 min by `scratchpad/scaleup/feed_loop.sh`, already running throughout).
This makes the sidecar behave like the main live loop — continuous ingest
of freshly-fed selfplay shards, periodic `trainer.pt` checkpoints, no
artificial stop — instead of requiring a manual re-`continue` every time the
on-pool metric plateaus. `trainer_best_eval.pt`/`.json` will no longer
update (that tracking is gated on `plateau_evals > 0`) — `trainer.pt` is now
the only checkpoint and is the one to warm-start any future leg from. The
step-loop `parity_probe.sh` (already running, fires every 1500 steps + on
any stop event) is now the sole governing signal for this sidecar; it will
also fire once on `credit_exhausted_idle` if live data ever stops flowing
for `--live-idle-exit-after` (600s, unchanged) — that remains the only
"real" exit path.
**PROBE BLINDNESS FIX (22:29): the no-plateau regime broke the probe** —
`probe_once` preferred `trainer_best_eval.pt` when present, which this run
never updates, so the 21:23/22:18 probes scored the SAME frozen 16:00
weights (identical 75.3cp/+1.84 lines while training advanced past step
1550). Fixed to score the NEWEST of best-eval/trainer.pt by mtime (in a
plateau-tracked run trainer.pt saves every 500 steps regardless, so
"latest trajectory" is what the lever decision should see in both regimes
— the stale-bar incident already showed best-eval can freeze on a lucky
outlier). Watcher restarted; verified 22:32: `ckpt=trainer.pt`, boot 77.3cp,
+3.89 [−4.91, +12.68] NS vs vdump_732 — first genuine reading of the
no-plateau run's own weights.
**FIRST FULL SWAP-GATE MEASUREMENT (07-10 22:32-22:5x,
`scratchpad/scaleup/gateread/`): GATE FAILS — on PANELS ONLY, and
decisively.** Boot snapshot = trainer.pt @~step 1650 of the no-plateau run
(≈30k cumulative steps); live side = ckpt737 audit dump + ckpt732 monitor
dumps. The policy half of the gate had NEVER been measured before this.
(1) VALUE: +3.89 [−4.91, +12.68] NS (probe, same-night) — passes.
(2) AUDIT POLICY (2000 pos, paired vs ckpt737 dump, join `key`): search
E[regret] +0.20 [−7.93, +9.39] and search top-1 −0.10 [−8.75, +9.47] —
DEAD PARITY; raw E[regret] −2.14 [−6.48, +2.13] NS, raw top-1 −5.90
[−16.94, +5.74] NS (boot point-worse on raw, search fully masks it) —
passes the letter of the gate.
(3) PANELS: boot v1 BLIND **26/35** vs live 13/35, severity paired −0.28
[−0.37, −0.19] SIG WORSE; v2 **77/113** vs 43/113, −0.26 [−0.31, −0.21]
SIG WORSE — "panels not worse" FAILS by a factor of ~2 on both panels.
READING: the boot reached broad-ruler parity (value + search policy) at
~30k offline steps — strong evidence FOR the capacity hypothesis — but has
essentially NONE of the blind-spot/tail knowledge the live net spent 5
weeks of FEN-seeded RL acquiring (the panels measure exactly that; the fed
pool contains dole-seeded shards but at far lower concentration than the
live loop's replay window ever held them). Raw-policy top-1 (−5.9 NS) is
the other lagging axis. IMPLICATION: the swap decision is now a
tail-transfer problem, not a broad-quality problem — levers in rough order
of cheapness: (a) keep feeding (panels should close slowly as dole-seeded
shards accumulate in the pool), (b) preferentially over-feed seeded/
blind-spot-heavy shards to the sidecar (offline analog of the seeding
lever; screenable vs the panel dumps in hours), (c) swap below panel
parity and rebuild awareness in-loop post-swap — (c) violates the
pre-committed gate and risks re-learning what took 5 weeks; not on the
table without a deliberate gate amendment. Re-measure panels at the next
natural pause (they're ~4 min each on the boot ckpt); the paired command
set is banked in `gateread/run_gateread.sh`.

**SEEDED OVER-FEED LAUNCHED (lever b) 2026-07-10 ~23:00** — hardlink-duplicated
the dole-era pool slice (598 shards, idx >= 30749, first fed 07-08 = dole
launch) 2x each via `scratchpad/scaleup/overfeed_seeded.sh` (undo:
`overfeed_seeded.sh undo`); pool 1406 -> 2602 entries, dole-era sampling
weight ~42% -> ~69% (~live-window recent-data concentration). No code
changes; the sampler credits the dups as fresh positions so the extra step
credit lands on the seeded slice. Hypothesis: the panel gap is a data-dose
problem, closable offline. YARDSTICK (pre-committed): panels v1+v2 on the
boot ckpt after >=12k more steps, paired vs the live732 dumps — auto-runs
via `gateread/panels_recheck.sh` (log: `gateread/panels_recheck.log`).
SUCCESS = v1 BLIND materially toward 13/35 with the paired-sev CI clearly
better than the pre-overfeed −0.28 (and the value probe not regressing,
watched by the step-loop probe). FAIL = panels ~unmoved after a full
overnight of ~69%-seeded training → evidence that tail knowledge transfers
only in-loop → the case for a deliberate gate amendment (swap at broad
parity, let live dole seeding close the tail), NOT for more sidecar seed
engineering. Confound: the same-night no-plateau continuation
(cont_0710_2120, lr 3e-5) is the only other change in the window.

**OVER-FEED READOUT 2026-07-11 04:05 — VERDICT: WORKED (by the pre-committed
rule); gate NOT yet passed, continue.** After ~10.3k over-fed steps
(recheck @step 12100): panel v1 BLIND 26/35 → 22/35 (live 13/35), paired
sev gap vs live732 shrank 0.28 → 0.14 [0.06, 0.22]; v2 BLIND 77/113 →
55/113 (live 43/113), sev gap 0.26 → 0.11 [0.06, 0.15]. Both CIs clearly
exclude the pre-overfeed gap = SUCCESS per the pre-committed rule; the
panel deficit is a DATA-DOSE problem, not an in-loop-only transfer problem.
Value stayed clean through the dose (probe 00:21: 72.5cp, −0.96 NS vs
live732 — first negative point delta ever; 01:07 +1.43 NS). DECISION: do
NOT swap yet (panels still short of the 13/35 / 43/113 gate bar); keep the
over-fed training running — gap halved in one night, so parity is
plausibly 1-2 more nights out. Re-run `gateread/panels_recheck.sh`
(MIN_STEPS bumped) or the full `run_gateread.sh` before any swap call.

**DOSE-2 KILLED 2026-07-11 11:53 — value regression (pre-committed kill).**
Dose 1 exhausted credit at step 19180 (natural stop); a second continuation
(cont_0711_0757, fresh full credit over the SAME over-fed pool) re-trained
the same data and the value probe degraded monotonically: 84.4 → 81.1 →
84.7 → 91.5cp, +19.33 [+8.65, +30.48] SIG vs live742 — the
offline-distillation value trap firing on a repeat epoch. Killed at step
13050; degraded weights banked (`gateread/boot_dose2_step13050_degraded.pt`,
forensics only — do NOT use). LESSON: the over-feed dose is ONE-PASS; fresh
credit over an already-consumed pool = memorization, not tail transfer.
**Swap candidate = dose-1 step-12100 snapshot**
(`gateread/boot_snap_recheck_0711_0404.pt`): panels v1 22/35 / v2 55/113,
value 76.9cp +4.76 [−3.96, +13.86] NS vs live742
(`gateread/vdump_boot_0404snap.jsonl`). Remaining panel gap needs FRESH
seeded data (live feed drip or post-swap in-loop), not more epochs.

**ARENA READ + GATE AMENDMENT — SWAP APPROVED 2026-07-11 (user decision).**
Paired-opening arenas, boot dose-1 snapshot (`boot_snap_recheck_0711_0404.pt`)
vs banked live ckpt737, production search settings: **sims=1 (raw policy):
+9.6 Elo [−22.9, +42.2]** over 200 pairs — statistical parity, boot
point-ahead (`arena/sims1.jsonl`). **sims=32: −66.2 [−115.1, −19.7]** at 93
pairs, SIG behind — the gap appears only under search
(`arena/sims32.jsonl`, stopped at ~100 pairs by design; the 400-game run
OOM'd twice concurrent with training — 512×16 search ≈ 700MB/game, eager
32-concurrent ≈ 22.5GB, do NOT run vs a live trainer). Search-tuning
sensitivity deliberately NOT tested (user call: pre-selfplay tuning has no
shelf life). AMENDMENT RATIONALE: the original gate (panels not worse)
assumed offline feeding could reach panel parity; dose-2 proved the offline
path is exhausted, the 46M net is the plateaued one (all reweighting dead,
value flat 3 weeks), boot learns faster (broad parity in 30k steps from
init), sims=1 parity says the nets are equal ex-search, and the sims-32
deficit is a search-integration gap that in-loop selfplay (training under
its own search) fixes fastest — precisely what the sidecar cannot teach.
**PRE-COMMITTED REVERT RULE: after ~5 days (or ~50 iters) post-swap, run a
32-sim cross-series arena, new net vs banked ckpt751 trunk
(`data/salvage/recover_ckpt751_20260711`); if not at parity or better
(CI excludes 0 on the losing side), salvage-restart the 46M trunk. Panels +
paired value at ~10 iters as the early canary (regression >2cp SIG = early
revert).** Runbook: `scratchpad/scaleup/swap_runbook.md` (trigger section
amended this session).

**cheese64 tranche FIRST READOUT @ckpt742 (2026-07-11 01:38) — ON TRACK.**
Pre-committed primary (v1 holds/improves vs 13/35): HELD at 13/35. v2
improved 43 → 38/113 (new best). Value 72.1cp, +3.73 NS vs 609 — kill
condition (value worse >2cp paired vs ckpt732) not triggered. Retirement
resolved 37/144, retired 6 → pool 138 (`blindspot_fens_retire_742.txt`).

**Seed reinstatement for the 512 net + restart onto #147 — LIVE-UNREAD (2026-07-11, ~iter 6 of trial 4c17c).**
Re-audit of all 40 ever-retired blind-spot seeds against the swap net
(4c17c ckpt_000004): 30 read AWARE (<= -0.4) — retirements transfer; 3 read
still-BLIND (> -0.2; +0.47/+0.45/-0.18) and are REINSTATED with fresh streak
state → `data/blindspot_fens_reinstate512_20260711.txt` (141 seeds =
retire_742 + 3). Active-list removals left to the 2x-consecutive auto-retire
(single-read churn 35-47%). Per-seed netq: scratchpad/seed_reaudit/ (session
scratchpad). Bundled into the SAME restart: PR #147 (broker gather fix,
UCI compact-bf16 gate fix 37c96dce, explicit v2_threats yaml default) —
performance/plumbing only, no target semantics. Yardstick: the swap canary
itself (panels + paired value vs pre-swap trunk at ~10 iters, >2cp SIG value
regression = revert to recover_ckpt751). Confounds: swap readout window —
accepted deliberately (3 seeds at dole-1/iter cannot plausibly move a 2cp
value bar; #147 changes no training targets).

**#147 speed check (pre-committed at the 2026-07-11 bundle restart).** #147's
broker gather fix claims a selfplay-side speedup. Pre-restart baseline on the
512x16 trial (iters 3-5, steady state, iter 2 excluded as post-restart burst):
~780-920 games/h selfplay, trainer 0.27-0.30 steps/s / ~140-150 samples/s,
~2350-2400 s/iter at ~500-620 games/iter. RULE: compare the same metrics over
the first 3-4 steady-state iterations after the restart (skip the first — the
restart burst inflates games/h via queued worker shards). SUCCESS = games/h
holds or improves; REGRESSION = games/h down >15% sustained with trainer
steps/s flat → suspect #147, consider reverting its broker path. Trainer
throughput is the control (should be untouched by a broker change).

**Pause-hold (PR #149): stop abandoning in-flight games at the train-phase
pause — PRE-COMMITTED, activates at the post-canary restart (2026-07-11).**
Finding (worker logs + code trace): since d9ac008b (03-20) every iteration's
pause_selfplay tore down play_batch, discarding ~384 in-flight games/worker
(vs ~180-235 completed per session on the 512 net → ~half of selfplay compute
discarded) and censoring games longer than one selfplay window out of replay
entirely. Fix: pause now HOLDS the loop (slots preserved, model hot-swaps at
unpause); teardown paths unchanged; CAE_WORKER_STOP_ON_PAUSE=1 reverts.
Hypothesis: games/h up materially (up to ~2x ceiling) AND long-game census
bias removed (replay mean game plies should RISE). YARDSTICK (one deciding):
paired games/h + replay avg plies over 3-4 steady-state iterations pre/post
the activating restart (pre = the #147-readout iterations, same net/window);
plus the new `in_flight_abandoned` log line ≈0 at pause boundaries. KILL:
games/h down >10% sustained, or worker deadlock symptoms (0 games generated
for 2 consecutive iterations with workers alive), or value/panel guardrails
of the live swap canary regressing coincident with activation → set
CAE_WORKER_STOP_ON_PAUSE=1 + restart (no salvage needed — no weights/replay
surgery). CONFOUND: activates in the swap-readout era; user chose immediate
activation (2026-07-11 evening, ~iter 8) over waiting for the 10-iter canary
— the per-iteration waste outweighs the small canary confound (no target
semantics change); the 5-day arena gate
tolerates a throughput-only change (does not alter targets; data-mix change
= longer games entering replay, which is the intended fix).

**#147 speed READOUT (2026-07-11 evening, n=1 by design — read early at the
user's call because PR #149 activates at the next boundary and confounds
games/h upward).** Only iter 7 is usable (iter 6 = restart iter, skipped per
rule). Headline: 641 selfplay games/h vs baseline 748-861 looks like -15-20%,
BUT the baseline iters 3-5 counted 187-198 STALE games/iter (pre-restart
backlog); iter 7 has stale=0. On a fresh-games basis iter 7 (430 fresh, 641/h)
BEATS baseline fresh (~386/293/293 → ~440-580/h). The control (trainer
steps/s) read 0.198 vs 0.27-0.30 baseline — NOT flat, so the rule's
broker-blame branch doesn't fire anyway; likely cause is the concurrent
seed-reaudit GPU scoring run during iter 7. VERDICT: NO REGRESSION (pass) —
fresh-games throughput at or above baseline; positive-speedup claim unproven
at n=1 and now unreadable (post-#149 iters measure the pause-hold instead).

**slot_oversubscribe ACTIVATION 2.0 (2026-07-12 ~00:00, restart @iter 9, LIVE).**
User chose immediate activation bundled with the #149 pause-hold restart era
(iteration had just begun; minimal compute lost). `selfplay.slot_oversubscribe: 2.0`
added to the live yaml AT the restart onto merged #151 code (restart-gated
key; added only once the running code defines it, per the live-yaml-key rule).
CI never triggered on PR #151 (zero workflow runs; close/reopen didn't kick
it) — merged on the strength of the full LOCAL test suite + lint gate in the
PR worktree plus the adversarial review (0 bugs). Yardstick per the #151
ledger entry (complete_gps + sf_block_starved share), read against the
post-#149 iterations 9+ — CONFOUND: #149 pause-hold and #150 coalescer
activate in the same window, so per-change attribution is impossible; the
combined games/h vs the iter-7 fresh-games baseline (641/h) is the deciding
number, and sf_block_starved share is #151's mechanism-specific check. WATCH:
SF pool queue depth (selfplay_fraction is 0.30 → ~2x concurrent curriculum SF
queries at factor 2.0), worker RSS (+0.5-1.8G expected), broker p95. KILL
(any): games/h below the 641/h baseline sustained 2 iters, worker RSS growth
>4G, broker p95 +20% → revert `slot_oversubscribe: 1.0` + restart (immediate,
no salvage).

**Triple-restart READOUT (iters 10-12, 2026-07-12 session): WORKED.** Combined
games/h vs the 641/h iter-7 fresh baseline: 1142 / 807 / 1142 (+78% clean
iters, +52% blended) — far above the +15% bar. #149 mechanism confirmed:
replay avg plies 105 -> 119-122 (long-game censoring gone), zero abandonment
at training-pause boundaries, trainer steps/s healthy (0.30-0.31), GIL settled
~1.1ms. #151 mechanism partial: sf_block_starved 0% during burst but ~2300%
of thread time again in steady state — the residual wall is SF pool capacity
(~20 procs serving 2x concurrent curriculum queries), not slot scheduling;
next speedup lever = SF pool sizing, NOT a higher factor. No kill criterion
hit (RSS fine, broker wait_ms ~0.03).
BUG FOUND during readout: a restart-classified recommended_worker key is
flapping — teardown restarts at the END of iters 10 and 12 (not 11), each
abandoning ~768 in-flight games, making every post-flap iteration ~807/h vs
1142/h. Worker log doesn't name the key; manifest-diff trap armed. Fix =
identify + either make the key live-applied or stop the server flapping it;
worth ~+35% games/h on affected iterations.

**Threaded live-reco fix (PR #153) ACTIVATED at restart (2026-07-12 ~04:30).**
Root cause of the flapping teardowns found in the triple-restart readout:
`opponent_wdl_regret_limit` (LIVE key) fell back to a full session restart
because the threaded path never registered states for live apply — every PID
lever move past deadband abandoned ~768 in-flight games (1142 vs 807 games/h
alternation). Fix: lock-guarded state registry, live fields transplanted onto
ALL thread states + late registrants (review race closed). YARDSTICK: worker
logs show `live reco (N state(s))` and NO `restarting selfplay session` at
PID moves; games/h holds ~1142 on ALL iterations. KILL: 0 games for 2
consecutive iters with workers alive, or games/h below the 641 baseline →
revert = git revert PR #153 + restart.

**SWAP CANARY READ @iter 20 (2026-07-12, scratchpad/canary_512_iter20/).**
Rule-as-written (live vs 46M ckpt751 trunk, >2cp SIG = early revert): FIRED —
paired value +8.42cp [+0.09, +16.86] SIG-marginal, panels v1 22/35 vs 11/35,
v2 63/113 vs 33/113. BUT the baseline disambiguation probe (same ruler on the
swap-time boot net data/salvage/swap_512x16_20260711) shows the gap is
PRE-EXISTING, not RL-caused: boot 76.9cp vs live 77.6cp, paired +0.74 NS
[-8.26, +9.60]; panel v1 unchanged from swap-time (22/35). The canary's
intent (catch in-loop value collapse) did NOT occur. Since swap: value flat,
v1 flat, v2 drifted 55->63/113 (mild negative, watch), policy/search regret
at PARITY with search E[regret] point-AHEAD (-1.70 NS) — consistent with the
swap thesis (search-integration gap, selfplay fixes it, judged at the 5-day
sims-32 arena). DECISION (user, 2026-07-12): HOLD to the 5-day sims-32 arena gate — no
in-loop regression occurred; the rule's trunk-baseline conflated the accepted
swap-time gap with regression. Follow-ups: repeat this canary ~iter 35
(baseline = the boot net AND the trunk, both banked dumps in
scratchpad/canary_512_iter20/), watch panel v2 drift.

**Post-speedup knob check (2026-07-12): NO CHANGES.** Steps auto-scale via
train_views_per_position=2.5 (verified: steps track ingest ~2x). Window
left at 1.5M: position-count semantics unchanged, but at ~2x ingest the
window now spans ~1 DAY of data, not ~2 — experiment clocks and revert
contamination both halve. User decision (2026-07-12): window increase
1.5M->3M PLANNED FOR LATER (restore ~2-day data lifetime at the new ingest
rate); not now, to avoid stacking another data change into the swap/speedup
readout window. Per-position reuse stays pinned at 2.5 views by design.

**PLANNED (drafted 2026-07-12, NOT LIVE — activate only after the swap/dole
readout window closes, alongside or after the v4 feed, never in the same
window as another data change): curriculum-seed refutation (SF-side seeding
at low dose).** HYPOTHESIS: the "fed but not learned" clusters (KDEF family
~0.59 blind despite 32 seeds; M_QLESS) need ON-BOARD REFUTATION — SF actually
punishing the bad plan move-by-move — which the selfplay-only dole cannot
provide (its SF signal is labels only; the stm telemetry exists because two
blind nets can agree a lost seed is fine). History: the prob-path curriculum
seeding co-drove the v1 WORKED verdict before the dole switch dropped it to
zero. Both original objections are resolved: (a) PID pollution — fixed in
code, finalize.py excludes fenlist* games from the PID sample; (b) SF-pool
starvation — the PID sample is now 400+ games/iter post-speedup (was ~5-7),
so a small crowding cost is affordable. CHANGE: keep the dole
(opening_fen_dole_per_iter: 1) AND set opening_fen_prob: 0.02 with
opening_fen_selfplay_only: false (legacy prob path seeds curriculum slots;
net forced onto the blundering seat per opening_fen_net_side_to_move).
YARDSTICK (one deciding): held-out panel severity on the stubborn clusters —
panel v1 BLIND count + v2 KDEF/M_QLESS subset vs the activation-day read,
judged at +10-15 iters by paired severity (scripts/paired_compare.py on the
per-position panel dumps). GUARDRAILS/KILL: (1) PID pooled sample n drops
below ~200/iter sustained (crowding) → revert prob to 0; (2) value_regret
paired CI vs activation-day dump >2cp worse → revert; (3) sf_block_starved
steady-state degrades materially vs the pre-activation iterations → revert.
BASELINE: bank a salvage-export + value dump + panel dumps at activation.
CONFOUND to record at launch: shares the window with nothing (hard rule);
if activated same restart as the v4 feed, the two are ONE combined entry
with a combined kill. Stacks with the SF-pool-sizing lever (more SF procs
pays for refutation games + PID sample together).

**INCIDENT (found 2026-07-12): FEN seed injection silently OFF since the
07-11 swap restart.** Pause-hold (#149) sessions start before the first dole
arrives; `fen_dole_queue = self._live_dole_queue or None` handed the session
None and every dole refilled an orphaned list ("dole: received 141 seed(s)
-> live session" logged every iteration, ZERO fenlist outcome keys in 40
recent shards — all selfplay games came from books). Fix = PR #154
(_promote_pending_dole hands the queue OBJECT, empty or not); worker code,
restart-gated — ACTIVATED EARLY 2026-07-12 (user call: pulled the iter-35
pause forward; restart bundled with the Cheese block, canary probes stay at
~iter 35 concurrent-safe). CONFOUNDS: (a) the iter-20
canary's panel-v2 drift (55->63/113) happened in a NO-SEED window — re-read
after seeds resume before treating drift as a swap property; (b) the dole
LIVE-UNREAD readout clock restarts at fix activation; (c) retirement reads
between 07-11 and activation scored a net trained WITHOUT seed exposure.

**INCIDENT (found 2026-07-12): C classifier shortened FEN-seed game budgets.**
The Python timeout path correctly compares plies played since the opening, but
the active native `classify_games` fast path compared absolute `CBoard.ply`
directly with `max_plies`. The current 141-seed production list starts at
absolute plies 4-179 versus `max_plies: 450`, so seeds were not instantly
dropped, but each could lose 4-179 plies of its intended continuation budget.
Fix: maintain a per-slot int32 starting-ply array across create/recycle and pass
it into the C classifier, which now tests `board.ply - starting_ply >=
max_plies`. Regression runs the same fullmove-69 seed through Python and C,
proving it remains live at zero played plies and times out after exactly the
configured two. Restart-gated; do not interrupt the live run solely for this,
but include it in the next graceful restart.

**Pause-hold invariant audit (2026-07-12, post-#154; report:
scratchpad/pausehold_audit_report.md).** Static end-to-end audit for more
#154-class bugs (session-start-only state vs hours-long #149 sessions): every
_RECO_LIVE_KEY verified to reach a RUNNING session; restart-key coverage
complete; threaded path sound. One structural finding ACCEPTED BY DESIGN:
held games span model publishes — early plies searched by the previous
weights are tagged with the NEW model sha at completion (~games_per_batch x
oversubscribe games per pause). Standard continuous-selfplay behavior
(LC0/KataGo train on such games); alternatives re-create the #149 waste.
Consequence for readers: "fresh-sha games" slightly overstate on-policy
freshness right after each publish. Hardening landed with this entry: TB
adjudication gate follows live state.game; dole-queue docstring forbids
rebinding (the #154 footgun); mid-session poll failures now log a warning
instead of a bare pass.

**Completion/upload overlap experiment -- WORKED OFFLINE; production read pending
(2026-07-12).** Post-#151
telemetry shows completion waves accumulating roughly 2400-5000% of wall time
in `finalize`, while broker waits and the GIL probe are small. Threaded workers
currently hold the shared upload-buffer/model-tag lock across pending-shard tar
packing and HTTP upload, serializing all 32 completion callbacks behind network
I/O. Hypothesis: restrict that lock to upload-buffer mutation and atomic
pending-shard creation, then use the existing independent pending-upload lock
for tar/HTTP work; this increases completion throughput without changing game,
sample, shard, model-tag, or retry semantics. ONE deciding offline yardstick:
`TMPDIR=/tmp python3 -m pytest -q tests/test_worker_model_update.py
tests/test_worker_upload_response.py tests/test_worker_small_uploads.py`, plus
the concurrency regression
`test_completed_game_does_not_wait_for_active_shard_upload`. SUCCESS: all
focused tests pass and a second callback completes while the first is held in a
synthetic upload; KILL: any lost/duplicated positions, model metadata mismatch,
upload outside the existing pending-upload serialization, or callback remains
blocked. Runtime mechanism read after the next natural restart: broker logs must
show `upload_busy > 0` without the same upload time appearing as serialized
completion lock-wait; overall games/hour is the production outcome but is not a
merge gate because training phase mix is nonstationary. No salvage/revert is
needed: this is scheduling-only, and revert is the code commit. **OFFLINE
VERDICT: WORKED.** The deciding focused gate passed 65/65; the concurrency test
held the first callback inside the pending uploader while a second callback
both flushed its completed game durably and returned without waiting. The broad
worker sweep passed 94/94, and ruff + basedpyright + vulture were clean. The
production outcome remains explicitly unread until a natural restart supplies
the broker completion telemetry above.

**Worker shard-materialization profile -- UNREAD (2026-07-12).** PR #159 moves
tar packing and HTTP outside the completion lock, but
`samples_to_arrays` plus compressed zarr creation still run inside it. The
post-oversubscription worker showed 2400-5000% cumulative `finalize` time during
completion waves, so disk materialization may remain a lock convoy even after
network overlap. Hypothesis: production-shaped 500-position shard conversion
and zarr creation cost >=50ms and >=50% of the local conversion+zarr+tar
pipeline, justifying a follow-up ownership-transfer queue that writes detached
buffers outside selfplay threads. ONE deciding yardstick:
`PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3 scripts/bench_worker_shard_pipeline.py
--positions 500 --rounds 7`. SUCCESS: both thresholds hold with stable array
hash/tar size and round-trip validation; otherwise FAILED and do not add an
asynchronous buffer queue. This is offline profiling only; no live/data change
or salvage snapshot.
**PROVENANCE CORRECTION BEFORE VERDICT:** the first harness reused one identical
board tensor for all 500 rows, making compression unrealistically easy. It was
corrected before judging the implementation to generate a distinct binary
175-plane tensor per position; thresholds and command are unchanged.
**VERDICT: WORKED.** Seven corrected measured rounds at 500 positions produced
median `samples_to_arrays` 155.198ms, compressed-zarr write 401.695ms, and tar
packing 29.043ms. The 556.893ms materialization section is 95.04% of the local
pipeline, with stable array hash `4f4eb31bc2e62c50` and stable 2,263,040-byte
tar. This clears both gates by a wider margin; proceed with detached-buffer
ownership transfer, keeping a global queued-position cap and failure retry at
the queue head.

**Detached shard-materialization experiment -- UNREAD (2026-07-12).**
Hypothesis: swapping a full `_BufferedUpload` for a fresh buffer under the
model-tag lock, then letting the single existing pending uploader materialize
the detached owner, removes the ~557ms completion lock convoy without losing or
duplicating samples. ONE deciding yardstick: focused worker/upload tests plus a
deterministic two-callback test that blocks the first materializer while the
second callback detaches another full buffer and returns. SUCCESS: second
callback completes before release, exact game/position totals are retained
across active+queued+pending state, same-second shards have unique paths,
materialization failure retains the original queue head, queued positions count
toward the OOM cap, and all worker tests/lint pass. KILL: any invariant fails;
otherwise WORKED. Runtime outcome after a natural restart is lower steady
`finalize` thread-time and higher completed games/hour, but is not the merge
gate because the offline lock hold falls from measured disk time to O(1) owner
transfer. No data semantics/config/salvage change.
**VERDICT: WORKED OFFLINE; production read pending.** The deterministic
two-callback test held the first callback in materialization/upload while the
second detached another complete buffer and returned before release; exact
positions remained split between the first durable materialization and the
second queued owner. Failure injection retained the original queue head and
position count; sidecar failure removed the partial zarr while preserving the
source buffer; same-second identical metadata produced distinct durable paths;
and detached positions triggered the existing global OOM cap. Validation:
70/70 focused worker/upload tests, 155/155 broad worker/shard/distributed tests,
15/15 continuous-selfplay tests, and ruff + basedpyright + vulture clean. Keep
the queue; after the next natural restart, use broker completion telemetry and
games/hour to quantify the production gain rather than extrapolating the
557ms lock-hold removal into a headline throughput percentage.

**Worker zarr codec experiment -- UNREAD (2026-07-12).** Corrected
production-shaped profiling attributes 401.7ms of each 500-position flush to
Blosc zstd level 3, the largest remaining shard-pipeline cost after detached
materialization. Hypothesis: a lower-cost lossless Blosc codec/level materially
reduces worker CPU and uploader occupancy without making local/network shards
unreasonably larger. ONE deciding yardstick: `PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_worker_shard_codecs.py --positions 500 --rounds 7`, alternating
zstd3 baseline with zstd1 and lz4 level 1/3 on identical pruned production-width
arrays. SUCCESS: one candidate median write time <=0.85x baseline, stored bytes
<=1.50x, eager read time <=1.10x, and exact decoded core-array hashes in every
round; choose the fastest qualifying candidate. Otherwise FAILED and retain
zstd3. Focused shard/server/worker tests and lint must pass before shipping.
This changes lossless storage representation only, not samples or training
semantics; no salvage/live readout.
**VERDICT: FAILED for the registered candidates.** zstd1 write ratio 0.534 and
size ratio 1.256 passed, but eager-read ratio 1.142 exceeded the 1.10 guardrail.
LZ4 levels 1/3 wrote at 0.424/0.396x and read faster, but inflated storage and
network bytes 3.805/3.811x, far beyond 1.50. All decoded hashes were the exact
stable `656869c08d2bcc3e`. Retain zstd3 unless the separately registered middle
zstd2 dose below passes every original constraint.

**Worker zarr codec zstd2 middle-dose experiment -- UNREAD (2026-07-12).**
The first dose ladder brackets the tradeoff: zstd3 is compact/slow, zstd1 is
fast but narrowly misses the read guardrail, and LZ4 is size-dead. Hypothesis:
zstd2 preserves enough decode/compression behavior to satisfy all three
original constraints while materially reducing writes. ONE deciding yardstick
is the same seven-round codec command, now including zstd2. SUCCESS: zstd2
write <=0.85x zstd3, bytes <=1.50x, eager read <=1.10x, exact stable decoded
hash. Otherwise FAILED and retain zstd3. No semantic/live change.
**VERDICT: WORKED.** Two seven-round alternating runs measured zstd2 write
ratios 0.486 and 0.537, eager-read ratios 0.811 and 1.081, and identical byte
ratio 1.070 (2,213,225 vs 2,068,037). Every decoded core array retained exact
stable hash `656869c08d2bcc3e`. Both independent runs clear every gate; use the
conservative second ratios as the headline. zstd2 is the selected local shard
codec; validate the production writer end to end before merge.
**PRODUCTION-WRITER VALIDATION:** the first pipeline invocation omitted the
repo-required `PYTHONPATH=.` and therefore imported the live checkout's zstd3
writer; it is invalid. Import provenance was then printed and verified as this
worktree's `replay/shard.py`. The corrected real-writer run measured
`samples_to_arrays` 157.488ms, zarr2 write 235.862ms, and tar 32.379ms: zarr
write is 41.3% below the corrected zstd3 pipeline anchor (401.695ms), while tar
bytes rise only 6.3% (2,406,400 vs 2,263,040) and the exact array hash remains
`4f4eb31bc2e62c50`. This confirms the selected codec through the actual writer.
Validation: 58/58 replay-shard, server-upload, and worker-buffer tests pass;
ruff + basedpyright + vulture are clean. Production activation requires only
the next worker/server restart onto the merged writer; old zstd3 and new zstd2
shards remain mutually readable through zarr codec metadata.

**Replay-finalization policy projection cache -- UNREAD (2026-07-12).** A
production-shaped 100-record profile measures 19.35ms/game in replay-sample
construction; policy-vector projection/renormalization costs 3.09ms and legal
mask projection 2.95ms. Each row is projected once for its own sample and again
as the two-ply-earlier sample's future target. Hypothesis: game-local lazy
caches keyed by record index preserve the immutable encoded arrays and remove
the duplicate gathers/normalizations, improving finalization without changing
any target values. ONE deciding yardstick:
`taskset -c 15 python3 scripts/profile_python_native_candidates.py
--only-finalize --finalize-records 100 --repeats 50`, paired baseline/candidate
in alternating rounds with output hashes. SUCCESS: candidate median <=0.92x
baseline and exact stable output hash; KILL: >0.98x, any parity/test failure, or
any target alias is mutated downstream. Otherwise MIXED. This is sample-build
scheduling only; no live activation, salvage, or data readout is needed.
**VERDICT: FAILED and reverted.** Three alternating 50-repeat pairs produced
baseline/candidate medians 20.512/19.099ms, 19.011/19.103ms, and
18.934/20.587ms. The candidate ratios were 0.931, 1.005, and 1.087 (geometric
mean ~1.006, slightly slower), while every run retained the exact stable hash
`a8f5fc45063e6f40`. Saved gathers did not repay per-row dictionary/helper
overhead under the contended production host, and the cache added complexity;
do not ship it. The result also rejects the tiny list-copy cleanup as a speed
PR: that copy is below the measured conversion noise and removing it alone is
not a meaningful simplification.

**Redundant policy-temperature normalization experiment -- UNREAD
(2026-07-12).** `apply_policy_temperature` currently computes `p/sum(p)`,
raises that vector to `1/T`, then normalizes again. The first scalar division
cancels exactly in real arithmetic because the final normalization removes the
common `sum(p)^(-1/T)` factor. Hypothesis: power the nonnegative input directly
and normalize once, reducing both code and one full-vector sum/division per
policy row with equivalent float32 targets. ONE deciding yardstick: three
alternating baseline/candidate runs of `taskset -c 15 python3
scripts/profile_python_native_candidates.py --only-finalize --finalize-records
100 --repeats 50`. SUCCESS: candidate geometric-mean ratio <=0.98, randomized
positive/zero/negative-input parity has max absolute error <=1e-7 and matching
degenerate behavior, and temperature/finalization tests pass. KILL: ratio
>1.00, parity exceeds the tolerance, or tests fail; otherwise MIXED. This is a
mathematically equivalent CPU simplification, not a training/config experiment;
no salvage or live readout is required.
**VERDICT: FAILED and reverted.** The production-shaped finalization hash stayed
exact (`a8f5fc45063e6f40`), but the two completed baseline/candidate pairs were
19.615/20.549ms and 18.934/19.859ms: ratios 1.048 and 1.049, decisively over
the >1.00 kill rule. Removing a mathematically redundant division did not
remove measurable wall work in this NumPy-sized regime and made the measured
path slower, so the original implementation remains.

**zstd2 bit-shuffle experiment -- UNREAD (2026-07-12).** The selected zstd2
codec uses Blosc byte-shuffle. Hypothesis: bit-shuffle better exposes the
binary-plane/float16 structure and either reduces write time or stored bytes
without harming the other dimension. ONE deciding yardstick: the same
`PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_worker_shard_codecs.py --positions 500 --rounds 7`, adding a
zstd2+bitshuffle arm. SUCCESS: versus zstd2+shuffle, either (a) write <=0.95x
and bytes <=1.00x, or (b) bytes <=0.90x and write <=1.05x; eager read <=1.10x
and exact stable hash in both cases. Otherwise FAILED and retain byte-shuffle.
No semantic/live change.
**VERDICT: WORKED.** Seven alternating rounds measured bit-shuffle vs selected
zstd2 byte-shuffle at 128.207 vs 154.605ms write (0.829x), 25.396 vs 27.572ms
eager read (0.921x), and 1,479,580 vs 2,213,225 bytes (0.669x), with exact
stable decoded hash `656869c08d2bcc3e`. It clears both success routes and is
also smaller than the original zstd3+shuffle shard. Promote zstd2+bitshuffle
and validate through the production writer.
**PRODUCTION-WRITER VALIDATION:** the real writer measured conversion 158.222ms,
zstd2+bitshuffle write 201.100ms, and tar 25.306ms. Against the corrected
original zstd3+shuffle pipeline, total materialization falls 556.893→359.322ms
(35.5%) and tar bytes fall 2,263,040→1,679,360 (25.8%), with unchanged exact
array hash `4f4eb31bc2e62c50`. All 58 replay-shard/server-upload/worker-buffer
tests pass; ruff + basedpyright + vulture are clean. This supersedes zstd2 byte-
shuffle as the codec change to publish.

**Shard stack+cast fusion experiment -- UNREAD (2026-07-12).** The corrected
500-position pipeline spends ~155ms in `samples_to_arrays`; required `x` and
`policy_target` currently allocate full float32 stacks and then allocate/copy
again to float16. Hypothesis: constructing those stacks directly at storage
dtype removes the float32 temporaries and one memory pass, reducing conversion
time and peak transient bytes with exact float16 arrays. ONE deciding yardstick:
`PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_shard_stack_cast.py --positions 500 --rounds 9`. SUCCESS: direct
typed construction median <=0.85x stack-then-cast, exact stable hashes/shapes,
and the full pipeline plus shard tests pass. KILL: ratio >0.98 or parity failure;
otherwise MIXED. No storage/data semantic or live change.
**VERDICT: FAILED; no code change.** Nine alternating rounds measured
stack-then-cast at 52.572ms and direct typed construction at 53.141ms, ratio
1.011 over the kill threshold. Both paths retained exact stable hash
`ab8a3ddae7131ca8`. NumPy's stack path offsets the avoided temporary with more
efficient bulk copying; retain the original implementation. The one-off
candidate harness was removed rather than adding dead tooling.

**NumPy stack-dtype fusion experiment -- UNREAD (2026-07-12).** NumPy 2 exposes
a distinct `np.stack(..., dtype=np.float16)` kernel, unlike the failed generic
`np.asarray(list, dtype=...)` path. Hypothesis: dtype-aware stack fuses the
bulk stack/cast without sacrificing the optimized stack implementation. ONE
deciding yardstick: `PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_shard_stack_dtype.py --positions 500 --rounds 9`. SUCCESS:
candidate <=0.85x reference with exact stable hash and shard tests; KILL >0.98x
or parity failure; otherwise MIXED. Requires NumPy 2, already the production
environment; packaging compatibility must be checked before shipping.
**VERDICT: MIXED.** Nine alternating rounds measured 66.817ms reference vs
56.855ms dtype-aware stack, ratio 0.850898, missing the 0.850000 success cutoff
by 0.09 percentage point while retaining exact hash `572af13523110668`. Do not
call this first gate WORKED. The candidate is retained only for the separately
registered modest-gain replication below because it also removes the explicit
post-stack cast and the user-approved policy accepts small simplifying wins.

**NumPy stack-dtype modest-gain replication -- UNREAD (2026-07-12).**
Hypothesis: the exact alternating rerun reproduces a useful >=5% gain from the
simpler dtype-aware stack despite missing the intentionally aggressive first
gate. ONE deciding yardstick is the same nine-round command. SUCCESS: candidate
<=0.95x, exact stable hash, NumPy 1.24 API compatibility confirmed, full
pipeline >=2% faster, and shard tests/lint pass. Otherwise FAILED and revert.
No semantic/live change.
**VERDICT: WORKED.** The exact replication measured 56.682ms reference vs
52.918ms dtype-aware stack, ratio 0.934 (6.6% faster), with exact stable hash
`572af13523110668`. NumPy documents the `dtype` stack parameter as added in
1.24, exactly the project's minimum dependency. Through the real zstd2 writer,
conversion improved 149.559→138.506ms and total materialization
378.597→362.847ms (4.2%), with unchanged full array hash `4f4eb31bc2e62c50`
and 2,406,400-byte tar. All 58 shard/server/worker-buffer tests pass and lint is
clean. Keep the simpler dtype-aware stack.

**Concurrent Blosc thread-count experiment -- UNREAD (2026-07-12).**
`numcodecs.blosc.get_nthreads()` defaults to 8 in every worker process; four
simultaneous shard writes can therefore launch 32 compression threads on the
8-physical-core production host, alongside Stockfish/selfplay. Hypothesis:
limiting each worker's zstd2 write to 1/2/4 threads improves aggregate four-
worker completion time by reducing oversubscription. ONE deciding yardstick:
`PYTHONPATH=. TMPDIR=/tmp taskset -c 0,2,4,6,8,10,12,14 python3
scripts/bench_concurrent_shard_writers.py --positions 500 --workers 4 --rounds
7`, alternating Blosc thread counts 1/2/4/8 on identical arrays and eight
physical cores. SUCCESS: a lower count median aggregate wall <=0.85x 8-thread
baseline, exact stable decoded hashes and sizes; choose fastest qualifier. KILL:
all lower counts >0.95x; otherwise MIXED. A production change must scope/restore
the process-global Blosc setting around the single serialized writer and pass
shard/worker tests. No data semantics/live activation.
**VERDICT: MIXED; no production change.** Four-worker medians relative to the
8-thread default were: 1 thread 0.935x, 2 threads 1.021x, and 4 threads 0.913x.
All outputs had identical 2,213,225-byte size and exact stable hash
`bd974f1197c70c97`. Lower thread counts reduce oversubscription slightly, but
none clears the 0.85 success gate; a scoped process-global Blosc mutation is
not justified for an 8.7% noisy scheduling point gain. Retain the library
default and the independently successful zstd2 change. The one-off benchmark
was removed rather than shipped as maintenance surface.

**Dense x_lc0_root shard-stack experiment -- UNREAD (2026-07-12).** Production
rows carry the alternate LC0-root planes densely, but `samples_to_arrays`
currently zero-allocates the optional array and casts/copies 500 rows through
the generic Python loop. Hypothesis: when every row carries `x_lc0_root`, one
dtype-aware stack plus filled presence flags is faster and exact, while the
existing generic path remains for mixed/missing shards. ONE deciding yardstick:
paired real-pipeline runs of `PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_worker_shard_pipeline.py --positions 500 --rounds 7`. SUCCESS:
conversion <=0.90x and total materialization <=0.97x baseline, exact stable
full hash/tar size, dense and mixed optional-field tests pass. KILL: conversion
>0.98x or parity failure; otherwise MIXED. No semantic/live change.
**VERDICT: FAILED and reverted.** Paired production-pipeline reads measured
baseline/candidate conversion 132.000/148.444ms (1.125x) and total
materialization 336.195/360.219ms (1.071x), over the kill threshold, while the
exact hash `4f4eb31bc2e62c50` and 1,679,360-byte tar stayed unchanged. The
preallocated row-assignment path is faster than another dense stack for this
optional field; retain it.

**Worker duplicate-value-validation experiment -- UNREAD (2026-07-12).** The
worker's trusted `ReplaySample` conversion scans every dense array for finite,
range, and distribution validity before local zarr write; the server then
eagerly reloads the upload and executes the same `validate_arrays` before
acceptance. Hypothesis: retain structural/declaration validation locally but
defer full value scans to the authoritative server for worker-generated shards,
removing duplicate CPU without weakening the training trust boundary. All
other `save_local_shard_arrays` callers retain full validation by default. ONE
deciding yardstick: paired `PYTHONPATH=. TMPDIR=/tmp taskset -c 15 python3
scripts/bench_worker_shard_pipeline.py --positions 500 --rounds 7`. SUCCESS:
total materialization <=0.90x baseline, exact hash/tar size, a deliberately
NaN worker shard is writable locally only through the explicit option and is
rejected/quarantined by the server, default writer still rejects it, and
worker/server/shard tests pass. KILL: >0.98x or any validation-boundary failure;
otherwise MIXED. No accepted data semantics/live change.
**VERDICT: MIXED; reverted.** The paired pipeline measured baseline/candidate
total materialization 370.326/339.809ms (0.918x), short of the 0.90 success
gate, while the directly affected zarr stage moved only 209.067/203.215ms
(0.972x). Conversion noise supplied most of the apparent total gain. Exact
hash `4f4eb31bc2e62c50` and 1,679,360-byte tar matched, but a 2.8% stage gain
does not justify weakening early local fault detection or adding a validation
mode. Keep full worker and server validation.

**RESTART 2026-07-13 ~11:00 (batch activation + stall recovery).** Trainer was
STALLED since 03:19 (inference broker wedged, GPU 0%, 111 worker timeouts,
iter 49 never finished — 7.5h lost; the #156 watchdog would have caught it,
NOT YET ON CRON — action item). Restarted onto origin/main, activating the
~15 speedup PRs merged by the other agent (#165-#179: Aurora CUDA-graph
replay + finite-check coalescing, SF pool head-of-line fix, ready-only future
collection, pending-label/selfplay overlap, compact broker metadata reuse,
threaded legal-array aliasing, native FEN-seed timeout fix, etc.) plus the
#157 hardening bundle. CONFOUND: many-changes-at-once restart — treat games/h
and iter-time shifts as a BUNDLE readout vs the iters 42-47 baseline
(~1300 g/h fresh @ regret 0.098-0.108, 2360-2420s/iter); no single-PR
attribution. Seeds verified flowing pre-restart (66 fenlist games/15 shards,
stm_l dominant). PID state: regret was RE-TIGHTENING (0.175 peak -> 0.098)
with EMA 0.60-0.64 — the post-airbag recovery watch item resolved HEALTHY.

**Stockfish pool engine-owning work-stealing experiment -- WORKED
(2026-07-15).** The per-engine executors from PR #177 prevent head-of-line lock
contention but permanently bind round-robin requests to queues. Production
mixes roughly quarter-budget fast-ply searches with full-budget labels, so an
engine can idle while another owns queued long work. Hypothesis: a shared FIFO
whose executor threads each permanently own one engine dynamically balances
actual completion times without engine locks, node estimates, callbacks, or
per-engine executors. ONE deciding yardstick: `PYTHONPATH=. taskset -c 14
python3 scripts/bench_stockfish_pool_work_stealing.py --workers 4 --requests 96
--rounds 9 --seed 20260715`. SUCCESS: work-stealing median makespan <=0.95x
fixed queues, no round regresses >1.05x, exact checksum, deterministic tests
prove a freed engine drains globally queued work while another remains blocked,
and focused Stockfish/selfplay tests plus lint pass. KILL: median >0.99x, tail
guard failure, engine contention, lifecycle regression, or materially more
complex code; otherwise MIXED. Offline scheduler mechanism only; no live/config
activation until a natural restart and no model/replay/training change.
**MECHANISM VERDICT: WORKED.** Nine alternating mixed-duration rounds measured
0.058687067s fixed-queue median versus 0.051551582s work-stealing median, a
**0.878415 ratio (12.2% faster)**. The worst individual round was 1.027657x,
inside the 1.05 tail guard, with exact checksum 77,823. The runtime replaces
four fixed executors plus round-robin binding with one shared FIFO/executor;
each executor thread claims exactly one engine at initialization, so no UCI
lock contention or node-cost accounting is introduced. A real four-engine
Stockfish smoke completed 24 mixed-node MultiPV requests without a zero move;
74 focused pool/timeout/nice/continuous-label/threaded/sparse-label/e2e tests
passed under a native+LTO build, and ruff/basedpyright/vulture were clean.
Activate only on a natural restart and judge whole throughput from
`sf_block_starved` plus games/h.

**Dtype-preserving replay policy mirror experiment -- UNREAD (2026-07-15).**
Array-backed replay augmentation copies each selected float16 policy field,
calls the general `mirror_policy_batch` API (which widens to float32 by
contract), then immediately casts it back to the source dtype for assignment.
Mirroring is only a column permutation. Hypothesis: use the width-specific
mirror index directly in this storage-preserving caller, removing both casts
and the float32 temporary with identical arrays. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_replay_policy_mirror.py
--rows 512 --width 1858 --fields 6 --iterations 100 --rounds 9`. SUCCESS:
candidate/reference median <=0.85x, exact hashes for float16/float32 and compact/
full widths, focused replay augmentation/trainer tests and lint pass. KILL:
ratio >1.00 or any dtype/value mismatch; otherwise MIXED and retain only if the
runtime is a net simplification. Replay augmentation only; no RNG mask, target,
model, optimizer, storage, config, or live-state change.
**VERDICT: WORKED.** Nine alternating production-shaped rounds measured
4.191702544s for the float32-widening reference versus 1.282233935s for the
dtype-preserving permutation, ratio **0.305898 (69.4% faster policy-field
mirroring)**, with exact checksum
`591ce45a2b0a501db1bc17595764cdf7c411b91ec50b8d3a662f0860258546e7`.
The hot loop is also simpler: one width-specific column gather replaces a
general float32 conversion followed by a source-dtype cast. Exact float16 and
float32 parity/dtype tests cover both compact 1858 and full 4672 policy spaces;
18 focused mirror, sparse-MultiPV, and relation tests pass after a clean
native+LTO build, and lint is clean. Keep the dtype-preserving permutation.
Activation is code-only on the next natural restart; whole training impact is
bounded by host prefetch overlap and should be read from `train_time_s`.

**Replay input-mirror redundant-copy experiment -- UNREAD (2026-07-15).**
`_mirror_x_batch` selects mirrored rows with boolean advanced indexing, which
already returns an independent array, then calls `.copy()` on that temporary
before assigning it back. Hypothesis: remove the second full selected-row copy
with identical positive-stride output and castling-plane swaps. ONE deciding
yardstick: `PYTHONPATH=. taskset -c 15 python3 scripts/bench_replay_x_mirror.py
--rows 512 --planes 175 --iterations 500 --rounds 9`. SUCCESS: candidate/
reference median <=0.90x, exact float16/float32 hashes, C-contiguous positive-
stride output, focused mirror/trainer tests and lint pass. KILL: ratio >1.00 or
any value/layout mismatch; otherwise MIXED and retain because the runtime is a
strict simplification. Replay augmentation only; no RNG, target, model,
optimizer, storage, config, or live-state change.
**VERDICT: MIXED; retained as a strict simplification.** Nine alternating
production-shaped rounds measured 3.810201700s reference versus 3.536325139s
without the second selected-row copy, ratio **0.928120 (7.2% faster)**, with
exact checksum
`91643cb1d5f60e7cfb638f18a3d7cb359dea804bbbf1270132d8219c75f0af62`.
This misses the aggressive 0.90 success bar but is well below the kill bar and
the shipped change deletes one redundant `.copy()` with no replacement branch.
Float16 benchmark output and float32 focused tests are exact, C-contiguous, and
positive-stride. Retain the simplification alongside the policy-field mirror
win above; no separate activation or configuration is required.

**Cross-optimizer-step training prefetch experiment -- UNREAD (2026-07-15).**
Production enables batch prefetch but uses `accum_steps: 1`; `_run_optimizer_step`
therefore invokes `_iter_prefetched_batches(count=1)`, whose explicit fast path
is synchronous, and destroys the iterator at every step. Hypothesis: keep one
phase-scoped iterator across all optimizer steps so replay sampling, target
rebuild, and mirroring overlap the previous GPU step while preserving the
three-attempt CUDA retry budget. ONE deciding yardstick: `PYTHONPATH=. taskset
-c 15 python3 scripts/bench_cross_step_prefetch.py --steps 100 --sample-ms 2
--compute-ms 8 --rounds 7`. SUCCESS: phase-scoped/reference median <=0.90x,
exact batch order/checksum, tests prove prefetch crosses an `accum_steps=1`
optimizer boundary and closes its thread on success/error, focused trainer/
retry tests and lint pass. KILL: ratio >1.00, batch loss/duplication, retry
exhaustion regression, or thread leak; otherwise MIXED and retain only if the
lifecycle stays localized to `train_steps`. Scheduling only; no RNG sequence,
batch contents, gradient, optimizer, target, replay, config, or live-state
meaning change.
**VERDICT: WORKED.** Seven alternating pinned-CPU rounds measured
1.134129402s with per-step synchronous sampling versus 0.897334813s with one
phase-scoped prefetch iterator, ratio **0.791210 (20.9% faster)**, with exact
batch-order checksum
`51f2696d83d68db933cbe2ba69586bfc8d17321ab3bac398a3c536fb4699ae29`.
The runtime allocates exactly `steps * accum_steps` batches for the normal
phase and extends that budget only by the number consumed by a failed CUDA
attempt, so retries neither duplicate nor lose successful microbatches and no
unused final batch advances replay RNG state. Tests prove the second batch is
already sampling across an `accum_steps=1` optimizer boundary, the executor
thread closes after success, retry order/budget is exact, and terminal errors
close the iterator. Five focused iterator/retry/scheduler tests passed; a wider
trainer/e2e/SWA/warmup/dropout run passed 44 tests before the known WSL `dxg`
kernel wait, and ruff/basedpyright/vulture are clean. Keep the localized
`train_steps` lifecycle. Production activation is natural-restart-only; read
whole-step `train_time_s` after restart because the synthetic 20.9% is the
host-preparation overlap ceiling, not an end-to-end GPU throughput claim.
**Replay-to-device deferred float16 cast experiment -- UNREAD (2026-07-15).**
Production replay stores the 175-plane input and four dense 1858-wide policy
fields as float16, but `collate_arrays` widens each one to float32 in NumPy
before pinning and copying it to the GPU. At batch 512 this materializes and
transfers roughly 34 MiB rather than 17 MiB for those five fields. Hypothesis:
preserve the stored float16 arrays through `torch.from_numpy`/pinning and ask
the device transfer to produce the same float32 tensors, eliminating the host
widening allocation and halving their H2D source bytes without changing any
consumer dtype or value. ONE deciding yardstick: `PYTHONPATH=. taskset -c 15
python3 scripts/bench_collate_deferred_cast.py --batch-size 512 --planes 175
--policy-size 1858 --policy-fields 4 --iterations 100 --rounds 9`. SUCCESS:
candidate/reference CPU preparation median <=0.80x, exact float32 tensor
equality/checksum, source-byte count <=0.51x, focused collation/loss/trainer
tests and lint pass. KILL: timing ratio >0.95, any value/dtype/contiguity
mismatch, or CUDA transfer requires a synchronous intermediate host widening;
otherwise MIXED and retain only if the helper/API change is a simplification.
Data movement only; no target, loss, gradient, model, replay storage, RNG,
config, or live-state meaning change. GPU end-to-end activation waits for a
natural restart and is judged from `train_time_s`; the CPU yardstick measures
the deterministic host-allocation mechanism, not GPU throughput.
**VERDICT: WORKED.** Nine alternating production-shaped rounds measured
5.143457092s for NumPy float32 widening plus the simulated pin copy versus
0.349399213s for copying the compact source, ratio **0.067931**, with exact
float32 checksum
`4bff3837d7ee0ba8998fe598b82c8e07b531c03a3b596f6a554d74da6ad79462`.
The five source arrays fell from 38,158,336 to 19,079,168 bytes, exact **0.50x**.
The runtime now preserves float16 only through `torch.from_numpy` and
`pin_memory`, while `_to_tensor(..., dtype=torch.float32)` keeps every model
and loss consumer unchanged. The installed PyTorch 2.10 CUDA `Copy.cu`
explicitly selects CUDA as the conversion device for nonblocking mixed-dtype
CPU/GPU copies, so this does not hide a synchronous host float32 intermediate;
it uses a temporary compact GPU source (about 18 MiB at the production batch),
which is negligible against training memory but should still be watched at the
first restart. Instrumented tests prove float16 reaches the transfer helper and
the resulting x/main-policy/SF-policy tensors are exact, contiguous float32;
all 12 collation tests pass and ruff/basedpyright/vulture are clean. Keep it.
End-to-end benefit remains restart-gated and should be judged from
`train_time_s`, with GPU memory as the guardrail.

**Replay-to-device compact integer/mask cast experiment -- UNREAD
(2026-07-15).** The same collation boundary widens three production
1858-wide legal masks from uint8 to float32 on the host, plus int8/int16 label
indices and many uint8 presence flags. The masks alone expand from about 2.7
MiB to 10.9 MiB per batch before pinning. Hypothesis: preserve any safely
widenable compact NumPy dtype through pinning and request the established
consumer dtype from the nonblocking device copy, extending the proven float16
mechanism without per-field branches. ONE deciding yardstick: `PYTHONPATH=.
taskset -c 15 python3 scripts/bench_collate_compact_masks.py --batch-size 512
--policy-size 1858 --mask-fields 3 --scalar-fields 20 --iterations 200
--rounds 9`. SUCCESS: candidate/reference host-preparation median <=0.80x,
exact float32 equality/checksum, source-byte ratio <=0.26x, focused tests cover
uint8->float32 and int8->int64 while float64/narrowing inputs still cast on the
host, and collation/loss tests plus lint pass. KILL: ratio >0.95, any dtype/
value/contiguity mismatch, or unsafe casts are deferred; otherwise MIXED and
retain only if one safe-cast helper replaces special cases. Data movement only;
no target, mask, loss, gradient, model, replay storage, RNG, config, or live
state change. It shares the float16 experiment's restart and GPU-memory guards.
**VERDICT: WORKED.** Nine alternating production-shaped rounds measured
3.170205812s for host float32 widening plus the simulated pin copy versus
0.062939495s for copying the compact sources, ratio **0.019853**, with exact
float32 checksum
`9b3255dae392e2c02a1f0dd52f80e4193a456c18faf4e92066360de446704c82`.
Pinned/H2D source bytes fell from 11,456,512 to 2,864,128, exact **0.25x**.
One `_transfer_array` helper now keeps a source compact only when NumPy marks
the conversion safe and its item size does not exceed the target; float64 to
float32 and int64 to int32 still convert on the host. This subsumes the earlier
float16 special case and also covers int8 WDL labels, int16 MultiPV rows, uint8
legal masks, and uint8 flags while preserving every consumer dtype. Thirteen
collation tests prove exact dtype/value/contiguity and safe/narrowing behavior;
ruff/basedpyright/vulture are clean. Keep it under the same natural-restart
activation and temporary-GPU-memory guard as the float16 transfer change.

**Inference-mode forward-context experiment -- UNREAD (2026-07-15).**
All eager, legal-policy, and AOT evaluator forwards run under `torch.no_grad`,
which disables gradient recording but retains view tracking and version-counter
updates. These outputs are inference-only and are detached/copied before they
leave evaluator ownership. Hypothesis: `torch.inference_mode` removes that
remaining autograd bookkeeping with exact outputs and no extra plumbing. ONE
deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_inference_context.py --batch-size 64 --width 64 --layers 12
--iterations 100 --rounds 9`. SUCCESS: inference-mode/no-grad median <=0.98x,
exact output checksum, focused direct/legal/AOT evaluator tests and lint pass.
KILL: ratio >1.00, any output/alias/lifetime failure, or compiled/evaluator
compatibility regression; otherwise MIXED and revert because a stricter tensor
mode is not a simplification by itself. Inference scheduling only; no model,
search, target, training, replay, config, RNG, or live-state change. Activation
would wait for a natural restart and whole match/selfplay throughput remains
the production outcome rather than the CPU context benchmark.
**VERDICT: MIXED; runtime unchanged.** Nine alternating pinned-CPU rounds
measured 0.468115401s under `no_grad` versus 0.460188073s under
`inference_mode`, ratio **0.983065 (1.7% faster)**, with exact checksum
`15741f00c9e3f948f4b514205a1b63377ed7023d210a3f3c157041880dd7c5fe`.
That narrowly misses the pre-committed 0.98 success gate. Inference mode is
stricter about tensor mutation/version behavior and does not simplify the
existing helper contract, while production compiled GPU forwards would likely
amortize even more of this CPU framework overhead. Keep `no_grad`; retain only
the reproducible benchmark and ledger result.

**Training loss-scalar synchronization coalescing -- UNREAD (2026-07-15).**
`_extract_loss_scalars` already stacks component losses so they cross the
CUDA/host boundary once, but both train and eval immediately call
`losses["total"].item()` as a second synchronization. Production performs
roughly 800 train steps per iteration. Hypothesis: include `total` in the same
stack/`tolist`, returning it as `loss`, so scalar reporting uses one host
materialization rather than two with identical values. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_loss_scalar_collection.py
--components 14 --iterations 100000 --rounds 9`. SUCCESS: coalesced/reference
CPU median <=1.02x, exact scalar/checksum parity, an instrumented unit test
proves one rather than two tensor materializations per batch, focused trainer
tests and lint pass. Otherwise FAILED and revert. The CPU timing is a
no-regression guard; the mechanism gain is removal of one mandatory CUDA host
synchronization per train/eval batch. Metrics only; no optimizer, gradient,
target, model, replay, config, or live-state change.
**VERDICT: WORKED.** Nine alternating pinned-CPU rounds measured 2.472764s
reference versus 2.473011s coalesced, ratio 1.000100, inside the 1.02 CPU
no-regression guard with exact checksum 860294.131562114. The runtime now
includes `total` in the existing stack/`tolist`, so every train/eval batch has
one tensor materialization rather than the prior component transfer plus
`total.item()`. The gradient-accumulation path preserves its exact divide-on-
device then rescale-on-host reporting order. The instrumented one-transfer
test and 31 focused loss/compute-loss tests pass; a wider trainer run passed
44 tests before the known WSL `dxg` kernel wait, and lint is clean. Keep the
coalescing. End-to-end GPU wall impact is restart-gated and should be read from
`train_time_s`; the deterministic mechanism removes about 800 host syncs per
production iteration.
**Compact native Gumbel root-state experiment -- UNREAD (2026-07-15).**
`start_gumbel_sims` currently allocates, zeros, and copies dense float64 prior
and Gumbel arrays of shape `(boards, 4672)` into C, but sequential halving reads
only the selected 2--32 candidates per board. At 384 roots the two C arrays are
27.4 MiB per search, excluding the Python inputs. Hypothesis: gather priors and
Gumbels into arrays parallel to `cands_flat`, keep them aligned during halving,
and eliminate the dense C state without changing the Python API or scores. ONE
deciding yardstick, run on the baseline and candidate native+LTO builds:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_gumbel_root_state.py --boards
384 --topk 16 --iterations 200 --rounds 9`. SUCCESS: candidate initialization
median <=0.50x baseline, native root-state bytes for priors+Gumbels fall >=99%,
exact remaining-candidate/checksum parity, randomized sequential-halving parity
and focused Gumbel/native/thread-safety tests plus lint pass. KILL: ratio >0.90,
any score/action/search-result mismatch, or more than one additional allocation;
otherwise MIXED and retain only if the state representation is simpler. Native
search bookkeeping only; no model, target, replay, config, or live-state change.
**VERDICT: WORKED.** On native+LTO builds, the exact registered baseline took
5.933741663s per 200 starts and the compact candidate took 0.031876409s, ratio
**0.005372 (186.1x faster initialization)**, with exact remaining-candidate
checksum `46a40aad285c222672b6d507c184985c3d26d8269d92537869d64c77f6c4be1b`.
At 384 roots/topk 16, prior+Gumbel C state falls from 28,704,768 to 98,304
bytes, a 99.6575% reduction; allocation count is unchanged. The representation
now gathers the dense API inputs once into arrays parallel to `cands_flat`, and
the halving sort moves actions, priors, and Gumbels together. A randomized
eight-seed alignment test with distinct scores selects the exact analytical
winner after all halving rounds. A clean native+LTO build, 142 focused Gumbel,
thread-safety, native-API, CBoard/history tests, and lint all pass. Keep the
compact state. Whole selfplay impact starts after a natural rebuild/restart and
is best read from MCTS init diagnostics and games/h.

**Candidate-aligned Python-to-native Gumbel input experiment -- UNREAD
(2026-07-15).** After compacting native root state, `gumbel_c.py` still creates
and zero-fills a 4672-wide float64 Gumbel vector per active board solely for
the legacy native input shape; the C entry then gathers the selected candidates
again. Hypothesis: allow each Gumbel input to be either the legacy dense vector
or a candidate-aligned vector, and have production pass `g[top_idx]`. ONE
deciding yardstick: `PYTHONPATH=. taskset -c 15 python3
scripts/bench_gumbel_input_packing.py --boards 384 --legal 30 --topk 16
--iterations 2000 --rounds 9`. SUCCESS: aligned/reference median <=0.20x,
exact candidate-value/checksum parity, dense and compact native inputs select
identical actions/probabilities across randomized halving tests, focused
Gumbel/native tests and lint pass. KILL: ratio >0.80, compatibility failure, or
more than one shape branch in native ingest; otherwise MIXED and retain only
if code is net simpler. Search-noise representation only; no RNG draw, score,
model, target, replay, config, or live-state change.
**VERDICT: MIXED; no runtime change.** Nine alternating rounds measured
7.648399s dense versus 2.545394s candidate-aligned packing, ratio 0.332801,
with exact checksum
`77396b7e9eb7cdf152e235031d72a77cc0ad3a781f9a3fca8490d33cb895f279`.
That is a real 3.0x packing reduction but misses the pre-committed 0.20 success
gate and saves under 1ms per 384-board pack in this mechanism. Accepting both
legacy dense and new aligned arrays would add a native shape branch, so it is
not the required net simplification. Keep the dense Python input API; the
successful compact C state above already removes the dominant 27.4 MiB native
copy. The one-off benchmark was removed.

**Background SF-starvation polling experiment -- WORKED (2026-07-15).** Live
post-restart telemetry shows states with zero runnable games spending roughly
850-970% cumulative thread time in `sf_block_starved`; the 16 Stockfish
processes plus Python keep the host 96-99% busy. Every one of 32 selfplay states
per worker currently wakes every 50ms to rescan futures/boards, although only
thread 0 performs the model/pause control poll. Hypothesis: retain the 50ms wait
for the control state and use 250ms for the other 31 states; FIRST_COMPLETED
still wakes immediately on useful SF work, while idle Python wakeups stop
stealing CPU from nice-19 Stockfish. ONE deciding yardstick: `PYTHONPATH=.
taskset -c 14 python3 scripts/bench_sf_starvation_polling.py --threads 32
--delay 0.5 --rounds 7`. SUCCESS: candidate/reference median idle wakeups
<=0.35, median completion-notification latency is no more than 10ms worse,
deterministic tests prove control/background timeout selection, and focused
continuous/label/threaded/e2e tests plus lint pass. KILL: wakeup ratio >0.60,
latency guard failure, or stop/pause teardown latency >300ms; otherwise MIXED.
Scheduler-only offline mechanism; no live/config/data/model/replay change and
activation waits for a natural restart.
**MECHANISM VERDICT: WORKED.** Seven alternating 32-state rounds measured 320
median idle wakeups with 50ms polling everywhere versus 72 when 31 background
states use 250ms, a **0.225 ratio (77.5% fewer wakeups)**. Median completion
notification moved from 1.369ms to 2.653ms, only +1.283ms and well inside the
10ms guard because FIRST_COMPLETED wakes immediately. The background timeout
test measured the no-completion bound below 300ms; 66 focused label/continuous/
distributed/CPU-e2e tests passed under a native+LTO build (three CUDA-only
threaded tests skipped as expected), and ruff/basedpyright/vulture were clean.
Production activation remains natural-restart-only, with host CPU share,
`sf_block_starved`, and games/h as the outcome read.
**Distributed Stockfish hash-size throughput experiment -- MIXED
(2026-07-15).** Production labels use about 700k nodes, MultiPV 40, and a 16 MB
transposition table per single-thread engine. The configured hash size was not
previously delivered to distributed workers; that plumbing is fixed separately
without changing the current value. Hypothesis: 32/64/128 MB reduces TT
collisions enough to improve fixed-node label wall time despite the larger CPU
cache footprint. ONE deciding yardstick: `PYTHONPATH=. nice -n 19 taskset -c 15
python3 scripts/bench_stockfish_hash.py --stockfish
/home/josh/projects/chess/e2e_server/publish/stockfish --nodes 700000 --multipv
40 --positions 16 --rounds 7`. Arms rotate order and each starts a fresh engine;
all searches within an arm retain the TT like production. SUCCESS: one larger
hash has median wall <=0.97x the 16 MB baseline and first-round best-move
agreement >=0.90; choose the fastest qualifier. KILL: every larger hash is
>0.99x or best-move agreement is <0.90; otherwise MIXED and do not change the
production value without a quiet-host replication. CPU-only offline mechanism;
no live YAML, worker, model, replay, or training state changes.
**VERDICT: MIXED; retain 16 MB pending a quiet-host replication.** Seven rotated
production-shaped rounds measured median ratios versus 16 MB of 1.0272 (8 MB),
0.9766 (32 MB), 1.0746 (64 MB), and 1.1148 (128 MB). The 32 MB arm had 93.75%
first-round best-move agreement and was the only larger qualifier on agreement,
but its 2.34% wall gain missed the pre-committed >=3% success bar; 64/128 MB
were clearly slower. Live training saturated 96-99% of the host during this
low-priority single-CPU sweep, so do not promote the near-threshold 32 MB point
without the exact quiet-host rerun. No config or runtime value changed.
**Persistent match CBoard experiment -- UNREAD (2026-07-15).** Match play
currently reconstructs every Gumbel root `CBoard` from its `python-chess`
board on every ply even though the C search accepts caller-owned root CBoards.
Hypothesis: create one CBoard per game, advance it alongside the authoritative
Python board, and pass the matching subsets into Gumbel/PUCT, eliminating the
repeated history replay with little added code. ONE deciding yardstick:
`PYTHONPATH=. taskset -c 15 python3 scripts/bench_match_cboard_reuse.py
--boards 64 --plies 160 --rounds 9`. SUCCESS: persistent-board glue median
wall <=0.50x reconstruction, saves >=0.20 ms per 64-board ply, every per-ply
FEN/checksum matches, match dispatch tests prove the matching CBoard subset is
passed, and focused arena/history tests plus lint pass. KILL: ratio >0.90,
absolute saving <0.10 ms/ply, any state mismatch, or materially complicated
fallback synchronization; otherwise MIXED and retain only if the final change
is a clear simplification. This is match/evaluation only; no training data,
search semantics, live config, or restart.
**VERDICT: MIXED; reverted.** The implementation-faithful registered run
measured 1.311999s reconstruction versus 0.996754s persistent, ratio 0.759722,
with 1.970ms saved per 64-board ply and exact checksum
`a5b29041c65eb77483d7b624c34b6cf061cb45d84f167972b9c94fac584bd8cf`.
That is a real glue reduction but misses the 0.50 success gate. Safely keeping
the two representations synchronized also requires remapping the selected move
back to its actual policy index because `index_to_move` can silently return a
legal fallback. The dual-state plumbing is not a simplification, so retain the
single authoritative Python board and rebuild roots. The one-off benchmark and
runtime/test changes were removed.

**AOT broker RETRY + speedup-merge restart (2026-07-15, post driver 610.74 +
reboot).** The 07-15 AOT KILL was a bridge-wedge side effect, not a
quality/perf failure (packages ==eager, forward ~5-6% faster). User updated
the NVIDIA driver 595.97 -> 610.74 and rebooted; bridge believed stable.
This restart lands as ONE bundle: (a) origin/main sync (Codex speedups #142-
#192, all correctness-reviewed CORRECT by parallel agents 07-15), (b) AOT
re-enabled (`distributed_inference_aot_dir: data/aot_models_512`), (c) Aurora
trainer CUDA graphs active by default (#170, first trainer-side cudagraphs;
user explicitly opted in). Yardstick: watchdog/recover_stall cadence + games/h
vs ~641 baseline + train_time_s. KILL (pre-committed): clockwork ~100-min
wedge/recover loop returns -> blank the aot key + set
`aurora_cuda_graphs: false` + restart (weights/optimizer/replay untouched);
if wedges persist even then, the driver didn't fix the bridge — revert the
whole merge is NOT needed (speedups are wedge-neutral eager code). Confounds:
bundle readout by design (speed-only changes); model/data-affecting changes: none.

**PLANNED — AOT bucket-ladder v2 (draft 2026-07-16, execute at next natural pause AFTER the AOT-retry readout closes).**
Interim 07-16 read of the retry: AOT live broker +~24% pos/s at high-load
windows vs the 07-15 reduce-overhead sessions (13.0-13.4k -> 16.2-16.6k);
zero watchdog fires since restart. Per-phase broker timers (fwd/out/scatter)
are NOT comparable across the AOT (dense+gather) and legal-rows paths — judge
on pos/s + games/h only. Also: "AOT packages loaded" goes to log.info, not
broker.out — verify engagement via /proc/<broker-pid>/maps (.chess_bNNN).
Plan (one restart bundle, perf-only): (1) build ~3 gap buckets in the
1190-4096 range — placed by the CURRENT hist these are b2720/b1792/b2336
(padded/actual on >=512 totals 1.234 -> 1.105; a full ladder doubling only
reaches 1.099, not worth 15 packages); (2) also build b64 and A/B it on the
quiet GPU (`scratchpad/aot_ab.py`) — deploy ONLY if it beats b128 for the
sub-64 call mass (90.9% of calls pad to 128 but are latency-bound; a busy-GPU
probe 07-16 was too contaminated to decide); (3) restart with
`CAE_BUCKET_HIST=scratchpad/bucket_hist_v2.json` to re-verify the batch-total
distribution post-speedup-merge — if the fresh hist moves the gap-bucket
placement materially, rebuild before deploying. Build with
`scripts/build_aot_packages.py` on the PAUSED GPU only (max-autotune compile
was the original bridge-hang trigger). Yardstick: pos/s at matched avg-batch
windows + games/h vs the post-retry baseline. KILL: any wedge recurrence
attributable to the larger package set -> drop back to the 14-bucket ladder.
Expected effect small (~2-4% games/h); this is opportunistic, not a bet.
Related knob NOT bundled (separate entry if pursued): batch coalescing toward
fuller forwards (broker --batch-wait-ms / adaptive-idle) — trades walker
latency for batch size; needs its own readout.
