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
| **Return-to-known-good bundle restart** (fast-ply revert + LR revert + #104 code, knobs off) | 2026-07-02 evening | **THE active readout**: over the next window refill (~1-1.5 days), policy net+search / raw top-1 vs **49.6 / 51.5** and value vs **76.6→toward 72.4** (protocol: `--max-positions 2000`). Recovery ⇒ throughput-era damage was config, not permanent; no recovery ⇒ window legacy or trunk state → consider salvage-restart from a clean pool. LR revert applied via the PBT-pinned mechanic (above) and VERIFIED: peak_lr=0.0003 on iter 482. Confound: rung-1 fracs (below) remain live by design. **Contamination (07-02 ~21:17–23:5x, iters 484–486): the branch-switch incident (see gotchas) reverted the yaml — fast-ply rows back ON (~150k value-only rows in those shards), labels re-capped 400k, blend back to 0.35/0.35 — restored 23:09; fast-ply-off re-propagation VERIFIED (shards 23:57–00:02 back to 100% has_policy). Recovery-read clock effectively restarts at iter ~487; interpret ckpt510-era reads with this in the window.** **Pre-committed recovery rule (added before the readout):** each read dumps per-position and pairs vs the banked ckpt457 dumps (`scratchpad/policy_ci/`). RECOVERED = raw top-1 paired CI vs ckpt457 includes 0 AND endgame-value paired CI includes 0 → bank fresh baseline dumps, proceed to #104 activation. NOT RECOVERED = raw top-1 CI still excludes 0 on the worse side after TWO window refills → stop and rebase: hand-build a salvage pool from the banked ckpt457 trainer + the then-clean replay window (offline-recovery playbook, memory `v2_threats_12layer_live_offline_recovery`) and salvage-restart. Do NOT rebase onto `prechange_20260702_ckpt479` — that pool carries the damaged trunk. **REFILL CLOCK RECALIBRATED (07-03 AM, live telemetry): window = 1.41M rows (not the 3M cap — still growing), turnover ~16.1k rows/iter at ~39 min/iter → ONE refill ≈ 87 iters ≈ 2.4 days (the "~1-1.5 days" assumed fast-ply-era 4× ingest). The fast-ply era ingested ~1.3-1.45M rows ⇒ the window was ~fully contaminated at the iter-487 restart, and ~75% of those rows are value-only — policy training stays diluted until they flush. INTERIM READ @ckpt500 (iter ~501, ~16% refilled; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt500.jsonl`): raw top-1 vs ckpt457 −11.2 [−21.8, −1.7] (endgame −20.9 [−36.4, −7.7]; middlegame +8.0 [−0.6, +19.6] — borderline BETTER); value vs ckpt457 −9.5 [−17.9, −1.4] (endgame −11.5); search E[regret] vs ckpt457 −0.7 NS (search still masks the raw damage); vs ckpt478 everything NS (raw top-1 −0.3, value −2.4) — no movement yet, consistent with 16%. Blind-spot panel 19/35 vs 21/35 baseline. NOT RECOVERED, expected at this refill depth — next read ~50% refill (07-04 AM), full-refill verdict read ~07-05, two-refill rebase deadline ~07-07. **TIMELINE SUPERSEDED by the window cut (row below): contamination flush ≈ 36 iters ≈ 24h → verdict read ~07-04, rebase deadline ~07-05.** TAIL ADDENDUM (same dumps, `scratchpad/policy_ci/tail_stats.py`): the mean is tail-dominated (median value regret 9–11cp vs mean ~70–80); >300cp blowups 4.2%→5.1% (457→500); paired flips 457→500 = 48 new vs 29 fixed (net +19) — the loop mints new tail blowups faster than it fixes old ones. Add the tail line + flip counts to every future read (secondary, descriptive).** |
| Value-blend rung 1: `search_wdl_frac` 0.35→0.20, `sf_wdl_frac_floor` 0.35→0.45 | iter 477 (07-02) | value_regret per ckpt vs the **76.6 pre-anchor** (ckpt478), judged at the FULL-refill read (post-bundle window). Pre-committed rule: SUCCESS = ≤70.4 (2cp below the 72.4 known-good — evidence rung 1 adds value beyond the bundle recovery); KEEP-BUT-UNREAD = 70.4–75.0 (recovered; bundle confound absorbs credit, rung 1 stays as principled default); KILL = >75.0 at full refill → revert pair 0.35/0.35 (live keys). **AMENDED 2026-07-02 late, BEFORE the readout: the ±2–5cp point bands sit inside the value yardstick's measured ±8.7cp paired noise floor — verdict now runs on `paired_compare` vs the ckpt478 dump (`scratchpad/paired_ci_smoke/dump_ckpt478.jsonl`): KILL = CI excludes 0 on the worse side; SUCCESS = CI excludes 0 on the better side; otherwise KEEP-BUT-UNREAD. Point bands demoted to descriptive. Also: iters 484–486 trained on the old blend (incident above) — rung-1 exposure restarts ~487** |
| PID `sf_pid_regret_tighten_streak_gain` 0.5→1.0 (temporary) | iter ~478 (07-02) | controller mode, not an experiment: restore 0.5 when EMA winrate ≈0.52. Expect the winrate sample to shift again post-uncap (congestion bias). 07-03 AM: EMA 0.565 falling ~0.005/iter — restore due later today |
| **`replay_window_max` 3M→800k, deepened same morning to a 250k ONE-ITERATION cut** (TEMP, live yaml) | iter ~502-503 (07-03 AM, user-approved; deepened on the user's "start now" push) | Ops lever, not an experiment: shrink evicts oldest rows. At 800k the flush was still ~24h (580k contaminated rows retained); the 250k deep cut evicts ALL contaminated rows at the next iteration boundary (newest ~236k rows are post-incident clean) → training ~100% clean immediately. Sequence: 250k lands (watcher) → flip cap to 800k + `replay_window_growth_frac` 0.10→1.0 (clean regrowth ~16k/iter, ~34 iters back to 800k) → at recovery banking, cap→1.5M (revisit growth_frac then). Verified live-reloadable (`tc` rebuilt per iteration; shrink via `buf.enforce_window()`; replay keys absent from topology/construction-bound lists). Rationale for waiting on #104/rebalance anyway: activating a training-distribution change inside the healing window confounds BOTH the rebase gate and the knob's own verdict (false-credit risk) — with the deep cut the gate shrinks to ~this evening, so nothing meaningful is lost. Recovery verdict read: earliest meaningful ~18-25 clean-exposure iters (~this evening); rule-quality read 07-04 AM. Cost accepted: one day of small-window recency bias (safe: views-targeting fixes per-row reuse at 2.5 regardless of window size — window size only sets batch recency/diversity, and 250k ≈ salvage-restart scale); refill-based verdicts get less exposure per refill. Note: per-row reuse invariance means the cut costs training QUALITY little while maximizing clean-fraction — strictly better healing. **EXECUTED + VERIFIED (07-03): cut landed iter 504 (1.41M→248,410 rows, all clean); cap flipped back to 800k + growth_frac 1.0 same hour; regrowth confirmed iter 506+ (~13.5k rows/iter avg — selfplay-heavy iters ~15-16k alternate with curriculum-heavy ~5k); winrate EMA steady ~0.56 through the surgery. INTERIM READ @ckpt510 (6 clean iters post-cut; dumps `scratchpad/policy_ci/{dump,vdump}_ckpt510.jsonl`): BOTH pre-committed recovery criteria formally MET — raw top-1 vs ckpt457 −4.1 [−15.4, +8.0] includes 0 (was −11.2 SIG at ckpt500), endgame value vs ckpt457 −1.9 [−10.2, +6.1] includes 0 (was −20.9 SIG). Direct healing evidence vs ckpt500: endgame value +9.6 [+0.1, +19.1] SIGNIFICANT in 10 iters; value >300cp tail flips 500→510 = 22 new vs 43 FIXED (net −21) — first read where the loop fixes tail blowups faster than it mints them; vs ckpt457 value tail now breakeven (30/32, was 48/29). Panel v1 BLIND 18/35 (best yet: 21@477, 19@500); panel v2 55/113 (flat vs 54@500). Still short of full: endgame raw top-1 point −10.4 vs 457 (NS), policy tail net +8 vs 457. Trajectory steep (+6.8cp value in 10 iters) ⇒ NOT steady state — hold #104 for the confirming read @~iter 524 (watcher armed) per protocol; treat that as the banking read if it confirms** |
| Blind-spot panel v2 — **BUILT, FROZEN 07-03** (`data/blindspot_panel_v2.jsonl`, 113 rows) | 07-03 AM | Mined the June 15-16 iter520 full-Cheese + June-1 rofChade losses with the v1 method (300k-node SF, first decisive collapse, time-forfeits excluded; session `blindspot_mine_v2.py`): 86 losses profiled → 78 new rows + v1's 35, deduped to 113 (74 iter520-Cheese / 35 ckpt150-Cheese / 4 rofChade). **ckpt500 v2 baseline: BLIND 54/113 (47.8%), AWARE 43/113** — rate consistent with v1's 19/35; binomial CI half-width ~9% vs ~16%. v1 stays THE frozen panel for #104's pre-committed rule (≤16/35 / ≥20/35); v2 pre-committed secondary thresholds vs the 54/113 baseline: SUCCESS ≤ 44/113, KILL ≥ 60/113 (≈ ±2σ binomial). **LOSS-MODE ANALYSIS (user confound challenge answered — "is the newer tune flakier?" NO):** same-opponent cross-net comparison: iter520 (OLD run) vs Cheese = 100% decisive-collapse, 81% one-error dominance, 86% mate-scale — at least as collapse-y as ckpt150 (NEW run: 100%/57%/77%). g465 vs rofChade = 33% collapse, median ZERO ≥150cp drops (genuine grinds, median 32 plies). Collapse mode tracks OPPONENT (Cheese ~3000, balanced-until-blunder) not tune; grind is the expected mode vs the stronger rofChade (~3300). **Open cell: current net has NEVER played rofChade (only g465 did) — queue a ~20-game match at the next training pause (GPU-bound)** **SEARCH-vs-VALUE DECOMPOSITION @ckpt510 (07-03 evening; `scratchpad/policy_ci/blindspot_search_probe.py`, dump `searchprobe_v2_ckpt510_s200.jsonl`): at fen_before with production PLAY-shape Gumbel @200 sims, the net STILL picks the recorded losing move on 23/113 (search) vs 25/113 (raw argmax) — ~7× the ~3% random-match base rate, and search provides ZERO net rescue (balanced-before subset sf_before≥−50, n=37: raw 7/37, search 7/37). Value-awareness does NOT prevent the choice: AWARE rows get picked at the same ~20-24% rate as BLIND rows (specimen: d3d7, sf −11→−2992, net_after −0.49 AWARE, root_q −0.24, search plays it anyway) — Q-propagation/one-ply incoherence, not just after-position misread. Implications: (1) selfplay-sims search inherits the blindness → the loop keeps generating+reinforcing these lines (mechanism behind 'loop does not self-correct tail errors'); (2) #104 gap-priority targets the 56 value-BLIND rows but NOT the aware-but-picks mode — the improved-policy target itself is wrong there; the lever for that mode is blind-spot FEN seeding (games FROM fen_before so deep outcomes refute the move). Caveats: 200 sims ≪ match sims (match-level rescue untested — the queued rofChade/Cheese match is the test); panel move ≠ only losing move (rescue rates are upper bounds). Panel `move` field is SAN, not UCI — parse before comparing (first probe run scored 0/113 on exactly this). **SHAPE + SIMS SWEEP (same evening, logs `searchprobe_v2_ckpt510_{selfplay,play}_s*.log`): picks-losing-move = raw 25/113; selfplay shape @256 (c_scale 0.1 linear, topk 16) 19/113; play shape @256 (c_scale 0.025 root-log, topk 32) 22/113; play @3200 22/113 — 16× search rescues ZERO net positions. Composition flips with depth: @3200 BLIND-row picks RISE 7→11/56 while AWARE-row picks fall 14→9/46 — deep search converges to the value head's beliefs, fixing aware-mode errors and ENTRENCHING value-blind ones. The ~20% floor tracks the value head exactly ⇒ match-depth search is NOT protective; the play shape's lower value-trust is WORSE on AWARE rows @256 (14/46 vs raw 9/46 — down-weights the head where it was right). VERDICTS: (a) do NOT flip selfplay search to the play shape for tail reasons — no offline signal, and the tail is a data problem; (b) the blind-spot fix must come from training data (FEN seeding, #104), not search knobs; (c) play shape stays for matches (its +301 Elo is on-distribution). NEXT BUILD: FEN-seeding feature — `selfplay/opening.py` only supports startpos move-sequence books; needs a flag-gated FEN-list book type (defaults off) so curriculum-vs-SF games start from panel fen_befores and SF refutes the losing moves on the board** |
| Blind-spot FEN seeding — **BUILT (PR #108), flags default-off** | 07-03 evening | Starts selfplay/curriculum games FROM historically misplayed positions (the probe sweep's verdict: search can't rescue these at any depth — only data can). Feature: `opening_fen_list_path`/`opening_fen_prob`/`opening_fen_net_side_to_move` (FEN-list opening source, net forced to the blundering seat, full server→worker distribution mirroring the books, `fenlist` opening-source telemetry). **Seed asset = 78 FENs (panel v2 MINUS v1): panel v1's 35 rows are DELIBERATELY HELD OUT so v1 measures GENERALIZATION (transfer to unseen collapse positions) while seeded-subset v2 rows measure direct learning.** ACTIVATION (pre-committed, do NOT activate in the same window as #104): (1) requires restart onto merged code BEFORE adding yaml keys (strict validator); natural slot = the graceful restart after #104's window; (2) starting dose `opening_fen_prob: 0.02` (~[games/iter]×0.02 seeded games/iter; each of 78 positions revisited every few iters — temperature + SF-regret variety diversifies traces); (3) readouts after ~1 day: PRIMARY = held-out panel v1 BLIND (generalization; success = clear drop from its pre-activation count, kill = rise), SECONDARY = v2 seeded-subset BLIND (direct learning — should move first; if v2-seeded moves and v1 doesn't, the fix memorizes without generalizing → reconsider dose/diversity), GUARDRAIL = value_regret paired CI vs pre-activation ckpt (±2cp) + fenlist game outcomes in stats; (4) fresh-match confirmation (Cheese) only after panels move |
| PR #104 gap-priority sampling — **MERGED, knobs default-off** | code live at bundle restart | activation = yaml `replay_sf_gap_priority_weight: 30` (live-tunable, no restart) AFTER the bundle recovery read banks. Pre-committed rule over one window refill from activation: SUCCESS = blind-spot panel ≤ 16/35 (≥5 positions un-blinded); KILL = panel ≥ 20/35 → set weight back to 0; GUARDRAIL = mean value_regret must not degrade >2cp vs its pre-activation read, else kill regardless of panel. SECONDARY: top-decile-gap resolution rate must turn positive — measure with `PYTHONPATH=. python3 scripts/gap_resolution.py --checkpoint-old <pre> --checkpoint-new <post>` (repro'd the 466→478 finding: top decile +0.0006 vs bottom half +0.0077). **PRE-ACTIVATION DRY-RUN (07-03, 16.5k rows / 25 newest clean shards): recording works (`has_priority_sf_search_gap` on 97.7% of rows; gap mean 0.132, p90 0.32, p99 0.75 value-units). At w=30 the gap term is 31.7% of priority mass; sampling gets BROADER not narrower (ESS 0.502→0.608, top-1% mass 5.9→4.6% — the flat-ish gap term dilutes the heavy-tailed base surprise); corr(base, gap)=+0.07 → orthogonal targeting; top-decile-GAP rows' priority-half mass 12.4%→20.3% (~1.6×; overall ~1.35× after the uniform sampling half). Verdict: safe to activate, moderate re-targeting, no starvation. w=60 adds nothing (ESS/top-mass identical) — 30 is the right first rung** |
| sf_p0 + regret teacher weights | June | proven (see WORKED); leave alone |

## Analysis findings (offline, no live change)

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
**Queued rung — "gradient rebalance" (do NOT stack with #104's window):**
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
  self-correct off-distribution value errors).
- **Cadenced cross-checkpoint arena + loop-health invariant monitor**: nothing runs on
  a schedule today; every regression so far was found by ad-hoc suspicion.

## Decision queue (2026-07-02 evening, in order)

1. ~~Fast-ply revert~~ / ~~LR decision~~ / ~~#104 merge~~ — DONE via the bundle
   restart (see LIVE table).
1b. Verify the 23:09 config restore re-propagated (fresh shards back to
   has_policy_frac ≈ 1.0); merge PRs #105/#106/#107 so main matches the live
   config and the branch-switch trap is disarmed.
2. Next full-refill read (~1-2 days; clock restarted ~iter 487 by the incident):
   the bundle recovery read — audit + value runs WITH `--dump-per-position`,
   judged by paired CIs vs the banked ckpt457 dumps (recovery target) and
   ckpt478 dumps (progress since the kill), blind-spot panel alongside.
3. Activate the gap boost (`replay_sf_gap_priority_weight: 30`, live yaml edit)
   once the recovery read banks. Judge by panel + tail, not the mean.
3b. After (or instead of — user's call) the gap-boost window: the gradient
   rebalance rung (`w_soft` 1.0→0.5 + `w_wdl` 1.0→1.5, live-tunable; see
   Analysis findings). Both levers push value — never live in the same window.
4. Build: tail metrics in value_regret; then the SF-frac ramp PR (per-row SF weight
   scaled by disagreement — premise validated 61%→70% monotone); then arena cadence
   + loop-health invariant monitor.
5. Restore `sf_pid_regret_tighten_streak_gain: 0.5` when EMA winrate ≈0.52.
6. Cheese rematch ONLY after the blind-spot panel moves (current net would still lose
   — same blunders, 21/35 blind).
