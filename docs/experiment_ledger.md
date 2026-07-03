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
| Value ranking | `PYTHONPATH=. python3 scripts/value_regret.py --checkpoint <ckpt> --max-positions 2000 --batch-size 128 --gpu-mem-fraction 0.15` — 1-ply deep-SF regret. NOT Brier/ECE (fooled by calibration) | 72.4 cp @ckpt457; BT4 reference **43.0** (2026-07-02) |
| Blind-spot panel | `PYTHONPATH=. python3 scripts/blindspot_panel.py --checkpoint <ckpt>` — frozen 35 Cheese-loss collapse positions (`data/blindspot_panel_v1.jsonl`); counts positions the net reads "fine" where deep SF says lost | 21/35 blind @ckpt477-478 |
| Real-game strength | `PYTHONPATH=. python3 scripts/arena_standard.py` paired openings, or external engine match | vs full Cheese: 0 wins / 35 real losses (Jun 21, ckpt150-era) |
| Progress-over-time | low-sim cross-checkpoint arena (current vs pinned older ckpt) | not yet cadenced — see Queue |

Copy checkpoints OUT of the tune dir before long audits (Ray prunes live
checkpoints). All three GPU yardsticks are safe concurrent with training at the
listed batch / mem-fraction settings.

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
| 2026-07-01→02 | throughput triple, leg 2: `record_fast_ply_value` (75% value-only rows) | **REVERTED 2026-07-02** (bundle restart, protected by ckpt479 pool). Evidence: raw policy top-1 stuck +10cp over baseline through the whole window (trunk dilution — value-only rows crowd the policy gradient) and value 72.4→76.6. Leg 3 (`train_views_per_position 2.5`) KEPT — proportional step budgeting, not implicated. Net throughput-triple verdict: quality-negative; only the views bookkeeping survives |
| 2026-07-01→02 | flat lower LR (0.0003 sqrt_release → 0.0001 flat, PB2-pinned) | **UNREADABLE — reverted 2026-07-02.** Its watch criterion ("must not slow the policy downtrend") was made unjudgeable by the throughput triple landing the same day. Reverted to 0.0003/sqrt_release in the same bundle restart so the recovery read targets the exact known-good recipe. Lesson: simultaneous launches destroy both readouts. (Mechanic for changing PB2-pinned params: see memory `flat_lower_lr_experiment_live` — yaml alone silently no-ops) |

## LIVE, UNREAD (as of 2026-07-02 evening — every open loop)

| change | live since | readout & rule |
|---|---|---|
| **Return-to-known-good bundle restart** (fast-ply revert + LR revert + #104 code, knobs off) | 2026-07-02 evening | **THE active readout**: over the next window refill (~1-1.5 days), policy net+search / raw top-1 vs **49.6 / 51.5** and value vs **76.6→toward 72.4** (protocol: `--max-positions 2000`). Recovery ⇒ throughput-era damage was config, not permanent; no recovery ⇒ window legacy or trunk state → consider salvage-restart from a clean pool. LR revert applied via the PBT-pinned mechanic (above) and VERIFIED: peak_lr=0.0003 on iter 482. Confound: rung-1 fracs (below) remain live by design |
| Value-blend rung 1: `search_wdl_frac` 0.35→0.20, `sf_wdl_frac_floor` 0.35→0.45 | iter 477 (07-02) | value_regret per ckpt vs the **76.6 pre-anchor** (ckpt478), judged at the FULL-refill read (post-bundle window). Pre-committed rule: SUCCESS = ≤70.4 (2cp below the 72.4 known-good — evidence rung 1 adds value beyond the bundle recovery); KEEP-BUT-UNREAD = 70.4–75.0 (recovered; bundle confound absorbs credit, rung 1 stays as principled default); KILL = >75.0 at full refill → revert pair 0.35/0.35 (live keys) |
| PID `sf_pid_regret_tighten_streak_gain` 0.5→1.0 (temporary) | iter ~478 (07-02) | controller mode, not an experiment: restore 0.5 when EMA winrate ≈0.52. Expect the winrate sample to shift again post-uncap (congestion bias) |
| PR #104 gap-priority sampling — **MERGED, knobs default-off** | code live at bundle restart | activation = yaml `replay_sf_gap_priority_weight: 30` (live-tunable, no restart) AFTER the bundle recovery read banks. Pre-committed rule over one window refill from activation: SUCCESS = blind-spot panel ≤ 16/35 (≥5 positions un-blinded); KILL = panel ≥ 20/35 → set weight back to 0; GUARDRAIL = mean value_regret must not degrade >2cp vs its pre-activation read, else kill regardless of panel |
| sf_p0 + regret teacher weights | June | proven (see WORKED); leave alone |

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
2. Next full-refill read (~1-2 days): the bundle recovery read — policy vs
   49.6/51.5, value vs 76.6 (rung-1 overlay), blind-spot panel alongside.
3. Activate the gap boost (`replay_sf_gap_priority_weight: 30`, live yaml edit)
   once the recovery read banks. Judge by panel + tail, not the mean.
4. Build: tail metrics in value_regret; then the SF-frac ramp PR (per-row SF weight
   scaled by disagreement — premise validated 61%→70% monotone); then arena cadence
   + loop-health invariant monitor.
5. Restore `sf_pid_regret_tighten_streak_gain: 0.5` when EMA winrate ≈0.52.
6. Cheese rematch ONLY after the blind-spot panel moves (current net would still lose
   — same blunders, 21/35 blind).
