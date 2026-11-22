# 2026-11-21: RL run 1 — sibling-ordering value screen (`--w-sibling`) on the pilot's collection (pre-registration)

**Status: registered, not launched.** It launches only after:
- review and merge of this record;
- the value-anchor screen (queue 2, `rl-run1-value-anchor/`) has finished. This screen
  queues AFTER it, and its Stage A needs that screen's instrument control.

It trains new cells on the pilot's frozen collection and the banked sibling-pair sidecar,
plays no new collection games, and changes nothing live. Outputs go to
`~/chess-artifacts/operations/rl-run1-sibling/` (`SB`); the banks it reads are read-only.

## Why

The owner asked for the value head to be trained on sibling ORDERING. The term now exists on
main (#1478, merge `febed2042`): `--w-sibling`, `--sibling-rows`, `--sibling-dir`, documented
in `docs/simple_rl_loop.md`, with the materialiser `scripts/simple_rl_sibling_pairs.py`.

**What is already measured** (`docs/experiments/2026-11-21-rl-loop-evidence-log.md`, rows
for the value diagnosis's addendum 1, the sibling scoping and the v2 sidecar; no training run
has used the term):

| Evidence | Number | Bank (sha256 prefix) |
|---|---|---|
| The 64-sim search orders clear sibling pairs better than the net (vs deep SF 8M, both children ≥ 2 visits, 14,304 pairs) | +0.0274 [+0.018, +0.037] | `RC/sibling-target-scoping/out/searched_order_vs_sf.json` [`5a267ec1`], README [`32a42f9e`] |
| Share of the searched sibling difference NOT explained by the net's own difference, at ≥ 2 / 4 / 8 visits | 20 / 29 / 35% | same README; `out/signal_pilot.json` [`faa1726e`] |
| Intra-game correlation (ICC by game): z − base / survivor_q − base / search_wdl − base / sibling residual | 0.62 / 0.013 / 0.028 / ≈ 0.001 | `out/icc_fixed_pov.json` [`e94618a4`] |
| The banked pilot sidecar | 1,490,177 pairs on 752,093 rows; 722,742 training rows, 713,407 with pairs; 1,024-pair hold-out probe | `RC/sibling-pairs-pilot/v2/gen_0001/sibling_manifest.json` [`bbd97a99`], README [`dd975485`], `v2/logs/trainer_load_check.txt` [`5006a7ce`]; `SIDECAR_VERSION` 2 |
| S's root value loss broke sibling ordering (clear pairs, vs base); Zq's did not | S −0.0096 [−0.0129, −0.0064]; Zq −0.0007 [−0.0032, +0.0017] | `RC/pilot-value-diagnosis/out/sibling_order.json` [`bb8e5b16`], README [`ae5e0866`] |
| Follow-up matches at the PIN, 6,400 games each | 5 (P0 single net vs base) +2.2 [−4.2, +8.6]; 7 (Zq + recal vs Zq) +21.4 [+15.1, +27.6]; 6b (S's value on the base policy vs base) −21.4 [−27.7, −15.2] | `rl-run1-followup/arenas/{p0_single,zq_recal_h2h,s_value_raw}/pool.json` [`22fc7f97`, `e1945cbd`, `3534eac9`] |

So the sibling target carries information the net lacks (the search is right more often on
clear pairs), its rows are nearly independent within a game (ICC ≈ 0.001), and the one value
recipe that damaged ordering (S) is also the one that lost most in play. Whether training on
the sibling difference helps the value in play is unmeasured. This screen measures it.

## What `--w-sibling` computes (read at main `febed2042`)

- **Loss** (`simple_rl/sibling.py`): `w · mean over pairs of (min(v_A, v_B) / 64) · Huber_0.2(Δv − dq)`.
  - Δv is the difference of the two children's values, each `L − W` of the child's own WDL
    softmax, both through the SAME net in the same step.
  - dq is the searched difference of the children's completed Q, both in the parent's POV.
- **Pairs:** the banked sidecar's (`--sibling-dir`), K 2 per row, children with ≥ 2 visits that
  the search did not score as terminal. In each batch, the first 128 pair-carrying rows present
  all their pairs (`--sibling-rows` 128, the default), so about 256 pairs per 512-row batch.
- **Hook.** The term rides the Trainer's single penalty hook (`attach_anchor`), as the KL
  anchor does. It is added to the training loss before `backward()`.
- **Hold-out.** Hold-out games' pairs are never trained (the learner refuses otherwise); the
  first 1,024 hold-out pairs are the probe.
- **Take-effect record:** `summary.sibling` (pairs, steps, per-step (pairs, loss), the probe's
  mean |Δv − dq| before and after) and the manifest input `sibling_gen_0001`.

## Code base: train on main, match at the PIN

The term exists only on main (`febed2042`), not at the pilot's PIN `0c795fb7a`. **Training runs
at main, pinned to `febed2042` in a detached worktree. Every match and every Stage A instrument
runs at the PIN**, as all pilot, follow-up and anchor-screen numbers did.

**Commits on main since the PIN that touch the trainer path** (`git log --no-merges
0c795fb7a..febed2042 -- chess_anti_engine/simple_rl chess_anti_engine/train
scripts/simple_rl_loop.py chess_anti_engine/model chess_anti_engine/uci
chess_anti_engine/inference.py`):

| Commit | What |
|---|---|
| `c661d2bb3`, `7f1a2f2c3`, `d4de13e0c`, `1deeb5afd`, `75e0fc39f`, `cd74e3d87` | simple_rl collect: game-granularity resume, VRAM record, journal, games.jsonl moves. Collect only; no training input changes |
| `84a47b785`, `8a97a23dd` | `--value-rows-per-game` (default 0 = every row, unchanged) |
| `fd9ddd17a`, `d18f82f3d`, `a0bc34416` | the sibling term (default 0 = off), its materialiser, docs and review fixes |

The diff touches `simple_rl/{collect,common,learn,sibling}.py`, `train/losses.py` (11 lines)
and `scripts/simple_rl_loop.py`. It touches no model, loader, encoder or `uci/` file. The one
encode-tripwire path it touches, `selfplay/network_turn.py`, changes only the collect's per-game
RNG plumbing, not `_batch_encoder_pair`.

**The pilot's guards, restated for PIN → main:**
- **(a) The decoded training input is identical.** The pilot launch record's guard (a) body,
  run at main with each pilot cell's own flags plus `--w-sibling 0`, must be byte-equal, key by
  key, to the banked `rl-run1-pilot/guard_a_{P0,S,Zq}_0c795fb7a.json`.
  **Run already (CPU, 2026-11-21) at `febed2042`:** **all three cells byte-equal** (P0, S and Zq; 722,742 rows each, every key and the row ids).
  Script and outputs: `SB/prelaunch/guard_a.py`, `run_guard_a.sh`, `guard_a_*_febed2042.{json,err}`,
  `guard_a_febed2042.verdict`.
- **(b) The encoder that wrote the stored `x` is identical.** It matters less here than at
  C0 → PIN: cells read the STORED `x`, and matches run at the PIN as before. It holds by
  construction, checked mechanically (run 2026-11-21, exit 0, and repeated at launch):
  `git diff --quiet 0c795fb7a febed2042 -- chess_anti_engine/encoding chess_anti_engine/mcts/_mcts_tree.c
  chess_anti_engine/moves chess_anti_engine/replay/shard.py chess_anti_engine/replay/packed_zarr.py
  chess_anti_engine/model chess_anti_engine/uci chess_anti_engine/inference.py`.
  `selfplay/network_turn.py`'s diff does not touch `_batch_encoder_pair`, and `x_hash` is
  unchanged. (The banked `guard_b_0c795fb7a_*.json` were made by an extended version of the
  launch record's guard (b) block whose script is not in the pilot bank, so it is not rerun
  here; the diff check covers every file it exercises.)
- **(c) Default off is off on the pilot's real inputs.** The code-base control cell `P0_main`
  (P0's flags at main, `--w-sibling 0`) is trained like every cell and must match the banked P0
  in Stage A within run-to-run noise (below). #1478's reviewer showed bit-identity of weights on
  a tiny phase; this is the same claim on the real phase, judged at the instruments.
- **(d) A main-written checkpoint loads and evaluates identically at the PIN.**
  `SB/prelaunch/check_load.py` loads `P0_main`'s `model.pt` once at the PIN worktree and once at
  the training worktree, through `uci.model_loader.load_model_from_checkpoint` (the loader the
  arena and every Stage A instrument use). It digests every state_dict tensor and forwards the
  first 512 hold-out rows of shard 0 on CPU in fp32. The two outputs must be bit-equal
  (`check_load.py --compare`). A failure stops the screen before Stage A. The same check was run
  on the SL base itself as the instrument's smoke: IDENTICAL for `sl_base.pt` at `0c795fb7a` vs `febed2042` (494 state_dict digests and both logit arrays); a one-ulp change to one WDL logit makes `--compare` print DIFFER and exit 1 (`SB/prelaunch/check_load_mutant.txt`).

## Strengths, derived from the measured step-0 gradients

**Measurement** (`SB/prelaunch/grad_scale.py` → `grad_scale.json`, CPU, fp32, at the SL base,
no optimizer step). A Trainer is built by `learn.build_trainer` from the base's arch, the base's
weights are loaded, and one `train_steps` call is made on a `ScheduledSampler` over pilot rows.
The penalty hook receives the trainer's real outputs and batch, takes the autograd gradient of
each term separately, then aborts the step. Rows: shard 0's 18,840 non-hold-out rows; pairs: the
v2 sidecar's shard 0. 4 batches of 512 rows.

| Pooled mean of 4 batches | Loss | ‖g‖ on `value_wdl` | ‖g‖ on the trunk |
|---|---|---|---|
| policy (w_policy 1) | 1.014 | 0 | 1.695 |
| S root WDL (w_wdl 1, search frac 1.0) | 0.632 | 0.739 | 0.739 |
| Zq root WDL (w_wdl 1, search frac 0.5) | 0.679 | 1.094 | 2.010 |
| sibling term at w = 1 (≈ 255 pairs from 128 rows) | 0.00143 | **0.00896** | **0.0246** |

Per batch, the sibling term's `value_wdl` norm ranged 0.0067–0.0126 (trunk 0.017–0.032), a factor
of about 1.9: the visit weight makes it heavy-tailed. Cosines with the root terms are small and
positive on the head (sibling vs S 0.29, vs Zq 0.19); on the trunk, sibling vs policy is 0.04.
Bank: `SB/prelaunch/grad_scale.json` (sha256 `782d52a2a358…`), script `grad_scale.py`
(`6a25772191f5…`).

**Reading.**
- The sibling term is small in absolute size: its mean weighted Huber is 0.00143, because
  the visit weight `min(v_A, v_B) / 64` averages well below 1 and most residuals sit in Huber's
  quadratic zone (mean |Δv − dq| 0.120). Its gradient norm at w = 1 is therefore a small
  fraction of the WDL CE's.
- Both optimizers here are scale-invariant per update (the anchor screen's derivation): the
  value head is in an AdamW group at 3e-6. So for parameters reached by ONE term, w barely
  changes the step. What w sets is the term's share of the gradient wherever terms mix: the
  value head in the S and Zq families, and the trunk in every family.
- **Strength rule:** w is set so that, at step 0, the sibling term contributes a share s of the
  combined gradient norm, s ∈ {0.25, 0.50, 0.75}:
  w = s / (1 − s) × ‖g_other‖ / ‖g_sib(w = 1)‖.
  - S and Zq families: g_other is the root WDL term's value-head gradient (the head is where
    the two value signals compete).
  - P family (`--w-wdl 0`): the value head gets only the sibling gradient, so its step is set
    by Adam whatever w is. w then sets the sibling share of the TRUNK gradient against the
    policy term's, so g_other is the policy term's trunk gradient.
  - The shares are the design parameter; the w values follow from the measurement. The shares
    are step-0 shares: as Δv approaches dq, the term's gradient shrinks.

| Cell | ‖g_other‖ / ‖g_sib‖ | s | `--w-sibling` | Step-0 share on `value_wdl` | Step-0 share on the trunk |
|---|---|---|---|---|---|
| `P_sib_w23` / `w69` / `w207` | 1.695 / 0.0246 = 69.0 (trunk, vs policy) | 0.25 / 0.50 / 0.75 | **23 / 69 / 207** | 1 (only term) | 0.25 / 0.50 / 0.75 |
| `Zq_sib_w41` / `w122` / `w366` | 1.094 / 0.00896 = 122.1 (head, vs Zq WDL) | 0.25 / 0.50 / 0.75 | **41 / 122 / 366** | 0.25 / 0.50 / 0.75 | 0.21 / 0.45 / 0.71 |
| `S_sib_w28` / `w83` / `w248` | 0.739 / 0.00896 = 82.5 (head, vs S WDL) | 0.25 / 0.50 / 0.75 | **28 / 83 / 248** | 0.25 / 0.50 / 0.75 | 0.22 / 0.45 / 0.71 |

Trunk shares are against the sum of the other terms' trunk norms (policy plus root WDL).
Values are rounded to 2–3 significant figures; the factor-1.9 batch spread means a nominal 0.50
share is 0.35–0.60 on a given batch, so the three strengths are distinct but not exact.

**No `--kl-wdl` cell here.** Whether an anchor helps, and at which λ, is exactly what the
value-anchor screen decides; a sibling + anchor cell would have to choose λ before that reading
exists (a forking path). A combination is registered later, under its own record, only if the
anchor screen nominates a λ.

## Cells

All cells train at main `febed2042` on the pilot's frozen collection (`--collect-from`, the
pilot's `COLLECT` and `--seed`) with the pilot's `$TRAIN` verbatim (it carries `--kl-wdl 0`).

| Cell | Value flags | Role |
|---|---|---|
| `P_sib_*` (3 strengths) | `--w-wdl 0 --w-sibling w --sibling-dir $SIDE` | (a) policy + sibling only: the root value gets no loss. The cleanest test of "does sibling training alone help, or at least not hurt, the value" |
| `Zq_sib_*` (3) | Zq's flags + `--w-sibling w --sibling-dir $SIDE` | (b) the nearest-to-adoptable root target plus the term |
| `S_sib_*` (3) | S's flags + `--w-sibling w --sibling-dir $SIDE` | (c) does the term repair the ordering S breaks? |
| `P0_main` | `--w-wdl 0` (term off) | code-base control (c) above; offline only |
| Controls, not retrained | the banked P0, S, Zq (`rl-run1-pilot/arm_*`) | references |

`$SIDE` = `~/chess-artifacts/operations/rl-claude-20261119/sibling-pairs-pilot/v2` (the
directory holding `gen_0001/`).

## Take-effect

`SB/stage_a/train_check.py` (one run, after every crash rerun). A field mismatch writes
`arm_<cell>/VOID`. For every cell:
- exit 0 and a sealed train manifest; init checkpoint sha256 = the SL base's (`a39f900c…`);
- `summary.realised_w_wdl` = 0 (P family, `P0_main`) or 1 (Zq, S families);
  `summary.realised_search_wdl_frac` = 0, 0.5 or 1.0 respectively;
- `summary.updates` = 1,412, `summary.drawn_rows` = 722,742, `presented.npz` sha256 =
  `63e227ac…` (the pilot's batches, proved at the consumer); no KL anchor.

For every sibling cell, additionally:
- `summary.sibling.w` = the cell's w, `sibling_rows` = 128, `steps` = `updates`, one per-step
  entry per update, the per-step pairs summing to `pairs` > 0;
- the sidecar record and the manifest input `sibling_gen_0001` both carry sha256 `bbd97a99…`
  (the banked v2 sidecar);
- probe of 1,024 pairs before and after. **A cell whose probe mean |Δv − dq| does not FALL is
  INERT** (`arm_<cell>/INERT`): scored in Stage A, never matched.

`P0_main` must carry no `summary.sibling` and no sidecar input.

**Crash rule.** A crashed cell is renamed `arm_<cell>.crash1` and rerun once, whole, before
`train_check.py` runs; only if no `.crash1` exists. A second crash is VOID.

## Deciding yardstick

### Stage A: offline veto and the targeted quantity, never a selector

**Instruments:** the value-anchor screen's banked adapted instruments, run by sha256 from
`rl-run1-value-anchor/stage_a/` at the PIN: `collect_logits.py` `956c238f88c7…`, `run_fits.sh`
`5017cea31e6e…`, `analyze_holdout.py` `3291449af64b…`, `sibling_order.py` `17eeb159f691…`,
`supp_outcome_distance.py` `2ad7f272d00a…`, and the unmodified banked `value_regret_recal.py`
`6cddd7d5afd1…`.
- **What changes for main-trained nets: nothing.** They take a net list and an output dir, and
  load each checkpoint with the PIN's loader, which guard (d) proves bit-faithful for a
  main-written file.
- **Instrument control:** the anchor screen's step 0 (`rl-run1-value-anchor/offline/control/CONTROL_PASS`)
  ran these exact files on the banked nets and reproduced the bank exactly. It is required, not
  rerun. If the anchor screen's control failed, this Stage A is void too.
- This screen's own files (`SB/stage_a/`: `cells.json`, `train_check.py`, `stage_a_decide.py`,
  `run_stage_a.sh`) are banked read-only with `SHA256SUMS`: `cells.json` `12be8a762466…`, `train_check.py` `1bb13026d34a…`, `stage_a_decide.py` `48408ecdc683…`, `run_stage_a.sh` `df63975bb1d5…` (and a fabricated-input test of the decision script in `fab_test/`). The pre-launch scripts are in `SB/prelaunch/` with their own `SHA256SUMS`: `grad_scale.py` `6a25772191f5…`, `cost_probe.py` `000fd1da8892…`, `check_load.py` `e8a870764bb5…`, `guard_a.py` `d247b0f0e665…`.

**Statistics** (raw heads; `stage_a_decide.py`):
- **Veto, exactly as the anchor screen:** VETOED iff the 95% CI upper bound of
  `sibling_order.json → paired_minus_base.all.clearplus_50plus.<cell>_raw` is below 0.
- **The primary offline quantity this term targets is the same statistic**: clear-pair
  sibling-ordering accuracy vs deep SF, cell minus base, game-clustered. For once a POSITIVE CI
  (lower bound > 0) is the interesting outcome, and it is reported as "improves ordering".
  - The ruler rows are the rl-audit collection's 2,433 leak-free rows, a different collection
    from the training games.
  - It still may NOT select between survivors. Three fit→Elo transfer failures are on record:
    v020 (+3.1 cp audit gain, −9.6 Elo), ft595 (−41.0 Elo for better fit), F-only (hold-out CE
    −0.043 nats for 0 cp). An offline gain on the term's own target is the "arm that is the
    gradient of the metric" case: it is expected, not evidence of strength.
- **INERT:** `train_check.py`'s marker (the probe did not fall).
- **Code-base control (`P0_main` vs banked P0).** On the clear-pair difference,
  `KL_base_net_raw` and `share_of_gap_closed`: |P0_main − P0| ≤ tol, where tol = √2 × the banked
  P0's 95% CI half-width for that statistic (a 95% bound on the gap between two exchangeable
  replicates if run-to-run noise is no larger than sampling noise). If the anchor screen's
  measured replicate gap |S_trunc_final − S| for that statistic is larger than half the bound,
  tol becomes 2 × that gap. **A failure means main trains differently from the PIN on these
  inputs: no Stage B, verdict "unread: code-base control failed".**
- **Report-only:** each cell's `KL_base_net_raw`, `share_of_gap_closed`, `KL_share_removed`
  by its own (T, b), FEN-only value regret, outcome distance, and the probe before/after.

### Stage B: matches

- **Selection, fixed now, not by offline score.** Families in this order: **Zq, P, S**. Per
  family, the smallest-w cell that is not VOID, VETOED or INERT. The first two families that
  have one are matched; at most two matches.
  - Why this order: Zq + sibling is the recipe nearest to adoption (Zq's root target kept
    ordering intact, and match 7 showed its loss is largely shape); P + sibling is the owner's
    question in its cleanest form; S + sibling is a repair test of a target that loses 20 Elo,
    so it has the least prospect of nominating a recipe.
  - Smallest w, as in the anchor screen: the least departure from the banked cell that
    survives.
- **Design.** Cell vs the SL base, plain seats, seat-balanced; 6,400 games (3,200 per run,
  1,600 openings); seeds **2026112193** (first selected cell) and **2026112194** (second), unused
  anywhere under `docs/` and in every bank; the follow-up launcher
  `rl-run1-followup/run_followup_match.sh` (sha256 `7c8fd065…`) at the PIN with output root
  `SB`, the pilot's `$ARENA` flags and the launch record's max-plies void rule; a 4-game smoke
  per seat first.
- **Rules, as match 5:** "damage" iff UB95 < 0; "adoptable" iff LB95 > −10; otherwise
  "unread". Match 5's tie-break is kept as written (it cannot trigger at the expected SE).
- **Power,** at the gate-scaled SE 3.37: P(adoptable | true 0) ≈ 0.84; family false-nomination
  ≈ 5% at the −10 boundary across two matches.
- **Report-only contrast** with the same-family banked cell:

  | Family | Banked result | SE of the difference | 80%-power MDE |
  |---|---|---|---|
  | Zq | Zq gate −10.0 (SE 6.74) | ≈ 7.5 | 21.1 Elo |
  | P | match 5, P0 single net +2.2 (SE 3.26) | ≈ 4.7 | 13.1 Elo |
  | S | S −20.0 (SE 8.5) | ≈ 9.1 | 25.6 Elo |

**What each outcome licenses:**

| Reading | Licenses |
|---|---|
| adoptable | NOMINATES that recipe (family + w) for a two-generation run with the policy, under its own record. Nothing is adopted here. For the Zq family the nomination record also carries a "Zq + (T, b)" alternative (match 7/8) |
| damage | The term at that strength does not make that value recipe safe |
| unread | Nothing |

**"Sibling training is not the fix"** for a family: every cell of it is vetoed or inert, or its
matched cell reads damage.

## Confounds

- **Same games, two targets.** The pairs come from the same games as the root rows. Hold-out
  here means hold-out GAMES (`is_holdout_game` at 0.05): the probe pairs and the Stage A
  hold-out rows are from those games, which no cell trains on. The sibling-ordering ruler rows
  (rl-audit, deep SF 8M) are a different collection.
- **Same search.** The sibling target is the same 64-sim search that S's root target came from;
  a cell that repairs S's ordering does it with the same teacher, not an independent one.
- **Policy-induced drift.** P0 showed the trunk moves the value with no value loss (KL 0.0072),
  and every cell trains the policy. The P family is compared with P0 (banked, and `P0_main`)
  for exactly that reason.
- **Code base.** Cells train at main, the banked controls trained at the PIN. Guards (a)–(d)
  are what licenses comparing them.
- **One data-affecting change** per cell versus its banked family cell: the term (and its
  sidecar); same rows, batches (`presented.npz`), seed and optimizer.

## Cost and feasibility

**Registered step 1, before any training: the cost probe** (`SB/prelaunch/cost_probe.py`, GPU,
alone on the card). Two runs of STEPS = 50 updates of the REAL trainer (settings from the
loop's own parser, Aurora resolved from the init checkpoint, the production `SiblingTerm` on
the banked pairs): the heaviest cell's flags (`Zq_sib`, largest w) and the same flags with
`--w-sibling 0`. It records the median step time over updates 11–50 and
`torch.cuda.max_memory_reserved`.
- **Abort the screen** if the sibling run's peak reserved memory exceeds **0.85 × the card's
  total memory** (headroom for allocator fragmentation and the CUDA context; the card runs
  nothing else during the screen). An abort is reported; no cell trains.
- **Hold for the owner** if the projected training total (below) exceeds 2 × the estimate.
- Projected wall per sibling cell = 1,412 × the measured sibling step time + about 150 s of
  sidecar sha verification.
- The probe script was smoke-run on CPU (`COST_PROBE_CPU_SMOKE=1`, 2 steps, Zq + w 366): both
  steps ran the production `SiblingTerm` (255 pairs each) and it exited 0
  (`SB/prelaunch/cost_probe_cpu_smoke.json`). That proves the logic, not the timing.

**Estimate until the probe reads** (the doc's "about double", plus 0.19 s host read per step):
pilot step ≈ 0.28 s (3.55 updates/s) → sibling step ≈ 0.75 s → ≈ 22 min per sibling cell.

| Item | GPU-h |
|---|---|
| Cost probe | ≈ 0.05 |
| Training: 9 sibling cells × ≈ 22 min + `P0_main` ≈ 7 min | ≈ 3.4 |
| Stage A: 10 nets × ≈ 6 min (no instrument control: reused) | ≈ 1.0 |
| Stage B: ≤ 2 × 1.52 | ≤ 3.0 |
| **Total** | **≤ ≈ 7.5** |

## Commands

**Step 0: preconditions and worktrees.**

```bash
SB=$HOME/chess-artifacts/operations/rl-run1-sibling; P=$HOME/chess-artifacts/operations/rl-run1-pilot
AV=$HOME/chess-artifacts/operations/rl-run1-value-anchor
PIN=$(cat $P/PIN.txt); PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
TSHA=febed2042b0859ea9216f796c5c4ace7790d1039; TW=$HOME/projects/chess-worktrees/rl-run1-sibling-febed2042
PY=$HOME/projects/chess/.venv/bin/python; SL=$P/sl_base.pt
SIDE=$HOME/chess-artifacts/operations/rl-claude-20261119/sibling-pairs-pilot/v2
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
SF=$HOME/local_stockfish_linux_latest/stockfish/stockfish-ubuntu-x86-64-bmi2
[ -e $AV/offline/cells/DONE ] && [ -e $AV/offline/control/CONTROL_PASS ] || exit 92   # anchor screen finished, instruments proven
(cd $SB/stage_a && sha256sum -c --quiet SHA256SUMS) && (cd $SB/prelaunch && sha256sum -c --quiet SHA256SUMS) || exit 93
[ -d $TW ] || git -C $PINW worktree add --detach $TW $TSHA             # a new detached worktree; no existing tree moves
cd $TW && [ "$(git rev-parse HEAD)" = "$TSHA" ] || exit 96
python3 scripts/build_production_extensions.py || exit 1
git diff --quiet 0c795fb7a $TSHA -- chess_anti_engine/encoding chess_anti_engine/mcts/_mcts_tree.c chess_anti_engine/moves \
  chess_anti_engine/replay/shard.py chess_anti_engine/replay/packed_zarr.py chess_anti_engine/model chess_anti_engine/uci \
  chess_anti_engine/inference.py || exit 1                             # guard (b)
grep -c EQUAL $SB/prelaunch/guard_a_febed2042.verdict | grep -qx 3 || exit 1   # guard (a), run 2026-11-21
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/test_simple_rl_sibling*.py \
  "tests/test_simple_rl_learn.py::test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer" || exit 1
```

**Cell flags** (all at `$TW`):

```bash
COLLECT="--openings-dir $P/openings --init-checkpoint $SL --device cuda --seed 2026112100 \
  --games 8000 --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 \
  --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450 --syzygy $TB --sf-path $SF"
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --kl-wdl 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
S_V="--value-outcome-weight 0 --value-search-weight 1 --value-search-source stored_search_wdl"
Z_V="--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl"
SIB="--sibling-dir $SIDE --w-sibling"
declare -A CELL=(
  [P_sib_w23]="--w-wdl 0 $SIB 23"   [P_sib_w69]="--w-wdl 0 $SIB 69"   [P_sib_w207]="--w-wdl 0 $SIB 207"
  [Zq_sib_w41]="$Z_V $SIB 41"       [Zq_sib_w122]="$Z_V $SIB 122"     [Zq_sib_w366]="$Z_V $SIB 366"
  [S_sib_w28]="$S_V $SIB 28"        [S_sib_w83]="$S_V $SIB 83"        [S_sib_w248]="$S_V $SIB 248"
  [P0_main]="--w-wdl 0 --w-sibling 0")
```

The `CELL` table must agree with `SB/stage_a/cells.json` (the checker's source of expected
values); `train_check.py` fails any cell whose realised w, w_wdl or search fraction differs.

**Step 1: the cost probe** (GPU alone):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u $SB/prelaunch/cost_probe.py $SB/cost_probe_sib.json 50 $COLLECT $TRAIN ${CELL[Zq_sib_w366]} || exit 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u $SB/prelaunch/cost_probe.py $SB/cost_probe_base.json 50 $COLLECT $TRAIN $Z_V --w-sibling 0 || exit 1
$PY - $SB <<'PYEOF' || exit 1
import json, sys
s, b = (json.load(open(f"{sys.argv[1]}/cost_probe_{k}.json")) for k in ("sib", "base"))
if s["peak_reserved_gib"] > 0.85 * s["device_total_gib"]:
    sys.exit(f"ABORT: peak reserved {s['peak_reserved_gib']:.1f} GiB > 0.85 x {s['device_total_gib']:.1f}")
cell_s = 1412 * s["step_s_median_11_on"] + 150      # update steps + the sidecar sha verification
proj = (9 * cell_s + 440) / 3600
print(f"sibling step {s['step_s_median_11_on']:.3f}s vs {b['step_s_median_11_on']:.3f}s; projected training {proj:.2f} GPU-h")
if proj > 2 * 3.4:
    sys.exit("HOLD for the owner: projected training > 2 x the registered estimate")
PYEOF
```

**Step 2: training, one crash rerun per cell, one take-effect check.**

```bash
train() { date -Is > $SB/arm_$1.started
  PYTHONPATH=. $PY -u scripts/simple_rl_loop.py run --run-dir $SB/arm_$1 --collect-from $P/collect \
    --generations 1 $COLLECT $TRAIN ${CELL[$1]} > $SB/arm_$1.log 2>&1; echo "$? $(date -Is)" > $SB/arm_$1.exit; }
CELLS="Zq_sib_w41 Zq_sib_w122 Zq_sib_w366 P_sib_w23 P_sib_w69 P_sib_w207 S_sib_w28 S_sib_w83 S_sib_w248 P0_main"
for c in $CELLS; do train $c; done
for c in $CELLS; do
  [ "$(cut -d' ' -f1 $SB/arm_$c.exit)" = 0 ] && continue
  [ -z "$(compgen -G "$SB/arm_$c.crash1*")" ] || continue
  for f in "" .log .exit .started; do [ ! -e $SB/arm_$c$f ] || mv $SB/arm_$c$f $SB/arm_$c.crash1$f; done; train $c
done
$PY $SB/stage_a/train_check.py $SB
```

**Step 3: guard (d), the main-written control loads identically at the PIN.**

```bash
C=$SB/arm_P0_main/gen_0001/train/model.pt
(cd $PINW && CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY $SB/prelaunch/check_load.py $C $SB/check_load_P0main_pin.npz) || exit 1
(cd $TW && CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY $SB/prelaunch/check_load.py $C $SB/check_load_P0main_main.npz) || exit 1
$PY $SB/prelaunch/check_load.py --compare $SB/check_load_P0main_pin.npz $SB/check_load_P0main_main.npz || exit 1
```

**Step 4: Stage A** (at the PIN; self-enforced one rerun, by running the same line again):

```bash
$SB/stage_a/run_stage_a.sh || exit 1      # writes $SB/offline/cells/stage_a_decision.json -> stage_b
```

**Step 5: Stage B**, for the i-th entry `X` of `stage_b` (i = 1, 2), seed `SEED` = 2026112193
(i = 1) or 2026112194 (i = 2):

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup
echo "7c8fd065865f3d5abcd8e9eb6f7fc1d48e882dfcfb88bbaa92d8696ce4a9510d  $F/run_followup_match.sh" | sha256sum -c --quiet || exit 93
[ "$($PY -c "import json;print(json.load(open('$SB/offline/cells/stage_a_decision.json'))['stage_b'][$i-1])")" = "$X" ] || exit 91
[ ! -e $SB/arm_$X/VOID ] && [ ! -e $SB/arm_$X/INERT ] || exit 91
M=$SB/arm_$X/gen_0001/train/model.pt
for O in $SB/smoke $SB; do n=$([ $O = $SB ] && echo 3200 || echo 4)
  $F/run_followup_match.sh $O sib_$X 1 A $SEED $n --candidate $M --reference $SL || exit 1
  $F/run_followup_match.sh $O sib_$X 1 B $SEED $n --candidate $SL --reference $M || exit 1
  d=$O/arenas/sib_$X; (cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $M \
    --a-games $d/sib_${X}_1_A.games.jsonl --b-games $d/sib_${X}_1_B.games.jsonl \
    --a-result $d/results.jsonl --b-result $d/results.jsonl \
    --a-label sib_${X}_1_A --b-label sib_${X}_1_B --out $d/pool.json > $d/pool.log 2>&1) || exit 1
done
```

The first pass is the 4-game smoke (2 openings); the real match runs only if it exits 0.
Plain seats, so no `check_seat.py` readback applies; every result row's `git_sha` must equal
`$PIN` and its candidate/reference sha256 the intended files, as in every follow-up match.
