# 2026-11-21: RL run 1 — sibling-ordering value screen (`--w-sibling`) on the pilot's collection (pre-registration)

**Status: registered, not launched.** It launches only after:
- review and merge of this record;
- the value-anchor screen (queue 2, `rl-run1-value-anchor/`) has finished. This screen
  queues AFTER it, and its Stage A needs that screen's instrument control and S replicate.

It trains new cells on the pilot's frozen collection and the banked sibling-pair sidecar,
plays no new collection games, and changes nothing live. Outputs go to
`~/chess-artifacts/operations/rl-run1-sibling/` (`SB`); the banks it reads are read-only.
Revision 2 applied the #1481 statistics review (evidence:
`~/chess-artifacts/operations/rl-claude-20261119/plan-review-1481/`). Revision 3 sets the training pin to
main **`9c0ff07dc`** (the merge of #1486, the Trainer-level take-effect test), registers a page-cache warm-up of
the sidecar bytes the cells read, and restates the Zq family's Stage B after match 8. Revision 4 applies the
statistics review at `fbb177c9f` (evidence: `plan-review-1481/delta-fbb177c9f/`):
- the P shape flag is removed, and the shape numbers are report-only;
- a Stage B damage reading retires the RAW recipe only, with a label per family;
- a control failure without the S replicate reads "undecidable";
- the Stage B order is P, Zq, S (smallest contrast MDE first);
- step 1a/1b text now matches the commands;
- a per-cell budget check runs after every sibling cell;
- `stage_a_decide.py` v3 is tested with 24 fabricated checks and nine mutants.

Revision 5 applies the review at `262a4e56b` (evidence: `plan-review-1481/delta-262a4e56b/`):
- **Stage B is two head-to-heads against each family's no-term counterpart**: P_sib vs `P0_main`
  (plain) and recalibrated Zq_sib vs recalibrated Zq (match 8's player); raw cell vs base is dropped;
  S only if neither P nor Zq has a PASS cell;
- revision 4's "cleanest evidence" paragraph is withdrawn: against the base, a P cell's "adoptable"
  was the expected result of a term that does nothing (0.86 at P0's point estimate, 0.71 allowing
  for its SE); both damage priors (Zq, S) are stated at the point estimate and predictively;
- match 2's (T, b) is fitted before any game, behind a fit control and a registered bounds stop;
- `stage_a_decide.py` v4 / `cells.json` v4 carry the head-to-head specs (27 fabricated checks, ten
  mutants); `SB/stage_b/` banks the two-seat readback and the fit gates, each tested and mutated.

Revision 6 applies the review at `376983c6c` (evidence: `plan-review-1481/delta-376983c6c/`):
- **"unread" is relabelled** "unread: CI contains 0 (this design had 80% power only for |effect|
  ≥ ≈ 9)": at SE 3.37 an interval containing 0 is consistent with true effects up to ≈ 13 Elo
  either way, so it does not show |effect| below the MDE;
- **a `Zq_main` cell** (Zq's flags at `TSHA`, `--w-sibling 0`) is match 2's opponent, with its own
  fit and bounds, and kl_pair(Zq ‖ `Zq_main`) is a second code-base control (it exercises the WDL /
  Zq value-loss branch, which `P0_main` never runs), gating the Zq and S families;
- the recalibration confound is stated exactly (calibratability is part of the treatment), and both
  seats' fit cross-entropies are reported;
- `check_both_seats.py` and `recal_gate.py` gain the cases the review's mutants needed (`recal_gate`
  now checks what was fitted, not only T and b); `stage_b_verdict.py` applies the rule to `pool.json`;
- the value-anchor screen's first Stage B result is added to the evidence (context, not a redesign).

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
| Share of the searched sibling difference NOT explained by the net's own difference, at ≥ 2 / 4 / 8 visits | 20 / 29 / 35% | same README; `RC/sibling-target-scoping/out/signal_pilot.json` [`faa1726e`] |
| Intra-game correlation (ICC by game): z − base / survivor_q − base / search_wdl − base / sibling residual | 0.62 / 0.013 / 0.028 / ≈ 0.001 | `RC/sibling-target-scoping/out/icc_fixed_pov.json` [`e94618a4`] |
| The banked pilot sidecar | 1,490,177 pairs on 752,093 rows; 722,742 training rows, 713,407 with pairs; 1,024-pair hold-out probe | `RC/sibling-pairs-pilot/v2/gen_0001/sibling_manifest.json` [`bbd97a99`], README [`dd975485`], `RC/sibling-pairs-pilot/v2/logs/trainer_load_check.txt` [`5006a7ce`]; `SIDECAR_VERSION` 2 |
| S's root value loss broke sibling ordering (clear pairs, vs base); Zq's did not | S −0.0096 [−0.0129, −0.0064]; Zq −0.0007 [−0.0032, +0.0017] | `RC/pilot-value-diagnosis/out/sibling_order.json` [`bb8e5b16`], README [`ae5e0866`] |
| Without the term, every banked arm moves the learner's probe AWAY from the search (mean \|Δv − dq\| minus the base's 0.1031; 1,024 pairs from 7 games, game bootstrap) | S +0.0079 [+0.0010, +0.0142]; Zq +0.0107 [+0.0034, +0.0169]; P0 +0.0088 [+0.0050, +0.0127] | `RC/plan-review-1481/probe_banked.json` |
| Follow-up matches at the PIN, 6,400 games each (registered verdicts in #1482) | 5 (P0 single net vs base) +2.2 [−4.2, +8.6]; 7 (Zq + recal vs Zq) +21.4 [+15.1, +27.6], "the loss is shape"; 6b (S's raw value on the base policy vs base) −21.4 [−27.7, −15.2]; 6a (the same, recalibrated) −16.1 [−22.3, −9.9], recovery R +5.3 [−3.5, +14.2]: **match 6 "unread"** | `~/chess-artifacts/operations/rl-run1-followup/arenas/{p0_single,zq_recal_h2h,s_value_raw,s_value_recal}/pool.json` [`22fc7f97` / `e1945cbd` / `3534eac9` / `0ff08c98`] |
| Match 8: recalibrated Zq (0.7036, −0.2271) vs the SL base | **+9.6 [+3.3, +16.0]**, "adoptable" and "improves on the base": nominates "Zq + per-generation recalibration" for a two-generation run under its own record; adopts nothing | `~/chess-artifacts/operations/rl-run1-followup/arenas/zq_recal_vs_base/pool.json` [`3e407ba8`] |
| Value-anchor screen, Stage B match 1 (2026-11-22): S + `--kl-wdl` 7.0 vs the SL base, 6,400 games, seed 2026112181 (unanchored S was −20.0) | **+5.81 [−0.53, +12.16]**, "adoptable" (nominates; does not improve on the base). Its Stage A: lowering `--w-wdl` to 0.1 / 0.03 does not brake the head (KL to base 0.017; clear-pair ordering −0.72 / −0.62 pt), the anchor does (λ 7.0: KL 0.0011, ordering −0.17 pt [−0.37, +0.03]). The Zq λ 11.3 match is running | `~/chess-artifacts/operations/rl-run1-value-anchor/arenas/anchor_S_k7.0/pool.json` [`0722d72b`], `~/chess-artifacts/operations/rl-run1-value-anchor/offline/cells/stage_a_decision.json` [`ea8ab67b`] |

So the sibling target carries information the net lacks (the search is right more often on
clear pairs), its rows are nearly independent within a game (ICC ≈ 0.001), and the one value
recipe that damaged ordering (S) is also the one that lost most in play. Whether training on
the sibling difference helps the value in play is unmeasured. This screen measures it. Since
match 8, the bar for a Zq-family recipe is recalibrated Zq, not the base: Stage B (revision 5) therefore
matches recalibrated Zq + sibling against recalibrated Zq.

## What `--w-sibling` computes (read at main `febed2042`; unchanged at `9c0ff07dc`)

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
  first 1,024 hold-out pairs are the learner's probe.
- **Take-effect record:** `summary.sibling` (pairs, steps, per-step (pairs, loss), the probe's
  mean |Δv − dq| before and after) and the manifest input `sibling_gen_0001`.

### Review item 1: the term's take-effect at the Trainer (#1486, merged as `9c0ff07dc`)

Revision 2 found that a mutant in `sibling.py` (`return self.w * loss` → `return None`: the term
still computed and recorded, never added to the loss) passed all 24 sibling tests. #1486 added
`tests/test_simple_rl_sibling.py::test_the_sibling_term_trains_the_value_head_through_the_trainer_hook`:
with `w_policy` 0, `w_wdl` 0 and `w_sibling` > 0, `train_steps` must give the value head a
non-zero gradient, and the probe must end LOWER than a same-seed run with the term off. The
reviewer's four mutants are all killed by it: the term not added; detached; backward on a total
that excludes the penalty; `0.0 * w * loss`. It passes at `9c0ff07dc` (run 2026-11-21, CPU) and
is in step 0's pytest line. #1486 also added tests for `--sibling-rows` and for refusing a
sidecar materialised from another collect.

**#1486 also changed `TrainSettings.validate`** (in `simple_rl/learn.py`, 5 lines): it refuses
an all-zero objective (`w_policy`, `w_wdl`, `w_sibling`, `kl_policy` and `kl_wdl` all 0). No cell
hits it: every cell keeps `w_policy` 1. Checked by passing every cell's flags through the loop's
parser, `train_settings` and `validate()` at `9c0ff07dc` (`SB/prelaunch/validate_cells.json`).

Run-level evidence (the reviewer's, above): with no term, all three banked arms RAISE the
probe's |Δv − dq| over the base. A cell whose all-pairs |Δv − dq| falls below its banked family
net's (the Stage A take-effect rule, below) has therefore moved against what training without
the term does.

## Code base: train on main, match at the PIN

The term exists only on main, not at the pilot's PIN `0c795fb7a`. **Training runs at the pin
`TSHA` = `9c0ff07dc0858c73def20678ce274c928d83dcec` in a detached worktree. Every match and every Stage A instrument except the
all-pairs take-effect statistic runs at the PIN**, as all pilot, follow-up and anchor-screen
numbers did. The take-effect statistic needs `sibling.py`, so it runs at `TSHA` with the same
loader; guard (b) shows `model/`, `uci/` and `inference.py` are unchanged.

**Every non-test file changed between the PIN and the training pin `9c0ff07dc`** (`git diff
--name-only 0c795fb7a 9c0ff07dc -- chess_anti_engine scripts/simple_rl_loop.py`), by whether it
is on the training path:

| Files | Change | On the training path? |
|---|---|---|
| `simple_rl/learn.py`, `simple_rl/sibling.py`, `simple_rl/common.py` (`is_holdout_game` moved here), `train/losses.py` (11 lines), `scripts/simple_rl_loop.py` | the sibling term (default 0), `--value-rows-per-game` (default 0 = every row), the `root_full_width` collect flag, #1486's all-zero-objective refusal | yes; every new behaviour is default-off or a refusal no cell triggers, and guard (a) and the `P0_main` control test that |
| `replay/dataset.py` (3 lines) | `collate_arrays` copies a `value_row_weight` key if present; `compute_loss` uses it only if present | yes, via `collate_arrays`; the key exists only when `--value-rows-per-game` > 0, which no cell sets |
| `replay/shard.py` (#1477, 5 lines), `replay/sample.py`, `replay/augment.py` | a new OPTIONAL stored field `root_prelude_visits` (`has_root_prelude_visits`), absent from the pilot's shards and not a training key | it is in the shard reader; guard (a) rerun at `6769c32f7` decodes the pilot's rows byte-identically (below) |
| `simple_rl/collect.py`, `mcts/gumbel_c.py` (per-game RNG plumbing), `mcts/gumbel.py`, `selfplay/{config,finalize,manager,network_turn,state}.py` | collect: game-granularity resume, VRAM record, journal, per-game RNG streams, `root_full_width` (#1477, default off) | no: collect only; these cells read a frozen collect |
| `tune/{distributed_runtime,trainable_config_ops,trial_config}.py`, `utils/config_yaml.py`, `worker.py` | `root_full_width` plumbing for the distributed path | no |

**Between `febed2042` (revision 1's pin) and `9c0ff07dc`**, the commits that touch
`chess_anti_engine/` or `scripts/`:
- **#1477** `root_full_width` (`48975a920`, `095059611`, `8f57af455`): `replay/{shard,sample,augment}.py`,
  `simple_rl/collect.py`, `scripts/simple_rl_loop.py`, `mcts/{gumbel,gumbel_c}.py`,
  `selfplay/{config,finalize,network_turn}.py`, `tune/*`, `utils/config_yaml.py`, `worker.py`.
  On the training path only through the shard reader and the loop's CLI; default off.
- **#1479** (`4c077ea0a`): `scripts/arena_standard.py` only (strict bool parsing of `--*-gumbel`
  flags). Arena code, and matches run at the PIN, so not on this screen's training path.
- **#1480**: docs and `tests/test_simple_rl_sibling.py` only.
- **#1486** (`fd8d98cd4`): `simple_rl/learn.py` (the `validate` refusal above) and tests.

Nothing else is on the training path. The record's code base is `9c0ff07dc`; anything merged
later is not used.

**The pilot's guards, restated for PIN → `TSHA`:**
- **(a) The decoded training input is identical.** This is ONE check. The pilot's three PIN
  digests `rl-run1-pilot/guard_a_{P0,S,Zq}_0c795fb7a.json` are byte-identical to each other,
  because `prepare_targets` does not depend on the value flags. The value-loss paths of S and Zq
  at main are covered by the `realised_*` fields that `train_check.py` checks, and by reading
  the code (the only change in `compute_loss` is the `value_row_weight` branch above).
  - Run at `febed2042` (CPU, 2026-11-21): all three byte-equal, as expected
    (`SB/prelaunch/guard_a_*_febed2042.json`, `guard_a_febed2042.verdict`; 37.2 GB peak RSS each).
  - Re-run once at `6769c32f7`, past #1477's shard-reader change (P0's flags; CPU, 2026-11-21):
    **byte-equal** (`SB/prelaunch/guard_a_P0_6769c32f7.json`, `guard_a_6769c32f7.verdict`).
  - **Re-run once at the training pin `9c0ff07dc`** (P0's flags; CPU, 2026-11-21): **byte-equal**
    (`SB/prelaunch/guard_a_P0_9c0ff07dc.json`, `guard_a_9c0ff07dc.verdict`). The only non-test
    change from `6769c32f7` is #1486's `validate` refusal, which cannot change the decoded rows;
    the re-run shows it directly. Step 0 checks this verdict.
- **(b) The encoder that wrote the stored `x` is identical.** It matters less here than at
  C0 → PIN: cells read the STORED `x`, and matches run at the PIN as before. Checked
  mechanically in step 0:
  - `git diff --quiet 0c795fb7a $TSHA` over `encoding/`, `mcts/_mcts_tree.c`, `moves/`,
    `replay/packed_zarr.py`, `model/`, `uci/` and `inference.py` must be empty (it is at
    `febed2042`, `6769c32f7` and `9c0ff07dc`);
  - `replay/shard.py`'s diff must be exactly #1477's optional-field addition: its sha256 is
    pinned (`418b6e96…`, measured at `6769c32f7`; #1486 does not touch it). Any other change to
    it stops the screen.
  - `selfplay/network_turn.py`'s diff does not touch `_batch_encoder_pair`, and `x_hash` is
    unchanged. (The banked `guard_b_0c795fb7a_*.json` were made by an extended version of the
    launch record's guard (b) block whose script is not in the pilot bank, so it is not rerun
    here; the diff check covers every file it exercises.)
- **(c) Default off is off on the pilot's real inputs.** The code-base control cell `P0_main`
  (P0's flags at `TSHA`, `--w-sibling 0`) is compared with the banked P0, PAIRED, in Stage A
  (below). #1478's reviewer showed bit-identity of weights on a tiny phase; this is the same
  claim on the real phase.
- **(d) A main-written checkpoint loads and evaluates identically at the PIN.**
  `SB/prelaunch/check_load.py` loads `P0_main`'s `model.pt` once at the PIN worktree and once at
  the training worktree, through `uci.model_loader.load_model_from_checkpoint` (the loader the
  arena and every Stage A instrument use). It digests every state_dict tensor and forwards the
  first 512 hold-out rows of shard 0 on CPU in fp32. The two outputs must be bit-equal
  (`check_load.py --compare`). A failure stops the screen before Stage A.
  - Smoke on the SL base: IDENTICAL for `sl_base.pt` at `0c795fb7a` vs `febed2042` (494
    state_dict digests and both logit arrays). A one-ulp change to one WDL logit makes
    `--compare` print DIFFER and exit 1 (`SB/prelaunch/check_load_mutant.txt`). The reviewer
    re-ran it on `sl_base` and on the banked P0: IDENTICAL, mutant killed.
  - `model/`, `uci/` and `inference.py` are unchanged from `febed2042` to `9c0ff07dc` (guard (b)),
    so the smoke carries to the pin; step 3 repeats it on `P0_main`.

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
of about 1.9. Cosines with the root terms are small and positive on the head (sibling vs S 0.29,
vs Zq 0.19); on the trunk, sibling vs policy is 0.04. Bank: `SB/prelaunch/grad_scale.json`
(sha256 `782d52a2a358…`), script `grad_scale.py` (`6a25772191f5…`).

**Independent check (the reviewer's; `RC/plan-review-1481/sib_grad_indep.json`)**: 24 random
128-row pair sets of sidecar shard 0, production `sibling_loss`: `value_wdl` norm 0.00919
(se 5.4%), trunk 0.0285 (se 4.5%). Both agree with the 4-batch figures within their noise.

**The strengths are nominal.** Four batches put about ± 28% on each w. That is acceptable for a
9× ladder, but the shares below are labels, not measurements of what a cell experiences:
- **Coherence.** On the reviewer's 24 sets, ‖mean gradient‖ / mean per-set ‖gradient‖ is 0.63 on
  the head and **0.28 on the trunk**. Under momentum, which averages across batches, the term's
  coherent trunk share is therefore well below its per-batch norm share. The same ratio for the
  policy term is **unknown**: measuring it on CPU (`SB/prelaunch/coherence.py`, 24 batches)
  peaked at 23.6 GiB RSS on the first batch, above the 15 GiB budget set for it, and stopped
  (`coherence.json` records that). So the trunk shares are not comparable across the families
  in coherent terms.
- **Huber regime.** At the base, 81% of residuals are in Huber's quadratic zone (|r| ≤ 0.2).
- **Weights.** `min_v / 64` is heavy-tailed: mean 0.112, median 0.0625, p99 0.49.
- **Drift.** Step-0 shares drift as the head fits: as Δv approaches dq, the term's gradient
  shrinks, and the root terms' gradients change too.

**Reading.**
- The sibling term is small in absolute size: its mean weighted Huber is 0.00143, because
  the visit weight averages well below 1 and most residuals sit in the quadratic zone (mean
  |Δv − dq| 0.120). Its gradient norm at w = 1 is therefore a small fraction of the WDL CE's.
- Both optimizers here are scale-invariant per update (the anchor screen's derivation): the
  value head is in an AdamW group at 3e-6. So for parameters reached by ONE term, w barely
  changes the step. What w sets is the term's share of the gradient wherever terms mix: the
  value head in the S and Zq families, and the trunk in every family.
- **The P family is NOT a head-dose ladder.** With `--w-wdl 0` the value head receives only
  this term, in an AdamW group, so w does not change the head's step. The three P cells are ONE
  head dose at three trunk shares.
- **Strength rule:** w is set so that, at step 0, the sibling term contributes a share s of the
  combined gradient norm, s ∈ {0.25, 0.50, 0.75}:
  w = s / (1 − s) × ‖g_other‖ / ‖g_sib(w = 1)‖.
  - S and Zq families: g_other is the root WDL term's value-head gradient (the head is where
    the two value signals compete).
  - P family: g_other is the policy term's trunk gradient (above).

| Cell | ‖g_other‖ / ‖g_sib‖ | s | `--w-sibling` | Step-0 share on `value_wdl` | Step-0 share on the trunk |
|---|---|---|---|---|---|
| `P_sib_w23` / `w69` / `w207` | 1.695 / 0.0246 = 69.0 (trunk, vs policy) | 0.25 / 0.50 / 0.75 | **23 / 69 / 207** | 1 (only term) | 0.25 / 0.50 / 0.75 |
| `Zq_sib_w41` / `w122` / `w366` | 1.094 / 0.00896 = 122.1 (head, vs Zq WDL) | 0.25 / 0.50 / 0.75 | **41 / 122 / 366** | 0.25 / 0.50 / 0.75 | 0.21 / 0.45 / 0.71 |
| `S_sib_w28` / `w83` / `w248` | 0.739 / 0.00896 = 82.5 (head, vs S WDL) | 0.25 / 0.50 / 0.75 | **28 / 83 / 248** | 0.25 / 0.50 / 0.75 | 0.22 / 0.45 / 0.71 |

Trunk shares are against the sum of the other terms' trunk norms (policy plus root WDL).
Values are rounded to 2–3 significant figures. The reviewer checked AMP safety at w = 366.

**No `--kl-wdl` cell here.** Whether an anchor helps, and at which λ, is exactly what the
value-anchor screen decides; a sibling + anchor cell would have to choose λ before that reading
exists (a forking path). A combination is registered later, under its own record, only if the
anchor screen nominates a λ.

## Cells

All cells train at `TSHA` (`9c0ff07dc`) on the pilot's frozen collection (`--collect-from`, the pilot's
`COLLECT` and `--seed`) with the pilot's `$TRAIN` verbatim (it carries `--kl-wdl 0`).

| Cell | Value flags | Role |
|---|---|---|
| `P_sib_*` (3 strengths) | `--w-wdl 0 --w-sibling w --sibling-dir $SIDE` | (a) policy + sibling only: the root value gets no loss. The cleanest test of "does sibling training alone help, or at least not hurt, the value" |
| `Zq_sib_*` (3) | Zq's flags + `--w-sibling w --sibling-dir $SIDE` | (b) Zq's root target (its loss vs the base is largely shape, match 7) plus the term |
| `S_sib_*` (3) | S's flags + `--w-sibling w --sibling-dir $SIDE` | (c) does the term repair the ordering S breaks? |
| `P0_main` | `--w-wdl 0` (term off) | code-base control (c) above; match 1's opponent |
| `Zq_main` (revision 6) | Zq's flags, `--w-sibling 0` (term off; reads no sidecar) | the Zq-branch code-base control (Stage A) and **match 2's opponent** |
| Controls, not retrained | the banked P0, S, Zq (`rl-run1-pilot/arm_*`) | references |

`$SIDE` = `~/chess-artifacts/operations/rl-claude-20261119/sibling-pairs-pilot/v2` (the
directory holding `gen_0001/`).

## Take-effect

**At training** (`SB/stage_a/train_check.py` v2, one run, after every crash rerun). A field
mismatch writes `arm_<cell>/VOID`. For every cell:
- exit 0 and a sealed train manifest; init checkpoint sha256 = the SL base's (`a39f900c…`);
- `summary.realised_w_wdl` = 0 (P family, `P0_main`) or 1 (Zq, S families, `Zq_main`);
  `summary.realised_search_wdl_frac` = 0, 0.5 or 1.0 respectively;
- `summary.updates` = 1,412, `summary.drawn_rows` = 722,742, `presented.npz` sha256 =
  `63e227ac…` (the pilot's batches, proved at the consumer); no KL anchor.

For every sibling cell, additionally:
- `summary.sibling.w` = the cell's w, `sibling_rows` = 128, `steps` = `updates`, one per-step
  entry per update, the per-step pairs summing to `pairs` > 0;
- the sidecar record and the manifest input `sibling_gen_0001` both carry sha256 `bbd97a99…`
  (the banked v2 sidecar);
- the learner's probe has 1,024 pairs before and after. Its values are **reported only**: the
  probe comes from 7 games, too few to decide on.

`P0_main` and `Zq_main` must carry no `summary.sibling` and no sidecar input (`train_check.py`'s
`w_sibling` 0 branch; `cells.json` v5 lists `Zq_main` with Zq's `w_wdl` 1 and search fraction 0.5).

**In Stage A: the take-effect statistic, which decides INERT** (`SB/stage_a/sibling_holdout.py`,
review item 5). On ALL hold-out sibling pairs of the v2 sidecar (the pilot's hold-out games;
416 games, none trained by any cell), each net's residual Δv − dq through the production
`load_sidecars` / `sibling_loss` at `TSHA`, eval mode, fp32. Per-pair residuals of every net are
banked first (`sibling_holdout_resid.npz`, with `game_id`, `dq`, `min_visits`).
- Statistic: mean |Δv − dq| of the cell minus that of its banked family net, PAIRED by pair:
  P cells vs P0, Zq cells vs Zq, S cells vs S (and `P0_main` vs P0, `Zq_main` vs Zq, reported). CI: 2,000 game
  bootstrap resamples, seed 20261121.
- **"Takes effect" iff the 95% CI lies entirely below 0. Otherwise the cell is INERT**: scored,
  never matched.
- Reported beside it: each net minus the base (this extends the reviewer's 7-game probe result
  to all hold-out pairs), the visit-weighted twin of the statistic, and the learner's probe.
- `sibling_holdout.py --max-pairs` is for the smoke only; the registered run has no limit.
- **CPU smoke** (2026-11-21, at `febed2042`, the first 4,096 hold-out pairs = 23 games; the
  banked nets, with a copy of P0 passed as the cell `P0_main`; `SB/stage_a/smoke/`):
  - base 0.1110;
  - vs base: P0 +0.0090 [+0.0068, +0.0115], S +0.0031 [+0.0003, +0.0061], Zq +0.0059
    [+0.0032, +0.0089]. That is the reviewer's direction, on other games;
  - the P0 copy gives `minus_family` exactly 0, and so INERT, as it must;
  - peak RSS 3.4 GB; 34.5 min of wall on CPU, most of it the sidecar's sha256 pass.

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
- **Ruler leakage** (reviewer, `RC/plan-review-1481/ruler_overlap.json`): the sibling-ordering
  ruler (rl-audit, 2,433 leak-free rows, 190 games) shares **0 game ids** with the pilot collect
  (761,979 rows, 8,000 games). **1 ruler row's exact `x` appears in the collect.**
- This screen's own files (`SB/stage_a/`, revision 6 plus the post-merge test addendum, read-only,
  `SHA256SUMS` `fbc8f863…` over every file; the revision-6 seal `610347b2…` kept as
  `SHA256SUMS.v5`; revision 1 files kept as `*.v1`, revision 2/3 as `*.v2`, revision 4 as `*.v3`,
  revision 5 as `*.v4`, each with its `SHA256SUMS.v*`):

  | File | sha256 prefix |
  |---|---|
  | `cells.json` (v5: adds `Zq_main` (family Zq, `control`); `zq_control_gates` Zq, S; Zq opponent `Zq_main`; the relabelled "otherwise" rule) | `93701afe1f2f…` |
  | `train_check.py` | `f739886c075f…` |
  | `sibling_holdout.py` | `0897bf777521…` |
  | `stage_a_decide.py` (v5: the Zq-branch code-base control) | `9d60b7e1dc3e…` |
  | `run_stage_a.sh` | `241de8fb1dab…` |

  **`fab_v6/` (post-merge addendum, review of revision 6 at `4c73b911b`)** tests the same v5
  decision script and cells: **36 checks, all pass**: fab_v5's 34 plus P (no usable S replicate, a
  `Zq_main` gap of 0.158 × KL(base ‖ Zq): the Zq control, on its floor alone, FAILS and both gated
  families read "unread: Zq code-base control undecidable (no replicate)", P only) and P2 (no
  replicate, a bit-identical `Zq_main`: passes, P and Zq). **Sixteen mutants, each killed**
  (`fab_v6/mutants/mutants.txt`): fab_v5's fifteen plus the reviewer's Zq floor 0.05 → 0.5, which
  survived fab_v5 (`fab_v6/fab_v5_on_m_zq_floor_0p5.log`: ALL_OK) and is killed by P.

  `fab_v5/` holds the v5 decision script's test (revision 6): **34 checks, all pass** (rerun on the
  banked files). v4's 27 (the spec check now expects `Zq_main`), plus: a bit-identical `Zq_main`
  passes and is never a cell; `Zq_main` VOID → P only; `Zq_main` far from Zq → the Zq control fails,
  P only, Zq and S labelled; the Zq control failing with no P cell → no S fallback; a small gap
  passes; a gap between 1× and 2× the S replicate's passes only under the registered 2×; the
  relabelled rule. **Fifteen mutants, each killed** with a FAIL line (`fab_v5/mutants/mutants.txt`):
  v4's ten regenerated on v5, plus the Zq tolerance at 1×, the Zq gate ignored, the S fallback
  ungated, `Zq_main` treated as a cell, and `Zq_main` VOID ignored.

  `fab_v4/` (kept) holds the v4 decision script's test (revision 5): 27 checks, all pass. v3's 24, with
  the labels check replaced by one on the head-to-head specs (P vs `P0_main`, plain; Zq vs Zq,
  both recalibrated; rule "LB95 > 0"), plus: P all VOID gives Zq ONLY (S is not added as a second
  match), Zq all INERT gives P only, and an S fallback carries its known-damage label. Ten mutants,
  each killed by exit code 1 (`fab_v4/mutants/mutants.txt`): `m_rep1`, `m_anchor_rows`,
  `m_shape_wl`, undecidable, specs from the wrong family, S added as a second match, smallest-w,
  control ignored, rows check, and the order Zq, P in `cells.json`.

  `SB/stage_b/` (revision 6 plus the post-merge test addendum, read-only, `SHA256SUMS` `02631123…`;
  the revision-6 seal `01432616…` kept as `SHA256SUMS.v2`; revision 5's files as `*.v1`):
  - `check_both_seats.py` `c205a6e755d5…` (unchanged; match 2's readback; the follow-up
    `check_seat.py` fails by design when the other seat is also a hybrid). Its test now has 11
    cases: candidate T off by 1e-4, reference T differs, reference `value_net=loaded` and reference
    row recal differs were added (review item 4). Four mutants are killed, including the reviewer's
    two survivors ("T matched to 3 decimals", "reference value_net unchecked").
  - `recal_gate.py` v2 `0231f67cd0b9…` (fit control and bounds, now checking what was fitted;
    passes on match 8's banked fit). Its test is v3 (addendum; v2 kept as `*.v2`): 20 cases, adding
    "bounds: 415 games" and "bounds: another collect"; **eleven mutants killed**, including two
    that skip the games or the collect check in bounds mode only. The games one passed test v2
    18 / 18 (`test/v2_on_mut_games_unchecked_bounds.txt`).
  - `stage_b_verdict.py` `9fb4d3bc1d6f…` (the registered label from `pool.json`; 7 cases, three
    mutants killed).
  - `SB/prelaunch/validate_cells.py` v2 `ba8f5a6fa0ae…`: `Zq_main`'s flags pass the loop's parser
    and `validate()` at `9c0ff07dc` (w_wdl 1, w_sibling 0); the other ten cells' output is
    byte-identical to v1's.

  `fab_v3/` (kept) holds the v3 decision script's test on fabricated inputs and the mutants:
  - 24 checks, all pass. They add the P, Zq, S order, the damage labels, the absence of any shape
    flag, a report-only ΔWL that must see a +W/−L shift, the "undecidable" control without a
    replicate, and a replicate whose `game_id` differs (scenario E3).
  - Nine mutants, each killed by exit code 1 (`fab_v3/mutants/mutants.txt`): the reviewer's three
    (`m_rep1`, `m_anchor_rows`, `m_shape_wl`, adapted to v3), plus undecidable, labels,
    smallest-w, control ignored, rows check, and the v2 order in `cells.json`.
  - `fab_v3/budget/` runs the step-2 `budget_check` function, extracted verbatim, on fabricated
    metrics (ok, exceed, the exact edge) with one mutant (2× budget), which is killed.

  `fab_v2/` (v2's 17 checks and six mutants) is kept.
- The pre-launch scripts are in `SB/prelaunch/` with their own `SHA256SUMS`: `grad_scale.py`
  `6a25772191f5…`, `cost_probe.py` (revision 3; revision 2 kept as `cost_probe.py.v2`)
  `ad41afdd6590…`, `check_load.py` `e8a870764bb5…`, `guard_a.py` `d247b0f0e665…`, `coherence.py`
  `bcdabcbed5c8…`, `warm_sidecar.py` `1989388fc967…`, `validate_cells.py` `2c4023cef494…`.

**Statistics** (raw heads; `stage_a_decide.py` v4, whose Stage A statistics are v3's unchanged):
- **VOID:** `train_check.py` failed the cell, or its VOID marker exists.
- **Veto, exactly as the anchor screen:** VETOED iff the 95% CI upper bound of
  `sibling_order.json → paired_minus_base.all.clearplus_50plus.<cell>_raw` is below 0.
- **The primary offline quantity this term targets is the same statistic**: clear-pair
  sibling-ordering accuracy vs deep SF, cell minus base, game-clustered. For once a POSITIVE CI
  (lower bound > 0) is the interesting outcome, and it is reported as "improves ordering".
  - The ruler rows come from the rl-audit collection, not the training games (leakage above).
  - It still may NOT select between survivors. Three fit→Elo transfer failures are on record:
    v020 (+3.1 cp audit gain, −9.6 Elo), ft595 (−41.0 Elo for better fit), F-only (hold-out CE
    −0.043 nats for 0 cp). An offline gain on the term's own target is the "arm that is the
    gradient of the metric" case: it is expected, not evidence of strength.
- **INERT:** the all-pairs take-effect statistic (above) is not entirely below 0.
- **Code-base control, paired (review item 8).** `P0_main` and the banked P0 are scored on the
  SAME hold-out rows (`holdout_logits.npz` of this Stage A and of the anchor screen's control).
  The script first checks that `game_id`, `wdl_target` and the base's logits are bit-equal in
  both files; otherwise the control fails.
  - Statistic: kl_pair = mean over rows of KL(p_P0 ‖ p_P0_main) on the WDL softmax.
  - Scale: kl_rep = mean over rows of KL(p_S ‖ p_S_trunc_final). The anchor screen's truncated-S
    final model is an S replicate: same flags, same code base, so the gap is GPU
    nondeterminism only.
  - **PASS iff kl_pair ≤ max(2 × kl_rep, 0.05 × mean KL(p_base ‖ p_P0)).** The floor applies if
    the S replicate is void or lands bit-identical. It is 5% of P0's own displacement from the
    base, a code-base effect far below the effects judged here (S 0.0167, Zq 0.0264 nats).
  - Reported beside it: whether the logits are bit-identical, the largest logit difference,
    mean |ΔWL|, the unpaired clear-pair / KL / share differences, and `P0_main` vs P0 on the
    all-pairs sibling statistic.
  - **A failure means main trains differently from the PIN on these inputs: no Stage B,
    verdict "unread: code-base control failed".**
  - **The Zq-branch control (revision 6, review item 2).** `P0_main` trains with `--w-wdl 0`, so it
    never exercises the WDL / Zq value-loss branch that `Zq_sib` and the pilot's Zq use; revision 5
    read that branch's equivalence from the diff only. So `Zq_main` (Zq's flags at `TSHA`, w 0) is
    scored the same way: kl_pair_Zq = mean KL(p_Zq ‖ p_Zq_main) on the same rows, **PASS iff
    kl_pair_Zq ≤ max(2 × kl_rep, 0.05 × mean KL(p_base ‖ p_Zq))** (the same rule, its floor scaled
    by Zq's own displacement). It gates the families whose value loss is on: **if it fails (or
    `Zq_main` is VOID), neither the Zq match nor an S fallback runs** ("unread: Zq code-base control
    failed", or "undecidable" without a replicate); the P match is unaffected. `Zq_main` is never a
    Stage B candidate.
  - **Except when the S replicate is VOID** (revision 4, review item 2). If the anchor screen's
    `logits_S_trunc_final` is missing, or its `game_id` / base logits are not bit-equal to the
    control's rows, kl_rep is None. Only the floor then applies, and a failure cannot be told apart
    from ordinary GPU nondeterminism. It reads **"unread: control undecidable (no replicate)"**,
    still with no Stage B. A pass against the floor alone still passes.
- **Level and draw mass, for every cell vs its family net (review item 4).** In the P cells
  (`--w-wdl 0`) only Δ(L − W) is fitted, so the value's level and its draw mass are free to move.
  The clear-pair veto cannot see that, because it is level-invariant. Shape alone cost Zq about
  21 Elo in match 7. So, paired on the hold-out rows with the same game bootstrap, the script
  reports:
  - the change in `signed_dWL_raw` and in `signed_dP_draw_raw`;
  - the change in `KL_share_removed`, each net by its own (T, b) fit;
  - both fits.

  **These numbers are report-only for every family, P included (revision 4, statistics review
  item 1).** Revision 3 turned a P cell's Stage B "damage" into "damage, shape-confounded" whenever
  any of the three CIs excluded 0. That flag could not be false:
  - It had no magnitude threshold. The game bootstrap on these 39,237 rows resolves a ±0.002
    logit shift (ΔWL 0.0004 [0.00038, 0.00040]).
  - P0 alone, with no value loss at all, moved ΔWL by −0.0120 [−0.0139, −0.0100] from the base.
  - A P cell passes INERT only if its value head moves, so every P cell that reached Stage B would
    have carried the flag, and "damage (otherwise)" could never fire.

  No magnitude threshold replaces it, because ΔWL magnitude does not track Elo here: P0's −0.012
  played +2.2, while Zq's −0.011 lost about 21 Elo on shape (match 7). The flag is removed. What a
  P-family match can and cannot conclude is stated under "What each outcome licenses".
- **Report-only:** each cell's `KL_base_net_raw`, `share_of_gap_closed`, `KL_share_removed`
  by its own (T, b), FEN-only value regret and outcome distance.

### Stage B: matches

**Revision 5 (review at `262a4e56b`; evidence `plan-review-1481/delta-262a4e56b/`): both matches
are head-to-heads against the family's no-term counterpart.** Revisions 1–4 matched each cell
against the SL base. Against the base, the likely outcome of each match was fixed by its family's
known result, not by the term:

| Match against the base (dropped) | Null prior: the term adds 0 | Information value (reviewer) |
|---|---|---|
| raw P_sib vs base | P0 played +2.2 (SE 3.26), so "adoptable, not improves" has probability **0.86** at P0's point estimate and **0.71** allowing for its SE; "improves" 0.10 / 0.17 | 0.65 bits (point) / 0.49 (predictive) |
| raw Zq_sib vs base | raw Zq −10.0 (SE 6.74): P(damage) **0.84** at the point estimate, **0.67** predictive | 0.65 / 0.24 |
| raw S_sib vs base | S −20.0 (SE 8.5): P(damage) **> 0.99** at the point estimate, **0.93** predictive | 0.08 / 0.10 |

A head-to-head against the family's counterpart has the null at 0 by construction: at the same
SE, P(adds | term adds 0) = 0.025 and P(unread) = 0.95, and its information value is 0.56 bits
whatever the family's level. **Revision 4's paragraph "Why the P match is still worth its cost"
is withdrawn.** It called an adoptable P + sibling cell "the cleanest evidence available that the
term itself carries a value signal". Against the base, "adoptable" was the expected reading of a
term that does nothing (0.86 / 0.71 above), so it was not evidence of any signal.

- **Selection, fixed now, not by offline score.** Per family, the **LARGEST-w** cell that is not
  VOID, VETOED or INERT.
  - **P and Zq are the primary families**: each that has such a cell is matched.
  - **S is a fallback**: matched only if neither P nor Zq has one. (Revision 6: the Zq and S
    matches also need the Zq-branch code-base control to pass.) S itself is known damage
    (−20.0 vs base), so an S match is labelled "repair of a known-damage recipe" and cannot
    nominate a recipe that competes with P or Zq.
  - So at most two matches, and the same two whatever the order.
  - Why the largest: the veto already guards against ordering damage, and the smallest dose is
    the weakest test of a term hypothesised to help.
- **Order: P first, then Zq.** The registered rule (smallest MDE first) is kept, but both
  head-to-heads now have the same SE, so it ties. **Tie-break: fewer confounds.** P is the
  family where the term alone moves the value head (`--w-wdl 0`); the Zq match adds a fitted
  recalibration on each seat. P takes seed **2026112193** and Zq **2026112194**; an S fallback
  match takes 2026112193. Both seeds were re-checked unused on 2026-11-22: absent from `main`,
  from every open PR other than this one (#1491 and #1484 included), and from the pilot,
  follow-up, value-anchor, root-full-width, generation-2 and sibling banks.

| | Match 1 (P) | Match 2 (Zq) | Fallback (S, only if neither) |
|---|---|---|---|
| Candidate | P_sib (largest-w PASS P cell), plain | Zq_sib (largest-w PASS Zq cell) + its own (T, b), fitted before any game | S_sib (largest-w PASS S cell), plain |
| Opponent | **`P0_main`** (trained at `TSHA` with w 0), plain | **`Zq_main`** (trained at `TSHA`, Zq's flags, w 0) **+ its own (T, b)**, fitted by the same command in the same session, before any game (revision 6) | the pilot's S, plain |
| Guard on the opponent | the P0 code-base control (P0 vs `P0_main`) passed, and guard (d) (`P0_main` loads identically at the PIN) | the Zq-branch code-base control (Zq vs `Zq_main`) passed, and guard (d) on `Zq_main`; both fits pass the fit control's procedure check and the bounds | the PIN-trained S, guarded by both code-base controls. **Why the Zq control covers the S fallback:** S and Zq both train the value head through the same `blended_wdl_ce` code at `w_wdl` 1 and differ only in the outcome / search coefficients; the PIN → `TSHA` diff has no branch that depends on those coefficients (its one new value-loss branch, the per-row `value_row_weight`, is keyed on `value_rows_per_game` > 0, off in every cell); `S_trunc_final` is PIN vs PIN and only sets the nondeterminism scale |
| Seats | plain both; result-row sha readback | both hybrid (own value + recal), as match 8 seated its candidate; `check_both_seats.py` readback | plain both; result-row sha readback |

- **Why `P0_main`, not the banked P0, for match 1.** P_sib and `P0_main` train at `TSHA` from
  the same rows, batches, seed and optimizer, so they differ by the term alone. P0 trained at the
  PIN would add the code base to the contrast. The code-base control (kl_pair ≤ its tolerance) is
  the guard that `P0_main` ≈ P0, which is what lets P0's banked +2.2 vs base describe it. Stage B
  runs only if that control passed.
- **Why `Zq_main` for match 2 (revision 6, review item 2).** Revision 5 used the pilot's Zq with
  match 8's banked (T, b), guarded only by a P0 control that never runs the Zq value-loss branch.
  With `Zq_main`, match 2 is recalibrated Zq_sib vs recalibrated `Zq_main`: they differ only by the
  term (plus GPU nondeterminism), and both fits come fresh from one procedure in one session.
  **The cost of the choice:** the opponent is no longer exactly match 8's player. That is
  acceptable because adoption of a Zq-family term is routed through the generation-2 record, not
  through this match. The fit control keeps its job: it shows the procedure reproduces the pilot
  Zq's banked fit.
- **Design.** 6,400 games (3,200 per seat run, 1,600 openings), seat-balanced, match 5's
  machinery: the follow-up launcher `rl-run1-followup/run_followup_match.sh` (sha256
  `7c8fd065…`) at the PIN with output root `SB`, the pilot's `$ARENA` flags and the launch
  record's max-plies void rule, a 4-game smoke per seat first, the result-row sha readback, and
  `arena_seat_balanced_pool.py` with `--arm` = the sibling cell.
- **Rules, per head-to-head (Elo of the sibling cell over its counterpart):**
  - **"the term adds"** iff LB95 > 0;
  - **"the term harms"** iff UB95 < 0;
  - otherwise **"unread: CI contains 0 (this design had 80% power only for |effect| ≥ ≈ 9)"**
    (revision 6). An interval containing 0 does NOT show the effect is below the MDE: at SE 3.37
    it is consistent with true effects up to ≈ 13 Elo either way.
  - At the gate-scaled SE 3.37, the 80%-power MDE is ≈ 2.8 SE ≈ 9 Elo. Power to read "adds" is
    0.32 at a true +5 and 0.84 at +10 (the reviewer's table, SE 3.37); false "adds" at a true 0
    is 0.025 per match.
  - **The label is applied by script, not by hand** (revision 6): `SB/stage_b/stage_b_verdict.py
    <pool.json>` prints it (match 5 was read by hand from `pool.json`). Tested on fabricated pools at
    the three boundaries (LB exactly 0 and UB exactly 0 read unread; LB 0.01 adds; UB −0.01 harms),
    a straddling interval and a malformed pool; three mutants killed (≥ for >, ≤ for <, swapped).

  | True effect vs counterpart (Elo) | −10 | −5 | 0 | +5 | +10 |
  |---|---|---|---|---|---|
  | P(the term adds) | < 0.001 | < 0.001 | 0.025 | 0.32 | 0.84 |
  | P(the term harms) | 0.84 | 0.32 | 0.025 | < 0.001 | < 0.001 |

- **Match 2's recalibration, fixed before any of its games** (Commands, step 5):
  1. **Fit control first.** `scripts/fit_value_recal.py` (banked sha256 `37e27420…`, at the PIN,
     `--device cuda`) on the pilot's Zq toward the SL base, on the pilot collection's hold-out
     (`--holdout-frac 0.05`), must reproduce match 8's banked (T, b) within 1e-3 each, as #1491's
     fit control does (GPU bf16 forwards are not bit-reproducible). If not, match 2 does not run.
     The control also checks WHAT was fitted (revision 6): the net is the pilot's Zq (`ad16d2f1…`),
     the reference the SL base (`a39f900c…`), the collect the pilot's (`61fc049f…`), 416 hold-out
     games, device cuda. Checked on match 8's banked fit: passes.
  2. **The fits.** The same command on Zq_sib AND on `Zq_main`: toward the SL base's value, the same
     416 hold-out games (which no cell trained on). Each (T, b) is applied exactly as match 8
     applied its own (`--candidate-value-recal` / `--reference-value-recal`, own value net).
  3. **Registered stop: the bounds, a SANITY stop, not a statistical bound.** ln T in
     [2 ln T1, −ln T1] = [−0.703, +0.352] (i.e. T in [T1², 1/T1] = [0.4951, 1.4212]) and b in
     [2 b1, −b1] = [−0.4542, +0.2271], with (T1, b1) match 8's fit. Derivation: the term
     is level-invariant, so Zq_sib's correction should lie near Zq's. The bounds allow up to twice
     Zq's correction, or an overshoot past the identity by Zq's whole correction. Outside them, the
     fit is not the registered procedure's regime and **match 2 stops before any game**; match 1
     is unaffected. **The same bounds apply to `Zq_main`'s fit**, and each bounds check requires the
     fit to be of the named net (its `model.pt` sha256) with the same reference, collect, 416 games
     and device. The bounds stay centred on the identity, not on Zq's fit (the reviewer's argument:
     re-centring would put the identity on the boundary, and a term that sharpens discrimination
     plausibly needs LESS correction).
  - Both gates are `SB/stage_b/recal_gate.py` v2 (tested: 20 fabricated cases; eleven mutants
    killed, including each unchecked field of what was fitted, in either mode).
  - **Reported beside the result (report-only):** both seats' `cross_entropy_identity` →
    `cross_entropy_fitted` from their fit files.
  - This supersedes revision 4's "no recalibrated readout" for match 2: match 2 is its own
    recalibrated match, fitted before its games. **The ban on re-reading a RAW pool as
    recalibrated stays**: a recalibrated player is a different player, so no raw game is ever
    re-read as its game.
- **Report-only contrast** (unchanged; decides nothing): each cell against its family's banked
  result vs the base, with the MDEs P 13.1, Zq 21.1, S 25.6 Elo (SE of the difference ≈ 4.7 /
  7.5 / 9.1).

**What each outcome licenses:**

| Reading | Licenses |
|---|---|
| the term adds, P | **NOMINATES the term for the P family** (P + sibling at that w). Nothing is adopted. Adoption against the base is a later record: P_sib vs the SL base. **P_sib > P0, together with P0 ≈ +2.2 vs base, does not license "P_sib improves on the base"**: the two intervals do not chain into one bound, and P0's +2.2 [−4.2, +8.6] does not exclude 0 |
| the term adds, Zq | **NOMINATES the term for the Zq family** (Zq + sibling + per-net recalibration at that w). Adoption is a later record: the term inside the generation-2 recipe (#1491's line), never this match's result alone |
| the term adds, S (fallback only) | The term repairs some of a known-damage recipe. Nominates nothing for adoption; S stays known damage until a match against the base says otherwise |
| the term harms, any family | **Retires the term at that w for that family's recipe.** For Zq, a harm reading may partly be worse calibratability under a 2-parameter recalibration (part of the treatment; see Confounds). For P, the reading is "harm; attribution (shape vs content) unread": the P family leaves the value's level and draw mass free, and the shape numbers are report-only. It is NOT "sibling training is not the fix" |
| unread: CI contains 0 | Nothing. The interval contains 0; it is consistent with true effects up to ≈ 13 Elo either way. This design had 80% power only for \|effect\| ≥ ≈ 9 |

**"Sibling training is not the fix" is not a reachable conclusion from Stage B**, in any family:
a head-to-head tests one w against one counterpart. For a family it is reachable only from
Stage A: every cell of the family is VOID, vetoed or inert.

**Competing nominations across the value lines (review item 9).** Three records can nominate a
value recipe: the anchor screen (≤ 2 matches), match 8 (1) and this screen (≤ 2). One already
exists: the anchor screen's S + `--kl-wdl` 7.0, "adoptable" (+5.8 [−0.5, +12.2]), beside match 8's
recalibrated Zq (+9.6); they are never ranked by point estimate. This screen's matches nominate a
TERM within a family, not a recipe against the base, so they do not compete with either on point
estimates. **If more than one recipe is nominated, the choice is made by a
fresh head-to-head under the next record, never by the best point estimate across screens.**

## Confounds

- **Same games, two targets.** The pairs come from the same games as the root rows. Hold-out
  here means hold-out GAMES (`is_holdout_game` at 0.05): the take-effect pairs, the learner's
  probe and the Stage A hold-out rows are all from those games, which no cell trains on. The
  sibling-ordering ruler rows are a different collection (leakage above).
- **Same search.** The sibling target is the same 64-sim search that S's root target came from;
  a cell that repairs S's ordering does it with the same teacher, not an independent one.
- **Policy-induced drift.** P0 showed the trunk moves the value with no value loss (KL 0.0072),
  and every cell trains the policy. The P family is compared with P0 (banked, and `P0_main`)
  for exactly that reason.
- **Free level in the P family.** See "Level and draw mass" above.
- **Code base.** Cells train at `TSHA`, the banked controls trained at the PIN. Guards (a)–(d)
  are what licenses comparing them.
- **Stage B opponents and the code base (revision 6).** Both opponents train at `TSHA` like their
  cells: match 1's is `P0_main`, match 2's is `Zq_main`, so each contrast is the term alone (plus
  GPU nondeterminism). The two paired code-base controls (P0 vs `P0_main`; Zq vs `Zq_main`, which
  exercises the WDL / Zq value-loss branch) are what let the banked PIN-trained nets describe them.
- **Match 2's recalibration (revision 6, review item 3).** The fit control proves the fit procedure
  is reproducible, not how good each fit is. Match 2's contrast therefore also includes (i) how well
  Zq_sib calibrates under a 2-parameter recalibration compared with `Zq_main`, and (ii) the two fits'
  unshared sampling error on the same 416 hold-out games. Neither is in the SE 3.37. **This is
  acceptable because the nominated recipe is "Zq + sibling + per-net recalibration", so
  calibratability is part of the treatment.** Both seats' `cross_entropy_identity` →
  `cross_entropy_fitted` are reported beside the result (report-only), and a "harms" reading may
  partly be worse calibratability.
- **One data-affecting change** per cell versus its banked family cell: the term (and its
  sidecar); same rows, batches (`presented.npz`), seed and optimizer.

## Cost and feasibility

**Registered step 1, before any training: the cost probe** (`SB/prelaunch/cost_probe.py`
revision 3, GPU, alone on the card; review item 2; revision 3 only adds the warm mode below). Revision 1 read 2 sidecar shards (about
3.4 GB, page-cached), so it could not see what a real cell pays. A cell random-reads about
255 × 2 child inputs per step from a 63 GB memmap on a rotational virtual disk, while the trainer
holds about 37 GB RSS on a 98 GB machine. Revision 2 does what a cell does, in production order:
- rows from `learn.load_window` over the whole window, with `prepare_targets`;
- the first 50 batches of the cell's own `plan_batches` plan, with its own seed;
- pairs from `sibling.load_sidecars` over ALL shards, with its sha256 pass timed as
  `sidecar_setup_s`, driven by the production `SiblingTerm`.

**Cold cache without root.** After `load_sidecars` (whose sha256 pass reads every file), the
probe calls `posix_fadvise(DONTNEED)` on every `x_a.npy` / `x_b.npy`. This needs no privilege.
It records the files' page-cache-resident bytes before and after with util-linux `fincore`.
The Windows host's cache below the WSL2 virtual disk is outside a guest's control, so "cold" is
cold in the guest only.

**What it reports:** per update, the wall after `torch.cuda.synchronize`; the seconds spent in
`SidecarPairs.child_inputs` (the host read); and `gpu_and_rest_s` = step − host read. Each is a
median over updates 11–50. It also reports peak reserved VRAM and peak RSS. It runs twice: with
the heaviest cell's flags (`Zq_sib_w366`), and with the same flags at `--w-sibling 0`.

**Gates:**
- **Abort the screen** if the sibling run's peak reserved memory exceeds **0.85 × the card's
  total memory** (headroom for allocator fragmentation and the CUDA context; the card runs
  nothing else during the screen). An abort is reported; no cell trains.
- **Projected wall per sibling cell** = 419 s + `sidecar_setup_s` + 1,412 × the sibling step
  median. The 419 s is the pilot Zq cell's measured fixed overhead: its process wall (859 s)
  minus its training loop's `wall_seconds` (440 s).
- **Hold for the owner** if the projected training total exceeds 2 × the estimate below.
- **Realised-wall gate (registered now).** The first sibling cell that exits 0 has its process
  wall compared with its projection. **If the wall exceeds 1.5 × the projection, the screen
  stops** (`SB/HOLD_WALL`) and the owner decides.
- **Per-cell budget check, on EVERY sibling cell (revision 4, review item 6).**
  - Why it is needed. The 1.5× gate covers only the first sibling cell. Residency after a cell's
    own sha pass is measured once, in step 1b (reported, not gated). Later cells could still
    degrade unseen, for example if the page cache is squeezed.
  - The check. After each sibling cell exits 0, its trainer's own `summary.wall_seconds` (the
    training loop) is compared with the loop budget = 3,367 − 419 − `sidecar_setup_s` (from the
    warm probe), where 3,367 s is the per-cell budget that keeps training under the HOLD limit.
  - **If it exceeds the budget, the screen stops before the next cell** (`SB/HOLD_BUDGET`) and the
    owner decides. The check costs one JSON read per cell.
- **The host read is handled by a page-cache warm-up** (revision 3, below), not by a trainer code
  change. No background reader is built.

**CPU smoke of revision 2** (`COST_PROBE_CPU_SMOKE=1`, 2 steps, Zq + w 366, 2026-11-21):
it exited 0 and ran the real plan: 722,742 rows, 1,412 batches in the plan, the production
`SiblingTerm` on all 78 sidecar files. Measurements:
- `load_window` took 353 s; `load_sidecars`, including its sha256 pass, took 68 s.
- The files held 20.0 GB in the guest page cache before `fadvise` and 0 after. Cold is
  achievable without root.
- **The cold host read was 1.60 s per step** (median of the 2 steps). It does not depend on the
  device. The registered estimate below assumed 0.19 s.
- Peak RSS 61.7 GiB (CPU: the model, its activations and the rows all on the host). Wall 12.5 min.

This proves the logic, not the GPU timing. Read cold, the HOLD rule would fire: a cell would be
about 419 + 68 + 1,412 × (0.75 − 0.19 + 1.6) s ≈ 59 min, and training about 9.0 GPU-h, over
2 × 4.3. The page cache cannot hold the whole 63 GB sidecar beside the trainer's rows. Hence
the warm-up.

### Revision 3: warm the page cache with only the bytes the cells read

**One touched set for all nine sibling cells (read at `9c0ff07dc`).** At step t a cell reads the
child inputs of every pair of the first `--sibling-rows` (128) pair-carrying rows of plan batch t
(`SiblingTerm._batch_pairs` on `ScheduledSampler.served[-1]`, which is the plan's own row array;
the sampler's RNG draws only the mirror). The plan is `plan_batches` over the window's row ids
with the rng `seed_for(seed, "train", 1)`, and each row's pairs were fixed when the sidecar was
materialised (RNG keyed by seed, game and ply). Neither w nor any value flag enters either. All
nine sibling cells share the collect, seed, batch size, hold-out split, `--sibling-rows` and
sidecar, so they read exactly the same bytes in the same order. `P0_main` reads none.

**`SB/prelaunch/warm_sidecar.py`** (CPU; imports the production functions and re-implements no
sampling):
- `plan` derives the touched list. It builds the row ids with `generation_shards` /
  `shard_row_meta` / `holdout_mask` (as `load_window` does, without loading the rows), the plan
  with `plan_batches`, and the pairs with `load_sidecars` and `SiblingTerm._batch_pairs` itself.
  It then checks the plan against the pilot's `presented.npz`, the batches every cell must
  reproduce (`train_check.py`). **They are equal**, or it refuses.
- `warm` reads every touched range twice in a row, file by file in offset order. It refuses if
  `MemAvailable` < touched bytes + 16 GiB.
- `check` measures the residency of the touched pages with `mincore(2)`, and exits 3 below 0.99.
- `bench` times the production `SidecarPairs.child_inputs` per step, cold and then warm.

**Measured (CPU, 2026-11-21, at `9c0ff07dc`; `SB/prelaunch/warm/`).** These ran with the
value-anchor screen's trainer resident: about 26 GB used, `MemAvailable` 72.5–74 GB of 98 GB.

| Quantity | Value |
|---|---|
| Plan | 1,412 batches; reproduces `presented.npz` exactly |
| Pairs read per cell | 358,198, each exactly once (distinct = total); median 254 per step |
| Distinct child inputs | 716,396 of the sidecar's 2,980,354 (24.0%) |
| Touched bytes | 16.05 GB (4,179,540 pages of 4 KiB = 17.1 GB); range-list sha256 `c54a78722450…` |
| `plan` | 83 s, 0.29 GB peak RSS |
| Cold read per step (`child_inputs`, steps 0–49, after DONTNEED) | 0.88 s median (0.95 mean) |
| One read pass, from cold | 61 s, **87.5% resident**; read per step afterwards 0.31 s (steps 50–99) |
| Two read passes per file, from cold (the registered `warm`) | **95 s, 99.999% resident** |
| After one `load_sidecars` sha256 pass (which every cell runs before step 1) | **89.3% resident**; read per step 0.148 s median (steps 100–149) |

- **Why two passes.** A page read once lands on the kernel's inactive file list. With a large
  active list (about 55 GB active against 6 GB inactive, measured), a single 16 GB stream evicts
  its own pages before they are used. A second read of each file's ranges promotes them to the
  active list.
- **Why a gate per cell, and why the warm probe.** Each cell's own `load_sidecars` sha256 pass
  reads all 63 GB and evicts part of the warmed set: about 11% on CPU. The Windows host's own
  cache below the WSL2 virtual disk is outside a guest's control too. So the screen measures
  residency before every cell and measures the real sequence (warm, then the sha pass, then the
  steps) with the GPU probe, instead of assuming either.

**Registered warm-up gates (all in the commands):**
- **Step 1a:** `warm_sidecar.py plan` only. Its range-list sha256 must equal `c54a78722450…` and
  its plan check must pass. Nothing is warmed in step 1a, so the cold probe that follows runs on a
  cold cache (revision 4, review item 5: the text now matches the commands).
- **Step 1b:** the cost probe is run cold (report only). Then the FIRST `warm`, then `check`
  ≥ 0.99, then the WARM probe (`COST_PROBE_CACHE=warm`, the same 50 batches). **The warm probe
  decides.** After it, `check` reports the residency left by the probe's own sha pass. That is the
  one measurement of residency after a sha pass; it is reported, not gated. Its host-read median must be
  ≤ host_read_max = (3,367 − 419 − `sidecar_setup_s`) / 1,412 − `gpu_and_rest_s`, both read from
  the warm probe. 3,367 s is the per-cell budget that keeps training under the registered HOLD
  limit: (2 × 4.3 × 3,600 − 653) / 9. (Revision 6: `Zq_main` reads no sidecar, so it is added to
  the HOLD limit once, at the pilot Zq cell's 859 s process wall, and the per-sibling-cell budget
  is unchanged.) With the estimate's 0.56 s step and 68 s setup,
  host_read_max ≈ 1.48 s; the CPU evidence after a sha pass is 0.15 s. Otherwise HOLD, as
  registered. The projection uses the warm probe.
- **Before EACH sibling cell:** `check` ≥ 0.99, or `warm` again and re-check. A second failure
  stops the screen (`SB/HOLD_WARM`).
- The realised-wall gate on the first sibling cell stays. The per-cell budget check
  (`wall_seconds`) runs after every sibling cell.

**Estimate until the probe reads** (the doc's "about double", plus 0.19 s host read per step):
pilot step ≈ 0.31 s (the Zq cell's 440 s / 1,412) → sibling step ≈ 0.75 s. Per sibling cell:
419 s + ≈ 150 s sidecar verification + 1,412 × 0.75 s ≈ 27 min.

| Item | GPU-h |
|---|---|
| Cost probe (three runs: cold, warm and no-term, each loading the whole window) | ≈ 0.45 |
| Training: 9 sibling cells × ≈ 27 min + `P0_main` ≈ 11 min + `Zq_main` ≈ 14 min (the pilot Zq cell's 859 s; revision 6) | ≈ 4.5 |
| Stage A: 11 nets through the instruments, ≈ 6 min each, plus the take-effect statistic on 15 nets | ≈ 1.3 |
| Stage B: ≤ 2 head-to-heads × ≈ 1.45, plus match 2's fit control and two fits (Zq_sib, `Zq_main`) ≈ 0.15 | ≤ ≈ 3.05 |
| **Total** | **≤ ≈ 9.3** |

**Memory per step (review item 10).** Every step runs alone; pre-launch CPU work was run one job
at a time.

| Step | Peak RSS |
|---|---|
| Guard (a), one run (CPU) | 37.2 GB (measured, `guard_a_*.err`) |
| Step-0 gradient checks: `grad_scale.py` / the reviewer's `sib_grad_indep.py` / `coherence.py` (first batch) | not recorded / 25 GB / 23.6 GiB |
| Cost probe (loads the whole window, like a cell) | 61.7 GiB on the CPU smoke (model and activations on the host); on GPU, expected near the trainer's ≈ 37 GB |
| Training, per cell | ≈ 37 GB (the trainer's rows; the reviewer's figure) |
| `check_load.py` | one model and 512 rows; not measured |
| `warm_sidecar.py` | 0.29 GB (`plan`) / 2.2 GB (`bench`) RSS, plus 16 GB of page cache that the kernel can reclaim |
| Take-effect statistic (`sibling_holdout.py`; ≈ 3.4 GB of child inputs held in RAM) | 3.4 GB (CPU smoke, 4,096 pairs; the full run holds ≈ 3.4 GB more of child inputs) |

## Commands

**Step 0: preconditions and worktrees.**

```bash
SB=$HOME/chess-artifacts/operations/rl-run1-sibling; P=$HOME/chess-artifacts/operations/rl-run1-pilot
AV=$HOME/chess-artifacts/operations/rl-run1-value-anchor
PIN=$(cat $P/PIN.txt); PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
TSHA=9c0ff07dc0858c73def20678ce274c928d83dcec; TW=$HOME/projects/chess-worktrees/rl-run1-sibling-${TSHA:0:9}
PY=$HOME/projects/chess/.venv/bin/python; SL=$P/sl_base.pt
SIDE=$HOME/chess-artifacts/operations/rl-claude-20261119/sibling-pairs-pilot/v2
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
SF=$HOME/local_stockfish_linux_latest/stockfish/stockfish-ubuntu-x86-64-bmi2
TAKE_EFFECT_TEST="tests/test_simple_rl_sibling.py::test_the_sibling_term_trains_the_value_head_through_the_trainer_hook"
[ -e $AV/offline/cells/DONE ] && [ -e $AV/offline/control/CONTROL_PASS ] || exit 92   # anchor screen finished, instruments proven
(cd $SB/stage_a && sha256sum -c --quiet SHA256SUMS) && (cd $SB/prelaunch && sha256sum -c --quiet SHA256SUMS) || exit 93
[ -d $TW ] || git -C $PINW worktree add --detach $TW $TSHA             # a new detached worktree; no existing tree moves
cd $TW && [ "$(git rev-parse HEAD)" = "$TSHA" ] || exit 96
$PY scripts/build_production_extensions.py || exit 1
# guard (b): encoder paths unchanged; shard.py changed only by #1477's optional field
git diff --quiet 0c795fb7a $TSHA -- chess_anti_engine/encoding chess_anti_engine/mcts/_mcts_tree.c chess_anti_engine/moves \
  chess_anti_engine/replay/packed_zarr.py chess_anti_engine/model chess_anti_engine/uci chess_anti_engine/inference.py || exit 1
[ "$(git diff 0c795fb7a $TSHA -- chess_anti_engine/replay/shard.py | sha256sum | cut -d' ' -f1)" = \
  418b6e96e29716160dfaf439b632896191be501018a15f4412b2686d8d2901ff ] || exit 1
# guard (a): banked EQUAL at the training pin itself
grep -qx "P0 EQUAL to guard_a_P0_0c795fb7a" $SB/prelaunch/guard_a_9c0ff07dc.verdict || exit 1
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/test_simple_rl_sibling*.py "$TAKE_EFFECT_TEST" \
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
  [P0_main]="--w-wdl 0 --w-sibling 0"
  [Zq_main]="$Z_V --w-sibling 0")                              # revision 6: Zq's flags at TSHA, no term, no sidecar
```

The `CELL` table must agree with `SB/stage_a/cells.json` (the checker's source of expected
values); `train_check.py` fails any cell whose realised w, w_wdl or search fraction differs.

**Step 1a: the warm-up's touched set** (CPU; can run while the anchor screen trains, subject to
its `MemAvailable` floor). It derives and checks the plan and defines `warm_ok`; it warms
nothing:

```bash
WARM="$PY $SB/prelaunch/warm_sidecar.py"
(cd $TW && PYTHONPATH=. $WARM plan $SB/warm) || exit 1          # refuses unless the plan reproduces presented.npz
[ "$($PY -c "import json;print(json.load(open('$SB/warm/plan.json'))['range_list_sha256'])")" = \
  c54a78722450f8adca516cfef99aeddee752d6bcd1016bc4a379d3932a77b345 ] || exit 1
warm_ok() { (cd $TW && PYTHONPATH=. $WARM check $SB/warm) || { (cd $TW && PYTHONPATH=. $WARM warm $SB/warm) && \
  (cd $TW && PYTHONPATH=. $WARM check $SB/warm); }; }
```

**Step 1b: the cost probe** (GPU alone): cold (report only), then the first warm-up, then warm
(decides), then no term.

**GPU check (post-merge addendum).** This screen's commands have no GPU guard of their own: the
operator runs them. So before step 1b, and again before each later GPU step block, the operator
runs `SB/operator/operator_gpu_check.sh IDLE_MIB`, with IDLE_MIB the GPU's measured idle
`memory.used` (this machine idles at about 3,382–3,452 MiB: the Windows side of WSL2). It is the
generation-2 bank's guard v3 rule: IDLE_MIB an integer ≤ 4,096; every memory line within
IDLE_MIB ± 512 MiB; utilisation < 10 % over 5 samples one second apart; `ps` readable and no
`arena_standard.py` / `simple_rl_loop.py` / `match_vs_uci.py` running; fail-closed, read-only.
`SB/operator/` (read-only, `SHA256SUMS` `029a3eb8…`; script `0e54fc624bb2…`): 28 cases under a
fake HOME with stubbed `nvidia-smi` / `ps` / `sleep`, one read-only case running its two probe
commands against the real card (integers: 3,451 MiB, 0 %); twelve mutants, one per clause, each
killed.

```bash
$SB/operator/operator_gpu_check.sh $IDLE_MIB || exit 8   # IDLE_MIB: the operator's measured idle memory.used
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u $SB/prelaunch/cost_probe.py $SB/cost_probe_sib_cold.json 50 $COLLECT $TRAIN ${CELL[Zq_sib_w366]} || exit 1
(cd $TW && PYTHONPATH=. $WARM warm $SB/warm) && warm_ok || exit 1
CUDA_VISIBLE_DEVICES=0 COST_PROBE_CACHE=warm PYTHONPATH=. $PY -u $SB/prelaunch/cost_probe.py $SB/cost_probe_sib.json 50 $COLLECT $TRAIN ${CELL[Zq_sib_w366]} || exit 1
(cd $TW && PYTHONPATH=. $WARM check $SB/warm)        # residency after the probe's own sha pass: report only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u $SB/prelaunch/cost_probe.py $SB/cost_probe_base.json 50 $COLLECT $TRAIN $Z_V --w-sibling 0 || exit 1
$PY - $SB <<'PYEOF' || exit 1
import json, sys
s, c, b = (json.load(open(f"{sys.argv[1]}/cost_probe_{k}.json")) for k in ("sib", "sib_cold", "base"))
assert s["cache"] == "warm" and c["cache"] == "cold"
if s["peak_reserved_gib"] > 0.85 * s["device_total_gib"]:
    sys.exit(f"ABORT: peak reserved {s['peak_reserved_gib']:.1f} GiB > 0.85 x {s['device_total_gib']:.1f}")
budget = (2 * 4.3 * 3600 - 653) / 9                  # per-cell seconds that keep training under the HOLD limit
read_max = (budget - 419 - s["sidecar_setup_s"]) / 1412 - s["gpu_and_rest_s_median_11_on"]
cell_s = 419 + s["sidecar_setup_s"] + 1412 * s["step_s_median_11_on"]
proj = (9 * cell_s + 653 + 859) / 3600               # + P0_main and Zq_main at the pilot P0 / Zq cells' process walls
print(f"warm sibling step {s['step_s_median_11_on']:.3f}s (host read {s['host_read_s_median_11_on']:.3f}s, max "
      f"{read_max:.3f}s); cold host read {c['host_read_s_median_11_on']:.3f}s; no-term step {b['step_s_median_11_on']:.3f}s; "
      f"projected cell {cell_s:.0f}s, training {proj:.2f} GPU-h")
json.dump({"cell_s": cell_s, "training_gpu_h": proj, "host_read_max_s": read_max}, open(f"{sys.argv[1]}/cost_projection.json", "x"))
if s["host_read_s_median_11_on"] > read_max or proj > 2 * 4.3 + 859 / 3600:   # Zq_main (no sidecar) counted once
    sys.exit("HOLD for the owner: the warm host read or the projected training exceeds the registered limit")
PYEOF
```

**Step 2: training, the realised-wall gate, one crash rerun per cell, one take-effect check.**

```bash
train() { date -Is > $SB/arm_$1.started
  PYTHONPATH=. $PY -u scripts/simple_rl_loop.py run --run-dir $SB/arm_$1 --collect-from $P/collect \
    --generations 1 $COLLECT $TRAIN ${CELL[$1]} > $SB/arm_$1.log 2>&1; echo "$? $(date -Is)" > $SB/arm_$1.exit; }
wall_gate() { [ -e $SB/wall_gate.json ] && return 0
  $PY - $SB $1 <<'PYEOF'
import json, sys
from datetime import datetime
sb, c = sys.argv[1:3]
t0 = datetime.fromisoformat(open(f"{sb}/arm_{c}.started").read().strip())
t1 = datetime.fromisoformat(open(f"{sb}/arm_{c}.exit").read().split()[1])
wall, proj = (t1 - t0).total_seconds(), json.load(open(f"{sb}/cost_projection.json"))["cell_s"]
json.dump({"cell": c, "wall_s": wall, "projected_s": proj, "ratio": wall / proj}, open(f"{sb}/wall_gate.json", "x"))
if wall > 1.5 * proj:
    open(f"{sb}/HOLD_WALL", "x").write(f"{c}: wall {wall:.0f}s > 1.5 x projected {proj:.0f}s\n")
    sys.exit(f"HOLD: {c} wall {wall:.0f}s > 1.5 x projected {proj:.0f}s")
PYEOF
}
budget_check() {   # every sibling cell: the trainer's own loop wall vs the per-cell loop budget (revision 4)
  $PY - $SB $1 <<'PYEOF'
import json, sys
sb, c = sys.argv[1:3]
w = json.load(open(f"{sb}/arm_{c}/gen_0001/train/metrics.json"))["summary"]["wall_seconds"]
setup = json.load(open(f"{sb}/cost_probe_sib.json"))["sidecar_setup_s"]
budget = (2 * 4.3 * 3600 - 653) / 9 - 419 - setup
with open(f"{sb}/budget_check.jsonl", "a") as fh:
    fh.write(json.dumps({"cell": c, "wall_seconds": w, "loop_budget_s": budget}) + "\n")
if w > budget:
    open(f"{sb}/HOLD_BUDGET", "x").write(f"{c}: loop wall {w:.0f}s > budget {budget:.0f}s\n")
    sys.exit(f"HOLD: {c} loop wall {w:.0f}s > budget {budget:.0f}s")
PYEOF
}
after_cell() { [ "$1" != P0_main ] && [ "$1" != Zq_main ] && [ "$(cut -d' ' -f1 $SB/arm_$1.exit)" = 0 ] || return 0
  wall_gate $1 && budget_check $1; }
CELLS="Zq_sib_w41 Zq_sib_w122 Zq_sib_w366 P_sib_w23 P_sib_w69 P_sib_w207 S_sib_w28 S_sib_w83 S_sib_w248 P0_main Zq_main"
for c in $CELLS; do
  case $c in P0_main|Zq_main) ;; *) warm_ok || { echo "$c: touched pages < 99% resident after a re-warm" > $SB/HOLD_WARM; exit 1; };; esac
  train $c
  after_cell $c || exit 1
done
for c in $CELLS; do
  [ "$(cut -d' ' -f1 $SB/arm_$c.exit)" = 0 ] && continue
  [ -z "$(compgen -G "$SB/arm_$c.crash1*")" ] || continue
  for f in "" .log .exit .started; do [ ! -e $SB/arm_$c$f ] || mv $SB/arm_$c$f $SB/arm_$c.crash1$f; done
  case $c in P0_main|Zq_main) ;; *) warm_ok || { echo "$c: touched pages < 99% resident after a re-warm" > $SB/HOLD_WARM; exit 1; };; esac
  train $c
  after_cell $c || exit 1
done
$PY $SB/stage_a/train_check.py $SB
```

**Step 3: guard (d), the main-written controls load identically at the PIN** (revision 6: both
`P0_main` and `Zq_main`, the two Stage B opponents).

```bash
for K in P0main Zqmain; do C=$SB/arm_${K/main/_main}/gen_0001/train/model.pt
  (cd $PINW && CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY $SB/prelaunch/check_load.py $C $SB/check_load_${K}_pin.npz) || exit 1
  (cd $TW && CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY $SB/prelaunch/check_load.py $C $SB/check_load_${K}_main.npz) || exit 1
  $PY $SB/prelaunch/check_load.py --compare $SB/check_load_${K}_pin.npz $SB/check_load_${K}_main.npz || exit 1
done
```

**Step 4: Stage A** (the instruments at the PIN, the take-effect statistic at `$TW`;
self-enforced one rerun, by running the same line again):

```bash
$SB/stage_a/run_stage_a.sh $TW $TSHA || exit 1      # writes $SB/offline/cells/stage_a_decision.json -> stage_b
```

**Step 5: Stage B (revision 5: head-to-heads; revision 6: `Zq_main`, the verdict script).** For the i-th entry `X` of `stage_b` (i = 1, 2),
its family `FAM` = `stage_b_specs[X].family`, seed `SEED` = 2026112193 (P, or an S fallback) or
2026112194 (Zq):

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup
echo "7c8fd065865f3d5abcd8e9eb6f7fc1d48e882dfcfb88bbaa92d8696ce4a9510d  $F/run_followup_match.sh" | sha256sum -c --quiet || exit 93
(cd $SB/stage_b && sha256sum -c --quiet SHA256SUMS) || exit 93
DEC=$SB/offline/cells/stage_a_decision.json
[ "$($PY -c "import json;print(json.load(open('$DEC'))['stage_b'][$i-1])")" = "$X" ] || exit 91
FAM=$($PY -c "import json;print(json.load(open('$DEC'))['stage_b_specs']['$X']['family'])")
[ ! -e $SB/arm_$X/VOID ] || exit 91
M=$SB/arm_$X/gen_0001/train/model.pt
RC8=0.7036189636786957,-0.22711530923843357                                    # match 8's banked (T, b)
case $FAM in
  P)  OPP=$SB/arm_P0_main/gen_0001/train/model.pt; SEED=2026112193; RA=(); RB=() ;;
  S)  OPP=$P/arm_S/gen_0001/train/model.pt;        SEED=2026112193; RA=(); RB=() ;;
  Zq) OPP=$SB/arm_Zq_main/gen_0001/train/model.pt; SEED=2026112194                 # revision 6: Zq_main
      [ ! -e $SB/arm_Zq_main/VOID ] || exit 91
      # match 2's recalibration, fixed before any game: fit control, both fits, both bounds stops (GPU, alone)
      echo "37e2742011c48be427a32ad2ecfb6c376228d484f8215b6650e659f150c3d3a5  $PINW/scripts/fit_value_recal.py" | sha256sum -c --quiet || exit 93
      fit() { (cd $PINW && PYTHONPATH=. $PY scripts/fit_value_recal.py --net $1 --reference $SL \
        --collect-dir $P/collect/gen_0001/collect --holdout-frac 0.05 --device cuda --out $2 > $2.log 2>&1); }
      fit $P/arm_Zq/gen_0001/train/model.pt $SB/stage_b_fit_control_Zq.json || exit 1
      $PY $SB/stage_b/recal_gate.py control $SB/stage_b_fit_control_Zq.json $RC8 > $SB/stage_b_fit_control.verdict \
        || { echo "fit control failed: match 2 does not run" > $SB/HOLD_FIT; exit 1; }
      for N in $X Zq_main; do NM=$SB/arm_$N/gen_0001/train/model.pt
        fit $NM $SB/stage_b_recal_$N.json || exit 1
        $PY $SB/stage_b/recal_gate.py bounds $SB/stage_b_recal_$N.json $RC8 $(sha256sum $NM | cut -d' ' -f1) \
          > $SB/stage_b_recal_bounds_$N.verdict \
          || { echo "$N (T, b) outside the registered bounds (or not its fit): match 2 stops before any game" > $SB/HOLD_BOUNDS; exit 1; }
      done
      RCX=$($PY -c "import json;print(json.load(open('$SB/stage_b_recal_$X.json'))['cli'])")       # the fits' own "T,b"
      RCO=$($PY -c "import json;print(json.load(open('$SB/stage_b_recal_Zq_main.json'))['cli'])")
      RA=(--candidate-value-recal $RCX --reference-value-recal $RCO)            # seat A: candidate = X
      RB=(--candidate-value-recal $RCO --reference-value-recal $RCX) ;;         # seat B: candidate = Zq_main
esac
for O in $SB/smoke $SB; do n=$([ $O = $SB ] && echo 3200 || echo 4)
  $F/run_followup_match.sh $O sib_$X 1 A $SEED $n --candidate $M --reference $OPP "${RA[@]}" || exit 1
  $F/run_followup_match.sh $O sib_$X 1 B $SEED $n --candidate $OPP --reference $M "${RB[@]}" || exit 1
  d=$O/arenas/sib_$X
  $PY - $d/results.jsonl sib_$X "$PIN" $M $OPP <<'PYEOF' || exit 1
import hashlib, json, sys
res, lab, pin, m, opp = sys.argv[1:6]
sha = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (m, opp)}
want = {f"{lab}_1_A": (sha[m], sha[opp]), f"{lab}_1_B": (sha[opp], sha[m])}
rows = [json.loads(x) for x in open(res) if x.strip()]
seen = {r["label"] for r in rows if r["label"] in want}
bad = [r["label"] for r in rows if r["label"] in want and
       (r["git_sha"] != pin or (r["candidate_sha256"], r["reference_sha256"]) != want[r["label"]])]
if seen != set(want) or bad:
    sys.exit(f"result rows: seen {sorted(seen)}, wrong git_sha or net sha256 in {bad}")
PYEOF
  if [ $FAM = Zq ]; then
    $PY $SB/stage_b/check_both_seats.py $d sib_${X}_1_A $RCX $RCO || exit 1
    $PY $SB/stage_b/check_both_seats.py $d sib_${X}_1_B $RCO $RCX || exit 1
  fi
  (cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $M \
    --a-games $d/sib_${X}_1_A.games.jsonl --b-games $d/sib_${X}_1_B.games.jsonl \
    --a-result $d/results.jsonl --b-result $d/results.jsonl \
    --a-label sib_${X}_1_A --b-label sib_${X}_1_B --out $d/pool.json > $d/pool.log 2>&1) || exit 1
done
$PY $SB/stage_b/stage_b_verdict.py $SB/arenas/sib_$X/pool.json > $SB/arenas/sib_$X/verdict.json || exit 1   # the label, by script
if [ $FAM = Zq ]; then for N in $X Zq_main; do   # report-only, beside the result
  $PY -c "import json;r=json.load(open('$SB/stage_b_recal_$N.json'));print('$N', r['cross_entropy_identity'], '->', r['cross_entropy_fitted'])"
done > $SB/arenas/sib_$X/fit_cross_entropy.txt; fi
```

- The first pass is the 4-game smoke (2 openings); the real match runs only if it exits 0.
- **Net identity:** the result-row check requires every row of both seats to carry `git_sha` =
  `$PIN` and the intended candidate and opponent sha256 (the cell, and `P0_main`, `Zq_main` or
  the pilot's S). `P0_main`'s and `Zq_main`'s identity at the PIN is also guard (d)'s `check_load` comparison
  (step 3); a cell's checkpoint is the file `train_check.py` passed.
- **Seats:** match 1 (and an S fallback) is plain on both seats, so no hybrid readback applies.
  Match 2 reads back both seats of each run with `check_both_seats.py`: each seat's "HYBRID built"
  line shows its own value net and exactly its registered (T, b), the result row carries the same,
  and the engine names differ in every game.
- The stored decision's `stage_b` already excludes VOID, VETOED and INERT cells, and the control
  gate.
