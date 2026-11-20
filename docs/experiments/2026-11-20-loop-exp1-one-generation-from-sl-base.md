# Loop experiment 1: one self-play generation from the SL base (prospective)

**Status:** DRAFT for owner approval. Nothing has been launched: no collection, no
training, no arena. This record exists so the experiment can be judged, and its
rule fixed, before any compute is spent (CLAUDE.md "Experiments" rule 1). It is
registered only once the owner approves it and the decisions below are filled in.
Nothing launches until (i) the owner's D1–D9 are recorded in this record by a
follow-up commit, (ii) the snapshot/collect-from driver PR is merged (otherwise the
4,000-game fallback applies), and (iii) pre-launch checks L1–L5 pass.
Revised 2026-11-20 after an independent review (findings M1–M7, m1–m11).

**Prerequisites:**
- **Collector fixes, on main since `b410456b4`:**
  - #1427: the collect evaluator cap is sized to the search's leaf buffer.
  - #1428: the C buffer-full fallback is replaced by flush-and-resume, with a counter
    (ABI 11).
- **Extensions rebuilt** in the run's tree at or after `b410456b4`.
- **Every collect's `stats.json` must show `buffer_full_flushes == 0` and
  `eval_max_batch_realised == leaf_rows_required`.** A collect that does not is void.
- **Snapshot / collect-from prerequisite PR (not yet merged; being commissioned).**
  Driver support for:
  - gates-only checkpoints at fixed update counts within a phase;
  - a native `--collect-from`-style option, so several training arms run on one
    sealed collection.

  Without it, D2 falls back to 4,000 games, and the multi-arm procedure in
  [Training](#training) applies.

Owner's goal, verbatim: "my goal is to get at least one or two steps of improvement
to show a RL loop can work."

Naming: "SL base" is the supervised starting net, sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0` (step 680,276).
F12, Q50, P1 and D1–D4 are as in the
[takeover record](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md).
The driver is `scripts/simple_rl_loop.py`
([docs](../simple_rl_loop.md)). Distance from the SL base is not a defect.

## Decisions for the owner

Approval can be one reply: "approve with defaults", or name the decisions that
change. Each decision is argued in the section it links to.

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| D1 | [Which arm is primary, given its prior](#prior-and-choice-of-primary) | **Arm P** (policy only): measured prior −10.9 (−27.7 to +5.9) on F12's rows, no named failure mechanism. The likely outcome is "unread" or "kill", which would register that one policy-only generation at this dose is neutral. | **Arm V** (value on the search value): possibly a better chance of a positive read, with a named risk (a flatter value, the mechanism of Q50's loss) |
| D2 | [Dose](#collection) | **16,000 games WITH snapshot gates** (needs the prerequisite PR) | **4,000 games** without snapshots (the fallback if the PR is not merged) |
| D3 | [Search budget of the stored targets](#collection) | **64 sims** | 256 sims at N/4 games, as a separate later experiment |
| D4 | [Softening used by Arm S](#arms) | **`--train-target-temp 2.0`** | `--policy-prior-mix 0.5 --prior-mix-source reference` (SL base) |
| D5 | [Primary arena layout](#deciding-yardstick) | **Seat-balanced:** half the games with the arm as candidate, half as reference, same openings; no separate base-vs-base control | One orientation plus a same-day base-vs-base control |
| D6 | [Positive control](#positive-control) | **Unconditional:** Arm V's recipe at 4,000 games from Q50 c12, played against Q50 c12. It cannot validate the primary: a pass is largely expected from fixing value calibration, and its dose is a quarter of the primary's | Skip it, saving about 3 GPU hours (2.6–3.4) |
| D7 | [Collect execution](#collection) | **Concurrency 128, `--collect-cuda-graphs` on** (after pre-launch check L4) | Graphs off |
| D8 | [Optimizer](#training) | **Fresh NAdamW, flat LR 3e-6, warmup 0**, as F12 | `--init-optimizer-state checkpoint` (Aurora): not recommended here |
| D9 | [Deciding N](#deciding-yardstick) | **Single fixed look at 3,200 games** (1,600 per orientation), read once | O'Brien–Fleming two-look: z 2.797 at 1,600, z 1.977 at 3,200 |

## Question and hypothesis

**Question.** Can ONE generation of self-play from the SL base produce a net that
beats the SL base head-to-head at 64 simulations, when the targets are chosen by the
diagnostics?

**Why one generation first.** Every earlier loop lost strength:
- F12 read −20.0 Elo on its own panel, and −34.9 on D4's instrument.
- Q50 read −63.2 on D4's instrument.

The diagnostics isolate what to change. A second generation is registered only after
a pass ([below](#second-generation-not-registered-here)).

**Hypothesis (primary, default D1).** Arm P will beat the SL base with a lower 95%
bound above 0. That arm is policy-only training, exactly once, on fresh self-play rows
from the large book.

The changes against P1:
- **Data:**
  - 153,842 TRAIN lines with no line reused, against P1's 192 recurring openings with
    18% exact repeats;
  - a 2-move book in the mix (5%).
- **Dose:** larger.
- **Collector:**
  - the cap defect is fixed;
  - the driver's root-noise schedule (Gumbel 1.0 → 0.5 over moves 12–15);
  - the driver's move-temperature schedule (0.5 → 0.1 over 10 plies after the book);
  - concurrency 128 against F12's 16 slots.

These confound a P-versus-P1 comparison only. The registered question is against the
SL base, which none of them confound.

**What would refute it.** Arm P reads with an upper bound below 0. That would mean
one policy-only generation at this dose costs strength even on clean data.

### Prior and choice of primary

The prior for Arm P is not favourable, and the record says so before any data:
- **P1** is the same training regime, on F12's rows. It read −10.9 (−27.7 to +5.9)
  against the SL base.
- **The +849 Elo of D1** (64 sims against 1 sim) is search against the raw net. It is
  the size of what a loop could distil, not a training gain anyone has measured.
- **Chance of success under the single 3,200-game look** (per-pair SD 0.254):

  | True effect | Chance of a success read |
  |---|---|
  | 0 | 2.5% |
  | +10 Elo | 62% (36% at 1,600 games) |
  | +15 Elo | 92% |

- **Likely outcome:** "unread" or "kill". Then the experiment mainly registers that
  one policy-only generation at this dose is neutral or harmful on clean data. That is
  informative, but it is not the owner's goal.

**The case for Arm V as primary.**
- Outcome targets are implicated: P1 read +42.6 (+30.1 to +55.2) against F12 c12.
- The 64-sim stored search value has lower error against deep search than the raw net:
  MSE 0.0145 against 0.035 (D3).
- V is the only arm that can improve the value. D4 found the value output, not the
  policy, carries most of the measured loss.

**The case against V.**
- The SL base is over-confident against the 800-sim survivor Q (value-recalibration
  study: best temperature fit T ≈ 1.4–1.55). Training toward the survivor value
  therefore moves the value flatter.
- D4 round 2 tied a flatter value to Q50's loss. Recalibrating it back toward the
  base's shape recovered 0.85 (0.50 to 1.18) of the value-output deficit.
- V's evidence is fit evidence, and fit has not transferred to Elo here before.

The default keeps P because its prior is measured and it has no named failure
mechanism. Choosing V is a legitimate owner call that accepts that named risk; the
G-T1 gate still catches a gross drift.

## Arms

All arms train on the SAME collected rows. They differ only in loss and target
settings, so collection is paid once and is not a confound between them.

| Arm | Train flags beyond the common set | Target, exactly | Evidence |
|---|---|---|---|
| **P** policy only | `--w-wdl 0` | stored 64-sim policy target | P1; D4 (Q50's loss nearly all in the value output: −55.2 with base policy + Q50 value, +4.3 with Q50 policy + base value) |
| **V** search value | `--w-wdl 1 --value-outcome-weight 0 --value-search-weight 1 --value-search-source survivor_q --value-draw-source net` | WDL = `search_value_wdl(root_survivor_q, draw = root_net_wdl[1])` (`simple_rl/learn.py`). Rows without a usable survivor Q fall back to the stored `search_wdl`. About 1% of rows carry the twofold-repetition exact zero (see Confounds). | P1 (outcome implicated); D3 (MSE 0.0145 against 0.035) |
| **S** softened policy | Arm P + `--train-target-temp 2.0` (D4) | `retemper_main_policy_target(stored target, temp=2)` (`train/losses.py`), applied in the loss | Stored targets 0.69 nats against the SL base's 1.12. The policy sharpened (1.117 → 0.787 by F12 c12) without better top-1 against deep search (0.667 → 0.667). In D3, temperature 2 gave 1.32 nats, the measured setting closest below the BT4 / Ceres band (1.49 / 1.53). |

D3 used the same `retemper_main_policy_target` function as `--train-target-temp`
(verified). All flags were read from `simple_rl_loop.py run --help`. No variant needs
new code.

**Multiplicity.** One pre-registered primary is decided at 95%. Two secondaries are
screened at 400 games (seat-balanced, 200 + 200). Only the better screen point
estimate is confirmed, at 1,600 games (seat-balanced) on fresh openings, and read at
97.5%: Bonferroni over the two that competed. Screen games are never pooled into the
confirmation. Head-to-head comparisons between arms are not registered.

## Collection

One collection from the SL base, at main ≥ `b410456b4`.

| Setting | This experiment | Driver default | F12 | Source / reason |
|---|---|---|---|---|
| Sims / topk / c_scale / c_visit | 64 / 16 / 0.1 / 50 | same | same | the F12 instrument and the D1 reads |
| Target q-rescale; tree reuse | on; on | same | same | stored targets include inherited visits (D3) |
| Root noise | Gumbel 1.0 → 0.5 over moves 12–15 | same | not re-read here | driver default |
| Move temperature | 0.5 → 0.1 over 10 plies after the book, then 0.1 | same | not re-read here | driver default |
| Openings | split 153,842 TRAIN / 4,000 EVAL; book mix 95% / 5%; `cycle` | same | 192 fixed starts | 18% exact repeats under F12's starts |
| Games | 16,000 with snapshots / 4,000 without (D2) | 2,000 | 192 per cycle | below |
| Concurrency | 128 (D7) | 64 | 16 slots | the only clean measured setting |
| Evaluator cap | auto (= `leaf_rows_required`) | auto | — | #1427 |
| `--collect-cuda-graphs` | on (D7), after L4 | off | — | documented bitwise identical; L4 checks it on this net |
| Max plies; tablebases | 450; strict rule50, production pair | 450 | 450 per game; strict | CLAUDE.md "Production Syzygy" |

**Dose (D2).**
- **Rows and updates.** A clean 128-game collect banked 12,372 rows, 96.7 per game
  (smoke `optimizer-start/mini/fix_c128_auto`). At that density:
  - 16,000 games ≈ 1.55M rows ≈ 5,700 updates at batch 256 after the 5% holdout;
  - 4,000 games ≈ 1,400 updates.
- **Against earlier loops.** P1 and F12 ran 948 updates over twelve cycles, and F12
  ran 69 per 192-game cycle (2026-11-18 record).
- **Why the larger dose is defensible.** No line is reused: 16,000 games use about
  10% of TRAIN. Every row is presented once. The evidence ties the earlier losses to
  the value loss and recurring openings, not to the number of updates.
- **What the larger dose risks.** No read exists at that many updates, and the
  policy-drift extrapolation below reaches KL 0.5–1.2. So the 16,000 default requires
  the snapshot gates; without them the dose falls back to 4,000 games, inside the
  measured range.

**Search budget (D3).** The evidence is mixed:
- the best 256-sim targets beat every transformed 64-sim target on mass, top-1 and
  regret (D3; for example cv0/cs0.5: mass 0.702, top-1 0.763, regret 0.014);
- the default 256-sim target does not beat the sharpest 64-sim cell on mass (0.674
  against 0.677), and the intervals overlap.

The default keeps 64 sims: it is the instrument of every banked read, and changing it
would add a second data-affecting change in one readout (rule 3).

**Throughput (unverified).** Clean collects at concurrency 128 measured 2,424–2,583
games/h, but on 128-game runs. Those two runs are effectively ONE collect: the same
128 games, with and without an explicit cap. The concurrency-256 reads (5,180–6,139
games/h) had the cap defect and are not used. The CUDA-graph speed-up is not banked
in a readable file and is not counted.

Estimate: about 6.2–6.6 h for 16,000 games, 1.5–1.7 h for 4,000.

## Training

**Common to every arm:**
- one phase, generation 1;
- `--presentations 1` (exact-once, shuffled), batch 256, `--mirror-prob 0.5`;
- `--lr 3e-6 --lr-floor 3e-6 --warmup-steps 0`;
- `--optimizer nadamw --optimizer-state fresh --init-optimizer-state fresh`;
- `--holdout-frac 0.05`, `--probe-rows 2048`;
- the same `--seed` everywhere.

The driver's `--lr` default is 3e-5, so 3e-6 is passed explicitly. The holdout is a
sha256 of the game id (`is_holdout_game`), so every arm holds out the same games.

**Optimizer (D8).** F12 used fresh NAdamW at flat 3e-6 with warmup 0 and showed no
first-step hit: this driver's trainer on F12's cycle-1 rows moves the probe by
|ΔQ| 0.018, KL 0.030 ([driver doc](../simple_rl_loop.md)). The `--warmup-start 0`
zero-first-update caveat applies only with `--warmup-steps > 0`.

The SL base's own optimizer state (Aurora, `mlp_out` scope) would change the
optimizer family, which is a second change in one readout.

**Several arms on one collection.** Prefer the native option from the prerequisite PR
once it is merged. Until then, lineage checks cannot prove a copied run dir: the
driver's `input_ref` stores paths as given, and `verify_stage` checks only in-run
inputs. So the manual procedure must record its own proof:
1. Collect generation 1 once in `collect/`.
2. Copy `collect/` (with its `run_config.json`) to `arm_P/`, `arm_V/`, `arm_S/`.
3. Record that `sha256_tree(arm_X/gen_0001/collect) == sha256_tree(collect/gen_0001/collect)`,
   and the same for `gen_0000`, with `simple_rl.common.sha256_tree`.
4. Run `train --generation 1` in each arm dir with the arm's flags and
   `--allow-config-change`, so each `run_config.json` records that arm's diff.
5. Record that `probe.npz` and the held-out row set have the same sha256 in every arm.

Pre-launch check L3 rehearses this on a tiny net, on CPU.

**Snapshots (with the prerequisite PR).**
- Each arm writes gates-only snapshots at 1,000, 2,000 and 4,000 updates and at the
  end of the phase.
- G-T1 and G-T2 are read at every snapshot. A snapshot that fails a gate ends the arm
  there with the verdict "gate failed". That is an early stop, not a selection.
- Only the final model is ever played. No snapshot replaces it.

## Gates before any arena

A failed gate means that arm's verdict is fixed by [Dispositions](#dispositions); no
arena compute is spent on it.

**Collect gates.**

The references are effectively TWO clean collects: the concurrency-128 collect with
the fixed cap (the two banked c128 runs are the same 128 games) and a concurrency-32
collect. "Capped" is the same 128 games at the defective cap 512. Sources: smoke
`optimizer-start/mini/fix_c128_auto`, `main_g128_c32`, `main_c128`; #1427.

| Gate | Pass | Clean | Capped |
|---|---|---|---|
| G-C1 `buffer_full_flushes` | = 0 | — | — |
| G-C2 `eval_max_batch_realised` vs `leaf_rows_required` | equal | 4096 = 4096 | 512 < 2048 |
| G-C3 reply-side Spearman(child Q, prior), fresh and reused roots | ≥ +0.50 both | +0.60 / +0.60 | −0.007 / +0.110 (smoke gen 1: −0.24 / −0.35) |
| G-C4 book-end mover's score | ≤ 0.70 | 0.582 | 0.820 (smoke gen 1: 0.90) |
| G-C5 `target_entropy` | 0.60–0.76 | 0.684–0.694 | 0.799 |
| G-C6 `kl_target_prior` | ≤ 0.50 | 0.290–0.332 | 0.676 |
| G-C7 rows per game | ≥ 80 | 96.7–101.0 | 56.9 |
| G-C8 `draw_rate` | 0.30–0.70 | 0.461–0.531 | 0.156 |
| G-C9 `mean_survivor_minus_node_q` | 0.12–0.30 | 0.181–0.212; D3 bank 0.235 | 0.157 (weak; recorded) |
| G-C10 `repeat_rate_within` | ≤ 0.05 | 0.000 | 0.000 |

**Book-end mover's score:** over all games, (wins + 0.5 × draws) / games for the side
to move at the end of the book line.

**G-C3 and G-C4 need a committed script** (pre-launch check L2). The banked
`loop-smoke-gpu/optimizer-start/scripts/side_stats.py` (sha256 `f3d237cb…`) cannot
serve:
- it samples only the first 3,000 rows per side and root class;
- it does not compute the mover score.

The committed version samples uniformly over the whole collect.

**Train gates (per arm, at the end of the phase and at every snapshot).**

**G-T1 confidence band on the fixed 3,080 D3 audit rows**, through the arena's own
evaluator, as in the D4 round-2 proof.
- **Arms P and S:** mean |Q| 0.37–0.45 and mean P(draw) 0.48–0.55.
- **Arm V:** mean |Q| 0.33–0.45 and P(draw) 0.48–0.55. A value fitted to the 800-sim
  survivor Q has mean |Q| ≈ 0.368 on these rows (value-recalibration study), so V's
  intended target sits near P's lower edge.

References:

| Net | Mean \|Q\| / P(draw) | Arena read |
|---|---|---|
| SL base | 0.409 / 0.513 | — |
| F12 c12 raw | 0.331 / 0.528 | lost 35 Elo |
| Q50 c12 raw | 0.298 / 0.515 | lost 63 Elo |
| Q50 recalibrated | 0.364 | value-only arm recovered 0.85 (0.50 to 1.18) |
| F12 recalibrated | 0.386 | full net recovered 0.97 (0.37 to 1.56); value-only arm unresolved |

**G-T2 probe drift against the SL base:** a gross-defect guard only. A read of KL > 1.5
or |ΔQ| > 0.4 fails. Every read is recorded. References, by instrument:

| Instrument | Run | Updates | KL | \|ΔQ\| |
|---|---|---:|---:|---:|
| `probe_vs_previous` | corrupted smoke, policy only | 46 | 0.148 | 0.295 |
| `probe_vs_previous` | driver trainer on F12 cycle 1 | 69 | 0.030 | 0.018 |
| P1 held-out cycle-2 rows | P1 c1 / c6 / c12 | about 69 / 470 / 948 | 0.038 / 0.215 / 0.299 | 0.020 / 0.048 / 0.066 |
| P1 held-out cycle-2 rows | F12 c12, value on | 948 | 0.288 | 0.107 |

Extrapolated, P1's KL reaches about 0.5–1.2 at 5,700 updates, so no tighter bound is
defensible at the default dose.

**G-T3 held-out policy CE.** After the phase it must not exceed the before-phase
value.
- **Arms P and V:** against the stored target (`heldout_metrics`).
- **Arm S:** `heldout_metrics` scores against the temperature-1 stored target, which S
  does not train on. S's CE is therefore scored against
  `retemper_main_policy_target(stored, temp=2)` by the gate script.

**G-T4 realised settings, read off the consumers** (`metrics.json`, train manifest):
- presentation histogram exactly `{1: trainable rows}`;
- `trained_on_capped_rows` absent;
- `realised_lr_param_groups` all 3e-6;
- `realised_policy_target_temp` 1.0 for P and V, 2.0 for S;
- **Arm S:** target entropy of the collected targets under
  `retemper_main_policy_target(·, temp=2)` recorded, and above Arm P's;
- **Arm V:**
  - `realised_search_wdl_frac` = 1.0;
  - `targets.value_from_survivor_q` ≥ 99% of rows;
  - `value_fallback_stored_search_wdl` and `value_without_search_target` recorded.

## Deciding yardstick

**Seat-balanced primary (D5), single fixed look at N = 3,200 (D9).** Two runs on the
same seed, so the same 800 opening pairs are played in both orientations:

```bash
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
COMMON="--sims 64 --games 1600 --mode matched_sims --search-shape training \
  --no-gumbel-noise --openings <run>/openings/eval.pgn \
  --opening-plies <summary.json eval_max_plies> --max-plies 300 \
  --syzygy $TB --compile off --seed 2026112001"

# A: arm as candidate
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON \
  --candidate <run>/arm_P/gen_0001/train/model.pt --reference <SL base> \
  --label exp1_primary_A --games-out <run>/arenas/exp1_primary_A.games.jsonl \
  --pgn-out <run>/arenas/exp1_primary_A.pgn --out <run>/arenas/results.jsonl

# B: arm as reference
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON \
  --candidate <SL base> --reference <run>/arm_P/gen_0001/train/model.pt \
  --label exp1_primary_B --games-out <run>/arenas/exp1_primary_B.games.jsonl \
  --pgn-out <run>/arenas/exp1_primary_B.pgn --out <run>/arenas/results.jsonl
```

**Pooling.** For each opening o, the arm's pair excess in each run is:
- run A: e_A(o) = (s₁ + s₂)/2 − 0.5, where s is the candidate's (the arm's) game score;
- run B: e_B(o) = 0.5 − (t₁ + t₂)/2, where t is the candidate's (the SL base's) game
  score.

Then d(o) = (e_A(o) + e_B(o)) / 2, the estimate is mean_o d(o), and the SE is
sd_o(d(o)) / √(number of openings): clustered by opening, because each opening
appears in both runs.

The 95% interval is estimate ± 1.96 SE, mapped to Elo by −400 log₁₀(1/(0.5 + x) − 1).
If an opening repeats within a run, all its pairs join one cluster. The script that
does this is committed before launch (L2) and fixed by this record.

**Why seat-balanced.** The pooled seat reading, +6.4 (−4.2 to +17.1 by pair; −5.0 to
+17.8 by opening; D4 round 2), does not exclude an offset. At a true seat offset of
+6.4, a zero-effect net would have passed about 34% of the time under the earlier
two-look rule. Balancing cancels any seat offset at no extra cost, and replaces a
separate base-vs-base control.

**Power (per-pair SD 0.254, from the banked matches).**

| Games | Elo SE | Minimum detectable effect (80% power, two-sided 5%) |
|---|---:|---:|
| 1,600 | 6.2 | 17.5 |
| 3,200 | 4.4 | 12.4 |

Correlation between the two orientations of one opening can widen the clustered
interval; the realised SE is reported.

**Pre-committed reading (primary, read once at 3,200 games):**
- **SUCCESS:** lower 95% bound > 0 Elo.
- **KILL:** upper 95% bound < 0 Elo.
- **Otherwise:** "unread, insufficient evidence". No extension.

A single look keeps the one-sided false-positive rate at 2.5%. The earlier rule (a
look at 1,600 games, then an extension, both at 1.96) gives 4.1%.

**D9 alternative: O'Brien–Fleming two-look.** Look 1 at 1,600 games (800 per
orientation) at z 2.797; look 2 at 3,200 at z 1.977. The false-positive rate stays at
2.5%, and it saves 1,600 games when the effect is large.

**Required in every arena record:**
- `eval_leaf_cap_bound` False;
- `buffer_full_flushes` 0. The arena writes it into the result record on main
  (`scripts/arena_standard.py`, read at `b410456b4`). The field's presence in a real
  run is unverified until the first run.
- `games == games_requested`;
- the same printed config sha across all arenas of this experiment.

Per-game rows, PGNs and seeds are banked.

**Secondaries.**
- **Screens:** V and S that passed their gates play 400 games each, seat-balanced
  (200 + 200), both on seed 2026112003, so they meet the same openings.
- **Confirmation:** the better screen point estimate is confirmed at 1,600 games,
  seat-balanced, on seed 2026112005, with `--openings-fen` built from the EVAL lines
  minus the screen's openings. Building that file is pre-launch check L5. If it
  cannot be built, the confirmation uses `eval.pgn`, and the overlap (about 40 of 800
  expected) is counted and recorded.
- **Reading:** success at 97.5%, lower bound > 0. Otherwise "not confirmed", with no
  extension.

**Openings across seeds.** The arena samples pairs by seed. Overlap between any two
matches is counted and recorded.

## Dispositions

Every outcome has a pre-committed verdict:

| Event | Disposition |
|---|---|
| Collect gate fails | All arms void. At most one re-collect, after a reviewed fix PR, under an addendum to this record. |
| Primary arm fails a gate (final or snapshot) | Verdict "gate failed — unread". No secondary is promoted to primary. |
| A secondary fails a gate | It is not screened. The other secondary, if it passed, is confirmed only if its screen is the better (it then is, by default). |
| Primary reads success, kill or unread | As registered above, read once. |
| Secondary confirmation | Success at 97.5% lower bound > 0. Otherwise "not confirmed", no extension. |
| Arena technical failure | Resume with the arena's own `--resume`: single-colour pairs are discarded and replayed whole, and `resumed_pairs` / `resumed_orphan_pairs` are recorded. A second failure of the same match makes it "unread". |
| Positive control fails a gate or reads unread | Recorded as such. It does not change the primary verdict. |

## Positive control

The question is whether this pipeline can show an improvement in one generation at
all.

**Design (D6 default: unconditional).**
- **Start:** Q50 c12 (sha256 `bb157a84…`), −63.2 (−81.1 to −45.7) against the SL base.
- **Recipe:** Arm V's, because Q50's loss sits in its value output. Arm P does not
  train the value and could fail for that reason alone.
- **Collection:** 4,000 games from the same split's TRAIN lines, with its own run dir
  and `--seed 2026112010` (its own opening permutation). The overlap with the primary
  collection's lines is counted and recorded.
- **Match:** the trained net against Q50 c12, seat-balanced, 1,600 games on EVAL
  openings, seed 2026112006. Same reading rule: success if the lower 95% bound is
  above 0.
- **Dose:** 4,000 games ≈ 1,400 updates, against about 5,700 for the primary.

**Why this start.** It is the clearly weaker same-architecture net with a known
recoverable deficit: in D4 round 2, a two-parameter value map recovered 0.77 (0.47 to
1.06) of Q50's full-net deficit. Other candidates:
- F12 c12 (sha256 `3d3ca6b8…`, −34.9) is less weak.
- No earlier SL checkpoint exists in the SL run's recovery directory, which holds only
  steps 680,213 and 680,276.
- A random-init start would be trivially positive.

**Gates for the control (gated, not record-only).** The control is gated so that a
broken run cannot be read as a pass or a fail. G-T1 is replaced, because Q50 c12 starts
at mean |Q| 0.298, below Arm V's 0.33 floor; a benign run could fail that band at the
start.
- **G-T1 (control):** mean |Q| on the 3,080 audit rows must not fall below the start's
  0.298, and mean P(draw) must stay in 0.45–0.60. Read at the end and at every
  snapshot.
- **Collect gates, gated:** G-C1 (= 0), G-C2 (equal), G-C7 (≥ 80 rows per game) and
  G-C10 (≤ 0.05) with the primary's bounds, and G-C3 at ≥ +0.30 on both root classes
  (the primary's bound is +0.50). These test the machinery, not the start net. G-C3 is
  looser here because no Q50 c12 collect reference exists and the statistic also
  reflects agreement between that net's own policy and value, which were trained
  differently; +0.30 still separates clearly from the defect signature (−0.007 /
  +0.110; smoke run −0.24 / −0.35).
- **Collect gates, recorded only:** G-C4, G-C5, G-C6, G-C8, G-C9. Their bands were set
  on SL-base collects and the expected values for a Q50 c12 collect are unknown.
- **G-T2:** the same gross guard, measured against Q50 c12 (the control's start), not
  the SL base: KL > 1.5 or |ΔQ| > 0.4 fails.
- **G-T3:** held-out policy CE against the stored target must not exceed the
  before-phase value.
- **G-T4:** the Arm V checks unchanged: presentation histogram `{1: trainable rows}`,
  `trained_on_capped_rows` absent, `realised_lr_param_groups` all 3e-6,
  `realised_policy_target_temp` 1.0, `realised_search_wdl_frac` = 1.0,
  `targets.value_from_survivor_q` ≥ 99% of rows.
- **Arena records:** the four required fields of every arena record apply.

A gate failure makes the control "gate failed — unread".

**Reading.**
- **Pass:** licenses only that this pipeline, with Arm V's recipe, can measurably
  improve a net whose deficit is mostly value calibration. A pass is largely expected
  from fixing calibration alone: in D4 round 2 the two-parameter recalibration moved
  Q50 c12 by +48.5 (+25.3 to +72.2) Elo with no training. It does not show that
  self-play policy improvement works, and it says nothing about Arm P.
- **Fail:** does not show that the recipe or dose cannot work. The control's dose
  (about 1,400 updates) is not the primary's (about 5,700). A fail says this recipe at
  this dose did not measurably improve Q50 c12. It points at the pipeline or the value
  fit as a question for the next experiment.
- **Either way,** the control cannot validate or invalidate the primary verdict.

It runs after the primary arena, alone on the GPU. The D6 alternative is to skip it.

## Compute budget

Single RTX 5090. Nothing else is measured during an arena or a collect.

| Stage | GPU hours (16,000-game default) | 4,000-game fallback | Basis |
|---|---|---|---|
| Pre-launch checks L1–L5 | about 0.1 (L4 only) | same | L4 is a 128-game collect |
| Collect | 6.2–6.6 | 1.5–1.7 | 2,424–2,583 games/h, unverified |
| Train ×3, incl. snapshot gates | 2.4–4.4 | 0.6–1.1 | 0.47–0.84 s per update (F12 record: 512 updates in 239–432 s) |
| Final gates | ≤ 0.2 | ≤ 0.2 | 3,080-row evaluator pass per arm |
| Primary arena, 3,200 games | 1.8–2.6 | same | 800-game D4 arms took 1,600–2,260 s |
| Screens, 2 × 400 games | 0.5–0.7 | same | |
| Confirmation, 1,600 games | 0.9–1.3 | same | |
| Positive control (collect 4,000 + train + 1,600-game arena) | 2.6–3.4 | same | |
| **Total** | **≈ 14.6–19.3** | **≈ 8.2–11.1** | |

The trains may run one after another, or two at a time if memory allows; they are not
measurements. Nothing runs during any arena.

## Pre-launch checks

- **L1.** Main ≥ `b410456b4`, extensions rebuilt (ABI 11). The run executes from a
  detached worktree at a pinned sha, recorded here before launch. For the 16,000-game
  default, the snapshot / collect-from PR is merged.
- **L2.** Committed scripts, sha256 recorded:
  - G-C3 / G-C4: uniform sampling over the whole collect, mover score as defined;
  - the audit-row G-T1 evaluator pass;
  - the S-specific G-T3;
  - the seat-balanced pooling.
- **L3.** A tiny-net CPU rehearsal of the multi-arm procedure: tree hashes equal,
  `probe.npz` and holdout hashes equal across arms, each arm's `run_config.json`
  change recorded.
- **L4.** Rerun the clean 128-game reference collect (`fix_c128_auto` settings) with
  `--collect-cuda-graphs` on. Its shard sha256 must equal the graphs-off shards;
  otherwise graphs stay off.
- **L5.** Build the confirmation `--openings-fen` file, or record that the fallback
  applies.

## Recovery plan

- **Everything goes in a new directory:** `~/chess-artifacts/operations/loop-exp1-20261120/`,
  with `openings/`, `collect/`, `arm_P/`, `arm_V/`, `arm_S/`, `posctl/`, `arenas/`,
  `gates/`.
- **Untouched:**
  - the SL base;
  - the live checkout;
  - the production recipe and live yaml;
  - every earlier bank (read-only inputs).
- **Nothing is promoted by this experiment,** even on success.
- **On a kill or a failed gate,** collections, gate outputs and arena banks are kept
  for re-analysis (rule 4). Arm models may be deleted by the owner after the verdict
  is recorded.
- **Recovery is deleting the run directory.**

## Confounds

- All arms share one collection; every arm verdict is conditional on that sample.
- The yardstick is not F12's instrument (root noise off, EVAL openings from the UHO
  split, 300-ply cap). F12's −20 is therefore not expected to reproduce.
- Stored targets include inherited visits under tree reuse (D3). With a move
  temperature above 0, the played move is sampled from the stored target.
- **Twofold repetition** is scored as a proven draw counting pre-root history, a known
  and unchanged condition. In the banked rows, 3.9% have such a child and about a
  quarter of those store q exactly 0, about 1% of all rows
  ([record](2026-11-19-twofold-repetition-as-proven-draw.md), parked). Arm V's
  survivor-Q target inherits it directly.
- **Same-coloured-bishops gap:** the C board recognises insufficient material with
  same-coloured bishops only for exactly K+B vs K+B (4 men). python-chess recognises
  any bishops-only position on one square colour. Known and unchanged
  ([twofold record](2026-11-19-twofold-repetition-as-proven-draw.md) §5).
  - Banked footprint: 0 positions in 4,608 games. Game end uses python-chess, and
    positions of 6 men or fewer are TB-adjudicated.
  - It can still matter at in-search nodes without a TB probe, and in the C
    `is_game_over` used for forced moves.
- The arena's strict tablebase rule stops a match on an unresolved 300-ply game. A
  resume replays the single-colour pairs, and only the capped pair is selected on
  length (D4 round 2).
- Throughput was measured on a shared GPU.

## Second generation (not registered here)

Conditional on a primary success only:
- generation 2 continues from the winning arm with the same recipe and a new
  collection from the continuing opening cursor;
- fresh optimizer;
- the deciding match is generation 2 against generation 1, seat-balanced, with a new
  record written before launch.

A second step of improvement is the owner's stated goal.

## Not verified

- The snapshot / collect-from prerequisite PR: not yet merged.
- That `buffer_full_flushes` appears in a real arena result row. The code writes it;
  no run on current main has been read.
- Clean collect throughput beyond 128-game runs; any CUDA-graph speed-up; graphs-on
  shard identity on this net (L4).
- The probe-drift trajectory at about 5,700 updates.
- That arena seeds draw disjoint EVAL pairs.
- The same-coloured-bishops gap's footprint in this collection (0 in the banked 4,608 games).
