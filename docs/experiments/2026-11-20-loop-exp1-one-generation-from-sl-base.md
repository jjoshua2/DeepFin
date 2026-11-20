# Loop experiment 1: one self-play generation from the SL base (prospective)

**Status:** DRAFT for owner approval. Nothing has been launched: no collection, no
training, no arena. This record exists so the experiment can be judged, and its
rule fixed, before any compute is spent (CLAUDE.md "Experiments" rule 1). It is
registered only once the owner approves it and the decisions below are filled in.

**Hard prerequisites (no collection before all hold):**
- **PR #1427 merged:** collect sizes the evaluator cap to the search's leaf buffer.
- **PR #1428 merged:** the C buffer-full fallback is replaced by flush-and-resume,
  with a counter (ABI 11).
- **Extensions rebuilt** in the run's tree after both merges.
- **Every collect's `stats.json` shows `buffer_full_flushes == 0` and
  `eval_max_batch_realised == leaf_rows_required`.** A collect that does not is void
  and is not trained on.

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
| D1 | [Primary arm](#arms) | **Arm P**: policy only, value loss off | Arm V (value on the search value) as primary |
| D2 | [Dose](#collection) | **16,000 games** in one collection (about 1.5M rows, about 5,700 updates per arm) | 4,000 games (about 1,400 updates, close to P1's 948) |
| D3 | [Search budget of the stored targets](#collection) | **64 sims** for this readout | 256 sims at 4,000 games, as a separate later collection with its own readout |
| D4 | [Softening used by Arm S](#arms) | **`--train-target-temp 2.0`** | `--policy-prior-mix 0.5 --prior-mix-source reference` (SL base) |
| D5 | [Same-day base-vs-base control](#deciding-yardstick) | **No**: the seat control is closed and banked | Yes: 1,600 games, about 1 more GPU hour |
| D6 | [Positive control](#positive-control) | **Conditional:** run only if the primary is not a success; start from Q50 c12 | Run it regardless, or skip it |
| D7 | [Collect execution](#collection) | **Concurrency 128, `--collect-cuda-graphs` on** | Concurrency 256 (no clean measurement exists), graphs off |
| D8 | [Optimizer](#training) | **Fresh NAdamW, flat LR 3e-6, warmup 0**, as F12 | `--init-optimizer-state checkpoint` (Aurora, the SL base's optimizer): not recommended here |

## Question and hypothesis

**Question.** Can ONE generation of self-play from the SL base produce a net that
beats the SL base head-to-head at 64 simulations, when the targets are chosen by the
diagnostics?

**Why one generation first.** Every earlier loop lost strength:
- F12 read −20.0 Elo on its own panel, and −34.9 on D4's instrument.
- Q50 read −63.2 on D4's instrument.

The diagnostics isolate what to change. A second generation is registered only after
a pass ([below](#second-generation-not-registered-here)).

**Hypothesis (primary).** Arm P will beat the SL base with a lower 95% bound above
0. That arm is policy-only training, exactly once, on fresh self-play rows from the
large book. The changes against P1 are:
- clean data: 153,842 TRAIN lines, no line reused, against P1's 192 recurring
  openings with 18% exact repeats;
- a larger dose;
- a collector with the cap defect fixed.

The value loss is the identified source of loss: P1 read +42.6 (+30.1 to +55.2)
against F12 c12 on the same rows. Removing that loss, and training on rows that no
longer recur, leaves the policy distillation of a search worth +849 Elo over the raw
net (D1, 64 sims against 1 sim).

**What would refute it.** Arm P reads with an upper bound below 0 at the fixed N.
That would mean policy distillation alone at this dose costs strength even on clean
data. P1's −10.9 (−27.7 to +5.9) against the SL base, on F12's rows, already allows
this.

## Arms

All arms train on the SAME collected rows (one collection, below). They differ only
in loss and target settings, so collection is not a confound between them and is
paid once.

| Arm | Train flags beyond the common set | Evidence for it | Role |
|---|---|---|---|
| **P** policy only | `--w-wdl 0` | P1 is the same regime on F12's rows: +42.6 (+30.1 to +55.2) against F12 c12, −10.9 (−27.7 to +5.9) against the SL base. D4: Q50's loss is nearly all in the value output (−55.2 with base policy + Q50 value; +4.3 with Q50 policy + base value). | **Primary** (D1) |
| **V** search value | `--w-wdl 1 --value-outcome-weight 0 --value-search-weight 1 --value-search-source survivor_q --value-draw-source net` | Outcome targets are implicated (P1). The 64-sim stored search value has lower error against deep search than the raw net: MSE 0.0145 against 0.035 (D3). | Secondary |
| **S** softened policy | Arm P + `--train-target-temp 2.0` (D4) | Stored 64-sim targets carry 0.69 nats against the SL base's 1.12. The policy sharpened (held-out entropy 1.117 → 0.787 by F12 c12) without better top-1 against deep search (0.667 → 0.667) (D3, P1 table). In D3, temperature 2 gave 1.32 nats, the measured setting closest below the BT4 / Ceres band (1.49 / 1.53). | Secondary |

**Why P is the recommended primary (D1).**
- It is the only arm whose exact training regime has an arena read (P1).
- Each of V and S adds exactly one change on top of it, so the secondaries read as
  "P plus one thing".
- **Arm V has a specific risk the evidence names.** The value-recalibration study
  found the SL base over-confident against the 800-sim survivor Q (best temperature
  fit T ≈ 1.4–1.55). Training the value toward the survivor value therefore moves it
  toward a flatter shape. D4 round 2 found a flatter value was the dominant measured
  loss in Q50: recalibrating it back toward the base's shape recovered 0.85 (0.50 to
  1.18) of the value-output deficit. V may still win, because what it fits is a search
  value and not an outcome. But the evidence for V is fit evidence, and fit has not
  transferred to Elo here before.
- The alternative (V as primary) is reasonable if the owner weighs D3's MSE result
  above that risk. The [gates](#gates-before-any-arena) catch a gross confidence drift
  either way.

**Flags that exist** (read from `simple_rl_loop.py run --help` at `73011b80b`):
- `--w-wdl`, `--value-outcome-weight`, `--value-search-weight`,
  `--value-search-source {survivor_q, stored_search_wdl}` and
  `--value-draw-source {net, zero}`;
- `--train-target-temp`, a power transform of the stored target at train time;
- `--policy-prior-mix λ` with `--prior-mix-source {reference, actor}` and
  `--prior-mix-reference`;
- `--kl-policy` / `--kl-wdl` with `--kl-reference`.

No variant here needs new code. Not verified: that D3's "temperature" transform is
the same as `--train-target-temp`. The train manifest records the realised target
temperature and `metrics.json` the target entropy, so the realised entropy is checked
at gate G-T4.

**Multiplicity.** One pre-registered primary is decided at 95%. The two secondaries
are screened at 400 games. Only the better of the two by screen point estimate is
confirmed, at the full N on a fresh seed. A confirmed secondary claims success only if
its 97.5% lower bound is above 0: Bonferroni over the two secondaries that competed.
Screen games are never pooled into the confirmation, so the confirmation estimate
carries no selection bias. Head-to-head comparisons between arms are not registered.

## Collection

One collection, run once, from the SL base, with the driver at main after
#1427 / #1428.

| Setting | This experiment | Driver default | F12 | Source / reason |
|---|---|---|---|---|
| Sims / topk / c_scale / c_visit | 64 / 16 / 0.1 / 50 | same | same | the F12 instrument and the D1 reads |
| Target q-rescale | on | on | on | unchanged |
| Tree reuse | on | on | on | production; stored targets include inherited visits (D3) |
| Root noise | Gumbel 1.0 → 0.5 over moves 12–15 | same | not re-read here | driver default |
| Move temperature | 0.5 → 0.1 over the first 10 plies after the book, then 0.1 | same | not re-read here | driver default |
| Openings | UHO books split 153,842 TRAIN / 4,000 EVAL, 95% / 5% book mix, `cycle` sampling | same | 192 fixed starts | 18% exact repeats under F12's 192 starts |
| Games | 16,000 (D2) | 2,000 | 192 per cycle | below |
| Concurrency | 128 (D7) | 64 | 16 slots | the only clean measured setting at scale |
| Evaluator cap | auto (= `leaf_rows_required`) | auto after #1427 | — | #1427 |
| `--collect-cuda-graphs` | on (D7) | off | — | outputs documented bitwise identical ([paths doc](../search_and_encode_paths.md)), so the same shards |
| Max plies | 450 | 450 | 450 per game | unchanged |
| Tablebases | strict rule50 route, production pair | — | strict | CLAUDE.md "Production Syzygy" |

**Dose (D2).** The recommendation is 16,000 games, the owner's stated real dose of
15–20k games per generation. The numbers behind it:
- **Rows and updates.** A clean 128-game collect at these settings banked 12,372 rows,
  96.7 per game (smoke `optimizer-start/mini/fix_c128_auto`). At that density,
  16,000 games ≈ 1.55M rows. After the 5% holdout that is about 5,700 updates at
  batch 256.
- **Against earlier loops.** That is 6× P1's and F12's 948 updates over twelve cycles,
  and 83× F12's 69 updates per 192-game cycle (2026-11-18 record).
- **Why the larger dose is defensible.** No line is reused: 16,000 games use about 10%
  of TRAIN. Every row is presented once. The evidence ties the earlier losses to the
  value loss and recurring openings, not to the number of updates.
- **What the larger dose risks.** No arena read exists at this many updates from the
  SL base, and the drift references in the gates below were measured at 46–948
  updates.
- **The alternative,** 4,000 games (about 1,400 updates), stays inside that measured
  range and costs a quarter of the collection. Its minimum detectable effect is the
  same, because the arena N does not change.

**Search budget (D3).** D3 found the best 256-sim targets beat every transformed
64-sim target on mass, top-1 and regret, for example cv0/cs0.5: mass 0.702, top-1
0.763, regret 0.014. The default 256-sim target did not beat the sharpest 64-sim cell
on mass (0.674 against 0.677), and the intervals overlap.

So "64 sims at N games" against "256 sims at N/4" (equal search compute) is not
decided by the evidence. The recommendation keeps 64 sims for this readout:
- it is the instrument of every banked loop and arena read;
- a 256-sim collection would be a second data-affecting change in the same readout
  (rule 3).

The 256-sim variant is a natural experiment 2, with its own collection and readout.

**Throughput is unverified.** The estimate is a range:
- Clean collects at concurrency 128 measured 2,424–2,583 games/h, but on 128-game
  runs, where the tail of the last games weighs heavily.
- The 5,180–6,139 games/h smoke reads at concurrency 256 had the cap defect: games a
  third shorter, corrupted reply-side targets. They are not usable.
- CUDA graphs were reported at 1.93× on one small run. That figure is not banked in a
  file this record could read.

Range for 16,000 games: about 3.2 h (5,000 games/h) to 6.6 h (2,400 games/h).

## Training

Common to every arm:
- one phase, generation 1;
- `--presentations 1` (exact-once, shuffled), batch 256, `--mirror-prob 0.5`;
- `--lr 3e-6 --lr-floor 3e-6 --warmup-steps 0`;
- `--optimizer nadamw --optimizer-state fresh --init-optimizer-state fresh`;
- `--holdout-frac 0.05`, `--probe-rows 2048`.

Note: the driver's `--lr` default is 3e-5, so 3e-6 must be passed explicitly.

**Optimizer (D8).** F12 used fresh NAdamW at flat 3e-6 with warmup 0 and showed no
first-step hit:
- the D2 probe read |ΔQ| 0.075, KL 0.039 for cycle 1;
- this driver's trainer on F12's cycle-1 rows reads |ΔQ| 0.018, KL 0.030
  ([driver doc](../simple_rl_loop.md), "Training details").

With warmup 0 the `--warmup-start 0` zero-first-update caveat does not arise. It
applies only when `--warmup-steps > 0`.

`--init-optimizer-state checkpoint` would adopt the SL base's Aurora state, which
changes the optimizer family (Aurora with the `mlp_out` matrix scope, matrix LR ×20).
That is a second change in the same readout and is not recommended.

**Running the three arms on one collection.** The driver pins one setting set per
run dir, and `run` collects and trains together. The plan:
1. Collect generation 1 once, in its own run dir.
2. Copy that dir's sealed `gen_0000/` and `gen_0001/collect/` into one run dir per
   arm. A copy keeps every sha256 the lineage checks.
3. Run standalone `train --generation 1` there, with the arm's flags and
   `--allow-config-change`, which records the change.

This mechanism is **not verified**. Pre-launch check L3 runs it end to end on a
tiny-net CPU run before any GPU work. If the driver refuses, the remedy is a small
code change in its own reviewed PR (for example `--collect-from`). Manifests are never
hand-edited.

Every arm must hold out the same games and build the same `probe.npz`: same seed,
same holdout rule. Their sha256 must match across arms.

## Gates before any arena

Each gate is cheap and pre-committed. A failed gate means the arm is not played and
its verdict is "gate failed"; no arena compute is spent on it. A failed collect gate
voids every arm.

**Collect gates.** Values are from `stats.json` unless stated. "Clean" is the
128-game concurrency-128 collect with the fixed cap. "Capped" is the same collect at
the defective cap 512 (smoke `optimizer-start/mini/fix_c128_auto` and `main_c128`;
#1427 table).

| Gate | Pass | Clean | Capped |
|---|---|---|---|
| G-C1 `buffer_full_flushes` | = 0 | — (new in #1428) | — |
| G-C2 `eval_max_batch_realised` vs `leaf_rows_required` | equal | 4096 = 4096 | 512 < 2048 |
| G-C3 reply-side Spearman(child Q, prior), fresh and reused roots | ≥ +0.50 both | +0.60 / +0.60 | −0.007 / +0.110 (smoke gen 1: −0.24 / −0.35) |
| G-C4 book-end mover's score | ≤ 0.70 | 0.58 | 0.82 (smoke gen 1, 256 games: 0.90) |
| G-C5 `target_entropy` | 0.60–0.76 | 0.694 (0.684 at concurrency 32) | 0.799 (smoke gen 1: 0.93) |
| G-C6 `kl_target_prior` | ≤ 0.50 | 0.290–0.332 | 0.676 |
| G-C7 rows per game | ≥ 80 | 96.7–101.0 | 56.9 |
| G-C8 `draw_rate` | 0.30–0.70 | 0.461–0.531 | 0.156 |
| G-C9 `mean_survivor_minus_node_q` | 0.12–0.30 | 0.181–0.212; D3 bank 0.235 | 0.157 (weak discriminator; recorded) |
| G-C10 `repeat_rate_within` | ≤ 0.05 | 0.000 | 0.000 |
| G-C11 `unresolved_games` | recorded; rows of ply-cap games are not kept | 0 | 0 |

G-C3 and G-C4 are not in `stats.json` today. They need the CPU script that produced
#1427's table, computed from stored root statistics and `games.jsonl`. That script
must be committed (or banked with its sha256) before launch. Without it, those two
gates cannot run and the collection is not trained on.

**Train gates (per arm):**
- **G-T1 confidence band on the fixed 3,080 D3 audit rows.** Mean |Q| must lie in
  0.37–0.45 and mean P(draw) in 0.48–0.55. Both are computed through the arena's own
  evaluator, as in the D4 round-2 proof.

  | Net | Mean \|Q\| / P(draw) | Arena read |
  |---|---|---|
  | SL base | 0.409 / 0.513 | — |
  | F12 c12 raw | 0.331 / 0.528 | lost 35 Elo |
  | Q50 c12 raw | 0.298 / 0.515 | lost 63 Elo |
  | Q50 recalibrated | 0.364 | value-only arm recovered 0.85 (0.50 to 1.18) of its deficit |
  | F12 recalibrated | 0.386 | full net recovered 0.97 (0.37 to 1.56); value-only arm unresolved |

  The lower |Q| edge sits just above recalibrated Q50. The upper edge mirrors it,
  because over-confidence is untested.
- **G-T2 probe drift against the SL base** (`probe_vs_previous`, generation 1).
  Policy-only arms: |ΔQ| ≤ 0.15. Every arm: KL ≤ 0.50. References:

  | Run | Updates | \|ΔQ\| | KL |
  |---|---:|---:|---:|
  | Corrupted smoke, policy only | 46 | 0.295 | 0.148 |
  | Driver trainer on F12 cycle 1 | 69 | 0.018 | 0.030 |
  | P1 c12, policy only | 948 | 0.066 | 0.299 |
  | F12 c12, value on | 948 | 0.107 | 0.288 |

  At the recommended dose no reference exists. These bounds are gross-defect guards,
  set above the clean full-dose reads. Arm V's |ΔQ| is recorded, not gated, because
  training the value moves it by design.
- **G-T3 held-out policy CE to the stored target.** After the phase it must not exceed
  the before-phase value. Both values are in `metrics.json`.
- **G-T4 presentation and target checks:**
  - presentation histogram exactly `{1: trainable rows}`;
  - `trained_on_capped_rows` absent;
  - Arm S: realised target entropy above Arm P's, by at least half of the difference
    D3 measured for temperature 2 (0.69 → 1.32 nats);
  - Arm V: `metrics.json` `targets` shows value-target fraction 1.0 search, 0.0
    outcome, read off the consumer.

## Deciding yardstick

One deciding match: the primary arm against the SL base, at fixed N, final read only:

```bash
PYTHONPATH=. python -u scripts/arena_standard.py \
  --candidate <run>/arm_P/gen_0001/train/model.pt \
  --reference <SL base> \
  --sims 64 --games 1600 --mode matched_sims --search-shape training \
  --no-gumbel-noise \
  --openings <run>/openings/eval.pgn --opening-plies <summary.json eval_max_plies> \
  --max-plies 300 \
  --syzygy "$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6" \
  --compile off --seed 2026112001 --label exp1_primary_vs_slbase \
  --games-out <run>/arenas/exp1_primary.games.jsonl \
  --pgn-out <run>/arenas/exp1_primary.pgn --out <run>/arenas/results.jsonl
```

- **Openings.** EVAL lines only, never TRAIN. The split's construction guarantees no
  TRAIN game starts at, passes through or is a prefix of an EVAL start. The
  production split's `eval_max_plies` is 16 on the smoke copy; this run reads its own
  `summary.json`.
- **Required in the result record:**
  - `eval_leaf_cap_bound` False;
  - `buffer_full_flushes` 0 (after #1428);
  - `games == games_requested`;
  - the config sha printed by `--search-shape training`, the same for every arena of
    this experiment.
- **Banked:** per-game rows, PGNs and seeds.

**N = 1,600 (800 opening pairs).** From banked matches of the same instrument:
- 800 games give a 95% interval of about ±17–18 Elo (the D4 800-game arms);
- 1,600 games give about ±12.5 (P1's deciding match: +30.1 to +55.2).

That puts the Elo SE at about 6.4 at 1,600 games. The minimum detectable effect at 80%
power, two-sided 5%, is about 2.8 × 6.4 ≈ **18 Elo**. A true gain of +10 is likely to
read "unread". This design cannot see effects that small at this N, and says so in
advance.

**Pre-committed reading (primary, pair-clustered interval):**
- **SUCCESS:** lower 95% bound > 0 Elo.
- **KILL:** upper 95% bound < 0 Elo.
- **Otherwise:** ONE pre-committed extension of 1,600 more games on seed 2026112002.
  The two matches are pooled (3,200 games; minimum detectable effect about 12.7 Elo)
  and read by the same rule. Still neither: final verdict "unread, insufficient
  evidence". No further extension.

**Secondaries.** Arms V and S that passed their gates play 400 games each against the
SL base, both on seed 2026112003, so they meet the same openings. The better screen
point estimate is confirmed at 1,600 games on seed 2026112005 and read at 97.5%
([multiplicity](#arms)). A 400-game match is a screen only: about ±35 Elo.

**Openings across seeds.** The arena samples each match's pairs from `eval.pgn` by
seed. Whether two seeds' pairs are disjoint is not guaranteed by the sampler. The
overlap of every later match with the primary's openings is counted and recorded, as
P1 did for TRAIN starts.

**Seat and sampling noise.** The seat control is closed (D4 round 2): pooled same-net
+6.4 (−4.2 to +17.1 by pair; −5.0 to +17.8 by opening). Matches are read against 0.
- **D5:** a same-day base-vs-base match at N = 1,600 is not recommended. It costs as
  much as the deciding match, and its information is already banked.
- The alternative is to run it on seed 2026112001, immediately before the primary,
  and report it beside the primary. It would not change the rule.

## Positive control

Can this pipeline show an improvement at this dose at all? The control is the same
one-generation recipe started from a net of the same architecture that is clearly
weaker than the SL base.

**Candidates on disk with banked arena reads:**
- **Q50 c12:** sha256 `bb157a84…`, −63.2 (−81.1 to −45.7) against the SL base.
- **F12 c12:** sha256 `3d3ca6b8…`, −34.9 (−53.1 to −16.8).

No earlier SL checkpoint exists in the SL training run's recovery directory: it holds
only steps 680,213 and 680,276. A `--random-init` start with the SL base's
architecture is possible but trivially positive. It tests plumbing, not this dose.

**Reading.** Collect from Q50 c12, train the primary recipe, and play against Q50 c12
at 1,600 games.
- **Pass** (lower bound > 0): the pipeline can move strength upward in one generation
  at this dose, from a net whose deficit is known to be largely recoverable. In D4
  round 2, a two-parameter value map recovered 0.77 (0.47 to 1.06) of Q50's full-net
  deficit.
- **Fail:** together with a failed primary, this points at the recipe or dose rather
  than the start.
- **Caveat:** Q50's loss is in its value output, and Arm P does not train the value.
  The control with Arm P can therefore fail for that reason alone. With Arm V it
  would be the more natural test.

**Cost.** One more 16,000-game collection, one train and one 1,600-game arena: about
5–9 GPU hours. That is material, hence **D6: run it only if the primary is not a
success**, after the primary, with its own record of the result. The alternatives are
to run it regardless, after the primary, or to skip it.

## Compute budget

Single RTX 5090. Nothing else is measured while an arena or a throughput-bearing
collect runs.

| Stage | GPU hours | Basis |
|---|---|---|
| Pre-launch checks L1–L3 (CPU) | 0 | tiny-net CPU run |
| Collect, 16,000 games | 3.2–6.6 | 2,400–5,000 games/h (see Collection) |
| Train ×3 (about 5,700 updates each) | 2.4–4.2 | 0.47–0.84 s per update (F12 record: 512 updates in 239–432 s) + probe / holdout metrics |
| Gates (audit rows, probe) | ≤ 0.2 | 3,080-row evaluator pass per arm |
| Primary arena, 1,600 games | 0.9–1.3 | 800-game D4 arms took 1,600–2,260 s |
| Secondary screens, 2 × 400 games | 0.5–0.7 | as above |
| Secondary confirmation, 1,600 games | 0.9–1.3 | as above |
| **Core total** | **≈ 8–14** | |
| Extension, 1,600 games (conditional) | 0.9–1.3 | |
| Base-vs-base control (D5, if chosen) | 0.9–1.3 | |
| Positive control (D6, conditional) | 5–9 | collect + train + arena |

The three trains may run one after another, or two at a time if memory allows; they
are not measurements. Nothing runs during any arena.

## Pre-launch checks (no GPU)

- **L1.** #1427 and #1428 merged; the run's tree rebuilt (ABI 11). The run executes
  from a detached worktree at a pinned main sha, recorded here before launch.
- **L2.** The G-C3 / G-C4 script committed or banked with its sha256.
- **L3.** The tiny-net CPU dress rehearsal of the whole plan:
  - split;
  - collect;
  - copy into three arm dirs;
  - three standalone trains with `--allow-config-change`;
  - the gate scripts;
  - one tiny arena.

  It must show identical `probe.npz` sha256 across arms and lineage checks passing on
  the copies.

## Recovery plan

- **Everything goes in a new directory:**
  `~/chess-artifacts/operations/loop-exp1-20261120/`, with `openings/`, `collect/`,
  `arm_P/`, `arm_V/`, `arm_S/`, `arenas/`, `gates/`.
- **Untouched:**
  - the SL base;
  - the live checkout;
  - the production recipe and live yaml;
  - every earlier bank (read-only inputs).
- **Nothing is promoted by this experiment,** even on success.
- **On a kill or a failed gate,** the collection and arm models are kept until the
  verdict is recorded, for re-analysis (rule 4). The owner may then delete the arm
  models; the collection, gate outputs and arena banks stay.
- **Recovery is deleting the run directory.**

## Confounds

- All arms share one collection, so every arm verdict is conditional on that one
  sample of games.
- The yardstick is not F12's instrument (root noise off, EVAL openings from the UHO
  split, 300-ply cap). F12's −20 is therefore not expected to reproduce. The
  head-to-head design does not depend on it.
- Stored targets include inherited visits under tree reuse (D3).
- With a move temperature above 0, the played move is sampled from the stored target.
- Twofold repetition is scored as a proven draw counting pre-root history, a known
  and unchanged condition. In the banked rows, 3.9% have such a child and about a
  quarter of those store q exactly 0
  ([record](2026-11-19-twofold-repetition-as-proven-draw.md), parked). Arm V's
  search-value targets inherit it directly.
- Throughput was measured on a shared GPU.
- The arena's strict tablebase rule stops a match on an unresolved 300-ply game. A
  resume replays the single-colour pairs, and only the capped pair is selected on
  length (D4 round 2).

## Second generation (not registered here)

Conditional on a primary success only:
- generation 2 continues from the winning arm with the same recipe and a new 16,000
  games from the continuing opening cursor;
- `--optimizer-state fresh`, unless a separate record argues otherwise;
- the deciding match is generation 2 against generation 1, with a new record written
  before launch.

A second step of improvement is the owner's stated goal.

## Not verified

- That the three-arm copy-and-train mechanism passes the driver's lineage and pinning
  checks (L3).
- That D3's target temperature equals `--train-target-temp`.
- Clean collect throughput above 128-game runs; the 1.93× CUDA-graph speed-up.
- That arena seeds draw disjoint EVAL pairs.
- The probe-drift bounds at about 5,700 updates: no reference exists at that dose.
- The G-C3 / G-C4 script: not on main today.
