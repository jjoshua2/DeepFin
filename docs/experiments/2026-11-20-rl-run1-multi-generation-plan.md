# RL run 1: a measured multi-generation self-play run from the SL base (plan)

**Status:** DRAFT for owner approval. Nothing has been launched. No stage below
starts until two things have happened:
- the owner has approved this record, with the [decisions](#decisions-for-the-owner)
  recorded in a follow-up commit (CLAUDE.md "Experiments" rule 1);
- the interim diagnosis it builds on ([Stage 0](#stage-0-what-is-known-before-any-training))
  has its full report and registered tests.

**Interim evidence.** Every number marked *interim* comes from an ongoing diagnosis
(bank `~/chess-artifacts/operations/rl-claude-20261119/search-compose/`, marked
NOT_A_CANDIDATE, with its pre-registration `PREREG.md`). Those numbers are pending
that diagnosis's full report and its registered tests T1 and T2. Where this record
re-read a number from the bank's own files, it says so.

**Why this record exists.** The owner set the direction on 2026-11-20, in these
words:
- "i do want the RL loop to be the main path to strength";
- "i expect to be working on this project for years but i would hope to have a
  very strong engine within weeks";
- on doses: "they both seem equally made up to me … i wish we could be more
  rigorous instead of guessing";
- on headroom: the SL base "gained at each more step and it should continue to do
  so as we increase the model; even lc0 and ceres do this at like 1500 elo
  stronger base";
- on the problem: "i think the main problem is we dont know how to calibrate the
  sharpness and distribution of policy and value head to use the information. i
  think it shouldnt be that hard to get half of that 800 elo with like 100,000
  positions if we do it right", and "you have to do it in training so the net
  doesn't flatten or sharpen too much";
- on RL over more SL data: "search goes towards the holes where policy is worse so
  fixing those is very important";
- on target depth: "I'm not convinced that 256 or more is necessary … Gumbel showed
  it could learn in chess at 16 nodes".

So the one-generation experiment is no longer the goal. The plan runs in stages:
1. what is known before any training (Stage 0);
2. an OFFLINE screen of training recipes on banked rows, with no matches (Stage
   1a);
3. the survivors trained on one small fresh collection and judged by a registered
   64-sim match (Stage 1b);
4. the multi-generation run, with an explicit entropy-drift invariant (Stage 2).

Every number that is still a guess is listed under
[What is still a guess](#what-is-still-a-guess), with the measurement that would
replace it.

**Relation to the loop-exp1 record.** The
[loop-exp1 record](2026-11-20-loop-exp1-one-generation-from-sl-base.md) is
superseded as a standalone experiment. This plan reuses it as the source of:
- the collection and `--collect-from` commands;
- the collect gates G-C1–G-C11;
- the model gates G-T1–G-T4, with `gt1` and `gt2`;
- the arena command and the seat-balanced pooling (`scripts/arena_seat_balanced_pool.py`);
- the SL base copy (step 0a), the openings split (step 0b), net write-protection
  and the sha256 provenance rules.

Naming: "SL base" is the supervised starting net, sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. The driver is
`scripts/simple_rl_loop.py` ([docs](../simple_rl_loop.md)).

## Decisions for the owner

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| O1 | **Which recipes enter the offline screen** | **All recipes that need no code (R1, R2-global, R4, R5, R6) first.** Then each recipe that needs code (R1 per-row, R2 agree-only, R3) once its own reviewed PR merges | Screen only the no-code recipes, and stop there |
| O2 | **Target search depth: an equal-compute arm only** | **Include R6: targets at 16, 32 and 64 sims, plus 64 without min-max rescale,** compared at equal compute. No depth is presumed better; the owner doubts 256 is needed | 64 sims only |
| O3 | **Stage 2's time cap** | **One week** from Stage 2's start, as the [success line](#what-success-means-at-one-week) assumes | Another cap |

## Stage 0: what is known before any training

### Search headroom (banked, final)

The SL base's search gain is banked in
`~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`, and
recorded in the [search-ceiling record](2026-11-20-sl-base-search-ceiling.md):

| Search vs search | Games | Elo (95% CI) |
|---|---:|---|
| 64 vs 1 sim | 200 | +848.7 (+690.7 to undefined) |
| 256 vs 64 sims | 400 | +272.4 (+243.1 to +305.3) |
| 800 vs 64 sims | 200 | +478.0 (+410.2 to +582.3) |

These matches were played in one orientation, on the default opening book. The
headroom is large. The open question is whether training transfers any of it into
a net that is stronger at 64 sims.

### 1-sim matches at the default temperature are veto-confounded (interim)

The loop nets read large gains over the SL base when both sides play at 1 sim. This
record recomputed those reads with the pooling script, from
`~/chess-artifacts/operations/rl-claude-20261119/raw-net-1sim/`:

| Net vs SL base, both sides at s sims (temperature 0.1) | s | Pooled Elo (95% CI) |
|---|---:|---|
| P1 c12 | 1 | +240.8 (+218.3 to +265.2) |
| F12 c12 | 1 | +211.7 (+190.3 to +234.7) |
| Q50 c12 | 1 | +128.7 (+109.4 to +148.8) |
| P1 c12 | 4 | +98.1 (+72.4 to +124.8) |
| P1 c12 | 16 | +20.0 (−2.2 to +42.3) |
| P1 c12 | 64 | −3.9 (−22.8 to +15.0) |

**These 1-sim reads are not evidence of transfer.** The diagnosis traced them to
the Gumbel root at 1 sim. With one simulation, only the top-prior child is visited.
The min-max-rescaled completed Q then gives about 5 logits either to the top move
or to everything else, depending on whether the value head rates the 1-ply child
above the root. In effect it is a value veto.

Re-read here from the bank's `onesim_veto.json` (1,050 audit positions; the
reference is the SL base's 800-sim survivor):

| Net | Veto rate | 1-sim move = 800-sim move | Raw argmax = 800-sim move | Prior entropy |
|---|---:|---:|---:|---:|
| SL base | 57.0% | 48.1% | 69.1% | 1.104 |
| P1 c12 | 38.2% | 57.6% | 68.6% | 0.787 |
| F12 c12 | 39.7% | 57.5% | 68.7% | 0.787 |

- **Why the loop nets look stronger at 1 sim.** They are vetoed less, because trunk
  drift made their value self-approving and the sharper prior overrides the veto
  more often. That makes their 1-sim move closer to their own (unimproved) argmax.
- **Their argmax did not improve.** Raw-argmax agreement with the 800-sim reference
  is unchanged by training: 68.6–68.7% against 69.1%.
- **The decay with search fits the veto.** The veto's weight falls with sims (1/2,
  1/5, 1/17, 1/65 at 1, 4, 16, 64 sims, *interim*), matching +241 / +98 / +20 / −4.

A registered temperature-0 1-sim match (T1 in `PREREG.md`, prediction 0 to +60) is
queued and decides this reading. Until it reads, every default-temperature 1-sim
match in this project is labelled **veto-confounded**, and none is used as
evidence of transfer.

### Search-time softening does not recover a gain

P1 c12 at 64 sims with search-time `policy_temp=1.5`, against the SL base at 1.0,
reads **+3.9 (−13.6 to +21.5)**, against −3.9 at 1.0 on the same openings. This
record pooled it from the bank's rows with the pooling script; the diagnosis
quoted +3.9 (−13.1 to +21.0). The control (the SL base at 1.5 against the SL base at
1.0) was still running. Either way, softening the trained prior at search time does
not show a gain.

### What training did (interim; re-read from `flip_by_cycle.json`)

The rows are the 3,080 rl-audit rows (cycle-1 F12 games, which are IN-SAMPLE for P1
and F12). The reference is the SL base's 800-sim survivor (R800).

| Net | Prior entropy | Top-1 vs R800 | Regret vs R800 | Flip rows learned | Agreeing rows broken |
|---|---:|---:|---:|---:|---:|
| SL base | 1.124 | 0.667 | 0.0706 | — | — |
| P1 c01 | 1.022 | 0.674 | 0.0646 | 11.1% | 2.2% |
| P1 c12 | 0.793 | 0.669 | 0.0590 | 23.2% | 8.2% |
| F12 c12 | 0.795 | 0.668 | 0.0605 | 22.6% | 8.3% |

- **Flips learned barely exceed rows broken.** A flip row is one of the 740 rows
  where the stored target's argmax differs from the base's. Of those, 23.2% were
  learned by c12, about 172 rows. Of the 2,340 agreeing rows, 8.2% were broken, about
  192 rows. Top-1 is flat. It peaked at c01, and CE to the in-sample targets was best
  at c01–c02, then worse. Only regret improves (−16%), and the entropy falls by 0.33
  nats.
- **Sharpening is a ratchet at fixed search settings** (*interim*, not in a bank
  file this record found). The target's entropy is about 0.7 × the prior's: base
  prior 1.11 → target 0.74; P1 prior 0.79 → target 0.56, on the same Q.
- **A target pre-screen** (*interim*, not in a bank file this record found). The
  table gives each target's flip rate, the share of flips where the target is
  right, the share where the base is right, and the maximum top-1 gain if every
  flip were learned and nothing broken:

  | Target | Flip rate | Target right | Base right | Max top-1 gain |
  |---|---:|---:|---:|---:|
  | Stored 64-sim | 24.0% | 54% | 19% | +8.4 pp |
  | R256 | 26.8% | 62% | 17% | +11.9 pp |
  | R64, no min-max rescale | 15.4% | 57% | 18% | +6.0 pp |

  Per sim, 64 beats 256 (about 2.8×). R8, R16 and R32 are being measured.

**Conclusion this plan builds on (interim):** the LEARNER is the binding
constraint, not target quality. The owner's calibration hypothesis is supported
in its training-side form: the net must be trained so that it neither sharpens nor
flattens too much, and it must learn flips without breaking agreeing rows.

## The readout protocol

- **Judged at 64 sims.** Every strength claim is a seat-balanced match against a
  fixed opponent at 64 sims, pooled by the pooling script, with its own registered
  rule. Later, the tournament path is added.
- **Secondary: temperature-0 low-sim matches.** These are reported, not judged:
  1, 4 and 16 sims at `--temperature 0`, which removes the 1-sim veto's sampling
  channel. They describe how far a prior gain survives search.
- **Never used as evidence:** default-temperature 1-sim matches (veto-confounded,
  above).
- **Offline descriptors on fixed rows**, never strength claims: flips learned,
  rows broken, top-1 and regret vs R800, prior entropy, veto rate,
  mean V − Q(top child), mean |Q| and P(draw).

## Stage 1a: offline recipe screen (no GPU matches)

**Question.** Which training recipe learns flips without breaking agreeing rows,
and without the entropy ratchet, on banked rows?

**Training rows.** The banked F12 cycle rows that P1 trained on (bank
`rl-claude-20261119/p1-policy-only/` and its sources), **excluding cycle 1**. The
3,080 rl-audit rows (cycle 1) are then held out. The current
`flip_by_cycle.json` reads are in-sample, so they are a reference, not a baseline
to beat on equal terms. The exclusion is pre-launch check P2.

**Baseline.** P1's recipe (policy only), with its optimizer and LR taken from the
P1 harness (`p1-policy-only/harness/`, `p1_cycle.py`), not restated here. It is
replayed on the same rows, with one snapshot per old cycle (69–84 updates each, per
`p1-policy-only/manifest.json`), up to P1's dose of 948 updates.

**Recipes:**

| # | Recipe | How it is run | Code needed |
|---|---|---|---|
| R1 | **Entropy-matched targets:** each row re-tempered so H(target) = H(current prior) | global approximation `--train-target-temp 1.5` now; exact per-row matching later | per-row: yes |
| R2 | **R1 + anchor on agreeing rows** | `--kl-policy λ` to the SL base (all rows), λ ∈ {0.1, 1}; KL only where the argmaxes agree later | agree-only: yes |
| R3 | **Flip-weighted or pairwise ranking loss,** optionally gated by target confidence | — | yes |
| R4 | **Dose with early stopping on the flip metric** (top-1 peaked at cycle 1) | the snapshot sequence of each recipe; the dose is the snapshot that maximises the screen metric | no |
| R5 | **Freeze value drift during policy training** | `--kl-wdl λ` to the SL base; the veto rate, mean V − Q(top child), \|Q\| and P(draw) are tracked | no |
| R6 | **Target arms at equal compute:** 16, 32 and 64 sims, and 64 without min-max rescale | no-rescale: rebuilt offline from stored root statistics (`rebuild_improved_policy`); 16 and 32: a re-search of the banked positions at those budgets | re-search: a script, no driver change |
| R7 | **Interior-node targets** | — | a collector change; last, not in this record's Stage 1 |

**The screen, on the 3,080 held-out audit rows, for every snapshot of every
recipe:**
- **Primary: net flips** = flip rows learned − agreeing rows broken, in rows. The
  base has 740 flip rows and 2,340 agreeing rows on this set.
- **Co-requirements:**
  - top-1 vs R800 does not fall below the SL base's 0.667. The stored target is
    right on only about 54% of its flips, so learning flips is not by itself
    improvement.
  - Prior entropy stays within ±0.15 nats of the SL base's 1.124. That is the
    ratchet guard, and the band is a guess.
- **Recorded:** regret vs R800, CE to the stored target, veto rate, mean
  V − Q(top child), |Q|, P(draw).

**Pre-written survival rule.**
1. A recipe's best snapshot (its R4 dose) survives when three things hold. First,
   its net flips exceed the baseline replay's best snapshot. That is judged on the
   discordant rows (one net right, the other wrong, by argmax == target argmax on
   the 3,080 rows), by an exact one-sided binomial test at 2.5%, Bonferroni over the
   recipes screened. Second, both co-requirements hold.
2. At most two recipes go to Stage 1b: the two with the largest net flips. A tie is
   two recipes within 10 rows; it goes to the higher top-1 vs R800, then to the
   recipe with no code change.
3. If no recipe survives, Stage 1b does not launch, and the record says so. The next
   step is the code-needing recipes (R1 per-row, R2 agree-only, R3), each through its
   own PR, then a re-screen.

**Why offline first.** It costs training compute only, about 0.1 GPU-h per recipe
at P1's dose. It is scored on fixed rows against a deterministic reference, so it is
re-analysable (CLAUDE.md "Experiments" rules 4 and 5).

It is a screen, not a claim. Three recorded cases of fit gains that did not
transfer to Elo are why it never decides strength
([below](#why-no-sizing-from-loss-curves)).

## Stage 1b: survivors on a fresh collection, judged by a registered match

**Collection.** One fresh collection, `lab/collect`, from the SL base: loop-exp1's
`$COLLECT` with `--games 1040 --seed 2026112300`. That is about 100,000 positions,
about 20 min at the measured 3,284 games/h. If a surviving recipe is an R6 target
arm, its own collection runs at its target sims and equal GPU time.
- It must pass loop-exp1's collect gates.
- It is frozen once arms reference it.

**Arms.** Each survivor at its R4 dose, converted from updates on banked rows to
the same number of updates here, through `--collect-from`. The driver's snapshot
mechanism supplies the dose. For example, `--snapshot-updates D` with the snapshot
at D is the played model when D is less than the phase's N.

**Registered match.** Each survivor plays the SL base at 64 sims:
- 2,000 games seat-balanced, on the 500-line `lab_confirm` block, which no other
  match uses;
- pooled with `--verdict-rule lb-gt-zero` at `--z 2.241` (97.5% two-sided:
  Bonferroni over two survivors; `--z 1.96` if only one survives);
- reported alongside, not judged: temperature-0 reads at 1, 4 and 16 sims on the
  400-line `lab_low` block, and the offline descriptors.

**Reading (pre-committed, per survivor):**
- **Success:** lower bound above 0.
- **Kill:** upper bound below 0.
- **Unread:** anything else.

Stage 2 launches on the successful survivor; with two, on the one with the higher
pooled estimate.

**Power.** At per-pair SD 0.254, 2,000 games give an SE of 5.6 Elo. That rises to
about 6.5 at the orientation correlation measured at 1 sim (+0.36); it is unmeasured
at 64 sims. At z 2.241, the match detects about +17 to +20 Elo with 80% power.

**Compute:**
- collect: 0.3 h;
- training: minutes;
- matches: 2.2–3.1 h per survivor (1,274–1,800 games/h);
- temperature-0 reads: minutes.

## Openings: disjoint blocks

The EVAL split (4,000 lines) is cut once, in `eval.jsonl` order:

```bash
mkdir -p $R/openings/blocks
python3 -c 'import json, sys
d = sys.argv[1]; lines = [json.loads(l)["line"] for l in open(d + "/eval.jsonl")]
assert len(lines) == 4000
cuts = [("lab_confirm", 0, 500), ("lab_low", 500, 900),
        ("readout_base", 2000, 2500), ("readout_prev", 2500, 3000), ("confirm", 3000, 3500), ("anchor", 3500, 4000)]
for name, a, b in cuts:
    open(f"{d}/blocks/{name}.fen", "w").write("".join(x + "\n" for x in lines[a:b]))
print(len(cuts), "blocks")' $R/openings
```

- Lines 900–1999 are held in reserve.
- A seat-balanced match of G games on a block of G/4 lines plays every line once
  per orientation (`--openings-fen`, `--games G/2` per run).
- The same cutting code, with other block boundaries, was run verbatim on the
  banked split. Each block loaded in the arena's FEN loader as unique rows, all with
  move history.
- The blocks are disjoint, so each match is independent of the others, and no claim
  is read on openings a screen or a selection used.

## Stage 2: the multi-generation run

Launches only on a Stage 1b success, with the surviving recipe.

### Driver settings

One run dir, `run --generations G`:
- **Start:** `--init-checkpoint` = the Stage 1b winner, copied into the run dir,
  sha-verified and write-protected. Generation 1 is the loop's second generation.
- **Generation size:** the winner's dose in rows (its R4 dose), at the winner's
  sims, concurrency 128, Stage 1b's collect flags. In every generation the played
  model is the snapshot at the dose (`--snapshot-updates D`) or the final model,
  as in Stage 1b.
- **Phase:** `--presentations 1`, the regime Stage 1 measured. Two presentations
  and `--sample-with-replacement --window` are not registered.
- **Optimizer:** `--optimizer-state fresh` each generation, at the winner's LR,
  warmup 0, flat. `--optimizer-state continue` is supported by the driver, but it is
  not registered.
- **Openings:** `--opening-sampling cycle`, `--seed 2026112200`; overlap with
  earlier collections is counted.
- **Holdout and probe:** `--holdout-frac 0.05`, `--probe-rows 2048`.
- **Evaluation:** `--no-eval`; the readouts below replace it.

### The entropy-drift invariant (first-class, every generation)

Sharpening is a ratchet at fixed search settings: the target is sharper than the
prior that searched for it (about 0.7× its entropy, *interim*). So every generation
measures the newest net's prior entropy on the fixed 3,080 audit rows, using the
flip script's definition.
- **Gate:** the entropy must stay within the band Stage 1a used. The default is
  ±0.15 nats around the SL base's 1.124, which is a guess
  ([below](#what-is-still-a-guess)). Leaving the band stops the run at the last
  generation inside it.
- **Recorded beside it:** the veto rate, mean V − Q(top child), |Q|, P(draw), and
  net flips against the previous generation.
- **Not done here: a feedback controller.** One option would raise
  `--train-target-temp` when entropy falls. It would be a separate, recorded
  change.

### Banked per generation (kept for the whole run)

- the collect;
- the Trainer checkpoint, with optimizer;
- `presented.npz` and `metrics.json`;
- the gate outputs, the entropy-invariant read and the content digest.

Nothing is pruned.

### Per-generation gates

- **Gated (machinery):** G-C1 = 0; G-C2 equal; G-C3 ≥ +0.30 on both root classes;
  G-C7 ≥ 80; G-C10 ≤ 0.05.
- **Gated:** the entropy invariant (above).
- **Gated (model):** G-T2 against the previous generation (KL > 1.5 or |ΔQ| > 0.4
  fails); G-T4 as loop-exp1.
- **Recorded, drift alarm:** G-C4, G-C5, G-C6, G-C8, G-C9. Two consecutive
  generations out of band pause the run for a report.
- **On a failure:**
  - A collect gate allows one redo, which replays the same lines from the sealed
    cursor. If the redo fails too, the run stops at g − 1.
  - Any other gated failure stops the run at g − 1.
  - Restarting needs an addendum.

### Readouts

Every k generations, the newest net plays two matches, both seat-balanced:
- **R-base,** against the SL base, on `readout_base`;
- **R-prev,** against the previous readout net, on `readout_prev`.

Each is 2,000 games at 64 sims, judged. Temperature-0 reads at 1, 4 and 16 sims
(800 games each) are reported beside them.
- **Size:** SE 5.6–6.5 Elo, which detects about 16–18 Elo with 80% power.
- **Cadence:** k is the smallest integer with k × g₁ ≥ 18 Elo, clamped to 1–4. g₁
  is Stage 1b's pooled estimate for the winner.
- **Two anchors that do not move.** Anchored Elo depended on the anchor by up to
  82 Elo in one recorded design (experiment ledger), so there are two:
  1. **The SL base,** via R-base.
  2. **Stockfish at fixed nodes,** via `scripts/match_vs_uci.py`, with our engine as
     `python -m chess_anti_engine.uci --checkpoint <net>`. The node count is fixed
     before Stage 2, where the SL base scores 25–75%. 1,000 games per readout on
     `anchor` (`--nodes-b <fixed> --enforce-nodes-b`).

  The transitivity check is that (net vs SF) − (SL base vs SF) agrees with R-base
  within the combined CI.

### Stop rule (pre-committed)

The run stops at the first of:
1. two consecutive readouts with an R-prev pooled upper 95% bound below 0;
2. three consecutive readouts with R-prev unread;
3. a gated failure that one re-collect does not clear, or the entropy invariant;
4. the owner's time cap (O3).

**Deliverable:** the readout net with the highest R-base estimate among those with
a lower bound above 0. That choice carries winner's curse, so it is re-measured
once: 2,000 games at 64 sims, seat-balanced, against the SL base on `confirm`, read
at 95%.

### Recovery

- Every generation's model and Trainer state are kept.
- To resume from the last passing generation g, use `--redo --redo-downstream` of
  generation g + 1's collect, or start a new run dir from generation g's model.
- The SL base, the live checkout, production and earlier banks are untouched.
  Nothing is promoted.

### Compute

With C = games per generation:
- **A generation:** C / 3,284 h of collect, plus
  (C × 96.8 × 0.95 / 256) × 0.47–0.84 s of training, plus minutes for gates and the
  entropy read. At C = 1,040, that is about 0.4 h.
- **A readout:** 4,000 64-sim games, 2.2–3.1 h, plus the temperature-0 reads
  (minutes) and the Stockfish anchor (rate unmeasured).

### What success means at one week

**The one-week target, pre-committed:** within 7 days of Stage 2's start, the
deliverable's confirmation match reads a 95% lower bound above 0 and a point
estimate of at least +50 Elo at 64 sims.
- Below +50 with the lower bound above 0: "gain confirmed, below target".
- Otherwise: "not shown at one week".

+50 is a guess. Stage 1b replaces it with a projection (generations in a week ×
g₁, halved), which is recorded before Stage 2 starts.

## Why no sizing from loss curves

The experiment ledger (2026-08-23) records three fit or audit gains that did not
transfer to Elo:
- F-only: −0.043 nats, for 0 cp;
- absorption: 11.6× the fit, for 0 Elo;
- v020: +3.1 cp, for −9.56 Elo (−29.59 to +10.41).

ft595 traded Elo for fit: −41.0 (−71.3 to −11.4).

This plan adds the veto lesson: a match setting can manufacture a large Elo gap
(+241 at 1 sim) that says nothing about the net's argmax. So the offline screen
only filters; only registered 64-sim matches decide.

## What is still a guess

| Number | Where it is used | The measurement that replaces it |
|---|---|---|
| Every *interim* number in Stage 0 (veto weights, the ratchet's 0.7×, the target pre-screen) | the plan's direction | the diagnosis's full report and its registered tests T1 / T2 |
| ±0.15 nats entropy band | Stage 1a co-requirement, Stage 2 invariant | Stage 1a's surviving recipes' realised entropy range |
| P1's dose (948 updates) as the screen's horizon | Stage 1a | R4's peak, per recipe |
| λ ∈ {0.1, 1} for the KL anchors | R2, R5 | Stage 1a |
| 3,284 games/h; 96.8 rows per game; 0.47–0.84 s per update; 1,274–1,800 arena games/h | compute | Stage 1b's own collect, train and match |
| Per-pair SD 0.254; orientation correlation at 64 sims | match sizing | Stage 1b's realised clustered SE |
| That Stage 1b's gain (g₁) repeats each generation | readout cadence, one-week projection | Stage 2's first readouts |
| `--presentations 1`, a fresh optimizer each generation | Stage 2 | a later comparison |
| +50 Elo at one week | success line | Stage 1b's projection |
| The Stockfish node count | second anchor | pre-Stage-2 calibration |
| The arena's 64 sims as the yardstick | every judged match | fixed, so matches stay comparable |

## Pre-launch checks

- **P1.** Every stage runs from a detached worktree at a pinned main sha, with
  extensions rebuilt; the sha is recorded here.
- **P2.** Stage 1a's training rows exclude cycle 1, which is checked by game id
  against the rl-audit rows.
- **P3.** The SL base is copied into the run dir and write-protected (loop-exp1
  step 0a). The split is reproduced (step 0b). The blocks are cut, with their
  sha256 recorded.
- **P4.** The diagnosis's full report and T1 / T2 results are read, and every
  *interim* number here is replaced or confirmed.
- **P5.** The flip script is re-run on the SL base and on P1 c12, and reproduces
  `flip_by_cycle.json`'s values. P1 c12's in-sample values are the reference.

## Confounds

- Stage 1a's training rows are old-loop rows (F12's collection, collected before
  the cap fix in some cycles; check the taint record before use). A recipe that
  survives on them is re-tested on fresh rows in Stage 1b.
- The R800 reference is the SL base's own 800-sim search. Agreeing with it rewards
  nets that think like a deeper SL base.
- The interim diagnosis ran on nets trained under the old loop.
- loop-exp1's confounds (twofold-repetition draws, the same-coloured-bishops gap)
  apply.
