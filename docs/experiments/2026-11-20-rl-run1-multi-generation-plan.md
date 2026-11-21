# RL run 1: a measured multi-generation self-play run from the SL base (plan)

**Amended (draft):** [count games, not rows](2026-11-21-rl-run1-amendment-games-not-rows.md)
proposes replacing Stage 1 below with a variance-vs-shape pilot and one large generation
read as a dose-response curve in games. Until the owner decides, both are drafts.

**Status:** DRAFT for owner approval. Nothing has been launched. No stage below
starts until two things have happened:
- the owner has approved this record, with the [decisions](#decisions-for-the-owner)
  recorded in a follow-up commit. Main's CLAUDE.md "Experiments" asks for the
  hypothesis, control, one deciding yardstick as an exact command, a pre-committed
  threshold and a recovery plan before any training compute;
- the stage's own pre-launch checks have passed ([below](#pre-launch-checks)).

**Interim evidence.** Every number marked *interim* comes from an ongoing diagnosis
(bank `~/chess-artifacts/operations/rl-claude-20261119/search-compose/`, marked
NOT_A_CANDIDATE, pre-registration `PREREG.md`). Those numbers are pending that
diagnosis's full report and its registered tests T1 and T2. T1's orientation A is
banked (`results/results.jsonl`, row `t0s1_P1_A`, 800 games) and **not read**: by
`PREREG.md` it is read once, after both orientations. T2 has not run. Numbers re-read here from bank files say
which file. The independent review of this plan (PR #1439) banked its evidence at
`~/chess-artifacts/operations/rl-claude-20261119/plan-review-1439/`; numbers from
it are marked *review* and were spot-checked here where stated.

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
  it could learn in chess at 16 nodes";
- on scope: "im not sure if its worth getting that fancy yet i mostly just want to
  get a working RL loop in selfplay even, which i think should be possible with 64
  node search … as long as on average its better if we get enough of it it should
  work. i wonder if maybe we need larger batch sizes to avg it out better?"

**The principle (owner, 2026-11-20).** "i want to be an independent project and not
merely distill lc0 … i want a pipeline that could in theory get us even stronger
than bt4 … right now we are so far below both of them we just need raw strength not
finding narrow holes."
- The run's strength must come from the project's own pipeline: self-play search
  targets and game outcomes. That pipeline can in principle exceed any teacher.
- BT4 and Ceres are used only as a ruler and a sharpness and calibration reference.
  Nothing in this plan trains on them. A BT4 raw-policy target diagnostic and a
  distance-to-BT4 match are [parked](#parked-with-launch-conditions); the
  diagnostic's net would never be a candidate.
- **Out of scope by the owner's direction:** opponent-exploitation and blind-spot
  arms. None is in this plan.

**The goal of this plan:** the first measured self-play gain at 64 sims against the
SL base, and then repeating it. Everything else is cut to what that needs.

**The plan, in order:**
1. **Stage 0** (done): what the banked matches and diagnostics already say.
2. **Stage 0b** (minutes of compute; the rulers are already built): score targets
   and nets on leak-free audit rows against searched BT4 and Ceres, with deep SF as
   a cross-check, and read the registered tests T1 and T2. **It does not gate
   Stage 1;** it only picks the ruler that chooses each cell's dose.
3. **Stage 1** (one generation from the SL base): one collection and four
   pre-committed cells, all at the SL base's Aurora split with fresh optimizer
   state: (a) the stored target, (b) the no-rescale target, (c) the stored target at
   train temperature 1.5, each at batch 512, plus (a) at effective batch 2048 (the
   owner's averaging hypothesis). **All four go to registered seat-balanced 64-sim
   matches.** The offline ruler only picks each cell's dose (its snapshot, defined
   in samples); it never decides which cells play.
4. **Stage 2** (only after a Stage 1 success): repeat the winning recipe for
   several generations. It is kept short here until Stage 1 has a winner.
5. **[Parked, with launch conditions](#parked-with-launch-conditions):** the
   distance to BT4 at 1 node (formerly Stage 0c), the BT4 raw-policy target
   diagnostic, and a value-first cell.

Every number that is still a guess is listed under
[What is still a guess](#what-is-still-a-guess).

**Relation to the loop-exp1 record.** The
[loop-exp1 record](2026-11-20-loop-exp1-one-generation-from-sl-base.md) is
superseded as a standalone experiment. This plan takes from it:
- the collection command and `--collect-from` arms;
- the collect gates G-C1–G-C11 and the model gates G-T1–G-T4, with `gt1` and `gt2`;
- the arena command and the seat-balanced pooling (`scripts/arena_seat_balanced_pool.py`);
- the SL base copy (step 0a), the openings split (step 0b), net write-protection
  and the sha256 provenance rules.

Naming: "SL base" is the supervised starting net, sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. The driver is
`scripts/simple_rl_loop.py` ([docs](../simple_rl_loop.md)).

## Decisions for the owner

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| O1 | **Stage 1's cells** | **Four pre-committed cells, all matched:** (a), (b), (c) at batch 512, and (a) at effective batch 2048 (below) | A different large batch than 2048, or a fifth cell |
| O2 | **Stage 1's optimizer state** | **Fresh** (the Aurora split as the SL base used it; the split is the owner's and is not an arm) | Continue the SL base's saved Aurora state |
| O3 | **Stage 1's collection size** | **1,040 games (≈ 100k rows)** | 2,080 games |
| O4 | **Stage 2's time cap** | **One week** from Stage 2's start | Another cap |
| O5 | **A 16-sim collection arm at matched GPU time** | **Deferred** until Stage 1 reads (the goal is a working loop at 64 sims) | Add it to Stage 1 as a fifth matched cell: 16 sims gives about **3.8×** the rows of 64 sims per GPU-hour (measured; the exchange-rate table at the end of Stage 1) |

## Stage 0: what is known before any training

### Search headroom (banked; the 64-vs-1 row is veto-confounded)

From `~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`,
recorded in the [search-ceiling record](2026-11-20-sl-base-search-ceiling.md):

| Search vs search | Games | Elo (95% CI) | Note |
|---|---:|---|---|
| 64 vs 1 sim | 200 | +848.7 (+690.7 to undefined) | **veto-confounded:** temperature 0.1, so the 1-sim side plays the vetoed move (below). The headroom over the raw prior is unmeasured |
| 256 vs 64 sims | 400 | +272.4 (+243.1 to +305.3) | |
| 800 vs 64 sims | 200 | +478.0 (+410.2 to +582.3) | |

These matches were played in one orientation, on the default opening book. The
owner's "half of that 800 Elo" is anchored on the 64-vs-1 row. Stage 0b replaces it
with a temperature-0 64-vs-1 match.

### Default-temperature 1-sim matches are veto-confounded (interim)

Recomputed here with the pooling script from
`~/chess-artifacts/operations/rl-claude-20261119/raw-net-1sim/`; reproduced by the
review to 0.1 Elo:

| Net vs SL base, both sides at s sims (temperature 0.1) | s | Pooled Elo (95% CI) |
|---|---:|---|
| P1 c12 | 1 | +240.8 (+218.3 to +265.2) |
| F12 c12 | 1 | +211.7 (+190.3 to +234.7) |
| Q50 c12 | 1 | +128.7 (+109.4 to +148.8) |
| P1 c12 | 4 | +98.1 (+72.4 to +124.8) |
| P1 c12 | 16 | +20.0 (−2.2 to +42.3) |
| P1 c12 | 64 | −3.9 (−22.8 to +15.0) |

P1 c12 at 64 sims has a second read: −10.9 (−27.7 to +5.9), in the
[takeover record](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md).

**These 1-sim reads are not evidence of transfer.** At one simulation, only the
top-prior child is visited. The min-max-rescaled completed Q then gives about 5
logits either to the top move or to everything else, depending on whether the value
head rates the 1-ply child above the root: a value veto.

Re-read from the bank's `onesim_veto.json` (1,050 audit positions; reference: the SL
base's 800-sim survivor, R800):

| Net | Veto rate | 1-sim move = R800 | Raw argmax = R800 | Prior entropy | 1-sim Elo vs base |
|---|---:|---:|---:|---:|---:|
| SL base | 57.0% | 48.1% | 69.1% | 1.104 | — |
| F12 c06 | 56.9% | 50.5% | 69.9% | 0.828 | not played |
| P1 c12 | 38.2% | 57.6% | 68.6% | 0.787 | +240.8 |
| F12 c12 | 39.7% | 57.5% | 68.7% | 0.787 | +211.7 |
| Q50 c12 | 58.8% | 50.7% | 67.5% | 0.762 | +128.7 |

- P1 c12 and F12 c12 are vetoed less, and their 1-sim move is closer to R800, while
  their raw argmax is not.
- **Q50 c12 gains +129 at 1 sim without a lower veto rate.** So the veto explains at
  most part of the 1-sim gap (interim).
- The veto's weight falls with sims (1/2, 1/5, 1/17, 1/65 at 1, 4, 16, 64; interim),
  in line with the decay +241 / +98 / +20 / −4.
- **How much of the 1-sim top-1 gain the veto explains** (interim,
  `veto_decomp.json`). The decomposition replays each net's own prior under the
  base's veto. The veto pattern accounts for 0.82–1.11 of P1 c12's and F12 c12's
  1-sim top-1 gain, depending on the ruler, but only 0.32–0.37 of their regret gain.
  For Q50 c12 the share is 0.26 on R800 and is unstable on the other rulers. **Q50
  is not a veto story.**
- **The veto rate depends on the position set, not the definition.**
  `onesim_veto.json` gives 57.0% (base) and 38.2% (P1) on the 1,050 audit
  positions. `pgn_onesim_check.json` gives 43.0% and 31.4%, with the same definition
  (own Q(top-prior child) < own V(root)), on 395 positions from the real 1-sim
  games. Both are re-read here; `PREREG.sha256.addendum` (addendum 2) corrects
  PREREG's label.

Until T1 reads, every default-temperature 1-sim match in this project is labelled
**veto-confounded** and none is used as evidence of transfer.

### Search-time softening shows no gain

P1 c12 at 64 sims with search-time `policy_temp=1.5`, against the SL base at 1.0:
**+3.9 (−13.6 to +21.5)**, against −3.9 at 1.0 on the same openings. Pooled here
and by the review. The diagnosis quoted +3.9 (−13.1 to +21.0).
- Its seat offset is −17.0 (−33.6 to −0.4), the only one of the seven reads whose
  offset excludes zero. Pooling cancels it.
- The control (the SL base at 1.5 against the SL base at 1.0) was still running.

### What training did (interim; re-read from `flip_by_cycle_v2.json`)

**Out of sample, the trained policies got worse after cycle 1, on every ruler.**
These are the 2,433 leak-free audit rows (`leak_mask.npz`, which drops the 647
leaked rows):

| Net | Entropy | Top-1 R800 | Top-1 BT4 | Top-1 C3 | Argmax changed | On changed rows, net / base right (BT4) |
|---|---:|---:|---:|---:|---:|---:|
| SL base | 1.191 | 0.658 | 0.704 | 0.679 | — | — |
| P1 c01 | 1.085 | 0.665 | 0.706 | 0.686 | 4.6% | 40% / 37% |
| P1 c02 | 1.004 | 0.661 | 0.698 | 0.680 | 6.4% | 33% / 41% |
| P1 c06 | 0.892 | 0.658 | 0.690 | 0.673 | 10.1% | 34% / 45% |
| P1 c12 | 0.866 | 0.649 | 0.675 | 0.666 | 12.1% | 30% / 49% |
| F12 c12 | 0.866 | 0.647 | 0.676 | 0.667 | 12.2% | 30% / 48% |

- c01 is neutral to slightly positive on every ruler. Every later cycle declines on
  every ruler.
- Where P1 c12 changed its argmax, BT4 says the net is right 30% of the time and
  the base 49%.
- This matches the 64-sim Elo ordering (base ≥ P1 ≥ F12).

**The in-sample view below looked "flat" only because of the leak.** All 3,080
rows (cycle-1 F12 games, in sample for P1 and F12), with two column groups:
- agreement with the **stored training target's argmax** (flip rows learned,
  agreeing rows broken; a flip row is one of the 740 rows where the stored target's
  argmax differs from the base's);
- agreement with independent references: R800, and the BT4-it332 and Ceres C3 raw
  policies banked for the same rows in `rl-audit/external_refs.npz` (*review*,
  `outputs/indep.txt`).

| Net | Entropy | Learned | Broken | Net flips | Top-1 R800 | Top-1 BT4 | Top-1 C3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SL base | 1.124 | — | — | 0 | 0.6666 | 0.7091 | 0.6909 |
| P1 c01 | 1.022 | 82 | 51 | **+31** | 0.6740 | 0.7143 | 0.6974 |
| P1 c02 | 0.941 | | | +5 | 0.6718 | 0.7058 | 0.6925 |
| P1 c06 | 0.828 | | | −2 | 0.6724 | 0.7010 | 0.6880 |
| P1 c12 | 0.793 | 172 | 192 | **−20** | 0.6685 | 0.6896 | 0.6857 |
| F12 c12 | 0.795 | 167 | 195 | −28 | 0.6675 | 0.6899 | 0.6873 |

- **By c12, rows broken (192) exceed flips learned (172).** Net flips peak at +31 at
  c01 and are ≤ +5 from c02 on.
- Under BT4, top-1 rises at c01, then falls below the base by c04 and keeps falling.
  Under R800 it stays flat, and the base ranks last of all 24 banked nets.
- **Sharpening is a ratchet at fixed search settings** (interim). Target entropy is
  about 0.7 × the prior's: base prior 1.11 → target 0.74; P1 prior 0.79 → target
  0.56, on the same Q.

### Every loop null so far used an optimizer the owner never tuned

The diagnosis's finding, partly re-checked here:
- **F12, P1 and Q50** all trained with NAdamW: torch NAdam, betas (0.9, 0.98), eps
  1e-7, decoupled weight decay 1e-4 on 190 weight tensors and 0 on 289 bias and norm
  tensors.
  - They used one flat LR of 3e-6 on both groups, batch 256, no gradient clipping
    and a fresh optimizer state.
  - Re-checked here: the F12 contract's `batch_size` 256, and the P1 log's groups
    `[(3e-06, 0.0001, 190), (3e-06, 0.0, 289)]`.
- **The SL base was trained with production's Aurora split.** Read here from the
  SL base file's saved optimizer state:
  - 48 matrices (scope `mlp_out`) on Aurora at LR 6e-5 (20 × 3e-6), with momentum
    0.95, Nesterov, weight decay 1e-4;
  - the other 431 tensors on AdamW at 3e-6, betas (0.9, 0.95), split into groups of
    142 (weight decay 1e-4) and 289 (no weight decay).
- So every loop null used an optimizer, split and batch the owner never tuned, at
  half his tested batch of 512. This is a confound of the earlier nulls, which
  Stage 1 removes by default.

### Target pricing depends on the ruler (review; spot-checked here)

Maximum top-1 gain over the base prior if every flip were learned and nothing
broken, against each reference. From `rl-audit/search_bank.npz`, `positions.npz`
and `external_refs.npz`; this record recomputed the BT4 column independently
(stored +0.68, no-rescale +3.67, R256 −0.49 pp):

| Target | Flip rate | vs R800 | vs BT4 | vs C3 | Flip rows right, target / base (BT4) |
|---|---:|---:|---:|---:|---:|
| Stored 64-sim (rescaled) | 24.0% | +8.4 pp | +0.7 | +1.4 | 41% / 38% |
| R256 | 26.8% | +11.9 | −0.5 | −0.1 | 38% / 40% |
| R64, no min-max rescale | 15.4% | +6.0 | **+3.7** | **+4.4** | 52% / 28% |

- Against R800, deeper search looks like a better target. Against BT4 and C3 raw,
  the stored target is barely better than the prior, R256 is no better, and
  no-rescale has the largest gain.
- **The deep-SF cross-check disagrees** (interim, `search-compose/sf_scores.json`,
  n = 127 leak-free rows, unpaired, still being labelled; *review*):

  | Move source | Deep-SF regret (cp) |
  |---|---:|
  | SL base prior | 42.6 |
  | Stored target | 25.2 |
  | R64 (rescaled) | 16.3 (blunder rate 1.6%) |
  | R64 no-rescale | 25.5 (blunder rate 3.9%) |
  | R256n | 23.5 |
  | R800 | 14.6 |
  | BT4 raw | 18.1 |
  | P1 c12 prior | 29.5 |

  On deep SF the stored target is much better than the prior, no-rescale is worse
  than the rescaled R64, and P1 c12 is better than the base, the opposite of its
  64-sim result. At n = 127 this decides nothing.
- **Target pricing is unresolved** until the full deep-SF labels and the searched
  BT4 ruler land. That is why all three targets play a match in Stage 1 instead of
  being screened out offline.
- Neither reference is truth. R800 agrees with BT4 less (0.682) than the base prior
  does (0.709).
- On the nets with known match Elo against the base, the rulers disagree. The known
  results: P1 c12 −3.9 / −10.9 at 64 sims; F12 c12 −20.0 (the
  [twelve-cycle record](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md)'s
  M12 read) and −34.9 (the takeover record's D4 table); Q50 c12 −63.2 (−81.1 to
  −45.7) (D4). The rulers rank them as follows:
  - R800 ranks the base last on the in-sample 3,080 rows, which is wrong; on the
    leak-free rows it ranks the base first (0.658 > P1 c12 0.649 > F12 c12 0.647,
    [above](#what-training-did-interim-re-read-from-flip_by_cycle_v2json));
  - BT4 and C3 rank the base first, which is right, but tie P1 and F12;
  - stored-target agreement orders the first three correctly, but P1 − F12 is 8
    rows at SE 8, and it is the training target.
- **No offline ruler has been shown to predict a positive at 64 sims.**

**What the previous version of this plan concluded, and why it is withdrawn.** It
said "the learner is the binding constraint, not target quality" and "per sim, 64
beats 256". Both rest only on the R800 ruler. The diagnosis has withdrawn both as
ruler-dependent (interim 6). No target is better than the others on every ruler.

**Candidate mechanism** (interim; to be tested, not assumed). Self-distillation
overwrites the SL prior's independent information with the base value head's
opinions. A 64-sim search re-applies those opinions anyway, so the update is
redundant at 64 sims, and it is a loss against external rulers. If this is right,
the loop gains only through a target that carries information the prior lacks, or
through a value head that learns new information. Stage 1's cell b512 tests the
first branch; the value branch is the parked value-first cell.

## The readout protocol

- **Judged at 64 sims.** Every strength claim is a seat-balanced match against a
  fixed opponent at 64 sims (loop-exp1's arena command, `--search-shape training`,
  temperature 0.1, no noise), pooled by the pooling script, with its own registered
  rule.
- **Secondary: temperature-0 low-sim matches** at 1, 4 and 16 sims
  (`--temperature 0`). They are reported, never judged.
- **Never used as evidence:** default-temperature 1-sim matches (veto-confounded).
- **Offline descriptors on fixed rows**, never strength claims: agreement with each
  reference, deep-SF regret once labelled, calibration gap, prior entropy, veto
  rate, mean V − Q(top child), |Q|, P(draw).

## Stage 0b: rulers and target pricing (minutes of compute, plus the registered T1/T2; no training)

**Question.** Which offline ruler orders nets the way 64-sim matches do, and how do
the candidate targets score on it?

**Rows:**
- **A1:** the 3,080 rl-audit rows, minus the 647 whose position also occurs in
  cycle 2–12 games (*review*, `outputs/leak.txt`: 99 of 740 flip rows and 548 of
  2,340 agreeing rows). The key is board, side to move, castling and en passant;
  counts are also reported with history. Both counts are reported.
- **A2:** a fresh audit set of about 3,000 rows from SL-base games started from the
  `a2` block (EVAL lines 800–1,799), which no training data or match uses.

**Labels.**
- **Primary rulers (owner, 2026-11-20): BT4 and Ceres searched at 64 and 256
  nodes** on A1 and A2: the root's most-visited move and visit distribution. The
  diagnosis is producing these; their bank is cited here when it lands. In the
  owner's words: "I think using bt4 or ceres can work instead of a deep stockfish?
  And you could even do those at like 64 nodes or 256 nodes."
  **Searched BT4 and Ceres are a ruler and a distance benchmark only; nothing in
  this plan trains on them.** The owner doubts they make a good target for a smaller net ("it
  might just be too hard, and it may be too sharp").
- **Cross-check column:** deep-SF labels (8M nodes; the diagnosis is starting them,
  interim 6), CPU only, with the production Syzygy pair.
- The raw BT4 and C3 policies already banked (`rl-audit/external_refs.npz`) stay as
  the 1-node column.

**Scored on A1 and A2:**
- **Targets:** stored, R64, R64 no-rescale, R256, R16, R32, stored at train temp
  1.5.
- **Nets:** SL base; P1 and F12 c01–c12; Q50 c12; the headswap hybrids.
- **By:** top-1 vs BT4 and Ceres at 1, 64 and 256 nodes; top-1 vs R800; deep-SF
  regret and top-1 (cross-check); calibration gap (mean p_top1 − top-1 agreement);
  entropy.
- **The disagreement table:** for each pair of rulers, the share of rows where they
  pick different moves, and the rank order each gives the nets.

**BT4 as the known-good sharpness reference.** BT4's raw policy is scored beside
ours, as what a healthy strong net looks like. It is a reference, not a target to
copy: BT4 is a different, larger net. Already computable from the bank, on the 2,433
leak-free rows (`external_refs.npz`, `leak_mask.npz`; computed here):

| Raw policy | Entropy | Mean p_top1 | Agreement / gap vs R800 | Agreement / gap vs C3 |
|---|---:|---:|---:|---:|
| SL base | 1.191 | 0.592 | 0.658 / −0.066 | 0.679 / −0.088 |
| BT4-it332 | 1.512 | 0.523 | 0.672 / −0.148 | 0.814 / −0.291 |
| Ceres C3 | 1.558 | 0.504 | 0.656 / −0.152 | — |

- **On these rulers the strong nets' raw policies are softer and more
  under-confident than ours, not sharper.** A calibration gap of zero is therefore
  not what a strong raw prior looks like.
- Stage 0b repeats this against the searched BT4 and Ceres rulers once they land.
  It adds BT4's value |Q| and draw rate on the probe rows if the adapter exposes the
  WDL head; the banked array holds the policy only.

**Ruler validation rule (pre-committed).** Each ruler is tested, the primary
rulers first. A ruler is *validated* when all three hold:
1. it is not the training target **and is not derived from the SL base's own net
   or search** (so R800, R256 and the stored target are excluded: a ruler that
   shares the target's generator rewards staying close to the base);
2. **negative control:** on A1 ∪ A2 it ranks the SL base above P1 c12, F12 c12 and
   Q50 c12, and Q50 c12 last of the four. Q50 c12 is scored on A1/A2 first (it is
   not in the bank yet, P5). A pair within one game-clustered SE counts as a tie,
   not an error;
3. **positive control:** it ranks the base's 64-sim search move (R64 argmax) above
   the base's raw argmax, paired and game-clustered, with the lower bound above 0.
   `t0_head` gives that gap's Elo.

A ruler that passes (2) but not (3) is recorded as "rewards closeness to the base"
and is not used. Given what is banked (BT4 raw prices the stored target at only
+0.7 pp; deep SF ranks P1 c12 above the base), possibly no ruler passes. Stage 1
does not depend on one: the ruler picks doses only, and all four cells play.

**The diagnosis's learner screen (Stage 0b, held until the owner finishes using the
GPU).** Arms N256, N512, N1024, N4096 (NAdamW), N4096 with linear LR, and A512
(the Aurora split), on the leak-free rows, scored at matched samples and matched
KL. Its bank (`search-compose/`, `batch_screen`) is cited here when it lands. It is
also what answers the legacy question "NAdamW-256 against the Aurora split at 512",
so Stage 1 has no legacy NAdamW cell.

**What Stage 0b decides for Stage 1.** Only which ruler picks each cell's dose
(the validated ruler, else searched BT4 at 256 nodes, labelled unvalidated). It
does not change which cells play.

**The registered matches in this stage (GPU, one arena at a time, about 2 h):**
- **T1 and T2**, exactly as `search-compose/PREREG.md` registers them (seed
  2026112050, then the fresh-seed confirmation if its rule calls for one).
- **True headroom:** `t0_head`, registered in `search-compose/PREREG.sha256.addendum`
  (addendum 2). The SL base at 64 sims against itself at 1 sim, both at
  `--temperature 0`, 400 games per orientation, seed 2026112050. Reported, not
  judged: it replaces the +849 as the headroom number. This plan runs no duplicate.
- The same addendum notes that T1's point prediction (0 to +60) may be too high,
  given the leak-free raw-argmax decline. Its rule is unchanged.

**T1 / T2 branches (pre-written):**
- **T1 refutes H1** (lower bound ≥ +180). The prior did improve at 1 sim, the 1-sim
  exclusion is revisited before Stage 1, and Stage 0's veto paragraphs are
  rewritten.
- **T2 shows a hidden policy gain** (P1 policy + base value, lower bound > 0,
  confirmed on the fresh seed). Value drift is binding: every Stage 1 arm keeps
  `--w-wdl 0` (already the default here) and adds a value co-requirement (G-T2's
  |ΔQ| on the probe rows ≤ the P1 c01 value). A KL-wdl anchor to the SL base
  (`--kl-wdl`, with `--kl-reference` explicit) is added to every policy cell. The
  `--accum-steps` PR refuses accumulation with a KL anchor, loudly (the anchor's
  penalty is a per-micro-batch mean), so the 2048 cell then needs a native 2048
  batch or anchor support for accumulation (code).
- **T2 shows no hidden gain** (upper bound < +20). Stage 1 proceeds unchanged.
- **Either test unread.** Stage 1 proceeds, and the reading is listed as an open
  confound.

**Scoring code.** Stage 0b's scorer lands on main in its own PR, with tests,
before it is used on Stage 1's snapshots. Until then the bank scripts it replaces are pinned by
sha256:

| Script | sha256 |
|---|---|
| `search-compose/scripts/flip_by_cycle.py` | `94a544090fe6f979b33491685287129addb2e89c7bdc221e7875deef81656c4f` |
| `search-compose/scripts/onesim_veto.py` | `5a7131c199ebb4c1a42ae587570baa014a80c2bfed24eb73b83d064cb1b8c71b` |
| `plan-review-1439/scripts/indep.py` | `fb3f5b0d95401bb654d97a4b2a4d4bab244525eb6c368cbbe93009e457df164e` |
| `plan-review-1439/scripts/calib.py` | `daabd793fe1a111c1d3e838090ef17d8a59f0ca2ada319b633fc625ded57369a` |
| `plan-review-1439/scripts/leak.py` | `435c31751268c116016498ca261c79fb333e8b099842cc08159b7efa9ec34ee0` |
| `rl-audit/external_refs.npz` | `3199a6acd82a6e29fbafb4ea4e26d27b161bc821eec2002b737a69e7ce642699` |
| `rl-audit/positions.npz` | `0d335d6487cd702a7e3b1d950fe10ca754a66794e6d42efdc9980f3d357df737` |
| `rl-audit/search_bank.npz` | `ee04cd33264b9917a7561b387d949a1705e467ea6c9d80c036d86d12fe8302e7` |

These bank scripts are hard-wired to the rl-audit rows and `root_priors.npz`, so
they cannot score Stage 1's new snapshots. That tooling is a prerequisite (P5).

## Stage 1: one generation, four pre-committed cells, at 64 sims

**Question.** Does one generation of the project's own self-play training, from the
SL base, gain at 64 sims? Which of three policy targets, and does a 4× larger
effective batch help?

### The collection

One SL-base collection: loop-exp1's `$COLLECT` with `--games 1040 --seed
2026112300`, which gives about 100k rows.
- **Cost:** about 12 min at 0.49M training rows per GPU-hour (64 sims, 3 processes
  × 128 games, `--collect-procs` from PR #1441, not merged). Without it, about 20
  min at the earlier single-process 3,284 games/h. Source: the collect-throughput
  bank `~/chess-artifacts/operations/rl-claude-20261119/collect-throughput/`
  (interim, the measuring agent's figures).
- It must pass loop-exp1's collect gates (G-C1–G-C11), and is frozen once arms
  reference it.

**One collection, three policy targets on the same games.** Two ways were considered:
- **Not used: two same-seed collections, one with `--no-target-q-rescale`.** That
  flag reaches play, not only the target. The collection plays at a positive move
  temperature (`--temp-start 0.5`, `--temp-floor 0.1`), and at a positive temperature
  the played move is sampled from the stored target
  (`selfplay/config.py`, the `gumbel_target_q_rescale` comment;
  `network_turn._resample_actions_with_temperature`). So the games diverge from the
  first sampled ply. The "survivor agreement 1.0" in `target_scores.json` is a
  temperature-0 survivor check and does not carry over.
- **Used: one collection and an offline rebuild.** Simple-RL collects always record
  root statistics (`collect.py`, `record_root_stats=True`):
  `root_log_prior`, `root_child_q`, `root_child_visits` and `root_net_value`.
  `rebuild_improved_policy(..., cfg with target_q_rescale=False)` re-derives the
  no-rescale target from them. The bank's numpy re-implementation
  (`rl-audit/analyze_targets.py::build_target`), fed the *rescaled* search's root
  statistics, reproduces the separately searched no-rescale target to 9.27e-8
  (`target_scores.json`). So the rescale knob changes only the target, not the
  search. In the same file the rescale-on rebuild differs from the stored target by
  up to 7.4e-4; the retarget PR measures main's own rebuild against the shard and
  states its tolerance.
- **The retarget step (code, P5)** writes a **full** collect in its own run dir: a
  sealed `gen_0000` with the same checkpoint, the same openings inputs, the copied
  rows with the rebuilt `policy_target`, and provenance naming the source collect
  and the rebuild. `--collect-from` accepts it (`precheck_source` refuses a
  reference-of-a-reference, so it cannot be a reference collect). Rows that never
  had a search (exact-shortcut and tablebase rows) keep the stored target and are
  counted; a searched row missing its root statistics refuses the step. The
  output's digest differs from its source's.

### The four cells

All four are policy only (`--w-wdl 0`), at the SL base's Aurora split with fresh
optimizer state, on the same collection:

| Cell | Policy target | Effective batch | Base `lr` | Matrix LR (20×) | Updates on ≈ 95.6k rows |
|---|---|---:|---:|---:|---:|
| **a512** | (a) stored, min-max rescaled | 512 | 3e-6 | 6e-5 | ≈ 187 |
| **b512** | (b) no-rescale, rebuilt offline | 512 | 3e-6 | 6e-5 | ≈ 187 |
| **c512** | (c) stored at `--train-target-temp 1.5` (entropy 0.693 → 0.999, `target_scores.json`) | 512 | 3e-6 | 6e-5 | ≈ 187 |
| **a2048** | (a) stored | 2048 (512 × `--accum-steps 4`) | 1.2e-5 | 2.4e-4 | ≈ 47 |

- **Why these.** The deep-SF cross-check and the independent raw-policy rulers
  disagree on how to price the targets
  ([above](#target-pricing-depends-on-the-ruler-review-spot-checked-here)), so
  all three play a match. a2048 is the owner's averaging hypothesis as one cell, not
  a batch axis.
- **What averaging can and cannot fix.** A larger batch averages away random error
  in the targets. It does not remove a consistent bias, such as the measured
  sharpening (target entropy ≈ 0.7 × the prior's, interim). That is why (b) and (c)
  are separate cells.
- **LR co-scaling (a guess): linear, anchored at 3e-6 for batch 512.** 3e-6 is
  P1/F12's fine-tuning LR. For Aurora's fixed-norm matrix steps, linear scaling
  restores equal movement per sample; for the AdamW groups it is approximate. The
  SL run's schedule peaked at 3e-5 base (matrix 6e-4) and decayed; a2048's 1.2e-5 is
  below that peak.
- **Start-up.** Each cell starts a fresh optimizer with warmup 0. a2048's first
  steps run at full matrix LR from an empty Nesterov buffer, and its momentum
  window (about 20 steps) is about 40% of its 47 updates. Recorded as a confound.
- **Cut from the previous version** (review, section 5): batch 256 and 1024 with
  Aurora, batch 4096 (23 updates spent mostly in the momentum start-up), the legacy
  NAdamW-256 cell (the diagnosis's `batch_screen` covers it), the continued-state
  cell (O2's alternative), and the value-first cell (parked).

### The optimizer: the owner-tested setup

- **The Aurora split as the SL base used it.**
  - The flags: `--optimizer aurora --matrix-optimizer-scope mlp_out`, with the
    Trainer's `matrix_lr_multiplier` of 20 (production's value). At `--lr 3e-6` the
    matrix group runs at 6e-5 and the AdamW groups at 3e-6, the SL run's end LRs.
  - The owner tuned this split and found 512 better than 256
    (`configs/pbt2_small.yaml`). It is taken as given, not screened.
- **What Aurora does** (`train/aurora.py`, `AuroraWithAuxAdam`; the owner: "aurora
  is built on muon on top of adamw"). The matrix group's update is the
  orthogonalised Nesterov momentum, a fixed-norm step whose size the LR alone sets.
  The other groups are AdamW.
- **The driver does not yet reproduce the SL run's optimizer** (code, P5):
  - the SL base's saved groups carry `aurora_pp_iterations` 3, `aurora_pp_beta`
    0.25, `aurora_polar_steps` 8, `aurora_polar_method` `polar_express` and
    `aurora_polar_dtype` fp16; the driver passes none, so the Trainer's defaults
    apply (2, 0.5, 12, `simple`, `auto`);
  - the SL run had zclip on (the checkpoint's `zclip`: initialised, warmup 25 steps);
    the driver forces `grad_clip_mode: none`;
  - the optimizer PR exposes the five knobs and a zclip option, defaulting from the
    init checkpoint, and records the realised param-group table (count, LR, betas,
    weight decay, optimizer class per group) and the realised clip mode in the
    manifest. A test asserts the table equals the SL checkpoint's.
- **The SL run's LR schedule decayed; the driver's is flat.** Not reproduced; a
  confound.
- **Realised values are read off the optimizer's param groups** in each run's
  metrics, never off the flags (P8).

### Snapshots, in samples

Snapshots are defined by rows presented, so the two batch sizes are read at the
same samples: **5,120; 10,240; 17,920; 35,328; 70,656** and the final model
(`--snapshot-rows`, from the `--accum-steps` PR). That is updates 10, 20, 35, 69
and 138 at batch 512. At 2048 each resolves to the first optimizer step whose rows
reach it (updates 3, 5, 9, 18, 35; at most +20% more rows, at the first one); the
resolved updates are recorded in the manifest. KL(SL base ‖ new) on the probe rows
(`probe_vs_previous.policy_kl_prev_to_new`, Σ p_old (log p_old − log p_new)) is
recorded at each snapshot, descriptively.

### Dose choice (the only thing the offline ruler decides)

Per cell, the played model is the snapshot with the highest **A1** top-1 against
the dose ruler, among snapshots inside the
[calibration band](#calibration-band-used-by-the-dose-choice-and-by-stage-2).
Snapshots within one game-clustered SE of the best count as tied, and the earliest
of them is played (less movement). If no snapshot is inside the band, the one
nearest the band is played. A2 is reported and never used to choose.
- **The dose ruler** is the Stage 0b-validated ruler, else searched BT4 at 256
  nodes, labelled unvalidated.
- The scorer is a P5 prerequisite. If it has not landed, every cell plays its
  final model, pre-committed.

### Exact commands

```bash
COLLECT1="${COLLECT/--games 16000/--games 1040}"; COLLECT1="${COLLECT1/--seed 2026112000/--seed 2026112300}"
python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/collect --generation 1 $COLLECT1
chmod -R a-w $R/s1/collect
# target (b)'s source: the retarget step (P5) writes the full sealed collect $R/s1/collect_norescale
#   (gen_0000 + gen_0001/collect), policy_target rebuilt with target_q_rescale=False.
AUR="--optimizer aurora --matrix-optimizer-scope mlp_out <SL-run Aurora knobs + zclip, from the init checkpoint (flag names from the optimizer PR)>"
BASE="--presentations 1 --mirror-prob 0.5 --warmup-steps 0 $AUR \
  --optimizer-state fresh --init-optimizer-state fresh \
  --holdout-frac 0.05 --probe-rows 2048 --no-eval \
  --snapshot-rows 5120,10240,17920,35328,70656"
cell() {  # $1 = name, $2 = source collect run dir, $3 = accum steps, $4 = lr, rest = arm flags
  local n=$1 src=$2 k=$3 lr=$4; shift 4
  local T="$BASE --batch-size 512 --accum-steps $k --lr $lr --lr-floor $lr"
  python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/$n --generation 1 \
    --collect-from $src $COLLECT1 $T "$@"
  python -u scripts/simple_rl_loop.py run --run-dir $R/s1/$n --generations 1 \
    --collect-from $src $COLLECT1 $T "$@"
}
cell a512  $R/s1/collect            1 3e-6   --w-wdl 0
cell b512  $R/s1/collect_norescale  1 3e-6   --w-wdl 0
cell c512  $R/s1/collect            1 3e-6   --w-wdl 0 --train-target-temp 1.5
cell a2048 $R/s1/collect            4 1.2e-5 --w-wdl 0
```

The command is final once the optimizer PR fixes its flag names.

### Registered matches: all four cells

Each cell's dose-chosen model plays 2,000 games against the SL base at 64 sims,
seat-balanced:
- **Openings:** the 500-line `s1_match` block.
- **Arena:** loop-exp1's `$COMMON` with `--games 1000 --openings-fen <block>` in
  place of `--openings`/`--opening-plies`, two runs on one seed with the cell as
  candidate (A) and as reference (B).
- **Pooling:**
  ```bash
  python scripts/arena_seat_balanced_pool.py --arm <model.pt> \
    --a-games <A.games.jsonl> --b-games <B.games.jsonl> \
    --a-result <results.jsonl> --b-result <results.jsonl> \
    --a-label <tag>_A --b-label <tag>_B --verdict-rule lb-gt-zero --z 2.498
  ```
- **z:** z = 2.498, Bonferroni over four matches (Φ⁻¹(1 − 0.025/4)).
- **Reported alongside, not judged:** temperature-0 reads at 1, 4 and 16 sims (800
  games each, on `s1_low`), and the offline descriptors.

**Reading and dispositions (pre-committed, per match):**
- **Success:** lower bound above 0. Stage 2 launches on that cell; with several, on
  the one with the highest pooled estimate.
- **Kill:** upper bound below 0. The cell is recorded as harmful at this dose.
- **Unread:** one fresh confirmation, with the same size, on `s1_confirm`, read
  alone at z = Φ⁻¹(1 − 0.025/m), where m is the number of cells taken to
  confirmation. If it is still unread, it is recorded "unread at N = 2,000".
  **Each cell can get two looks, so the family-wise one-sided error is up to about
  5%, not 2.5%.** Stated, and accepted for a first read.
- **No success in any cell:** Stage 2 does not launch. The parked BT4-target
  diagnostic's launch condition is then met; O3's larger collection with the same
  four cells is the alternative.

**Power.** The banked P1 64-sim match gives a per-cluster SD of d of 0.195 (200
clusters, ρ = +0.12; *review*). That is an SE of 6.06 Elo at 500 clusters, and an
80%-power MDE of (2.498 + 0.842) × 6.06 = **20.2 Elo** at z 2.498, about 10%
larger at ρ = +0.36.

**Compute (GPU hours):**
- collect: about 0.2 h (0.33 h without #1441); the retarget step is CPU;
- training: four cells, minutes each;
- dose scoring: CPU minutes;
- matches: 1.1–1.6 h per 2,000 games, so **4.4–6.4 h for four**; confirmations add
  up to the same again; the temperature-0 low-sim reads are unmeasured and small;
- total about 5–7 GPU hours without confirmations, up to about 12–14 with all four.

### Calibration band (used by the dose choice and by Stage 2)

The gap (mean top-1 probability − top-1 agreement) against the dose ruler must
stay within [g_BT4, +0.02]. g_BT4 is BT4's raw gap on the same ruler (−0.148
against R800, Stage 0b).
- The upper edge stops over-confidence, the direction the loop drifted (P1 c12 is
  +0.034 against R800).
- The lower edge allows a prior as soft as a known strong net's, but no softer.
- BT4 is a reference point, not a target: it is a different, larger net. Both
  edges are guesses.
- Entropy is recorded beside the gap.

**Later, only after Stage 1 has read:**
- learner recipes (anchor, R1 per-row entropy matching, ranking loss);
- other effective batches (256 and 1024 with Aurora, 4096), if a2048 differs from
  a512;
- search-depth arms at matched GPU time, sized from the measured exchange rate.
  The collect-throughput bank (interim) gives training rows per GPU-hour on the SL
  base, at the best measured multi-process setting each (PR #1441, not merged):

  | Target sims | Rows per GPU-hour | Setting (processes × games) | Share of the forward ceiling |
  |---:|---:|---|---:|
  | 16 | 1.86M | 6×96 or 8×64 | 0.89 |
  | 32 | 0.88M (**provisional**; 8×64 not yet measured) | 3×128 | — |
  | 64 | 0.49M | 3×128 | 0.86 |

  So 16 sims gives about 3.8× the rows of 64 sims per GPU-hour once the process
  count is raised, and 2.8× at 3×128. 100k rows take about 12 min at 64 sims and
  about 3 min at 16. The usable VRAM limit is about 26 GB in total. A 16-sim arm at
  matched GPU time is the owner's choice (O5, default deferred).

These are not planned in detail here.

## Parked, with launch conditions

None of these is on the path to the stated goal. Each launches only on its
condition, with its own record.

| Item | Launch condition |
|---|---|
| Distance to BT4 at 1 node (formerly Stage 0c) | The owner asks for it, or Stage 2 has a confirmed gain and a concrete gap number is wanted |
| BT4 raw-policy target diagnostic | Every Stage 1 cell fails its match (kill, or unread after confirmation) |
| Value-first cell (d) | Stage 1 has read, and either T2 shows value drift is binding or the owner asks; it then gets a pre-committed match slot, never an offline eligibility rule |

The parked designs, kept for when they launch:

### Distance to BT4 at 1 node (parked)

The owner's milestones, in order:
1. a measured gain from the loop at 64 sims;
2. "our 800-sim search matches BT4";
3. long term, closing the node handicap to 1:1.

"i think we are too far away from bt4 to use it as a good anchor yet", so BT4 is
not a per-readout anchor. One optional reading gives the gap in concrete terms.
- **Pairing:** the SL base at 800 sims (UCI, the arena's training-shape flags,
  `go nodes 800`) against BT4-it332 at 1 node (its raw policy).
- **Engine:** the existing CPU binary `~/local_engines/lc0/lc0` (v0.32.1, `eigen`
  backend; about 1 s per move at 1 node). The local lc0 builds are CPU-only, so
  BT4 above 1 node is out of scope here. A wrapper around the diagnosis's
  `ext_search.py`, or a CUDA lc0 build, would be needed for that. Neither is
  proposed now, and a CUDA build would need the owner's go-ahead.
- **Harness:** `scripts/match_vs_uci.py` (cutechess is broken here), 200 games on
  100 `anchor` lines, colours swapped. The production Syzygy pair for our side and
  for adjudication (`--syzygy-adjudicate-path`); lc0 gets `SyzygyPath` too.
- **Reading:** the score with its pair-clustered 95% CI, in Elo. If the upper bound
  is below 0.15 (about −300 Elo), the answer is recorded as "BT4's raw policy beats
  our 800-sim search by more than about 300 Elo", and nothing more is played.
- **Cost:** about 1 s per BT4 move plus our 800-sim moves, a few GPU-minutes; it is
  unmeasured until it runs.

### BT4 raw-policy target diagnostic (parked)

**Launch condition: every Stage 1 cell fails its match** (kill, or
unread after its confirmation). It separates two causes: "the learner or pipeline
cannot absorb a good target" and "our targets are not good enough".
- **Setup:** the same Stage 1 collection, snapshots and policy-only setup, at batch 512. The
  policy target is BT4-it332's raw (1-node) policy on those positions, which the
  owner called the safest distillation. BT4's raw entropy is about 1.5 against our
  base's 1.2, so it does not sharpen.
- **Its net is never a candidate and never an initialisation for Stage 2.** This
  record never presents it as a result. The run's strength must come from its own
  pipeline (see the principle at the top).
- **Match:** one registered seat-balanced 2,000-game 64-sim match against the SL
  base on `confirm`, which is free because Stage 2 did not launch; read alone at
  95%. Played model: the final model.
- **Reading:**
  - A gain means our targets are the binding problem, and the next step is target
    work (depth arms at matched GPU time; interior-node targets).
  - No gain means the learner or pipeline is suspect, and the next step is a
    pipeline audit before any recipe screen.
- **Cost and implementation:** BT4 raw forwards via the bank's
  `rl-audit/external_refs.py` path (ONNX, about 43 ms per row on 8 CPU threads,
  about 1.2 h per 100k rows, an estimate). It also needs a supplied-array mode in
  the retarget step. P9 covers the history rep_fix regime and the 1858 / castling
  remap checks.



## Openings: disjoint blocks

The EVAL split (4,000 lines) is cut once, in `eval.jsonl` order. No training
collection draws from it.

| Block | Lines | Use |
|---|---|---|
| `reserve` | 0–99 | unused (the true-headroom match is the diagnosis's `t0_head`, on its own seed) |
| `s1_match` | 100–599 | every Stage 1 registered match |
| `s1_low` | 600–799 | Stage 1 temperature-0 reads |
| `a2` | 800–1,799 | game starts for Stage 0b's A2 audit set (SL-base self-play, about 3 rows per game) |
| `s1_confirm` | 1,800–2,299 | Stage 1 unread confirmations |
| `anchor` | 2,300–2,499 | the parked distance-to-BT4 match, the Stockfish K tuning ladder and the Stage 2 Stockfish anchor |
| `readout_base` | 2,500–2,999 | every Stage 2 R-base readout |
| `confirm` | 3,000–3,499 | Stage 2 deliverable confirmation |
| `confirm2` | 3,500–3,999 | a second deliverable confirmation, if the first fails |

- A seat-balanced match of G games on a block of G/4 lines plays every line once
  per orientation (`--openings-fen`, `--games G/2` per run). The 200-line
  blocks play 800.
- Blocks shared by several matches (`s1_match`, `readout_base`, `anchor`) make those matches
  correlated with each other. Each match's own read is unaffected.
- Cutting code: loop-exp1's, run verbatim on the banked split with other
  boundaries. Each block loaded in the arena's FEN loader as unique rows, all with
  move history.

## Stage 2: repeat the winning recipe (short until Stage 1 has a winner)

Launches only on a Stage 1 success, with that cell's target, effective batch, LR
and dose.
- **Driver:** one run dir, `run --generations G`, `--init-checkpoint` = the Stage 1
  winner (copied, sha-verified, write-protected).
  - Each generation is Stage 1's collection size at 64 sims, with
    `--presentations 1` and `--optimizer-state fresh`.
  - `--opening-sampling cycle`, `--seed 2026112200`, `--no-eval`.
  - `--kl-reference <SL base copy>` is passed explicitly if any KL term is on; it
    defaults to generation 0, which here is the winner.
  - **If b512 wins,** Stage 2 needs a per-generation retarget hook inside
    `run --generations G` (a driver change, separate PR, written only if (b) wins).
    The standalone retarget step is not enough.
- **Gated every generation (hard gates):** loop-exp1's collect gates (G-C1 = 0,
  G-C2 equal, G-C3 ≥ +0.50, G-C7 ≥ 80, G-C10 ≤ 0.05) and G-T2 / G-T4. A failure
  allows one re-collect, then stops the run at g − 1.
- **Drift alarms (tests, not point estimates):**
  - **Ruler drift:** stop only if the A2 top-1 against the dose ruler is below the
    SL base's with the game-clustered upper 95% bound below 0, at two consecutive
    generations. At a true difference of 0, each generation trips with probability
    0.025, so a stop needs two in a row: about 0.06% per pair and about 1% over 15
    generations, assuming independent generations. Generations share the A2 rows,
    so the realised rate is higher; it is re-estimated from the first readouts.
    (A point-estimate gate would stop a neutral run about half the time per
    generation, review N5.)
  - **Calibration:** stop only if the gap leaves the band by more than 2 SE at two
    consecutive generations.
  - Matches remain the only deciders.
- **Readouts:** on a GPU-time schedule, with an arena share ≤ 30%. The next readout
  comes once the generations' GPU time is at least 7/3 × the last readout's arena
  time: about every 6–9 generations at ≈ 0.4 h per generation (20-min collect), and
  about every 9–15 at ≈ 0.25–0.3 h with #1441 (12-min collect plus 187 updates).
  - Each readout is R-base: 2,000 games at 64 sims, seat-balanced, against the SL
    base, on `readout_base`.
  - Beside it, the second anchor: Stockfish at a fixed node count K, near the net.
    Our agent runs via UCI with R-base's arena settings at `--nodes-a 64
    --enforce-nodes-a`; Stockfish runs at `--nodes-b K --enforce-nodes-b`. 400
    games on `anchor`, with the production Syzygy pair for both.
  - K is tuned before Stage 2 so the SL base scores 40–60%. It is re-tuned only at
    a readout where the net leaves 25–75%, and that readout plays the old and new K
    so the scale chains.
  - Verdicts read both anchors (the single-anchor trap: anchored Elo depended on
    the anchor by up to 82 Elo in one recorded design).

### Stop rule (pre-committed, simulated)

The run stops at the first of:
1. the R-base point estimate at readout r is not above the one at r − 3, at two
   consecutive readouts (first possible at readout 5);
2. the R-base upper 95% bound below 0 at two consecutive readouts;
3. a hard-gate failure that one re-collect does not clear, or a drift alarm;
4. the owner's time cap (O4).

**Simulated error rates.** Each readout's R-base estimate is drawn as N(r · δ, 6.06²),
independently across readouts. δ is the true gain per readout. Readouts share one
block, so their noise is positively correlated; that shrinks the variance of the
differences rule 1 reads, so the independent model overstates its false-stop rate.

| True gain per readout | Rules 1–2: stop by readout 6 | Stop by readout 12 | Previous rule (R-prev, two UB < 0 or three unread): stop within 10 |
|---:|---:|---:|---:|
| −5 | 99.8% | 100% | — |
| 0 | 37.6% | 86.3% | 99.9% |
| +5 | 0.3% | 1.2% | 98.8% |
| +10 | 0.0% | 0.0% | 74.0% |
| +18 | 0.0% | 0.0% | 2.6% |

The previous rule's column reproduces the review's `stop.py` (sha256
`2b77ca2a1fddd1f3797a41777645823b4c13933361fc60cf4901f42645458b6b`). The rules 1–2
column comes from this code:

```python
import numpy as np
rng = np.random.default_rng(20261120); se = 6.06; N = 100_000; R = 12
def stops(delta):
    est = rng.normal(delta * np.arange(1, R + 1), se, (N, R)); stop = np.full(N, R + 1)
    for r in range(R):
        r1 = (r >= 4) and (est[:, r] - est[:, r - 3] <= 0) & (est[:, r - 1] - est[:, r - 4] <= 0)
        r2 = (r >= 1) and (est[:, r] + 1.96 * se < 0) & (est[:, r - 1] + 1.96 * se < 0)
        s = (np.asarray(r1) | np.asarray(r2)) & (stop == R + 1); stop[s] = r + 1
    return np.mean(stop <= 6), np.mean(stop <= R)
for d in (-5, 0, 5, 10, 18): print(d, stops(d))
```

At a true plateau (δ = 0), the rule stops within 12 readouts 86% of the time (the code above, run verbatim, prints these values). O4
bounds the rest.

**Deliverable:** the readout net with the highest R-base estimate among those with
a lower bound above 0. That choice carries winner's curse, so it is re-measured
once: 2,000 games at 64 sims, seat-balanced, against the SL base on `confirm`, read
at 95%.
- **Confirmed:** lower bound above 0.
- **Not confirmed:** recorded with both numbers. The previous qualifying readout
  net, if any, is confirmed next on `confirm2`. Otherwise the run's result is
  "no confirmed gain".

### Recovery

- Every generation's model and Trainer state are kept; nothing is pruned.
- To resume from the last passing generation g, use `--redo --redo-downstream` of
  generation g + 1's collect, or start a new run dir from generation g's model.
- The SL base, the live checkout, production and earlier banks are untouched.
  Nothing is promoted.

### What success means at one week

Within 7 days of Stage 2's start, the deliverable's confirmation reads a 95% lower
bound above 0 and at least +50 Elo at 64 sims. +50 is a guess, replaced before
Stage 2 by a projection from Stage 1's gain (halved, for winner's curse).

## Why no sizing from loss curves

The experiment ledger (2026-08-23) records three fit or audit gains that did not
transfer to Elo:
- F-only: −0.043 nats, for 0 cp;
- absorption: 11.6× the fit, for 0 Elo;
- v020: +3.1 cp, for −9.56 Elo (−29.59 to +10.41).

ft595 traded Elo for fit: −41.0 (−71.3 to −11.4).

This plan adds two lessons:
- A match setting can manufacture a large Elo gap (+241 at 1 sim) that says nothing
  about the net's argmax.
- An offline ruler can invert against independent references (R800 vs BT4/C3).

So offline scores only pick a cell's dose; which cells play is pre-committed, and
only registered 64-sim matches decide.

## What is still a guess

| Number | Where it is used | The measurement that replaces it |
|---|---|---|
| Every *interim* number in Stage 0 (veto weights, the 0.7× ratchet) | direction | the diagnosis's full report, T1 / T2 |
| The ruler validation rule, including the 1-SE tie rule and the positive control's paired lower bound | Stage 0b, dose ruler | Stage 1's matches |
| The fallback dose ruler: searched BT4 at 256 nodes, unvalidated, with `ext_search.py`'s PUCT constants | Stage 1 dose | Stage 0b's validation |
| A2: about 3,000 rows, about 3 per game, from the `a2` block | Stage 0b validation, reporting, Stage 2 drift alarm | — (a budget) |
| 1,040 games / ≈ 100k rows | Stage 1, Stage 2 | Stage 1's read and O3 |
| Linear LR scaling anchored at 3e-6 for batch 512 (matrix LR 20× = 6e-5, the SL run's end LRs); 1.2e-5 at 2048 | a2048 | the Stage 0b learner screen; a2048's match |
| Effective batch 2048 as the one large-batch cell | a2048 | its match |
| Snapshot rows 5,120 … 70,656; the 1-SE tie rule picking the earliest; nearest-to-band fallback | Stage 1 dose | Stage 1's held-out curves |
| `--train-target-temp 1.5` | target (c) | Stage 1 |
| Calibration band [g_BT4, +0.02] | dose choice, Stage 2 alarm | Stage 1's realised gaps |
| Four matched cells; z 2.498; the confirmation's z for m cells | Stage 1 | — (a budget) |
| Zclip on (as the SL run had it), from the init checkpoint; flat LR (the SL run decayed) | every cell | — (stated confounds) |
| The Stage 2 drift alarms (upper bound < 0 twice; band ± 2 SE twice) | Stage 2 | its first readouts |
| Arena settings: 64 sims, temperature 0.1, no noise, `--search-shape training` | every match | — (loop-exp1's arena) |
| Arena share ≤ 30% (7/3 factor) | Stage 2 schedule | Stage 2's realised times |
| Stop-rule constants (lag 3, two consecutive) and SE 6.06 | Stage 2 | Stage 2's realised clustered SE |
| The Stockfish K band (40–60%), the 25–75% re-tune trigger, 400 games | Stage 2 anchor | the tuning ladder and first readouts |
| G-C3 ≥ +0.50; holdout 0.05; probe 2,048 | gates | not tuned |
| +50 Elo at one week, and "halved" | success line | Stage 1's projection |
| 0.49M rows per GPU-hour at 64 sims (3,284 games/h single-process); 96.8 rows per game; 1,274–1,800 arena games/h | compute | Stage 1's own runs |
| The exchange rate: 1.86M / 0.88M (provisional) / 0.49M rows per GPU-hour at 16 / 32 / 64 sims | later depth arms | the full collect-throughput report, after #1441 |
| BT4 raw forward cost (about 43 ms per row on 8 CPU threads) | the parked BT4-target diagnostic | its own run, if it launches |

## Pre-launch checks

- **P1.** Every stage runs from a detached worktree at a pinned main sha, with
  extensions rebuilt; the sha is recorded here.
- **P2.** A1 excludes every audit position that occurs in cycle 2–12 games, by
  position key; both counts are reported. A2 shares no position key with any
  training collection.
- **P3.** The SL base is copied and write-protected (loop-exp1 step 0a). The split
  is reproduced (step 0b). The blocks are cut, with their sha256 recorded.
- **P4.** The diagnosis's full report and T1 / T2 are read, and every interim
  number here is replaced or confirmed.
- **P5.** These have merged and were run once on a small collect. Owner of each:
  the implementer working this plan (Claude), CPU only unless stated.
  - **`--accum-steps` and `--snapshot-rows`** (a2048; snapshots in samples), tested
    so an accumulated step's gradient equals the native big batch's on CPU. It
    refuses a KL anchor and an SL share under accumulation. Default 1.
  - **The optimizer PR:** the five Aurora knobs, per-group LRs and a zclip option,
    defaulting from the init checkpoint, fresh or continued state, with the realised
    param-group table and clip mode in the manifest, tested equal to the SL
    checkpoint's table.
  - **The retarget step:** a full sealed collect with the rebuilt no-rescale target,
    lineage to its source and a distinct digest, accepted by `--collect-from`;
    tested against a searched no-rescale target and, rescale on, against the
    shard's `policy_target`, at the tolerance the PR measures. The per-generation
    hook for Stage 2 is written only if b512 wins.
  - **A snapshot scorer** (a small script PR): forwards any model on A1/A2 and
    scores it against the ruler labels (top-1, calibration gap, entropy,
    game-clustered SEs). Needed for the dose choice; without it every cell plays
    its final model. CPU, minutes per snapshot.
  - **Q50 c12 scored on A1/A2** with that scorer (validation needs it). CPU minutes.
  - **The A2 builder** (same script PR), with an exact command: SL-base games from
    the `a2` block, about 3 rows per game, the P2 position-key check. Its labels:
    searched BT4/Ceres via the diagnosis's `ext_search.py` (defaults to
    `--device cuda`: GPU, or slow CPU), and deep-SF labels at 8M nodes (CPU-hours,
    unmeasured). A2 is reporting and Stage 2's drift alarm; it does not block
    Stage 1's launch.
- **P6.** For the Stockfish anchor and the parked distance-to-BT4 match, the UCI engine's
  settings match the arena's training-shape search on fixed positions. If they
  cannot be matched, both are reported as measuring the UCI agent.
- **P7.** Stage 0b's scores on the banked rows use the scorer on main or the bank
  scripts at the sha256 listed in Stage 0b; Stage 1's snapshots need the scorer
  (P5), which the bank scripts cannot replace.
- **P8.** Each run's metrics carry every param group's realised LR. The batch arms'
  values are checked against the cell table before the dose is chosen.
- **P9.** Only if the parked BT4-target diagnostic launches, its BT4 policy array:
  - the rep_fix regime is applied and stamped;
  - 0 index-convention mismatches and 0 support outside the legal mask;
  - an independent gather check over castles and promotions;
  - its sha256 is recorded.

## Confounds

- The R800 reference is the SL base's own 800-sim search; BT4 and C3 are other
  nets' raw policies. None is truth; Stage 0b's validation rule is the arbiter.
- The interim diagnosis ran on nets trained under the old loop and the P1 harness,
  with a continued optimizer. Stage 1 uses fresh simple-RL collects and a fresh
  optimizer.
- P1's snapshots confound dose with data vintage (snapshot k trained on rows from
  F12 c01…c(k−1)). Stage 1's single collection removes this.
- a2048 against a512: linear LR scaling is exact only for Aurora's fixed-norm
  matrix steps; the AdamW groups' steps also depend on gradient noise. a2048 spends
  about 40% of its 47 updates inside the matrix momentum's start-up. Dose is read
  at matched samples, not matched movement.
- **Zclip:** the SL run had zclip on. The optimizer PR adds the option and
  defaults it from the init checkpoint; until then the driver forces clipping off,
  and any cell run that way says so.
- **Schedule:** the SL run's LR decayed from a 3e-5 peak; every cell runs a flat LR.
- The earlier nulls (P1, F12, Q50) ran NAdamW at batch 256, not the SL base's own
  Aurora split at 512. Stage 1 differs from them in optimizer and batch as well as
  in the collection. The diagnosis's `batch_screen` (N256 against A512) measures
  the optimizer's part offline.
- Selecting the played snapshot is selection. It is done on A1, never on A2 or a
  match, and the matches are on their own block.
- loop-exp1's confounds (twofold-repetition draws, the same-coloured-bishops gap)
  apply.
