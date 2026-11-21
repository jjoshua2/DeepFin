# RL run 1: a measured multi-generation self-play run from the SL base (plan)

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
diagnosis's full report and its registered tests T1 and T2, neither of which has
run (`results/` and `pool/` are empty). Numbers re-read here from bank files say
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
- BT4 and Ceres are used only as a ruler, a sharpness and calibration reference,
  and a distance benchmark (Stage 0c). Nothing in this plan trains on them, except one
  conditional diagnostic whose net is never a candidate
  ([below](#conditional-diagnostic-a-bt4-raw-policy-target)).
- **Out of scope by the owner's direction:** opponent-exploitation and blind-spot
  arms. None is in this plan.

**The goal of this plan:** the first measured self-play gain at 64 sims against the
SL base, and then repeating it. Everything else is cut to what that needs.

**The plan, in order:**
1. **Stage 0** (done): what the banked matches and diagnostics already say.
2. **Stage 0b** (minutes of compute; the rulers are already built): score targets
   and nets on leak-free audit rows against searched BT4 and Ceres, with deep SF as
   a cross-check, and read the registered tests T1 and T2. **It does not gate
   Stage 1,** unless it shows the standard target is worse than the prior.
3. **Stage 1** (one generation from the SL base): one collection, a grid of
   policy target × effective batch size (plus a value-first cell), screened offline
   on held-out positions. At most three cells go to registered seat-balanced 64-sim
   matches.
4. **Stage 2** (only after a Stage 1 success): repeat the winning recipe for
   several generations. It is kept short here until Stage 1 has a winner.
5. **Optional:** a one-off distance to BT4 at 1 node, with the existing CPU lc0
   binary (Stage 0c).

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
| O1 | **Stage 1's grid** | **3 policy targets × 4 effective batch sizes, plus a value-first cell** (below) | Fewer batch sizes (256 and 1024 only) |
| O2 | **Stage 1's optimizer** | **Production's: `--optimizer aurora --matrix-optimizer-scope mlp_out`** (Muon-style on the FFN and attention-output matrices, AdamW on the rest). The owner tuned that split, so it is taken as given, not an arm | `nadamw`, the driver's default, which is what the earlier nulls effectively ran (below) |
| O3 | **Stage 1's collection size** | **1,040 games (≈ 100k rows)** | 2,080 games |
| O4 | **Stage 2's time cap** | **One week** from Stage 2's start | Another cap |

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

- Against R800, deeper search looks like a better target. Against both independent
  references, the stored target is barely better than the prior, R256 is no better,
  and **no-rescale is the only target with a material gain.**
- Neither reference is truth. R800 agrees with BT4 less (0.682) than the base prior
  does (0.709).
- On the nets with known match Elo against the base, the rulers disagree. The known
  results: P1 c12 −3.9 / −10.9 at 64 sims; F12 c12 −20.0 (the
  [twelve-cycle record](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md)'s
  M12 read) and −34.9 (the takeover record's D4 table); Q50 c12 −63.2 (−81.1 to
  −45.7) (D4). The rulers rank them as follows:
  - R800 ranks the base last, which is wrong;
  - BT4 and C3 rank the base first, which is right, but tie P1 and F12;
  - stored-target agreement orders the first three correctly, but P1 − F12 is 8
    rows at SE 8, and it is the training target.
- **No offline ruler has been shown to predict a positive at 64 sims.**

**What the previous version of this plan concluded, and why it is withdrawn.** It
said "the learner is the binding constraint, not target quality" and "per sim, 64
beats 256". Both rest only on the R800 ruler. The diagnosis has withdrawn both as
ruler-dependent (interim 6). R64 no-rescale is the only target that improves on
every ruler.

**Candidate mechanism** (interim; to be tested, not assumed). Self-distillation
overwrites the SL prior's independent information with the base value head's
opinions. A 64-sim search re-applies those opinions anyway, so the update is
redundant at 64 sims, and it is a loss against external rulers. If this is right,
the loop gains only through a target that carries information the prior lacks, or
through a value head that learns new information. Stage 1's arms (b) and (d) are
built around it.

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
rulers first. A ruler is *validated* when two things hold. First, it is not the training target. Second, on A1 ∪ A2 it ranks the SL base
above P1 c12, F12 c12 and Q50 c12, and Q50 c12 last of the four. A pair whose
difference is within one SE (game-clustered) counts as a tie, not an error.

**What Stage 0b decides for Stage 1.** It does not gate Stage 1, with one exception.
If searched BT4 and Ceres at 256 nodes both price the stored target at or below the
prior (maximum top-1 gain ≤ 0 on A1 ∪ A2), target (a) leaves the grid, and (b) and
(c) stay. The validation result decides one more thing: which ruler Stage 1's
offline screen uses (below).

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
  (`--kl-wdl`, with `--kl-reference` explicit) is added to every policy cell. The anchor refuses
  gradient accumulation (`attach_anchor`), so the 1024 and 4096 cells then need
  either native batches or anchor support for accumulation (code).
- **T2 shows no hidden gain** (upper bound < +20). Stage 1 proceeds unchanged.
- **Either test unread.** Stage 1 proceeds, and the reading is listed as an open
  confound.

**Scoring code.** Stage 0b's scorer lands on main in its own PR, with tests,
before Stage 1's screen reads. Until then the bank scripts it replaces are pinned by
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

## Stage 0c (optional): distance to BT4 at 1 node

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

## Stage 1: one generation, target × effective batch, at 64 sims

**Question.** Does one generation of the project's own self-play training, from the
SL base, gain at 64 sims? Which policy target, and which effective batch size, makes
the difference?

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
  no-rescale target from them. It reproduces a searched no-rescale target to 9.3e-8
  (`target_scores.json`, `offline_rebuild_norescale_maxabs_vs_searched_norescale`).
  **This needs code:** a `retarget` step that writes a sealed reference collect
  whose `policy_target` is the rebuilt one, so an arm can `--collect-from` it. That
  lands in its own PR, with a test that the rebuilt stored-arm target equals the
  shard's `policy_target` when rescale is on, before Stage 1 launches.

### The grid

**Targets** (policy only, `--w-wdl 0`):
- **(a)** the stored rescaled target;
- **(b)** the no-rescale target, rebuilt offline from the same collection;
- **(c)** the stored target at `--train-target-temp 1.5`, which brings its entropy
  from 0.693 to 0.999 (`rl-audit/target_scores.json`), near the prior's.

**Effective batch** (the owner's averaging hypothesis): 256, 512, 1024 and 4096
rows per optimizer step, at an equal number of samples seen.
- **256** is what the banked P1/F12 runs used (the F12 contract's `batch_size`,
  checked here).
- **512** is production's setting (`configs/pbt2_small.yaml`, Aurora; the owner
  found it better than smaller sizes).

**Plus one value-first cell (d):** value only, on the game outcome (`--w-policy 0
--w-wdl 1 --value-outcome-weight 1 --value-search-weight 0`), at batch 512. This
tests the other branch of the candidate mechanism: the loop gains only if the value
head learns information the base lacks. **Named risk:** Q50's loss sat in its value
output (base policy + Q50 value: −55.2, takeover record D4).

**What averaging can and cannot fix.** A larger batch averages away random error in
the targets. It does not remove a consistent bias. The measured sharpening (target
entropy ≈ 0.7 × the prior's, interim) is a bias every row shares. So the batch axis
is crossed with the target arms, not a replacement for them.

### The optimizer, read from the code

- **Production:** `optimizer: aurora`, `matrix_optimizer_scope: mlp_out`,
  `matrix_lr_multiplier: 20`, `batch_size: 512` (`configs/pbt2_small.yaml`). The
  owner tuned the Aurora/AdamW split and found 512 better than 256 in his own
  experiments. Stage 1 takes the split as given, and **512 is the baseline batch.**
- **What Aurora does** (`train/aurora.py`, `AuroraWithAuxAdam`; the owner: "aurora
  is built on muon on top of adamw"):
  - The matrix group's update is the orthogonalised (polar) Nesterov momentum,
    with momentum 0.95. That is a fixed-norm step: the LR alone sets how far it
    moves, whatever the gradient's size or noise.
  - The matrix group's LR is `lr × matrix_lr_multiplier` = 20 × `lr`
    (`train/trainer.py`). The driver does not expose the multiplier, so the
    Trainer default of 20, which matches production, applies.
  - The other groups are AdamW at `lr`, with betas (0.9, 0.95).
- **The simple-RL driver's defaults differ from production:** `--optimizer nadamw`
  and `--batch-size 256` (`scripts/simple_rl_loop.py`). NAdam is β1 0.9, β2 0.98,
  ε 1e-7, with two groups, decay (1e-4) and no-decay (0), both at `--lr`.
- **What P1/F12 actually ran:**
  - **Batch 256:** the F12 contract's `batch_size`, checked here.
  - **Two parameter groups, both at 3e-6** (the P1 log: `[(3e-06, 0.0001, 190),
    (3e-06, 0.0, 289)]`). That is an Adam-family layout. An Aurora run would show
    a matrix group at 20 × `lr`, and there is none. The optimizer class is not named
    in the contract; the two-group layout is the evidence.
  - **Confound of the earlier nulls:** P1/F12 departed from the production
    optimizer setup (no Aurora matrix group, batch 256 against 512). Their 64-sim
    nulls may partly reflect that. Stage 1 runs the production setup.
- **The realised LR of every group** is read off the optimizer's param groups in
  each run's metrics, never off the flag (P8).
- **Momentum's averaging window** is about batch × 1/(1 − β) samples. That is 20
  steps for Aurora's matrix momentum (10,240 samples at batch 512; 81,920 at
  4096), and 10 steps for AdamW's first moment.

### The batch axis is confounded with dose unless movement is matched

At equal samples, a larger batch takes fewer steps:

| Effective batch | Updates on ≈ 95.6k training rows | Aurora matrix momentum window (steps) | Base `lr` (linear scaling) | Matrix LR (20×) |
|---:|---:|---:|---:|---:|
| 256 | ≈ 373 | 20 | 1.5e-6 | 3e-5 |
| **512 (baseline)** | ≈ 187 | 20 | **3e-6** | 6e-5 |
| 1024 | ≈ 93 | 20 | 6e-6 | 1.2e-4 |
| 4096 | ≈ 23 | 20 (most of the phase) | 2.4e-5 | 4.8e-4 |

- **Fixed-norm steps.** With a fixed LR, the 4096 arm takes an eighth as many
  fixed-size matrix steps as the 512 arm, and moves the net much less. The batch
  axis would then just be the dose axis again.
- **LR co-scaling (a guess): linear, LR ∝ batch, anchored at 3e-6 for the 512
  baseline.** 3e-6 is P1/F12's fine-tuning LR. For Aurora's fixed-norm matrix steps,
  linear scaling restores equal movement per sample. For the AdamW groups it is
  only approximate, because their step also depends on the gradient's
  signal-to-noise ratio. Every batch arm uses linear scaling. A fixed-LR batch arm
  would test a different hypothesis (dose), already covered by the snapshots.
- **So cells are compared at matched samples AND at matched movement.** Movement is
  KL(new ‖ SL base) on the probe rows, recorded at every snapshot. The screen reads
  each cell at the snapshot whose KL is closest to a common set of KL values: those
  the 512 arm reaches at 10, 35 and 69 updates. Only P1 c01 (69–84 updates at 256)
  was non-negative out of sample.
- **Fresh optimizer and warmup.** Each arm starts a fresh optimizer with warmup 0.
  Aurora's matrix momentum needs about 20 steps to fill, which is most of the 4096
  arm's 23 updates. So that arm is the most exposed to start-up effects. This is
  recorded as a confound, not corrected.

**Gradient accumulation needs code.** The Trainer has `accum_steps`, but the driver
hard-codes `"accum_steps": 1` (`simple_rl/learn.py`, `trainer_kwargs`). The KL
anchor also refuses accumulation (`attach_anchor`). A small PR adds
`--accum-steps K`, with a test that K × B accumulated equals one native K·B batch
on CPU. It applies only to arms without the KL anchor. 256 and 512 fit natively
(512 is production's); 1024 and 4096 use accumulation unless a native run fits in
VRAM.

### Snapshots

At 10, 20, 35, 50, 69, 138, 207 and 276 updates, plus the final model, for each
cell whose N reaches them. The 4096 arm's N is about 23, so it has only its 10 and
20 snapshots plus its final model.

### Exact commands

```bash
COLLECT1="${COLLECT/--games 16000/--games 1040}"; COLLECT1="${COLLECT1/--seed 2026112000/--seed 2026112300}"
python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/collect --generation 1 $COLLECT1
chmod -R a-w $R/s1/collect
# target (b)'s source: the retarget step (new PR, P5) writes the sealed reference
#   collect $R/s1/collect_norescale, whose policy_target is rebuilt with target_q_rescale=False.
BASE="--presentations 1 --mirror-prob 0.5 --warmup-steps 0 --optimizer aurora \
  --matrix-optimizer-scope mlp_out --optimizer-state fresh --init-optimizer-state fresh \
  --holdout-frac 0.05 --probe-rows 2048 --no-eval"
# effective batch -> native micro-batch, accumulation (new flag, P5), base lr (linear scaling)
declare -A MB=([256]=256 [512]=512 [1024]=512 [4096]=512)
declare -A AC=([256]=1 [512]=1 [1024]=2 [4096]=8)
declare -A LR=([256]=1.5e-6 [512]=3e-6 [1024]=6e-6 [4096]=2.4e-5)
cell() {  # $1 = name, $2 = effective batch, $3 = source collect, rest = arm flags
  local n=$1 b=$2 src=$3; shift 3
  local T="$BASE --batch-size ${MB[$b]} --accum-steps ${AC[$b]} --lr ${LR[$b]} --lr-floor ${LR[$b]}"
  local SN=$(for k in 10 20 35 50 69 138 207 276; do [ "$k" -lt "$(planned_updates $b)" ] && printf '%s,' $k; done)
  python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/$n --generation 1 \
    --collect-from $src $COLLECT1 $T --snapshot-updates ${SN%,} "$@"
  python -u scripts/simple_rl_loop.py run --run-dir $R/s1/$n --generations 1 \
    --collect-from $src $COLLECT1 $T --snapshot-updates ${SN%,} "$@"
}
for b in 256 512 1024 4096; do
  cell a_$b $b $R/s1/collect --w-wdl 0
  cell b_$b $b $R/s1/collect_norescale --w-wdl 0
  cell c_$b $b $R/s1/collect --w-wdl 0 --train-target-temp 1.5
done
cell d_512 512 $R/s1/collect --w-policy 0 --w-wdl 1 --value-outcome-weight 1 --value-search-weight 0
```

`planned_updates b` is loop-exp1's step-2 `planned_phase_updates` reader at that
batch. The command is final only once the `--accum-steps` PR fixes the flag's name
and its interaction with `planned_phase_updates`.

### Screening order (matches are the expense)

1. **Offline screen, every cell and snapshot:**
   - on A1 (leak-free) and A2;
   - top-1 against the screen ruler, plus the calibration gap and prior entropy;
   - the screen ruler is the Stage 0b-validated ruler, else searched BT4 at 256
     nodes, labelled unvalidated;
   - read at matched KL (above);
   - selection on A1, report on A2.
2. **At most three cells go to registered matches:** the three with the highest
   A2 top-1 among those inside the [calibration band](#calibration-band-used-by-the-screen-and-by-stage-2). At most one
   cell per target, and (d) is eligible by its value read (|V − deep-SF value| on
   A1, reported on A2).
3. **If no ruler is validated and the screen is unvalidated,** the three cells are
   pre-committed instead: (a), (b) and (c), each at batch 512 and the final model.

### Registered match, per selected cell

2,000 games against the SL base at 64 sims, seat-balanced:
- **Openings:** the 500-line `s1_match` block.
- **Arena:** loop-exp1's `$COMMON` with `--games 1000 --openings-fen <block>` in
  place of `--openings`/`--opening-plies`, two runs on one seed with the cell as
  candidate (A) and as reference (B).
- **Pooling:**
  ```bash
  python scripts/arena_seat_balanced_pool.py --arm <model.pt> \
    --a-games <A.games.jsonl> --b-games <B.games.jsonl> \
    --a-result <results.jsonl> --b-result <results.jsonl> \
    --a-label <tag>_A --b-label <tag>_B --verdict-rule lb-gt-zero --z 2.394
  ```
- **z:** z = 2.394, Bonferroni over three matches (Φ⁻¹(1 − 0.025/3)). With fewer
  matches, z is recomputed for the actual count.
- **Reported alongside, not judged:** temperature-0 reads at 1, 4 and 16 sims (800
  games each, on `s1_low`), and the offline descriptors.

**Reading and dispositions (pre-committed, per match):**
- **Success:** lower bound above 0. Stage 2 launches on that cell; with several, on
  the one with the highest pooled estimate.
- **Kill:** upper bound below 0. The cell is recorded as harmful at this dose.
- **Unread:** one fresh confirmation, with the same size and z, on `s1_confirm`,
  read alone. If it is still unread, it is recorded "unread at N = 2,000".
- **No success in any cell:** Stage 2 does not launch. The
  [conditional diagnostic](#conditional-diagnostic-a-bt4-raw-policy-target) runs
  next, then O3's larger collection with the same grid.

**Power.** The banked P1 64-sim match gives a per-cluster SD of d of 0.195 (200
clusters, ρ = +0.12; *review*). That is an SE of 6.06 Elo at 500 clusters, and an
80%-power MDE of 19.6 Elo at z 2.394, about 10% larger at ρ = +0.36.

**Compute:**
- collect: about 12 min (20 without #1441);
- training: 13 cells, minutes each;
- offline screen: minutes;
- matches: 1.1–1.6 h per 2,000 games, so 3.3–4.8 h for three, plus confirmations.

### Conditional diagnostic: a BT4 raw-policy target

**It runs only if every own-pipeline cell in Stage 1 fails its match** (kill, or
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

### Calibration band (used by the screen and by Stage 2)

The gap (mean top-1 probability − top-1 agreement) against the screen ruler must
stay within [g_BT4, +0.02]. g_BT4 is BT4's raw gap on the same ruler (−0.148
against R800, Stage 0b).
- The upper edge stops over-confidence, the direction the loop drifted (P1 c12 is
  +0.034 against R800).
- The lower edge allows a prior as soft as a known strong net's, but no softer.
- BT4 is a reference point, not a target: it is a different, larger net. Both
  edges are guesses.
- Entropy is recorded beside the gap.

**Later, only after Stage 1 has read:**
- learner recipes (anchor, R1 per-row entropy matching, ranking loss), screened
  the same way;
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
  about 3 min at 16. The usable VRAM limit is about 26 GB in total.

These are not planned in detail here.

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
| `anchor` | 2,300–2,499 | optional Stage 0c, the Stockfish K tuning ladder and the Stage 2 Stockfish anchor |
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
  - Cell (b) runs the retarget step every generation.
- **Gated every generation:**
  - loop-exp1's collect gates (G-C1 = 0, G-C2 equal, G-C3 ≥ +0.50, G-C7 ≥ 80,
    G-C10 ≤ 0.05) and G-T2 / G-T4;
  - the calibration band, and top-1 against the screen ruler on A2 not below the SL
    base's.
  - A failure allows one re-collect, then stops the run at g − 1.
- **Readouts:** on a GPU-time schedule, with an arena share ≤ 30%. The next readout
  comes once the generations' GPU time is at least 7/3 × the last readout's arena
  time, about every 6–9 generations.
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
3. a gated failure that one re-collect does not clear, or the calibration
   invariant;
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

So offline scores only filter, and only after a ruler is validated; only registered
64-sim matches decide.

## What is still a guess

| Number | Where it is used | The measurement that replaces it |
|---|---|---|
| Every *interim* number in Stage 0 (veto weights, the 0.7× ratchet) | direction | the diagnosis's full report, T1 / T2 |
| The ruler validation rule | Stage 0b | Stage 1's results as a positive control |
| 1,040 games / ≈ 100k rows | Stage 1, Stage 2 | Stage 1's read and O3 |
| Linear LR scaling anchored at 3e-6 for batch 512 (matrix LR 20×) | Stage 1 batch axis | the matched-KL read |
| The matched-KL points (the 512 arm at 10 / 35 / 69 updates) | Stage 1 screen | Stage 1's held-out curves |
| Snapshot grid 10…276 | Stage 1 dose | Stage 1's held-out curves |
| `--train-target-temp 1.5` | target (c) | Stage 1 |
| Outcome-only value target, value-only training, at batch 512 | cell (d) | Stage 1 |
| Calibration band [g_BT4, +0.02] | screen, Stage 2 gate | Stage 1's realised gaps |
| At most three matched cells; z 2.394 | Stage 1 | — (a budget) |
| Arena share ≤ 30% (7/3 factor) | Stage 2 schedule | Stage 2's realised times |
| Stop-rule constants (lag 3, two consecutive) and SE 6.06 | Stage 2 | Stage 2's realised clustered SE |
| The Stockfish K band (40–60%), the 25–75% re-tune trigger, 400 games | Stage 2 anchor | the tuning ladder and first readouts |
| G-C3 ≥ +0.50; holdout 0.05; probe 2,048 | gates | not tuned |
| +50 Elo at one week, and "halved" | success line | Stage 1's projection |
| 0.49M rows per GPU-hour at 64 sims (3,284 games/h single-process); 96.8 rows per game; 1,274–1,800 arena games/h | compute | Stage 1's own runs |
| The exchange rate: 1.86M / 0.88M (provisional) / 0.49M rows per GPU-hour at 16 / 32 / 64 sims | later depth arms | the full collect-throughput report, after #1441 |
| BT4 raw forward cost (about 43 ms per row on 8 CPU threads) | the conditional diagnostic | its own run, if it runs |

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
- **P5.** Two small PRs have merged and were run once on a small collect:
  - the retarget step (target (b)), tested so the rebuilt target equals the shard's
    `policy_target` when rescale is on;
  - `--accum-steps` (the batch axis), tested so accumulated equals native on CPU.
- **P6.** For the Stockfish anchor and the optional Stage 0c, the UCI engine's
  settings match the arena's training-shape search on fixed positions. If they
  cannot be matched, both are reported as measuring the UCI agent.
- **P7.** The screen's scorer is on main, or the bank scripts are used at the
  sha256 listed in Stage 0b.
- **P8.** Each run's metrics carry every param group's realised LR. The batch arms'
  values are checked against the linear-scaling table before the screen reads.
- **P9.** Only if the conditional diagnostic runs, its BT4 policy array:
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
- The batch axis: linear LR scaling is exact only for Aurora's fixed-norm matrix
  steps; the AdamW groups' steps also depend on gradient noise, so matched KL is
  the comparison, and the 4096 arm sits mostly inside the optimizer's start-up.
- The earlier nulls (P1, F12, Q50) ran an Adam-family two-group optimizer at batch
  256, not production's Aurora split at 512. Stage 1 differs from them in optimizer
  and batch as well as in the collection.
- Selecting the played snapshot is selection. It is done on held-out games of the
  collection, never on a match, and each match is on its own block.
- loop-exp1's confounds (twofold-repetition draws, the same-coloured-bishops gap)
  apply.
