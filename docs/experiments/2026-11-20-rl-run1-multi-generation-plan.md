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
  it could learn in chess at 16 nodes".

**The plan, in order:**
1. **Stage 0** (done): what the banked matches and diagnostics already say.
2. **Stage 0b** (CPU plus about 2 GPU-h, no training): validate a ruler (primary:
   BT4 and Ceres searched at 64 and 256 nodes), price the targets on it, and run
   the registered tests T1 and T2 and the true headroom match.
3. **Stage 0c** (GPU, no training): the node-odds ladder, with the SL base against
   BT4 at fixed nodes. It gives the baseline gap and the owner's milestone: our
   256-sim search matches BT4 at 64 nodes.
4. **Stage 1** (one generation): a target test at 64 sims. One SL-base collection;
   three policy-only arms that differ only in the policy target, plus a value-first
   arm; each judged by a registered seat-balanced 64-sim match. This is also the first possible positive
   control for any offline screen.
5. **Stage 1c** (after Stage 1): learner recipes, screened offline only on a ruler
   that Stage 0b validated, and search-depth arms at matched GPU time.
6. **Stage 2**: the multi-generation run, with a gated calibration invariant and
   two anchors (the SL base, and BT4 at fixed nodes).

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
| O1 | **Stage 1's arms** | **(a) stored rescaled, (b) no-rescale, (c) rescaled at `--train-target-temp 1.5`, (d) value-first on game outcome** | (b) and (d) only, the two built on the candidate mechanism (MDE 18.7 instead of 20.2 Elo) |
| O2 | **Stage 1's collection size** | **1,040 games (≈ 100k rows)**, a guess with its power stated below | 2,080 games, which doubles the dose range the snapshots can reach |
| O3 | **Search depth** | **An equal-GPU-time arm in Stage 1c only** (16/32/64 sims at measured positions per GPU-hour); no depth presumed better | Leave depth at 64 |
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

## Stage 0b: ruler validation and target pricing (CPU, plus about 2 GPU-h; no training)

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

**Ruler validation rule (pre-committed).** Each ruler is tested, the primary
rulers first. A ruler is *validated* when two things hold. First, it is not the training target. Second, on A1 ∪ A2 it ranks the SL base
above P1 c12, F12 c12 and Q50 c12, and Q50 c12 last of the four. A pair whose
difference is within one SE (game-clustered) counts as a tie, not an error. If no
ruler is validated, Stage 1 still runs, because it is judged by matches, and Stage
1c's offline screen does not run. If several are validated, the searched BT4 ruler
at 256 nodes is used, then at 64, then Ceres, then the rest.

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
  |ΔQ| on the probe rows ≤ the P1 c01 value). Stage 1c's R5 becomes part of every
  baseline.
- **T2 shows no hidden gain** (upper bound < +20). Stage 1 proceeds unchanged.
- **Either test unread.** Stage 1 proceeds, and the reading is listed as an open
  confound.

**Scoring code.** Stage 0b's scorer lands on main in its own PR, with tests,
before Stage 0b reads. Until then the bank scripts it replaces are pinned by
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

## Stage 0c: node-odds ladder against BT4 (the baseline gap, in concrete terms)

**Why.** This is the owner's headline yardstick (2026-11-20): "even if we get as good
as them matching one 1 node for 1 node, it will already be huge since our net is
smaller. I expect even if our 256 node search matched their 64 it would be amazing
for our elo and show that we can be competitive in tcec tournaments with all the
top engines etc".

**The milestones, recorded here:**
- **The milestone:** our 256-sim search matches BT4 at 64 nodes.
- **The stretch:** node for node, at the same node count.

**The ladder.** The SL base at 64, 256 and 800 sims plays BT4-it332 at 1, 64 and
256 nodes: nine cells. The SL base goes first, so the gap is known before any
training. Stage 2 then repeats the milestone cell at its readouts.

**What exists, checked here:**
- **An lc0 that runs BT4 on the GPU:**
  `~/chess-artifacts/operations/rl-week-one-20261112-v1/external-Lc0-CUDA-build-runtime-v1/build/lc0`.
  It is v0.32.1+git.dirty, and `--backend` offers `cuda-fp16`, `cuda-auto` and
  `cuda`. Its BT4 GPU operability bank (`external-Lc0-BT4-GPU-operability-results-v3`)
  passed its 23 predicate controls.
- **The other two lc0 builds are CPU-only:** `~/local_engines/lc0/lc0` (v0.32.1)
  and `~/local_engines/lc0_033/build/lc0` (v0.33.0-dev+git.d8ce482). Both offer the
  `eigen` backend but no CUDA one.
- **Weights:** `data/lc0/BT4-it332.pb.gz`, sha256
  `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`.
- **No banked match against an lc0 engine can be reused.** The one joint SL-base vs
  Lc0 GPU match (`external-A39-Lc0-joint-timed-engineering-results-v1`) failed after
  1 game. Lc0 died with `std::system_error: Resource temporarily unavailable` at
  `Threads=8` (`external-A39-Lc0-joint-failure-thread-source-note-v1.md`). The
  "champion vs lc0-LAST2x −619" anchors in project notes are matches between our own
  nets, one of them trained on lc0 data. They are not matches against the lc0
  engine.

**Settings for both sides:**
- **Harness:** `scripts/match_vs_uci.py`; cutechess is broken here. Each opening is
  played twice with colours swapped, so every cell is seat-balanced by construction,
  and the script's CI is over pairs.
- **Our engine:**
  - `python -m chess_anti_engine.uci --checkpoint <net>`, with the arena's
    training-shape search flags (`--topk 16 --c-scale 0.1 --c-visit 50`, and the
    root flags from the arena row's `search_candidate.gumbel`);
  - `go nodes S`, where our node count is simulations;
  - it plays the halving survivor, so it is deterministic at temperature 0.
- **BT4:**
  - `--weights=<BT4> --backend=cuda-fp16`, `go nodes N`;
  - `SmartPruningFactor=0`, so it spends the whole budget;
  - `Threads=2`, below the 8 that crashed;
  - lc0's UCI default temperature is 0, so it is deterministic up to
    multi-threaded search order.
- **Syzygy:** the production pair for both sides (`SyzygyPath` on each), plus
  `--syzygy-adjudicate-path`.
- **Openings:** the 200-line `anchor` block, with move history. 400 games per cell.

**How lc0 counts nodes, versus our sims: unmeasured.** Our `nodes` is the
simulations run this move. lc0's `go nodes N` limit may count visits already in
the subtree reused from the previous move. If it does, BT4's realised new search
per move is below N. Pre-launch check P8 logs both engines' reported `nodes` per
move (`--move-log-out`) over 20 games per side, and the record states the realised
per-move new-visit counts beside each cell. `--enforce-nodes-*` (an external stop
at the reported count) is used only for an engine that ignores `go nodes`.

**One cell, exact command:**

```bash
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
LC0=~/chess-artifacts/operations/rl-week-one-20261112-v1/external-Lc0-CUDA-build-runtime-v1/build/lc0
BT4=data/lc0/BT4-it332.pb.gz          # sha256 e6ada9d6…bdd618, checked before each cell
ENG_A="python -m chess_anti_engine.uci --checkpoint <net> --device cuda --topk 16 --c-scale 0.1 --c-visit 50 <root flags>"
ENG_B="$LC0 --weights=$BT4 --backend=cuda-fp16"
PYTHONPATH=. python -u scripts/match_vs_uci.py --engine-a "$ENG_A" --engine-b "$ENG_B" \
  --label-a ours_s$S --label-b bt4_n$N --games 400 --nodes-a $S --nodes-b $N \
  --openings $R/openings/blocks/anchor.fen --max-plies 300 \
  --option-a SyzygyPath=$TB --option-b SyzygyPath=$TB \
  --option-b Threads=2 --option-b SmartPruningFactor=0 \
  --syzygy-adjudicate-path $TB \
  --games-out $R/ladder/ours_s${S}_bt4_n$N.games.jsonl --pgn-out $R/ladder/ours_s${S}_bt4_n$N.pgn \
  --move-log-out $R/ladder/ours_s${S}_bt4_n$N.moves.csv
```

UCI option names are checked against each engine's `uci` listing before the first
cell (P8).

**Size and power.** This is a measurement, not a judged test. Each cell reports the
score with its pair-clustered 95% CI, mapped to Elo.
- At 400 games (200 pairs), with the per-pair SD of about 0.26 from our 64-sim
  self-matches (unmeasured against BT4), the score SE is about 0.018.
- That is about ±25 Elo (95%) near a 50% score and about ±33 at 25% or 75%. It is
  wider in Elo towards the extremes, where a cell far from 50% mostly says which
  side of the crossing it is on.
- **Milestone reading:** the milestone cell's 95% lower bound on score ≥ 0.50. A
  point estimate ≥ 0.50 with the lower bound below 0.50 is reported as "level,
  unconfirmed".

**Compute: unmeasured.** It needs BT4's GPU nodes per second at these budgets and
our 800-sim UCI move time. A 20-game smoke per engine (P8) measures both before the
ladder runs. One arena at a time; never concurrent with another GPU job.

## Stage 1: one-generation target test at 64 sims

**Question.** Does a one-generation update from the SL base, trained on fresh
self-play rows, gain at 64 sims? Which policy target, or the value-first arm, makes
the difference?

**Hypotheses.**
- (b) no-rescale beats (a) stored: the two independent references price it at +3.7
  / +4.4 pp against +0.7 / +1.4.
- (c) tests the owner's calibration hypothesis in its simplest form.
  `--train-target-temp 1.5` brings the stored target's entropy from 0.693 to 0.999
  (`rl-audit/target_scores.json`), near the prior's.
- (d) tests the candidate mechanism's other branch: the loop gains only if the value
  head learns information the base lacks. The game outcome is the one value signal
  the base value head does not already produce.

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

**Arms:**
- **(a)** policy only (`--w-wdl 0`), the stored rescaled target;
- **(b)** policy only, the no-rescale target, rebuilt offline from the same
  collection;
- **(c)** policy only, the stored target with `--train-target-temp 1.5`;
- **(d)** value first: value only, on the game outcome (`--w-policy 0 --w-wdl 1
  --value-outcome-weight 1 --value-search-weight 0`). **Named risk:** Q50's loss sat
  in its value output (base policy + Q50 value: −55.2, takeover record D4).
- **LR:** 3e-6 (the loop-exp1 default) and 1e-5, each flat. Both are guesses.

So there are eight trained models. If T2 shows a hidden policy gain, (a)–(c) keep
the value frozen, as they already do.

**Dose.** Snapshots at 10, 20, 35, 50, 69, 138, 207 and 276 updates, plus the final
model at N (≈ 374 updates at batch 256 for ≈ 100k rows, holdout 0.05). The only
non-negative out-of-sample point in the banked loop is P1 c01, at 69–84 updates
(leak-free table above), so the grid is dense below 69.
- **Which model plays the match, per arm** (pre-committed):
  - Policy arms (a)–(c), if Stage 0b validated a ruler: the (LR, snapshot) that
    maximises it. Selection is on A1 and the report is on A2.
  - Arm (d), if deep-SF labels exist: the (LR, snapshot) with the lowest mean
    |V − deep-SF value| on A1, reported on A2. That is not its training target.
  - Otherwise: the final model at each LR, so the arm plays two matches.

**Collection and arms, exact commands** (loop-exp1's `$COLLECT` and `$TRAIN`,
except where stated):

```bash
COLLECT1="${COLLECT/--games 16000/--games 1040} "; COLLECT1="${COLLECT1/--seed 2026112000/--seed 2026112300}"
python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/collect --generation 1 $COLLECT1
chmod -R a-w $R/s1/collect
# (b)'s source: the retarget step (new PR), writing a sealed reference collect
#   $R/s1/collect_norescale whose policy_target is rebuilt with target_q_rescale=False.
SNAPS=10,20,35,50,69,138,207,276
for lr in 3e-6 1e-5; do
  T="${TRAIN/--lr 3e-6 --lr-floor 3e-6/--lr $lr --lr-floor $lr} --snapshot-updates $SNAPS"
  for a in a b c d; do
    src=$R/s1/collect; [ $a = b ] && src=$R/s1/collect_norescale
    extra="--w-wdl 0"; [ $a = c ] && extra="--w-wdl 0 --train-target-temp 1.5"
    [ $a = d ] && extra="--w-policy 0 --w-wdl 1 --value-outcome-weight 1 --value-search-weight 0"
    python -u scripts/simple_rl_loop.py collect --run-dir $R/s1/arm_${a}_$lr --generation 1 \
      --collect-from $src $COLLECT1 $T $extra
    python -u scripts/simple_rl_loop.py run --run-dir $R/s1/arm_${a}_$lr --generations 1 \
      --collect-from $src $COLLECT1 $T $extra
  done
done
```

The snapshot list is checked against N before each run is pinned. A snapshot at or
above N is dropped, and the record says so.

**Gates:** loop-exp1's collect gates (G-C1–G-C11) and model gates (G-T1–G-T4),
unchanged. A collect gate failure allows one re-collect from the sealed cursor;
a second failure stops Stage 1, and the record says so.

**Registered match, per played model.** 2,000 games against the SL base at 64
sims, seat-balanced:
- **Openings:** the 500-line `s1_match` block, shared by every Stage 1 match, so
  arms are compared on the same openings.
- **Arena:** loop-exp1's `$COMMON` with `--games 1000 --openings-fen <block>` in
  place of `--openings`/`--opening-plies`, two runs on one seed with the arm as
  candidate (A) and as reference (B).
- **Pooling:**
  ```bash
  python scripts/arena_seat_balanced_pool.py --arm <model.pt> \
    --a-games <A.games.jsonl> --b-games <B.games.jsonl> \
    --a-result <results.jsonl> --b-result <results.jsonl> \
    --a-label <tag>_A --b-label <tag>_B --verdict-rule lb-gt-zero --z <z>
  ```
- **z:** Bonferroni over the m matches played: z = Φ⁻¹(1 − 0.025/m). With one model
  per arm, m = 4 and z = 2.498. With two per arm, m = 8 and z = 2.734. If both
  selection rules apply to only some arms, m is the actual count.
- **Reported alongside, not judged:** temperature-0 reads at 1, 4 and 16 sims (800
  games each, on the 200-line `s1_low` block); the Stage 0b ruler scores; the
  offline descriptors.

**Reading and dispositions (pre-committed, per match):**
- **Success:** lower bound above 0. Stage 2 launches on that arm; with several, on
  the one with the highest pooled estimate.
- **Kill:** upper bound below 0. The arm is recorded as harmful at this dose.
- **Unread:** anything else. One fresh confirmation is run, with the same size and
  z on `s1_confirm`, read alone (several unread arms share that block). If it is still unread, it is recorded "unread
  at N = 2,000", and that arm does not launch Stage 2.
- **No success in any arm:** Stage 2 does not launch. The next step is Stage 1c,
  with an offline screen only if Stage 0b validated a ruler. If it did not, the next
  step is O2's larger collection, which doubles the reachable dose, with the same
  arms.

**Power.**
- The banked P1 64-sim match gives a per-cluster SD of d of 0.195 (200 clusters,
  ρ = +0.12; *review*). At 500 clusters the SE is 6.06 Elo.
- The 80%-power MDE is 20.2 Elo at m = 4 and 21.7 at m = 8. It is about 10% larger
  if ρ is +0.36, as at 1 sim.
- **Expected effect:** unknown. A c01-sized dose moved KL(new‖base) by 0.038. About
  374 updates moves it by about 0.17, the movement of P1 c04–c06, which were ≈ 0 at
  64 sims (*review*). With the stored target, "unread" is the likely outcome. Arms
  (b) and (d) exist because of the independent references and the candidate
  mechanism.

**This is the first possible positive control.** A Stage 1 success is the first net
with a positive 64-sim result. It is then added to Stage 0b's ruler validation.

**Compute:**
- collect: about 0.3 h at the measured 3,284 games/h;
- training: eight models, minutes each;
- matches: 1.1–1.6 h per 2,000 games, at 1,274–1,800 games/h, so 4.4–12.6 h for four
  to eight matches;
- temperature-0 reads: minutes.

## Stage 1c: learner recipes and depth arms (after Stage 1)

These run only after Stage 1 has read.

**Learner recipes,** screened offline. The screen runs only if Stage 0b validated a
ruler, and only on that ruler, which is never the training target. Selection is on
A1 and the test is on A2 (or a game-split half of A1). The co-requirement is a
calibration guard (below).

| # | Recipe | Driver flag | Code needed |
|---|---|---|---|
| R1 | Entropy-matched targets | `--train-target-temp` (global); per-row: new code. `--policy-prior-mix λ` (target = λ · stored + (1 − λ) · prior) is an entropy-preserving alternative that may make per-row code unnecessary | per-row only |
| R2 | R1 + anchor | `--kl-policy λ --kl-reference <SL base copy>`; screened as anchor-only and as R1 + anchor | agree-only KL: yes |
| R3 | Flip-weighted or pairwise ranking loss | — | yes |
| R4 | Dose with early stopping | the snapshot grid | no |
| R5 | Freeze value drift | `--kl-wdl λ --kl-reference <SL base copy>` | no |
| R7 | Interior-node targets | — | collector change |

- **Runs on fresh simple-RL collects, not on the old F12 rows.** No path runs the
  driver's recipe flags on the old rows without code. Those rows came from the P1
  harness (`p1_cycle.py`, another tree, a continued optimizer). The driver's
  `--collect-from` needs a collect played by the arm's own generation g − 1 model.
  The old shards have no per-child Q or visit arrays. Stage 1c's rows are Stage 1's
  collection, so the optimizer regime (`--optimizer-state fresh`) matches Stage 2.
- **`--kl-reference` is always passed explicitly.** It defaults to generation 0,
  which in Stage 2 is the Stage 1 winner, not the SL base. G-T4 reads the realised
  reference sha256.
- **Multiplicity:** Bonferroni over arms (recipe × setting), not recipes.

**Depth arms (R6):** 16, 32 and 64 target sims, run only as collection arms at
matched GPU time. Each collects for the same GPU-hours, at the positions per
GPU-hour measured at that sim count. The first exchange-rate point (interim): one
process × 256 games at 16 sims gives 6,451 games/h, against 3,372 at 64 sims, which
is 1.9×, not 4×. Three processes × 128 at 64 sims give 5,021 games/h, via
`--collect-procs` in PR #1441 (not merged). So the depth arms' collections are sized
from the measurement, never from 4×. Each is judged at 64 sims by
Stage 1's match design. They never run on fixed banked rows: there a 16-sim arm
would get a quarter of the compute.

**The calibration guard** replaces the earlier fixed entropy band. That band was
centred on an under-confident net. The gap is mean top-1 probability minus top-1
agreement, on the 2,433 leak-free rows (interim, `flip_by_cycle_v2.json`):

| Net | Entropy | Gap vs R800 | Gap vs BT4 |
|---|---:|---:|---:|
| SL base | 1.191 | −0.066 | −0.113 |
| P1 c01 | 1.085 | −0.046 | −0.087 |
| P1 c03 | 0.973 | −0.010 | −0.045 |
| P1 c06 | 0.892 | +0.018 | −0.014 |
| P1 c12 | 0.866 | +0.034 | +0.008 |

- Against R800 the gap crosses zero near entropy 0.95. Against BT4 it crosses only
  at c12, where top-1 is worst. **So calibration alone does not select a good net,
  and the ruler matters.**
- **Guard:** |mean p_top1 − top-1 agreement| ≤ 0.02, against the validated ruler.
- Entropy is recorded as a descriptor. Entropy conventions differ between nets: BT4
  and C3 read 1.49 and 1.53 on these rows.
- The driver's `--tripwire --min-policy-entropy / --min-target-entropy` stay as
  coarse alarms, not as the guard.

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
| `anchor` | 2,300–2,499 | BT4 node-odds ladder (Stage 0c) and the Stage 2 BT4 anchor |
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

## Stage 2: the multi-generation run

Launches only on a Stage 1 success, with that arm's target, LR and dose.

### Driver settings

One run dir, `run --generations G`:
- **Start:** `--init-checkpoint` = the Stage 1 winner, copied into the run dir,
  sha-verified and write-protected.
- **Generation size:** Stage 1's collection size at 64 sims. The played model is
  the snapshot at the winner's dose (`--snapshot-updates D`) or the final model, as
  in Stage 1.
- **Phase:** `--presentations 1`. **Optimizer:** `--optimizer-state fresh` each
  generation, at the winner's LR, warmup 0, flat.
- **Openings:** `--opening-sampling cycle`, `--seed 2026112200`.
- **Holdout and probe:** `--holdout-frac 0.05`, `--probe-rows 2048`. **Evaluation:**
  `--no-eval`; the readouts below replace it.
- **KL:** if the winner uses `--kl-policy` or `--kl-wdl`, `--kl-reference <SL base
  copy>` is passed explicitly.
- **Target:** if the winner is arm (b), every generation runs the retarget step
  before training.

### The calibration invariant (gated, every generation)

The target is sharper than the prior that searched for it (about 0.7× its entropy;
interim). Every generation measures the newest net on A1 ∪ A2:
- **Gate:** the calibration gap against Stage 0b's validated ruler stays within
  ±0.02, and top-1 against that ruler on A2 is not below the SL base's (one-sided
  2.5% on discordant rows). Calibration alone admits a worse net (above). If no ruler was validated, R800 is used, and the gate is labelled
  provisional. Leaving the band stops the run at the last generation inside it.
- **Recorded:** prior entropy, veto rate, mean V − Q(top child), |Q|, P(draw), and
  agreement with each reference.

### Per-generation gates

- **Gated (machinery):** G-C1 = 0; G-C2 equal; G-C3 ≥ +0.50 on both root classes
  (loop-exp1's primary value); G-C7 ≥ 80; G-C10 ≤ 0.05.
- **Gated (model):** the calibration invariant; G-T2 against the previous
  generation (KL > 1.5 or |ΔQ| > 0.4 fails); G-T4 as loop-exp1.
- **Recorded, drift alarm:** G-C4, G-C5, G-C6, G-C8, G-C9. Two consecutive
  generations out of band pause the run for a report.
- **On a failure:** a collect gate allows one redo from the sealed cursor. A second
  failure, or any other gated failure, stops the run at g − 1. Restarting needs an
  addendum.

### Readouts on a GPU-time schedule

- **R-base,** at each readout: the newest net against the SL base, 2,000 games at
  64 sims, seat-balanced, on `readout_base` (the same block every readout). Temperature-0 reads at 1, 4 and 16 sims
  (800 games each, on `s1_low`) are reported beside it.
- **Schedule:** the next readout falls once the generations' GPU time since the
  last readout is at least 7/3 × that readout's measured arena time, which keeps
  the arena share ≤ 30%. At 0.4 h per generation and 1.1–1.6 h per 2,000 games, that
  is about every 6–9 generations.
- **The second anchor: BT4 at fixed nodes.** At every readout, the newest net plays
  the milestone cell (our 256 sims against BT4 at 64 nodes), using Stage 0c's command
  and settings, 400 games on `anchor`. At the deliverable it plays the full nine-cell
  ladder. The anchor is reported, not judged; R-base judges.
- **Why BT4, and why it replaces the Stockfish anchor:**
  - It is the owner's yardstick and names his milestone.
  - Stage 0c measures the SL base on it, so the gap is known in advance.
  - As an lc0-family engine it is closer to ours in style, which should make it
    more sensitive to policy and value gains.
  - **The single-anchor trap.** Anchored Elo depended on the anchor by up to 82 Elo
    in one recorded design (experiment ledger). So two unlike anchors stay: the SL
    base (our own family, via R-base) and BT4 at fixed nodes. A gain claimed on one
    and absent on the other is reported as anchor-dependent.
  - The Stockfish anchor is dropped. Matching its settings to our agent was itself
    an open check, and a third anchor would raise the arena share past 30% with
    nothing to decide.
- **Transitivity check:** the change in the milestone cell's score against the
  change in R-base, both since Stage 0c, agree within the combined CI. A
  disagreement is recorded, not resolved.

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

### Compute

- **A generation:** about 0.4 h at 1,040 games (collect 0.3 h, training minutes,
  gates and the calibration read minutes), plus the retarget step for arm (b).
- **A readout:** 1.1–1.6 h of 64-sim games, plus the temperature-0 reads. The
  BT4 anchor's rate (one cell, 400 games) is unmeasured until P8.

### What success means at one week

**The one-week target, pre-committed:** within 7 days of Stage 2's start, the
deliverable's confirmation match reads a 95% lower bound above 0 and a point
estimate of at least +50 Elo at 64 sims.
- Below +50 with the lower bound above 0: "gain confirmed, below target".
- Otherwise: "not shown at one week".

+50 is a guess. Before Stage 2 starts it is replaced by a projection: generations in
a week × g₁ / generations per readout, halved. g₁ is the Stage 1 winner's pooled
estimate, which carries winner's curse; the halving is for that, and is itself a
guess.

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
| The ruler validation rule (rank order of four nets, one-SE ties) | Stage 0b | Stage 1's results as a positive control |
| 1,040 games / ≈ 100k rows | Stage 1, Stage 2 generation size | Stage 1's read and O2 |
| LR 3e-6 and 1e-5, batch 256 | Stage 1 | Stage 1 |
| Snapshot grid 10…276 | Stage 1 dose | Stage 1's held-out curves |
| `--train-target-temp 1.5` | arm (c) | Stage 1 |
| Outcome-only value target, value-only training | arm (d) | Stage 1 |
| The 16-vs-64-sim exchange rate (1.9×, one point, interim) | Stage 1c depth arms | the throughput measurement's full curve |
| Calibration tolerance ±0.02 | Stage 1c guard, Stage 2 invariant | Stage 1's realised gaps |
| Arena share ≤ 30% (7/3 factor) | Stage 2 schedule | Stage 2's realised times |
| Stop-rule constants (lag 3, two consecutive) and the simulation's SE 6.06 | Stage 2 | Stage 2's realised clustered SE |
| 400 games per ladder cell; the per-pair SD 0.26 against BT4 | Stage 0c, Stage 2 anchor | Stage 0c's realised CIs |
| lc0 node counting vs our sims | Stage 0c, Stage 2 anchor | P8's per-move node logs |
| G-C3 ≥ +0.50 | Stage 2 gate | loop-exp1's primary value; revisited if a clean collect fails it |
| Holdout 0.05, probe 2,048 | Stage 1, 2 | not tuned |
| +50 Elo at one week and the "halved" projection | success line | Stage 1's projection |
| 3,284 games/h; 96.8 rows per game; 1,274–1,800 arena games/h | compute | Stage 1's own collect and matches |
| Per-cluster SD 0.195, ρ at 64 sims | match sizing | Stage 1's realised clustered SE |
| A2's ≈ 3,000 rows | Stage 0b | the disagreement table's SEs |

## Pre-launch checks

- **P1.** Every stage runs from a detached worktree at a pinned main sha, with
  extensions rebuilt; the sha is recorded here.
- **P2.** A1 excludes every audit position that occurs in cycle 2–12 games, by
  position key; both counts are reported. A2 shares no position key with any
  training collection.
- **P3.** The SL base is copied into the run dir and write-protected (loop-exp1
  step 0a). The split is reproduced (step 0b). The blocks are cut, with their
  sha256 recorded.
- **P4.** The diagnosis's full report and T1 / T2 are read, and every interim
  number here is replaced or confirmed.
- **P5.** The retarget step's PR has merged, with its rebuild-equals-shard test,
  and was run once on a small collect.
- **P6.** The UCI engine's settings match the arena's training-shape search on
  fixed positions, because the BT4 ladder plays our net through the UCI path. If
  they cannot be matched, the ladder is reported as measuring the UCI agent.
- **P8.** BT4 smoke, before Stage 0c:
  - the lc0 CUDA build loads BT4 (sha256 checked) on the GPU at `Threads=2`;
  - 20 games per side complete without an engine death;
  - UCI option names are checked against each `uci` listing;
  - per-move reported `nodes` are logged for both engines, which settles how lc0
    counts nodes against our sims;
  - BT4 nodes per second and our 800-sim move time are measured, which prices the
    ladder.
- **P7.** Stage 0b's scorer is on main, or the bank scripts are used at the sha256
  listed above.

## Confounds

- The R800 reference is the SL base's own 800-sim search; BT4 and C3 are other
  nets' raw policies. None is truth; Stage 0b's validation rule is the arbiter.
- The interim diagnosis ran on nets trained under the old loop and the P1 harness,
  with a continued optimizer. Stage 1 uses fresh simple-RL collects and a fresh
  optimizer.
- P1's snapshots confound dose with data vintage (snapshot k trained on rows from
  F12 c01…c(k−1)). Stage 1's single collection removes this.
- Selecting the played snapshot is selection. It is done on held-out games of the
  collection, never on a match, and each match is on its own block.
- loop-exp1's confounds (twofold-repetition draws, the same-coloured-bishops gap)
  apply.
