# 2026-11-22: RL run 1, generation 2 — Zq + per-generation recalibration (pre-registration)

**Status: registered, NOT AUTHORISED.** This record starts a new collection, so it launches only
after the owner's explicit go-ahead. Each stage has its own go-ahead file, `stage0/OWNER_GO` and
`gen2/OWNER_GO`, which the coordinator writes after the owner says yes, one line each: who, when,
the message. Each stage launcher refuses without its file (exit 90). Nothing here was run on the
GPU; the sizing below is CPU work plus measured costs.

**What it is.** Match 8 nominated "Zq + per-generation recalibration" for a 2-generation run under
its own record. This is that record. It has three parts:
- **Stage 0:** a replication of match 8 under a new seed, which confirms step 1;
- **Generation 2:** collect from the recalibrated generation-1 net, train with the same Zq recipe,
  recalibrate by a procedure fixed now, and play two matches. Gen 2 vs gen 1 is the step-2 claim;
  gen 2 vs the SL base is the cumulative test;
- **Options**, costed but not registered.

## Why

The owner's goal is **1–2 measured steps of RL improvement from the SL base**, and a
multi-generation loop is the main path to strength. Where step 1 stands, with every number from the
[follow-up results](2026-11-21-rl-run1-followup-results.md) and the
[evidence log](2026-11-21-rl-loop-evidence-log.md):

- **Match 8, recalibrated Zq vs the SL base: +9.61 [+3.27, +15.96]**, 6,400 games, seed
  2026112191, measured clustered SE 3.24. By its registered rule it reads **"adoptable" and
  "improves on the base"**. That reading only **nominates**; it adopts nothing.
- **Its caveats are why step 1 needs confirmation:**
  - (T, b) = (0.7036189636786957, −0.22711530923843357) was fitted **post hoc**, after the pilot
    gate, on this one net, toward the base's value shape on the pilot's hold-out games
    (`pilot-value-diagnosis/out/recal_Zq.json`).
  - It is one net from one generation.
  - **Seven registered readings were made that day with no multiplicity correction** (pilot
    matches 1, 3, 4; follow-ups 5, 7, 6, 8).
- **What the other cells say:**
  - P0 single net: +2.2 [−4.2, +8.6].
  - Raw Zq vs base: −10.0 [−23.2, +3.2].
  - S's value: −21.4, not rescued by recalibration.
  - The policy-ordering penalty was scoped and declined (#1488).

The recipe that won match 8 is therefore Zq's training (0.5 outcome + 0.5 stored search WDL) plus
a two-parameter WDL recalibration toward the previous generation's value shape. This record runs
that recipe once more, from its own output, and measures each step.

## (A) Confirming step 1 — three ways, one chosen

**What a new arena seed changes.** `arena_standard.py` builds `np.random.default_rng(seed)`. From
it, `load_paired_openings` draws 1,600 openings, weighted by book frequency, from the pilot's
`eval.pgn` (4,000 lines). The same seed also drives the temperature-0.1 move sampling.
- **Openings.** Two seeds share about 40% of their draw: 622–669 of 1,600 openings across every
  pair among 2026112191 (match 8) and this record's 2026112281/83/84/86, computed with the arena's
  own `load_paired_openings`. The most frequent book lines recur.
- **Games do not repeat across seeds**, and do not even repeat under the same seed:
  - Graft looks 1/2/3 (the same two nets; seeds 2026112141/42/43) shared 74–82 (opening, colour)
    keys per pair. Exactly 0 games were identical (one pair had 1). The same result occurred on
    0.32–0.48 of the shared keys, against a chance agreement of 0.34–0.36.
  - Match 8's smoke and real run used the SAME seed: 4 shared keys, 0 identical games. Batched
    evaluation is not bit-deterministic at temperature 0.1.
  - So a new-seed replication of the same two nets is close to independent at game level. The
    opening is a covariate, not a source of bias. What a replication does **not** change: the
    nets, the (T, b) fit and the eval book.
- **Unused openings.** 3,678 of the 4,000 eval openings were used by at least one banked match
  (match 8 alone used 1,600). Only about 322 were never drawn, so **no fresh disjoint 1,600-opening
  set exists** in this book. A replication shares openings with match 8 at the independent-seed
  rate (644 for 2026112281), the same as every earlier pair of matches.

**The three ways**, with power on the measured SE 3.24. The planning effects are match 8's +9.6,
which is likely an overestimate (it was selected as the winner of seven readings), and +5:

| Way | What it confirms | Cost (GPU-h) | P(LB95 > 0) at true +9.6 / +5 | Licenses |
|---|---|---|---|---|
| (i) replicate match 8, same seats, new seed | the same net + the same post hoc (T, b) beat the base | 1.45 + smoke | 0.84 / 0.34 | step 1 confirmed for **this net**; says nothing about the recipe's repeatability |
| (ii) fresh generation-1 replicate: new collect seed, Zq training, (T, b) refit by the registered procedure BEFORE the match, 6,400 vs base | the **recipe**, with no post hoc fit | ≈ 3.4 (collect 1.54, train 0.27, fit 0.1, match 1.45) | 0.84 / 0.34 | the recipe repeats; a second, independent gen-1 net |
| (iii) fold confirmation into gen 2 | gen 2 vs base is a fresh, preregistered cumulative test | 0 extra | — | if gen 2 beats the base, step 1 + step 2 together beat it; it cannot say which generation contributed |

**Chosen: (i) as Stage 0, then (iii) inside generation 2. (ii) stays an option** (part C).
- **Why (i).** It is the cheapest independent test of the claim generation 2 builds on. Its
  failure mode decides whether generation 2 runs:
  - if Stage 0 reads **damage** (UB95 < 0), generation 2 does not launch (the launcher refuses,
    exit 91);
  - any other reading lets generation 2 proceed. An "unread" replication does not stop the
    record: (iii) is the confirmation that counts, and (i) only guards against building on a
    reversal.
- **Why not (ii) first.** It costs 2.3× more and delays the second step by about 2 GPU-h. Its
  question, whether the recipe repeats, is answered more usefully by generation 2 itself: the
  same recipe applied again, from its own output, with a refit made before any match.

**Stage 0 rules** (`pool.json` → `result.pooled.elo_ci95`; the arm is the recalibrated seat):

| Reading | Condition | Consequence |
|---|---|---|
| damage | UB95 < 0 | match 8 is not replicated; generation 2 does not launch; back to the owner |
| replicates | LB95 > 0 | step 1 replicated on a new seed for this net |
| adoptable | LB95 > −10 and not "replicates" | consistent with match 8; does not confirm it |
| unread | otherwise | nothing |

Stage 0 is the eighth reading in this family. It is reported as such, with no correction, and
nothing is adopted on it alone.

## (B) Generation 2

Everything in generation 2 runs at the pilot's PIN `0c795fb7ab5a46e20848f6b0013de5a686fcb8ad`:
the collect, training, both fits and every match.
- **The collect's play path is the same code.** The pilot collected at C0 `6c72b821b`. The only
  change to `chess_anti_engine/simple_rl/collect.py` between C0 and the PIN is `a5ebcf8c7`
  (#1470): seal statistics are streamed shard by shard, a post-play change. The rest of the
  C0 → PIN diff is evaluate/arena/pool/fit/audit scripts and tests.
- Running at the PIN removes the code base as a confound, and the PIN carries the seal-OOM fix
  the pilot needed.

### B1. Bake the recalibration into the WDL head

`bake_recal.py` (in the bank's `prelaunch/`, sha256 below) writes a checkpoint whose plain forward
equals the recalibrated forward.
- **Why it is exact.** The WDL logits are the output of one linear layer, `value_wdl.net.2` (3 ×
  128 weight + bias). Nothing after it touches the logits, and the coupled categorical head reads
  the 128-d hidden layer, not the logits. So `l / T + [0, b, 0] = (W / T) h + (c / T + [0, b, 0])`.
- **What the bake does.** It replaces W with W / T and c with c / T + [0, b, 0], computed in
  float64 and stored in the tensor's dtype. Every other tensor and checkpoint field is copied
  unchanged. It refuses to overwrite, and writes a provenance JSON (sha256 of source and output,
  T, b, changed keys).
- **Test (run on CPU, fp32).** `bake_recal.py verify` loads the source and baked files with the
  arena's loader (`uci.model_loader.load_model_from_checkpoint`) and forwards the first 2,048
  hold-out rows of the pilot collect's `shard_00000`. It requires:
  - max |baked logits − (source logits / T + [0, b, 0])| ≤ 1e-4;
  - max |softmax difference| ≤ 1e-5;
  - the policy output bit-identical.

  On Zq with the banked (T, b) (`bake_test/verify.log`):

  | Measure | Value |
  |---|---|
  | max logit difference | 5.4e-6 |
  | max probability difference | 1.0e-7 |
  | policy | bit-identical |
  | how far the recalibration itself moves these probabilities (max) | 0.136 |

  The baked file loads with `weights_only=True`. Over all 39,237 pilot hold-out rows, the
  recalibration moves Zq's WDL probabilities by 0.0375 per entry on average (p99 0.128).
- **Mutants**, both run through the same `verify` and both **fail** (exit 1, `bake_test/mutants.txt`):
  - the draw bias omitted (`c / T` only): max logit difference 0.227;
  - W multiplied by T instead of divided: max logit difference 14.5.
- **On the GPU.** The arena and collector run bf16 inference, so there the baked net and the
  recalibrated seat differ only by bf16 rounding (W/T rounded, rather than the result scaled after
  rounding). The collect's take-effect check (B3) uses a tolerance that covers this.

### B2. Collect from the baked generation-1 net

- **Net:** `gen1_baked.pt` = Zq (`arm_Zq/gen_0001/train/model.pt`, sha256 `ad16d2f1…`) + the
  banked (T, b), baked and verified by the launcher's first step.
- **Flags:** the pilot's collect flags verbatim. Only the checkpoint and the seed change:

  ```
  collect --run-dir gen2/collect --generation 1 --openings-dir $P/openings
    --init-checkpoint gen2/gen1_baked.pt --device cuda --seed 2026112282 --games 8000
    --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 --topk 16 --c-scale 0.1
    --c-visit 50 --max-plies 450 --syzygy <production pair> --sf-path <stockfish>
  ```
- **The search uses the recalibrated value.** The collector's MCTS evaluates `gen1_baked.pt`, and
  that net's WDL output is the recalibrated one. Every leaf value, the root Q and the stored
  `search_wdl` target are therefore built on the recalibrated value, not on raw Zq. This is
  deliberate: the recalibrated net is the one that measured as stronger.
- **Generation 1's collection, for comparison:** the SL base collected, seed 2026112100, 8,000
  games, 761,979 rows, draw rate 0.49, CUDA graphs, 0 buffer-full flushes. The launcher writes
  both collections' games, rows, draw/decisive rate, mean plies, resumes and buffer-full flushes
  to `collect_compare.json`.

### B3. Collect checks and the take-effect proof

- **Gates:** 8,000 games and no resumes, or the launcher stops.
- **Take effect.** `take_effect_collect.py` proves that the collect searched with the baked net,
  not the raw one, from the consumer's own record: the collector writes the WDL of the net it
  searched with as `root_net_wdl` on each row.
  - On every hold-out row it compares the stored value with a fresh forward of (a) the baked
    file and (b) raw Zq.
  - **Pass iff** mean |stored − baked| ≤ 0.004 and mean |stored − raw| ≥ 0.02.
  - Scale: the pilot's stored-vs-fresh-forward noise for the SL base was 0.00085 per entry
    (`holdout_analysis.json`), and the recalibration moves Zq by 0.0375.
- **Self-test on the pilot collect, CPU, 2,048 hold-out rows:**
  - the true collecting net (SL base) as "baked" and Zq as "raw": **pass** (stored − SL base 0.0011, stored − Zq 0.0452);
  - swapped, the mutant: **fail**, exit 1 (0.0452 / 0.0011) (`take_effect_test/`).

### B4. Train generation 2

**From generation-1 weights, with the same Zq recipe.**
- **Lineage:** `--init-checkpoint gen2/gen1_baked.pt`, in a fresh single-generation run dir
  `gen2/arm_Zq2`, with `--collect-from gen2/collect` (the pilot's cell pattern).
- **Sha chain:** gen 2 → `gen1_baked.pt`, recorded in `arm_Zq2/gen_0000/manifest.json` inputs →
  Zq, recorded in the bake JSON's `src_sha256` → the SL base, recorded in the pilot's arm_Zq
  manifest.

| Flag | Value |
|---|---|
| `--presentations` | 1 |
| `--mirror-prob` | 0.5 |
| `--batch-size` | 512 |
| `--lr`, `--lr-floor` | 3e-6, 3e-6 |
| `--lr-schedule` | flat |
| `--warmup-steps` | 0 |
| `--optimizer` | aurora |
| `--optimizer-hparams-source` | init-checkpoint |
| `--optimizer-state`, `--init-optimizer-state` | fresh, fresh |
| `--train-target-temp` | 1.0 |
| `--kl-policy`, `--kl-wdl` | 0, 0 |
| `--sl-mix-frac` | 0 |
| `--holdout-frac` | 0.05 |
| `--probe-rows` | 2048 |
| `--no-eval` | set |
| `--value-outcome-weight`, `--value-search-weight` | 0.5, 0.5 |
| `--value-search-source` | stored_search_wdl |
| collect flags (validated, not re-run) | as B2 |

- **Why fresh optimizer state.**
  1. Generation 1 trained from fresh moments, so fresh moments keep the training step identical
     and leave the collecting net as the only data-affecting change.
  2. Generation 1's end moments come from 1,412 updates on generation-1 data. The bake rescaled the
     WDL head's weights by 1/T, so that layer's second-moment estimates no longer match its
     weights.
  3. `continue` has never been measured here.
- **Hyper-parameters come from the init checkpoint.** I checked that Zq's saved param groups equal
  the SL base's in every field except the realised `lr`/`initial_lr`, which `--lr` sets (the Aurora
  group keeps its 20× ratio). The baked file's groups equal Zq's.
- **Take effect.** After training, the launcher requires the trainer's own summary to show:
  - `realised_w_wdl` 1.0, `realised_search_wdl_frac` 0.5, `anchor` none, target temperature 1.0,
    SL fraction 0;
  - `realised_optimizer` and `realised_lr_param_groups` **equal to generation 1's**.
- **KL is off, as in generation 1.** The value-anchor screen (#1475) is testing `--kl-wdl` toward
  the SL base on the pilot's data. Whatever it finds does not change this recipe (recipe frozen,
  below).
- **Why a fresh single-generation run, not the loop's multi-generation path.** `run --generations
  2` would train generation 2 from the sealed generation-1 model of the same run dir, but:
  - generation 1 lives in the frozen pilot bank (`arm_Zq`);
  - the loop has no bake step between generations, so its generation-2 collect would search with
    raw Zq, which is the net that measured −10.0, not the champion.

  Single runs keep the pilot bank frozen and put the bake explicitly in the chain. The cost is
  that lineage is carried by sha, not by the loop's own `gen_000N` dirs, so the launcher records
  every sha. Moving the bake into the loop so `--generations N` works is the change to make
  before generation 3. It is not made here; it would change the code pin.

### B5. Per-generation recalibration — the procedure, fixed now

For generation k ≥ 2, before any match:
- fit (T_k, b_k) with `scripts/fit_value_recal.py` at the PIN (sha256 `37e27420…`):
  `--net` = generation k's trained model, `--reference` = generation k−1's **baked** file,
  `--collect-dir` = generation k's own collect, `--holdout-frac 0.05`, `--device cuda`. The fit
  uses hold-out games only;
- bake (T_k, b_k) into generation k's model with `bake_recal.py`, and run `verify`;
- generation k plays only as its baked file.

**Fit control.** The same script on Zq vs the SL base on the pilot collect must reproduce the banked
(T, b) within 1e-3 on each parameter. GPU bf16 forwards are not bit-reproducible, so the check is
not exact equality. If it fails, the launcher stops before the collect. The control also proves
that the procedure is the one match 8's (T, b) came from.

**Why toward the previous generation, not the SL base.** The recalibration's job is to keep each
generation's value shape where the last measured champion's was, so that a generation's
over-sharpened or flattened WDL from its own targets does not reach the search uncorrected. Fitting
to the SL base every time would anchor every generation to the base's shape. That is a separate
recipe, and the anchor screen is the experiment that asks about it.

### B6. Matches

Two registered matches, at the PIN, with match 5's rules.
- **Seats:** plain seats, both files baked; no `--*-value-recal` flags.
- **Launcher and smoke:** the follow-up launcher (sha256 `7c8fd065…`), 6,400 games each (3,200
  per seat), a 4-game-per-seat smoke first.
- **Readback, after both smoke and real runs:** every result row's `git_sha` must equal the PIN,
  and `candidate_sha256` / `reference_sha256` must equal the intended files in each seat.

| Match | Seats | Seed | Claim |
|---|---|---|---|
| **gen 2 vs gen 1** | `gen2_baked.pt` vs `gen1_baked.pt` | 2026112283 | **the step-2 claim**: "improves" iff LB95 > 0 |
| gen 2 vs SL base | `gen2_baked.pt` vs `sl_base.pt` | 2026112284 | cumulative, fresh and preregistered: (iii) |

Readings for both: damage iff UB95 < 0; adoptable iff LB95 > −10; **improves** iff LB95 > 0;
otherwise unread. Match 5's tie-break is kept.

**Power** (SE ≈ 3.24):
- **Gen 2 vs gen 1:** a second step as large as match 8's (+9.6) reads "improves" with
  probability 0.84; a +5 step with 0.34.
- **Gen 2 vs base:** if the steps add (about +19), LB95 > 0 with probability ≈ 1.00. Elo is not
  transitive across opponent pairs, so this is a planning figure.

| Outcome | Licenses |
|---|---|
| gen 2 vs gen 1 **improves** | **the second measured step of RL improvement**; with gen 2 vs base, the 2-step claim. Generation 3 under a new record, after the bake is moved into the loop |
| gen 2 vs gen 1 adoptable, not improves | generation 2 is safe to collect from, but no second step is measured; the owner decides between generation 3 and option (ii) |
| gen 2 vs gen 1 damage | the recipe does not repeat from its own output; stop; the owner decides |
| gen 2 vs base improves (with any gen 2 vs gen 1) | the cumulative claim, which includes step 1 (the (iii) confirmation) |

**Recipe frozen.** Within this record the recipe is exactly B2–B5. A competing nomination, for
example from the value-anchor screen or the sibling screen, does not change it mid-record. It is
settled by a head-to-head match against this record's champion, under the next record.

### B7. Seeds and what differs from generation 1

**New seeds.** Each was checked unused in the repo at `origin/main` (`git grep`), in every open
PR's diff, in the rl-run1-pilot / followup / value-anchor / sibling / root-full-width /
rl-claude-20261119 banks' scripts, records and JSON, and in every banked `results.jsonl` `seed`
field. Each appears only in this record's own bank.

| Seed | Use |
|---|---|
| 2026112281 | Stage 0 replication match |
| 2026112282 | generation-2 collect |
| 2026112283 | gen 2 vs gen 1 |
| 2026112284 | gen 2 vs SL base |
| 2026112285 / 2026112286 | option (ii): collect / match (not registered) |
| 2026112287 | option C2 graft match (not registered) |

**Everything that differs, generation 1 → generation 2** (rule 3: one data-affecting change per
readout window):

| Item | Generation 1 | Generation 2 | Data-affecting? |
|---|---|---|---|
| collecting net | SL base | `gen1_baked.pt` (Zq + recal) | **yes: the intended change** |
| collect seed | 2026112100 | 2026112282 | yes, a re-roll (unavoidable) |
| code at collect | C0 `6c72b821b` | PIN `0c795fb7a` | no (play path unchanged; see B) |
| training init | SL base | `gen1_baked.pt` | yes, part of "a generation" |
| init checkpoint's `grad_clip_mode` field | none | `zclip` | no: the realised optimizer must equal generation 1's (checked) |
| recalibration reference | SL base (match 8's fit) | `gen1_baked.pt` | the registered per-generation rule |
| training data | generation-1 collect | generation-2 collect | follows from the collecting net |
| every training flag | as B4 | identical | — |
| match settings | match 8's | identical, plain seats with baked files | no: the bake is exact (B1) |

**Confounds.** The collecting net and the training init change together. That is the definition
of a generation, not two changes inside one readout. The re-rolled collect seed is unavoidable.

## (C) Options — costed, not registered

**C1. Generation 2 from P0.** P0 trained the policy only (`--w-wdl 0`) and measured +2.2
[−4.2, +8.6] as a single net vs base, so it is unread, with no evidence that it is stronger.
Collecting from P0 would build generation 2 on a net that has not measured above the base, and
the step-2 claim would then be relative to an unconfirmed step 1. Cost is the same as B (≈ 4.9
GPU-h). **Not recommended.**

**C2. A P0-policy + recalibrated-Zq-value graft.** P0 and Zq are separate fine-tunes of the whole net,
and both heads read their own trunk. So no single checkpoint carries P0's policy and Zq's value:
the graft is a hybrid two-forward seat (`--candidate-value-from` + `--candidate-value-recal`). A
collector cannot play a two-net seat, so the graft could only be a match diagnostic, never a
generation-2 collector without new code. A graft vs recal-Zq match would cost about 2.2 GPU-h for
6,400 games (hybrid seats are measured at 0.27 GPU-h per 800 games), and it answers only whether
P0's policy adds to Zq's. It decides nothing in this record's chain, so **it is not proposed as a match.** It
becomes worth running only if generation 2 reads unread vs generation 1 and the owner wants to
know whether the policy target or the value target is carrying the gain.

**C3. Option (ii), a fresh generation-1 replicate** (part A): ≈ 3.4 GPU-h, seeds 2026112285 /
2026112286. It confirms the recipe with no post hoc fit. Recommended only if Stage 0 reads unread
and the owner wants step 1 confirmed before generation 2.

## (D) Cost and queue

All figures are measured on this machine. No GPU work was run for this record.

| Item | GPU-h | Source |
|---|---|---|
| Stage 0 (6,400 games + smoke) | ≈ 1.47 | 1.45 per 6,400 recal games (match 8) |
| bake + verify, generation 1 and 2 | 0 (CPU, ≈ 6 min each; the GPU is held idle meanwhile, ≈ 0.2 h of wall) | `bake_test/verify.log`: 5:57 wall, 3.6 GB RSS |
| fit control + fit (T2, b2) | ≈ 0.1 | fit_value_recal on 39k rows, two forwards, cuda |
| collect, 8,000 games | 1.54 | the pilot's play time, slowest process 5,534 s |
| collect checks + take effect | ≈ 0.05 | two cuda forwards over about 38k hold-out rows |
| train Zq2 | 0.27 | the anchor screen's first cell (16 min); the pilot's Zq was 859 s |
| two matches (12,800 games + smokes) | ≈ 2.95 | 1.45 per 6,400 |
| **total** | **≈ 6.4** (Stage 0 1.47, generation 2 ≈ 4.9) | |

**Per-step RAM floors.** The launcher refuses a step (exit 89) unless `MemAvailable` meets the floor:

| Step | Floor | Basis |
|---|---|---|
| collect, train | 40 GiB | the pilot's training loader holds 20–23 GiB at ≈ 760k rows (launch record) |
| fits, checks, bakes | 12 GiB | bake verify peaked at 3.6 GB RSS |
| matches, Stage 0 | 16 GiB | |

**Queue.** The anchor screen runs until about 05:00, then:
- the sibling screen, ≤ about 9 GPU-h;
- the full-width stages 0–2, about 5–6 GPU-h.

**Recommendation for the fastest second measured step:** run this record **directly after the
anchor screen**, before the sibling and full-width screens. The order is Stage 0 (≈ 1.5 h), then
generation 2 (≈ 4.9 h), which puts the step-2 reading about 6.5 GPU-h after the anchor screen ends
instead of about 21 GPU-h.
- **Neither screen is a prerequisite.** Each tests a recipe change that, by the frozen-recipe
  rule, would be settled against this record's champion in the next record anyway.
- **The anchor screen's cost.** If its Zq cells read well, they become a competing nomination,
  settled by head-to-head under the next record. Waiting for it gains nothing for this record.
- **The order is the operator's to set,** through `GPU_RELEASED_BY_OPERATOR` (below). This record
  only recommends it.

## Guards (as the sibling and full-width records)

- **Code pin.** Every GPU step runs in the PIN worktree, checked at HEAD = PIN with no tracked
  changes. `fit_value_recal.py` is sha-checked at the PIN, and every match result row must carry
  `git_sha` = PIN.
- **Matches at the PIN, with seat readback:**
  - Stage 0 uses `check_seat.py` (sha256 `8f4ce9eb…`), as match 8 did.
  - Generation 2's plain seats are proven by the result rows' `candidate_sha256` /
    `reference_sha256` against the baked files' sha256, smoke and real.
- **The GPU guard** (`gpu_guard.sh`, adapted from the full-width bank's v2; only the bank path and
  stage names differ) refuses unless all of these hold:
  - GPU memory used < 1024 MiB;
  - no `arena_standard.py` / `simple_rl_loop.py` / `match_vs_uci.py` process (token match on
    `ps` args);
  - both queue markers exist (`rl-run1-followup/chain.exit`, `rl-run1-value-anchor/queue2.exit`);
  - the operator's `GPU_RELEASED_BY_OPERATOR` in this bank has a line whose first word is the
    stage (`stage0` or `gen2`);
  - and it then takes a bank-local `flock`.
- **The launchers** (`set -euo pipefail`) check, in this order:
  1. OWNER_GO (exit 90);
  2. the guard: an existence check, then `source … || exit 8`, before any sha check or other path
     use;
  3. Stage 0 not "damage" (gen2 only; exit 91);
  4. no existing collect (95) or match dir (95);
  5. launcher and script sha256s (93);
  6. the PIN worktree (96/97);
  7. the RAM floor per step (89).
- **Test.** `guard_test/guard_test.sh` (v2) runs everything under a fake HOME (an empty dir), a
  PATH that shadows `python`, `python3`, `nvidia-smi` and `ps` with stubs, and a fake bank and
  operations dir. It **asserts** that isolation before any case runs.
  - It has 27 cases, covering every guard refusal, every launcher refusal, a missing guard file
    and a guard whose `source` fails, for each launcher. It passes 27/27, and no case reaches
    python.
  - **Mutants**, each killed:

    | Mutant | Result |
    |---|---|
    | launcher without `|| exit 8` | "source fails" case exits 3 |
    | gen2 with no guard | |
    | guard with no memory check | |
    | guard with no release-line check | |
    | guard with a substring `ps` match | |
    | harness with stubs last on PATH | ABORT |
    | harness with another HOME | ABORT |

## Provenance and incident

- **Incident (2026-11-21, 23:51–23:58 EST).** Version 1 of this bank's guard test ran the
  launchers under the real HOME, with a fake bank that held no `gpu_guard.sh`. Version 1 of
  `run_stage0.sh` had no `set -e`, and it sourced the guard without `|| exit`. The failed `source`
  fell through to the real follow-up launcher, which started a real 4-game arena (recalibrated Zq
  vs the SL base, 64 sims, output inside the test's temp dir). This happened in each of five test
  runs.
  - **Overlap.** The arenas shared the GPU with the anchor screen's cell `Zq_k18.1` (started
    23:52:31). That cell is matched by updates, not by the clock, and its log shows no OOM, CUDA
    error or traceback.
  - **Evidence.** No arena output survives (the test's temp dirs were deleted). A CUDA JIT cache
    entry written at 23:57:07 is consistent with at least one arena reaching CUDA initialisation.
    No file outside the temp dirs was written by these runs.
  - **Fix.** The fixes are the v2 launchers and test described under Guards. Since then every
    command in this work has run with `CUDA_VISIBLE_DEVICES=` exported.
  - The anchor screen's results record carries the overlap window on `Zq_k18.1`.
- **Bank** `$OPS/rl-run1-gen2/` (`$OPS` = `~/chess-artifacts/operations`). It holds nothing yet
  except the files below and the CPU test outputs.

| File | sha256 |
|---|---|
| `gpu_guard.sh` | `a7b4deb29d436321f15019c0d7c2d74d831df2222be29dec001db9f4add95dab` |
| `stage0/run_stage0.sh` | `0a02aead0994e1879c74a7797401f5d5db4c51d396f2ab0db84967a1c5b05a28` |
| `gen2/launch_gen2.sh` | `9a61eefbe56201ffeb3b8e7e551f0ed945f08adbe44d02e4816149bcbe150675` |
| `prelaunch/bake_recal.py` | `786bb537e0d1130ed2ae20bf81c7791239f4903095fd22a2f40840e3e8a06230` |
| `prelaunch/take_effect_collect.py` | `9029f146958166c82b83da64de93852ce9adc5edc25171229f4c033f2607e37e` |
| `guard_test/guard_test.sh` | `fd0aa78af4e1d1507aab92831045879bccfd4c8c8738568e79da5edda602e847` |

`prelaunch/SHA256SUMS` lists the two prelaunch scripts, and `launch_gen2.sh` checks it.

## Commands

One command per stage. Each refuses without its OWNER_GO file and the operator's release line.

```bash
OPS=$HOME/chess-artifacts/operations
# Stage 0 (after: owner go -> stage0/OWNER_GO; operator -> "stage0 <time>" in GPU_RELEASED_BY_OPERATOR)
setsid nohup $OPS/rl-run1-gen2/stage0/run_stage0.sh > $OPS/rl-run1-gen2/stage0/run.out 2>&1 < /dev/null &
# Generation 2 (after Stage 0's verdict.json is not damage; owner go -> gen2/OWNER_GO; operator -> "gen2 <time>")
setsid nohup $OPS/rl-run1-gen2/gen2/launch_gen2.sh > $OPS/rl-run1-gen2/gen2/run.out 2>&1 < /dev/null &
```

Results go into a separate results record, in the same session as the readout, judged by the rules
above.
