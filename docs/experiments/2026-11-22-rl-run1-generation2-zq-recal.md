# 2026-11-22: RL run 1, generation 2 — Zq + per-generation recalibration (pre-registration)

**Status: registered; launch authorised by the owner's standing statement** (2026-11-22, verbatim):
"You dont have to wait for me to approve gpu use it dont have time ill just tell you to stop if
necessary". It launches on the operator's release once this record is merged. The go-ahead files
stay as the mechanism: `stage0/OWNER_GO` and `gen2/OWNER_GO`, one line each (who, when, the
message), written by the coordinator quoting that statement. Each stage launcher still refuses
without its file (exit 90), and an owner's stop ends the run. Every hold below that concerns a
RESULT (a stop rule, a registered note, a reading that goes "to the owner") is unchanged: those
holds stop the run and the owner decides; the statement authorises starting, not continuing past
them. Before this revision nothing here ran on the GPU; the sizing below is CPU work plus measured
costs. The first GPU step is Stage 0; the 3-update training memory smoke runs after Stage 0's
verdict and before the generation-2 release (revision 5; below).

**Revision 2** applies the #1491 statistics review at `a785315b6` (evidence:
`$OPS/rl-claude-20261119/plan-review-1491/`):
- the licensing rule for "two measured steps";
- a collect gate that can fail, with the pilot's resume explained;
- EXIT-trap markers and a resume path;
- a fail-closed guard;
- a sealed bank, with the guard's sha pinned;
- the baked-start confound, named and sized;
- what the recalibration rule can and cannot learn, with registered bounds on (T2, b2);
- Stage 0 plays the baked file;
- the order of readings;
- the incident wording corrected.

**Revision 3** applies the #1491 statistics review at `730e07509` (evidence:
`$OPS/rl-claude-20261119/plan-review-1491/delta-730e07509/`):
- **the baked start with no compensation is the multi-generation rule** (the coordinator's
  decision, on the reviewer's measured step-0 trunk gradients; B4);
- `move_aside.sh` can no longer re-draw a registered reading: it refuses completed steps,
  completed matches and a completed Stage 0, and accepts only plain step names;
- the licensing table names the contradictory cell and carries the cumulative claim whenever gen 2
  vs base improves (48-cell grid re-fabricated and banked);
- the (T2, b2) bounds text: [0.4951, 1.4212] are the T bounds; the ln T bounds are
  [−0.703, +0.352]; it is a sanity stop, not a statistical bound;
- completion checks verify data and inputs (actor, seed, shard digests, model and source-collect
  shas; a `<step>.inputs` record compared on resume);
- the confound line is the measured trunk gradient ratio, and generation 2 re-measures it on its
  own rows before training (report-only, with a registered stop for the owner).

**Revision 4** applies the #1491 statistics review at `950be80e3` (evidence:
`$OPS/rl-claude-20261119/plan-review-1491/delta-950be80e3/`):
- **a fully played match can no longer be moved aside and replayed.** Both matches and Stage 0
  resume PER SEAT: a finished seat (exit 0, exactly n games, unique pairs, one header and one result
  row with the seed, the nets' sha256 and the PIN) is never replayed; only the missing seat plays,
  then readback and pool run again. `move_aside.sh` refuses a whole match or Stage 0 once any real
  seat's `.exit` is 0, and moves one interrupted seat only with `--seat A|B`. A seat whose result row
  names other nets (a sha readback failure) is owner-only: a registered note, never a move;
- `trunk2` runs 8 batches, re-measured (9:17 wall, 14.1 GB RSS), with per-batch values and their SE;
  the stop range is a mean-based sanity range [0.59, 1.35], symmetric in log space around the baked
  start, and only holds for the owner;
- both anchor-screen results are in the evidence, as existing competing nominations (three
  nominations from one generation each, never ranked by point estimate); the multiplicity count is
  nine readings before Stage 0;
- **the GPU guard could never have passed on this machine** and is replaced (guard v3, the
  coordinator's addition): v2 required every `nvidia-smi` memory line < 1,024 MiB, but this
  machine's IDLE baseline is 3,382 MiB (the Windows side of WSL2). v3 judges memory against the
  operator's measured `idle_mib` (± 512 MiB), adds a 5-sample utilisation check (< 10 %), and
  re-checks before every step and every arena seat;
- **the first GPU step is a 3-update training memory smoke** (revision 5 moves it after Stage 0's
  verdict; what the sibling screen lacked; its
  registered step-1b probe died of CUDA out-of-memory on its 2nd update):
  `prelaunch/train_mem_smoke/train_mem_smoke.py` runs generation 2's own training configuration (the
  launcher's `$COLLECT2 $TRAIN $ZQ` flags) for 3 updates and writes the peak allocated and reserved
  memory after EACH update. The screen aborts if peak reserved exceeds 0.85 × the card's total
  (details under Guards);
- **the collect stage's expected GPU memory** is stated from the pilot's banked collect at the same
  flags (under Guards);
- **launch is authorised** by the owner's standing statement of 2026-11-22 (Status, above).

**Revision 5** applies the #1491 statistics review at `d8f91bcab` (evidence:
`$OPS/rl-claude-20261119/plan-review-1491/delta-d8f91bcab/`):
- **the memory smoke could not have run where revision 4 put it.** It was sequenced before Stage 0,
  but its `--init-checkpoint` is `gen1/gen1_baked.pt`, which Stage 0 writes; the reviewer's CPU
  run of the exact command died in `train_settings` 2 s in, and the record's own rule would have
  turned that into ABORT + `HOLD_MEM`. The smoke now runs **after Stage 0's verdict and before the
  generation-2 release line**, on the baked generation-1 file itself;
- **one command, nothing retyped.** `prelaunch/train_mem_smoke/run_smoke.sh IDLE_MIB` runs the
  operator's GPU gate itself (the sibling bank's `operator_gpu_check.sh`, sha-pinned), refuses
  before Stage 0's verdict, with damage or with a gen1 file other than the one Stage 0 played, and
  passes `$COLLECT2 $TRAIN $ZQ` from **`gen2/gen2_flags.sh`, the one sha-pinned file the launcher
  also sources**. The harness proves the smoke and the training step receive identical strings;
- **the exact operator command was run once in CPU logic mode from a staged copy, end to end**
  (stand-in baked file and Stage 0 verdict; output banked), and that is now a pre-launch checklist
  item for every operator command (Guards);
- `seat_state.py`'s two untested DONE clauses (unique `(pair_id, half)`, the header seed) and
  Stage 0's verdict thresholds are now exercised (the reviewer's surviving mutants R3–R5 die);
- the flaky TERM case waits for the launcher's trap (bound 120 s), and the mutant runner reruns a
  kill by that case alone;
- the collect's measured whole-card peak is cited beside the model; the smoke's PASS confirmation
  adds "reserved memory flat between updates 2 and 3", and the allocations it does not cover are
  named with their margin;
- `trunk2` is deterministic on the same inputs, so moving `HOLD_TRUNK2` aside cannot clear the
  hold by re-rolling (B4);
- **Commands is the reviewer's operator sequence, steps 0–10.**

**What it is.** Match 8 nominated "Zq + per-generation recalibration" for a 2-generation run under
its own record. This is that record. It has three parts:
- **Stage 0:** a replication of match 8's player under a new seed;
- **Generation 2:** collect from the recalibrated generation-1 net, train with the same Zq recipe,
  recalibrate by a procedure fixed now, and play two matches. Gen 2 vs gen 1 is the step-2 claim
  and the PRIMARY reading; gen 2 vs the SL base is the cumulative test;
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
    matches 1, 3, 4; follow-ups 5, 7, 6, 8). An eighth and a ninth followed on 2026-11-22 (the
    anchor screen's two Stage B matches; see B6).
- **What the other cells say:**
  - P0 single net: +2.2 [−4.2, +8.6].
  - Raw Zq vs base: −10.0 [−23.2, +3.2].
  - S's value: −21.4, not rescued by recalibration.
  - The policy-ordering penalty was scoped and declined (#1488).
  - **Value-anchor screen, Stage B match 1 (2026-11-22): S + `--kl-wdl` 7.0 vs the SL base, 6,400
    games, seed 2026112181: +5.81 [−0.53, +12.16], "adoptable"** (nominates; it does not improve on
    the base). `~/chess-artifacts/operations/rl-run1-value-anchor/arenas/anchor_S_k7.0/pool.json`
    [`0722d72b`]. It is an existing competing nomination (B6).
  - **Value-anchor screen, Stage B match 2 (2026-11-22): Zq + `--kl-wdl` 11.3 vs the SL base, 6,400
    games, seed 2026112182, PIN `0c795fb7a`, 0 unresolved: +3.58 [−2.62, +9.79], "adoptable"**
    (LB95 > −10), not damage (raw Zq was −10.0 [−23.2, +3.2]); seats +7.6 / −0.4, seat offset
    +4.0 [−1.8, +9.9].
    `~/chess-artifacts/operations/rl-run1-value-anchor/arenas/anchor_Zq_k11.3/pool.json`
    [`2c890633`]. By the anchor screen's record it nominates the anchored Zq recipe only together
    with a "Zq + (T, b)" alternative, and it supports no claim that the anchor fixes content for Zq.

The recipe that won match 8 is therefore Zq's training (0.5 outcome + 0.5 stored search WDL) plus
a two-parameter WDL recalibration. This record runs that recipe once more, from its own output, and
measures each step.

## What this record can license (revision 2, review item 1)

"Gen 2 > gen 1" and "gen 2 > base" together do **not** imply "gen 1 > base": Elo across different
opponent pairs is not transitive, and the second match can be won on generation 2's gain alone. So:

| Stage 0 | gen 2 vs gen 1 | gen 2 vs base | Licensed claim |
|---|---|---|---|
| **replicates** (LB95 > 0) | **improves** (LB95 > 0) | **not damage** | **two measured steps** of RL improvement from the SL base |
| replicates | improves | **damage** (UB95 < 0) | **"inconsistent readings; no claim; owner"**: gen 1 > base, gen 2 > gen 1 and gen 2 < base cannot all hold, so no claim is made and the owner decides (revision 3) |
| adoptable or unread | improves | improves | **one measured step (step 2) and a cumulative improvement over the base; attribution between generations unread** |
| adoptable or unread | improves | not improves | one measured step (step 2) |
| any non-damage | not improves | **improves** | **no second measured step; a cumulative improvement over the base (gen 2 vs base improves); attribution between generations unread** (revision 3: the B6 claim is carried; 9 of the 48 cells) |
| any non-damage | not improves | not improves | no second measured step |
| damage | — | — | generation 2 does not run |

- **Grid (revision 3).** `guard_test/grid/grid_v3.py` extracts the launcher's verdicts code
  verbatim and runs it on all 48 non-damage combinations (Stage 0 × gen 2 vs gen 1 × gen 2 vs
  base, each reading fabricated): **48 of 48 equal this table**. Two mutants are killed: dropping
  the damage clause (1 cell differs) and dropping the cumulative-only row (9 cells differ).

- Generation 2 proceeds on any **non-damage** Stage 0.
- `verdicts.json` carries the licensed claim, computed by this table from the stored readings.
- **Order of readings (review item 9; recounted in revision 4).** Nine registered readings come
  before Stage 0. Seven are in the [evidence log](2026-11-21-rl-loop-evidence-log.md): (1) pilot
  match 1, Zq vs base; (2) pilot match 3, S vs base; (3) pilot match 4, the P0 hybrid vs base;
  (4) match 5, P0 single net vs base; (5) match 7, Zq recal vs Zq; (6) match 6, S's value recovery
  R = 6a − 6b (one registered reading from two matches); (7) match 8, Zq recal vs base. Two are in
  the [anchor screen's record](2026-11-21-rl-run1-value-anchor-screen.md) and not yet in the log:
  (8) S + `--kl-wdl` 7.0 vs base; (9) Zq + `--kl-wdl` 11.3 vs base. Stage 0 is the **tenth**.
  Gen 2 vs gen 1 is the **eleventh** and the **primary** reading; gen 2 vs base is the
  **twelfth**. No correction is registered; each is read at its own 95% interval.
- **No pooled match 8 + Stage 0 estimate is computed.** Match 8 was selected as a winner, so
  pooling would carry its selection bias; Stage 0 alone is the unbiased estimate of that player.

## (A) Confirming step 1 — three ways, one chosen

**What a new arena seed changes.** `arena_standard.py` builds `np.random.default_rng(seed)`. From
it, `load_paired_openings` draws 1,600 openings, weighted by book frequency, from the pilot's
`eval.pgn` (4,000 lines). The same seed also drives the temperature-0.1 move sampling.
- **Openings.** Two seeds share about 40% of their draw: 622–669 of 1,600 openings across every
  pair among 2026112191 (match 8) and this record's 2026112281/83/84/86, computed with the arena's
  own `load_paired_openings`. The most frequent book lines recur. The reviewer reproduced 644 for
  match 8 vs Stage 0 and measured the between-match error correlation from shared openings at
  ≈ 0.01.
- **Games do not repeat across seeds**, and do not even repeat under the same seed:
  - Graft looks 1/2/3 (the same two nets; seeds 2026112141/42/43) shared 74–82 (opening, colour)
    keys per pair. Exactly 0 games were identical (one pair had 1). The same result occurred on
    0.32–0.48 of the shared keys, against a chance agreement of 0.34–0.36.
  - Match 8's smoke and real run used the SAME seed: 4 shared keys, 0 identical games. Batched
    evaluation is not bit-deterministic at temperature 0.1.
  - So a new-seed replication of the same player is close to independent at game level. The
    opening is a covariate, not a source of bias. What a replication does **not** change: the
    net, the (T, b) fit and the eval book.
- **Unused openings.** 3,678 of the 4,000 eval openings were used by at least one banked match
  (match 8 alone used 1,600). Only about 322 were never drawn, so **no fresh disjoint 1,600-opening
  set exists** in this book.

**The three ways**, with power on the measured SE 3.24. The planning effects are match 8's +9.6,
which is likely an overestimate (it was selected as the winner of seven readings), and +5:

| Way | What it confirms | Cost (GPU-h) | P(LB95 > 0) at true +9.6 / +5 | Licenses |
|---|---|---|---|---|
| (i) replicate match 8's player, new seed | the same net + the same post hoc (T, b) beat the base | 1.45 + smoke | 0.84 / 0.34 | step 1 replicated for **this net**; says nothing about the recipe's repeatability |
| (ii) fresh generation-1 replicate: new collect seed, Zq training, (T, b) refit by the registered procedure BEFORE the match, 6,400 vs base | the **recipe**, with no post hoc fit | ≈ 3.4 (collect 1.54, train 0.27, fit 0.1, match 1.45) | 0.84 / 0.34 | the recipe repeats; a second, independent gen-1 net |
| (iii) gen 2 vs base inside generation 2 | a fresh, preregistered cumulative test | 0 extra | — | a cumulative improvement over the base; it cannot attribute it between generations |

**Chosen: (i) as Stage 0, plus (iii) inside generation 2. (ii) stays an option** (part C).
- **Why (i).** It is the cheapest independent test of the claim generation 2 builds on, and its
  "replicates" reading is required for the two-step claim (table above).
  - If Stage 0 reads **damage** (UB95 < 0), generation 2 does not launch (the launcher refuses,
    exit 91).
  - Any other reading lets generation 2 proceed, but caps what the record can license.
- **Why not (ii) first.** It costs 2.3× more and delays the second step by about 2 GPU-h. Its
  question, whether the recipe repeats, is answered more usefully by generation 2 itself: the same
  recipe applied again, from its own output, with a refit made before any match.

**Stage 0 plays the BAKED generation-1 file (revision 2, review item 8).** Stage 0 first bakes the
banked (T, b) into Zq (B1; CPU, verified) and plays `gen1/gen1_baked.pt` as a plain seat. That is
the exact player that collects generation 2 and sits in the gen 2 vs gen 1 match. Match 8 played
Zq + `--*-value-recal`, the same function computed in a different order.
- **Measured equivalence** (CPU bf16 autocast proxy, 1,024 hold-out rows,
  `prelaunch/equiv_test/pass.json`, matching the reviewer's numbers): the baked file differs from
  the recal seat by mean |Δp| 0.00027 (max 0.0044), with 0 WDL-argmax disagreements. The recal
  seat's own bf16-vs-fp32 noise is 0.00098.
- So **Stage 0 replicates match 8's player within measured inference noise**.
- Generation 2's first GPU step banks the same comparison on the GPU (B3a).

**Stage 0 rules** (`pool.json` → `result.pooled.elo_ci95`; the arm is the baked seat):

| Reading | Condition | Consequence |
|---|---|---|
| damage | UB95 < 0 | match 8's player is not replicated; generation 2 does not launch; back to the owner |
| replicates | LB95 > 0 | step 1 replicated on a new seed for this player; required for "two measured steps" |
| adoptable | LB95 > −10 and not "replicates" | consistent with match 8; does not replicate it |
| unread | otherwise | nothing |

Nothing is adopted on Stage 0 alone.

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
  The logits are side-to-move WDL, so the draw-bias index is POV-safe.
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

### B2. Collect from the baked generation-1 net

- **Net:** `gen1/gen1_baked.pt` = Zq (`arm_Zq/gen_0001/train/model.pt`, sha256 `ad16d2f1…`) + the
  banked (T, b), baked and verified by Stage 0. Generation 2 refuses (exit 92) unless the file's
  sha256 equals the one in Stage 0's `verdict.json` (`played_sha256`), its bake JSON names Zq's
  sha256 and the banked (T, b), and its CPU verify passed.
- **Flags:** the pilot's collect flags verbatim. Only the checkpoint and the seed change:

  ```
  collect --run-dir gen2/collect --generation 1 --openings-dir $P/openings
    --init-checkpoint gen1/gen1_baked.pt --device cuda --seed 2026112282 --games 8000
    --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 --topk 16 --c-scale 0.1
    --c-visit 50 --max-plies 450 --syzygy <production pair> --sf-path <stockfish>
  ```
- **The search uses the recalibrated value.** The collector's MCTS evaluates `gen1_baked.pt`, and
  that net's WDL output is the recalibrated one. Every leaf value, the root Q and the stored
  `search_wdl` target are therefore built on the recalibrated value, not on raw Zq. This is
  deliberate: the recalibrated net is the one that measured as stronger.

### B3. Equivalence, collect checks and the take-effect proof

**B3a. Equivalence on the GPU (the first GPU step, `equiv_check.py`).**
- **What it does.** It loads raw Zq as the arena's recal seat (`HybridValueNet(raw, None,
  recal=(T, b))`) and `gen1_baked.pt` as a plain net. Both forward the first 2,048 pilot hold-out
  rows under the arena's own `inference_autocast` and in fp32.
- **Registered tolerance:** mean |p_baked − p_seat| under autocast ≤ the seat's OWN autocast-vs-fp32
  noise measured in the same run, and WDL-argmax disagreements ≤ 0.1% of rows.
- **Self-test on CPU** (bf16 proxy, 1,024 rows): **pass**, 0.00027 against a noise of 0.00098, with
  0 disagreements.
- **Mutant:** a seat with the wrong draw bias (b = −0.20 instead of −0.227) **fails** (exit 1):
  0.0022 against a noise of 0.00098, with 4 disagreements (`prelaunch/equiv_test/wrong_b.json`).

**B3b. The collect gate (revision 2, review item 2).** Revision 1's "no resumes" gate read
`game_resumes`, a key that does not exist at the PIN. It came back as None and the gate passed
always. `collect_gate.py` now reads by subscript, so a missing key is a FAIL. It requires:
- `games == 8000`;
- `stats["procs_resumed"] == []`;
- `stats["unresolved_games"] == 0`;
- `stats["duplicate_game_ids"] == 0`, and 8,000 distinct ids in `games.jsonl`.

**What happened in the pilot's collect** (`procs_resumed` `[0, 1, 2]`): no game was replayed or
dropped.
- **Mechanism.** `collect.py` keeps a process from an earlier attempt only whole: its `done.json`
  must exist, and its shard count and `games.jsonl` sha256 must match. It never resumes a process
  part-way.
- **The pilot's first attempt.** It started all three processes at once (`started_at`
  1795261292, 06:41:32). All three wrote `done.json` by 08:13–08:14 (2,667 + 2,667 + 2,666 games)
  and merged `games.jsonl` at 08:14:42.
- **The failure was in the seal.** The seal's statistics were then OOM-killed at 08:26:46 (exit
  137, anon-rss 83.7 GB; fixed by #1470).
- **The reseal played nothing.** It kept all three processes (hence `procs_resumed [0, 1, 2]`,
  `games_per_hour` None).
- **Byte-level proof.** The final `games.jsonl` is byte-identical to the one backed up before the
  reseal (`collect_backup_20261121_0827/games.jsonl`, sha256 `0ac540bf…`). Each process's
  `games_sha256` in the final stats equals its pre-reseal `done.json`. There are 8,000 distinct
  game ids, and no process was replayed.
- **What the gate does here.** Under this record the gate still **fails** on any kept process: a
  generation-2 collect must be one uninterrupted attempt. A seal-only failure therefore costs a
  replay. The seal-OOM cause is fixed at the PIN.

**Gate tests** (`prelaunch/collect_gate_test/`):

| Input | Expected | Result |
|---|---|---|
| the pilot collect | FAIL | **FAIL** on `procs_resumed [0, 1, 2]`, for the reason above |
| fabricated clean copy (`procs_resumed []`) | pass | **pass** |
| fabricated: `procs_resumed` key missing | FAIL | **FAIL** (KeyError) |
| fabricated: `unresolved_games` 1 | FAIL | **FAIL** |
| fabricated: a repeated `game_id` in `games.jsonl` | FAIL | **FAIL** |
| mutant: the gate reads the key with `.get(…, [])` | — | passes the missing-key case, so the case kills it |

**B3c. Comparison with generation 1's collection** (the readout for the confounds in B7). The same
script writes `collect_gate.json` with both collections' games, rows, rows per game, draw and
decisive rate, mean total plies, the **draw share of the outcome target** (rows of drawn games /
all rows) and buffer-full flushes. Revision 1 read a `plies` field that does not exist; it is now
`total_plies`. Generation 1 reads:

| Measure | Generation 1 |
|---|---|
| games | 8,000 |
| rows | 761,979 (95.2 per game) |
| draw rate | 0.49 |
| mean total plies | 111.6 |
| draw share of the outcome target | 0.500 |
| buffer-full flushes | 0 |

**B3d. Take effect.** `take_effect_collect.py` proves that the collect searched with the baked net,
not the raw one, from the consumer's own record: the collector writes the WDL of the net it searched
with as `root_net_wdl` on each row.
- On every hold-out row it compares the stored value with a fresh forward of (a) the baked file and
  (b) raw Zq.
- **Pass iff** mean |stored − baked| ≤ 0.004 and mean |stored − raw| ≥ 0.02.
- Scale: the pilot's stored-vs-fresh-forward noise for the SL base was 0.00085 per entry
  (`holdout_analysis.json`), and the recalibration moves Zq by 0.0375.
- **Self-test on the pilot collect, CPU, 2,048 hold-out rows:**
  - the true collecting net (SL base) as "baked" and Zq as "raw": **pass** (stored − SL base 0.0011,
    stored − Zq 0.0452);
  - swapped, the mutant: **fail**, exit 1 (0.0452 / 0.0011) (`take_effect_test/`).

### B4. Train generation 2

**From generation-1 weights, with the same Zq flags.**
- **Lineage:** `--init-checkpoint gen1/gen1_baked.pt`, in a fresh single-generation run dir
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

**Training starts from the BAKED net, with no compensation: this is the multi-generation rule
(revision 3).** Revision 2 kept the baked start because the PIN cannot train from raw weights on a
collect played by the baked net (`reference.py::reference_collect` requires the arm's generation-0
sha to equal the source's and the collect to be played by the arm's own generation g−1 model; a raw
start needs code). It sized the bake's effect on the trunk as 1/T1 = **× 1.421**, "≈ `--w-wdl`
1.42". **That figure was measured against raw Zq, but generation 1 did not start from raw Zq.** The
reviewer measured the step-0 trunk gradients directly, with the trainer's own batch construction
and `compute_loss` at the PIN, on the same pilot rows (2 × 256 rows, CPU fp32, generation-1
targets; `plan-review-1491/delta-730e07509/trunk/`, script `trunk_pull.py` sha256 `39173757…`):

| Start | Trunk ‖g_value‖ / ‖g_policy‖ at step 0 (per batch) |
|---|---|
| SL base (generation 1's actual start) | **0.998** (1.28, 0.70) |
| baked Zq (generation 2's start, as registered) | **0.894** (0.97, 0.83) |
| raw Zq (a raw start) | 0.594 (0.54, 0.65) |
| baked Zq with `--w-wdl` = T1 (a compensated start) | ≈ 0.63 |

I re-ran the reviewer's script for the baked file (banked copy, collect made a parameter,
default unchanged): trunk value 1.6481881737709045 / policy 1.8444833159446716 = **0.894**,
bit-identical to the reviewer's (`prelaunch/trunk_check/trunk_repro.json`; 3:13 wall, 14.0 GB
peak RSS).

- **Why the baked start is base-like.** Zq's training barely moved the final layer (‖W_out‖ 0.9311
  for the base, 0.9297 for raw Zq) and flattened the value through the body. Baking (T1, b1) into
  the final layer restores a base-like output shape, and with it a base-like pull on the trunk. A
  raw start would pull the trunk less (0.59) than generation 1 did; compensating the baked start
  by `--w-wdl` = T1 would do the same (≈ 0.63). **Neither is adopted: the coordinator's decision
  (revision 3), and the reviewer withdrew the raw-start preference.**
- **The final layer itself** (`value_wdl.net.2`, AdamW group 2, weight decay 1e-4): after the bake
  its weights are larger (‖W‖ 1.32 vs 0.93) while Adam's step size is unchanged, so its relative
  step per update is **≈ T1 ≈ 0.70×** generation 1's. Generation 1 moved ‖W‖ by only ≈ 0.001, so
  this changes little. Decoupled weight decay is multiplicative, so its relative effect is
  unchanged. Fresh moments (below) mean no stale second-moment estimate meets the rescaled layer.
- **The rule for generation k ≥ 2:** collect, train and play from the BAKED generation k−1 net; fit
  (T_k, b_k) of the trained generation k toward that baked net, on generation k's own hold-out
  (B5); bake (T_k, b_k) into generation k. Because each generation starts already recalibrated,
  (T_k, b_k) measures the drift of ONE generation of training, not the cumulative drift from the
  base; that is why the B5 bounds are centred on one generation's drift, not on T1.
- **Re-measured on generation 2's own rows, before training (step `trunk2`, CPU, report-only).**
  The same script (`prelaunch/trunk_pull_gen2.py`: the reviewer's file with one change, the collect
  is `TRUNK_COLLECT`; sha256 below) runs on `gen1_baked.pt` over **8 × 256 rows** (revision 4) of
  generation 2's own collect. The statistic is the reviewer's: mean value norm / mean policy norm
  over the batches. `trunk2_verdict.json` also carries the 8 per-batch ratios, their mean and its
  SE. **If the statistic leaves [0.59, 1.35]**, the launcher writes the registered note
  `gen2/HOLD_TRUNK2` and **stops for the owner before training**.
  - **What the range is: a sanity stop, not a test.** It is mean-based and symmetric in log space
    around the baked start's pilot-row mean 0.894: [0.894 / 1.505, 0.894 × 1.505], where 1.505 =
    0.894 / 0.594 is the factor between the baked and the raw start's pilot-row means. Revision 3's
    upper edge (1.28) was generation 1's largest single batch, a per-batch extreme set against a
    mean; it is replaced. Leaving the range says the trunk pull moved by more than the whole
    raw-vs-baked difference; it holds the run for the owner and licenses nothing.
  - **Deterministic on the same inputs (revision 5, review item 8), so a relaunch cannot clear the
    hold by re-rolling.** The batches are a fixed permutation (seed 20261122) of the hold-out-free
    rows, each batch's sampler is seeded by its index, `mirror_prob` is 0, and it runs CPU fp32 at a
    fixed 4 threads with no optimizer step. Observed: my rerun of the reviewer's script was
    bit-identical to the reviewer's (`trunk_repro.json`), and the first two batches of the 8-batch
    run equal the reviewer's 2-batch run. Its inputs cannot be re-drawn by `move_aside.sh`: the
    collect is a completed step (refused) and `gen1_baked.pt` is sha-checked. Moving
    `HOLD_TRUNK2` aside and relaunching therefore reproduces the same ratio and the same hold.
  - **Re-measured with 8 batches on the pilot rows** (`prelaunch/trunk_check/trunk8_baked.json`,
    CPU fp32, 4 threads): baked Zq start **0.898** (mean of per-batch ratios 0.902, SE 0.042;
    batches 0.97, 0.825, 0.69, 1.045, 0.968, 1.007, 0.886, 0.825; the first two equal the
    reviewer's 2-batch run). **9:17 wall, 14.1 GB peak RSS.** Generation-2 targets differ from
    generation 1's, so the pilot-row figures are the reference, not a prediction.
- **The realised-optimizer equality check still holds:** the groups and hyper-parameters are
  unchanged by the bake.

**Other choices:**
- **Why fresh optimizer state.**
  1. Generation 1 trained from fresh moments, so fresh moments keep the training step identical
     and leave the collecting net as the main data-affecting change.
  2. Generation 1's end moments come from 1,412 updates on generation-1 data, and the bake
     rescaled the final layer.
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
    raw Zq, the net that measured −10.0, not the champion.

  Single runs keep the pilot bank frozen and put the bake explicitly in the chain. The cost is
  that lineage is carried by sha, so the launcher records every sha. Moving the bake into the loop
  so that `--generations N` works is the change to make before generation 3. It would change the
  code pin, so it is not made here.

### B5. Per-generation recalibration — the procedure, its limit, and its bounds

**The procedure**, for generation k ≥ 2, before any match:
- fit (T_k, b_k) with `scripts/fit_value_recal.py` at the PIN (sha256 `37e27420…`):
  `--net` = generation k's trained model, `--reference` = generation k−1's **baked** file,
  `--collect-dir` = generation k's own collect, `--holdout-frac 0.05`, `--device cuda`. The fit
  uses hold-out games only;
- bake (T_k, b_k) into generation k's model with `bake_recal.py`, and run `verify`;
- generation k plays only as its baked file.

**Fit control.** The same script on Zq vs the SL base on the pilot collect must reproduce the banked
(T, b) within 1e-3 on each parameter. GPU bf16 forwards are not bit-reproducible, so the check is
not exact equality. If it fails, the launcher stops before the collect. The reviewer's CPU rerun
of this control was stopped after 41 min (it projected about 2.5 h), so the 1e-3 tolerance has not
been checked independently. The control itself decides on the GPU.

**What this rule can and cannot learn (review item 7).**
- Generation k is fitted toward generation k−1's baked value, which was itself fitted toward the
  base. Transitively, the rule pins every generation's **global** temperature and draw bias to the
  SL base's, up to fit drift.
- The recipe can learn **ordering and per-position value**. It can **never learn a global
  sharpness or draw calibration different from the base's**.
- The base's value is known to be under-confident, so that under-confidence is **locked in**.
- That is acceptable for 1–2 measured steps, and it is how match 8's player was made. It is a
  **known limit to revisit once the loop climbs**, in a later record, not this one.

**Registered bounds on (T2, b2): a stop BEFORE the matches.**
- **The bounds: a registered SANITY stop, not a statistical bound.** With (T1, b1) = Zq's banked fit
  (0.7036, −0.2271), the launcher stops after the fit, before any match, unless:
  - ln T2 ∈ [2 ln T1, −ln T1] = **[−0.703, +0.352]**, i.e. T2 ∈ [T1², 1/T1] = **[0.4951, 1.4212]**
    (revision 2 printed the T bounds as the ln T bounds; the code was right);
  - b2 ∈ [2 b1, −b1] = **[−0.4542, +0.2271]**.
- **Derivation.** Generation 2's head STARTS recalibrated (B4), so (T2, b2) measures the drift of
  one more generation of the same training.
  - The pilot measured one such drift from the base: Zq (T 0.7036, b −0.227).
  - The bounds allow up to **twice** that drift in the same direction, or an equal drift in the
    opposite direction.
  - The pilot's other two fits sit inside: P0 (T 1.028, b +0.042), whose value head was untrained
    and drifted only through the trunk, and S (T 0.853, b −0.120).
- **What is expected.** If the coordinator's raw start were possible, T2 would absorb the cumulative
  drift (about T1 × the per-generation drift) and the bounds would have to be centred on T1. With
  the baked start, about one generation's drift is expected.
- **Caveat.** The generation-2 hold-out is generation-2 self-play, a different position
  distribution from the pilot hold-out that produced (T1, b1).
- **A stop is a FAIL of step `fit2`.** No match is played, and the owner decides.

### B6. Matches

Two registered matches, at the PIN, with match 5's rules.
- **Seats:** plain seats, both files baked; no `--*-value-recal` flags.
- **Launcher and smoke:** the follow-up launcher (sha256 `7c8fd065…`), 6,400 games each (3,200
  per seat), a 4-game-per-seat smoke first.
- **Readback, after both smoke and real runs:** every result row's `git_sha` must equal the PIN,
  and `candidate_sha256` / `reference_sha256` must equal the intended files in each seat.

| Match | Seats | Seed | Claim |
|---|---|---|---|
| **gen 2 vs gen 1** (primary; reading 11) | `gen2_baked.pt` vs `gen1_baked.pt` | 2026112283 | **the step-2 claim**: "improves" iff LB95 > 0 |
| gen 2 vs SL base (reading 12) | `gen2_baked.pt` vs `sl_base.pt` | 2026112284 | cumulative, fresh and preregistered: (iii) |

Readings for both: damage iff UB95 < 0; adoptable iff LB95 > −10; **improves** iff LB95 > 0;
otherwise unread. Match 5's tie-break is kept.

**Power** (SE ≈ 3.24):
- **Gen 2 vs gen 1:** a second step as large as match 8's (+9.6) reads "improves" with
  probability 0.84; a +5 step with 0.34.
- **Gen 2 vs base:** if the steps add (about +19), LB95 > 0 with probability ≈ 1.00. Elo is not
  transitive across opponent pairs, so this is a planning figure.

| Outcome | Licenses |
|---|---|
| gen 2 vs gen 1 **improves** | **one measured step (step 2)**. It becomes **two measured steps** only with Stage 0 "replicates" (table at the top). Generation 3 under a new record, after the bake is moved into the loop |
| gen 2 vs gen 1 adoptable, not improves | generation 2 is safe to collect from, but no second step is measured; the owner decides between generation 3 and option (ii) |
| gen 2 vs gen 1 damage | the recipe does not repeat from its own output; stop; the owner decides |
| gen 2 vs base improves | a **cumulative improvement over the base**; attribution between generations unread unless Stage 0 replicated and gen 2 vs gen 1 improved |

**Recipe frozen.** Within this record the recipe is exactly B1–B5. A competing nomination, for
example from the value-anchor screen or the sibling screen, does not change it mid-record. **Three
nominations now exist, each from one generation:** match 8's recalibrated Zq, +9.61 [+3.27,
+15.96], the only one whose LB95 clears 0; the anchor screen's S + `--kl-wdl` 7.0, +5.81 [−0.53,
+12.16]; and its Zq + `--kl-wdl` 11.3, +3.58 [−2.62, +9.79] (nominated only together with a
"Zq + (T, b)" alternative). They are **never ranked by point estimate** (their intervals overlap,
and each is the winner of its own screen); this record's recipe stays frozen. A competing
nomination is settled by a head-to-head match against this record's champion, under the next
record.

### B7. Seeds and what differs from generation 1

**New seeds.** Each was checked unused in the repo at `origin/main` (`git grep`), in every open
PR's diff, in the rl-run1-pilot / followup / value-anchor / sibling / root-full-width /
rl-claude-20261119 banks' scripts, records and JSON, and in every banked `results.jsonl` `seed`
field. Each appears only in this record's own bank; the reviewer's independent grep agrees.

| Seed | Use |
|---|---|
| 2026112281 | Stage 0 replication match |
| 2026112282 | generation-2 collect; also the training's batch plan and mirror draws (the same `--seed`) |
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
| training batch plan and mirror draws | from seed 2026112100 | from 2026112282 (the same `--seed` draws them) | yes, a re-roll (unavoidable: `--collect-from` requires the source's seed) |
| draw rate, game length, draw share of the value target | 0.49 / 111.6 plies / 0.500 | measured in `collect_gate.json` | yes, follow from the collecting net; reported, not controlled |
| code at collect | C0 `6c72b821b` | PIN `0c795fb7a` | no (play path unchanged; see B) |
| training init | SL base | `gen1_baked.pt` | yes, part of "a generation" |
| **value pull on the trunk at step 0** (trunk ‖g_value‖ / ‖g_policy‖, same pilot rows, 2 × 256, generation-1 targets) | **1.00** (SL base start) | **0.89** (baked start); a raw start would be 0.59; final layer's relative step × 0.70 (‖W‖ 1.32 vs 0.93; generation 1 moved it by ≈ 0.001) | **a named confound, measured** (B4); re-measured on generation 2's own rows before training (`trunk2`) |
| init checkpoint's `grad_clip_mode` field | none | `zclip` | no: the realised optimizer must equal generation 1's (checked) |
| recalibration reference | SL base (match 8's fit) | `gen1_baked.pt` | the registered per-generation rule |
| training data | generation-1 collect | generation-2 collect | follows from the collecting net |
| every training flag | as B4 | identical | — |
| match settings | match 8's | identical, plain seats with baked files | no: within inference noise (A, B3a) |

**Confounds.**
- The collecting net and the training init change together. That is the definition of a
  generation, not two changes inside one readout.
- The re-rolled collect seed, batch plan and mirror draws are unavoidable.
- **The start's value pull on the trunk** (revision 3, measured, replacing revision 2's "× 1.421 (≈
  `--w-wdl` 1.42)"): trunk value / policy gradient ratio at step 0 on the same pilot rows is **1.00
  for generation 1's start and 0.89 for generation 2's baked start**; a raw start would be 0.59
  (2 × 256 rows, generation-1 targets). The baked start is the closer of the two to generation 1's
  pull, so it is kept with no compensation as the rule for every generation (B4). The final layer's
  relative step is × 0.70 (‖W‖ 1.32 vs 0.93; Δ‖W‖ ≈ 0.001 over generation 1). The residual
  difference cannot be separated from a generation-2 result; generation 2's own ratio is reported
  before training (`trunk2`).
- Concurrent screens (the sibling and root-full-width screens) use separate banks and nets; the
  guard keeps the card exclusive to one stage at a time, so they share no data or GPU time with
  this run.

## (C) Options — costed, not registered

**C1. Generation 2 from P0.** P0 trained the policy only (`--w-wdl 0`) and measured +2.2
[−4.2, +8.6] as a single net vs base, so it is unread, with no evidence that it is stronger.
Collecting from P0 would build generation 2 on a net that has not measured above the base, and
the step-2 claim would then be relative to an unconfirmed step 1. Cost is the same as B (≈ 4.9
GPU-h). **Not recommended.**

**C2. A P0-policy + recalibrated-Zq-value graft.** P0 and Zq are separate fine-tunes of the whole
net, and both heads read their own trunk. So no single checkpoint carries P0's policy and Zq's
value: the graft is a hybrid two-forward seat (`--candidate-value-from` + `--candidate-value-recal`).
- A collector cannot play a two-net seat, so the graft could only be a match diagnostic, never a
  generation-2 collector without new code.
- A graft vs recal-Zq match would cost about 2.2 GPU-h for 6,400 games (hybrid seats are measured
  at 0.27 GPU-h per 800 games), and it answers only whether P0's policy adds to Zq's.
- It decides nothing in this record's chain, so **it is not proposed as a match.** It becomes worth
  running only if generation 2 reads unread vs generation 1 and the owner wants to know whether the
  policy target or the value target carries the gain.

**C3. Option (ii), a fresh generation-1 replicate** (part A): ≈ 3.4 GPU-h, seeds 2026112285 /
2026112286. It confirms the recipe with no post hoc fit. Recommended only if Stage 0 reads unread
and the owner wants step 1 confirmed before generation 2.

## (D) Cost and queue

All figures are measured on this machine. No GPU work was run for this record.

| Item | GPU-h | Source |
|---|---|---|
| Stage 0: bake + verify (CPU, ≈ 6 min, the GPU held idle), then 6,400 games + smoke | ≈ 1.57 | 1.45 per 6,400 recal games (match 8); `bake_test/verify.log` 5:57 wall, 3.6 GB RSS |
| equivalence check (two forwards, 2,048 rows, cuda) | ≈ 0.02 | |
| fit control + fit (T2, b2) | ≈ 0.1 | fit_value_recal on 39k rows, two forwards, cuda |
| collect, 8,000 games | 1.54 | the pilot's play time, slowest process 5,534 s |
| collect gate + take effect | ≈ 0.05 | two cuda forwards over about 38k hold-out rows |
| `trunk2`: trunk gradient ratio on generation 2's rows, 8 × 256 (CPU, 4 threads, GPU idle) | 0 (≈ 0.16 h wall) | 8 batches on the pilot rows: 9:17 wall, 14.1 GB peak RSS |
| train Zq2 | 0.27 | the anchor screen's first cell (16 min); the pilot's Zq was 859 s |
| bake + verify generation 2 (CPU, GPU held idle) | ≈ 0.1 | as Stage 0's |
| two matches (12,800 games + smokes) | ≈ 2.95 | 1.45 per 6,400 |
| **total** | **≈ 6.6** (Stage 0 ≈ 1.6, generation 2 ≈ 5.0) | |

**Per-step RAM floors.** The launcher refuses a step (exit 89, before its `.started`) unless
`MemAvailable` meets the floor:

| Step | Floor | Basis |
|---|---|---|
| collect, train | 40 GiB | the pilot's training loader holds 20–23 GiB at ≈ 760k rows (launch record) |
| `trunk2` | 20 GiB | 8 batches peaked at 14.1 GB RSS |
| equivalence, fits, checks, bakes | 12 GiB | bake verify peaked at 3.6 GB RSS |
| matches, Stage 0 | 16 GiB | |

**Queue.** The anchor screen runs until about 05:00, then:
- the sibling screen, ≤ about 9 GPU-h;
- the full-width stages 0–2, about 5–6 GPU-h.

**Recommendation for the fastest second measured step:** run this record **directly after the
anchor screen**, before the sibling and full-width screens. The order is Stage 0 (≈ 1.6 h), then
generation 2 (≈ 5.0 h), which puts the step-2 reading about 6.6 GPU-h after the anchor screen ends
instead of about 21 GPU-h.
- **Neither screen is a prerequisite.** Each tests a recipe change that, by the frozen-recipe
  rule, would be settled against this record's champion in the next record anyway.
- **The anchor screen's cost.** If its Zq cells read well, they become a competing nomination,
  settled by head-to-head under the next record. Waiting for it gains nothing for this record.
- **The order is the operator's to set,** through `GPU_RELEASED_BY_OPERATOR` (below). This record
  only recommends it.

## Guards, markers and resume

- **GPU memory, training: the 3-update smoke, after Stage 0's verdict and before the generation-2
  release** (revision 5; revision 4 put it before Stage 0, where its start net does not exist yet).
  Run by the operator, alone on the card, as ONE command: `run_smoke.sh IDLE_MIB` (Commands, step 7).
  - **The runner** (`prelaunch/train_mem_smoke/run_smoke.sh`) first runs the operator's GPU gate
    itself: the sibling bank's `operator_gpu_check.sh IDLE_MIB` (read-only, sha256-pinned:
    memory within IDLE_MIB ± 512 MiB, utilisation < 10 % over 5 samples, the shared `gpu_scan.sh`).
    It then refuses (nothing runs) on: the checks library, the flags file, the smoke script or the
    gate not their pinned sha256 (93); `CUDA_VISIBLE_DEVICES` set to empty (94); Stage 0 with no
    verdict or with damage (91); `gen1_baked.pt` not the verified file Stage 0 played (92, the
    launcher's own check); the PIN worktree not at the PIN or dirty (96 / 97); a launch of this bank
    holding its lock (9; the smoke then holds the lock for its whole run). It passes INIT =
    `gen1_baked.pt` and the flags `$COLLECT2 $TRAIN $ZQ` from `gen2/gen2_flags.sh`, the one file
    `launch_gen2.sh` also sources (pinned in both): no string is copied or retyped.
  - What it runs: `train_mem_smoke.py` from the PIN worktree, with generation 2's training flags.
    It uses the PIN's own `load_window` over the pilot collect's whole window (the same size as
    generation 2's, 8,000 games), `prepare_targets`, the first 3 batches of the real
    `plan_batches` plan, `build_trainer` with `gen1_baked.pt`'s weights (generation 2's actual start),
    then `Trainer.train_steps` one update at a time.
  - What it records: after EACH update, one JSON line in `smoke_mem.jsonl` (flushed and fsynced):
    `max_memory_allocated` and `max_memory_reserved` since the start, `memory_reserved` now, the
    device total and the update's wall. Then `smoke_mem.json` holds the verdict and (revision 5) the
    update 2 → 3 changes of reserved-now and peak reserved.
  - **Abort rule (registered, unchanged): ABORT iff peak reserved > 0.85 × the device total** (exit
    3). An update that dies, CUDA OOM included, and any refusal or other non-zero exit are an ABORT
    too. On ABORT the operator writes `gen2/HOLD_MEM`, nothing launches, and the owner decides.
  - **What the operator confirms for PASS** (a check, not a rule change; revision 5): exit 0,
    `"verdict": "PASS"` in `smoke_mem.json`, and reserved memory flat between updates 2 and 3
    (`reserved_now_delta_2_to_3_gib` and `peak_reserved_delta_2_to_3_gib` ≈ 0: a step that still
    grows after update 2 is not a steady state). Anything else is `HOLD_MEM`.
  - Expectation, NOT a substitute for the measurement: generation 2 adds no second
    gradient-carrying forward, so about 17 GiB reserved. The sibling diagnostic's no-term cell, at
    the same flags from the SL base, peaked at 16.96 GiB reserved / 15.38 GiB allocated of 31.82 GiB
    over 6 updates (`~/chess-artifacts/operations/rl-run1-sibling/oprun/diag_mem/RESULT.md` [`0468d345`],
    with `mem_rows0.json` beside it). Update 1 takes 5–7 s (kernel warm-up), so the smoke's walls are not a step-time
    estimate.
  - **What the smoke does not cover, and the margin that does.** Three later allocations in the
    training stage never run in the smoke: the hold-out metrics (eval-mode forwards under
    `no_grad`, smaller than a training step), `probe_drift`'s second copy of the net (≈ 0.25 GiB of
    fp32 weights for 61 M parameters) and `torch.save` (serialisation; no new activations). Against
    an expected ≈ 17 GiB peak, the abort line (27.05 GiB) leaves ≈ 10 GiB and the card (31.82 GiB)
    ≈ 15 GiB; each of the three is well inside that.
  - Tested on CPU at the PIN on a tiny fixture (`tms_test.sh`, 7 / 7): three per-update lines; PASS
    at 27.00 of 31.82 GiB (0.85 × total = 27.047); ABORT (exit 3) at 27.10; refuses to overwrite;
    refuses a CPU run without the logic-smoke flag. The verdict is exercised on CPU through an
    injected (peak, total) pair, accepted only in CPU logic-smoke mode. Three mutants, each killed
    (`mutants/mutant_results.txt`): the abort fraction at 0.95; the abort ignored; the per-update
    line not written. Not covered by a test: a write deferred to the end of the run (the lines
    exist either way); the code writes and fsyncs each line inside the update loop. In CPU
    logic mode (revision 5) the exact flags are parsed first, then the device is forced to cpu
    without autocast, so the operator's own command line (`--device cuda` included) can be run on
    the CPU.
  - **The exact operator command, run once in CPU logic mode** (revision 5, review item 1):
    `run_smoke.sh 1138` with `GEN2_BANK` a scratch copy of the bank files, the banked baked Zq
    (`prelaunch/bake_test/Zq_baked.pt`, the bake of the same file with the same (T, b)) as the
    stand-in `gen1_baked.pt`, a stand-in Stage 0 verdict naming its sha, and `TRAIN_MEM_SMOKE_RUN`
    one pilot shard (18,840 rows, 37 batches of 512). The full `$COLLECT2 $TRAIN $ZQ` parsed,
    including `--optimizer-hparams-source init-checkpoint` reading INIT; 3 updates ran; exit 0,
    3 per-update lines, verdict null (CPU). **Footprint: 24.0 GB peak RSS, 3:55 wall** (4 threads,
    nice 15), mostly the 61 M-parameter net at batch 512 on CPU. Banked:
    `prelaunch/train_mem_smoke/cpu_logic_run/`.
  - Harness (v7): the runner's refusals, the gate (a busy GPU, IDLE + 513 MiB, a listed GPU job, a
    tampered gate), INIT and the output stem, and **the smoke and the training step receiving
    IDENTICAL flag strings** (both calls logged by the stand-in and compared).
  - Not enforced by the launchers: they are pinned and do not read `smoke_mem.json`. The smoke
    is an operator step, and `gen2/HOLD_MEM` is a registered note.
- **GPU memory, collect.** Generation 2's collect uses the pilot's flags (`--collect-procs 3
  --concurrency 128 --sims 64`). The pilot's banked collect recorded its own pre-launch VRAM check
  (`~/chess-artifacts/operations/rl-run1-pilot/collect/gen_0001/collect/stats.json` [`48e8239a`],
  `gpu_memory_check`):
  - modelled peak need 24.55 GiB for 3 processes (`collect.vram_need_gib`: a fitted peak × a safety
    factor, about 8.2 GiB per process);
  - usable 26.43 GiB = min(free 29.71, 0.83 × total 31.84);
  - the collect then ran all 8,000 games with no OOM.

  That is a model, not a measured peak. The same check runs at the start of generation 2's collect
  and refuses before any work if the need exceeds what is usable then. Expected: about 24.6 GiB
  at most. The card must be otherwise idle, which the guard checks.

  **Measured at the same shape** (revision 5, review item 6):
  - the collect-throughput sweep's `mp3_c128_s64_prof` cell (`~/chess-artifacts/operations/
    rl-claude-20261119/collect-throughput/logs/sweep.txt` line 37: 3 processes, concurrency 128, 64
    sims; manifest matching the pilot's on every setting, checked by the reviewer): **22,104 MiB
    whole-card peak** `memory.used`;
  - the operator's diagnostic today at the same shape: **20,566 MiB** peak `memory.used` (1 s
    samples) at an idle of 1,138 MiB
    (`~/chess-artifacts/operations/rl-run1-root-full-width.opdiag/mem_20261122/RESULT.md` [`0a062442`]).

  Both are whole-card readings (idle included), below the model's 24.55 GiB and the card's 31.8.

- **Pre-launch checklist** (revision 5). Before any operator step:
  - **every operator command in Commands was executed once in CPU / stub mode from the staged bank,
    output banked.** Revision 5: Stage 0, generation 2 and `move_aside.sh` by the harness (stub
    mode, every refusal and path); `run_smoke.sh` by the harness and once for real in CPU logic
    mode (above; 24.0 GB RSS, ≈ 4 min: run it beside nothing else that needs the RAM). This is the
    second time a first GPU step failed on something a dry run of the exact command would have
    caught;
  - the live bank verifies against `BANK_SHA256SUMS` after the swap;
  - the queue markers `rl-run1-followup/chain.exit` and `rl-run1-value-anchor/queue2.exit` exist.
- **Code pin.** Every GPU step runs in the PIN worktree, checked at HEAD = PIN with no tracked
  changes. `fit_value_recal.py` is sha-checked at the PIN, and every match result row must carry
  `git_sha` = PIN.
- **Matches at the PIN, with seat readback.** All seats are plain (Stage 0 included, since
  revision 2). They are proven by the result rows' `candidate_sha256` / `reference_sha256` against
  the intended files' sha256, smoke and real.
- **The GPU guard** (`gpu_guard.sh` **v3**, revision 4) refuses (exit 8) unless all of these hold.
  **Why v3: v2's memory bar could never pass on this machine.** v2 required every memory line
  below 1,024 MiB, but this machine's idle baseline is **3,382 MiB** (the Windows side of WSL2;
  the coordinator's 30 one-second samples with all queues finished: min = max = 3,382 MiB,
  utilisation mean 0.0 %, max 1 %; 3,430 / 3,452 MiB between anchor cells). Stage 0 could never
  have launched; v2 had only ever run against stubs. v3 uses the same rule as the full-width
  bank's guard:
  - the operator's release line for the stage carries the **measured idle baseline** as a token
    `idle_mib=<integer>` (for example `stage0 idle_mib=3382 2026-11-22T09:00`); missing,
    non-integer or **> 4,096** (a sanity cap) refuses;
  - **every** `nvidia-smi --query-gpu=memory.used` line is an integer within **[idle_mib − 512,
    idle_mib + 512] MiB**, and there is at least one line (below the band means the baseline is
    stale);
  - **utilisation**: `nvidia-smi --query-gpu=utilization.gpu`, 5 samples one second apart, every
    line an integer, maximum **< 10 %** (idle is 0–1 %; when the owner is gaming, the GPU is not
    ours);
  - every probe fails closed (non-zero exit, no line, a non-integer);
  - **no GPU job of this user, by the shared process scan `gpu_scan.sh` v4** (one file,
    byte-identical in the full-width, generation-2 and sibling banks, declared final by its owner at
    sha256 `949408e727e7ccbf942dd28f58ad800f958267577a93822950529ac20734f466`, pinned in the guard;
    the #1495 review; the file's header is its full specification). For each same-uid process
    outside the caller's own ancestry and descendants it refuses on any of:
    - **exact argv** (`/proc/<pid>/cmdline`) naming a listed GPU entry point: python's first
      non-option argument (`-X` / `-W` values skipped; `-c` ends the scan) or
      `-m chess_anti_engine…`, a shell's script argument (`bash -c` ends the scan), or the lc0 /
      Ceres binary itself. A shell or `python -c` that merely mentions a name is not a job;
    - **the GPU device, identified by rdev**: every fd is `stat -L`'d, and one that is a character
      device with the major:minor of this machine's `/dev/dxg` refuses, whatever path opened it. A
      file merely named `/dev/dxg` (another agent's namespaced test harness) is a NOTE, not a
      holder; if the scanner's own `/dev/dxg` is not a character device the scan refuses;
    - **fds that cannot actually be read** (listed and stat'ed, never judged by `access(2)`), except
      a process whose argv[0] is exactly `/usr/bin/ssh-agent` or `/usr/bin/gpg-agent` (both
      non-dumpable by design): skipped for the fd clauses only, with a NOTE per pid.

    A launcher's re-guard does not trip on itself, while a **sibling copy** of the same launcher
    still refuses. It replaces v2's `ps -o args` token scan, which passed GPU jobs outside its three
    names and refused agent shells that merely mentioned one. The list of entry points is derived,
    and the derivation is written in the file's header: every `scripts/*.py` at main matching
    `cuda|--device`, every script in the RL banks that uses cuda or starts one of those programs,
    the launchers they start, and the lc0 / Ceres binaries. Measured cost 0.26–0.75 s per scan.
  - **CPU jobs are exempt by their environment, GPU jobs never.** A listed process is exempt from
    the argv clause (never from the device or unreadable-fd clauses) iff its exec environment
    (`/proc/<pid>/environ`) is readable and holds exactly `CUDA_VISIBLE_DEVICES=` with an empty value
    and no other entry for that variable. That is how agents' CPU test children
    (`python -m chess_anti_engine.uci … --device cpu`) stop blocking launches. The stated limit: the
    environment is the one at exec, so a process that sets the variable itself before importing
    torch is caught only by the device clause, once its CUDA context exists. **So the operator's and
    every launcher's GPU jobs must be started WITHOUT `CUDA_VISIBLE_DEVICES` set to empty.** Both
    launchers assert `[ "${CUDA_VISIBLE_DEVICES-unset}" != "" ]` among their refusals, before any
    GPU step (exit 94), so a job cannot silently fall back to CPU or crash mid-stage;
  - **Memory and utilisation cannot see a job that is still loading.** Measured: the value-anchor
    reproduction's Zq λ 11.3 cell read idle (3,448–3,463 MiB, utilisation 0–1 %) from its start at
    04:54:28 until 05:00:09, about six minutes; the first busy sample is 05:00:15 (5,488 MiB, 81 %),
    then 20–22 GB (`~/chess-artifacts/operations/rl-run1-value-anchor-repro/arm_Zq_k11.3.gpu.txt`
    [`1bfc3754`]). So **the process scan and the operator's release file are the primary defences,
    and the memory band is the backstop**;
  - both queue markers exist (`rl-run1-followup/chain.exit`, `rl-run1-value-anchor/queue2.exit`);
  - the operator's `GPU_RELEASED_BY_OPERATOR` in this bank has a line whose first word is the
    stage (`stage0` or `gen2`), with its `idle_mib` (above);
  - and it then takes a bank-local `flock` (exit 9 if held).
  - **Re-guard (revision 4).** The guard defines `gen2_gpu_idle WAIT_S`. The generation-2
    launcher calls it before every step that runs, **before** that step's marker (a refusal is
    exit 8 with no step marker: relaunch when the GPU is idle), and `play_seats` calls it before
    every arena seat, in both launchers (a refusal fails that match step or Stage 0, which then
    resume per seat). The bound is `GEN2_REGUARD_WAIT_S`, **default 900 s**, which must be a
    non-negative integer (else the first guard refuses). It matches the full-width bank's guard. The
    holder found on 2026-11-22 was `torch.cuda.is_available()` in an agent's CPU test: under WSL2
    it opens `/dev/dxg` and holds it for the process's lifetime even with
    `CUDA_VISIBLE_DEVICES=` empty, so a holder can last minutes. **It retries the whole check, not
    only memory:**
    - the operator's release line is re-read, and memory band, utilisation and the process scan
      are re-run, every 5 s until the bound has passed (waits counted, and wall time);
    - each distinct refusal reason is logged once, with its timestamp, to the stage's `guard.log`
      (`stage0/guard.log` or `gen2/guard.log`), and so is the outcome with its duration;
    - only the **last** attempt decides, and a refusal names that last reason;
    - **a release-line refusal (revoked, missing, no `idle_mib`) is the operator's intent:
      refused at once, never retried.**

    **Not covered by the scan:** other-uid processes are not scanned (the memory band is the
    backstop for them); and before CUDA initialises, a process that sets `CUDA_VISIBLE_DEVICES`
    itself after exec is covered ONLY by the operator's release decision. **Do not run guard
    harnesses while a real stage relies on a re-guard:** their fake jobs are visible from the host
    through the child pid namespace and would make the re-guard wait or refuse.

    **What a wait costs:** a step (or seat) delayed by up to the bound, logged with its reason and
    duration in `<stage>/guard.log`. An agent's test touching the device during a stage delays that
    step; it does not abort a multi-hour stage. **The FIRST guard of a stage, before anything is
    spent, is single-shot**: one attempt, refused with its reason.
  - **Checked against the real card (read-only).** The harness runs the guard's two probe
    commands, extracted from the guard file, against the real `nvidia-smi` (`CUDA_VISIBLE_DEVICES=`
    does not affect it) and asserts integers: 3,382 MiB and 1 % at this revision. A direct source of
    v3 against the real card with `idle_mib=3382` passed the memory and utilisation checks. The
    v2 `ps` scan then refused on another session's shell line that merely named a script; that
    false refusal is what the exact-argv scan removes. The harness also runs the real
    `gpu_scan.sh` read-only and prints its verdict. On 2026-11-22 that verdict was rc 0 with NOTEs
    only (`/usr/bin/ssh-agent` allowlisted; a fake-HOME launcher copy exempt by its empty
    `CUDA_VISIBLE_DEVICES`) on some runs, and rc 8 on others: an agent's CPU test process held the
    real device (rdev `a:102`) despite an empty `CUDA_VISIBLE_DEVICES`. The measured cause: under
    WSL2, `torch.cuda.is_available()` opens `/dev/dxg` and holds it for the process's lifetime even
    with the variable empty; `PYTORCH_NVML_BASED_CUDA_CHECK=1` avoids it, and a conftest fix is in a
    PR. It is why the re-guard retries. The verdict in the banked
    `guard_test/guard_test.out` is the one at banking time.
- **Launcher refusals** (`set -euo pipefail`; none of them writes a marker):
  1. OWNER_GO (exit 90; revision 4: the coordinator writes it quoting the owner's standing
     statement, Status). (Revision 4: `CUDA_VISIBLE_DEVICES` set to empty refuses, exit 94.)
  2. The guard FIRST: it must exist (8), match the sha256 pinned in the launcher (93; review
     item 5), and source cleanly (`|| exit 8`). Then (generation 2) the checks library
     `gen2/gen2_checks.sh`, pinned by sha256 in both the launcher and `move_aside.sh` (93).
  3. Stage 0: no earlier attempt on disk (95).
  4. Generation 2: Stage 0 not "damage" (91), and `gen1_baked.pt` is the verified file Stage 0
     played (92).
  5. Launcher and script sha256s (93).
  6. The PIN worktree (96/97).
  7. Generation 2's resume prescan (below).
  8. The RAM floor per step (89).
- **Markers (review item 3).** After the refusals, an EXIT trap whose first statement saves `$?`
  writes `stage0/exit` or `gen2/exit` as "<code> <time>"; generation 2 also appends
  `gen2/exit.history`.
  - On INT/TERM the launcher first kills its own tracked child (each GPU step runs under `setsid`
    as its own process group, killed by pid, never by pattern), then exits 130/143. **Known gap:**
    a grandchild that calls `setsid` itself leaves that process group, so the group kill does not
    reach it (none of the called scripts does this today; it is not tested).
  - `verdict.json` / `verdicts.json` are written to a temp file and renamed.
- **Generation 2 resume (revision 3).** Each step writes `<step>.inputs`, then `<step>.started`,
  then `<step>.exit`, and has a completion check in `gen2/gen2_checks.sh` that verifies **data and
  inputs, not existence** (review item 4):
  - `collect`: `prelaunch/stage_checks.py collect` — the sealed manifest's actor
    (`inputs.checkpoint.sha256`) is the current `gen1_baked.pt`, its seed (`seeds.base` and
    `config.settings.seed`) is 2026112282, and its shard, games and starts digests re-verify now
    (shards by the loop's own `sha256_tree`, copied verbatim). Checked on the real pilot collect:
    passes with its actor and seed 2026112100; fails with another actor or seed
    (`prelaunch/stage_checks_test/`).
  - `arm_Zq2`: the trainer-summary check plus `stage_checks.py arm` — `model.pt`'s sha256 equals the
    arm manifest's `outputs.model`, the init checkpoint is `gen1_baked.pt`, and the reference
    collect's `source_collect_manifest` sha256 is generation 2's collect manifest now. Checked on
    the pilot's `arm_Zq`: passes.
  - `<step>.inputs` records the launcher's sha256, the checks library's, `prelaunch/SHA256SUMS`'s,
    the step's seed and the sha256 of every file the step reads. **On resume a done step is reused
    only if its recorded inputs equal the inputs now; a mismatch refuses (exit 84), never skips and
    never silently reruns.** Every run and skip line in `resume.log` carries the launcher's sha256.
  - A step whose `.exit` is 0 AND whose check passes AND whose inputs match is skipped and reused.
    A transient refusal after the 1.5 h collect does not cost the collect.
  - A step with a non-zero `.exit` (or a passing exit with a failing check) refuses with exit 88.
  - A step with `.started` but no `.exit` (interrupted) refuses with exit 87.
  - Both refusals name the one documented command, `gen2/move_aside.sh <step>`.
  - **Except the two match steps (revision 4): they resume per seat.** A failed or interrupted
    match does not refuse the relaunch (its recorded inputs must still equal the inputs now, else
    84). `play_seats` (in `gen2_checks.sh`, shared with Stage 0) judges each seat, smoke then real,
    with `prelaunch/seat_state.py` (the full-width bank's `seat_state.py` v2/v3 with one addition:
    the row's `git_sha` must be the PIN):
    - **DONE** (`.exit` 0; exactly one header, with the seed; exactly n games with unique
      (pair_id, half); no unparsable line; exactly one result row with the seed, the two nets'
      sha256 as they are now and the PIN): **never replayed**;
    - **ABSENT** (none of the seat's own files): played, then it must read DONE;
    - **PARTIAL** (anything else, including an exit-0 seat with too few games): the relaunch stops
      (1) and names `move_aside.sh <step> --seat <A|B>`;
    - **STALE** (complete, but the row names other nets or another commit: a sha readback failure):
      writes the registered note `HOLD_READBACK_<label>` and stops for the owner (86). It has no
      move_aside path.

    After both seats are DONE, readback and pool run again. A relaunch after a pool failure plays
    no game. Checked on match 8's real seats: both read DONE; the seats swapped, another commit
    and a wrong seed each read STALE or PARTIAL (`prelaunch/seat_state_test/`).
- **`move_aside.sh` (revision 3, review item 1): it moves only failed or interrupted work.** It
  renames to `*.aside.<UTC with nanoseconds>.<pid>`, refuses if any destination exists, and never
  deletes. **It refuses, with no override flag:**
  - an argument that is not exactly one plain step name from its list (no `/`, `.`, `..`; every
    source and destination must resolve inside the bank);
  - while a launch holds the bank lock, or while a later step has markers;
  - **a completed step** (`.exit` 0 and its completion check passes — the launcher's own checks,
    from the shared library);
  - **a match step, always, once it completed (`.exit` 0), has a real `pool.json`, or once
    `verdicts.json` exists**, and (revision 4) **once any of its real seats' `.exit` reads 0**;
  - **`stage0` once Stage 0 completed** (exit 0 or `verdict.json`), **whenever `gen2/` has any
    marker**, and (revision 4) **once any real Stage 0 seat's `.exit` reads 0**.
  - **`--seat A|B` (revision 4; the match steps and `stage0` only)** moves one interrupted seat's
    own files (`.games.jsonl`, `.pgn`, `.log`, `.exit`; smoke and real) and nothing else. It refuses
    a real seat whose `.exit` is 0 (a finished smoke seat is left in place), a seat with a result
    row (the row cannot move without rewriting `results.jsonl`: the owner's), a made reading, and
    when there is nothing to move.

  A registered reading is re-drawn only under a new record. A moved-aside seat's games are never
  read or pooled. **Stage 0 resumes per seat like the matches** (revision 4, replacing "replayed
  whole"): a completed Stage 0 (`verdict.json`) refuses the next launch (95); otherwise a DONE seat
  is kept, an ABSENT seat plays and a PARTIAL seat stops until `move_aside.sh stage0 --seat`.
- **Test.** `guard_test/guard_test.sh` **v7** (revision 5; v6 and v4 kept) runs everything under the isolation
  described next.
  - **The isolation.** A fake HOME holds only a fake pilot bank, a fake follow-up launcher and a
    throwaway git repo as the PIN worktree. Stubs on PATH replace `python`, `python3`,
    `nvidia-smi`, `ps`, (v4) `date` (a frozen clock only when a case sets it) and (v6) `sleep` (the
    guard's 1 s / 5 s pauses are skipped and counted). Inline checks run
    on the system python3, which is asserted torch-free; script calls run on a stand-in
    (`fakepy.py`) that imports nothing from the repo, except that it runs the REAL
    `prelaunch/stage_checks.py` and (v5) `prelaunch/seat_state.py` (pure stdlib) on manifests and
    seat logs written in the real shape. The v5 fake follow-up launcher refuses an existing game log
    as the real one does, writes a header, n games and a `.exit`, and logs every arena call. The
    launchers are copies whose pinned sha constants are rewritten to the fake files. The harness
    asserts the isolation before any case runs.
  - **Revision 5 (v7): 247 cases, 247 pass** (v6's 219, the TERM case strengthened, plus 28 new; run
    from the staged bank, `guard_test/guard_test.out`):
    - **Stage 0's verdict thresholds** (review item 4; the fake pool returned [1, 20] for every
      case before): CI [−12, −1] reads damage and generation 2 then refuses (91); [−12, 1] reads
      unread; [−8, 5] reads adoptable; [−0.5, 0.5] reads adoptable (hi > 0 is not damage);
    - **seats that exit 0 but are not DONE** (review item 3): a duplicated `(pair_id, half)` and a
      header carrying another seed each read PARTIAL (`seat_state.py`'s own output asserted) and
      the relaunch refuses;
    - **the smoke's runner** (review items 1–2): refused before Stage 0's verdict (91) and on
      damage (91), on a gen1 file Stage 0 did not play (92), on a drifted flags file (93; the
      launcher refuses it too), on another smoke script (93), on a held bank lock (9), on an empty
      `CUDA_VISIBLE_DEVICES` (94, allowed in CPU logic mode), without `IDLE_MIB` (2); **the real
      operator gate** (copied read-only into the fake HOME) refuses a busy GPU, IDLE + 513 MiB, a
      listed GPU job and a tampered gate; no refused call reaches the smoke; after Stage 0 the
      smoke gets INIT = `gen1_baked.pt`, 3 updates and the output stem `gen2/smoke_mem`, and **its
      flag string equals the training step's** (both calls logged by the stand-in);
    - the launchers' pins of the flags file, and the runner's pins of the checks library, flags
      file, smoke script and gate, equal the files under test;
    - **the TERM case** (review item 5) waits for `stage0/started` (written after the trap is
      installed) and the sleeping seat, up to 120 s, and asserts it got there.
  - Revision 4 (v6): **219 cases, 219 pass** (v4's 102, adapted, plus 117 new). The stand-in GPU device is
    `/dev/zero` (a real character device with another rdev) bind-mounted over `/dev/dxg` inside
    the namespace, asserted before any case; the real node is never touched. (The scan owner's own
    harness uses `/dev/full`.) Launchers run with `CUDA_VISIBLE_DEVICES` **unset** inside the
    namespace, as in production, since they refuse an empty value (exit 94); fake GPU jobs run under
    `setsid env -u CUDA_VISIBLE_DEVICES` and are killed by process group.
    - **the shared scan and its pins (35 cases):** the bank's `gpu_scan.sh` is the adopted file and the guard
      pins it; the launchers' own unpatched pins equal the guard and checks library under test (a
      stale pin from an earlier guard revision was found and fixed this way); real processes with the
      argv of each entry point (`python -u` arena_standard / simple_rl_loop / match_vs_uci /
      cost_probe / fit_value_recal / equiv_check / take_effect_collect, `-X` / `-W` values skipped,
      `python -m chess_anti_engine.run`, the lc0 binary, a sibling arena launcher) each refuse; a
      shell that only mentions a name and an unlisted program with a listed substring pass; the guard
      called by a listed launcher (its ancestor) and a listed child of the guard's own caller (a
      descendant) pass, while a sibling copy of the launcher refuses; an unlisted holder of the
      stand-in device refuses; a nested-namespace process holding a regular file named `/dev/dxg`
      passes; the scanner's own `/dev/dxg` not a character device refuses; with a capability-free
      scanner, a non-dumpable process refuses, argv[0] exactly `/usr/bin/ssh-agent` passes, the same
      name at another path refuses, and the capability-free control passes; the environ cases
      (empty value exempt; absent, `=0`, prefix- and suffix-confusable names, and an unreadable
      environ refused; exempt but holding the device refused: seven cases); a tampered and a missing scan file
      refuse; each launcher started with `CUDA_VISIBLE_DEVICES` set to empty refuses (94);
    - **guard v3 (the coordinator's addition):** the guard's two probe commands against the REAL
      `nvidia-smi` print integers (read-only); memory at idle ± 512 passes and ± 513 refuses; v2's
      bar (1,023 MiB) now refuses against idle 3,382; a release line without `idle_mib`, with a
      non-integer or empty one, or with 4,097 refuses, 4,096 passes; each stage's own `idle_mib` is
      used; utilisation 9 % passes, 10 % refuses, a spike in the 5th sample only refuses, 5 samples
      with 4 one-second pauses, a non-integer, a second GPU at 50 %, a failing probe and no line each
      refuse; the re-guard passes when memory returns within the wait, refuses after it (24 polls of
      5 s for 120 s, bounded), does not wait at 0, and refuses a non-integer wait; generation 2
      refused by the first step's re-guard writes no step marker and relaunches cleanly; Stage 0
      refused before its first seat plays no seat; the GPU taken mid-run stops at the next step's
      re-guard with every earlier step done and no marker for the refused one;
    - **re-guard retry (14 cases):** a listed holder that exits after 10 s (it appears at one
      memory probe and is killed at the next-but-one, so two 5 s waits): the re-guard passes, and
      `guard.log` holds its reason once, timestamped, then the pass with its duration; a holder that
      persists: refused at the deadline (24 waits for 120 s) naming the last reason, logged as one
      retrying line plus the refusal with its duration; memory high, then a holder, then idle:
      passes, with each distinct reason logged once in order; the first guard refuses a holder at
      once, with one attempt and nothing logged; the release revoked during the retries is refused
      at the next attempt (2 waits, not 24) naming the release, and the release file gone at a
      re-guard on an idle GPU is refused with no wait; a non-integer `GEN2_REGUARD_WAIT_S` refuses
      at the first guard; at launcher level, a persistent holder at generation 2's first step is
      refused after the **default 900 s** (180 waits) and logged, `GEN2_REGUARD_WAIT_S=10`
      overrides it (2 waits), and a transient holder at generation 2's first step and at Stage 0's
      first seat delays it, the run completes, and that stage's `guard.log` names it. The holder is
      started detached from the `nvidia-smi` stub, so, like an agent's test, it is not a descendant
      of the guard's caller;
    - **per-seat resume (review item 1):** both real seats complete with the pool failing →
      `move_aside` of the match refused (4), and the relaunch completes with **0 arena calls** for
      that match (readback + pool only); seat A done, seat B failed → the relaunch plays only seat
      B; a PARTIAL seat B → relaunch refused, whole-match move refused, `--seat A` refused,
      `--seat B` moves only B (seat A untouched), a second `--seat B` has nothing to move (7), then
      the relaunch plays only B; a finished seat whose row names other nets → stops (86) with the
      `HOLD_READBACK` note and `--seat` refuses it; an exit-0 seat with too few games → relaunch
      refused and `--seat` refuses it; `--seat` on a non-match step and `--seat C` refused (2);
      with no real seat finished, a real `pool.json` or `verdicts.json` still refuses a whole-match
      move (these isolate revision 3's match rule from the new finished-seat rule), and without them
      the failed match may move whole;
    - **isolating cases** (each refusal rule alone, with no real seat finished, so the new
      finished-seat rule cannot shadow it): Stage 0 with a generation-2 marker, and with
      `verdict.json`, refused, and otherwise movable whole; `--seat` refuses a seat whose `.exit`
      reads 0 with no result row, and a seat with a result row whose `.exit` is non-zero; a
      finished seat whose row names another commit is STALE (86);
    - **Stage 0:** a finished seat A with a failed seat B → whole-Stage-0 move refused, the relaunch
      plays only seat B and completes; a completed Stage 0 refuses a relaunch (95); a PARTIAL seat B
      → refused, `--seat A` refused, `--seat B` moves only B (kept, not deleted), then the relaunch
      completes without replaying A;
    - `trunk2`: 1.4 stops, the edge 1.35 passes, and the verdict carries 8 per-batch values and an
      SE.

    v4's cases (v3's 64, plus):
    - `move_aside`: a completed final match and first match refused (4) with the pool left in
      place; a completed Stage 0 refused, both after generation 2 consumed it and before any
      generation-2 marker; Stage 0 refused while `gen2/` has any marker; a match refused when its
      `.exit` is 0 but its check now fails, when it has a real `pool.json` but a non-zero `.exit`, and
      once `verdicts.json` exists (these isolate the match rule from the completed-step rule); a
      completed step refused;
      `../stage0`, `.`, `..`, `collect/../..`, `../../x`, an empty name, two arguments,
      `m_g2_vs_base/` and no argument all refused (2) and nothing moved; two moves of the same step
      under one frozen clock reading keep both attempts; a failed, unpooled match may move;
    - data checks: a tampered shard, a collect manifest with another seed, and a model that is not
      the arm manifest's output each refuse on relaunch (88), never skip;
    - inputs: a changed input file of a done step, and a changed launcher, each refuse (84);
      `resume.log` carries the launcher's sha256; every run step wrote its `.inputs`;
    - `trunk2`: a ratio of 0.5 stops with `HOLD_TRUNK2` before training (v5 moves the upper cases
      to 1.4 / 1.35);
    - licensing: replicates + gen 2 > gen 1 + gen 2 vs base damage reads "inconsistent readings; no
      claim; owner"; gen 2 vs gen 1 unread + gen 2 vs base improves carries the cumulative claim.
  - **Revision 5 mutants** (generator `make_mutants_v7.py` and runner `run_mutants_v7.sh`, banked
    OUTSIDE the sealed bank with the full sweep's results at
    `~/chess-artifacts/operations/rl-run1-gen2.mutants_v7/`, which has its own README and
    SHA256SUMS; one exact replacement each from the FINAL revision-5 files; a replacement that does
    not match exactly once aborts). 103 in all: the 63 carried-over shell / python mutants
    (every anchor still matches), the 20 scan clause mutants, five pin mutants (P1–P5: stale
    `GUARD_SHA` and `FLAGS_SHA` in the launcher, stale `CHECKS_SHA` in `move_aside.sh`, stale
    `SMOKE_SHA` and `GATE_SHA` in the runner) and 15 new:

    | Mutant | File |
    |---|---|
    | m65 the launcher redefines `TRAIN` after sourcing the flags file (drift) | `launch_gen2.sh` |
    | m66 the runner adds a flag (drift) | `run_smoke.sh` |
    | m67 no Stage 0 verdict check | `run_smoke.sh` |
    | m68 no gen1 provenance check | `run_smoke.sh` |
    | m69 INIT = raw Zq | `run_smoke.sh` |
    | m70 the gate skipped | `run_smoke.sh` |
    | m71 the flags pin ignored | `run_smoke.sh` |
    | m72 the flags pin ignored | `launch_gen2.sh` |
    | m73 batch 256 in the flags file (both see it) | `gen2_flags.sh` |
    | m74 the bank lock not taken | `run_smoke.sh` |
    | m75 empty `CUDA_VISIBLE_DEVICES` accepted | `run_smoke.sh` |
    | m76 = reviewer's R3: damage at hi < −5 | `run_stage0.sh` |
    | m77 = reviewer's R4: `(pair_id, half)` uniqueness dropped | `seat_state.py` |
    | m78 = reviewer's R5: header seed dropped | `seat_state.py` |
    | m79 adoptable at lo > −15 | `run_stage0.sh` |

    **At this push: the priority set of 20 (the 15 new mutants and P1–P5) are all KILLED.** The
    carried-over 79 (63 shell / python and the 20 scan mutants; m1–m4 of them were already KILLED
    at the push) are re-run on the unchanged final files, the bank at `BANK_SHA256SUMS`
    `2d6e1b67…`. Their results are banked outside the sealed bank at the path above (runner output
    per mutant, and the md5 of every final file they ran against) and reported in the PR's review
    record. A survivor means a new revision, not an edit here.
    The runner (review item 5) reruns ALONE any kill whose only FAIL line is the TERM case, and the
    rerun decides; P* mutants count only when a pin case fails; a non-pin kill whose only FAILs are
    pin cases is vacuous. `train_mem_smoke.py`'s own three mutants, regenerated on the revision-5
    file: all KILLED (`prelaunch/train_mem_smoke/mutants/mutant_results.txt`; `tms_test.sh` 7 / 7).
  - **Revision 4 mutants: 83 regenerated from the FINAL revision-4 files by one exact
    replacement each and run against harness v6** (`guard_test/mutants_v6/`: `make_mutants_v6.py`,
    the shared scan's `make_scan_mutants.py`, `run_mutants_v6.sh`, every mutant and its output,
    `mutant_results.txt`; a replacement that does not match exactly once aborts). **All 83 are KILLED,
    none vacuous**, plus the 2 pin cases (each KILLED). The runner honoured the operator's hold file
    (`GPU_CHECK_HOLD`: no new pass starts while it exists, in-flight passes finish). The longest
    single pass took about 9 min (526 s, an upper bound from result timestamps). A
    kill counts only with a RESULT line (every case ran) or an isolation ABORT; a run with neither is
    a harness failure, rerun alone; a kill whose only FAIL lines are the two pin cases counts as
    vacuous. The shards wrote disjoint output files.
    - 63 shell / python mutants: v4's twenty-nine and revision 4's first eleven (below), the guard v3
      mutants (no cap, no upper or lower band, release line without `idle_mib` defaulted,
      utilisation ignored, `<= 10`, one sample, non-integer utilisation or memory passing, the
      second GPU's utilisation ignored, no wait), no step re-guard, no seat re-guard, a re-guard
      refusal skipped, empty `CUDA_VISIBLE_DEVICES` accepted by either launcher, the scan call and the
      scan pin ignored, and the re-guard retry's own: the final status ignored, reasons not
      logged, only memory retried, **a revoke retried**, the bound fixed at 120 s, the
      `GEN2_REGUARD_WAIT_S` override ignored, the bound not validated;
    - 20 clause mutants of the shared `gpu_scan.sh` (argv branches, `-m`, `-X` / `-W`, `bash -c`,
      the device clause by rdev and type, a regular file named `/dev/dxg`, unreadable fds, the
      allowlist by basename, ancestors, descendants, and the six environ cases);
    - the launchers' own pins, by hand: a stale `GUARD_SHA` in `launch_gen2.sh` and a stale
      `CHECKS_SHA` in `move_aside.sh` each fail the pin case (`guard_test/mutants_v6/pin_cases.txt`).

    Revision 4's first eleven:

    | Mutant | File |
    |---|---|
    | a DONE seat is replayed | `gen2_checks.sh` |
    | a STALE seat (sha readback failure) treated as DONE | `gen2_checks.sh` |
    | whole-match move ignores a finished real seat | `move_aside.sh` |
    | whole-Stage-0 move ignores a finished real seat | `move_aside.sh` |
    | `--seat` moves a finished seat | `move_aside.sh` |
    | `--seat` moves a seat that has a result row | `move_aside.sh` |
    | a failed match refuses the relaunch (no per-seat resume) | `launch_gen2.sh` |
    | `trunk2` back to 2 batches | `launch_gen2.sh` |
    | Stage 0 replays whole (any attempt on disk refuses) | `run_stage0.sh` |
    | game count unchecked | `seat_state.py` |
    | PIN (`git_sha`) unchecked | `seat_state.py` |

    The licensing grid (`grid_v3.py`) rerun on the final revision-4 launcher: 48 / 48, unchanged
    (`guard_test/grid_v6/`).
  - **Revision 3 mutants, all made from the FINAL files by one exact replacement each
    (`guard_test/mutants_v4/make_mutants_v4.py`; a replacement that does not match exactly once
    aborts) and each killed** (`guard_test/mutants_v4/mutant_results.txt`). v3's fourteen shell
    mutants are regenerated on the final files: launcher without `|| exit 8`; gen2 with no guard;
    guard with no memory check, no release-line check, a substring `ps` match, `ps` failing open,
    only the first GPU line; Stage 0 trap losing the code; resume skipping without checking `.exit`;
    fit2 bounds ignored; move-aside deleting; gen2 trap set before the refusals; harness with stubs
    last on PATH (ABORT); harness with another HOME (ABORT). New in v4:

    | Mutant | File |
    |---|---|
    | a completed match moves | `move_aside.sh` |
    | any name accepted (no plain-name check, no allowlist) | `move_aside.sh` |
    | second-resolution stamp, no destination check, overwriting `mv` | `move_aside.sh` |
    | Stage 0 moves under generation-2 markers | `move_aside.sh` |
    | a completed step moves | `move_aside.sh` |
    | a completed Stage 0 moves | `move_aside.sh` |
    | collect check = existence only | `gen2_checks.sh` |
    | arm check = existence only | `gen2_checks.sh` |
    | inputs never compared | `gen2_checks.sh` |
    | trunk range ignored | `launch_gen2.sh` |
    | claim ignores gen 2 vs base damage | `launch_gen2.sh` (also killed by the grid: 1 cell) |
    | claim drops the cumulative-only row | `launch_gen2.sh` (also killed by the grid: 9 cells) |
    | shards digest unchecked | `stage_checks.py` |
    | seed unchecked | `stage_checks.py` |
    | model sha unchecked | `stage_checks.py` |

    The collect-gate mutant (B3b) is unchanged and banked with its own test.

## Provenance and incident

- **Incident (2026-11-21, 23:51–23:58 EST).** Version 1 of this bank's guard test ran the
  launchers under the real HOME, with a fake bank that held no `gpu_guard.sh`. Version 1 of
  `run_stage0.sh` had no `set -e`, and it sourced the guard without `|| exit`. The failed `source`
  fell through to the real follow-up launcher, which can start a real 4-game arena (recalibrated
  Zq vs the SL base, 64 sims, output inside the test's temp dir). Each of five test runs reached
  that point.
  - **Overlap.** The window 23:51:14–23:58:18 is recorded as possibly shared with the anchor
    screen's cell `Zq_k18.1` (started 23:52:31). That cell is matched by updates, not by the
    clock, and its log shows no OOM, CUDA error or traceback. Operator note:
    `$OPS/rl-run1-value-anchor/INCIDENT_gpu_overlap_20261121.md`.
  - **Evidence.** Whether any test arena reached the GPU is unknown in both directions. No arena
    output survives (the test's temp dirs were deleted). The CUDA JIT cache entry written at
    23:57:07 is most likely `Zq_k18.1`'s own: every anchor cell writes one a few minutes after it
    starts, and this one came 4.6 min after that cell started, the same lag as the cell before it.
    No file outside the temp dirs was written by these runs.
  - **Fix.** The fixes are the v2/v3 launchers and tests described under Guards. Since then every
    command in this work has run with `CUDA_VISIBLE_DEVICES=` exported.
- **Bank** `$OPS/rl-run1-gen2/` (`$OPS` = `~/chess-artifacts/operations`).
  - **What is read-only:** the scripts, `BANK_SHA256SUMS` (which no longer lists itself; review
    item 5) and the test dirs. The `stage0/` and `gen2/` dirs stay writable for outputs.
  - Revision 1 files are kept as `*.v1`, revision 2 files as `*.v2` (the launcher, `move_aside.sh`,
    the harness, `fakepy.py`, its output, `prelaunch/SHA256SUMS` and `BANK_SHA256SUMS`).
  - `prelaunch/SHA256SUMS` lists the seven prelaunch scripts (revision 4 adds `seat_state.py`;
    revision 3 added `stage_checks.py` and
    `trunk_pull_gen2.py`); both launchers check it.
  - `BANK_SHA256SUMS` (sha256 `2d6e1b67849cd850c74e28cc8dce59f76e63d42ef6a16b58d13a27187ba5e6f2`) lists
    482 files and verifies with `sha256sum -c` (the staged revision-5 bank,
    `$OPS/rl-run1-gen2.staging`, swapped in for the live bank only on this PR's MERGE verdict;
    Commands step 1). The revision-5 mutant sweep's results are banked OUTSIDE this sealed set
    (`$OPS/rl-run1-gen2.mutants_v7/`), so the seal and the table below are final.
  - Revision 4 files are kept as `*.v4` (`launch_gen2.sh`, the harness, `fakepy.py`, its output,
    `train_mem_smoke.py`, `tms_test.out`, `BANK_SHA256SUMS`; the smoke's revision-4 mutants as
    `prelaunch/train_mem_smoke/mutants.v4/`). Revision 5 adds `gen2/gen2_flags.sh`,
    `prelaunch/train_mem_smoke/run_smoke.sh` and `prelaunch/train_mem_smoke/cpu_logic_run/` (the
    exact operator command in CPU logic mode, with its driver and stand-ins).
  - Revision 3 files are kept as `*.v3` (the launchers, `move_aside.sh`, `gen2_checks.sh`, the
    harness, `fakepy.py`, its output, `prelaunch/SHA256SUMS` and `BANK_SHA256SUMS`). Revision 4
    evidence: `prelaunch/trunk_check/trunk8_baked.*` (the 8-batch measurement),
    `prelaunch/seat_state_test/` (match 8's real seats), `guard_test/mutants_v6/`,
    `guard_test/grid_v6/`, `prelaunch/train_mem_smoke/` (the memory smoke, its test, output and
    mutants; listed in `BANK_SHA256SUMS`, not in `prelaunch/SHA256SUMS`, which the launchers check
    and which is unchanged by the smoke), and `gpu_scan.sh` (the shared scan, final at v4).
  - Revision 3 evidence: `prelaunch/trunk_check/` (the reviewer's script verbatim, the
    reproduction), `prelaunch/stage_checks_test/` (the pilot collect and arm checks),
    `guard_test/mutants_v4/` (generator, runner, every mutant and its output),
    `guard_test/grid/` (the 48-cell grid and its two mutants).

| File | sha256 |
|---|---|
| `gpu_guard.sh` (revision 4: v3 + the re-guard retry; pinned in both launchers) | `a7a6765c66fd7c6d0c0a9da20c524cdc49b238214509656c4353db3dcab85604` |
| `gpu_scan.sh` (shared, final v4; pinned in the guard) | `949408e727e7ccbf942dd28f58ad800f958267577a93822950529ac20734f466` |
| `stage0/run_stage0.sh` (revision 4, unchanged) | `9280b38f131beb97633c9fff476976f7d6002c05844d5c533a06fec9b31efe0f` |
| `gen2/launch_gen2.sh` (revision 5: sources the pinned flags file) | `55973fa937a031d50ef883682d70277573d65a9d56abf430fe2fb282c3c10f6d` |
| `gen2/gen2_flags.sh` (new; pinned in the launcher and the smoke's runner) | `7f4d7adff874febd3d0051df090e80a88c3c03676aa666f3fb03cc62fb7be3eb` |
| `gen2/move_aside.sh` (revision 4) | `a5dcd10191d5251441affe000061485f37985307083edaf6d7a64dd6e9131954` |
| `gen2/gen2_checks.sh` (revision 4; pinned in both) | `76dc0f691f72b8a49ea9fef871c4e20090b2f89b6c3a424bc4001cb075d33bf3` |
| `prelaunch/SHA256SUMS` (seven scripts) | `b907ea98504c778e36697ab64ea3dbb40ff2690f81bd15ce3b4e6826aec454f5` |
| `prelaunch/seat_state.py` (new) | `a0f80af1286ca90d38eb6ddc77e84395292f0b8b0f73ad35654d579c842ceef0` |
| `prelaunch/stage_checks.py` | `362abc13f19c7b12c7cf290483c9b3dd67b741e7c3c0d8679378b568cb5ca89d` |
| `prelaunch/trunk_pull_gen2.py` (the reviewer's `39173757…` with the collect as a parameter) | `cfe660c2332b451817b0b8348d0205e60cbaca267a5dc3c78a40b96c5bfc5075` |
| `prelaunch/bake_recal.py` | `786bb537e0d1130ed2ae20bf81c7791239f4903095fd22a2f40840e3e8a06230` |
| `prelaunch/equiv_check.py` | `9af37c1b33bae5008067173d99422151ebe1402cc36989a722702697fe59574b` |
| `prelaunch/collect_gate.py` | `470a76aebe3a0e1fa94d54ff802024ed803026fae28df2056063e04b6b3023ff` |
| `prelaunch/take_effect_collect.py` | `9029f146958166c82b83da64de93852ce9adc5edc25171229f4c033f2607e37e` |
| `prelaunch/train_mem_smoke/train_mem_smoke.py` (revision 5: CPU mode forces cpu after parsing; 2 → 3 deltas; pinned in the runner) | `2ce38774e589c124266381b138ef9f7b689c2b9e2aace05e689c38d1a06d5c54` |
| `prelaunch/train_mem_smoke/run_smoke.sh` (new: the operator's one smoke command) | `dd523072dcbaea7d8f24c1fce48a8ef79be1ac6dfe3c74f64a954e2b99b1a736` |
| `prelaunch/train_mem_smoke/mutants/mutant_results.txt` (3 KILLED, revision-5 file) | `48bb0c77e36c941199557a79aa138c7c5b266b6c7d26ffe25ed7fb9cc88e9224` |
| sibling bank `operator/operator_gpu_check.sh` (read-only; pinned in the runner) | `2c4efb706548b69c78db644e16b4d7b7e60fb0a998236b3e3b1b4630d25e6ccd` |
| `prelaunch/train_mem_smoke/tms_test.sh` (new) | `13c1817c7f080ea9557fb635786729b8833d6ecd6c97fba71e2833a51b24ef09` |
| `guard_test/guard_test.sh` (v7) | `556f034de7d45803eac78862786d88363128da56dd5353ae39a25843aca54336` |
| `guard_test/fakepy.py` (v7) | `22959db83b843c3044e0854261ed683b6138a57b0e47c61a7db81342cdf2aa75` |
| `guard_test/guard_test.out` (247 / 247, run from the staged bank) | `37762cd7a48578f8e95951ae384bdd8057a683eacd281645d77cbfaa8fc07b0e` |
| `guard_test/mutants_v6/mutant_results.txt` (83 KILLED) | `0aeb0e03c68b6c36d4294562d66095481c37a0f6fef214cb8d1381ccdc216950` |
| `guard_test/mutants_v6/pin_cases.txt` (2 KILLED) | `869827a8b01992b1dad24b2ebfeac68962eaed449b515c105b8ed75f3530914b` |
| `guard_test/mutants_v6/make_mutants_v6.py` | `78885df926aff7f413c5bed1c48ee58ea28bdfe83d406cc969826cfe40b086d3` |
| `guard_test/mutants_v6/make_scan_mutants.py` | `e4d99db8b3e8d89c0432805648faea3e5ea78e7948bc95eb0d20daf602ae656b` |
| `guard_test/mutants_v6/run_mutants_v6.sh` | `6151e79be10a74d705f676ae03a519b974a80ef2cb3246cf6cc48e87e1e983a8` |

## Commands

**The operator sequence is the statistics reviewer's** (review of revision 4 at `d8f91bcab`, steps
0–10, under fix (a)), written here so the coordinator's driver and this record cannot diverge. Step
7 is one command that runs the gate itself, so nothing is retyped.

0. Before starting: the fixed head is merged with green CI. The GPU is free (the guard refuses
   otherwise).
1. Swap the staged bank into the live bank, then run `sha256sum -c BANK_SHA256SUMS` on the live bank
   (expect the staged value).
2. Confirm `rl-run1-followup/chain.exit` and `rl-run1-value-anchor/queue2.exit` exist.
3. The coordinator writes `stage0/OWNER_GO`, quoting the owner's statement.
4. The operator measures idle (`nvidia-smi` memory.used, utilisation under 10 %) and writes
   `stage0 idle_mib=<MiB> <time>` in `GPU_RELEASED_BY_OPERATOR`.
5. `setsid nohup run_stage0.sh`. It bakes and verifies `G1` on CPU, plays the 4-game smoke, then
   6,400 games. Read `stage0/verdict.json`.
6. If the reading is damage: stop. Generation 2 refuses with 91; the owner decides.
7. Otherwise, run the training memory smoke: `run_smoke.sh <idle_mib>`. It runs the named
   operator gate, takes `$COLLECT2 $TRAIN $ZQ` from the live bank's `gen2/gen2_flags.sh` (the file
   the launcher sources) and runs `train_mem_smoke.py … gen2/smoke_mem 3 $G1 $COLLECT2 $TRAIN $ZQ`.
   It passes only with exit 0 and PASS in `smoke_mem.json`, with reserved memory flat between
   updates 2 and 3. Anything else: write `gen2/HOLD_MEM` and stop.
8. The coordinator writes `gen2/OWNER_GO`. The operator re-measures idle and writes
   `gen2 idle_mib=<MiB> <time>`.
9. `setsid nohup launch_gen2.sh`. On `HOLD_TRUNK2`, a fit2 FAIL, `HOLD_READBACK`, or exit
   84 / 86 / 88: stop for the owner. Only a PARTIAL seat gets `move_aside.sh <match> --seat <A|B>`
   and a relaunch.
10. Read `gen2/verdicts.json` and write the results record in the same session.

```bash
OPS=$HOME/chess-artifacts/operations; B=$OPS/rl-run1-gen2
# 1. swap (on this PR's MERGE), then verify
mv $B $OPS/rl-run1-gen2.rev4.$(date -u +%Y%m%dT%H%M%SZ) && mv $OPS/rl-run1-gen2.staging $B
(cd $B && sha256sum -c --quiet BANK_SHA256SUMS && sha256sum BANK_SHA256SUMS)
# 2. queue markers
ls $OPS/rl-run1-followup/chain.exit $OPS/rl-run1-value-anchor/queue2.exit
# 5. Stage 0 (after stage0/OWNER_GO and the operator's "stage0 idle_mib=<MiB> <time>" line)
setsid nohup $B/stage0/run_stage0.sh > $B/stage0/run.out 2>&1 < /dev/null &
cat $B/stage0/verdict.json
# 7. the training memory smoke (after a non-damage verdict), ONE command; the gate runs inside it
$B/prelaunch/train_mem_smoke/run_smoke.sh <idle_mib> > $B/gen2/smoke_mem.log 2>&1; echo "exit $?"
cat $B/gen2/smoke_mem.json   # exit 0 + "verdict": "PASS" + both *_delta_2_to_3_gib ~ 0, else: echo "<why> $(date -Is)" > $B/gen2/HOLD_MEM
# 9. generation 2 (after gen2/OWNER_GO and the operator's "gen2 idle_mib=<MiB> <time>" line)
setsid nohup $B/gen2/launch_gen2.sh > $B/gen2/run.out 2>&1 < /dev/null &
# only for a PARTIAL seat named by the refusal:
$B/gen2/move_aside.sh <m_g2_vs_g1|m_g2_vs_base|stage0> --seat <A|B>
```

Results go into a separate results record, in the same session as the readout, judged by the rules
above.
