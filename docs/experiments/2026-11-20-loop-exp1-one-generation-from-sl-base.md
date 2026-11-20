# Loop experiment 1: one self-play generation from the SL base (prospective)

**Status:** DRAFT for owner approval. Nothing has been launched: no collection, no
training, no arena. This record exists so the experiment can be judged, and its
rule fixed, before any compute is spent (CLAUDE.md "Experiments" rule 1). It is
registered only once the owner approves it and the decisions below are filled in.
Nothing launches until (i) the owner's D1–D6, D8 and D9 are recorded in this record
by a follow-up commit (D7 is decided), and (ii) pre-launch checks L1–L4 pass. The driver support it needs
(#1430) is on main.
Revised 2026-11-20 after an independent review (findings M1–M7, m1–m11). Aligned
2026-11-20 with the driver as merged in #1430: commands, gate scripts and report
fields. D7 was decided by the owner on 2026-11-20 (fast collect path only). D1–D6,
D8 and D9 are unchanged and still open.
Revised 2026-11-20 after a dry run of L1–L4 on main `f51c51f20` (bank
`~/chess-artifacts/operations/rl-claude-20261119/exp1-prelaunch-dryrun/`, marked
NOT_A_CANDIDATE: a tiny-net CPU rehearsal, one 128-game collect, an audit
reproduction and a tiny arena pair; nothing in it is a result). L1–L4 passed. The
gaps it found are closed here: the openings split command, the G-T1 and G-T2
verdict commands, `--z` on the pooling script, the L4 build script, snapshots
derived from N, the SL base copied into the run dir, and recalibrated G-C5 and
G-T4 bounds and throughput.

**Prerequisites:**
- **Collector fixes, on main since `b410456b4`:**
  - #1427: the collect evaluator cap is sized to the search's leaf buffer.
  - #1428: the C buffer-full fallback is replaced by flush-and-resume, with a counter
    (ABI 11).
- **Extensions rebuilt** in the run's tree at or after `b410456b4`.
- **Every collect's `stats.json` must show `buffer_full_flushes == 0` and
  `eval_max_batch_realised == leaf_rows_required`.** A collect that does not is void.
- **Snapshot / collect-from driver support, #1430, on main since `885ab7b54`:**
  - `--snapshot-updates`: weights-only checkpoints at fixed update counts within a
    phase ([Train snapshots](../simple_rl_loop.md#train-snapshots));
  - `--collect-from`: several training arms on one sealed collection
    ([Training arms on one collection](../simple_rl_loop.md#training-arms-on-one-collection));
  - the gate instruments `scripts/simple_rl_collect_gates.py`
    ([Collect gates](../simple_rl_loop.md#collect-gates)) and
    `scripts/simple_rl_model_gates.py` ([Model gates](../simple_rl_loop.md#model-gates)).

  The 4,000-game fallback existed in case this PR did not merge. That reason no
  longer applies; 4,000 games remains the owner's D2 alternative. #1430 changed no C
  code, so the extension build above stands.

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
change. Each decision is argued in the section it links to. D7 is already decided.

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| D1 | [Which arm is primary, given its prior](#prior-and-choice-of-primary) | **Arm P** (policy only): measured prior −10.9 (−27.7 to +5.9) on F12's rows, no named failure mechanism. The likely outcome is "unread" or "kill", which would register that one policy-only generation at this dose is neutral. | **Arm V** (value on the search value): possibly a better chance of a positive read, with a named risk (a flatter value, the mechanism of Q50's loss) |
| D2 | [Dose](#collection) | **16,000 games WITH snapshot gates** (snapshots at 1,000 / 2,000 / 4,000 updates, derived from N by the [snapshot rule](#snapshots)) | **4,000 games** (one snapshot, at 1,000 updates, by the same rule; the commands are unchanged apart from `--games`) |
| D3 | [Search budget of the stored targets](#collection) | **64 sims** | 256 sims at N/4 games, as a separate later experiment |
| D4 | [Softening used by Arm S](#arms) | **`--train-target-temp 2.0`** | `--policy-prior-mix 0.5 --prior-mix-source reference` (SL base) |
| D5 | [Primary arena layout](#deciding-yardstick) | **Seat-balanced:** half the games with the arm as candidate, half as reference, same openings; no separate base-vs-base control | One orientation plus a same-day base-vs-base control. **Would need commands and a reading rule before selection:** none are written here, and the pooling script reads only the seat-balanced layout |
| D6 | [Positive control](#positive-control) | **Unconditional:** Arm V's recipe at 4,000 games from Q50 c12, played against Q50 c12. It cannot validate the primary: a pass is largely expected from fixing value calibration, and its dose is a quarter of the primary's | Skip it, saving about 3 GPU hours (2.6–3.6) |
| D7 | [Collect execution](#collection) | **DECIDED by the owner, 2026-11-20:** the fast (CUDA-graphs) collect path only, concurrency 128. "I don't think we need to maintain a fast and a slow path. It's okay to be a little bit different. And just be faster." | ~~Graphs off~~ (struck) |
| D8 | [Optimizer](#training) | **Fresh NAdamW, flat LR 3e-6, warmup 0**, as F12 | `--init-optimizer-state checkpoint` (Aurora): not recommended here |
| D9 | [Deciding N](#deciding-yardstick) | **Single fixed look at 3,200 games** (1,600 per orientation), read once | O'Brien–Fleming two-look: z 2.797 at 1,600, z 1.977 at 3,200. Look 2 [would need commands before selection](#deciding-yardstick) |

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

One collection from the SL base, at main ≥ `885ab7b54`.

| Setting | This experiment | Driver default | F12 | Source / reason |
|---|---|---|---|---|
| Sims / topk / c_scale / c_visit | 64 / 16 / 0.1 / 50 | same | same | the F12 instrument and the D1 reads |
| Target q-rescale; tree reuse | on; on | same | same | stored targets include inherited visits (D3) |
| Root noise | Gumbel 1.0 → 0.5 over moves 12–15 | same | not re-read here | driver default |
| Move temperature | 0.5 → 0.1 over 10 plies after the book, then 0.1 | same | not re-read here | driver default |
| Openings | split 153,842 TRAIN / 4,000 EVAL; book mix 95% / 5%; `cycle` | same | 192 fixed starts | 18% exact repeats under F12's starts |
| Games | 16,000 / 4,000 (D2) | 2,000 | 192 per cycle | below |
| Concurrency | 128 (D7) | 64 | 16 slots | the only clean measured setting |
| Evaluator cap | auto (= `leaf_rows_required`) | auto | — | #1427 |
| CUDA graphs | on: the only CUDA collect path (D7, decided) | always on for CUDA since #1433 (no flag) | — | owner's decision; no on-vs-off identity check |
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
  policy-drift extrapolation below reaches KL 0.5–1.2. So the 16,000 default is
  paired with the snapshot gates (#1430). The 4,000-game alternative stays closer to
  the measured range.

**Search budget (D3).** The evidence is mixed:
- the best 256-sim targets beat every transformed 64-sim target on mass, top-1 and
  regret (D3; for example cv0/cs0.5: mass 0.702, top-1 0.763, regret 0.014);
- the default 256-sim target does not beat the sharpest 64-sim cell on mass (0.674
  against 0.677), and the intervals overlap.

The default keeps 64 sims: it is the instrument of every banked read, and changing it
would add a second data-affecting change in one readout (rule 3).

**Throughput (128-game runs only).** On current main the fast path measured:
- 3,284 games/h: the dry run's collect on `f51c51f20`, this record's step 1 verbatim
  apart from `--games 128` (`stats.json` `games_per_hour`; 140 s wall);
- 3,760 games/h: #1433's `bench_after_default` (seed 20261121).

The earlier 2,424–2,583 games/h were measured before #1433 on the eager path and no
longer apply. The concurrency-256 reads (5,180–6,139 games/h) had the cap defect and
are not used. No collect longer than 128 games has been timed.

Estimate: about 4.3–4.9 h for 16,000 games, 1.1–1.2 h for 4,000.

## Training

**Common to every arm:**
- one phase, generation 1;
- `--presentations 1` (exact-once, shuffled), batch 256, `--mirror-prob 0.5`;
- `--lr 3e-6 --lr-floor 3e-6 --warmup-steps 0`;
- `--optimizer nadamw --optimizer-state fresh --init-optimizer-state fresh`;
- `--holdout-frac 0.05`, `--probe-rows 2048`;
- `--seed 2026112000` for the collection and every arm. The driver refuses an arm
  whose `--seed` differs from its source's, because the seed also draws the train
  phase's batch plan.

The driver's `--lr` default is 3e-5, so 3e-6 is passed explicitly. The holdout is a
sha256 of the game id (`is_holdout_game`), so every arm holds out the same games.

**Optimizer (D8).** F12 used fresh NAdamW at flat 3e-6 with warmup 0 and showed no
first-step hit: this driver's trainer on F12's cycle-1 rows moves the probe by
|ΔQ| 0.018, KL 0.030 ([driver doc](../simple_rl_loop.md)). The `--warmup-start 0`
zero-first-update caveat applies only with `--warmup-steps > 0`.

The SL base's own optimizer state (Aurora, `mlp_out` scope) would change the
optimizer family, which is a second change in one readout.

**Several arms on one collection (native, #1430).** The collection is its own run
dir, `collect/` (the source), which only collects. Each arm is its own run dir that
references the source's sealed generation-1 collect with `--collect-from`, pins its
own `run_config.json`, and holds no copy of the rows. The driver creates the
reference only when all of these hold, and refuses otherwise:
- **Generation 1 only.** The source collect must have been played by the arm's own
  generation g−1 model (the on-policy actor rule). Arms share generation 1, which the
  shared generation-0 model played; no arm trains a second generation on these rows.
- **Same `--init-checkpoint`** as the source: every arm passes the SL base file the
  source was started from.
- **Collect flags equal to the source's.** Only device, evaluator cap, dup-sketch
  budget and exact-repeat-vs-previous are exempt. CUDA graphs are no longer a setting
  (#1433): every CUDA collect replays them. The commands below pass
  one shared `$COLLECT` to the source and to every arm.
- **The arm's `--seed` equals the source's.**

**Freeze the source.** Once the first arm references it, `collect/` is made read-only
and is never redone or edited. The source cannot know its arms exist. A redo or edit
there makes every arm refuse its next stage, because the reference re-hashes the
source's manifest, shards, `games.jsonl`, `starts.txt` and generation-0 model.
Recovering means re-referencing and retraining every arm.

**Commands.**

```bash
export PYTHONPATH=.
R=~/chess-artifacts/operations/loop-exp1-20261120
TB=~/projects/chess/data/syzygy_3-4-5:~/projects/chess/data/syzygy_6
BOOKS=~/projects/chess/data/opening_books
mkdir -p $R/gates $R/arenas

# 0a. The SL base: copied ONCE into the run dir, verified, and write-protected. The shared
#     bank file is never chmodded; every command below uses the copy.
cp -n <shared SL base file> $R/sl_base.pt
echo "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0  $R/sl_base.pt" | sha256sum -c - \
  || { echo "SL base sha256 mismatch"; exit 1; }
chmod a-w $R/sl_base.pt
SL=$R/sl_base.pt

# 0b. The openings split (TRAIN / EVAL), then its check against the banked split.
python -u scripts/simple_rl_loop.py split-openings --out $R/openings \
  --book $BOOKS/8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip:16 \
  --book $BOOKS/2moves_v2_no8overlap_uho060110.pgn.zip:4 --eval-lines 4000 --salt simple-rl-v1
python3 -c 'import hashlib, json, sys
d = sys.argv[1]; m = json.load(open(d + "/manifest.json")); s = json.load(open(d + "/summary.json"))
want = {"train.jsonl": "1a98416db7263005da0412403267e722b8b0e814c2569728ee77e35421bcef1f",
        "eval.jsonl": "1d3d13ed11d669d0b2c7263fbd0bdc35059cbc2bb02d2babb0023159e8b5e325",
        "eval.pgn": "067675c264fb6b88251a2afdcdb298a720646a19b26aa1a90e7ca779eaf2a4de"}
got = {k: hashlib.sha256(open(d + "/" + k, "rb").read()).hexdigest() for k in want}
counts = (m["counts"]["train"], m["counts"]["eval"], s["eval_max_plies"])
print(json.dumps({"sha256": got, "train_eval_maxplies": counts}))
sys.exit(0 if got == want and counts == (153842, 4000, 16) else 3)' $R/openings; echo "exit=$?"
# D2: --games 4000 for the alternative. D7 (decided): the CUDA-graphs path, which since #1433
# is the only CUDA collect path (no flag).
COLLECT="--openings-dir $R/openings --init-checkpoint $SL --device cuda --seed 2026112000 \
  --games 16000 --concurrency 128 --sims 64 --topk 16 --c-scale 0.1 --c-visit 50 \
  --max-plies 450 --syzygy $TB --sf-path <stockfish>"
TRAIN="--presentations 1 --batch-size 256 --mirror-prob 0.5 --lr 3e-6 --lr-floor 3e-6 \
  --warmup-steps 0 --optimizer nadamw --optimizer-state fresh --init-optimizer-state fresh \
  --holdout-frac 0.05 --probe-rows 2048 --no-eval"
declare -A ARM=(
  [P]="--w-wdl 0"
  [V]="--w-wdl 1 --value-outcome-weight 0 --value-search-weight 1 --value-search-source survivor_q --value-draw-source net"
  [S]="--w-wdl 0 --train-target-temp 2.0")

# 1. The collection: the source run dir, collect stage only.
python -u scripts/simple_rl_loop.py collect --run-dir $R/collect --generation 1 $COLLECT
chmod -R a-w $R/collect          # frozen from here on

# 2. The phase's planned updates, from the sealed collect's row metadata. Record N here.
python - "$R/collect" <<'EOF'
import importlib.util, sys
from pathlib import Path
from chess_anti_engine.simple_rl.learn import planned_phase_updates
spec = importlib.util.spec_from_file_location("loop", "scripts/simple_rl_loop.py")
loop = importlib.util.module_from_spec(spec); spec.loader.exec_module(loop)
src = Path(sys.argv[1])
a = loop.parser().parse_args(["train", "--run-dir", str(src), "--openings-dir", "unused",
                              "--generation", "1", "--batch-size", "256",
                              "--holdout-frac", "0.05", "--presentations", "1"])
print(planned_phase_updates(src, 1, loop.train_settings(a, src)))
EOF
N=<the number printed above>
# The snapshot rule: the members of {1000, 2000, 4000} strictly below N. Empty below 1,001
# updates, and then the flag is omitted.
SNAPS=$(for s in 1000 2000 4000; do [ "$s" -lt "$N" ] && printf '%s,' "$s"; done); SNAPS=${SNAPS%,}
SNAPARG=${SNAPS:+--snapshot-updates $SNAPS}
echo "N=$N SNAPS=$SNAPS"

# 3. Each arm: seal the reference (a standalone collect pins nothing), then `run`,
#    which checks the snapshot plan against the sealed collect BEFORE pinning
#    run_config.json, skips the sealed collect after re-verifying the source, and trains.
for a in P V S; do
  python -u scripts/simple_rl_loop.py collect --run-dir $R/arm_$a --generation 1 \
    --collect-from $R/collect $COLLECT $TRAIN ${ARM[$a]} $SNAPARG
  python -u scripts/simple_rl_loop.py run --run-dir $R/arm_$a --generations 1 \
    --collect-from $R/collect $COLLECT $TRAIN ${ARM[$a]} $SNAPARG
done
```

**Openings split.** Step 0b reproduces the banked split
(`~/chess-artifacts/operations/rl-claude-20261119/loop-smoke-gpu/openings`, written by
the same command): 158,106 parsed lines → 153,842 TRAIN / 4,000 EVAL (3,725 / 275 by
book), `eval_max_plies` 16. Its check compares the three output files against the
sha256 the banked `manifest.json` records for them, and the counts. The manifest
file's own sha256 cannot be compared: it records the write time and runtime, so it
differs on every run. The book inputs' sha256 (`70d0dfa5…`, `aa0f2fd3…`) are in the new
manifest's `inputs`, to be recorded here. Step 0b, run verbatim from main `f51c51f20`
into a scratch directory (about 6 min of CPU), reproduced all three files' sha256 and
the counts; its check printed `exit=0`.

The planned-updates snippet was checked read-only on the banked 128-game collect
(`fix_c128_auto`, 12,372 rows): it prints 47. The SNAPS lines print `1000,2000,4000`
for N = 5,700 and 4,001, `1000` for N = 1,400 and 2,000, and nothing (flag omitted)
for N = 900. The dry run's L3 rehearsal ran this block on a tiny net (with the
snapshot set scaled down to its N).

**Pre-arena identity check (recorded, before any arena).**
- `sha256(arm_P/probe.npz) = sha256(arm_V/probe.npz) = sha256(arm_S/probe.npz)`;
- `sha256(arm_X/gen_0001/train/presented.npz)` equal across the three arms (same
  holdout, same batch plan; the driver's test asserts this for arms that differ only
  in loss flags);
- every arm's `gen_0001/collect/manifest.json` names the same
  `collect_reference.source_collect_dir`.

A mismatch voids all arms under the collect-gate disposition.

**Collect identity (lineage).** After the source collect seals, its
`simple_rl_collect_gates.py --content-digest` output (`content_sha256` over every
decoded shard array and row sidecar, and `games_sha256`) is recorded here, so later
re-analyses can prove they read the same rows. Shard file sha256 is not used for this:
the compressed chunk files and the metadata timestamp are not reproducible.

### Snapshots

- **Rule.** An arm's snapshots are the members of {1,000, 2,000, 4,000} strictly
  below the phase's planned update count N. N is computed from the sealed collect
  (step 2) before any arm starts, the list is derived from N by the SNAPS lines in
  step 2, and N and the list are recorded here. At about
  5,700 planned updates (16,000 games) the list is 1,000, 2,000, 4,000; at about
  1,400 (4,000 games, or the positive control) it is 1,000 alone. The driver allows a
  snapshot AT N; "strictly below" is this record's choice, so no snapshot duplicates
  the final model.
- **Refusal.** The train stage refuses a list that reaches past its planned updates.
  Because the arm's collect is sealed before its `run` starts, that check runs before
  `run_config.json` is pinned.
- **The final model is the last readout:** `gen_0001/train/model.pt`. It is not a
  snapshot.
- **Gates.** G-T1 and G-T2 are read at every snapshot and on the final model. A
  snapshot that fails a gate ends the arm there with the verdict "gate failed". That
  is an early stop, not a selection.
- Only the final model is ever played. No snapshot replaces it.

## Gates before any arena

A failed gate means that arm's verdict is fixed by [Dispositions](#dispositions); no
arena compute is spent on it.

**Collect gates.**

The clean references are 128-game collects from the SL base at concurrency 128 with
the fixed cap:
- `fix_c128_auto` (smoke `optimizer-start/mini/`, seed 20261119, eager path, before
  #1433), plus a concurrency-32 collect (`main_g128_c32`) for the older ranges;
- #1433's `bench_after_default` (`one-fast-path/`, seed 20261121, fast path);
- the dry run's collect (seed 2026112000, this experiment's own seed and command, on
  main `f51c51f20`, fast path; column "Dry run").

"Capped" is the `fix_c128_auto` games at the defective cap 512 (`main_c128`; #1427).

| Gate | Report field (`simple_rl_collect_gates.py`) | Pass | Clean (earlier) | Dry run | Capped |
|---|---|---|---|---|---|
| G-C1 buffer flushes | `buffer_full_flushes` | = 0 | — (written from #1428 on) | 0 | — |
| G-C2 evaluator cap | `eval_max_batch_realised` and `leaf_rows_required` | both = 4096 | 4096 = 4096 | 4096 = 4096 | 512 < 2048 |
| G-C3 reply-side Spearman(child Q, prior), fresh and reused roots | `spearman_childq_prior.reply_fresh.spearman_childq_prior`, `spearman_childq_prior.reply_reused.spearman_childq_prior` | ≥ +0.50 both | +0.596 / +0.602 | +0.576 / +0.608 | −0.053 / +0.096 |
| G-C4 book-end mover's score | `book_end_mover_score` | ≤ 0.70 | 0.582 | 0.434 | 0.820 (smoke gen 1: 0.90) |
| G-C5 target entropy | `target_entropy` | 0.60–0.78 | 0.684–0.694; `bench_after_default` 0.720 | 0.758 | 0.799 |
| G-C6 KL(target‖prior) | `kl_target_prior` | ≤ 0.50 | 0.290–0.332 | 0.294 | 0.676 |
| G-C7 rows per game | `rows_per_game` | ≥ 80 | 96.7–101.0 | 100.5 | 56.9 |
| G-C8 draw rate | `draw_rate` | 0.30–0.70 | 0.461–0.531; `bench_after_default` 0.562 | 0.617 | 0.156 |
| G-C9 survivor − node Q | `mean_survivor_minus_node_q` | 0.12–0.30 | 0.181–0.212; D3 bank 0.235 | 0.206 | 0.157 (weak; recorded) |
| G-C10 within-collect repeats | `repeat_rate_within` | ≤ 0.05 | 0.000 | 0.000 | 0.000 |
| G-C11 fast path ran | `stats.json`: `evaluator_mode`, `cuda_graph_stats.replays`; eager counters recorded | `evaluator_mode` = `"cuda_graphs"` and `replays` > 0 (reported, voids nothing; see below) | — | `cuda_graphs`, 1,791 replays, 627 eager over max rows | — |

**Calibration of G-C5 and G-C8 (three clean references).** The three clean 128-game
collects at concurrency 128, by seed 20261119 / 20261121 / 2026112000, read target
entropy 0.694 / 0.720 / 0.758 and draw rate 0.461 / 0.562 / 0.617. The G-C5 upper
bound 0.76 was set from the first collect alone; the third sits 0.002 under it.
- **Expected 16,000-game value.** The best estimate is the three-collect mean:
  entropy 0.724 (SD across collects 0.033, so SE of the mean 0.019), draw rate 0.547
  (SD 0.079, SE 0.046).
- **Sampling SD of one 16,000-game collect** around its own expectation is small:
  about 0.003 for entropy (0.033 × √(128/16,000), an upper bound since that SD is not
  all game sampling) and 0.004 for the draw rate (binomial).
- **What sets the band is the uncertainty in the mean, not the 16,000-game sampling.**
  The spread across collects is larger than game sampling alone (draw-rate SD 0.079
  against a binomial 0.044 at 128 games): the seeds draw different openings, and the
  three ran on three code versions.
- **G-C5: widened to 0.60–0.78.** 0.76 sat 1.9 SE above the estimated mean, so a
  clean collect could plausibly void every arm on it. 0.78 is 3.0 SE above the mean;
  0.60 is 6.6 SE below. The capped collect's 0.799 still fails, and G-C5 is not the
  capped signature's only detector: on that collect G-C2, G-C3, G-C6 and G-C7 all
  fail too.
- **G-C8: kept at 0.30–0.70.** That is 5.4 SE below and 3.3 SE above the mean.

G-C5, G-C6, G-C9 and G-C10 are the collect's own `stats.json` statistics, recomputed
by the script over every row. G-C3 and G-C4 come from a uniform seeded sample over all
shards (20,000 rows, `--seed 0`) and from every game.

**Book-end mover's score:** over all games, (wins + 0.5 × draws) / games for the side
to move at the end of the book line.

**Spearman references use average ranks.** The script ranks ties by their average
rank. The scratch `side_stats.py` used ordinal ranks, which differ on the capped
collect's many tied child Qs (ordinal capped −0.007 / +0.110, reproduced exactly by
the script with ordinal ranks). Only average-rank references are cited here, from the
[driver doc](../simple_rl_loop.md#collect-gates).

**G-C2 is gated three ways.** #1431 (`054548fb8`) added `{"equals": v}` bounds, for
numbers and booleans, to the gate scripts' `--thresholds`. G-C2 keeps the two numeric
gates, `eval_max_batch_realised` and `leaf_rows_required`, each with min = max = 4096
(the leaf rows of the clean concurrency-128 reference, the decided D7 setting), and
adds `"eval_max_batch_equals_leaf_rows": {"equals": true}`. G-C1 is
`{"equals": 0}`.

**G-C11: the fast path ran.** D7 makes the CUDA-graphs collect the only path, so
something must show it ran. Read from the collect's own `stats.json`:
- **Reported:** `evaluator_mode` is `"cuda_graphs"` (the realised evaluator mode main
  banks since #1433) and `cuda_graph_stats.replays` > 0. If either
  fails, "fast path did not run" is recorded here before any training proceeds. It
  voids nothing: graphs are intended to change speed only (their identity on this net
  is unchecked; see [Not verified](#not-verified)), so the collect is kept as data.
- **Recorded only:** the eager counters `eager_over_max_rows`,
  `eager_over_max_graphs` and `eager_relations` (dense forwards that ran eager instead
  of replaying a graph). A non-trivial eager share is expected: on the banked
  graphs-on bench collect, 573 dense forwards ran eager against 1,590 replays. The
  eager share is therefore recorded, not gated.

The check writes its own output file (no pipe, so the exit code seen is Python's) and
prints the exit code, which is recorded here. Exit 0 = the fast path ran; exit 3 =
"fast path did not run" (reported, not a void); any other code (for example 127, no
interpreter) means the check itself did not run and is repeated.

```bash
gc11() {  # $1 = sealed collect dir, $2 = output JSON
  python3 -c 'import json, sys
s = json.load(open(sys.argv[1])); g = s.get("cuda_graph_stats") or {}
keys = ("replays", "captures", "recaptures", "eager_over_max_rows", "eager_over_max_graphs", "eager_relations")
out = {"evaluator_mode": s.get("evaluator_mode"), "cuda_graphs": s.get("cuda_graphs"), **{k: g.get(k) for k in keys}}
open(sys.argv[2], "w").write(json.dumps(out) + "\n")
print(json.dumps(out))
sys.exit(0 if s.get("evaluator_mode") == "cuda_graphs" and (g.get("replays") or 0) > 0 else 3)' "$1/stats.json" "$2"
  echo "exit=$?"
}
gc11 $R/collect/gen_0001/collect $R/gates/collect_primary.gc11.json          # primary
gc11 $R/posctl/collect/gen_0001/collect $R/gates/collect_control.gc11.json   # control
```

This block was run verbatim (the function, then a call per collect) on two banked
bench collects. A collect written by the #1433 code (`bench_after_default`) printed
`exit=0`, with `evaluator_mode` `"cuda_graphs"`, `replays` 1,590 and
`eager_over_max_rows` 573. A graphs-off collect from before #1433 printed `exit=3`
(no `evaluator_mode`, `cuda_graphs` false).

**Command.** Exit code 3 means a gate failed or its field was missing or NaN:

```bash
python -u scripts/simple_rl_collect_gates.py $R/collect/gen_0001/collect --train-target-temp 2 \
  --thresholds $R/gates/collect_primary.json --out $R/gates/collect_primary.out.json
```

`$R/gates/collect_primary.json`, verbatim:

```json
{
 "buffer_full_flushes": {"equals": 0},
 "eval_max_batch_realised": {"min": 4096, "max": 4096},
 "leaf_rows_required": {"min": 4096, "max": 4096},
 "eval_max_batch_equals_leaf_rows": {"equals": true},
 "spearman_childq_prior.reply_fresh.spearman_childq_prior": {"min": 0.50},
 "spearman_childq_prior.reply_reused.spearman_childq_prior": {"min": 0.50},
 "book_end_mover_score": {"max": 0.70},
 "target_entropy": {"min": 0.60, "max": 0.78},
 "kl_target_prior": {"max": 0.50},
 "rows_per_game": {"min": 80},
 "draw_rate": {"min": 0.30, "max": 0.70},
 "mean_survivor_minus_node_q": {"min": 0.12, "max": 0.30},
 "repeat_rate_within": {"max": 0.05}
}
```

The control's file is in [Positive control](#positive-control).

Checked read-only, with the gate script at `054548fb8`, on the banked clean 128-game
collect (`fix_c128_auto`): every field resolves, and every gate passes, including
`eval_max_batch_equals_leaf_rows` `{"equals": true}`, except `buffer_full_flushes`. That collect predates
#1428 and does not carry the field, so the gate fails as missing, as the script
intends. On the dry run's collect (main `f51c51f20`), every gate passes, under the
earlier 0.76 bound too.

**Train gates (per arm, at the end of the phase and at every snapshot).**

**G-T1 confidence band on the fixed 3,080 D3 audit rows**
(`~/chess-artifacts/operations/rl-claude-20261119/rl-audit/positions.npz`, sha256
`0d335d6487cd702a…`), measured by `scripts/simple_rl_model_gates.py --audit-rows`. It
loads the model with the arena's loader and runs the forward under bf16 autocast on
CUDA, as the banked references were made. Banked reproduction for the SL base: this
script reads 0.4086 / 0.5133 on CPU fp32, against 0.4088 / 0.5132 banked (bf16 CUDA).
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

Commands, for each snapshot and for the final model of arm X:

```bash
for m in $R/arm_X/gen_0001/train/snapshots/step_*/model.pt $R/arm_X/gen_0001/train/model.pt; do
  python -u scripts/simple_rl_model_gates.py --model $m --device cuda \
    --audit-rows ~/chess-artifacts/operations/rl-claude-20261119/rl-audit/positions.npz \
    --thresholds $R/gates/gt1_X.json --out $R/gates/arm_X_$(basename $(dirname $m))_gt1.json
done
```

The threshold files, verbatim:
- `gt1_P.json` and `gt1_S.json`:
  `{"audit.mean_abs_q": {"min": 0.37, "max": 0.45}, "audit.mean_p_draw": {"min": 0.48, "max": 0.55}}`
- `gt1_V.json`:
  `{"audit.mean_abs_q": {"min": 0.33, "max": 0.45}, "audit.mean_p_draw": {"min": 0.48, "max": 0.55}}`

Outputs go to `$R/gates/` (`step_<N>` or `train` in the name), never into a sealed
stage directory.

**G-T1 verdict.** The loop above discards each run's exit code, so the verdict is read
from the output files. The expected files are named from `$SNAPS` (one per snapshot,
plus `train`), so a missing file fails rather than being skipped. This prints, per
file, the model and `gates_passed` (or `MISSING`), then the expected count
(snapshots + 1), and exits 3 if any file is missing or not `true`:

```bash
gt1() {  # $1 = arm letter; reads $R and $SNAPS (step 2)
  python3 -c 'import json, os, sys
r, arm, snaps = sys.argv[1], sys.argv[2], [s for s in sys.argv[3].split(",") if s]
files = [f"{r}/gates/arm_{arm}_step_{s}_gt1.json" for s in snaps] + [f"{r}/gates/arm_{arm}_train_gt1.json"]
bad = 0
for f in files:
    if not os.path.isfile(f):
        print("MISSING", f); bad = 1; continue
    o = json.load(open(f)); ok = o.get("gates_passed") is True
    print(o.get("model"), "gates_passed", o.get("gates_passed"), f); bad |= not ok
print(len(files), "file(s) expected")
sys.exit(3 if bad else 0)' "$R" "$1" "$SNAPS"
  echo "exit=$?"
}
gt1 P; gt1 V; gt1 S
```

Exit 0 = G-T1 passed on every expected file; exit 3 = a gate failed or a file is
missing; any other code means the check did not run and is repeated. Run on the dry
run's tiny-net rehearsal outputs with its `SNAPS=2,4,8` (4 files for arm P), it
printed each `gates_passed` false and `exit=3`, as it should for an untrained tiny
net. With `SNAPS=2,4,8,16` it also printed `MISSING` for `step_16`.

**G-T2 probe drift against the SL base:** a gross-defect guard only. A read of KL > 1.5
or |ΔQ| > 0.4 fails. Every read is recorded. The fields are
`probe_vs_previous.policy_kl_prev_to_new` and `probe_vs_previous.mean_abs_dq`, in each
snapshot's `metrics.json` and in the final `gen_0001/train/metrics.json`, measured
against the phase's init model (the SL base; Q50 c12 for the positive control).
References, by instrument:

| Instrument | Run | Updates | KL | \|ΔQ\| |
|---|---|---:|---:|---:|
| `probe_vs_previous` | corrupted smoke, policy only | 46 | 0.148 | 0.295 |
| `probe_vs_previous` | driver trainer on F12 cycle 1 | 69 | 0.030 | 0.018 |
| P1 held-out cycle-2 rows | P1 c1 / c6 / c12 | about 69 / 470 / 948 | 0.038 / 0.215 / 0.299 | 0.020 / 0.048 / 0.066 |
| P1 held-out cycle-2 rows | F12 c12, value on | 948 | 0.288 | 0.107 |

Extrapolated, P1's KL reaches about 0.5–1.2 at 5,700 updates, so no tighter bound is
defensible at the default dose.

The snapshot `metrics.json` carries `probe_vs_previous` at top level; the final
`gen_0001/train/metrics.json` carries it under `summary`. This reads both, prints every
read and exits 3 if any expected file is missing (named from `$SNAPS`, as in
`gt1`) or any read is above a bound, missing or not finite:

```bash
gt2() {  # $1 = arm run dir; reads $SNAPS (the arm's own list)
  python3 -c 'import json, math, os, sys
d, snaps = sys.argv[1] + "/gen_0001/train", [s for s in sys.argv[2].split(",") if s]
files = [f"{d}/snapshots/step_{s}/metrics.json" for s in snaps] + [d + "/metrics.json"]
bad = 0
for f in files:
    if not os.path.isfile(f):
        print("MISSING", f); bad = 1; continue
    m = json.load(open(f))
    p = m.get("probe_vs_previous") or (m.get("summary") or {}).get("probe_vs_previous") or {}
    kl, dq = p.get("policy_kl_prev_to_new"), p.get("mean_abs_dq")
    ok = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (kl, dq)) and kl <= 1.5 and dq <= 0.4
    print(f, "policy_kl_prev_to_new", kl, "mean_abs_dq", dq, "pass" if ok else "FAIL"); bad |= not ok
print(len(files), "file(s) expected")
sys.exit(3 if bad else 0)' "$1" "$SNAPS"
  echo "exit=$?"
}
gt2 $R/arm_P; gt2 $R/arm_V; gt2 $R/arm_S
SNAPS=<the control's own list, from its N>; gt2 $R/posctl/arm_V    # the control, against Q50 c12
```

Exit codes as for G-T1. Checked on the dry run's rehearsal arms P and V with
`SNAPS=2,4,8` (3 snapshots + final, every read printed, `exit=0`); with
`SNAPS=2,4,8,16` (`MISSING` for `step_16`, `exit=3`); and on a hand-made file with KL
1.6 and a missing |ΔQ| (both reads FAIL, `exit=3`).

**G-T3 held-out policy CE.** After the phase it must not exceed the before-phase
value, on generation 1's held-out games. This is a manual comparison of two numbers
from two JSON outputs (no script gates it); both numbers and the comparison are
recorded here.
- **Arms P and V:** `summary.heldout_before.policy_ce` against
  `summary.heldout_after.policy_ce` in the train manifest (temperature 1, the target
  they train on).
- **Arm S:** `heldout.policy_ce` from `scripts/simple_rl_model_gates.py` at
  `--train-target-temp 2`, run on BOTH the before model and the final model:

  ```bash
  for m in $R/arm_S/gen_0000/model.pt $R/arm_S/gen_0001/train/model.pt; do
    python -u scripts/simple_rl_model_gates.py --model $m --device cuda \
      --collect $R/arm_S --gen 1 --train-target-temp 2 \
      --out $R/gates/arm_S_gt3_$(basename $(dirname $m)).json
  done
  ```

  The CE logged in S's `metrics.json` and manifest is against the temperature-1
  target, which S does not train on, and it rises by design (1.06 → 1.16 in #1430's
  45-update check). S's G-T3 is never read off `metrics.json`.
- The script at temperature 1 equals the train stage's `heldout_after` (tested), so
  the same command at `--train-target-temp 1` cross-checks P and V.

**G-T4 realised settings, read off the consumers** (`metrics.json`, train manifest):
- presentation histogram exactly `{1: trainable rows}`;
- `trained_on_capped_rows` false (the field is always written);
- `summary.snapshot_updates`, and `[s["update"] for s in summary.snapshots]`
  (`summary.snapshots` is a list of dicts), both equal the recorded snapshot list;
- `realised_lr_param_groups` all 3e-6;
- `realised_policy_target_temp` 1.0 for P and V, 2.0 for S;
- **Arm S:** `retempered_target_entropy.2.overall` from the collect-gate report
  (`--train-target-temp 2`) recorded, and above the temperature-1 `target_entropy`
  that Arm P trains on;
- **Arm V:**
  - `realised_search_wdl_frac` = 1.0;
  - `targets.value_from_survivor_q` ≥ 98.5% of `targets.rows`;
  - `value_fallback_stored_search_wdl` and `value_without_search_target` recorded.

  **The 98.5% bound, from evidence.** The share of rows with a usable survivor Q
  (`has_root_survivor_q`, a finite `root_survivor_q` and `has_root_net_wdl`, as
  `prepare_targets` counts them) over every row of the three clean 128-game
  collects: 98.77% (`fix_c128_auto`), 98.89% (`bench_after_default`), 99.12% (dry
  run); mean 98.93%, SE of the mean 0.10 pp. The earlier ≥ 99% would have failed two
  of the three clean collects. 98.5% is 4.2 SE below the mean and 0.27 pp below the
  lowest clean collect; a 16,000-game collect's own sampling SD is about 0.02 pp. A
  broken survivor-Q path would read far lower.

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
  --candidate <run>/arm_P/gen_0001/train/model.pt --reference <run>/sl_base.pt \
  --label exp1_primary_A --games-out <run>/arenas/exp1_primary_A.games.jsonl \
  --pgn-out <run>/arenas/exp1_primary_A.pgn --out <run>/arenas/results.jsonl

# B: arm as reference
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON \
  --candidate <run>/sl_base.pt --reference <run>/arm_P/gen_0001/train/model.pt \
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
The script's `--z` (default 1.96) sets the half-width the verdict reads, and the
output records `z` and its two-sided coverage. Every block also keeps its 1.96
interval (`*_ci95`) beside the one at `z` (`*_ci`).
If an opening repeats within a run, all its pairs join one cluster. The script is
`scripts/arena_seat_balanced_pool.py` (#1434):
- **Inputs:** the two `--games-out` logs, plus each run's result row (matched by
  game-log fingerprint) for the fields only the row carries.
- **Checks:** the opening key is `opening_fen` + `opening_uci`. Run A's and run B's
  index → opening maps must be identical. Both runs must be complete, with
  `buffer_full_flushes` 0 and the realised evaluator cap recorded, and their settings
  must be equal after the seat swap. `--arm` (required) must be run A's candidate and
  run B's reference.
- **Net identity, by sha256:** the arena banks `candidate_sha256`, `reference_sha256`
  and `openings_sha256` (plus per-file `candidate_files` / `reference_files`) in every
  result row, and the same block, unfingerprinted, in each game-log header. A resume
  onto files whose sha256 differs is refused. The pooling script then asserts three
  things: A's candidate sha256 is B's reference sha256 and vice versa; every row
  sha256 equals the file's sha256 at pool time; and the two runs' openings sha256
  are equal. The older mtime heuristic applies only to legacy rows without these
  fields, under `--allow-legacy-rows`, which a registered read never uses. The output
  records which check ran (`net_check`: `row_sha256` here).
- **Outputs:** estimate, SE, 95% CI, Elo, n_clusters, the realised A/B correlation and
  the implied seat offset. Provenance: the sha256 of the four input files and of every
  net file at pool time.
- **Verdict:** `lb-gt-zero` is decided on the excess scale (lower bound of the excess
  > 0 is success; upper bound < 0 is kill). This is the same rule as an Elo lower
  bound > 0, and it still holds where an Elo end is undefined at an extreme score.

```bash
PYTHONPATH=. python -u scripts/arena_seat_balanced_pool.py \
  --arm <run>/arm_P/gen_0001/train/model.pt \
  --a-games <run>/arenas/exp1_primary_A.games.jsonl --b-games <run>/arenas/exp1_primary_B.games.jsonl \
  --a-result <run>/arenas/results.jsonl --b-result <run>/arenas/results.jsonl \
  --a-label exp1_primary_A --b-label exp1_primary_B \
  --verdict-rule lb-gt-zero --out <run>/arenas/exp1_primary_pooled.json
```

The primary uses the default `--z 1.96`. The other registered readings use the same
command with these changes:
- **Secondary confirmation (97.5%, two-sided z 2.2414):** `--arm` the confirmed arm's
  model, its `exp1_confirm_A` / `exp1_confirm_B` logs and labels, `--z 2.2414`,
  `--out <run>/arenas/exp1_confirm_pooled.json`.
- **D9 alternative, O'Brien–Fleming, look 1:** runs A and B at `--games 800` each
  (1,600 games), pooled with `--z 2.797`.
- **D9 alternative, look 2 would need commands before selection.** Look 2 reads at
  `--z 1.977` the look-1 games PLUS 800 new pairs per orientation. Neither tool does
  that today: the arena cannot extend a log (`games` is in its resume fingerprint),
  and the pooling script reads one log per orientation. A fresh pair of 1,600-game
  runs would replay look 1's openings as new games rather than add to them, so it is
  not the registered design. The default single look needs none of this.

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
- `candidate_sha256`, `reference_sha256` and `openings_sha256` present and non-null,
  with `game_log_provenance` `this_run` or `recorded` (never `absent`: every log of
  this experiment is written by an arena that records provenance);
- `eval_leaf_cap_bound` False;
- `eval_cuda_graphs` true, in both orientations. Since #1433 the arena replays CUDA
  graphs under `--compile off` on CUDA matched-sims; the pooling script also requires
  the field to be equal between runs A and B.
- `buffer_full_flushes` 0. The arena writes it into the result record on main
  (`scripts/arena_standard.py`, read at `b410456b4`). The field's presence in a real
  run is unverified until the first run.
- `games == games_requested`;
- the same printed config sha across all arenas of this experiment.

Per-game rows, PGNs and seeds are banked.

**Write-protect the nets.** From the end of each arm's training until its pooling is
recorded, every net that plays in an arena of this experiment is made read-only with
`chmod a-w`: each arm's `gen_0001/train/model.pt` and (for the control)
`posctl/arm_V/gen_0001/train/model.pt`. The two start nets are shared bank files
and are never chmodded: each is copied once into the run dir, verified and
write-protected there, and only the copy is used (`$R/sl_base.pt`, step 0a;
`$R/posctl/q50_c12.pt`, [Positive control](#positive-control)). This complements
the sha256 check. The check detects a rewrite after the fact; the write protection
prevents one.

**Secondaries.**
- **Screens:** V and S that passed their gates play 400 games each, seat-balanced
  (200 + 200), both on seed 2026112003, so they meet the same openings. Their
  `--games-out` logs are `$R/arenas/exp1_screen_{V,S}_{A,B}.games.jsonl`.
- **Confirmation:** the better screen point estimate is confirmed at 1,600 games,
  seat-balanced, on seed 2026112005, with `--openings-fen` built from the EVAL lines
  minus the screen's openings. Building that file is pre-launch check L4:

  ```bash
  python -u scripts/arena_openings_minus_screen.py --openings-dir $R/openings \
    --screen-games 400 --screen-seed 2026112003 \
    --check-games 1600 --check-seed 2026112005 --out $R/openings/confirmation.fen
  ```

  The script rebuilds the screens' 200 opening pairs from `--screen-games` /
  `--screen-seed` with the arena's own sampler and rng, writes the remaining EVAL
  lines as full-history `<root_fen> | <uci ...>` rows, and loads the file as the
  confirmation arena will. It refuses (exit 2, file removed) unless all 800 pairs
  load with their history and none is in the REBUILT screen set. It prints the counts
  and the file's sha256, which is recorded here. At L4 the screens have not been
  played, so this proves only that the arguments rebuild a set that the file
  excludes: a wrong seed or game count would rebuild a different set, and nothing at
  L4 can see that. On the banked split it wrote 3,800 rows (4,000 − 200), sha256
  `8e1973153d666893…`, 800 pairs with history: byte-identical to the dry run's L4
  file.

  **Screen-log check (after both screens, before the confirmation).** The same
  command is re-run with the four screen logs, which proves the rebuilt set is the
  set the screens actually played:

  ```bash
  python -u scripts/arena_openings_minus_screen.py --openings-dir $R/openings \
    --screen-games 400 --screen-seed 2026112003 \
    --screen-log $R/arenas/exp1_screen_V_A.games.jsonl --screen-log $R/arenas/exp1_screen_V_B.games.jsonl \
    --screen-log $R/arenas/exp1_screen_S_A.games.jsonl --screen-log $R/arenas/exp1_screen_S_B.games.jsonl \
    --check-games 1600 --check-seed 2026112005 --out $R/openings/confirmation_checked.fen
  ```

  It refuses (exit 2) unless the rebuilt set equals each log's set of played
  (`opening_fen`, `opening_uci`) keys, and its printed sha256 must equal L4's. Only
  then does the confirmation run, with `--openings-fen $R/openings/confirmation.fen`
  in place of `--openings` / `--opening-plies`; "no confirmation pair was played in a
  screen" is established by this step, not by L4. Checked on the dry run's two real
  matched-sims logs (8 games, seed 2026112001): the right seed passes, and seed
  2026112004 is refused. If the file cannot be built or the check refuses, the
  confirmation uses `eval.pgn`, and the overlap (about 40 of 800 expected) is counted
  and recorded.
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
- **Collection:** 4,000 games from the same split's TRAIN lines, in its own source
  run dir `posctl/collect`, with `--seed 2026112010` (its own opening permutation).
  The overlap with the primary collection's lines is counted and recorded.
- **Start file:** Q50 c12 is a shared bank file, so it is copied once into the run
  dir and only the copy is used:

  ```bash
  mkdir -p $R/posctl
  cp -n <shared Q50 c12 file> $R/posctl/q50_c12.pt
  echo "bb157a8412a2085b7dbb77cee64aef0c5d3498f065648c6244a02804347f9c33  $R/posctl/q50_c12.pt" \
    | sha256sum -c - || { echo "Q50 c12 sha256 mismatch"; exit 1; }
  chmod a-w $R/posctl/q50_c12.pt
  ```
- **Procedure:** the primary's, with `$COLLECT` changed only in
  `--init-checkpoint $R/posctl/q50_c12.pt --games 4000 --seed 2026112010`. One arm,
  `posctl/arm_V`, references it with `--collect-from $R/posctl/collect`, the same
  init checkpoint and seed, and Arm V's flags. `posctl/collect` is frozen once the arm
  references it. Snapshots follow the [rule](#snapshots): at about 1,400 planned
  updates, one at 1,000.
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
  own read, and mean P(draw) must stay in 0.45–0.60. Read at the end and at every
  snapshot. The floor is the value `simple_rl_model_gates.py` reads for
  `posctl/arm_V/gen_0000/model.pt` (Q50 c12) on the same device before training,
  recorded here; banked 0.2980 (bf16 CUDA) and 0.2982 (CPU fp32). Pre-launch check L2
  writes that value into `gt1_control.json` before training. The file's shape, with
  the value still to be written shown here as the placeholder FLOOR (FLOOR is not
  valid JSON and must be replaced by the measured number):
  `{"audit.mean_abs_q": {"min": FLOOR}, "audit.mean_p_draw": {"min": 0.45, "max": 0.60}}`.
- **G-C11 (reported, voids nothing):** the same `gc11` check on
  `posctl/collect/gen_0001/collect` (the second call in the G-C11 block), with its
  printed exit code recorded.
- **Collect gates, gated:** G-C1 (= 0), G-C2 (equal), G-C7 (≥ 80 rows per game) and
  G-C10 (≤ 0.05) with the primary's bounds, and G-C3 at ≥ +0.30 on both root classes
  (the primary's bound is +0.50). These test the machinery, not the start net. G-C3 is
  looser here because no Q50 c12 collect reference exists and the statistic also
  reflects agreement between that net's own policy and value, which were trained
  differently; +0.30 still separates clearly from the defect signature (average
  ranks: −0.053 / +0.096).

  `$R/gates/collect_control.json`, verbatim:

  ```json
  {
   "buffer_full_flushes": {"equals": 0},
   "eval_max_batch_realised": {"min": 4096, "max": 4096},
   "leaf_rows_required": {"min": 4096, "max": 4096},
   "eval_max_batch_equals_leaf_rows": {"equals": true},
   "spearman_childq_prior.reply_fresh.spearman_childq_prior": {"min": 0.30},
   "spearman_childq_prior.reply_reused.spearman_childq_prior": {"min": 0.30},
   "rows_per_game": {"min": 80},
   "repeat_rate_within": {"max": 0.05}
  }
  ```
- **Collect gates, recorded only:** G-C4, G-C5, G-C6, G-C8, G-C9. Their bands were set
  on SL-base collects and the expected values for a Q50 c12 collect are unknown.
- **G-T2:** the same gross guard, measured against Q50 c12 (the control's start), not
  the SL base: KL > 1.5 or |ΔQ| > 0.4 fails.
- **G-T3:** held-out policy CE against the stored target must not exceed the
  before-phase value (`summary.heldout_before` / `heldout_after`, as for Arm V).
- **G-T4:** the Arm V checks unchanged: presentation histogram `{1: trainable rows}`,
  `trained_on_capped_rows` false, `summary.snapshot_updates` and the `update` of each
  `summary.snapshots` entry as recorded, `realised_lr_param_groups` all 3e-6,
  `realised_policy_target_temp` 1.0, `realised_search_wdl_frac` = 1.0,
  `targets.value_from_survivor_q` ≥ 98.5% of `targets.rows` (calibrated on SL-base
  collects; a Q50 c12 collect's share is unknown, and a read below it is recorded
  with the share).
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

| Stage | GPU hours (16,000-game default) | 4,000-game alternative (D2) | Basis |
|---|---|---|---|
| Pre-launch checks L1–L4 | ≈ 0 | same | CPU rehearsal and file checks |
| Collect | 4.3–4.9 | 1.1–1.2 | 3,284–3,760 games/h on main's fast path, 128-game runs only |
| Train ×3, incl. snapshot gates | 2.4–4.4 | 0.6–1.1 | 0.47–0.84 s per update (F12 record: 512 updates in 239–432 s) |
| Collect gates (`simple_rl_collect_gates.py`) | 0 (CPU, about 10–15 min per collect) | same | two collects: primary and control |
| Model gates (`simple_rl_model_gates.py`) | 0.2–0.5 | same | a 3,080-row audit pass per snapshot and final model; held-out CE before and after for S (and the P / V cross-check) |
| Primary arena, 3,200 games | 1.8–2.6 | same | 800-game D4 arms took 1,600–2,260 s |
| Screens, 2 × 400 games | 0.5–0.7 | same | |
| Confirmation, 1,600 games | 0.9–1.3 | same | |
| Positive control (collect 4,000 + train + 1,600-game arena) | 2.6–3.6 | same | 1.1–1.2 + 0.6–1.1 + 0.9–1.3 |
| **Total** | **≈ 12.7–18.0** | **≈ 7.7–11.0** | |

The trains may run one after another, or two at a time if memory allows; they are not
measurements. Nothing runs during any arena.

## Pre-launch checks

- **L1.** Main ≥ `885ab7b54` (contains #1427, #1428 and #1430), extensions rebuilt
  (ABI 11). The run executes from a detached worktree at a pinned sha, recorded here
  before launch.
- **L2.** The gate instruments are committed on main (#1430):
  `scripts/simple_rl_collect_gates.py` (G-C1–G-C10, and S's retempered target
  entropy) and `scripts/simple_rl_model_gates.py` (G-T1, and S's G-T3). Their sha256
  at the pinned sha is recorded, with the threshold files above. The seat-balanced
  pooling script, `scripts/arena_seat_balanced_pool.py` (#1434), satisfies the rest of
  L2 once it merges; its sha256 is recorded too. Also:
  - a model-gates read of the SL base on the audit rows reproduces 0.4086 / 0.5133
    (CPU) or 0.4088 / 0.5132 (CUDA) within 0.0005;
  - `simple_rl_model_gates.py --model <posctl run>/gen_0000/model.pt --audit-rows
    <positions.npz> --device cuda` is run on the control's start, and its
    `audit.mean_abs_q` is written into `gt1_control.json` as the FLOOR before the
    control trains.
- **L3.** A tiny-net CPU rehearsal of the exact commands above: source collect, the
  `chmod`, each arm's `collect --collect-from` then `run`; `probe.npz` and
  `presented.npz` sha256 equal across arms; an arm with a different `--seed` and a
  snapshot list past the planned updates are both refused, before `run_config.json`
  is pinned.
- **L4.** Build the confirmation `--openings-fen` file with
  `scripts/arena_openings_minus_screen.py` ([Secondaries](#deciding-yardstick)) and
  record its printed counts and sha256, or record that the fallback applies. L4
  checks the arguments only; the screen-log re-run after the screens, which must
  print the same sha256, checks them against what was played.

The dry run on `f51c51f20` passed L1–L4 (bank above). It predates the commands
added in this revision (steps 0a and 0b, the SNAPS lines, `gt1`, `gt2`, `--z`, the L4
script), each of which was checked separately as stated where it appears.

## Recovery plan

- **Everything goes in a new directory:** `~/chess-artifacts/operations/loop-exp1-20261120/`,
  with `sl_base.pt` (write-protected copy), `openings/`, `collect/` (the source; read-only once an arm references it),
  `arm_P/`, `arm_V/`, `arm_S/`, `posctl/q50_c12.pt` (write-protected copy), `posctl/collect/`, `posctl/arm_V/`, `arenas/`,
  `gates/`.
- **Untouched:**
  - the SL base and Q50 c12 bank files (the run uses write-protected copies);
  - the live checkout;
  - the production recipe and live yaml;
  - every earlier bank (read-only inputs).
- **Nothing is promoted by this experiment,** even on success.
- **On a kill or a failed gate,** collections, gate outputs and arena banks are kept
  for re-analysis (rule 4). Arm models may be deleted by the owner after the verdict
  is recorded.
- **Recovery is deleting the run directory,** after `chmod -R u+w` on it: the
  frozen source run dirs (`collect/`, `posctl/collect/`) are read-only and refuse a
  plain delete.

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
- **The calibration saw part of the registered collect's stream.** The dry run's
  128-game collect used this record's own collect command and seed (2026112000), so
  its games are probably the first 128 of the registered collection's stream. It is
  one of the three references that set G-C5 (0.60–0.78) and G-T4's survivor-Q bound
  (≥ 98.5%). Those are collect-quality checks on untrained data, not a result, so the
  overlap is harmless; it is recorded.
- **G-C5's margin over the defect is thin.** The only banked capped collect at
  concurrency 128 read target entropy 0.799, against the new 0.78 upper bound. G-C5
  is a backstop behind G-C2, G-C3, G-C6 and G-C7, which all fail clearly on that
  collect.

## Second generation (not registered here)

Conditional on a primary success only:
- generation 2 continues from the winning arm with the same recipe and a new
  collection from the continuing opening cursor;
- fresh optimizer;
- the deciding match is generation 2 against generation 1, seat-balanced, with a new
  record written before launch.

A second step of improvement is the owner's stated goal.

## Not verified

- The arm procedure (a standalone `collect --collect-from`, then `run`) on the
  production net. The driver's tests cover it on tiny nets; L3 rehearses the exact
  commands.
- `buffer_full_flushes` in a real collect-gate report on current main.
- The gate scripts' runtimes on a 16,000-game collect (10–15 min is an estimate).
- That `buffer_full_flushes` appears in a real arena result row. The code writes it;
  no run on current main has been read.
- Collect throughput beyond 128-game runs (3,284–3,760 games/h on 128-game runs).
  Graphs-on vs graphs-off identity on this net is not checked (D7: the fast path is
  the only path).
- The probe-drift trajectory at about 5,700 updates.
- That arena seeds draw disjoint EVAL pairs.
- The same-coloured-bishops gap's footprint in this collection (0 in the banked 4,608 games).
