# RL: value-side follow-ups after deep/fast phase B (pre-registration; Step 1 launch-ready, Steps 2–3 design only)

Date: 2026-11-27. Status: registered. Step 1 launches once deep/fast phase B has finished (its verdict is expected at
about 16:00 EST today) and the coordinator writes the GO file. Steps 2 and 3 get their launchers in an addendum after
Step 1's record is reviewed. Parent of every arm: `kC_baked.pt`, sha256 `c48a7b2c1cc734a7…`.

## Why

The owner decided the following today:
- **Effect size to hunt:** about 15 Elo on the **value** side at 64 sims.
- **Policy already learns but does not carry to search:** +75 to 90 Elo at 1 sim, and the gain vanishes at 64 sims.
- **The pilot showed value is the lever, either way:** a trained value head with the base's everything-else scored
  −21.4, and blend plus recalibration scored +9.6 (the [RL run 1 pilot results](2026-11-21-rl-run1-pilot-results.md);
  the [evidence log](2026-11-21-rl-loop-evidence-log.md) has every arm).
- **Screen size:** the standard screen is now 3,200 games (about ±9 Elo). Extend to 12,800 only if the point lands in
  about [+5, +15].
- **Sim counts:** new nets are read at 1 and 16 sims as well as 64, because those reads are cheap.
- **Arm discipline:** one variable per arm. Banked rows are reused, and each row is used once per arm.

## What "value from X, policy from Y" means here

Each checkpoint is a whole network: one trunk feeding a policy head and a value head. After training, C, D and kC
share no weights; the trunk is the same architecture, but not the same tensors. So "kC's policy with C's value" is not
one trunk with two heads. It is two complete networks evaluated on the same input batch:
- kC's network supplies the policy logits.
- C's network, with its own trunk, supplies the WDL logits.

The search uses those two outputs together. That is what
`chess_anti_engine/eval/hybrid_value.py` (`HybridValueNet`) builds, through `scripts/arena_standard.py`
`--candidate-value-from` / `--reference-value-from`. **It costs a second full forward per evaluation.**

The earlier head-swap ran the same way: Q50 and F12 against the SL base, in the evidence log's D4 rows. There, the
hybrid arms took 2.4–3.5× the wall time of a plain arm without CUDA graphs. `hybrid_value.py` is the in-repo
replacement for that bank's monkeypatching `headswap_arena.py`.

Two consequences:
- A head-swap arm measures the value net **including its trunk**, read through kC's search prior. Any change that
  training made to C's own policy cannot reach the result.
- Seat identity controls passed in that bank: hybrid(base, base) reproduced plain base game for game, and the
  win/loss and policy-permutation mutants scored 0/40.

`C_baked.pt` and `D_baked.pt` already carry phase B's recalibration baked into
`value_wdl.net.2.{weight,bias}` (`C_baked.pt.json`: T 0.7214, b −0.2336; D: T 0.7026, b +0.0422). So the hybrids use
the recalibrated value with no `--*-value-recal`.

## Step 1: phase B's value nets under the parent's policy (no training)

**Question.** Do D's sparse rows from 45,718 games teach value better than C's dense rows from 27,955 games?

**Arms.**
- `hC`: policy `kC_baked.pt` + value `C_baked.pt` (`e1d62064…`).
- `hD`: policy `kC_baked.pt` + value `D_baked.pt` (`fb869143…`).
- Each plays against `kC_baked.pt` in two seats over the same openings:
  - seat A: the hybrid is the candidate;
  - seat B: the hybrid is the reference.

**Control.** `kC_baked.pt` itself, which is also the policy of both arms. Between the arms, the only difference is
which phase B net supplies the WDL.

**Settings.** These are phase B's match settings at phase B's pin `d08b03adb` (the code every phase B match ran):
- `matched_sims`, search shape `training`, no Gumbel noise, compile off;
- the pilot's `eval.pgn`, 16 opening plies, max plies 600 with abort on reaching them;
- the production Syzygy pair, 128 concurrent games.

Look 1 is 1,600 games per seat (3,200 per arm) at seed 2026112721. Both arms use the same seed, so pair id p is the
same opening in all four runs. `contrast_v.py` refuses the pairing otherwise.

**The ONE deciding yardstick:**
`python3 ~/chess-artifacts/operations/rl-value-followup/contrast_v.py 64 <seed>:<hC64_A>:<hC64_B>:<hD64_A>:<hD64_B> --out step1/contrast64_look1.json`.
The launcher runs it as step `read64_1`, with the four look-1 game logs under `step1/arenas/hC64|hD64/`.

What it computes:
- Δ = E_D − E_C, the paired difference of the two arms' seat-balanced per-game excess over kC, per opening key.
- The SE is clustered by opening. The result is in Elo through the slope at 0.5 (694.87 Elo per unit excess), with a
  95 % interval.

The per-arm readings are reported beside it: each arm's E against kC, from both `contrast_v.py` and the official
`scripts/arena_seat_balanced_pool.py --arm kC_baked.pt`.

**Pre-committed rule** (`contrast_v.py` `rule()`, fixture-tested):

| Reading | Look 1 (3,200 per arm) | Final (12,800 per arm) |
|---|---|---|
| Δ | \|Δ\| < 5: stop; read as "not resolved at 3,200 games" and quote the interval (about ±9 Elo or wider, so a 10 to 14 Elo difference is not ruled out). Δ ≥ +15 with LB > 0: D's rows teach value better. Δ ≤ −15 with UB < 0: C's rows teach value better. Anything else: extend. | LB > 0: D better. UB < 0: C better. Otherwise: no detectable difference. |
| Each arm's E vs kC | ≥ +15 with LB > 0: value gain over the parent. ≤ −15 with UB < 0: value loss. In [+5, +15): extend. Otherwise: stop; read as "no gain resolved at 3,200 games" and quote the interval. | LB > 0: gain. UB < 0: loss. Otherwise: no detectable difference. |

Extension is all or nothing. Look 2 is played for **both** arms iff the Δ rule or either arm's rule says extend. Look 2
is 4,800 games per seat at seed 2026112722, so 12,800 per arm in all. The final reading pools looks 1 and 2.

The 16-sim reads (1,600 per seat, seed 2026112721) are reported only.

**There is no 1-sim read for a hybrid.** At 1 sim, the move comes from kC's prior, which is the same network as the
control. The one way a value could enter at 1 sim is through the completed-Q of a single visited child, and I could not
check that without torch. The 1-sim reads belong to Steps 2–3, where the nets are new.

**Why the yardstick is Δ and not each arm vs kC.** The question compares the two collections. Both arms share kC's
policy, the openings and the move RNG seed, so the paired Δ removes most of the shared variance that two separate
readings would carry. The per-arm readings say whether either value beats the parent at all.

**GPU-hours.** Measured basis: phase B's `C_vs_kC` seats took 4,587 s and 4,534 s per 6,400 games at 64 sims, so a
plain 3,200 games takes 0.63 h. The hybrid adds a second forward; I assumed 1.5–2× (CUDA graphs on), not measured.

| Read | Estimate |
|---|---|
| 64 sims, 2 arms × 3,200 | ≈ 1.9–2.5 h |
| 16 sims, 2 arms | ≈ 0.6–0.8 h |
| Smoke | < 0.1 h |
| **Step 1 total** | **≈ 2.6–3.4 h** |
| Plus the extension if triggered (2 × 9,600 at 64 sims) | ≈ 5.7–7.6 h more |

**Recovery.** Nothing trains, and no live state, banked file or yaml changes. To stop:
`kill -TERM $(cat ~/chess-artifacts/operations/rl-value-followup/step1/step1.pid)`. A relaunch skips finished steps and
DONE seats and never re-rolls a seat. A PARTIAL seat stops the launcher (87). The owner moves that seat's four files
into a new `aborted_*` directory and relaunches.

### Step 1 bank and launcher

Bank: `~/chess-artifacts/operations/rl-value-followup/`, listed in
`~/chess-artifacts/operations/rl-value-followup/SHA256SUMS` [`56a83cba`]. The launcher pins that file's sha256 and the
guard's. Files:

- **`~/chess-artifacts/operations/rl-value-followup/step1/launch_step1.sh` [`aab7bec6`].** Ported from deep/fast phase
  B's `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/launch_phase_b.sh` [`12145ed8`] (P4 rev 2), keeping its refusals and exit codes:

  | Exit | Refusal |
  |---|---|
  | 90 | no `step1/STEP1_GO` |
  | 94 | `CUDA_VISIBLE_DEVICES` empty |
  | 93 | a pinned file differs |
  | 92 | `kC_baked.pt`, `C_baked.pt` or `D_baked.pt` differs |
  | 91 | phase B not finished: `phaseb/exit` and `phaseb/m_D.exit` must both read 0 |
  | 96/97 | the pin worktree |
  | 81 | a HOLD/STOP file |
  | 84 | the inputs changed |
  | 86/87 | a STALE or PARTIAL seat |
  | 8/9 | the guard |

  It writes its own pid to `step1/step1.pid` past the refusals, and its exit marker to `step1/exit`.
- **`~/chess-artifacts/operations/rl-value-followup/gpu_guard_v.sh` [`279267b0`].** Phase B's guard with three
  changes:
  - the stage is `vstep1`, released by a line `vstep1 idle_mib=<MiB> <ISO time>` in
    the bank's `GPU_RELEASED_BY_OPERATOR` file, which the coordinator creates;
  - the bank is this one (with the pinned `gpu_scan.sh` v7 copied here, `d1aee351`);
  - it **also takes the deep/fast bank's launch lock, opened read-only**, so it refuses while any deep/fast launcher
    runs.
- **`~/chess-artifacts/operations/rl-value-followup/run_v_match.sh` [`54b13133`].** Phase B's `run_df_match.sh` with
  two changes: sims as an argument (64 or 16 only), and seat args that may carry `--<seat>-value-from`.
- **`~/chess-artifacts/operations/rl-value-followup/seat_state_v.py` [`fd5a0e8c`].** Phase B's `seat_state.py` plus the
  hybrid's value block. Both seats load `kC_baked.pt`, so the policy shas alone cannot tell the hybrid from the parent.
  DONE therefore also requires the row's `candidate_value` / `reference_value` to name the value-from file's current
  sha256 with recalibration null, and a plain seat to carry no value block.
- **`~/chess-artifacts/operations/rl-value-followup/contrast_v.py` [`2be3fee9`].** Stdlib only.

**Steps:**
1. `smoke`: 4 games per seat per hybrid, under `step1/smoke`.
2. `h64_C`, `h64_D`: look 1.
3. `read64_1`: the per-arm pools plus `contrast64_look1.json`.
4. `x64_C`, `x64_D`, `read64_2`: only if `read64_1` says extend.
5. `h16_C`, `h16_D`, `read16`.
6. `step1/STEP1_DONE`.

**Launch** (the coordinator, after phase B's `phaseb/exit` reads 0, this record is merged, `step1/STEP1_GO` is written
and the `vstep1` release line is added). First check:
`sha256sum ~/chess-artifacts/operations/rl-value-followup/step1/launch_step1.sh` must print
`aab7bec6fe0034f5fc9f5d7b0b1309e397e91633857749cf947eaab15eaf152f`. Then:
`V=~/chess-artifacts/operations/rl-value-followup/step1; setsid nohup bash $V/launch_step1.sh >> $V/step1.log 2>&1 < /dev/null &`.
Stop: `kill -TERM $(cat $V/step1.pid)`, then check `ps` for leftover arena processes whose command line names this bank.

**Tests so far:**
- **Torch-free only**, because phase B holds the GPU and any torch import holds `/dev/dxg`.
- **`~/chess-artifacts/operations/rl-value-followup/step1/test/contrast_fixtures.py` [`f4173717`]: 14 of 14 pass.**
  - Synthetic logs with known excesses: E_C 0, E_D +0.25, Δ +173.7 Elo, and the sign flips when the arms are swapped.
  - Refusals: an incomplete pair, a different opening map, a wrong seed, a sims mismatch.
  - Every band of the rule.
- **`~/chess-artifacts/operations/rl-value-followup/step1/test/refusals_torchfree.sh` [`ffd712e1`]: 6 of 6 pass.** Fake
  HOME with a copy of the bank: no GO (90), `CUDA_VISIBLE_DEVICES` empty (94), phase B unfinished and phase B exit 1
  (91), a tampered bank file (93), a HOLD file (81).

**Not verifiable without torch, to be checked by the one unmutated pass once the coordinator frees the GPU (no mutant
series):**
- `arena_seat_balanced_pool.py --arm kC_baked.pt` accepts a hybrid arm whose policy file is the base's own. Its
  hybrid-seat check should pass (A's `candidate_value` is B's `reference_value`), but it imports torch through
  `chess_anti_engine.eval`.
- The result rows carry the `candidate_value` / `reference_value` blocks that `seat_state_v.py` reads.
- The second forward's real wall time.

The registered form of that pass is the launcher's own `smoke` step on the real nets. It is the first step, so a
wiring failure stops Step 1 before any look is spent. I did not build a fake-arena harness. Phase B's harness fakes the
GPU steps and so would not exercise the hybrid wiring that is in question here.

## Step 2: the value target (design; launcher in a later addendum)

Training uses banked `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/pool_C`: 27,955 games, 2,648,696 drawn
rows, 5,174 updates. Each arm trains from `kC_baked.pt` at the pin with phase B's `train_C` flags byte for byte except
the value weights. That keeps the same pool, seed 2026112703, rows, order, optimizer and lr, so each row is used once
per arm.

**Arms:**
- `vO`: outcome only, `--value-outcome-weight 1 --value-search-weight 0`.
- `vS`: search value only, `0 / 1`, with `--value-search-source stored_search_wdl` as in phase B.

**Control:** the current blend, 0.5 / 0.5. This is phase B's C net, reused, not retrained.

Each arm then gets phase B's fits on C's pool collect, the bounds and the bake, so all three are recalibrated alike;
the pilot showed raw value training loses and recalibration rescues it. Arms are **read through the Step 1 head-swap**:
kC policy + the arm's value, against kC, at seed 2026112721. That makes each arm paired with Step 1's `hC` games on the
same openings, so `hC` is reused as the control's reading. `contrast_v.py` gets arm names in Step 2's addendum.

**Deciding yardstick:** Δ_arm = E_arm − E_hC at 64 sims, 3,200 games per arm, with the Step 1 bands; extension to
12,800 only from [+5, +15]. The winner is the arm with Δ LB > 0 at its final reading. If neither clears that, the blend
stays and Step 3 runs on it.

**Reported, not deciding:**
- the same hybrid at 16 sims;
- each arm's own full net (its own policy) at 1, 16 and 64 sims vs kC, 3,200 games each.

**Can the trainer at main hold the policy fixed while value trains? No, not exactly.**
- `--w-policy` exists (`scripts/simple_rl_loop.py:831`, default 1.0; `--w-wdl` 0 = policy only). `--w-policy 0` removes
  only the policy loss. The trunk is shared, so value-loss gradients still move the trunk and with it the policy
  output.
- Under the default matrix scope, the policy head sits in the AdamW aux groups (2/3) with weight decay. So group 2's
  decay would still shrink its weights with no gradient. That group assignment is read from
  `chess_anti_engine/train/trainer.py` `_matrix_optimizer_filter`; it is not checked on the real net, which needs
  torch.
- `--kl-policy` (`scripts/simple_rl_loop.py:845`) anchors the policy to a frozen reference softly; it does not hold it.

The change needed for an exact hold, **not written in this PR**:
- a flag, for example `--train-only-prefix value_wdl.`, that calls `p.requires_grad_(name.startswith(prefix))` on every
  parameter before the trainer builds its optimizer;
- `_split_decay_groups` (`trainer.py:1180`) already skips parameters without `requires_grad`, so frozen parameters get
  neither gradient nor decay;
- plus a manifest field and a take-effect check that policy logits are bitwise unchanged after training.

That would train the value head alone on a frozen trunk, which is a different and smaller capacity than the arms
above. **So the arms are read through the head-swap instead**, where the policy is kC's by construction.

**GPU-hours (estimate):**

| Item | Estimate |
|---|---|
| Training per arm (phase B's `train_C` wall: 1,437 s) | 0.4 h |
| Fits and bake per arm | < 0.2 h |
| Hybrid read, 64 sims, 3,200 games | 1.0–1.3 h |
| Hybrid read, 16 sims | 0.3–0.4 h |
| Full net at 64 / 16 / 1 sims | ≈ 0.63 / 0.2 / 0.05 h |
| **Per arm** | **≈ 2.8–3.2 h** |
| **Two arms** | **≈ 5.6–6.4 h** |
| Plus extension if triggered | up to ≈ 7.6 h |

## Step 3: the value-side learning rate (design, on the Step 2 winner)

**What the realised optimizer looked like** (phase B `opt_C.json`, kC's saved groups): group 0 at 6e-5 (Aurora, the
trunk's block matrices; base from the init checkpoint, closed form) and groups 1–3 at `--lr` 3e-6 (flat). Under the
default matrix scope, the value head's own weights (`value_wdl.*`) are in groups 2/3 with the policy heads, embeddings
and norms. The trunk, which carries most of what value learns, is in group 0.

**Proposed arms (one variable: the groups-1–3 lr, value weights as the Step 2 winner):**
- `lr1e-5`: `--lr 1e-5 --lr-floor 1e-5 --matrix-lr-multiplier 6`, about 3.3× the current value.
- `lr3e-5`: `--lr 3e-5 --lr-floor 3e-5 --matrix-lr-multiplier 2`, 10×, half of group 0's 6e-5.

**Group 0 does not stay fixed by itself (corrected after the Codex inline finding on PR #1551).** `init-checkpoint`
imports the checkpoint's matrix/aux *ratio* (20), not group 0's absolute rate (`simple_rl/learn.py`,
`matrix_lr_multiplier = mat lr / aux lr`), and the trainer sets `matrix_lr = lr × matrix_lr_multiplier`
(`train/trainer.py`). Without compensation the two arms would run the trunk at 2e-4 and 6e-4, a second variable. The
explicit multipliers 6 and 2 hold group 0 at 6e-5. Before either arm's match, the realised rate of every optimizer
group is read from the trained run (Phase B's `opt_check.py` pattern) and the arm is void if group 0 is not 6e-5.

The policy heads share groups 2/3, so their lr moves too. Under the head-swap reading the policy is kC's, so that
cannot reach the result.

**Before Step 3 launches:** a torch one-liner listing which group holds `value_wdl.*` on `kC_baked.pt`, so the arm
really moves the value head.

Same yardstick and bands as Step 2, against the Step 2 winner's reading. GPU-hours as Step 2: ≈ 5.6–6.4 h, plus
extension.

## Confounds (named)

- **Step 1: D's value net also trained on D's own policy targets.** Both arms trained policy and value jointly. The
  head-swap removes the policy head's effect but not how policy training shaped the trunk that the value reads from.
  That is part of "what the rows taught the value net", and it is stated, not removed.
- **Steps 2–3 reuse pool_C, the rows phase B's C used.** "One use per row" holds per arm. Across arms the same rows are
  reused by design, so arm contrasts are paired.
- **Search-config freeze:** every arena runs phase B's settings at its pin. Nothing in this record changes search.

## Recovery

None of the steps changes live training, the live yaml or any banked file. Each step's outputs are new directories
under the new bank. Stop with the pid file. Read-only inputs: phase B's nets and pool, the pilot's openings, and
`kC_baked.pt`.

**Review notes (2026-11-27, non-author review on PR #1551).** (1) The look-1 stop bands above are decided on the point
estimate; they are stopping rules, not evidence of no effect, and every readout quotes the interval. (2)
`launch_step1.sh` and `run_v_match.sh` read an environment override `VF_PY` for the interpreter. It is not part of the
recorded inputs, the launch command does not set it, and the launcher must be run with `VF_PY` unset; a later revision
of the bank drops it. (3) At look 1 an arm with E ≥ +15 whose lower bound is ≤ 0 is labelled "no gain" by
`contrast_v.py` and not extended; at about ±9 Elo this is not reachable in practice, and such a reading would be
reported as unresolved. The reviewer settled two items the author could not check without torch, from source at
`d08b03adb`: result rows carry `candidate_value` / `reference_value` for a hybrid seat (`scripts/arena_standard.py`),
and `arena_seat_balanced_pool.py` accepts an arm whose policy file is the parent's.


## Steps 2–3 launch addendum (2026-11-27)

The launchers for Steps 2 and 3 were written while Step 1 runs. Nothing below has been launched.
- **Step 2** launches after `step1/STEP1_DONE` exists, this addendum is reviewed and merged, `step2/STEP2_GO` is written
  and a `vstep2 idle_mib=<MiB> <ISO time>` line is in the bank's `GPU_RELEASED_BY_OPERATOR`.
- **Step 3** launches after `step2/STEP2_DONE` and `step2/winner.txt` exist, plus `step3/STEP3_GO` and a `vstep3` line.

### Files

Step 1's `SHA256SUMS` and every file it pins are unchanged (byte for byte; its launcher pins that file by hash). The
Steps 2–3 files are listed in their own `~/chess-artifacts/operations/rl-value-followup/SHA256SUMS.step23` [`e0ed8638`],
which both launchers pin by hash, together with the guard's hash. It lists:
- **`common23/lib23.sh`**: the shared step machinery, the train / opt-check / fits / bounds / bake sequence of phase B's
  `stage2b_lib.sh`, and Step 1's seat / match / read pattern.
- **Copied byte for byte** from phase B's bank: `common23/bake_recal.py` [`786bb537`], `common23/recal_bounds.py`
  [`1a50be12`], `common23/planned_updates.py` [`208995af`].
- **Derived, each change stated in its header:**
  - `common23/opt_check_v.py`: phase B's `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/opt_check.py` [`4f8b306f`] with three changes. (1) The realised search-WDL
    fraction is an argument (0.0 / 1.0 / 0.5) instead of the fixed 0.5. (2) The matrix multiplier is an argument
    (`ckpt` keeps the original expression; a number gives base0 = lr × MULT). A new check requires base0 within 1e-6
    of 6e-5. (3) `lr_floor`, `peak_lr` and `lr_schedule.eta_min` / `group_eta_min` (the scheduler's floor,
    `simple_rl/learn.py` `realised_optimizer`) leave the non-LR comparison. A positive check then requires
    `eta_min` = the arm's `--lr-floor` and `group_eta_min` null. (Review fix on #1553: the first version left
    `eta_min` in, so Step 3's two arms, whose floors differ by design, would have failed the comparison on a correct run.)
  - `common23/train_smoke_v.py`: phase B's `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/train_mem_smoke.py` [`2ce38774`] with one change: every update's record also
    carries each optimizer group's realised lr.
  - `common23/run_v2_match.sh`: Step 1's `run_v_match.sh` with 1 sim also registered.
  - `common23/gpu_guard_v23.sh`: Step 1's guard with stages `vstep2` / `vstep3`. It uses the same bank lock and the same
    read-only deep/fast lock.
  - `common23/contrast_v2.py`: named arms against a control. It imports Step 1's `contrast_v.py` (estimator unchanged).
    It also refuses, after the #1553 review:
    - any run of a look whose header settings differ from the look's other runs, apart from net paths;
    - any run whose results row puts the arm on the wrong side. In the A run the arm must be the candidate and
      `kC_baked.pt` the reference; in the B run, the reverse. A head-swap arm's side carries its value block.
    It reports the realised arm-control correlation.
- **Step 1's own files, read through their pinned bytes:** `seat_state_v.py`, `contrast_v.py`, `gpu_scan.sh`.

Phase B's `stage2_flags.sh` (`043b4810…`), `plan_C.json` and `pool_C` manifest are read in place, pinned by sha256 in
the launchers, never copied or written. So are `kC_baked.pt`, `C_baked.pt` and `scripts/fit_value_recal.py` at the pin
(`37e27420…`).

**Why opt_check and the smoke are derived rather than copied.** Copied unchanged, phase B's `opt_check.py` would fail
every Step 2 arm: it requires a search fraction of 0.5. It would also fail every Step 3 arm: it expects
base0 = lr × 20 and compares `lr_floor` as a non-LR field. And phase B's smoke does not record the per-group lr that
Step 3's void rule reads. `opt_check_v.py` with `0.5 ckpt` reproduces phase B's verdict on phase B's own C and D train
dirs (all seven checks true, `ok` true). With `0.0` it fails them, as it must.

### What each launcher runs

| Launcher | sha256 |
|---|---|
| `~/chess-artifacts/operations/rl-value-followup/step2/launch_step2.sh` [`1a00398c`] | `1a00398c8ac202973179e8e1e78d1076edae97fb6f5d8e13b6d25dc0055db5fb` |
| `~/chess-artifacts/operations/rl-value-followup/step3/launch_step3.sh` [`5ffac6f0`] | `5ffac6f0494714e9327a1e01b467ab2e1281711ce5214e2b79eb781b9cb137b9` |

**Training.** Each arm trains from `kC_baked.pt` at pin `d08b03adb` with:
- `--collect-from` phase B's `pool_C` (read only);
- phase B's collect flags (arm C, seed 2026112703, the pool's games);
- phase B's `TRAIN`, then the arm's flags (a later flag wins);
- `--snapshot-updates 2587,5174` (phase B's `plan_C.json`).

The arms:

| Step | Arm | Flags |
|---|---|---|
| 2 | `O` | `--value-outcome-weight 1 --value-search-weight 0 --value-search-source stored_search_wdl` |
| 2 | `S` | `0 / 1`, same source |
| 3 | `L1` | the Step 2 winner's value mix + `--lr 1e-5 --lr-floor 1e-5 --matrix-lr-multiplier 6` |
| 3 | `L3` | the Step 2 winner's value mix + `--lr 3e-5 --lr-floor 3e-5 --matrix-lr-multiplier 2` |

For Step 3, `step2/winner.txt` names the winner: `O`, `S`, or none, in which case the mix is the 0.5 / 0.5 blend.

Per arm, in order:
1. **The 3-update GPU memory smoke**, with the arm's exact flags. ABORT above 0.85 × device memory writes `HOLD_MEM`.
   Then the **realised-lr check after update 1**: group 0 within 1e-6 (relative) of 6e-5, and groups 1–3 at the arm's
   `--lr`. A miss writes `HOLD_LR_<arm>`, exits non-zero, and the arm is **void**: it is not trained and plays no
   match. In Step 2 the same check confirms phase B's optimizer: 6e-5 and 3e-6.
2. **Training** (host-RAM floor 86 GiB, phase B's floor for this pool), then the planned-updates check: realised =
   planned = 5,174, snapshots as registered.
3. **`opt_check_v.py`, right after that arm trains and before the next arm trains.** The non-LR reference is phase B's
   `arm_C` train dir (its `metrics.json` is pinned by sha256 in the launchers). The realised search fraction must be the
   arm's, base0 must be 6e-5, and `eta_min` must be the arm's floor. A failure writes `HOLD_TRAIN_CHECK` before the
   second arm spends GPU time. (The first version compared the two arms with each other, so it could only fail after
   both had trained.)
4. **Phase B's fits** on C's pool collect against `kC_baked.pt`: snapshots 2587 and 5174, plus the final net.
5. **Phase B's bounds** (`recal_bounds.py`: R1 box and R2). A miss writes `HOLD_RECAL_<arm>`; the owner decides.
6. **Phase B's bake and verify.**

**Readings.** These are Step 1's settings at the pin. Seeds: look 1 2026112721, look 2 2026112722.
- A 4-game-per-seat smoke per arm.
- **Head-swap** (`kC` policy + the arm's baked value vs `kC_baked.pt`): 1,600 games per seat (3,200 per arm) at 64
  sims; 4,800 more per seat at look 2 if the rule extends; 1,600 per seat at 16 sims.
- **Each arm's own full net vs `kC_baked.pt`** at 64, 16 and 1 sims, 1,600 per seat each. These are reported only.

**The control's games are read, never replayed.**
- Step 2's control is Step 1's `hC` (`step1/arenas/hC64`, `hC16`).
- Step 3's control is the Step 2 winner's head-swap (`step2/arenas/h<O|S>64`, `h<O|S>16`), or `hC` when Step 2 has no
  winner.
- If the rule extends and no step has played the control's look 2 (seed 2026112722, 4,800 per seat), the extending step
  plays it once, as step `c64_2`, into its own directory. That is a new reading, not a replay. Every later step reads
  it from there.

### The deciding yardstick and the rule (Steps 2 and 3)

**Yardstick:** step `read64_1` runs, for Step 2 (full paths; `B=~/chess-artifacts/operations/rl-value-followup`):
`python3 $B/common23/contrast_v2.py 64 --out $B/step2/contrast64_looks1.json --base ~/chess-artifacts/operations/rl-run1-volume/kC_baked.pt --control C --look 2026112721 C=$B/step1/arenas/hC64/hC64_1_A.games.jsonl,$B/step1/arenas/hC64/hC64_1_B.games.jsonl O=$B/step2/arenas/hO64/hO64_1_A.games.jsonl,$B/step2/arenas/hO64/hO64_1_B.games.jsonl S=$B/step2/arenas/hS64/hS64_1_A.games.jsonl,$B/step2/arenas/hS64/hS64_1_B.games.jsonl`.
Step 3 is the same with `step3`, arms `L1` and `L3`, and the control named above.

For each arm X, Δ_X = E_X − E_control. It is paired by opening key, uses Step 1's seat-balanced estimator unchanged,
has a clustered SE, and is reported in Elo through the slope at 0.5 with a 95 % interval. Each arm's own E vs kC is
reported beside it, and so is the realised arm-control correlation (phase B's C-vs-D pairing: +0.046).

**Rule** (`contrast_v2.py` `rule()`), per arm:

| Reading | Look 1 (3,200 games per arm) | Final (12,800 per arm) |
|---|---|---|
| Δ_X | \|Δ\| < 5: stop; "not resolved at 3,200 games; quote the interval". Δ ≥ +15 with LB > 0: better than the control. Δ ≤ −15 with UB < 0: worse. Anything else (\|Δ\| in [5, 15), or \|Δ\| ≥ 15 with an interval that includes 0): extend. | LB > 0: better. UB < 0: worse. Otherwise: "not resolved at 12,800 games; quote the interval". |
| Own E vs kC | ≥ +15 with LB > 0: gain. ≤ −15 with UB < 0: loss. Any other E ≥ +5, including ≥ +15 with LB ≤ 0: extend. Otherwise: "not resolved at 3,200 games; quote the interval". | LB > 0 gain / UB < 0 loss, else "not resolved at 12,800 games". |

The labels are stopping labels, not findings of no effect. At 3,200 games the paired Δ cannot exclude about 17 Elo (see
the power table), and every readout quotes the interval. **This extension rule supersedes the Step 2 design text's
"extension to 12,800 only from [+5, +15]".**

- **Extension is all or nothing:** if look 1 extends anything, every arm and the control get look 2, and the final
  reading pools both looks.
- **Winner:** among arms whose final Δ says "better than the control", the one with the largest Δ. Otherwise "none (the
  control stays)". It is written to `step<N>/winner.txt`.
- The 16-sim and full-net readings decide nothing.

**Power of the paired contrast.** Measured on phase B's logs, subsampled to 800 pairs per seat for 3,200 games, with
arm-control ρ = +0.05 (from the statistics review of #1553):

| Games per arm | Per-arm half-width | Paired Δ half-width | MDE at 80 % power, two-sided 5 % |
|---|---|---|---|
| 3,200 | ±8.5 Elo | ±11.7 Elo | ≈ 17 Elo |
| 12,800 | ±4.3 Elo | ±5.9 Elo | ≈ 8.5 Elo |

The head-swap arms share kC's policy, so ρ may be higher and the band narrower. Each readout reports the realised ρ.

**Selection.** Two arms and up to two looks. Under the global null (no arm differs from the control), simulation in the
review gives:
- a family-wise false-winner rate of about 3–5 %;
- a false winner's reported Δ averaging +7 to +8 Elo;
- a winner's Δ that is biased upward in general: one arm truly +8 wins 69 % of the time with a reported +9.7.

A Step 2 or Step 3 winner's Δ is quoted with that caveat.

### Confounds (Steps 2–3)

- **Step 3's control is a selected arm's banked games.** The Step 2 winner was chosen because its Δ against hC was
  high, so its banked E carries positive noise. Step 3's Δ = E_L − E_winner is biased low:
  - by about +3.8 Elo if the winner's true Δ is 0;
  - by about +0.7 at a true +8;
  - by about 0 at a true +15.

  A Step 3 "better than the control" stays conservative. **A Step 3 "worse than the control" is anti-conservative and
  is NOT read as "a higher learning rate harms".** Each Step 3 readout quotes the control's E beside its Step 2 value.
  The bias-free alternative would replay the control at a fresh seed pair, about 0.95 h per look; it is not
  registered.
- **S (search only) falls back to the outcome target** on any row without a stored search WDL
  (`train/value_blend_guard.py` `search_effective_frac`). The trainer logs that share but does not record it in
  `metrics.json`. Each Step 2 readout therefore reports pool_C's effective search share per arm: the share of drawn
  rows carrying a stored search WDL, computed on the CPU from pool_C's shards. It is not measured yet.
- **Step 3's lr also moves the policy heads** (groups 2/3). That cannot reach the head-swap reading, where the policy
  is kC's, but it does reach the full-net readings.
- **The hC control games were played earlier**, in Step 1, on the same pin, settings and seeds. The same holds for a
  Step 2 winner's games used by Step 3.

### Interpreter override

The launchers read `VF_PY` (interpreter) and `VF_TEST_MEMINFO` (meminfo path) for the harness only. **Both must be
unset at a real launch.** The launchers refuse with 82 if either is set and the bank has no `HARNESS_FAKE_HOME` marker.
Only the harness's fake bank carries that marker.

### GPU-hours

Measured basis:

| Item | Time |
|---|---|
| Phase B smoke | 5.5 min |
| Phase B `train_C` | 33 min |
| Phase B fits, both arms | 31 min |
| Phase B bake | 3 min |
| Step 1's live hybrid at 64 sims | ≈ 3,370 games/h, so 3,200 games ≈ 0.95 h, 1.5× a plain arm |

Per step (two arms):

| Item | Estimate |
|---|---|
| Smokes | 0.2 h |
| Training | 1.1 h |
| Fits and bake | 0.6 h |
| Hybrid 64, look 1 | 1.9 h |
| Hybrid 16 | ≈ 0.6 h (estimate) |
| Full nets at 64 / 16 / 1 sims | ≈ 1.8 h |
| **Total per step** | **≈ 6.2 h** |
| Extension: two arms' look 2, plus the control's if missing (3 × 9,600 hybrid games) | up to ≈ 8.6 h more |

### Launch and stop

**Step 2:**
1. `sha256sum ~/chess-artifacts/operations/rl-value-followup/step2/launch_step2.sh` must print the value above.
2. `V=~/chess-artifacts/operations/rl-value-followup/step2; setsid nohup bash $V/launch_step2.sh >> $V/step2.log 2>&1 < /dev/null &`
3. Stop: `kill -TERM $(cat $V/step2.pid)`, then check `ps` for leftover arena or trainer processes whose command line
   names this bank.

Step 3 is the same with `step3`.

A relaunch skips every step whose exit reads 0, and every DONE seat. A failed match step reruns with its DONE seats
skipped and never re-rolled. A PARTIAL seat (87), a STALE seat (86), or any `HOLD_*` / `STOP` file (81) stops the
launcher for the owner.

### Harness

**`~/chess-artifacts/operations/rl-value-followup/step2/test/harness_step23.sh` [`2695d3e0`]: 25 of 25 pass**
(`~/chess-artifacts/operations/rl-value-followup/step2/test/harness_step23.out` [`67ad9ea8`]).

**The review blocker reproduced, then fixed.** The fake trainer now writes `lr_schedule.eta_min` = the run's floor. With
the old `opt_check_v.py` the harness failed 17 of 21: Step 3 `HOLD_TRAIN_CHECK`, `non_lr_fields_equal_reference`
false (`~/chess-artifacts/operations/rl-value-followup/step2/test/harness_step23.old_strip.out` [`0d43e896`], run on
the previous launcher structure). With the fix it passes.

Setup:
- Fake HOME with a copy of the bank, and a fake finished Step 1 (hC at 64 and 16 sims, played through the real
  `run_v2_match.sh`).
- Read-only symlinks to phase B's flags, plan, pool manifest and `C_baked.pt`, to `kC_baked.pt`, and to the pin
  worktree.
- A stub interpreter that runs the stdlib tools for real (`contrast_v2.py`, `contrast_v.py`, `seat_state_v.py`,
  `recal_bounds.py`, `opt_check_v.py`, and every inline python of the launchers) and
  `~/chess-artifacts/operations/rl-value-followup/step2/test/fake23.py` [`aacf683b`] in place of each torch call:
  smoke, trainer, plan, fit, bake, arena.
- The fake trainer writes phase B's real `arm_C` `metrics.json` / `manifest.json` with these fields set from the run's
  flags: settings `lr`, `lr_floor`, `value_outcome_weight` / `value_search_weight`; `realised_optimizer` `base_lrs`,
  `param_group_lr`, every table `lr`, `lr_floor`, `peak_lr`, `lr_schedule.eta_min`; summary
  `realised_lr_param_groups`, `realised_search_wdl_frac`, `snapshot_updates`. Every other field is phase B's.
- The fake arena biases scores for the sides whose net name carries a registered bias, so the rule has something to
  decide.
- No torch; `CUDA_VISIBLE_DEVICES` is empty for every python.
- **One stated difference from a real launch:** the fake bank's guard is a stub, and the copy's two pinned constants
  are rewritten to the stub's hashes. The real guard's scan would refuse while Step 1's arena holds the GPU.

What it proves:
- **Step 2 refusals:** no GO (90), `CUDA_VISIBLE_DEVICES` empty (94), Step 1 not done (91), a tampered `common23`
  file (93), a HOLD file (81), a test-only variable (82), `VF_PY` set outside the harness (82).
- **`contrast_v2.py` hardening:** the registered control and arm are accepted; a control whose header settings differ
  (temperature) is refused; a seat-swapped control (B passed as A) is refused.
- **A full Step 2 pass:**
  - `hC` look 1 is read, not replayed.
  - Look 1 extended, so the control's look 2 was played once in `step2` and the final reading pooled both looks.
  - Realised search fractions are 0.0 / 1.0, and both opt checks are ok.
  - Full nets are read at 64 / 16 / 1 sims; the winner file, the pid file and the exit marker are written.
- **A Step 2 relaunch** that trains and plays nothing.
- **Step 3:** no GO (90); Step 2 not done (91).
- **The void:** group 0 at 1.1 × 6e-5 after update 1 gives a non-zero exit, `HOLD_LR_L1`, no train dir and no
  arena.
- **A full Step 3 pass:** the control is the Step 2 winner's reading, read not replayed. Realised lr is L1 group 0
  6e-5 / groups 1–3 1e-5, and L3 6e-5 / 3e-5; opt checks are ok.
- **A Step 3 relaunch** that plays nothing.

No mutant series, per the owner's rule.

### Not verified without torch (stated plainly)

- **The real trainer's realised per-group lr with `--matrix-lr-multiplier 6` / `2`** on top of
  `--optimizer-hparams-source init-checkpoint`. The code reads "explicitly set knobs win" (`simple_rl/learn.py`
  `resolve_optimizer_hparams`), so group 0 should be 1e-5 × 6 and 3e-5 × 2. The void rule catches it at update 1 if
  not.
- **That `trainer.opt.param_groups[i]["lr"]` after one `train_steps` call is the realised lr.** `train_smoke_v.py`
  records it; only the stand-in was run.
- **Whether the outcome-only and search-only arms' recalibration falls inside phase B's R1 box.** The box was fitted
  on blend nets; an arm outside it writes `HOLD_RECAL_<arm>` and stops for the owner. That is the registered behaviour,
  not a defect.
- **Wall times** beyond the measured basis above.
- **Which group holds `value_wdl.*` on the real net** (the Step 3 pre-launch one-liner).

## Step 1 verdict (read 2026-11-27)

Step 1 exited 0 at 18:21:24 EST (`step1/exit` newer than `step1.pid`; launched 15:54:07, launcher `aab7bec6`);
every step exit reads 0 and `STEP1_DONE` exists. Read once by the registered rule from
`~/chess-artifacts/operations/rl-value-followup/step1/contrast64_look1.json` [`10513cd2`]. Each arm is kC's policy with
that net's value head, against `kC_baked.pt`, 3,200 games (1,600 per seat, 800 opening pairs per seat), seed
2026112721.

| Reading | 64 sims (decides), Elo [95% CI] | 16 sims (reported only) |
|---|---|---|
| hC: C_baked's value on kC's policy, vs kC | −3.69 [−12.12, +4.74] | −7.49 [−17.18, +2.19] |
| hD: D_baked's value on kC's policy, vs kC | −0.98 [−9.23, +7.27] | +2.71 [−7.16, +12.59] |
| **Δ = E_D − E_C**, paired by opening | **+2.71 [−8.92, +14.35]** | +10.21 [−2.93, +23.34] |

**Registered verdict: |Δ| < 5 and neither arm's own E in the extend band, so Step 1 stops at look 1: not resolved
at 3,200 games.** The bank's label strings predate the wording fix and read "no C-vs-D value difference at screen
resolution" / "no value gain at screen resolution"; per this record's review note they mean "not resolved", and the
interval is the result. Look 2 was not played. The 16-sim file is
`~/chess-artifacts/operations/rl-value-followup/step1/contrast16.json` [`d1e448c5`].

**What this licenses.**

- No claim that the deep/fast value head differs from the stock one, in either direction. The paired contrast at
  this size detects about 17 Elo (power table above); the interval excludes a D value advantage above +14.4 and a C
  advantage above +8.9. Realised arm-control correlation: +0.028 at 64 sims, +0.098 at 16.
- Neither value head, placed on the parent's policy, is shown to gain or to lose against the parent.
- Descriptive only: the full nets read C −5.81 and D −3.34 against the same parent at 12,800 games, and the
  head-swap points are −3.69 and −0.98. The points are compatible with the value head carrying most of each net's
  deficit and with it carrying none; this step cannot separate them. The full-net and head-swap readings used
  different opening sets (seed 2026112711 and 2026112721), so the four points are not paired.
- Descriptive only: D − C is positive in the full-net match (+2.5), here at 64 sims (+2.7) and at 16 sims (+10.2).
  None is resolved; the three share the same two nets and the two head-swap readings share openings, so this is
  not three independent replications, and the 16-sim reading decides nothing.

hC's banked games are Step 2's control. Step 2 was launched at 18:23:08 EST (pid in `step2/step2.pid`, launcher
`1a00398c`). Measured before it: all 2,785,627 rows of pool_C carry a stored search WDL (`has_search_wdl`, 141
shards, read on the CPU), so the search-only arm's effective search share is 1.0 and the outcome-fallback confound
is empty (the trainer sets the dampen knobs, `wdl_terminal_outcome_plies` and `sf_wdl_frac` to 0 at the pin,
`simple_rl/learn.py:1078-1081`, so no taper moves search weight to the outcome).

## Amendment 1: max-plies draws (2026-11-28)

**What happened.** Step 2 exited 1 at 00:17:03 EST on 2026-11-28. The failure was in step `c64_2`, the control's look 2
(hC at 64 sims, seed 2026112722, 4,800 games per seat):
- Seat A finished with exit 0 (4,800 games, results row `on_max_plies` abort, 0 unresolved).
- Seat B aborted after 2,418 games. One game (pair 884, half 0, kC vs kC + hC value) reached `--max-plies 600` without
  a result by the rules or by Syzygy. Its final position is a queen ending, `8/1k6/3p3p/PPpP3P/1p3Q2/1P6/K1q5/8 w - - 4 309`.
  Under `--on-max-plies abort` (`common23/run_v2_match.sh`) the arena refused to score it. It banked the game in
  `hC64_2_B.games.jsonl.unresolved.jsonl` and wrote **no results row** for seat B.

Every earlier step and seat is sealed (exit 0): both arms' training, fits, bakes, smokes, `h64_O`, `h64_S` and
`read64_1`. Step 1's seats had 0 unresolved games.

**Rejected options.**
- **Replaying seat B from the same seed under `abort`.** Openings are a pure function of the seed and the game count
  (`np.random.default_rng(seed)`, `scripts/arena_standard.py:3382` at the pin). The same generator is then passed into
  the rolling loop (`rng=rng`, line 2777), which consumes it in batch order, and the loop is single-threaded. So a
  same-seed, same-N replay repeats the same games up to GPU float non-determinism, and would probably reach the same
  600-ply game again.
  - The play is **not** a function of (seed, pair) alone. The 4-game smoke seats and the 1,600-game seats share seed
    2026112721 and openings, but all 16 of the smoke games differ from their 1,600-game counterparts in moves and
    length. Exact reproduction of one 600-ply game was not tested; it would need the GPU.
  - Either way, `abort` remains the wrong policy for a seat that has to finish.
- **Keeping the 2,418 finished games.** This is survivorship-biased: the missing half of the seat includes the long
  games.

**Registered handling.**
1. **Set aside seat B.** Its five files (`hC64_2_B.games.jsonl`, `.games.jsonl.unresolved.jsonl`, `.pgn`, `.log`,
   `.exit`) move into `step2/arenas/hC64/aborted_<UTC>/` with a `SHA256SUMS` of them. They are preserved, never
   deleted, and excluded from every read. The operator does this by hand with the commands below. The launcher stops on
   the PARTIAL seat (87) until it is done.
2. **Every remaining seat of Steps 2 and 3 plays with `--on-max-plies draw`.** This includes the seat B replay, which
   uses its registered seed 2026112722 and 4,800 games. The seats go through the new `common23/run_v3_match.sh`.
3. **Sealed seats stay as they are.** They played under `abort` with 0 unresolved games, so `draw` would have given
   them identical scores.
4. **Each read reports, per seat:** `on_max_plies`, the number of `max_plies` draws, and the games.
5. **HOLD bound:** a read HOLDs (`HOLD_MAXPLIES`, the launcher stops for the owner) if any seat's max-plies draws exceed
   **0.25 % of its games**: more than 12 of 4,800, or more than 4 of 1,600.
   - The observed rate here is 1 in about 20,000 games played so far by Steps 1–2. The evidence log's strict-Syzygy
     rate is about 3e-4 per game.
   - At 3e-4, a 4,800-game seat expects 1.4 draws and a 1,600-game seat 0.5. The bound sits about 9 Poisson standard
     deviations above the 4,800-game mean.

**Size.** One max-plies draw that "should" have been decisive moves its seat's score by 0.5 / 4,800 = 1.04e-4, which
is **0.072 Elo** on that seat (slope 1600 / ln 10 per unit excess). Each arm's seat-balanced E and Δ average the
seat with its partner, so the effect on them is **0.036 Elo**. At the bound (12 draws in one seat), the worst case
is 0.87 Elo on that seat and 0.43 Elo on the arm, against a paired half-width of ±11.7 (look 1) and ±5.9 (final).

**The draw takes effect at the pin (`d08b03adb`, `scripts/arena_standard.py`).**
- `run_arena` validates the flag (3508–3512): `draw` without `--syzygy` exits.
- For `draw`, it binds `on_max_plies` into the header settings and the resume fingerprint (3513–3518). An abort log and
  a draw log can therefore never resume into each other.
- In `play_paired_games_matched_sims_rolling` (2655–2707), a board with no rules result and no Syzygy result, whose
  plies are ≥ `max_plies`, reaches `if tb_probe is not None and on_max_plies == ON_MAX_PLIES_ABORT` (2672).
  - Under `draw` that branch is skipped. The game gets `res = "*"` and `termination = "max_plies"` (2681).
  - `_record` (2605–2631) scores it 0.5 and writes it through `pgn_sink` as `"1/2-1/2"` with termination
    `max_plies`. `pgn_sink` is `_on_game` (3888), whose `game_log.write_game(row)` puts it into the `.games.jsonl`.
- The results row's `on_max_plies` is the policy applied (4546). `max_plies_unresolved` is counted from the game log's
  latest rows with termination `max_plies` (`count_max_plies_games`, 3189–3209). `max_plies_aborts_banked` counts the
  sidecar beside the log (3212).
- The chunked loop has the same guard (2397, 2422). The rolling loop is the one `matched_sims` uses here (the abort
  traceback names it: line 2704 in `play_paired_games_matched_sims_rolling`).

**The readers.**
- **`seat_state_v.py`** (Step 1's file, unchanged) checks the exit, the header seed, the game count, uniqueness, and
  one results row with the right shas, `git_sha`, seed and value blocks. It reads neither argv nor `on_max_plies`.
  - Sealed seats therefore stay DONE, not STALE.
  - The replayed seat B is DONE once it finishes: a fresh log, one header, 4,800 games, one results row.
  - Before the set-aside, seat B is PARTIAL (exit 1, no row), so the launcher stops with 87.
- **`contrast_v2.py`** (amended):
  - It drops `on_max_plies` from the header-settings comparison. The arena writes that key only for `draw`, so
    without this change the abort-sealed seat A and the draw-replayed seat B would REFUSE.
  - In its place, each run is checked three ways:
    - the results row's `on_max_plies` is abort or draw and equals the header's (absent means abort);
    - the log's count of games with termination `max_plies` equals the row's `max_plies_unresolved`, and is 0 under
      abort;
    - `max_plies_aborts_banked` is 0.
  - It outputs `max_plies` per run, plus `max_plies_bound_frac`.
  - Above the bound, it writes `<out>.HOLD.json` instead of `<out>` and exits 3. `lib23.sh`'s new `contrast()` turns
    exit 3 into `HOLD_MAXPLIES`.
  - Rerun on the real Step 2 look-1 logs, it reproduces the sealed `contrast64_looks1.json` exactly: rule, Δ and E
    identical, with every seat reported as abort with 0 draws.
- `contrast_v.read_log` scores a max-plies draw as `score_candidate` 0.5, like any draw.

### Amendment 1 files (bank `~/chess-artifacts/operations/rl-value-followup/`)

| File | sha256 | Change |
|---|---|---|
| `common23/run_v3_match.sh` | `68856fbc…` | new: `run_v2_match.sh` with `--on-max-plies draw`, and a refusal (95) while an abort sidecar is beside the seat |
| `common23/lib23.sh` | `2dbda08d…` (was `647456a8`) | `seat()` calls `run_v3_match.sh`; reads go through `contrast()` (exit 3 → `HOLD_MAXPLIES`); the PARTIAL message names the sidecar |
| `common23/contrast_v2.py` | `ab514cae…` (was `afabc94a`) | as above |
| `SHA256SUMS.step23` | `9f7cc7b2d418af4a9647daf612876e8a4cfa0813842e2d3cb953d7f9c7cf837d` (was `e0ed8638`) | lists `run_v3_match.sh`; new lib23 / contrast_v2 hashes |
| `step2/launch_step2.sh` | `2ba394c657b11705302b1217dd5f2edbbac0a58fe9aabd7362746248566a744b` (was `1a00398c`) | re-pinned `SUMS23_SHA`; `PRIOR_INPUTS_A1` |
| `step3/launch_step3.sh` | `9c400db92dd53240a6b18c8dbc1ffd607abbdac340a80360fcf37fd35c17da33` (was `5ffac6f0`) | re-pinned `SUMS23_SHA` |

- `run_v2_match.sh`, Step 1's files and every sealed seat are unchanged. The pre-amendment lib23, contrast_v2, both
  launchers and `SHA256SUMS.step23` are kept byte for byte in `step2/test/pre_amend1/`.
- **`inputs.txt`.** Step 2's `inputs.txt` records the launcher's sha256, so any new launcher would refuse with 84.
  `launch_step2.sh` therefore accepts the pre-amendment line exactly once. If `inputs.txt` equals `PRIOR_INPUTS_A1`
  (the recorded line, launcher `1a00398c`, sums `e0ed8638`), it moves the file to `inputs.pre_amend1.txt`, writes the
  new line and logs this to `resume.log`. Any other difference still refuses with 84. This runs after every other
  refusal.
- **Relaunch.** It skips every step whose exit reads 0. `c64_2` reads 1, so it reruns: seat A is DONE (skipped, logged
  "not replayed") and seat B is ABSENT after the set-aside, so it is replayed. Then `x64_O`, `x64_S` and `read64_2`
  follow, and the rest of `run_all`.
- **Step 3** sources the same `lib23.sh`, so all of its seats play through `run_v3_match.sh`.
- **The existing release line stays valid:** `vstep2 idle_mib=3011 2026-11-27T18:22:59-05:00`. The guard checks the
  idle baseline, not its age.

### Amendment 1 operator commands

Run these after this PR is merged, Stage 0 has released the GPU, and no Step 2 process is alive (`step2/step2.pid`
names a dead pid; `ps` shows no arena naming this bank).
```bash
V=~/chess-artifacts/operations/rl-value-followup
# 1. set seat B aside (preserved, never deleted, read by nothing)
U=$(date -u +%Y%m%dT%H%M%SZ); X=$V/step2/arenas/hC64/aborted_$U
mkdir "$X" && mv $V/step2/arenas/hC64/hC64_2_B.games.jsonl $V/step2/arenas/hC64/hC64_2_B.games.jsonl.unresolved.jsonl \
  $V/step2/arenas/hC64/hC64_2_B.pgn $V/step2/arenas/hC64/hC64_2_B.log $V/step2/arenas/hC64/hC64_2_B.exit "$X"/ && (cd "$X" && sha256sum * > SHA256SUMS)
# 2. check the launcher and the sums
sha256sum $V/step2/launch_step2.sh $V/SHA256SUMS.step23   # 2ba394c6… / 9f7cc7b2…
(cd $V && sha256sum -c --quiet SHA256SUMS.step23) && echo ok
# 3. relaunch Step 2 (resumes at c64_2: seat A skipped, seat B replayed with --on-max-plies draw)
setsid nohup bash $V/step2/launch_step2.sh >> $V/step2/step2.log 2>&1 < /dev/null &
```
Stop as before: `kill -TERM $(cat $V/step2/step2.pid)`. Step 3's launch command is unchanged
(`sha256sum $V/step3/launch_step3.sh` must print `9c400db9…`).

### Amendment 1 harness

**`step2/test/harness_step23.sh` [`661909ea`] with `step2/test/fake23.py` [`b89a49f4`]: 36 of 36 pass**
(`step2/test/harness_step23.out` [`eddcd23c`]; the pre-amendment 25-of-25 output [`67ad9ea8`] moved to
`step2/test/pre_amend1/harness_step23.out`).
- The fake bank is first built from `pre_amend1/`, and Step 2 runs with one fake unresolved game in `hC64_2_B`. The
  fake arena follows the real abort semantics: earlier games are written, the game goes to the sidecar, there is no
  results row, and the exit is 1. That reproduces the real state: launcher 1, `c64_2` 1, seat A 0, seat B 1, sidecar
  1 row, no B row.
- The amendment is then deployed over it as in the real bank.
- **A second stated difference** from a real launch: the copy's `PRIOR_INPUTS_A1` is rewritten to the fake launch's
  `inputs.txt`, because the fake pre-amendment launcher carries the stub-rewritten hashes.

What it proves:
- **The refusals on the amended launcher:** the existing ones, plus a tampered `run_v3_match.sh` (93), and an
  `inputs.txt` that is not exactly the prior line (84, not migrated).
- **Relaunch before the set-aside:** stops at PARTIAL (87) and plays nothing. The inputs are migrated once.
- **The operator's set-aside commands**, run verbatim against the fake bank.
- **Relaunch after the set-aside** (the same game unresolved again, plus 12 in `hO64_2_A`):
  - it exits 0 with `STEP2_DONE`;
  - all 15 steps before `c64_2` are skipped and seat A is "DONE: not replayed";
  - no sealed seat is played again;
  - every results row's argv carries `abort` for the 9 rows sealed before the amendment and `draw` for every other;
  - the replayed B row has `on_max_plies` draw, `max_plies_unresolved` 1 and `max_plies_aborts_banked` 0, with the
    game written as a `max_plies` draw;
  - `contrast64_looks12.json` reports hC64_2_A abort 0, hC64_2_B draw 1, and hO64_2_A 12, which equals the bound and
    is not over it;
  - the set-aside files are intact.
- **`contrast_v2.py`:** it reads abort-sealed A together with draw-replayed B, and refuses a draw row whose
  `max_plies_unresolved` disagrees with its log.
- **Step 3:** 5 draws in a 1,600-game seat HOLD the read: exit 3, `HOLD_MAXPLIES`, `.HOLD.json`, and no read file. The
  relaunch is then refused (81).
- The full Step 3 pass and its relaunch are as before.

**Mutant (one).** `lib23.sh`'s `seat()` was changed back to `run_v2_match.sh`, meaning the draw flag is accepted in the
record and ignored on the path. The harness then gives **21 pass, 15 WRONG**:
- the replayed seat B aborts again (relaunch exit 1);
- B is still PARTIAL on the next relaunch (87);
- no look 1+2 read is made;
- Step 3 never starts.

With the amended file it passes 36 of 36.

### Amendment 1: not verified (stated plainly)

- **Whether a same-seed replay would reproduce the 600-ply game.** It needs the GPU. The handling does not depend on
  it.
- **The real arena under `--on-max-plies draw`** was not run. The take-effect proof above reads the code at the pin.
  The fake arena only mirrors its semantics (header key, termination, row counters).
- **Whether `draw` adds wall time.** A 600-ply game is one game of 4,800; the replay of seat B costs one seat, about
  80 min (seat A of this look took 80 min).
