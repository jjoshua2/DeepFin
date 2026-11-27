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
- `lr1e-5`: `--lr 1e-5 --lr-floor 1e-5`, about 3.3× the current value.
- `lr3e-5`: `--lr 3e-5 --lr-floor 3e-5`, 10×, half of group 0's 6e-5.

Group 0 stays at its init-checkpoint value. Changing it needs `--optimizer-hparams-source` other than
`init-checkpoint`, which is a second variable.

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

