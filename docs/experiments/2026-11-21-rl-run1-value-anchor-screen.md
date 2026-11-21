# 2026-11-21: RL run 1 — value-anchor screen (`--kl-wdl` to the SL base) on the pilot's collection (pre-registration)

**Status: registered, not launched.** It launches only after review and merge of this
record, AND after the follow-up chain releases the GPU (expected ≈ 21:30). It trains new
cells on the pilot's frozen collection, plays no new collection games, and changes nothing
live.

## Why

**The owner's request.**
- On 2026-11-19 the owner asked for the KL penalty on policy AND value. It was never run:
  - no banked run has `kl_wdl` or `kl_policy` non-zero;
  - the 11-20 plan made it a conditional branch (`2026-11-20-rl-run1-multi-generation-plan.md` ≈ L462);
  - the amendment set both to 0 (`2026-11-21-rl-run1-amendment-games-not-rows.md` ≈ L323: "nothing below is called anchored").
- His words today: a "kl penalty from last known best" for the value, because the value
  loss is "always clearly worse", not noise.

**What was already measured.** I grepped `docs/experiments/2026-11-21-rl-loop-evidence-log.md`
and `~/chess-artifacts/operations/rl-claude-20261119/INDEX.md` for `kl-wdl` / `kl_wdl` /
`kl-policy`, value dose (`--w-wdl` < 1) and `--sl-mix-frac`:
- **No measurement of any of the three exists.** The only hits are the plan and amendment
  lines cited above.
- The closest evidence is the pilot itself, all at `--kl-wdl 0`:

  | Cell | vs base |
  |---|---|
  | S (value on search WDL only) | −20.0 [−36.8, −3.3] |
  | Zq (half outcome, half search) | −10.0 [−23.2, +3.2] |
  | P0 (no value loss), as a single net | +2.2 [−4.2, +8.6] (match 5) |

- The value diagnosis (`rl-claude-20261119/pilot-value-diagnosis/`, README `0bdac6a5…`,
  Addendum 1 `PREREG.addendum1.md` `2a07543c…`) found that S's value loss broke sibling
  ordering, and that recalibration does not repair it.

## What `--kl-wdl` computes (read at the pin, `0c795fb7a`)

- **Code path.** `scripts/simple_rl_loop.py` → `TrainSettings.kl_wdl` →
  `learn.ReferenceAnchor(reference, policy_coef=kl_policy, wdl_coef=kl_wdl)`, installed by
  `attach_anchor` on the Trainer's penalty hook (`train/trainer.py` ≈ L5916). The penalty is
  added to the training loss before `backward()`.
- **Direction.** It is **KL(reference ‖ student)** on the WDL head:
  `Σ q · (log q − log p)`, where q = softmax(reference WDL) and p = softmax(student WDL),
  computed per row. This is the same direction as the diagnosis's "KL(base ‖ net)".
- **Rows.** Rows with `is_network_turn` AND `has_policy` (`frozen_policy_anchor.eligible_legal`).
  The pilot collection is self-play at `--full-prob 1.0`, so every training row is eligible,
  and the anchor sees the same rows as the WDL loss.
- **Weighting.** The total is `losses.total + kl_wdl · mean(KL)`, where the WDL term
  inside `losses.total` is `w_wdl · CE(target, p)`.
  - The gradient of KL(q ‖ p) with respect to the logits equals that of CE(q, p).
  - So the head is pulled toward the blended target **t_λ = (w_wdl · t + λ · q) / (w_wdl + λ)**.
  - With w_wdl = 1, the anchored fixed point is a = 1 / (1 + λ) of the way from the base to
    the target.
- **Reference.** `--kl-reference` defaults to the run's generation-0 model, which is the
  init checkpoint, the SL base. Every pilot cell lost to the base, so the base is "the last
  known best". **This record passes `--kl-reference $SL` explicitly anyway.**
- **`--value-rows-per-game`** does not exist at the pin. On main, the anchor's rows are the
  eligible rows, NOT the value-row subset. The pilot did not use it, and neither does this
  screen.
- **Aurora and `--optimizer-hparams-source init-checkpoint`: no special interaction.**
  - The anchor's gradient is part of the same total loss, so it reaches whatever parameter
    groups own the value head and trunk, at their realised rates. Those are the SL base's
    Aurora split: matrix group 6e-5, AdamW groups 3e-6. They are recorded in
    `config.realised.optimizer`.
  - **Consequence: both optimizers are scale-invariant per update.** Aurora orthogonalises,
    and Adam divides by √v. So for parameters that ONLY the value terms reach (the value
    head), multiplying the value loss by a constant barely changes the step.
  - So `--w-wdl` < 1 mainly lowers the value's share in the SHARED trunk, not the head's
    movement.
  - The anchor is different: it changes the gradient's DIRECTION (toward t_λ), which a
    scale-invariant step does not undo. The dose arm below is designed and read with this in
    mind.
- **Guards at the pin.**
  - The anchor refuses `accum_steps` > 1, prefetch and compile. The simple loop builds the
    Trainer with `prefetch_batches: False` and `use_compile: False`.
  - It refuses exact-epoch replay. The pilot's `--presentations 1` uses `ScheduledSampler`,
    which sets `stored_row_quota`, not `exact_without_replacement`, so the anchor is
    accepted.
  - No banked or e2e run has used `--kl-wdl` through `run_train`. A refusal is loud: it
    raises before training.

**The observation that proves the anchor took effect** (consumer-side; a cell lacking any of
these is void):
1. `gen_0001/train/metrics.json` → `summary.anchor` must show:
   - `wdl_coef` == λ;
   - `reference_forwards` == the number of optimizer updates;
   - `rows` == the trained rows;
   - `mean_wdl_kl` > 0.

   `mean_wdl_kl` is the anchor's own accumulated KL, averaged over every anchored row. The
   anchor keeps a run sum, not a per-step log; this run mean is the consumer-side record.
2. The training manifest's `inputs.kl_reference.sha256` == the SL base's sha256
   (`a39f900c…`).
3. Offline: the cell's measured KL(base ‖ cell) on hold-out rows falls with λ (Stage A).

A cell whose head moves exactly like the unanchored cell is reported as **"anchor not
taking effect"**, whatever its `summary.anchor` says.

## Pin

**The pilot's `$PIN` `0c795fb7a`, from the existing worktree
`~/projects/chess-worktrees/rl-run1-pilot-0c795fb7a`.**
- `--kl-wdl` works there: the anchor code is identical between the pin and main (918785204).
  `git diff 0c795fb7a 918785204` touches neither `ReferenceAnchor` / `attach_anchor` nor
  `train/trainer.py`. It changes only `--value-rows-per-game` in `learn.py` and the
  loop, and adds a `losses.py` change that is not on this path.
- At the pin, the cells train exactly as S / Zq / P0 did.
- The pilot's C0 → PIN guards (a) and (b) already passed for this collection at this pin, so
  no guard needs restating.
- The matches run at the same pin as every pilot and follow-up match, so the within-run
  comparison rule holds.

**Pre-launch check (CPU, minutes):** at the pin, run
`tests/test_simple_rl_learn.py::test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer`
(with `w_wdl = 0`, the value head learns only through the anchor) and
`tests/test_simple_rl_accum_steps.py`. Both must pass.

## Strengths: derived from the measured scales, not picked round

**Measured on the pilot's hold-out rows** (39,237 rows, 416 games), from
`pilot-value-diagnosis/out/holdout_logits.npz`:

| Cell | KL(base ‖ cell), unanchored | KL(base ‖ its target) | Fraction moved toward its target, g |
|---|---|---|---|
| S | 0.0167 | 0.100 | **0.277** |
| Zq | 0.0275 | 0.229 | **0.116** |

g is the regression slope of the cell's W−L move on the target − base W−L gap. Zq's target
contains the one-hot outcome, so it is farther away, and Zq covered less of it.

**Two models of anchored training, both from first principles:**
- **Scale-invariant steps (Adam / Aurora).** The head heads for t_λ at roughly the
  unanchored speed, so its displacement ≈ min(g, a) of the gap, with a = 1 / (1 + λ). The
  anchor binds only once a < g.
- **Linear (gradient-flow) transient.** The displacement is
  a · gap · (1 − (1 − g)^{1 + λ}).

The two models agree to within about 0.05 at the strengths chosen. **Targets:** hold the
head to ≈ 0.70, 0.45 and 0.25 of its unanchored displacement, so KL ≈ 0.5, 0.2 and 0.06 of
the unanchored KL:

| Cell | λ (`--kl-wdl`) | Predicted displacement / unanchored | KL(base ‖ cell) it should hold to |
|---|---|---|---|
| S-anchor weak | **2.6** | ≈ 0.69 | ≈ 0.008 |
| S-anchor medium | **6.2** | ≈ 0.45 | ≈ 0.0034 |
| S-anchor strong | **13** | ≈ 0.25 | ≈ 0.0010 |
| Zq-anchor weak | **7.6** | ≈ 0.65 | ≈ 0.012 |
| Zq-anchor medium | **16** | ≈ 0.44 | ≈ 0.0053 |
| Zq-anchor strong | **34** | ≈ 0.25 | ≈ 0.0017 |

For scale: P0's value head drifted KL 0.0072 from the base with NO value loss. The medium
and strong cells are held below that policy-induced drift.

## Cells

All cells train on the pilot's frozen collection. They use the pilot's `COLLECT` (needed by
`--collect-from`, which refuses another seed) and its `$TRAIN` **verbatim, except
`--kl-wdl` and the value terms**:

```bash
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
```

This is the pilot's `TRAIN` with `--kl-wdl 0` removed; each cell passes its own.

| Cell | Value flags |
|---|---|
| S-anchor λ ∈ {2.6, 6.2, 13} | S's: `--value-outcome-weight 0 --value-search-weight 1 --value-search-source stored_search_wdl --kl-wdl λ --kl-reference $SL` |
| Zq-anchor λ ∈ {7.6, 16, 34} | Zq's: `--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl --kl-wdl λ --kl-reference $SL` |
| S-dose w ∈ {0.1, 0.03} (no anchor) | S's value flags, `--w-wdl w --kl-wdl 0` |
| P0-anchor (no value loss) | `--w-wdl 0 --kl-wdl 6.2 --kl-reference $SL` |
| Controls (no retraining) | the banked S, Zq and P0 (`rl-run1-pilot/arm_*`) |

**The dose arm** is the alternative explanation "it is just less movement".
- It is applied to the S target only, because S is the target with the established ranking
  damage. Two doses keep the screen to 9 trained cells.
- **Prediction, from the scale-invariance above:** the dose leaves the value HEAD's movement
  nearly unchanged and lowers only the value's share of the trunk gradient. So:
  - if the dose cells' KL(base ‖ cell) stays near 0.0167, "less movement" is not what
    `--w-wdl` delivers here, and that is reported as such;
  - if their KL does fall like the anchor's, the dose is a live alternative explanation.

**The P0-anchor cell separates the two drifts.** It has no value loss, so the only value
pull is the anchor, which holds the head against the policy-induced trunk drift alone.
- Its λ equals S-medium's, so the anchor's share of the trunk gradient matches that cell's.
- Comparing P0-anchor's drift with P0's (KL 0.0072, uniformly pessimistic −0.012 W−L) shows
  how much of any anchored cell's effect is "anti-policy-drift" rather than "anti-value-loss".
- P0-anchor is offline-only here. Matches on it would extend match 5's +2.2 and are not part
  of this screen.

## Deciding yardstick

### Stage A: an offline VETO, not a selector

**Instruments:** the banked instruments of `rl-claude-20261119/pilot-value-diagnosis/`, run
on every trained cell:
- **clear-pair sibling ordering** (`scripts/sibling_order.py`): pairwise accuracy against
  deep SF on clear pairs (gap ≥ 50 cp, the `clearplus_50plus` bin), paired against the base,
  with game-clustered CIs; base 0.8554;
- **1-ply value regret** (`scripts/value_regret_recal.py`, which takes `--net` / `--recal`);
- **KL to the base and the (T, b) fit** (`scripts/fit_value_recal.py --collect-dir`, the
  pilot's hold-out games);
- **outcome calibration and toward-search movement on hold-out games**
  (`scripts/collect_logits.py`, `analyze_holdout.py`, `supp_outcome_distance.py`).

**How the banked scripts are reused.** They hard-code their net list and their output
directory, and they refuse to overwrite. Stage A therefore runs **copies** of them with
exactly two edits, the net list and the output directory, writing under
`~/chess-artifacts/operations/rl-run1-value-anchor/offline/`. Before any cell is scored,
the copies' diff against the banked files is saved beside them with both sha256s. The diff
must touch nothing but those two lines; a copy with any other change voids Stage A.

**Rules:**
- **VETOED** iff the cell's paired clear-pair accuracy difference (cell − base, game-clustered
  95% CI) lies entirely below 0. That is what S's did: −0.0096 [−0.0129, −0.0064].
- **INERT** ("did it learn anything?") iff the cell's top-decile share of the
  search-vs-base gap closed (`analyze_holdout.py` → `toward_search_top_decile.share_of_gap_closed`,
  the 3,924 rows with |search − base| ≥ 0.241) has a 95% CI that includes 0, or a point
  estimate below 0.1 × the unanchored cell's. The unanchored values are S 0.293 and Zq
  0.346, so the thresholds are 0.029 and 0.035.
  - The dose cells use S's threshold.
  - Inert cells are reported as "inert" and are not matched: passing the veto by not moving
    at all is not a recipe.
- **No ranking.** The offline numbers may NOT rank survivors. There are three fit→Elo
  transfer failures on record:
  - v020: +3.1 cp audit gain, −9.6 Elo;
  - ft595: −41.0 Elo for better fit;
  - F-only: hold-out CE −0.043 nats for 0 cp.

  Stage A only removes cells.

### Stage B: matches

- **Selection, fixed now by λ, not by offline score:** for each target, the **smallest-λ
  cell that is neither vetoed nor inert**. That is at most two matches, one S-anchor and one
  Zq-anchor. Dose and P0-anchor cells are not matched.
- **Design.**
  - The cell vs the SL base, plain seats, seat-balanced.
  - 6,400 games: 3,200 per run, 1,600 openings.
  - Seeds: **2026112181** for the S-anchor match, **2026112182** for the Zq-anchor match.
    Neither appears in any pilot or follow-up match nor anywhere under `docs/`.
  - The follow-up launcher (`rl-run1-followup/run_followup_match.sh`, from the follow-up
    record) with out root `rl-run1-value-anchor/`, the pilot's `$ARENA` flags, and the
    launch record's max-plies void rule.
- **Rules, as match 5:**
  - "damage" iff UB95 < 0;
  - "adoptable" iff LB95 > −10;
  - otherwise "unread".

  Read on the measured clustered SE, with match 5's tie-break: below SE ≈ 2.55, both facts
  are reported as "adoptable within the margin, with a measured loss".
- **Pre-declared contrast, report-only:** the anchored cell's estimate minus its unanchored
  cell's banked result.

  | Target | Unanchored result | SE of the difference |
  |---|---|---|
  | S | −20.0 (SE 8.5 at 800 games) | ≈ √(3.37² + 8.5²) ≈ 9.2 |
  | Zq | −10.0 (SE 6.7) | ≈ √(3.37² + 6.74²) ≈ 7.5 |

  Separate seeds, same pin; the SEs treat the matches as independent.

**What each outcome licenses:**

| Reading | Licenses |
|---|---|
| adoptable | NOMINATES that value recipe (the target plus λ) for a 2-generation run together with the policy, under its own record. Nothing is adopted here |
| damage | The anchor at that strength does not prevent the value damage |
| unread | Nothing |

**What counts as "the anchor is NOT the fix"** for a target:
- (a) every anchored cell of that target is vetoed or inert;
- OR (b) its matched cell reads "damage".

If the S-dose cells also pass the veto without being inert, "less movement" remains a live
alternative to "the anchor's direction", and any adoptable reading is reported with that
alternative named.

## Confounds

- **Policy-induced drift.** P0 showed that the trunk moves the value even with no value loss
  (KL 0.0072). So an anchored cell also anchors against policy-induced drift. The P0-anchor
  cell measures that part alone (see above).
- **One data-affecting change.** Each anchored cell differs from its banked unanchored cell
  only in `--kl-wdl` / `--kl-reference`: same collection, seed, batch plan, pin and optimizer.

## Cost

| Item | Cost |
|---|---|
| Training, 9 cells | ≈ 7 min each (measured 398–440 s for S / Zq / P0) ≈ **1.1 GPU-h**. The anchor adds one no-grad reference forward per batch, so the realised wall time is reported |
| Offline Stage A, 9 cells | ≈ 6 min each ≈ **0.9 GPU-h**, capped at 0.24 of the card as banked |
| Stage B | ≤ 2 matches × 1.52 GPU-h ≈ **≤ 3.0 GPU-h** |
| **Total** | **≤ ≈ 5.0 GPU-h** |

## Commands

Outputs go to `~/chess-artifacts/operations/rl-run1-value-anchor/` (`V`). The pilot bank
(`P`) is read-only.

```bash
V=$HOME/chess-artifacts/operations/rl-run1-value-anchor; P=$HOME/chess-artifacts/operations/rl-run1-pilot
PIN=$(cat $P/PIN.txt); PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
PY=$HOME/projects/chess/.venv/bin/python; SL=$P/sl_base.pt
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
SF=$HOME/local_stockfish_linux_latest/stockfish/stockfish-ubuntu-x86-64-bmi2
COLLECT="--openings-dir $P/openings --init-checkpoint $SL --device cuda --seed 2026112100 \
  --games 8000 --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 \
  --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450 --syzygy $TB --sf-path $SF"
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
S_V="--value-outcome-weight 0 --value-search-weight 1 --value-search-source stored_search_wdl"
Z_V="--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl"
declare -A CELL=(
  [S_k2.6]="$S_V --kl-wdl 2.6 --kl-reference $SL"  [S_k6.2]="$S_V --kl-wdl 6.2 --kl-reference $SL"
  [S_k13]="$S_V --kl-wdl 13 --kl-reference $SL"
  [Zq_k7.6]="$Z_V --kl-wdl 7.6 --kl-reference $SL" [Zq_k16]="$Z_V --kl-wdl 16 --kl-reference $SL"
  [Zq_k34]="$Z_V --kl-wdl 34 --kl-reference $SL"
  [S_w0.1]="$S_V --w-wdl 0.1 --kl-wdl 0"           [S_w0.03]="$S_V --w-wdl 0.03 --kl-wdl 0"
  [P0_k6.2]="--w-wdl 0 --kl-wdl 6.2 --kl-reference $SL")
cd $PINW && [ "$(git rev-parse HEAD)" = "$PIN" ] || exit 96
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider \
  "tests/test_simple_rl_learn.py::test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer" \
  tests/test_simple_rl_accum_steps.py || exit 1
mkdir -p $V
for c in S_k2.6 S_k6.2 S_k13 Zq_k7.6 Zq_k16 Zq_k34 S_w0.1 S_w0.03 P0_k6.2; do
  date -Is > $V/arm_$c.started
  PYTHONPATH=. $PY -u scripts/simple_rl_loop.py run --run-dir $V/arm_$c --collect-from $P/collect \
    --generations 1 $COLLECT $TRAIN ${CELL[$c]} > $V/arm_$c.log 2>&1; echo "$? $(date -Is)" > $V/arm_$c.exit
done
# Take-effect check: every anchored cell (dose cells: anchor must be null)
for c in "${!CELL[@]}"; do $PY - $V/arm_$c/gen_0001/train "$c" <<'EOF' || echo "VOID $c"
import json, sys
d, c = sys.argv[1], sys.argv[2]
s = json.load(open(f"{d}/metrics.json"))["summary"]; a = s["anchor"]
lam = float(c.split("_k")[1]) if "_k" in c else None
if lam is None:
    sys.exit(0 if a is None else f"{c}: an anchor ran on a dose cell")
ok = (a is not None and abs(a["wdl_coef"] - lam) < 1e-12 and a["reference_forwards"] == s["updates"]
      and a["rows"] > 0 and (a["mean_wdl_kl"] or 0) > 0)
m = json.load(open(f"{d}/manifest.json"))
ok = ok and m["inputs"]["kl_reference"]["sha256"].startswith("a39f900c")
print(c, a, "updates/s", s["updates_per_second"]); sys.exit(0 if ok else f"{c}: anchor did not take effect")
EOF
done
```

The check requires `reference_forwards` == `summary.updates` (one reference forward per batch).

**Stage A** follows the procedure above, writing under `$V/offline/`. **Stage B** then runs,
for each selected cell `X` with seed `SEED`:

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup
$F/run_followup_match.sh $V anchor_$X 1 A $SEED 3200 --candidate $V/arm_$X/gen_0001/train/model.pt --reference $SL || exit 1
$F/run_followup_match.sh $V anchor_$X 1 B $SEED 3200 --candidate $SL --reference $V/arm_$X/gen_0001/train/model.pt || exit 1
d=$V/arenas/anchor_$X; cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $V/arm_$X/gen_0001/train/model.pt \
  --a-games $d/anchor_${X}_1_A.games.jsonl --b-games $d/anchor_${X}_1_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl \
  --a-label anchor_${X}_1_A --b-label anchor_${X}_1_B --out $d/pool.json > $d/pool.log 2>&1
```

**Recovery.** Nothing to recover: no live change. The pilot bank is read only; the arms
reference its collect, and `simple_rl/reference.py` documents that it never writes the
source.
