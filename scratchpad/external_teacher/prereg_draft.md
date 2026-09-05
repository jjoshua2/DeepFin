# PREREG DRAFT — an EXTERNAL teacher (BT4), and why the audit-first rule cannot judge it

**Status: DRAFT. NOT LAUNCHED. No GPU spent. Not a ledger entry until Josh says go.**
Written 2026-08-13 while Tier-13 arm B trains.

## Why this experiment exists

Every target the loop trains on is derived from the loop. Measured, not argued:

- policy target = our own search over our own priors (`KL(target‖prior)` is the signal);
- value target = `0.69·sf_wdl + 0.31·search_wdl`, and `search_wdl`'s draw axis is the net's
  own raw output, which additionally CLAMPS the searched q (ledger d0eb060bf, 34aa63922);
- replay = our own games; difficulty = a PID servo on our own winrate;
- even the "improvement" ruler most often quoted (`wdl_regret`) measures the agent, not the net.

The one input that is genuinely external is Stockfish — and it is external only as a LABEL
(cp-logistic at 150-200k nodes, MultiPV 6), never as a move-ranking teacher on our own
positions. Measured consequence: absorption works 11.6× in-window and buys ZERO held-out
generalisation; the policy stopped generalising around iter 249.

**Hypothesis.** The loop's plateau is not capacity, not views, not loss weighting — it is
that the training signal is (almost) a function of the thing being trained, so gradient
descent has a fixed point that is not "strong play". An external teacher whose errors are
DECORRELATED from both our net and Stockfish should move held-out policy CE where absorption
never did. BT4 (191M lc0 net) is on disk, loads in our stack, and yields policy + WDL
(`Lc0OnnxEvaluator`, `onnx/load.py::OnnxChessNet`, `data/lc0/onnx/BT4-it332-vanilla-winner.onnx`).

## ⚑⚑ THE AUDIT-FIRST RULE HAS A BLIND SPOT HERE, AND IT WOULD KILL THIS BEFORE IT RAN

`docs/eval_protocol.md`: *every training-target candidate is scored against the frozen deep-SF
audit set BEFORE any training compute, and one that loses the direct audit is killed without
training.* That rule is correct for candidates that are supposed to approximate SF. **It is
structurally incapable of crediting a teacher for being DECORRELATED from SF**, because the
ruler IS SF:

- `value_regret` / `audit_targets` score an evaluator by the deep-SF regret of the move it
  picks. An evaluator is rewarded for AGREEING with SF.
- Our SF label already scores **16.1 cp** on that ruler (it is SF, at lower nodes).
- BT4 is a different engine. Even if BT4 is stronger in play, it will score WORSE than the SF
  label on an SF-graded ruler, essentially by construction.

So running the audit first would produce "SF label 16.1 beats BT4 — killed", and that verdict
would be an artifact of the ruler, not a fact about the teacher. This is the same shape as
[[same_name_different_population]] and the `wdl_regret` trap: **a presence of agreement is
not a measure of strength.**

**Resolution, pre-committed:** the audit-first rule is HONOURED but RE-POINTED. BT4 is not
screened for SF-agreement; it is screened for **decorrelation with usable direction**, on the
frozen audit set, with three numbers that an SF-graded ruler cannot fake:

1. **Disagreement rate**: fraction of audit positions where BT4's top move ≠ SF's top move.
   (If ~0, BT4 adds no information and the experiment is dead for free.)
2. **Conditional quality of the disagreements**: on exactly those positions, the deep-SF
   regret of BT4's move. If BT4's disagreements are cheap (small regret) it is a plausible
   second opinion; if they are expensive it is just wrong.
3. **Coverage of OUR errors**: on positions where OUR net's top move has high deep-SF regret,
   how often does BT4 pick the SF-preferred move? This is the only one of the three that
   directly measures "can this teacher fix what we get wrong", and it is the gate.

**KILL BEFORE TRAINING (pre-committed):** if BT4 fixes < 25% of our high-regret positions
(our-regret > 50 cp), the teacher does not cover our failure mode and no distillation runs.
**PROCEED** at ≥ 25%. Both computed on the frozen audit set, CPU/GPU-light, one pause window.

## If it proceeds: the training arm

- **Policy-side, not value-side.** The prior note ([[value_ceiling_bt4_distill_direction]])
  proposed VALUE distillation. Superseded by measurement: the value head's own designated
  ruler says it IMPROVED 17.8% while play went flat, and the value target's defect is now
  identified and separately fixed (Tier-14). The head that demonstrably stopped generalising
  is POLICY.
- Arm: add a BT4 soft-policy KL term at low weight, blended with (never replacing) the
  existing target, on our own replay distribution. Offline first, on the retarget rig — same
  donor/pool/steps as Tier-10/11/12 so it joins an existing comparison group.
- **Deciding yardstick (ONE):** held-out policy CE on the exposure-clean split — the exact
  measurement where absorption gained nothing (in-window ÷4.0, held-out −2.5%). That is the
  falsifier that distinguishes "this teacher adds information" from "this teacher adds fit".
  Brier/ECE banned; in-window CE banned as a success criterion for exactly this reason.
- **Kill:** held-out CE not better than control by ≥ the noise floor measured in the same run.

## Confounds and traps, stated before launch

- ⚑ The lc0 policy index is NOT our 1858 index — they agree on 46 of 1858 slots, and a static
  remap previously under-weighted castling priors 49-120× in 9.3% of positions. The correct
  mapping is board-aware (`moves/leela_index.py`, verified 0 mismatches over 55,586 moves).
  ANY BT4 policy comparison predating that fix is suspect and must not be cited as a baseline.
- BT4's encoder is lc0's 112-plane format, not our 175-plane v2_threats. The adapter handles
  it; the repetition plane differs (lc0 counts whole-game, we keep 7 plies).
- This is an INSTRUMENT path today. Nothing here touches production selfplay, and the
  offline arm must stay offline until a held-out CE win exists.
