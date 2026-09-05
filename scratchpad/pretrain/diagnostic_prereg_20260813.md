# PREREG DRAFT — BT4 distillation as a DIAGNOSTIC, not a direction

**Status: DRAFT. NOT LAUNCHED. Needs Josh's go + a GPU pause window. Sits behind Tier-13/14.**
Written 2026-08-13 while Tier-13 arm B trains (CPU-only).

## The reframe that defines the design

Josh, 2026-08-13: *"I guess it's ok to try distilling BT4 to see what kind of thing we can
expect, but i don't want to be merely an lc0 clone since the goal was to RL into exploiting
Stockfish and be something different, but it seems like it might help us diagnose if we have
problems with our data or labels or architecture."*

This is a better framing than the one in `strategy_20260813.md`, and it changes what gets
built. The earlier framing asked "should we pretrain on external data" — a DIRECTION question,
which is the one Josh is declining. The right question is a DIAGNOSTIC one: **the loop is not
gaining Elo; is the defect in our DATA, our LABELS, or our ARCHITECTURE?** We have never been
able to separate those three, because every instrument we own is downstream of all three at
once.

BT4 separates them because it is a **fixed external function with no self-reference**: no PID,
no curriculum, no replay composition, no dependence on our own net. That makes it usable as a
RULER and as a CONTROL TARGET, which is a completely different use from making it our teacher.

**⇒ Every arm below outputs a NUMBER. No arm ships weights.** The distilled checkpoints are
diagnostic byproducts and can be deleted after readout. Nothing here proposes replacing the
anti-engine objective, and nothing here goes near production selfplay.

## D0 — LOCATE OURSELVES ON AN EXTERNAL LADDER. **Zero training compute. Run this FIRST.**

Josh, 2026-08-13, pointing at `https://github.com/dje-dev/CeresNets`: *"it even posts the
approximate elo difference of each one. some are bigger than ours and some are smaller and
there is a 512x15 that is quite close to ours … about 200 elo range. from about 150 elo larger
than our size to 100 elo below for smaller."*

Confirmed against the repo. Ten ONNX nets, GPL-3.0, with a published relative-Elo column:

| net | arch | their Elo | vs 512x15 |
|---|---|---|---|
| C1-256-10 | 256x10 | −96 | −105 |
| C1-384-12 | 384x12 | −42 | −51 |
| **C1-512-15** | **512x15** | **9** | **0 ← nearest to ours (we are 512×16)** |
| C1-768-15 | 768x15 | 54 | +45 |
| C1-512-25 | 512x25 | 74 | +65 |
| C1-640-25 | 640x25 | 124 | +115 |
| C1-640-34 | 640x34 | 160 | +151 |
| C1-768-26 | 768x26 | 150 | +141 |

**256 Elo of span, +151 above our size and −105 below.** Josh's read is exact.

**⇒ THIS IS THE NEGATIVE CONTROL D1 NEEDED, SUPPLIED EXTERNALLY AND FOR FREE.** D1 was going
to spend GPU hours distilling into a small net purely to give a fidelity number a scale.
CeresNets supplies a whole *strength-vs-size curve* from one consistently-trained family, and
C1-512-15 is a near-exact size analogue of our 512×16 net. No training required — arena only.

**We already own the adapter.** `chess_anti_engine/onnx/load.py::OnnxChessNet` opens with
*"Load a foreign ONNX chess net (CeresNets / LC0) for use in our search"* — it slices the
112 LC0 planes, declares its own input contract (`input_history_encoding="lc0_root"`), does the
board-aware policy remap via `moves/leela_index.py`, and returns our `policy_own`/`wdl` dict.
Our on-disk BT4 files already carry Ceres export naming (vanilla/optimistic/soft/opponent ×
winner/q/st). This is configuration, not construction.

**THE READ, pre-committed:**
- **We beat or match C1-512-15** ⇒ our training is not broken *at our size*; the lever is
  scale, and Josh's ~100 Elo architecture claim is live and quantified (+151 available on this
  family by going to 640×34).
- **We lose badly to C1-512-15** ⇒ **we sit far below the curve at our own size**, and scaling
  parameters is the WRONG move — a bigger net trained by our loop lands further below a higher
  curve point. Fix training first. This outcome would also retire the architecture question
  without spending a single GPU-hour on it.

**Second-order payoff: a non-self, non-Stockfish Elo anchor ladder.** Every anchor we own is
one of our own checkpoints, which is why [[arena_elo_is_anchor_dependent]] bit us (differencing
off by up to 82 Elo) and why the Cheese handicap ladder proved unreliable. Three or four rungs
of an external family give absolute placement AND free transitivity checks.

**The data number that lands separately.** Ceres reports training positions per net ranging
**3.8B to 34.6B**. At our measured 4.1M unique positions/day that is **2.5 to 23 YEARS** of
continuous RL. This is the starvation claim from the 3d5df283a entry, now confirmed against an
external comparable instead of my own estimate.

**Confounds and traps, stated before running:**
- ⚑ **The published Elo column confounds SIZE with DATA** — later nets are both bigger and
  trained on more positions (Sep 24 → Apr 25). "+151 from 512×15 → 640×34" is an UPPER BOUND
  on the pure architecture effect, not an architecture measurement. Fine for locating
  ourselves; NOT quotable as "architecture is worth 151 Elo".
- ⚑ **Their Elo is their measurement under their conditions.** Treat the ORDERING as reliable
  and re-measure the SPACING ourselves — which running multiple rungs gives us for free.
- ⚑ **Our search is tuned for our net** (`c_scale` 0.1 calibrated on us). Both sides running
  our MCTS is the right net-vs-net contrast, but it is NOT neutral —
  [[same_setting_both_sides_is_not_neutrality]]: a shared knob still moves B−A when one net
  gains more from it. Bias favours us by an unknown amount. Mitigate by running ≥2 search
  configs and requiring the ordering to be stable.
- ⚑ **NODE limits, never time.** A 63M net and a 640×34 net have very different inference cost;
  a time ruler converts that into phantom Elo.
- GPL-3.0 matters only if we ever distribute a derived net. A measurement distributes nothing —
  another reason the diagnostic framing is the right one.

**⇒ REVISED ORDER: D0 → D2 → D1(only if still needed) → D3.** D0 gates the rest: it is free,
it is the fastest, and a bad D0 result changes what D1/D2 are even for.

## The three training-side diagnostics, and what each one isolates

### D2 — LABEL QUALITY. **Run this first: sharpest, cheapest, existing rig.**

Isolate the TARGET axis: hold architecture, positions, donor, steps, seed and draw sequence
constant, vary ONLY what the net is trained to predict.

Rig: `scripts/retarget_retrain.py`, which already does exactly this and is hardened against
the ways it could go wrong (identical seed per variant, paired draw sequences, a shard-list
change guard, a draw-sequence de-pairing guard, and a dead-override guard that fails if a
variant sets a key that reaches nothing). Two variants on the SAME pool:

- **(a) `prod`** — our production targets: `0.69·sf_wdl + 0.31·search_wdl` value, our search
  policy target.
- **(b) `bt4`** — BT4's policy and value on the SAME positions.

New machinery is one buffer view, `_BT4TargetBuffer`, modelled directly on the existing
`_SoftPolicyAsMainBuffer` (`retarget_retrain.py:350`) and activated per-variant by
`rig_bt4_targets=1`. ⚑ It MUST carry the same fail-loud guard as its model: if a sampled row
has no BT4 label, raise — never fall through to the stored target, because a silent fallback
turns arm (b) into a second copy of arm (a) and the experiment reports a null. That is this
codebase's signature defect in its exact canonical form.

**Reading, pre-committed:**
- `bt4` beats `prod` in the paired arena by more than the instrument's resolution ⇒ **our
  LABEL PIPELINE is a real limiter.** Most actionable outcome available: it says fix targets,
  not architecture, and it is upstream of the anti-engine goal rather than in tension with it.
- No separation ⇒ **our labels are not the limiter.** Combined with the absorption result
  (11.6× in-window fit, zero held-out gain), that points at loop DYNAMICS — self-reference —
  and rules out the cheaper explanation.

### D1 — ARCHITECTURE CEILING. **Needs a negative control or it is uninterpretable.**

Fit our 63.08M architecture to BT4's function on abundant positions; track held-out fidelity to
BT4 (top-1 policy agreement, policy KL, value MAE) against positions seen.

⚑ **A fidelity number alone means NOTHING** — there is no scale on which "we reached 0.72
top-1 agreement" is good or bad. The control that supplies the scale: run the SAME distill into
a **deliberately smaller net** (e.g. 256-dim). Then:
- 63M ≫ small ⇒ capacity is being used; the architecture is doing work and the curve tells us
  where it saturates.
- **63M ≈ small ⇒ we are DATA-limited inside the distill itself, not capacity-limited**, and
  the whole D1 read is void until the corpus grows. Without this control that outcome is
  indistinguishable from "our architecture is fine", which is the opposite conclusion.

This is the [[shuffle_the_labels_negative_control]] / [[a_resampled_pool_measures_itself]]
discipline: a ceiling measured on one pool converges to the pool, not to the architecture.

### D3 — POSITION DISTRIBUTION. Run last; needs external position sourcing.

Hold the target source (BT4) and architecture constant, vary the POSITIONS: our replay
positions vs external positions (PGN / lc0 public data). Separation ⇒ our selfplay diet is
narrow or self-similar; null ⇒ position distribution is not the issue.

## Shared prerequisite, and the one place this silently breaks

All three need BT4 labels on chosen positions. Build once: a labeling pass writing a sidecar
keyed by row, so labels are precomputed and DETERMINISTIC across arms. Inline inference in the
sampler is rejected — it is slow and it de-pairs the arms, defeating the rig's guards.

**⚑⚑ THE POLICY INDEX IS THE FAILURE MODE.** BT4's policy is in lc0's 1858 index; ours is our
own 1858 encoding, and **they agree on only 46 of 1858 slots**. The board-aware map is
`moves/leela_index.py` (verified 0 mismatches over 55,586 moves). A wrong or static map here
does not crash — it silently trains every BT4 arm on a permuted policy, D2 reports a confident
null, and we conclude "our labels are fine" from an experiment that never ran. Gate before any
arm: re-verify the map over a fresh sample and assert 0 mismatches, and additionally assert
BT4's remapped top-1 move is legal in 100% of sampled positions. Also note BT4 consumes lc0's
112-plane encoding, not our 175-plane `v2_threats` (adapter exists; repetition-plane semantics
differ — lc0 counts whole-game, we keep 7 plies).

## What this cannot tell us, stated before launch

- It cannot say whether a BT4-distilled net is a good ANTI-ENGINE net. It is not being asked
  to; the anti-engine objective is a post-training question and no arm here touches it.
- A D2 null is "labels are not the limiter", **not** "labels are good" — the arms share our
  position distribution, so a defect that lives in the positions cancels in the contrast.
  That is what D3 exists for, and it is why D3 should not be dropped if D2 reads null.
- All three inherit the arena's resolution: 800 pairs = ±15.5 Elo, 80% power at ~22 Elo. An
  effect smaller than that is not detectable here and must not be reported as a null with a
  point estimate quoted as if it were meaningful.

## Cost and order

Labeling is paid once per position pool and shared by D2/D3. D2 reuses an existing rig, so its
marginal cost is the labeling pass plus two short retrains plus one paired arena. Order:
**D2 → D1 (with its small-net control) → D3**, one readout window each, per the
one-data-affecting-change-per-window rule.
