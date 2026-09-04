# PREDICTIONS — BT4 cascade gate cross-tab (task #247)

Written **BEFORE** any number was computed. Rig `scratchpad/bt4cascade/xtab.py`.
Reads ONLY banked artifacts under `scratchpad/target_vs_bt4/`. Zero GPU, zero Stockfish,
no live change.

## What is being decided

Josh's design: a **cascade**, not a blanket screen.

1. gate 1 (FREE, stored): is the target's argmax inside SF's MultiPV-6? (`tgt_listed`)
2. gate 2 (FREE, stored): the target's own `top1_mass`
3. only rows failing the gates get a **BT4 forward pass**
4. downweight where BT4 corroborates SF; **keep** where BT4 prefers our move

The open question this rig answers, and ONLY this one: **does gate 2 add anything
INSIDE the not-listed rows, or is it redundant with gate 1?** The banked C=0.7815
is the `tgt_listed=False` cut; the banked confidence table is over ALL disagreement
rows. Those are two different populations and nobody has crossed them.

## ⚑ POPULATION, declared before measuring

Phase 2's C table says n=1158 disagreement rows (no trained mask). Phase 2's dQ /
screen tables use `trained = ~postckpt_mask` (n≈1007 with features). **Same names,
different populations.** PRIMARY = disagreement rows, `trained` mask APPLIED, finite
Q on cand 0 (`tgt`) and 1 (`sf`). The no-mask population is reported as a robustness
row, never as the headline.

`C = P(BT4 prefers SF's move) = mean(Q_tgt - Q_sf < 0)`. Tie rate reported
separately; if ties exceed 2% the definition is ambiguous and must be restated.

## ⚑ RESOLUTION BEFORE THRESHOLD (the rule #440 broke twice)

Cell n's and the bootstrap half-width are computed and printed FIRST. If the smaller
of the two compared cells has **n < 30**, the verdict is **NO RESOLUTION** — that is a
stop, not an invitation to lower the bar.

## Deciding statistic — one, pre-committed

    dC = C(not-listed AND top1_mass < 0.5) - C(not-listed AND top1_mass >= 0.9)

95% CI by 10,000-resample bootstrap over rows.

| observed | verdict |
|---|---|
| CI lower bound **> 0.10** | **GATE 2 EARNS ITS PLACE** — gate on both; skip the BT4 call on high-confidence not-listed rows and keep them |
| CI contains 0.10, lower bound **> 0** | **WEAK** — trigger BT4 on `not_listed` alone; put `top1_mass` in the downweight FUNCTION, not the gate |
| CI contains **0** | **REDUNDANT** — gate 1 already captures it; do NOT add a second knob for nothing |
| CI upper bound **< 0** | **INVERTED** — contradicts the banked confidence table ⇒ one population is mis-specified. STOP, re-derive, do not build the cascade |

## Numbered predictions (scored honestly, misses included)

- **P1 REPRODUCTION CONTROL.** C on the `tgt_listed=False` cut reproduces the banked
  **0.7815** to within ±0.02. *If this misses, the rig is wrong and every other
  number here is void.* This is the negative-control equivalent and it comes first.
- **P2** The four banked confidence-bin C values reproduce to within ±0.03:
  0.702 / 0.578 / 0.577 / 0.380.
- **P3 EXACT COUNTS.** not-listed n = **326** ± 15 (16.3% of 2000, before the trained
  mask); after the trained mask, **284** ± 20.
- **P4** not-listed rows skew LOW confidence: median `top1_mass` on not-listed is
  **below** the median on listed.
- **P5 THE RISK I EXPECT TO BITE.** The `not-listed AND top1_mass >= 0.9` cell is
  **SMALL — I predict n between 15 and 45**, i.e. plausibly under the n=30 floor. If
  it lands under 30 the honest outcome is NO RESOLUTION at this n, and the follow-up
  is a wider banked sample, not a softer threshold.
- **P6** dC lands in **[+0.10, +0.30]** ⇒ I expect GATE 2 EARNS ITS PLACE. Stated so
  a WEAK or REDUNDANT reading is a recorded miss, not a re-framing.
- **P7** Among not-listed rows where BT4 prefers OUR move, mean dQ is **positive**
  (these are the keep-worthy rows, the ones a blanket top-6 drop would destroy).
- **P8** The share of ALL rows the cascade would downweight is **11-14%**
  (≈16.3% x 0.78), against 16.3% for a blanket not-listed drop.

## What a PASS here does NOT establish

- **Not Elo, and not even target quality.** This is a cross-tab of a banked ruler. It
  sizes the gate; it does not show the gate helps. The training arm is a separate,
  GPU-gated entry that does not exist yet.
- **BT4 is the FILTER here, so BT4 can never judge the result.** Readout must be a
  different non-SF net (C1 ladder) plus a paired arena. Never SF at any budget.
- **One checkpoint era.** C=0.7815 is from the iter100/218-era shards; knob effects
  reverse sign between checkpoints here, so a re-measure on current shards is owed
  before any threshold ships.
