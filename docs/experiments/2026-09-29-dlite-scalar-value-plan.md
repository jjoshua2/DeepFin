# D-lite: cheap scalar Stockfish value as a separate candidate

Status: prospective cost and training plan. No new strength result or production
recipe is claimed. This extends the [current 500M plan](2026-09-29-500m-practical-next-tests.md)
by testing affordable SF value supervision separately from SF policy correction.

## Why test it

D and E use the same per-row half-BT4/half-Ceres policy. D's value is approximately
one third each SF, BT4 and Ceres; E's is half each neural teacher. The direct
1,152-game result is E minus D **-12.37 Elo**, 95% interval **[-27.18, +2.39]**.
That is a reason to keep testing SF value, not a proved recoverable 12 Elo.
Historical D also inherits stored target rounding. See the
[matched arena](2026-09-29-e-vs-d-576-strict.md).

A scalar root search supplies a value without paying for all-legal-move SF policy
labels. Shallow scalar search is a different teacher profile from D's historical
adaptive/full-width search. Matching its mixture coefficient does not reproduce D.

## First target comparison

This is a matched selected-route comparison. Its chosen-teacher policy equals the
D/E neural policy mixture in expectation, not historical per-row policy bytes. Keep
the position bank, row order, neural labels, chosen-teacher route, main policy,
loss masks, initial weights and training schedule identical. The control uses the
authenticated stored float16 selected-neural value. The candidate widens that
same stored value to float32, blends it with the float32 D-calibrated scalar
value, then rounds the resulting main WDL once to float16:

```text
q_control = q_chosen
q_candidate = (1/3) q_SF_scalar + (2/3) q_chosen
```

With a source-independent fair BT4/Ceres route, the candidate has equal-thirds
teacher weights in expectation. It is not the exact per-row three-teacher average;
selected targets introduce sampling variation, and the chosen neural value was
already rounded once before the candidate blend. Soft cross-entropy is linear in
its target before nonlinear target transformations, masking or rounding. Preserve
the existing selected route and record these differences rather than claiming byte
identity. A treatment that averages both neural teachers would add label cost and
change the comparison.

Use the historical SF CP-to-WDL mapping (slope 0.006, draw width 120 CP), with raw
CP/mate scores kept separately from native UCI WDL. Require side-to-move orientation
and full recorded UCI history. Ceres retains its existing calibrated dual-value
blend. Write the candidate into the same active main-WDL target path as the control;
keep auxiliary heads and outcome fractions identical. Missing SF scores are missing
labels, never zero-CP replacements. No policy correction belongs in this comparison.

## Cost screen before committing training

Measure 10,000 uniformly hash-selected, history-bearing positions from the closed
38,615-row d8_c4 bank at scalar d6, d8 and d10, with one and eight persistent
single-thread engines. Each depth needs its own inclusive measured wall and accepted
row count. Report startup, TT reset, search, history/parser and serialization time,
phase/material composition, failures and engine scaling. Keep TT cold between
independent labels while retaining the qualified tablebase cache. Request strict
six-man Syzygy with the 50-move rule enabled. The bank's all-move d8 scores are an
agreement proxy, not deep truth or an Elo ruler.

The often quoted 36.57 d8 and 35.74 d10 labels/s came from 64-row single-engine
screens dominated by reset overhead. Later retained-tablebase measurements changed
that overhead. Those old rates cannot select the current depth or establish fleet
throughput. Choose the cheap depth after observing both actual cost and label
stability; greater nominal depth alone does not establish better training targets.

At 95% availability, 500M scalar labels alone require **203.05 accepted labels/s**
to finish in 30 days. If all 58.09M legacy rows qualify and retain their existing value profile without
SF re-labeling, the remaining 441.91M require **179.46/s**. A separate full-cost refinement of 10%
raises these thresholds to 223.36/s and 197.41/s if each refinement costs the same
as the base label. Warm continuation has a different cost and must be timed.

These thresholds omit SF-origin generation, strict replay, retries and contention.
When SF generation and scalar labels use the same CPU pool, add their compute
requirements; they do not overlap for free. Likewise, CPU/GPU co-location needs a
measured interference cost. Reusing an SF-origin root score can reduce labeling
cost, but its search profile and value calibration must be recorded and tested.
The GPU source/label/training budget remains a separate constraint.

## Deeper refinement and the strength decision

First decide the shallow-only candidate against its unchanged selected-neural
control. Deeper refinement of roughly 10% is a later separate intervention, with a
frozen cheap router and an equal-search-budget random control. Possible inputs are
material/ply, chosen-neural entropy, or the already-paid scalar SF versus chosen
neural value gap. Computing BT4-versus-Ceres disagreement everywhere would require
both neural labels and erase part of the selected-label saving. MultiPV1 does not
supply a runner-up move margin.

The scalar screen can establish cost and agreement only. Before a training launch,
freeze the actual matched data/checkpoint, epochs, few-hour compute budget, strict
six-man arena and success/kill rules in this record. Do not spend days on a small
proxy result, promote an untrained router, or promise a month-scale 500M run from
single-engine arithmetic.
