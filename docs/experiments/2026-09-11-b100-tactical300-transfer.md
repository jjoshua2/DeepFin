# B100 Tactical300 mass-transfer policy experiment

Registered September 11, 2026.

## Status

Implementation and focused tests are prepared in this branch. No real corpus rewrite,
training run, arena, playing-strength result, or production adoption is claimed.

**Real Tactical300 materialization and training are now explicitly blocked on the
stacked retrospective G10 calibration.** That calibration must measure how often a
one-node BT4 policy disagreement with a large d9 Stockfish margin is subsequently
vindicated by the already-recorded d10/d12 searches, including the fraction that can
actually be adjudicated after G10's narrowed search rosters. A later training
admission must pin that completed calibration receipt or adopt a separately
registered depth-stable gate justified by it.

The existing incumbent remains **B100**: pure BT4 policy supervision sharpened at
teacher temperature 0.5, with the original Stockfish-derived value target unchanged.

The earlier `B100Tactical100` experiment is complete and unresolved: its 400-simulation
comparison scored -13.6 Elo versus B100 with paired 95% interval [-52.4, +24.9].
That experiment broadly attenuated BT4 moves using a 100 cp allowance, exponential
decay and a 0.1 multiplier floor. It therefore changed many ordinary positions but
could only alter relative odds by at most about 10x, and stored zero BT4 mass stayed
zero.

This experiment tests a materially different hypothesis: **use Stockfish only when
its tactical evidence is overwhelming, and then correct the policy strongly.**

## Hypothesis

A one-node neural teacher can be strategically stronger than Stockfish policy
imitation overall while still miss tactics that a short search exposes. Conversely,
a large shallow-Stockfish gap is not automatically truth: a deeper search can overturn
its preferred move, and a neural teacher can encode positional information that a
short tactical search has not resolved.

The mechanism implemented here uses a d9 best-vs-second gate of **strictly more than
300 effective centipawns**. That 300 cp threshold is an experimental candidate, not
a calibrated confidence statement. Do not change ordinary B100 targets below or at
the boundary.

When the gate fires, move probability mass toward the Stockfish-best move/set instead
of merely multiplying inferior moves downward. Whether d9 alone is strong enough to
trigger that transfer is the subject of the mandatory calibration below.

## Target construction

Let `B` be the normalized stored B100 float16 policy over legal moves.

### Ordinary positions

Find the exact Stockfish-best non-mate move set. If the best-vs-second-best
non-mate gap is `<= 300 cp`, preserve B100 bytes exactly.

If the gap is `> 300 cp`:

- remove 50% of the total probability mass on inferior non-mate moves;
- add that mass to the SF-best set;
- preserve B100 relative odds inside the SF-best set;
- if B100 assigned zero total mass to the SF-best set, distribute the transferred
  mass uniformly across the tied SF-best moves.

This can deliberately restore support to an SF tactical winner that B100 stored
at zero probability.

Example: if the unique SF winner has B100 mass 0.01, the 50% transfer changes it
to about 0.505 rather than the roughly 0.09 produced by a 10x relative-odds
correction.

### Winning mates

If one or more winning-mate moves exist, ignore encoded mate-distance differences.

Transfer 75% of all probability mass outside the winning-mate set into the
winning-mate set. Preserve B100 odds within that set, or use a uniform split if the
set had zero B100 mass.

### Losing-mate alternatives

If no winning mate exists but some moves are losing mates:

- transfer 75% of losing-mate mass to the best non-losing move set;
- independently apply the ordinary 50% transfer among non-losing moves if their
  best-vs-second gap is strictly greater than 300 cp.

The donor sets are disjoint.

### All forced losses

If every legal move is a losing mate, preserve B100 bytes exactly rather than
inventing a mate-distance policy.

## Controlled quantity

Only `policy_target` changes.

The producer inherits and rechecks the original schema-3 raw/source join, the
qualified B100 parent recipe, legal support, original q-policy reconstruction,
B100/SF shard structure, non-policy compressed-byte parity, source stability and
final publication guards.

The original Stockfish value target and all other non-policy arrays remain unchanged.
Shared-trunk learning can still alter the learned value head after training; the
supervision intervention itself is policy-only.

## New implementation

`scripts/sf_tactical_transfer.py` provides:

- `tactical_transfer_target`, the pure row-level target operator;
- strict `>300 cp` gating;
- 50% ordinary and 75% mate mass transfers;
- zero-support restoration for SF-best recipients;
- exact categorical mate handling;
- full-corpus rewriting from qualified B100 plus original SF/raw lineage;
- fresh-namespace publication, STOP/disk guards and failure preservation;
- per-shard source/output hashes and storage identities;
- aggregate diagnostics for gate count, transferred mass, SF-best mass before/after,
  support gains/losses and storage error.

The completed summary is `bt4_sf_tactical_transfer_summary.json`.

Focused tests cover:

- exact 300 cp identity versus 301 cp activation;
- zero-mass winner restoration;
- tied-best B100 ratio preservation;
- winning-mate 75% transfer;
- disjoint losing-mate and ordinary transfers;
- all-forced-loss identity;
- ambiguous score-domain rejection;
- real shuffled raw/SF/B100 joins;
- byte preservation of all non-policy arrays;
- source-pin and partial-output refusal.

## Mandatory retrospective calibration gate

Before **any real Tactical300 materialization or training**, use the qualified G10
rows that already contain deeper recorded Stockfish observations plus a source-bound
BT4 policy to measure the failure mode that the original recipe cannot see.

For ordinary, non-mate rows, report at least the following threshold grid using the
d9 best-vs-BT4-top disagreement:

- `>100 cp`, `>200 cp`, `>300 cp`, `>500 cp`, and `>1000 cp` d9 margins;
- BT4 top-probability strata `<0.25`, `[0.25,0.5)`, `[0.5,0.75)`, and `>=0.75`;
- d10 versus d12 final selected depth;
- whether the later narrowed roster scores both the d9-best set and the BT4-top set.

Among rows where both competing sets are scored at the final recorded depth, classify
whether deeper SF:

1. still favors the original d9-best set;
2. favors the BT4-top set;
3. ties them; or
4. favors neither set / a third move.

Also report the one-sided roster cases separately (`BT4 scored / d9-best absent`,
`d9-best scored / BT4 absent`, neither scored). A narrowed roster is evidence about
what was retained by the prior search, not an invented score for a move that was not
searched. Do not silently count those cases as pairwise wins.

For mate rows, separately report whether a d9 winning-mate set persists at the final
recorded depth and whether BT4's top set agreed with it. Do not convert encoded mate
distance differences into ordinary centipawns.

The diagnostic must preserve source/config/game/ply identities and its BT4 sidecar
provenance. Deeper d10/d12 restricted searches are **not ground truth**: d12 is
selected by the d10 gate, later rosters are narrowed, and searches share historical
transposition-table state. The purpose is calibration of the intervention, not a
claim about optimal play.

### Decision rule

The calibration is intentionally asymmetric because a strong d9-only transfer can
remove useful BT4 information.

- If **5% or more** of adjudicable `>300 cp` BT4-disagreement rows are later pairwise
  won by the BT4-top set, a d9-only Tactical300 training treatment is **blocked**;
  a follow-up recipe must require deeper-search stability (or otherwise justify a
  different threshold/intervention).
- If the observed rate is below 5%, report it together with its denominator,
  adjudicability and one-sided roster counts. This does **not** automatically admit
  training; it only removes this specific mandatory stability trigger.
- Regardless of that rate, a completed calibration receipt must precede full-corpus
  materialization/training admission.

The 5% boundary is an experiment-allocation rule, not an accuracy confidence bound.
Do not tune the threshold repeatedly against the same diagnostic bank and then call
the selected value confirmed.

## Subsequent geometry and strength experiment

Only after the calibration gate is satisfied should a bounded original-corpus
geometry pass inspect:

1. fraction of rows with the ordinary selected gate;
2. mean B100 mass already on the SF-best set in gated rows;
3. total probability mass transferred by ordinary and mate rules;
4. mean/quantiles of target total variation and support gains.

A real 18,910,484-row Tactical300 treatment must not be produced merely because the
rewriter exists. Its admission must bind the chosen post-calibration rule, completed
producer receipt, and exact historical B100 training runtime/schedule.

If eventually admitted for training, keep the historical B100 initialization/runtime,
one-epoch row schedule and original SF value supervision fixed. Compare the resulting
checkpoint directly with B100 at the same 400-simulation / swapped-opening setting
used for the prior tactical screen.

Do not combine this policy change with Ceres policy, neural-value mixtures,
adaptive G10 values, AVI targets, a longer training horizon or search-prior changes
in the first strength test.

## Scope limitation

The original 18.91M source is a single-phase d9 corpus. The current mechanism is
therefore a **d9 candidate recipe**, not evidence that d9 is sufficiently reliable.
The stacked G10 calibration exists specifically to determine whether a deeper-stable
gate is required before this mechanism can be trained.
