# B100 Tactical300 mass-transfer policy experiment

Registered September 11, 2026.

## Status

Implementation and focused tests are prepared in this branch. No real corpus rewrite,
training run, arena, playing-strength result, or production adoption is claimed.

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
imitation overall while still miss tactics that a short search exposes. In
particular, a d9 Stockfish search can identify positions where one move is hundreds
of centipawns better than every alternative, or where a winning mate exists.

If Stockfish's best non-mate move beats the next-best non-mate move by **strictly
more than 300 effective centipawns**, treat that as a sparse tactical-confidence
gate. Do not change ordinary B100 targets below or at the 300 cp boundary.

When the gate fires, move probability mass toward the Stockfish-best move/set
instead of merely multiplying inferior moves downward.

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

## Intended next experiment

After independent implementation review and a bounded real-corpus diagnostic,
materialize the full 18,910,484-row treatment only if the intervention frequency
and target geometry are sensible.

The most useful pre-training readout is:

1. fraction of rows with the ordinary `>300 cp` gate;
2. mean B100 mass already on the SF-best set in gated rows;
3. total probability mass transferred by ordinary and mate rules;
4. mean/quantiles of target total variation and support gains.

If admitted for training, keep the historical B100 initialization/runtime, one-epoch
row schedule and original SF value supervision fixed. Compare the resulting
checkpoint directly with B100 at the same 400-simulation / swapped-opening setting
used for the prior tactical screen.

Do not combine this policy change with Ceres policy, neural-value mixtures,
adaptive G10 values, AVI targets, a longer training horizon or search-prior changes
in the first strength test.

## Scope limitation

The original 18.91M source is a single-phase d9 corpus. This implementation therefore
tests **sparse strong correction from the same d9 evidence used by the earlier
tactical experiment**, not d10/d12 stability.

A later G10 transfer should be a separate experiment and should preferentially
require agreement of the selected move across the recorded deeper adaptive
observations before treating `>300 cp` as high-confidence search evidence.
