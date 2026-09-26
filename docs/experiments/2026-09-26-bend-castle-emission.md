# Actual castling emission and guard provenance

## Scope and acceptance

Base: PR #889, `8608169d517a3eefeffed459bb7d7c0631f7c4c4`.
Pinned compiler: `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, unchanged 84-source fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Target the actual output-list theorem, not another coordinate-only model: show that
`Chess.castle_side` returns its exact input tail or one guarded, correctly formed new
move followed by that tail. Derive membership provenance and actual update consistency
from the true producer guard. Keep existing accepted laws and production code unchanged.

Acceptance is a complete six-law public source consumer with meaningful semantic and
policy controls, bounded current-pin native producer checks against an independent
coordinate/set reference, original compiler/pin tests and unchanged repository lint.
Tests are opt-in and CPU-only; no routine perft increase, GPU, model or training work.
No merge or deployment. Self-review only unless a separate review actually occurs.

This record is written after successful local source construction and the first
native report, before final whole-suite and hosted qualification. It is not presented
as a prospective performance experiment or retroactively preregistered measurement.

## The six contracts

The new suite lives at `native/bend_engine/standalone/proofs/castle_emission/`.
Its public LAWS file imports actual Chess and the independent Board invariant.
PROOF explicitly imports LAWS and discharges every obligation.

1. Actual output is the input tail or a single exact guarded move prepended to it.
2. A result member absent from the input tail entails the input producer guard.
3. That new member equals the complete computed castling Ply.
4. A true guard supplies initial rook-landing freshness in the independent predicate.
5. Applying a new member preserves input Board consistency, composed with the existing
   public castling update theorem rather than assuming output consistency.
6. A false guard preserves the complete actual `(table, tail)` pair.

The guard contains the selected castling-right bit, owned source king, owned corner
rook and empty between path. A source proof, not a textual source guard, connects it
to the actual function. Structural Word proofs reflect actual masked occupancy into
independent freshness and narrow that freshness to the rook landing square.

## Domains that must remain distinct

The input tail is arbitrary and can already contain flag-two moves or duplicates.
The membership results require absence from the input tail; a closed counterexample
with an existing castling entry and false current guard is retained. The extension
law is occurrence-aware even when values duplicate the tail.

Source statements quantify arbitrary actual affine tables. Positive output-list
refinement does not prove table-content preservation or lifetime; false-guard
rejection separately preserves the complete pair. A nonempty source witness uses a
zero-filled arbitrary table, establishing satisfiability, not attack-table correctness.

Native tests build actual current Tables.build and call actual castle_side. The
independent reference checks geometric start/transit attacks, not a candidate-provided
answer. Destination-attacked cases are intentionally retained by castle_side; later
legal_moves filtering and its semantic king-safety theorem remain separate.

No claim covers the whole legal_moves list, valid metadata history, king counts,
legal reachability, final king safety, legal-generation soundness/completeness,
native ownership/lifetime, performance or new Python application migration.

## Execution status

Six public laws and their importing consumer passed locally with exact safe output.
The first native report passes 852 distinct producer requests, all four build modes,
and three compiled/executed actual producer corruptions. Complete final qualification
and any failed intermediate attempts will be recorded below with their exact sources.
