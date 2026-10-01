# Accepted castling destination safety

Three public contracts link actual retained moves to their destination check.
The prepared-filter law needs a producer certificate; the complete-generator law
derives it from actual membership and flag 2. Both return the complete original
array and False from in_check, for arbitrary source arrays and Boards. They do not
assume initialized attacks, Board consistency, valid side metadata or king counts.
Those broad source domains do not certify unsafe native layouts/indices.

The initialized law replaces the actual check with independent target-centred
coordinate geometry at the returned move's destination. It requires a consistent
initial Board, a singleton moving-side king at home, valid Boolean side metadata,
and initialization depth17 / 128 slider blocks / 64 extras with arbitrary seed.
The public caller does not supply a castling guard, final singleton, query bound,
filter decision, or correct attack masks: they are derived from existing producers.

This is destination safety of castling members of actual full legal_moves, not
just a query performed after generation and not the isolated single-side adapter.
Starting/transit geometric safety for the complete generator, universal
forward/reverse attack equivalence, historical rights, other move classes,
completeness, and absence of duplicates remain separate.

Spec's fixed-table folds are proof-only models using actual in_check answers.
Filter proves complete result-pair equality using existing storage preservation.
Accept derives retained-member implications by list induction. Generated connects
both castling calls and the ordinary scan's preserved table to final filtering and
uses the accepted actual-generator castling provenance. Logical duplication of a
membership certificate is explicitly derived by list/sum induction, not assumed.
Geometry composes the resulting False with the existing initialized stage theorem.

Commands (opt-in, pinned compiler; no routine perft change):

```sh
BUN=/path/to/bun python3 native/bend_engine/standalone/proofs/accepted_castle/focused.py /path/to/pinned/bend --report /tmp/accepted-source.json
BUN=/path/to/bun CC=clang python3 native/bend_engine/standalone/proofs/accepted_castle/verify_native.py /path/to/pinned/bend --report /tmp/accepted-native.json
```

The focused runner uses Bun --smol and serializes the expensive complete consumer
before lightweight semantic mutations. It requires exact safe checker output;
timeouts, parser/ownership errors and absent files are not semantic rejections.
The native candidate uses actual Chess/Tables functions and no expected decisions.
Only castling move-set results receive independent oracle comparison; noncastling
move correctness is not inferred. Repeated modes are not independent datasets.
