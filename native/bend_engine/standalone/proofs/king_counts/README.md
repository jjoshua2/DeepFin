# Actual king-count preservation

`consumer.use_child` proves exact Nat-count preservation of both fixed white/king
and black/king intersections for an actual full-Ply member of
`Chess.legal_moves(table, board)`. Its premises are row partition, coherent
`Position.valid_ep(get_ep(board), board)`, and zero rights. The child theorem accepts
arbitrary tables and raw turns and needs no initial count-one, canonical-turn,
initialization or target-geometry premise. Both colors follow the implementation's
`turn == 1` choice directly.

The actual occupied source decoder supplies king membership. Promotions 1–4 insert
no king, and their actual pawn source has no king under row partition. Ordinary
promotion0 uses the decoded source piece. Destination and capture masks are
king-clear: the actual target producer supplies destination clearance, while an
ordinary EP tag is exactly `pawn && destination == ep`. Coherent EP makes its
`destination xor 8` victim a pawn. Thus the actual removal mask affects each king
intersection only at the owned source. Structural deletion/insertion counts preserve
that intersection exactly; nonking moves erase an absent source bit.

`Emission.destinations` keeps arbitrary fuel, coherent actual empty flags, multibit
masks and individually certified physical tail cons. `after` takes the actual
`(table, targets)` pair. Promotions use selectors 1–4 and flag0; ordinary moves use
promotion0 and `Bool.to_u32(pawn && destination == ep)`. No tail deduplication occurs.
Zero rights makes both actual castle guards false; nonzero-right castling is outside
this increment.

`consumer.use_replay` proves preservation through actual ordered `Protocol.moves`,
including first complete-Ply lookup and each command cons, with the output table
equal to the exact input. Its conditional parent premises are rows/canonical turn,
initial coherent EP and zero rights. It retains exact `Init.run`, depth17,
table-count128 and extra64. Existing initialized EP preservation supplies coherence
and row/canonical invariants for every next step. Surviving games retain both initial
Nat counts; `None` remains a conditional failure result.

`Initialized.use_replay` retains every PR1080 result as well: metadata, coherent EP,
rights, back-rank pawn exclusion, both color popcounts at most16, both pawn/color
popcounts at most8, and pawn counts bounded by their initial counts.
`Root.use_replay` derives initial EP coherence, material premises and king
popcount-one from actual `Position.valid`, and proves both actual U32 king
intersection popcounts are one for surviving replay results. Initial rows,
canonical turn and zero rights remain explicit; `Position.valid` alone does not
assert row partition.

The malformed-EP fixture has valid rows and zero rights but EP target16 with pawn8
and king24. Its actual flag1 move8→16 erases king24. This demonstrates why initial
EP coherence cannot be dropped. Other fixtures retain twelve physical occurrences,
duplicate commands, all promotion/EP fields, an exact ragged table, raw turn2 and
the first full-Ply text match. Fixtures support the production theorem.

No full child `Position.valid`, replay acceptance or full legal-move correctness
claim is made. This increment adds the missing king-count obligation to the prior
conditional invariants.

Run the fail-closed qualifier from `native/bend_engine` with fresh external paths:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m standalone.proofs.king_counts.qualify_king_counts \
  /path/to/pinned/bend --checker-manifest /external/checker-tree.json \
  --report /external/fresh/qualification.json --evidence-dir /external/fresh/checks
```

The immutable checked candidate adds only this suite to PR1080 exact head
`91e4eadded397ffe388379e2566e06ceb03145d8`, tree
`12a03baf2f86bac95ba5b480cb04838dfee0cbe5`. Reused dependencies must match its Git
blobs. Qualification checks three entries and41 strict mutations, including field,
EP/row/source/clearance, count-one, duplicate, consumer/replay, exact-table and all
four initialization obligations. A control must fail with unequal types at its
declared obligation; parser, inference, affine-use, timeout or resource failures
earn no credit.

Runtime: Bend2.0.21+U64 pin `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2,
84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Checks run serially on CPUs1/3 with two threads, 6GiB AS/RSS, 16MiB stream caps and
86400seconds per positive or negative. Each check hashes source, immutable snapshot,
all84 runtime files, exact Git head/tree and canonical single-file mutant contents
before and after. Commands, raw streams, rusage and hashes remain external.
