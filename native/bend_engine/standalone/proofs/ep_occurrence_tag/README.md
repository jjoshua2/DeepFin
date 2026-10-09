# Actual EP flag and priority-decoder provenance

Stacked on PR1046 `26f10a2cc672e2010fdc058648ad01dd46884d7c`.

`consumer.legal_member` takes full-Ply membership in the actual
`Chess.legal_moves(table, board)` plus flag 1 and proves promotion 0, a pawn
bit at the source, destination equal to the parent EP field, and destination
below 64. `consumer.decoded_kind` connects that pawn bit to the actual priority
decoder and proves `Chess.piece(board, source) == 0`.

No caller source bounds, target-hit, promotion-shape, partition, canonical-turn,
parent-EP or accepted-root premise is hidden in this producer contract. It
applies to the exact arbitrary input array, including initialized arrays; it
does not assert initialized geometry after arbitrary writes. Board partition
and accepted-root validation remain separate contracts in inherited suites.

`Source.destinations` retains the actual fuel and requires its empty argument
to equal `U64.is_zero(bb)`. The nonempty branch obtains CTZ range from the pinned
all-U32 structural proof. Flag correlation follows the implementation's pawn
bit and destination/EP comparison. Promotion blocks overwrite flag with 0.
`Source.after` consumes the actual `(table, targets)` pair, while scan, both
castle wings and the actual safety-filter suffix preserve the predicate of
every incoming full-Ply node. Arbitrary certified tails may contain duplicates.

Concrete checks cover multi-bit targets with repeated tails, ordinary versus
EP flags, promotions, EP64, nonpawn sources and exact scan-after input table.
The stale EP example is partition-valid and is an actual legal occurrence for
its specified arbitrary table, yet removes the opposing king. Its parent EP
is invalid. The arbitrary-table 32-to-48 child EP failure is also retained.
This increment supplies a missing producer bridge toward EP king survival;
king survival, accepted-root preservation, final legal suffix array equality
and full legal-move correctness remain unproved here.

Qualification uses pinned Bend 2.0.21+U64
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2 and the verified
84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
One checker runs at a time on CPUs 1 and 3, with a 6 GiB address/RSS ceiling,
16 MiB output-file cap and 86400-second positive/negative deadline. Parse,
affine, timeout and resource failures cannot qualify a negative control.

```sh
BUN="$HOME/.bun/bin/bun" python3 -B -m native.bend_engine.standalone.proofs.ep_occurrence_tag.qualify_ep_occurrence_tag /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json --report /fresh/qualification.json --evidence-dir /fresh/evidence
```

Frozen source closures, Git/base identities, commands, raw checker output,
negative mutations and hashes are recorded outside all source checkouts.
Internal independent review precedes draft publication. No production source,
GPU launcher, runtime deployment or merge is changed by this increment.
