# Actual legal-child metadata

Stacked on PR1040 `a2d0d37928dec8681f8081c8064e71d8c32a3d3a`.
Production and every prior proof are unchanged.

`consumer.legal_children` consumes an arbitrary actual input array and establishes
`Children.every` for the exact `Chess.children` of the actual `Chess.legal_moves`
output. Each child has a valid piece/color partition, the opposite canonical turn,
and EP either below64 or exactly sentinel64. Parent assumptions are precisely
`Board.valid(b)==True` and `get_turn(b)==Bool.to_u32(white)`.
No parent EP range/coherence, king existence or uniqueness, lookup contracts,
target geometry, initialized builder, king safety, or full legal-position
reachability is assumed or established. Partition validity has its existing PR1040
meaning, including boards without kings.

`Source.bend` carries bounded source squares and the exact full-Ply field
alternatives through the actual scan, both guarded castle producers, and the
actual full/fast legal filter. The table/move-list pair is consumed once at each
continuation. Scan source bounds compose the prior all-U32 bit-squares range
result; castles use their actual4/60 source constants. Promotion fields and flags
are certified by the existing implementation-coupled classifier.

`Range.bend` unfolds the actual make_move EP projection. Its trigger is source
kind zero and `src xor dst ==16`, regardless of promotion/flag. XOR cancellation
reconstructs `dst==src xor16`;64 checked source cases bound the midpoint.
The arithmetic lemma allows any U32 destination and kind and retains source
bounds; it does not assume destination geometry or a bounded destination.
The prior checked turn-flip and PR1040 full-Ply partition consumer are composed
in `Post.bend`. `Children.lift` preserves the parent, list order, and every
duplicate occurrence, including an arbitrary list with per-occurrence
certificates.

The fixtures exercise actual legal outputs for ordinary moves, EP both colors,
all four promotions and both castle wings/colors, repeated Queen/Knight/Queen
children, empty lists, and an unrestricted maximum-U32 parent EP field.
They also check two concrete limits:

* With a white pawn8, white king0, black king24 and stale EP16, the actual
  zero-leaf attack array emits and retains `Ply{8,16,0,1}`.
  The EP guard rejects16, but the emitter still tags the normal advance because
  destination equals metadata. The update removes24; the next king selection
  returns64. Both parent king planes were nonempty and the partition and turn
  were valid. Thus these premises do not imply nonempty next-side kings.
* An arbitrary leaf attack mask can emit a pawn capture32→48. The actual update
  writes EP40, which is in range but fails the next black side's EP guard.
  EP coherence is not an invariant under unconstrained attack tables.

These are proof-scope obstructions, without any production fix or assertion about
initialized-table reachable play.

Pinned qualification: Bend2.0.21+U64
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2,84 checker files,
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The checked full fixture import closure is frozen and bound to the exact head
and unchanged base dependencies. Eight coupling mutations and eight concrete
false witnesses must each reject at the intended typed obligation with distinct
expected/observed terms. Parse, usage, resource and timeout failures get no credit.

Each checker runs serially with wall86400s, CPU affinity1,3, two worker threads,
6GiB address-space/RSS allowance and16MiB per-output-file cap.
Raw stdout/stderr, commands, timing/RSS, source/checker hashes, full mutant
closures and control classification are retained outside the checkout.

```text
BUN=~/.bun/bin/bun python3 -B -m native.bend_engine.standalone.proofs.child_metadata.qualify_child_metadata /tmp/deepfin-king-away-checker-aaeb9bc --checker-manifest CHECKER_TREE_JSON --report FRESH_REPORT_JSON --evidence-dir FRESH_EVIDENCE_DIR
```

Evidence root:
`~/chess-artifacts/deepfin-child-metadata-20261005/evidence/2026-10-05`.
A development PASS is a checked partial milestone; publication requires the
immutable-head qualification, independent internal review, and final evidence
audit. CI is reported separately from this bounded checker qualification.
