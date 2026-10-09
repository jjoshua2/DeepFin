# Actual legal-move children preserve board representation

This proof stacks on PR1039 commit
`805a72dc08d7296185065b1d3fcead5661098415`, tree
`67726d621bac19cb68110a852b5dc5214b5f14ba`.

The specialized ordinary, en-passant, typed-promotion and guarded-castle update
proofs already preserve the per-square bitboard partition. This increment removes
the need to choose a specialized consumer or assume post-board validity when
applying an actual generated or actual legal move.

`Post` consumes the exact full-Ply alternatives from the actual generated-field
classifier. Ordinary and EP witnesses retain promotion 0 and their respective
flags 0/1. Promotion retains its typed choice and flag 0. The castle alternative
contains the actual producer guard and full-Ply equality; the existing guard
consumer derives rook-target freshness from the checked path. Raw flag 2 alone
is insufficient. Equality transport preserves the actual source, destination,
promotion and flag used by `Chess.make_move`.

`consumer.prefilter_member` and `legal_member` derive post-board validity for
an actual occurrence. `prefilter_all` and `legal_all` cover every occurrence of
the actual producers. `legal_children` then carries the certificates through
actual `Chess.children`. Its affine array/list pair is unpacked once; only the
duplicable move list is reused. The child mapper preserves its supplied order
and duplicate multiplicity. No candidate is replaced, removed or deduplicated.

The sole semantic input premise is `Board.valid(parent)==True`. Arrays and raw
metadata are arbitrary. Callers supply no post-board validity, promotion bounds,
castle-target freshness, canonical turn, nonempty king plane, target geometry or
attack lookup contract. Member consumers also require their actual full-Ply
membership certificate. This discharges representation validity for the next
ply, rather than establishing all of that ply's remaining preconditions.

`Board.valid` describes exactly an empty square or one piece kind and one color.
It does not check king existence, king uniqueness, metadata bounds, move geometry,
check safety or initialized attack tables. This result is not full chess legality,
legal-move completeness or unconditional fast/full correctness. Arbitrary caller
tails are not certified by generation; arbitrary malformed moves remain outside
the theorem.

The concrete fixtures use small arbitrary leaf arrays, not `Tables.build`.
They invoke the actual-member and actual-list bridges for ordinary moves, EP in
both colors, all four promotions, and both castle sides in both colors. Repeated
actual promotion members produce the exact Queen/Knight/Queen child list, with
three occurrences. Complete EP boards distinguish removal of the off-destination
victim. Counterexamples retain malformed promotion tag 6, an unguarded castle
whose rook target is occupied, and an invalid parent square untouched by the move.

## Qualification

Use Bend 2.0.21+U64 commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2 and the
84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Each positive and negative check permits 86400 seconds, CPUs 1/3, two worker
threads, 6 GiB address space/RSS and 16 MiB per output file. Checks run serially
against detached immutable closures outside the worktree. All declarations remain
in the full positive fixture check. The qualifier binds every source/support file
to the frozen HEAD and reused dependencies to the exact base.

Sixteen controls are classified as eight contract-coupling checks and eight
concrete false witnesses. They cover the parent-validity premise, ordinary/EP/
promotion dispatch, guarded castle preservation, exact generated/legal occurrence,
child tail/list coupling, malformed inputs, child order/duplicates/kinds and EP
victim/flag fields. A coupling rejection is an interface check, not a semantic
counterexample. Credit requires one intended typed mismatch with different
expected and observed types; syntax, import, linearity, resource or incidental
failures receive no credit.

```sh
python -B -m native.bend_engine.standalone.proofs.generated_child_validity.qualify_generated_child_validity \
  /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```

