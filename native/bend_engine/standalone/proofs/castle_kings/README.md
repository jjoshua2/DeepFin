# Castling mover-king invariants

Four source contracts derive the singleton selected king through actual king transit
and castling, preserve a Boolean side transition, and connect actual in_check to the
intended stage square and opposite attacker side. Both components of the check/query
pair are included; arbitrary affine tables remain inputs.

Initial conditions are representation consistency, exactly one selected-side king at
its home square, and raw turn equal to that Boolean side. Final castling additionally
requires the actual producer guard. Transit alone does not require the guard. These
are explicit input conditions, not assumed output singleton or decoder results.

The guard supplies initial rook-landing freshness. Consistency then rules out even
hidden/unowned king bits there, preventing recoloring from adding a second king.
The consumer includes a consistent blocked enemy-king counterexample with false guard.
Invalid raw turn3 becomes2 under XOR; valid-side flipping does not sanitize metadata.

Run from a checked-out exact source tree:

```sh
bun native/bend_engine/standalone/proofs/castle_kings/focused.js /path/to/pinned/bend --report focused.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/castle_kings/verify_native.py /path/to/pinned/bend --report native.json
```

The source gate checks all producers with the unmodified pinned CLI and runs ten
semantic/refinement rejections, eight policy checks and one synthetic warning check.
Four implementation mutations may fail accepted imported implementation bridges;
these are deliberately retained as integration checks, not independent new laws.

Native tests build actual tables and compare complete start/transit/final Boards,
selected king planes/index, check and attack values against an external coordinate/set
reference. Missing kings use the probe-only sentinel2 and skip the invalid in_check
call. Modes repeat fixtures, not exhaustive game coverage. Native comparisons do not
prove every table cell or lifetime. The formal stage result routes to actual attacked;
independent initialized semantics are provided by the preceding PR, not reproved here.

No universal forward/reverse attack correspondence, opposing-side singleton, historical
castling rights, legal reachability or full-generator semantic safety follows from
these four contracts alone. No production function or prior accepted proof changes.
