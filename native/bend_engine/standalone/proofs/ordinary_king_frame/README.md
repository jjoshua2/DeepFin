# Actual ordinary king preservation and separate old/new queries

This increment connects actual `Chess.make_move(b, Ply{src,dst,0,0})` to
the king square used by actual `Chess.in_check`. It builds on PR1031's
prepared king selection and PR1030's geometric path and whole-pair lookup results.

`Algebra.bend` proves finite Boolean laws, structural Word laws and their U64
composition. `Frame.bend` unfolds the actual ordinary update, proves preservation
of the entire king mask, then of the original-side king plane and its `ctz`.
`Source.bend` derives source range/ownership from actual full-Ply candidate
membership; `Board.valid` and the actual bypass branch give source king clearance
and actual decoder nonking status. `Enemy.bend` proves exact removal of source and
destination bits from the enemy color mask and each of its six piece-kind masks,
which implies subset. `consumer.bend` composes these with the production functions.

The king-plane consumer assumes actual ordinary candidate membership,
`Board.valid`, `filter_requires(sensitive(b,rays),ply) == False`, and explicit
destination disjointness from **all** king bits. The destination condition is
not yet derived from candidate membership. No additional destination range or
geometry assumption is introduced. Plane preservation itself allows arbitrary
U32 turns and an empty or multiple-bit king plane. The query consumer additionally
requires a nonempty moving-color king plane to obtain PR1031's in-range index.

The original `get_turn(b)` is passed to the post-update check: `make_move` flips
the board's metadata, so selecting the new board's current turn selects the other
side. The equality preserves the entire `(table, Bool)` returned by `in_check`.
The old and new queen queries use **separate** whole-pair rook and bishop lookup
contracts for the old and actual post-update occupancy, respectively, at the
preserved king square. Each returns the exact input `O.pack(c)` table. These
lookup contracts remain explicit; no arbitrary table is presumed valid.

Enemy erasure has a separate canonical-side certificate
`get_turn(b) == Bool.to_u32(white)` (turn 0 or 1). It does not require `Board.valid`.
For noncanonical turns, `color(turn)` and `color(turn xor 1)` can both select black,
so calling the latter enemy does not imply erasure. The checked turn-2 witness
shows a new pawn target bit absent from the old mask; the helper named
`noncanonical_pawn_intersection_empty` asserts only the intersection fact.

The concrete actual-candidate witness establishes the old-side frame at square 8.
Additional total-update witnesses show the flipped metadata, enemy rook capture,
a king destination clearing the plane, the invalid-board decoder fallback adding
a king, and the noncanonical side counterexample. These raw update witnesses are
representation facts; they do not assert legal generation of those moves.

The scope stops at king preservation, enemy-mask erasure and exact query rewrites.
It does not prove post-move `in_check == False`, full attack safety, destination
provenance, table-builder validity, unique kings, full filter equivalence or full
legal-move correctness. Promotion, EP and castle moves are outside this ordinary
update theorem.

## Reproduce the checked gate

Use the existing pinned Bend 2.0.21+U64 checkout at commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` and Bun 1.4.2.
The verifier and supplied 84-file manifest must match fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Supply fresh external output paths; do not place evidence in this checkout.

```sh
export BUN="$HOME/.bun/bin/bun"
export TMPDIR="$EVIDENCE_ROOT/control-tmp"
mkdir -p "$TMPDIR"
taskset -c 1,3 python3 -m \
  native.bend_engine.standalone.proofs.ordinary_king_frame.qualify_ordinary_king_frame \
  "$BEND_CHECKER" --checker-manifest "$CHECKER_MANIFEST" \
  --report "$EVIDENCE_ROOT/qualification.json" \
  --evidence-dir "$EVIDENCE_ROOT/qualification-logs"
```

The gate hashes its dependency closure, support code and all suite files, pins
reused dependencies to the exact PR1031 base, binds published sources to Git HEAD,
and freezes a source snapshot. Positive and each serial negative check default to
86,400 seconds, two allowed CPU cores, 6 GiB address space/RSS and a 16 MiB cap for
each output file. Parser, import, linearity, timeout and resource failures do not
count as rejection at an expected obligation. Every negative must give one
expected/observed type mismatch at the named declaration.

There are 14 contract-coupling controls (actual candidate, board consistency,
destination exclusion, bypass, ordinary flag/promotion, canonical side, four
lookup contracts, returned table, king presence and new occupancy), and seven
concrete false-witness controls. Coupling controls test the composed certificate
dependency; they do not by themselves prove logical necessity of every premise.
The false witnesses test concrete source-linked representation assertions.

The sealed qualification and internal independent review are external artifacts;
the draft PR records the exact head, result hashes and durable evidence location.
No evidence snapshot or mutable output belongs in the tracked source suite.
