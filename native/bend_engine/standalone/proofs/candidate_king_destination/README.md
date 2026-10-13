# Actual candidate destinations exclude all king bits

This increment discharges PR1032's explicit destination-king-exclusion premise.
It follows production `Chess.piece_targets`, `Chess.destinations`, `scan_after`,
`scan_step` and `scan`, then composes the actual ordinary king-frame consumer.

`Occupied.bend` lifts `Board.valid` through its finite row invariant to prove
that every king bit is occupied. `Algebra.bend` proves Boolean, Word and U64
mask laws, including an all-index bridge from actual `test_bit == False` to
mask disjointness. This covers out-of-range zero bits without a new destination
geometry premise. `Targets.bend` follows the production guards: non-pawn
targets exclude own pieces OR all kings; pawn capturables remove kings; pawn
advances and EP targets require an empty occupied square. The production
range, rank and EP victim checks remain part of the actual expression.

`Bits.bend` composes the existing actual nonzero-ctz range and selected-bit
lemmas. `Emission.bend` proves that every emitted move's destination bit is
disjoint from all king bits. Its structural destination lemma accepts arbitrary
fuel with an empty flag agreeing with the actual mask. Both `put_move` branches
retain the caller-tail predicate: the four promotion entries share their
destination, and ordinary entries retain the actual computed EP flag. Scan
steps and arbitrary source-key lists preserve that predicate and do not assume
source range or ownership. The final theorem uses the exact scan-based
`C.candidates` list; it does not include the separately appended castle suffix.

`Provenance.bend` carries the existing source range/ownership predicate and
the new destination predicate through one actual scan with one linear input
table. It reuses the checked source destination lemma and the all-emitted
bit_squares range/set results, then extracts both properties from one membership
occurrence. The ordinary consumer passes the source facts to PR1032's checked
source helper and composes its king frame. It requires full-Ply membership for
`Ply{src,dst,0,0}`, `Board.valid`, and the actual bypass-False branch; callers no
longer supply destination exclusion. Arbitrary input tables, arbitrary U32
turns, empty or multiple-bit king planes remain allowed. The query consumer
retains a nonempty moving-color king-plane premise and separate OLD and NEW
whole-pair rook/bishop lookup contracts at the preserved square and actual old
and post-update occupancy. The original side selects the post-update check.
All pair continuations retain the actual returned table; no table is replaced.

Concrete checks include a valid actual pawn candidate, the resulting king
frame, exclusion of an enemy king from pawn captures, and rejection of an
occupied EP target. An invalid board with an uncolored king at a pawn advance
destination supplies actual membership plus a set destination king bit;
`Board.valid` cannot simply be dropped. Whole-pair scan witnesses preserve a
multi-bit target mask, an arbitrary duplicate tail, promotion values 1-4 with
flag 0 even when the destination equals EP, and the ordinary EP flag 1.

This establishes destination provenance and king preservation. It does not
prove post-move `in_check == False`, full attack safety, table-builder validity,
king uniqueness or full legal-move correctness. Lookup contracts remain explicit.

## Reproduce the checked gate

Use the existing Bend 2.0.21+U64 checker at commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` with Bun 1.4.2 and the verified
84-file manifest, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Keep fresh evidence output outside the source checkout.

```sh
export BUN="$HOME/.bun/bin/bun"
export TMPDIR="$EVIDENCE_ROOT/control-tmp"
mkdir -p "$TMPDIR"
taskset -c 1,3 python3 -m \
  native.bend_engine.standalone.proofs.candidate_king_destination.qualify_candidate_king_destination \
  "$BEND_CHECKER" --checker-manifest "$CHECKER_MANIFEST" \
  --report "$EVIDENCE_ROOT/qualification.json" \
  --evidence-dir "$EVIDENCE_ROOT/qualification-logs"
```

The gate freezes and hashes its dependency closure, support code and suite files,
pins reused dependencies to exact PR1032 head/tree and binds published sources
to Git HEAD. Positive and each serial negative default to 86,400 seconds, two
allowed CPU cores, 6 GiB address space/RSS and 16 MiB per output file. Parser,
import, linearity, timeout and resource failures are not accepted as controls.

The ten contract-coupling controls check candidate and board certificates,
actual bypass, the derived consumer, actual empty flag, caller tail, capture
king mask, exact table and old/new rook contracts. They establish dependency
coupling rather than logical necessity of every premise. Nine concrete false
witness controls check destination/frame, the invalid-board counterexample,
EP and pawn exclusion, duplicate tail, promotion and EP fields.

The sealed qualification, raw commands/logs/hashes and independent internal
review remain external artifacts, identified in the stacked draft PR. No
generated evidence is tracked in the proof suite.
