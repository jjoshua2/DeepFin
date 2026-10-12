# Actual EP king-mask and original-side plane preservation

For actual full-Ply membership in `Chess.legal_moves(table,b)` and flag1, the
public consumer derives promotion0, source pawn bit, destination
equal to the parent EP field, and destination below64 from the checked producer
consumer. It derives source range from the actual pawn test bit. It then combines the explicit parent `Board.Spec.valid` partition
certificate and actual `Position.valid_ep(get_ep(b),b)` certificate with the
actual priority decoder and EP raw-update bridge. The source, destination and
`destination xor8` victim are disjoint from every king bit. The actual child
retains the entire king mask and original-side king plane; its CTZ is unchanged.

One affine input array and one full-Ply membership certificate flow through
the actual producer consumer. Source range follows from its true pawn test bit;
out-of-range actual U64.bit is zero. No array or membership duplication is used. No replacement table or geometry premise is introduced. This also
applies to initialized arrays, without asserting geometry after writes. Full-Ply
identity is retained, and the inherited multibit scan witnesses retain both
certified duplicate tail nodes and the input table. This occurrence consumer
does not assert uniqueness or discard duplicates.

The concrete two-tail certificates were checked in the exact PR1047 base.
This gate checks the inherited count/table witnesses and imported generic
consumer; it does not rerun that base fixture entry. The evidence audit rehashes
the preserved PR1047 qualification receipt and its complete sealed evidence.

`Rows` checks finite local partition/decoder facts. `Parent` extracts the actual
coherent-EP target-empty and victim-decoder fields via the existing bounded
validator unfolding, and derives the victim bound in64destinationcases. `Frame`
checks the exact source|victim|destination mask and raw update, then the king and
color-plane frame. `consumer` derives all move fields/ranges internally and
queries the original parent turn after metadata flips. Canonical turn, unique
kings, nonempty planes, and target geometry are unnecessary for this theorem.

Both pawn directions use actual legal-list fixtures. The stale-EP actual legal
occurrence still loses the opposing king; the overlapping pawn/king update loses
its source king and violates partition. A partition-valid/coherent-EP board with
multiple kings fails `Position.valid`, preserving the distinction from accepted
roots. The arbitrary-array32-to48child-EP failure remains a direct update witness.
These are retained counterexamples, not premises silently repaired by the proof.

This suite proves a conditional EP frame. It does not prove post-move attack
safety, accepted-root provenance/preservation, final legal suffix array equality,
or full legal-move correctness. It changes no production source.

## Reproduce

Use pinned Bend2.0.21+U64 commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2 and the verified84file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Use fresh external evidence paths:

```sh
export BUN="$HOME/.bun/bin/bun"
export TMPDIR="$EVIDENCE_ROOT/control-tmp"
mkdir -p "$TMPDIR"
taskset -c 1,3 python3 -m \
  native.bend_engine.standalone.proofs.ep_king_frame.qualify_ep_king_frame \
  "$BEND_CHECKER" --checker-manifest "$CHECKER_MANIFEST" \
  --report "$EVIDENCE_ROOT/qualification.json" \
  --evidence-dir "$EVIDENCE_ROOT/qualification-logs"
```

The fail-closed gate freezes/hash-checks its complete closure and support code,
pins reused dependencies to exact PR1047, and binds published source to HEAD.
Each serial positive/negative check defaults to86400seconds, CPUs1/3,6GiB
address/RSS ceiling,16MiB per output file. Eleven contract-coupling controls and
ten concrete false witnesses must reject at one expected typed mismatch;
parser/inference/affine/resource/timeout failures do not qualify. Coupling mutations test
certificate use, not logical necessity of every premise.
