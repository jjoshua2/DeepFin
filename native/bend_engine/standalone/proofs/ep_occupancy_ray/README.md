# Actual EP occupancy and conditional victim-aware ray subset

`EP.normalization` proves the actual `Chess.make_move(b,Ply{s,d,0,1})`
occupancy is `OR(AND_NOT(occupied(b), source_bit OR bit(d xor8)), target_bit)`.
The source is the actual flag-one dispatch and complementary color updates;
destination clearing is absorbed by reinsertion. Board rows, raw turn, and U32
source/destination are arbitrary, including overlaps, absent source, equal
squares and out-of-range actual bit behavior. No parent premise is needed for
this total identity. The ordinary-move formula would wrongly retain the victim.

`EP.actual_ray` rewrites that identity into the checked clear/insert theorem.
Its input-only `Route.trace` records actual coordinate steps and fuel. Its
separate explicit premise is that BOTH source and captured-pawn bits are disjoint
from the OLD first-blocker-inclusive `Path.attack(xs,occupied(b))`. The result
is a subset of the old actual `Tables.ray`, with the same fuel, both actual
False flags and arbitrary shared accumulator. This premise is not inferred
from parent board validity or legal-list membership.

The public `legal_member` calls the pinned EP producer consumer once using one
exact arbitrary affine input array and one full-Ply occurrence certificate.
It derives promotion0/flag1 shape while preserving source/destination identity.
Its result combines the new occupancy/ray result with the reused PR1048 king-mask
and original-parent-side king-plane frame under explicit parent `Board.Spec.valid`
partition and actual `Position.valid_ep(get_ep(b),b)` certificates. Those parents
support the reused king result; they are unnecessary for total occupancy alone.
No canonical turn, unique/nonempty king plane, replacement table or target
geometry premise is introduced. Occurrence membership is not uniqueness.

Both pawn directions instantiate the new public result with actual legal-list
fixtures and rays that shrink at the new target. A separate parent satisfies
partition, coherent EP and actual `Position.valid`, yet capturing its first
ray blocker expands the king32-to-rook39 ray from three squares to seven.
Source36 is outside the old observation; victim35 is observed. The specified
arbitrary table's legal occurrence is checked separately. This preserves the
boundary between actual list membership and geometric attack correctness.

Stale-EP and pawn/king-overlap king-loss witnesses, partition/coherent-EP but
multiple-kings `Position.valid` rejection, arbitrary-array child-EP failure,
duplicate count3, wrong full-Ply field absence and the exact table are retained.
The two concrete duplicate-tail certificates are inherited from the exact
sealed PR1047 ancestry; this entry checks the inherited count/table witnesses.

This theorem bounds actual `Tables.ray`. It does not establish arbitrary-array
`Chess.attack`/`in_check` lookup correctness, opponent slider-plane framing,
king safety, accepted-root provenance/preservation, final suffix array equality
or full legal-move correctness. No production/runtime source changes.

## Reproduce

Use Bend 2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, Bun 1.4.2 and
the verified 84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Run from the repository root with fresh external evidence paths:

```sh
export BUN="$HOME/.bun/bin/bun"
export TMPDIR="$EVIDENCE_ROOT/control-tmp"
mkdir -p "$TMPDIR"
taskset -c 1,3 python3 -m \
  native.bend_engine.standalone.proofs.ep_occupancy_ray.qualify_ep_occupancy_ray \
  "$BEND_CHECKER" --checker-manifest "$CHECKER_MANIFEST" \
  --report "$EVIDENCE_ROOT/qualification.json" \
  --evidence-dir "$EVIDENCE_ROOT/qualification-logs"
```

The gate freezes/hash-checks the source closure and support helpers, pins every
reused dependency to exact PR1048 and binds clean published source to HEAD.
Each serial positive/negative defaults to 86400 seconds, CPUs1/3, 6GiB AS/RSS
and 16MiB per output file. Fourteen coupling controls and eleven concrete false
witnesses must reject at the expected typed obligation. Parser, inference,
affine, resource and timeout failures do not qualify. Coupling controls test
certificate use rather than logical necessity of every premise.
