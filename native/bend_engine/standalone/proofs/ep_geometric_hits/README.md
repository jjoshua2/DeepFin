# Actual EP colored nonpawn erasure and conditional geometric hits

`Plane.erasure` and `Plane.colored_erasure` connect directly to the actual
`Chess.make_move(b,Ply{s,d,0,1})` through `en_passant/Actual.unfold`. For a
source decoded as pawn0 and an explicit nonpawn piece tag, each child piece
plane and each fixed-query-side colored plane is the parent plane AND_NOT the
full source/victim (`dst xor8`)/destination mask. The actual color insertion
at the destination is masked away by the nonpawn destination clear. The color
bridge threads the actual next turn, rights and EP fields. Arbitrary board
rows, raw turn, source/destination U32s and fixed query side are allowed. No
partition, coherent EP, canonical turn, range or king premise is needed for
these low-level identities. They are erasure/subset results, not preservation
equalities: malformed overlapping nonpawn rows can lose bits.

`Hits.subset` composes those colored planes with the unchanged PR1049 actual
single `Tables.ray` subset. Its separate input-only `Route.trace` fixes actual
coordinate steps and fuel. BOTH source and victim must be outside the OLD
first-blocker-inclusive `Path.attack(xs,occupied(parent))`. `EP.removed` is
the source/victim mask for this condition; `Plane.mask` includes destination
for piece erasure. The result bounds the actual nested U64 gate
`AND(AND(ray,OR(rook-or-bishop,queens)),color(board,by))`, with the same fuel,
square, direction, False/False flags, selected R/B class, fixed query side and
arbitrary shared accumulator. `Hits.clear` also transfers an explicit zero
parent gate to a zero child gate. The R/B class is a gate selector; this proof
does not impose or infer a direction/class match or aggregate eight rays.

The public `legal_member` and `clear_member` each consume one exact arbitrary
affine input table and one full-Ply legal occurrence certificate through
`Tag.legal_member` once. They derive promotion0/flag1 and the actual pawn
priority decoder from that producer. Each result retains the PR1049 actual
occupancy, king-mask/original-parent-side-plane and conditional ray result
under explicit parent `Board.valid` and actual
`Position.valid_ep(get_ep(parent),parent)` certificates. Those certificates
support the reused king frame; the new colored erasure needs only decoder0
and a nonpawn tag. The old ray condition remains separate from legal-list
membership. Membership preserves full fields and multiplicity; it is not
uniqueness. No replacement table or target-geometry certificate is added.

`Fixtures.bend` checks both actual EP pawn directions on boards with nonzero
opponent rook, bishop and queen rows. Orthogonal and diagonal public instances
shrink nonzero old gates to zero at the new destination blocker. A public
zero-transfer instance keeps those nonzero rows outside its ray. `Witnesses`
checks old/new gate values, queen inclusion, arbitrary shared accumulator,
raw-turn7 and overlapping-row erasure, full destination/victim-mask necessity,
the excluded pawn/source-kind/nonzero-promotion/query-side generalizations,
and exact table/duplicate count3/wrong promotion/EP flag distinctions. It also
checks the retained validated-parent actual legal occurrence whose captured
first blocker exposes a king32-to-rook39 hit. That gate grows from zero to
bit39 when the old-victim condition fails, even though the colored rook plane
does not grow. Parent validity and legal membership cannot discharge it.

No theorem here establishes `Tables.slider` aggregation, arbitrary-array
`Chess.attack`/`in_check` geometry, king safety, accepted-root preservation,
suffix-array equality or full legal-move correctness. The exact arbitrary
table fixture is a producer occurrence, not a geometric table qualification.

Run the fail-closed qualifier from the repository root with an already pinned
checker and its 84-file qualification manifest:

```text
python -m native.bend_engine.standalone.proofs.ep_geometric_hits.qualify_ep_geometric_hits CHECKER --checker-manifest MANIFEST --report REPORT --evidence-dir FRESH_EVIDENCE
```

It freezes and hashes the union of the consumer/fixture and witness closures,
checks the full Witnesses closure once, and runs29 one-declaration negative controls
(17 typed contract couplings and 12 concrete false witnesses). Only typed
rejection at each declared obligation counts. Parser/import/affine/inference,
resource/output/timeout failures do not qualify. It pins PR1049 base
`3bd05d0d2678b19bf099d9839507c4a39fe7f693`, tree
`14e225046aa5c13586e9c962f07bcebcbb0aaf06`, Bend2.0.21+U64 commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2 and the verified84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Each positive/negative gets86400 seconds, CPUs1,3, 6GiB AS/RSS and16MiB
per stdout/stderr stream. Reports, logs, snapshots, commands and hash evidence
belong outside the code checkout; no prior artifact is overwritten.
