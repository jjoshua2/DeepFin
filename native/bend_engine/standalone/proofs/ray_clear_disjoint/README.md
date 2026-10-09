# Clearing a mask disjoint from the old ray attack

Algebra.disjoint(removed,old_attack) is the equality-valued Type certificate
U64.and(removed,old_attack) == U64.zero(). The theorem's explicit premise uses
the OLD imported Path.attack(xs,occ), never an asserted desired attack result.

New structural Boolean/Word/U64 partition and zero lemmas show that a disjoint
removal mask leaves each observed bit in its complement. disjoint_subset uses
actual subset certificates and AND association. Every head bit belongs to the
old attack; the tail belongs to it only when the old observation is clear.

Clear.agreement derives existing blocker-aware agreement by structural induction:
preserve the reached head observation, and recurse from tail disjointness only
in the old-clear continuation. An old blocker ends agreement. The proof never
infers agreement from attack equality. No range/uniqueness/geometry premise is
added to the arbitrary-list theorem.

Clear.attack proves Path.attack(xs, AND(occ,NOTremoved)) == Path.attack(xs,occ).
Clear.scan and actual_ray reuse checked Blocker framing, Fold and Traversal
with one shared arbitrary accumulator. Actual fuel is preserved and initial
stopped=False/mask=False. The existing input-only Spec.trace certificate binds
coordinate steps, listed squares and fuel; it supplies no occupancy or attacks.

actual_source specializes removed to U64.bit(U32.to_nat(src)) for src<64. Its
range certificate records that specialization; the general theorem is range-free.
clear_then_insert composes the checked PR1025 insertion theorem: after disjoint
clearing, OR insertion yields an actual ray attack subset of the original ray.

Checked positive: [1,2,3,4], old occ=12, removed=8 clears a square beyond the
first blocker; old/new actual attacks are both 6, while old interior agreement
is False. Removing the first blocker (4) produces actual attack 14 and is not
disjoint from old attack 6. Bounded source 3 and arbitrary-accumulator clearing
and clear-then-insert consumers are checked.

Duplicate edge: [1,1], occ=2 versus 0 yields equal independent attacks 2 but
both old and blocker-aware agreements False. A genuinely disjoint duplicate
clearing instance is separately checked. These duplicate witnesses concern the
arbitrary independent list theorem, not an invented production route.

This is one conditional production-ray ingredient. The full board/candidate/
move/table invariant that supplies old-observation disjointness remains open.
Unconditional bypass retention, full lookup framing and chess legality are not
derived. No raw Pro code was supplied or applied.

## Qualification

Exact base PR1025: 66788977a85acf4730bed9306a47e76b0c1e9503, tree
e22d35b0b28c63721a68d3d5235672165c573a0a. A shallow checkout needs that object.

From repository root, choose fresh external evidence paths:

    python3 -m native.bend_engine.standalone.proofs.ray_clear_disjoint.qualify_ray_clear_disjoint \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

Require Bend 2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, Bun1.4.2,
all84 files and fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
Exact-base check/evidence/mutation/strict-classifier helpers are reused and hashed.
All qualified support and Bend dependencies match base or clean HEAD Git blobs.

The caller selects available CPUs; checkers inherit at most two. This run uses
1,3 and excludes P2's CPUs30-31. Positive and each negative allowances are86400s,
with6GiB address-space/RSS checks and16MiB per-output caps. Prior evidence,
qualified source and compiler identity must remain unchanged.

Seven declaration-local semantic controls reject deletion of the first blocker,
its false disjointness certificate, fake duplicate agreement, a vacuous old
observation, a disconnected actual consumer, lost shared accumulator, and
deletion substituted for insertion in the composed actual-ray theorem. Parser,
import, resource, timeout or linearity failures cannot count as controls.

Independent internal review is required before stacked draft publication.
No merge, live adoption, P2, GPU, runtime edits or new installation.
