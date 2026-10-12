# Occupancy insertion cannot expand a ray

Lattice.subset(small,big) is a Type-valued equality certificate:
U64.and(big,small) == small. It is not a Boolean used as a proof type.

The new structural Boolean/Word/U64 lemmas establish shared-OR subset lifting,
AND-over-OR distribution, and the actual comparison/nonzero behavior. Existing
decoder.Bits.nonzero_any and cmp_limbs connect real Word/U64 comparisons to
structural bit observations. test_or and insert_preserves apply to every Nat
index through actual U64.bit; no square-range assumption is required.

Insertion.attack_subset proves, for arbitrary Nat lists and U64 occ/added,
that Path.attack(xs, U64.or(occ,added)) is a subset of Path.attack(xs,occ).
Old blockers remain occupied; a new blocker can discard only a suffix; the
current head square remains included. Duplicates and out-of-range list entries
require no extra premise. scan_subset composes Fold.accumulator with shared-OR
lifting using one shared arbitrary accumulator.

actual_ray composes the result with existing Traversal.actual. Its existing
Spec.trace premise checks actual coordinate steps, fuel and listed squares;
it contains no occupancy or desired attack facts. Production fuel is preserved,
initial stopped=False and mask=False, with the same accumulator on both sides.
No board/move/table invariants, unconditional bypass retention, full slider
lookup result or chess legality are derived.

The consumer checks production and independent path [1,2,3]: old occupancy 0
attacks 14, adding bit 2 (4) attacks 6. Reversed inclusion fails, since
6 AND 14 is 6, not 14. With old occupancy 4 and added 4, OR keeps attack 6;
replacing insertion by AND NOT deletes the blocker and expands attack to 14.
Direct production values and failed-inclusion Boolean witnesses are checked.
There are also closed duplicate/out-of-range witnesses alongside the structural
arbitrary-list proof.

## Qualification

Exact base: PR1024 head 9620dd5d05a4376096c937d5aab6a53d6229d332, tree
299900ffcb56080e977364a5acef3e1a360f06c5. A shallow checkout must fetch this
exact base object before qualification:

    git fetch --depth=1 origin 9620dd5d05a4376096c937d5aab6a53d6229d332

From the repository root, use fresh external evidence paths:

    python3 -m native.bend_engine.standalone.proofs.occupancy_insertion.qualify_occupancy_insertion \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

Require pinned Bend 2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,
Bun 1.4.2, all 84 compiler files and fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.

The qualifier reuses exact-base fast_full_equivalence bounded checking,
mutation, evidence and strict semantic-classification helpers. All imported
Python support, all reused Bend dependencies, and the five new suite files are
hashed and verified against base/HEAD Git blobs. Preserve prior evidence.
The caller chooses available CPUs; checkers inherit at most two. This run uses
CPUs 1,3 and excludes P2's CPUs 30-31. Positive and each negative have an
86400-second allowance, 6 GiB address-space/RSS checks and 16 MiB per-output cap.

Seven declaration-local controls reject reversed inclusion, insertion changed
to deletion, a wrong reachable actual attack, a disconnected production
consumer, loss of the shared accumulator, an unjustified extra path bit and
denial of a preserved blocker. The reverse/deletion controls require typed
rejections at the named declarations; their concrete false equality is 6=14.
Parser, import, resource, timeout and linearity failures never count as semantic
controls.

Reconstructed from actual source and the user's goal/counterexamples. No raw Pro
code was supplied or executed. Independent internal review precedes publication.
Draft only; no merge, live adoption, GPU/runtime/P2 changes.
