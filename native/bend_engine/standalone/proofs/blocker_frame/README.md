# Blocker-aware ray framing

The new agreement predicate accepts Nil and singleton paths. At every nonfinal
square, both occupied bits must be equal: both True succeeds without evaluating
or constraining the suffix; both False checks the remainder; a mismatch fails.
The remainder is passed as a thunk and called only in the both-clear branch.

Blocker.attack_equal proves equality of the existing independent Path.attack,
including the first blocker. It applies to arbitrary Nat lists and U64
occupancies without range, uniqueness or chess-board assumptions.
interior_implies_agree proves the old Path.agree premise implies this predicate.

The consumer checks strict weakening: path [1,2,3,4], occupancies 4 and 12 give
new agreement True, old agreement False, and both independent attacks 6.
The actual Tables.ray with fuel 4, start square 0, direction 0, stopped False and
mask False also gives 6 for both inputs, for one shared arbitrary accumulator.
The zero-accumulator numeric instances are separately checked.

The existing Traversal.actual and Fold.accumulator bridges compose this framing
with actual production Tables.ray. The route certificate Spec.trace checks only
actual coordinate steps, fuel and listed squares. It supplies no occupancy or
expected attack equality. This increment does not derive board/move/table
invariants, bypass retention, lookup framing, or full chess legality.

The one-sided counterexample [1,2], a=0, b=2 has agreement False and attacks
6 versus 2. The same numeric difference is checked against actual Tables.ray.
No unsound Pro acceptance code was applied.

## Qualification

Exact base: PR1023 head eb8a3ac05c69b1e07bb8488d1d2cc4f49a7aa56b, tree
b3d82b3472aff98b86904a48ea68c8aa649b2726. A shallow checkout must fetch that
exact base object before qualification:

    git fetch --depth=1 origin eb8a3ac05c69b1e07bb8488d1d2cc4f49a7aa56b

From the repository root, use fresh report and evidence paths outside checkout:

    python3 -m native.bend_engine.standalone.proofs.blocker_frame.qualify_blocker_frame \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

Pinned Bend 2.0.21+U64 revision aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,
Bun 1.4.2, all 84 compiler files and fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4 are required.

The qualifier reuses the exact-base fast_full_equivalence helper functions for
bounded checking, evidence, isolated mutation and strict semantic rejection.
That Python source and all imported qualification support are included in its
hashes and base/HEAD Git-blob checks. Every reused Bend source matches the base;
a clean published head may change only this suite's four new files.

The caller selects an available CPU set; the qualifier inherits it and restricts
each checker to at most two CPUs. Avoid P2's assigned CPUs 30-31. Positive and
each negative allowance are 86400 seconds, with 6 GiB address-space/RSS checks
and 16 MiB per-output-file caps. Existing evidence and qualified sources must
remain unchanged.

Seven semantic controls test acceptance of each one-sided blocker mismatch,
skipping the both-clear suffix, a false closed agreement, a wrong actual attack
with a reachable bit difference, a disconnected production consumer, and loss
of the arbitrary accumulator. Acceptance requires exactly one nonempty unequal
expected/observed pair on single lines and its expected proof location.
Resource, parser, import, timeout and linearity failures cannot count.

Independent internal review is required before publishing a draft stacked PR.
No merge, live adoption, GPU work or changes to P2 are part of this increment.
