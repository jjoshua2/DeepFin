# Complete-tree route separation

Opt-in source continuation of #827, pinned to the unchanged Bend U64 compiler.

```bash
bun native/bend_engine/standalone/proofs/complete/verify.js /path/to/pinned/bend --report /tmp/complete-proofs.json
bun native/bend_engine/standalone/proofs/complete/verify_native.js /path/to/pinned/bend --report /tmp/complete-native.json
```

The aggregate retains the unchanged 79-law/140-control normalization gate and
adds four public laws and 16 controls. `focused.js` is bounded development only;
it explicitly reports `inherited_gate_run:false` rather than a complete aggregate.

## Contracts and representation

`Spec.full(d)` is the complete binary constructor shape at depth d. Actual
`Array.new` produces it; actual table and extras loops preserve it for arbitrary
source Nat counts. These shape contracts do not assert native allocation or
termination feasibility at unbounded depths/counts.

For arbitrary real affine arrays of that shape at depth 17, any two distinct
U32 indices below 131072 satisfy the existing `Spec.separate` predicate. No
path-separation premise or desired read equality is supplied by the caller.
The new actual write/read theorem consequently preserves the other query value
and the entire updated array in the returned pair. The importing consumer derives
shape from the actual initialized table/extras pipeline before applying separation.

`Arithmetic` proves a right-half subtraction stays in its child interval and
fixed-subtrahend ripple subtraction is injective. `Core.routes` recurses on depths
0..17; it compares actual unsigned U32 halves and rebases right indices using
actual U32 subtraction. Only 17 depth constants and a Boolean full-adder truth
table are split; addresses are not enumerated. `Actual` composes this with the
existing normalization and certified affine representation/read-write lemmas.
Keeping depth symbolic until scalar specialization avoids expanding the full
131072-leaf proof term. No compiler edits, holes, foreign witnesses or new axioms.

Capacity alone remains insufficient: a ragged shape with a two-leaf left subtree
and a single right leaf reports size four but aliases indices two and three.
The consumer preserves that counterexample and the excluded endpoint alias.
Small concrete positive examples exercise cross-branch and same-right-half recursion.

## Validation and limits

Eight semantic controls mutate shape, routes, bounds, equality/shape premises,
actual allocation and actual table-loop storage. Eight controls enforce mandatory
laws/proofs/imports, regular files, no holes/foreign/unsafe dependencies and exact
checker output. Crashes, timeouts and missing files are not semantic rejection.
Status zero alone does not pass: output must be exactly `All terms check.`.

Native tests reuse the unchanged supported public-API normalization probe. All
17 possible divergence bits are exercised. The reference predicts the query's
seed/write value using unsigned remainder; proof predicates and models do not run
in the candidate. Same-index and out-of-range cases retain aliasing counterexamples.
The native checks observe query values, capacity and normalization, not every cell,
and do not execute internal raw Array helpers or requalify the full table builder.

This closes route separation for bounded indices in a complete source tree.
Actual prefix-plus-lookup bounds, fill-clear certificates, all initialized final
contents and independent blocker-ray lookup refinement remain separate P2 work.
Source equality does not prove physical pointer identity, native allocation or
lifetime, compiler lowering, or hardware correctness. No application logic newly
moves out of Python, and no model/GPU/perft/training/benchmark work is added.
