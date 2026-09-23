# Numeric fill intervals and real block frame laws

This opt-in six-law increment extends the 89-law relative-address baseline.
It does not change production code, earlier laws, the pinned compiler, or routine tests.

```sh
bun native/bend_engine/standalone/proofs/fill_interval/verify.js /path/to/pinned/bend --report /tmp/fill-proofs.json
CC=clang-18 bun native/bend_engine/standalone/proofs/fill_interval/verify_native.js /path/to/pinned/bend --report /tmp/fill-native.json
```

The aggregate retains the unchanged 89-law/176-control parent, then requires
six new laws and 19 controls: 95 laws and 195 controls total. `focused.js` alone
excludes the inherited gate; `--controls-only` also excludes the importing consumer
and cannot pass the aggregate. Native CC applies to that command only, not Python setup.

## Actual source contract

The two `bounded_clear_*` laws derive the existing `separation/Spec.clear`
predicate from complete depth-17 shape and **mathematical Nat count budgets**.
They do not assume any per-write path certificate. `Arithmetic.inc_value` relates
actual U32 increment to Nat successor while the address is bounded. Induction on
the count proves that every actual increment stays in its interval. The after
case admits the exclusive endpoint as a protected query; before requires strict
query < start. A zero-count fill is covered. Overflowing budgets are excluded,
not silently computed modulo U32. The before theorem requires end < 131072;
this includes all actual slider-block endpoints, not every conceivable allocation-end fill.

The two `bounded_fill_preserves_*` laws apply the existing real fill frame theorem
to those derived certificates. They preserve the queried U64 value, **not a newly
claimed whole-array contents equality**. The two `block_fill_preserves_*` laws
specialize to the actual mask, count, start, square, bishop flag and zero initial
subset used for each valid key. `Spec.block` calls original `Tables.fill`.
`Certified` imports the existing prefix proof producer, and `Block.count` turns
its widened endpoint theorem into an exact Nat budget using structural addition
and carry lemmas. No caller-supplied size/endpoint or clear-path certificate is needed.

The generic laws retain arbitrary slider parameters and arbitrary affine initial
arrays with complete shape. The block laws cover all 128 keys and arbitrary
outside queries. Shape is a real premise (earlier allocation/pipeline laws supply
it); capacity alone is not substituted. No desired read value occurs in a premise.

## Checks and limits

The consumer checks both sides of a positive-count fill and the full universal
block composition, retaining overlap and overflow counterexamples. Source success
requires zero status and exactly `All terms check.`. Eight semantic controls fail
at new arithmetic/budget/refinement locations; two actual fill mutations first
fail imported storage refinements and are reported as such. Nine policy/output
controls retain laws/proofs/imports and reject holes, foreign/symlinked inputs and
unsafe warnings. Missing files, wrong-layer failures, crashes and timeouts do not
count as the intended semantic result.

The native probe uses actual Tables.build headers, poisons a queried cell, runs
actual full/partial/zero/overlong fills, and observes that query plus first/last
block cells and capacity. No proof predicate runs natively and no host-supplied
expected table is given to the candidate. The independent oracle uses signed board
coordinates, BigInt scattering and ordinary integer prefixes. It does not compare
every returned cell or prove all stored metadata correct. All 128 keys have full
block cases; overlong and inside-block cases are deliberate outside-domain controls.

These laws establish whole-fill outside-location preservation. Stored header
correctness, final computed inside-block contents and lookup equality with an
independent blocker-ray specification remain P2 work. No native allocation/lifetime,
full engine/model/GPU/training or performance guarantee follows. Source laws trust
the pinned checker/Base; runtime and toolchain remain separate trust boundaries.
