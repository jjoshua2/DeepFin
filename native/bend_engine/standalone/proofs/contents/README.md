# Computed entries inside actual bounded fills

Three contracts connect the actual `Tables.fill` result to the value written at
any selected interior index. They complement the inherited fill-interval laws,
which preserve locations outside a fill. No runtime code or prior law changes.

## Statements and limits

`bounded_fill_entry` applies to an arbitrary affine U64 array of complete depth-17
shape, arbitrary slider parameters and starting subset, and every mathematical
`i < n`, with `n + to_nat(at) <= to_nat(end)` and `end < 131072`. Reading actual
`U32.add(at,U32.from_nat(i))` after the entire fill returns the actual slider
computation for iteration i. `Spec.state` only traverses the existing
`Subsets.next`; `Schedule` proves the traversal/address correspondence.

`bounded_zero_fill_read` starts at zero and uses the **previously proved
`Sequence.at`**, not another claimed enumerator. It establishes the complete
returned pair: the actual final updated array and the selected computed value.

`full_block_entry` specializes to every valid chess key and every relative index
below its actual block size. It obtains the mathematical count budget and safe
exclusive endpoint from the existing prefix proof producer. The caller supplies
only the valid key, selected-index bound, and complete initial shape, not an
expected value, a per-write noninterference premise, or an assumed size equality.
The actual blocks' endpoints are below 131072; the general fill law deliberately
retains this inherited strict exclusive-endpoint requirement.

The result refers to actual `Tables.slider`, **not yet an independently proved
ray specification**. `Block.block` is the real fill at the certified scalar
prefix; it does not establish stored-header correctness or persistence through
later metadata, other-block and extras writes in the complete builder. Those
compositions remain necessary before claiming final `Chess.bend` lookup refinement.

## Proof structure

`First` uses same-location write/read plus the inherited bounded suffix frame to
show the first computed value survives. `Entries` inducts over the selected offset,
retaining actual updated storage and the remaining budget, then handles arbitrary
`i < n` through an exact Nat count split. `Schedule` proves modular U32 address
addition and correspondence to the accepted zero-start subset recurrence.
`Actual` transports the values and full returned pairs through the existing
certified affine representation. `Certified` supplies actual block-size/prefix
facts from accepted producers. No new axiom, unsafe dependency, foreign equality
witness or proof hole is used. The engine never imports the proof representation.

## Commands and cost

From the repository checkout with the exact pinned compiler available:

```bash
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/contents/focused.js /path/to/bend --report /tmp/contents-focused.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/contents/verify.js /path/to/bend --report /tmp/contents-aggregate.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/contents/verify_native.js /path/to/bend --report /tmp/contents-native.json
```

The full wrapper requires the unchanged **95-law / 195-control** parent gate and
then this three-law / 17-control focused gate. Its prospective total is 98 laws
and 212 controls; the dated execution record states what was actually run.
`--controls-only` is a development option on `focused.js` only. It returns a
`controls_gate`, explicitly labels the consumer not run, and never claims a full
source or inherited aggregate pass. Native and source gates remain separate.
All new tests are bounded and opt-in, not added to routine CI/perft budgets.

Seven controls require failures in the new contents/schedule proofs. Two actual
fill mutations deliberately fail the inherited implementation-linked Build proof;
these are labeled as inherited-dependency rejections, not new-law failures. Eight
more retain required manifests/imports and reject holes, foreign/symlinked input,
and unsafe output even when the raw checker exits zero. Missing files, crashes,
timeouts and affine-use errors do not count as valid semantic rejection.

The native probe builds and retains actual table metadata, then runs actual fills
on a fresh, poisoned complete allocation for each case. The independent reference
uses signed-coordinate rays, compact bit deposition and mathematical prefix sums.
All observed seed values differ from computed values. It checks selected entries,
full mask/prefix values and scalar metadata, not every returned array cell or the
native execution of the proof model. The extra native write-zero mutation is
recorded separately and is not added to gate-control counts.

Source equality is not native pointer identity, allocation feasibility or physical
lifetime safety. Compiler/checker, native lowering, runtime, ABI, toolchain and
hardware remain separate trust boundaries. No Python application component, model,
GPU, training or performance result is added by this proof increment.
