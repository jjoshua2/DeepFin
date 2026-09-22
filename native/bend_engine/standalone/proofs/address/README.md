# Exact U32 slider indices

This additive suite connects the actual `Sliders.pext_index` to full-width PEXT
and the previously proved Nat-indexed subset enumeration. It changes no runtime
code, previous accepted law or compiler input. It is not an affine-table theorem.

## Accepted contracts

| Law | Function and domain |
| --- | --- |
| `low_projection_exact` | Actual `U64.low`; arbitrary U64 `x`, mathematical Nat `k <= 32`, and `value(x) < 2^k`. |
| `compact_index_exact` | Actual `Sliders.pext_index`; arbitrary occupancy and mask with population at most 32. |
| `chess_lookup_exact` | Actual lookup equals the full PEXT value for every valid key and arbitrary U64 occupancy. |
| `chess_lookup_word` | Actual `U64.from_u32` of that index equals the full PEXT word. |
| `chess_sequence_ordinal` | Actual lookup on `Sequence.at(i,mask)` equals `i` below mathematical capacity. |
| `chess_sequence_coverage` | The actual lookup value indexes the recurrence back to every mask-contained state. |
| `chess_lookup_injective` | Equal actual lookup indices imply equal mask-contained states. |
| `chess_lookup_redeposit` | PDEP of the widened actual lookup recovers the occupancy masked to the relevant bits. |

The chess domain is exactly `Nat.is_lt(U32.to_nat(key),128n) == True{}`.
`Layout.mask` calls the existing `Tables.slider`; neither its implementation nor
`Sliders.pext_index` is copied into a shadow candidate. `Sequence.at` is the
existing imported recurrence that calls the actual `Subsets.next`.

`Projection.bend` proves retained-width value preservation by structural Word
induction. Its symbolic `k` includes 32 without expanding a literal `2^32` into
a huge concrete Nat during proof elaboration. `Correspondence.bend` composes
this with the existing extraction bound, actual count certificate and ordinal.
The accepted laws do not ask the caller to assume a population equality:
`PROOF.bend` explicitly imports the old layout proof and obtains that certificate
from `LayoutLaws.mask_population`. The helper separation avoids repeatedly
checking the expensive finite geometry proof while developing structural lemmas.

## Run from the repository root

Use the unchanged pinned compiler at
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` and Bun 1.4.2:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/address/verify.js \
  /path/to/pinned-bend --report /tmp/exact-index-proofs.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/address/verify_native.js \
  /path/to/pinned-bend --report /tmp/exact-index-native.json
```

The source gate verifies the 84-input compiler identity, invokes the unchanged
48-law/55-control layout gate once, then checks the new importing consumer and
all eight new laws. It requires successful execution and exactly
`All terms check.`; a raw zero with unsafe/foreign warnings is rejected.
The new suite has 17 negative controls: eight semantic helper-proof mutations,
eight import/manifest policies, and one unsafe dependency checked both ways.
The in-range constant-zero mutation demonstrates why a range theorem alone
cannot establish an exact index. Timeouts, crashes and unrelated diagnostics
are not accepted semantic rejections. All mutations use disposable copies.

These gates are opt-in, not part of ordinary pytest or a new permanent workflow.
They use one compiler at a time. The source gate has bounded subprocess timeouts;
the inherited finite-key proof can dominate wall time. Native fixtures cap rows
and recurrence indices at 4,096 and do not increase any engine perft depth.

## Native boundary evidence

The probe consumes runtime operands and uses actual lookup/PEXT/PDEP and recurrence
functions. Keys 0..127 select the actual chess masks; key 128 selects an explicit
raw mask for generic width boundaries. Other keys and excessive budgets fail.
The independent reference walks signed file/rank coordinates, gathers/scatters
BigInt bits directly, and computes the ordinal modulo mathematical capacity; it
does not use carry-rippler subtraction to calculate expected recurrence states.

Each generic, forced-portable, native-target and UBSan C executable is compared
on the same 2,070 fixture rows. The suite includes population 32 and explicit
wider-mask truncation counterexamples, not a false claim that arbitrary U64
PEXT values fit U32. Wider-mask rows test the specified low-word behavior outside
the exactness theorem. Repeated native environments are not disjoint datasets.

## Limits and trust

No prefix sum, offset-addition overflow, region separation, Array initialization,
affine ownership, actual `Chess.slide` read, or blocker-ray refinement is proved
here. Those are the next obligations. Source laws remain conditional on the
unchanged checker/Base semantics; native lowering, effects/ABI, runtime, C toolchain
and hardware remain independently tested/trusted. This suite does not exercise
trained models, CUDA, selfplay/training, performance or playing strength.

See `docs/experiments/2026-09-21-bend-exact-slider-indices.md` for the exact
qualification outcomes, development failures and publication status.
