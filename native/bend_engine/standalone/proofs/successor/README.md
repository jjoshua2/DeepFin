# Actual subset successor and ordinal laws

This additive suite stacks on the nine compact-index laws. It proves **seven
new universal laws**, reusing the eight original engine and 16 compiler U64 laws.
No production runtime source, previous statement, compiler or pin is changed.

## What is now proved

Let `cap(mask)` be mathematical `2^popcount(mask)` (the exact capacity equality
is imported from `index/LAWS.bend`). Let `at` be the existing Nat recurrence that
calls the actual `Subsets.next`, from zero.

| Law | Actual implementation / guarantee | Preconditions |
| --- | --- | --- |
| `subtraction_refinement` | Two-U32 `U64.sub`, including actual low-half comparison and high-half borrow, equals `Word.sub(64,...)`. | All U64 operands. |
| `step_successor` | PEXT of actual `Subsets.next` has the next compact value, wrapping to zero at capacity. | The starting state is mask-contained. |
| `sequence_ordinal` | `value(pext(at(i,mask),mask)) = i`. | Mathematical Nat `i < cap(mask)`. |
| `sequence_nonduplicating` | Equal recurrence states imply equal indices. | Both indices strictly below capacity. |
| `sequence_coverage` | Every mask-contained state is reached at a bounded index. Witness: its actual PEXT value. | Mask membership only. |
| `cycle_endpoint` | `at(cap(mask),mask) = 0`. | All U64 masks. |
| `sequence_periodic` | `at(i+cap(mask),mask) = at(i,mask)`. | All Nat indices and U64 masks. |

The `Spec.cycle` observation is explicitly piecewise: return `v+1` if it is less
than capacity, otherwise zero. On the separately proved compact range this is
wraparound successor modulo capacity. This suite does not assert an additional
refinement of the library's division/remainder implementation.

`Borrow.bend` proves generic ADC concatenation, comparator/final-carry agreement,
and equivalence of high-half decrement with borrow-in. `Step.bend` connects the
real arithmetic to a structural masked-bit increment, then proves its extracted
Nat value. `Ordinal.bend` inducts over the imported recurrence and derives coverage,
nonduplication and period; `Value.bend` supplies injectivity of actual Word/U64
numeric observations. The specification/helper definitions are **not** runtime
replacement algorithms. Production `Tables.fill` still calls unchanged Subsets.

No machine-width power appears in these bounds. Empty masks have capacity one;
full masks have mathematical capacity `2^64`. Proofs are structural and quantified,
not a runtime enumeration of that period. The importing consumer exercises arbitrary
masks, actual PDEP-produced members and cross-half/high-bit boundaries.

## Opt-in verification

```sh
bun native/bend_engine/standalone/proofs/successor/verify.js /path/to/pinned/bend
bun native/bend_engine/standalone/proofs/successor/verify_native.js /path/to/pinned/bend
```

Both accept `--report FILE`. The aggregate invokes the unchanged index gate and
therefore retains all prior laws and 26 previous negative controls. It requires
exactly `All terms check.` and status zero, verifies compiler identity, validates
imports and required law names, and rejects unsafe/foreign proof dependencies.
Sixteen new controls cover missing/hole/omitted obligations, false bounds and
membership, incorrect wrap, actual step/recurrence mutations and unsafe imports.
Three disposable Base mutations specifically fail the **new subtraction bridge**:
dropped borrow, borrow on equal halves, and a zeroed high result. The protected
checker and original Base are never edited. Manifest/policy rejection is distinguished
from semantic checker failure; not every negative is a semantic counterexample.

The separate native probe calls actual `U64.sub`, `Subsets.next` and PEXT. It compares
1,605 operand pairs against full-width BigInt subtraction and independent bit-position
scatter/gather of the next compact index. Generic/portable/native/UBSan repeat the
same fixtures, including populations 0..64, 815 low-half borrows, 790 no-borrow cases,
234 wraps and 1,371 advancing states. Four malformed/budget requests reject per mode.
These native checks do not prove the compiler or replace source laws.

## Remaining boundary

The general P1 ordinal/coverage result is now proved; **chess-mask population bounds
and P2 affine table refinement remain**. This suite does not relate every Array write,
region offset, initialization or `Chess.bend` lookup to independent ray geometry.
Nor does it requalify the full engine, model, GPU, or training pipeline. The pinned
checker/Base, native lowering/runtime, ABI, toolchain and hardware remain explicit
trust boundaries. See `docs/bend_migration_proofs.md` and the dated readout.
