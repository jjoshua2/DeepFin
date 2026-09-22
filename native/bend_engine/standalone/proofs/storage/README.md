# Actual affine-array storage laws

This opt-in suite continues the exact scalar-index work in `../address/`. It
changes no runtime code, earlier law, compiler input, test or permanent workflow.

## Accepted contracts

All eight statements in `LAWS.bend` mention actual `Array<U64>` operations or the
actual imported `Tables` functions. There is no hidden valid-buffer certificate
that assumes the desired result, nor a new axiom about storage.

| Contract | What the equality observes |
| --- | --- |
| `read_preserves_storage` | `Array.get(a,i)` returns exactly the original array as its first component. |
| `write_read_same_location` | Reading after `Array.set(a,i,v)` returns the entire updated array paired with `v`. |
| `initialized_read` | Any source-level read of `Array.new(d,v)` returns the entire new array and `v`. |
| `write_preserves_shape` | An actual write preserves the complete binary constructor topology. |
| `fill_preserves_shape` | The actual `Tables.fill` loop preserves that topology for every Nat iteration count. |
| `tables_preserve_shape` | The actual mask/header/block builder preserves topology. |
| `extras_preserve_shape` | The actual knight/king/pawn initialization loop preserves topology. |
| `fill_one_read` | Reading the first address after one actual fill step returns the actual `Tables.slider` output and the full updated array. |

The arbitrary-array contracts include nonuniform constructor trees at source
level. The new-array contract quantifies over mathematical Nat depth, not a
promise that a native allocation of every such depth can succeed. Only U64-valued
arrays are covered by these new laws, not every possible generic element type.

`Representation.reify` consumes an affine array once and yields a duplicable
constructor image plus a proved equality back to the actual array. Induction is
performed on this image and transported through that equality. `Read`, `Write`,
`Refinement` and `Roundtrip` connect actual Base APIs to structural observations;
`Build` inducts over the real `Tables.fill/tables/extras` definitions. Their scalar
abbreviations must reduce to the actual expressions for these equalities to check.
Neither the reifier nor `Model` is imported by the application or native probe.

## Important non-claims

`Array.get` and `Array.set` mask indices. The two-leaf consumer counterexample
writes index 2 and reads index 0 successfully: distinct numeric indices can alias.
Thus these laws do **not** establish prefix offsets, no arithmetic overflow,
disjoint regions, preservation of other physical slots, complete initialized
slider contents or equality with an independent blocker-ray specification.
A read after initial allocation returns the seed; that is not proof that every
lookup later reads a computed attack rather than a seed.

Shape preservation is topology preservation, not unchanged values. A no-op writer
could pass that property; `write_read_same_location` and `fill_one_read` supply
separate value-sensitive checks. The latter proves storage of the actual slider
result, not that the slider result itself equals correct chess geometry.

Source laws trust the pinned checker/Base. Native array lowering, physical
ownership/lifetime, allocator failure, toolchain and hardware remain separate.

## Checks and costs

From the repository root with the pinned compiler checkout:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/storage/verify.js COMPILER --report /tmp/storage-proofs.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/storage/verify_native.js COMPILER --report /tmp/storage-native.json
```

The aggregate runs the unchanged 56-law/72-control parent once, then the new
importing consumer and 18 controls. It checks 64 accepted laws in separate modules;
it does not redundantly normalize the parent's finite chess-domain proof again
inside this consumer. Every source success requires both exit zero and exactly
`All terms check.`. Semantic controls require an ordinary expected/observed
rejection. Policy controls are labeled separately; a crash is not a valid rejection.

Ten semantic/refinement mutations test no-op writes, wrong stored values, damaged
returned storage, discarded siblings, wrong read routing, wrong fill values and
addresses, and destroyed base-case storage in all three actual builders. Eight
manifest/import/output controls retain all obligations and reject missing imports,
proof holes, foreign/unsafe dependencies and symbolic-link inputs. Compiler
mutations operate only on disposable copies; the real fingerprint is rechecked.

The native probe uses only the actual Base APIs, actual Tables and input parsing.
An external independent flat-array/reference-geometry harness compares every
returned logical cell, not merely a checksum produced by the candidate. It tests
all 128 chess fill keys, partial actual table/header construction, all extra attack
slots, zero-count builders, U64 boundaries and wrapped-index alias cases. Four
modes run identical fixtures: generic, forced-portable, native-target and UBSan.
Native fixtures use complete binary arrays only; source coverage of arbitrary
constructor trees is not native validation of those shapes.

No model, GPU, perft, full-engine build or benchmark is part of this suite. See
`docs/experiments/2026-09-22-bend-table-storage.md` for executed qualification,
identities, limitations and remaining P2 work. Self-review is not independent review.
