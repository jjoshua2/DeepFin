# Exact runtime slider-index correspondence

This opt-in increment extends PR #819 without changing production code, the
compiler pin, any accepted parent law, or any existing gate. `LAWS.bend` imports
the actual `Sliders.pext_index`, the existing `Tables.slider` mask observation,
and the already-proved Nat-indexed recurrence.

## Six accepted targets

- `low_projection_exact`: for any symbolic width k <= 32 and U64 value below
  mathematical 2^k, observing the actual low U32 preserves its whole Nat value.
- `lookup_index_exact`: for every valid chess key and arbitrary occupancy, the
  actual U32 lookup index equals the full PEXT value. This strengthens a bound
  into equality; an in-range wrong index is not accepted.
- `lookup_sequence_ordinal`: for every Nat index strictly below the actual U32
  block size, lookup of that recurrence state returns exactly the index.
- `lookup_collision_sound`: equal actual indices for a valid chess mask imply
  equal masked occupancies, not necessarily equal complete boards.
- `lookup_collision_complete`: equal masked occupancies imply equal indices for
  every U64 mask. Unlike the converse, this direction needs no population bound.
- `lookup_state_recovery`: enumerating to an arbitrary occupancy's actual lookup
  index reconstructs exactly its relevant masked occupancy.

`Projection.bend` is structural Word induction. Its exponent remains symbolic;
no gigantic unary 2^32 literal is normalized. `Correspondence.bend` combines the
existing population certificate, full-width extraction/ordinal laws, and numeric
observation injectivity. Its helper population-certificate parameter is supplied
by the existing `Layout.mask_population` proof in `PROOF.bend`; it is **not** an
extra hypothesis on the six public laws. The aggregate checks those prior proofs.

`Boundary.bend` independently checks the actual runtime function without importing
layout proofs. It covers bit 63, the 31/32 transition, the third compact bit, all
32 compact bits, and a counterexample showing why unrestricted wide-mask collision
soundness is false. A U64 mask with 33 selected bits can collapse a relevant bit
when the runtime index retains only the low U32. The collision-soundness law retains its
chess-mask restriction; irrelevant occupancy bits are not falsely reconstructed.

## Commands

From the repository root with the unchanged verified compiler checkout:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/exact_index/verify.js /path/to/bend --report /tmp/exact-index-proofs.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/exact_index/verify_native.js /path/to/bend --report /tmp/exact-index-native.json
```

The first command invokes the unchanged parent gate: 48 prior laws and 55 prior
rejection controls. It then checks the new complete importing consumer and 15
additional controls. Manifest/import checks are reported separately from checker
rejections. Successful exit plus exactly `All terms check.` is required. Signals,
timeouts, crashes and warnings are not accepted as semantic mutation rejections.

The second command compiles one focused probe at a time in generic, forced-portable,
native-target and UBSan modes. For each of all 128 chess masks it traverses the
actual subset step for the complete block, comparing every state with independent
BigInt bit scatter, its full PEXT and actual U32 index, and the same index after
setting all irrelevant bits. Each cycle endpoint must be zero. Invalid keys,
malformed integers and a bounded request overflow must reject. No oracle data
enters the candidate. Repeated modes exercise the same data, not disjoint samples.

## Limits

No prefix-offset arithmetic, storage initialization, affine-array writes/reads or
independent blocker-ray attack equality is proved here. `Sequence.at` calls the
production step, but is not the actual `Tables.fill` array loop. This increment
closes the exact scalar-index prerequisite; P2 storage/geometry work remains.

The source checker and Base remain trusted. Native lowering/runtime, effects/ABI,
C toolchain and hardware are separate tested/trusted components. No full-engine,
model, GPU, perft, training, strength or performance qualification is implied.
Python export/control/data/training and the transitional LibTorch/AOTI backend
remain unchanged. There is no added ordinary pytest or permanent workflow cost.
