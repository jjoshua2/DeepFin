# Source contracts for the production subset step

This is the first **partial P1 source-proof increment**, not a proof of the whole
engine, slider tables, native compiler or neural model. The executable application
already constructed its tables in Bend. `Tables.fill` now calls
[`Subsets.next`](../Subsets.bend), which retains exactly `(subset - mask) & mask`.
No Python responsibility or model arithmetic is newly migrated here.

## Checked contracts and their limits

`LAWS.bend` imports that actual production module; `PROOF.bend` imports LAWS and
discharges all eight accepted obligations. `Mask.bend` supplies structurally
recursive Word/Boolean lemmas, not an alternative enumerator.

| Law | Quantification and established property |
| --- | --- |
| `step_in_mask` | Every U64 input and mask: the next state has no bits outside its mask. |
| `sequence_in_mask` | Every mathematical Nat index and U64 mask: the zero-start recurrence stays masked. |
| `sequence_extract_deposit` | Every Nat index and U64 mask: extracting and depositing that state recovers it exactly. |
| `empty_mask_sequence` | Every Nat index: the empty mask always gives zero. |
| `cross_low_half` | Closed subtraction/borrow witness crossing the low 32-bit half. |
| `cross_bit63` | Closed transition into bit 63. |
| `mask_cycle_end` | Closed return-to-zero witness for the cross-half mask. |
| `full_word_wrap` | Closed return-to-zero witness with an all-ones mask. |

The first four laws have no narrowed chess-mask precondition. The other four are
closed equalities, **not universal order/period theorems**. The Nat recurrence calls
the imported implementation at every step; it does not wrap its index at 2^32 or
2^64. Connecting this recurrence to `Tables.fill`'s affine array writes is still
open. Lossless extraction does not establish that consecutive extracted indices
are 0, 1, 2, ... or that every occupancy occurs once.

The `u64/` files are **byte-for-byte reuse**, not 16 newly proved engine laws, from
`jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae:demos/proof_u64`.
`provenance.json` and the gate bind all three original source blobs to this pin.
The aggregate checks these 16 inherited obligations along with the eight new ones.
No holes, unsafe helpers, foreign equality witnesses or additional axioms are used.

## Opt-in source gate

From the repository root, with the existing standalone compiler source checkout:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/verify.js \
  /path/to/pinned/bend/source --report /tmp/deepfin-subset-proofs.json
```

The gate verifies the existing 84-input compiler fingerprint before and after
checking. Success requires exit status zero **and exactly `All terms check.`**;
exit zero plus an unsafe/foreign warning is not accepted. It requires all eight
named laws and the original inherited files, traverses the actual import graph,
and rejects unsafe code, holes and foreign imports. `Base` and the checker are
trusted through the unchanged source pin, not proven by this gate.

Eleven disposable negative controls exercise missing proofs/imports, a false law,
a hole, missing final masking, reversed subtraction, a stuck-at-zero difference,
truncated subtraction, an unsafe proof helper, changed inherited provenance and
a missing inherited proof. Each implementation control must fail at its intended
law. The unsafe helper currently produces a raw CLI exit of zero; the gate still
rejects its warning. No real compiler or working implementation is mutated.

## Separate native/reference gate

```sh
bun native/bend_engine/standalone/proofs/verify_native.js \
  /path/to/pinned/bend/source --report /tmp/deepfin-subset-native.json
```

This compiles only bounded subset/table probes and the unchanged external C table
reference, one compiler at a time. It uses neither Python nor a model. It checks
all 107,648 occupancy states across 128 chess masks, plus 5,305 states in 225
synthetic cases spanning empty masks, single bits, every population 1..64 and
cross-half/full-width wrap boundaries. Larger-than-chess populations use bounded
prefix/suffix cases; they are **not exhaustively enumerated**. There are 112,953
reported positive state rows per mode, repeated in generic, forced-portable,
native-target and UBSan C, not four disjoint sets.

The oracle deposits compact index bits directly into independently constructed
geometric masks. It does not copy the carry-rippler recurrence. Independently
computed rays and the unchanged separate CBoard reference check all 108,160 logical
`Tables.build` entries. Native checks also reject five malformed/over-budget probe
requests per mode. These counts do not imply source-proved ray geometry, buffer
ownership, initialized-region bounds or compact-index order. UBSan covers the
compiled probes, not arbitrary target hardware.

Neither gate is added to ordinary pytest, existing perft fixtures or permanent CI.
No performance result is claimed. See the [migration/proof matrix](../../../../docs/bend_migration_proofs.md)
and [dated evidence record](../../../../docs/experiments/2026-09-21-bend-subset-source-laws.md).
