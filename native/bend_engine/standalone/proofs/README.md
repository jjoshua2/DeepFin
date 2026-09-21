# Slider-index laws for the production Bend engine

These proofs import the actual `Tables.bend` and `bitboard_probe/Sliders.bend`.
They do not replace the engine with a verification-only chess implementation.
`../LAWS.bend` declares ten contracts; `../PROOF.bend` discharges them.

## Run the complete gate

Build or fetch the compiler revision in `../toolchain.json`, then from the
repository root run:

```sh
bun native/bend_engine/standalone/verify_laws.js /path/to/pinned/bend \
  --report /tmp/deepfin-slider-laws.json
```

This verifies all 84 compiler/effect inputs, byte-identical proof dependencies
against the pinned Git objects, the universal bridge, 17 rejection controls,
and all ten application laws. Checking the complete finite domain is real
source-level computation, not a cheap sampled test; the checker has a bounded
30-minute timeout. `--controls-only` is a development shortcut and explicitly
reports that the full laws were NOT checked. CI does not use that shortcut.

To invoke the proof checker directly, without the pin/failure controls:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/PROOF.bend --check-only
```

Only the exact clean verdict `All terms check.` is accepted. An unsafe/foreign
warning can accompany exit status zero; zero by itself is not proof acceptance.
The gate never runs a Bend `main` during proof checking.

## What the laws establish

Nine contracts are universally quantified, with explicit preconditions where
needed; one is an exhaustive closed computational certificate.

- `index_ignores_off_mask`: arbitrary bits outside the mask cannot change the
  actual U32 lookup index. The supporting word theorem works at arbitrary width.
- `index_full_width` and `index_recovers_masked_input`: when the full PEXT result
  has zero high word, the production U32 index is lossless and depositing it
  recovers the masked occupancy. That high-word precondition is NOT omitted for
  unrestricted masks. The bit-63/all-ones-mask negative control rejects omission.
- `reflection_preserves_conditions`: for every input, the faster proof-only
  predicate equals the original eight conditions expressed with public U64.pext
  and the actual Sliders.pext_index. No safety condition is dropped.
- `fill_matches_lookup`: for any array and a checked walk, the actual sequential
  `Tables.fill` equals a specification that writes each attack using the actual
  `Sliders.pext_index` and the address expression checked against `Sliders.lookup`. This is induction over the real
  Array.set loop, not a comparison between two test-only implementations.
- `lookup_uses_checked_address`: the actual reader uses the address expression
  certified by the index/bounds proofs; its real Array.get call is checked.
- `every_visited_slot_is_safe`: every ordinal below the walk length inherits
  the slot conditions, not only its first or last member.
- `canonical_tables_checked`: check all 128 production masks and every one of
  their 107,648 generated occupancy subsets. The checked carry-rippler step is
  `(subset - mask) & mask`; `Core.bend` verifies it, the mask constructor and
  segment size against the unmodified production loop, not merely a test model.
  At each state PEXT equals its sequential ordinal, its high word is zero, the
  ordinal fits the segment, and the full-width offset sum equals the actual
  unwrapped address without overflow. Every address is in `[512,108160)`.
  Every segment cycles back to zero at exactly its size; the final cursor is
  108160, and each mask has at most 12 selected bits.
- `every_canonical_segment_is_safe`: every ordinal below 128 inherits its
  descriptor/count and complete walk certificate.
- `tables_match_lookup`: lift the fill proof through the actual production
  metadata writes and the outer table loop, for any initial array. Instantiate
  it with the canonical certificate to connect all segments, not just a sample.

`Spec.bend` spells out the predicates and the PEXT-indexed specification;
`Core.bend` proves the implementation/specification bridge;
`Bits.bend` supplies generic bit and Boolean-to-equality lemmas.
`Lookup.bend` checks the actual read expression. All pre-existing engine Bend
files are unchanged; proof-friendly specifications remain outside the runtime.

To keep exhaustive normalization practical, `Fast.bend` gathers selected bits
into a list and pads the result once. Induction proves it equal to the public
`Word.pext` for every word width, input and mask; `Core.step_agrees` transfers
that equality to all eight original safety conditions. Both proofs are checked
as part of the complete module. The engine and its native PEXT path are not
changed. This is verified reflection, not trusting a second implementation.

The finite certificate is checked by Bend's source normalizer, not generated C,
Python, an SMT oracle, or trusted generated answers. It enumerates the entire
canonical finite domain rather than proving the carry-rippler for every possible
64-bit mask. The surrounding bridge and ordinal-selection theorems are generic
inductive proofs. Both methods are explicit in the source.

## Reuse and dependencies

Application proof files can import both declarations and implementations:

```bend
import ./LAWS.bend as Contracts
import ./PROOF.bend as Proven
```

Reference theorem names through `Contracts`, for example
`Contracts.index_ignores_off_mask(occupied, mask)`. Importing the proofs does not
implicitly re-export their aliases. Importing the full module checks the complete
finite certificate, so small lemma-development files should import `Bits.bend`
or `Core.bend` directly until their full integration check.

`u64/` contains verbatim copies of the three proof files and the Apache 2.0
license from the exact compiler commit. `dependencies.json` records their source
paths and SHA-256 hashes. The gate checks those bytes against immutable Git
objects and requires the same revision as the compiler manifest. Upgrade them
as one reviewed change; do not silently mix a new compiler and older proofs.

## Failure controls and review boundary

Seven cases must be rejected by the checker: high-word indexing, wrong subset
order, shifted lookup addresses, shifted production writes, wrong production
segment offsets, wrong production metadata, and unrestricted-mask truncation.
Two further checker controls corrupt the reflection extractor or replace its
entire safety predicate with True; both must invalidate the equivalence proof.
The pinned cyclic-template false-proof regression is also rejected. Four policy
controls reject missing proofs/imports, holes and unsafe annotations. One checks
that a raw status-zero unsafe warning is not accepted; two check vendored proof
content/revision drift. These total 17 controls, with the rejecting layer named
in the JSON report. The separate existing compiler suite has 12 integrity tests.

The policy is a CI guard, not protection against an author who can change the
laws, predicates, verifier, and CI together. Review specification changes.
No unsafe annotation, foreign proof, admitted axiom, hole or checker change is
introduced. The ordinary engine's existing IO/serve unsafe boundary is not
imported into this pure proof graph.

## What remains separate

These are source properties conditional on Bend's checker. They do not verify
the C lowering, C compiler, CPU, complete move generation, draw rules, neural
encoding, or playing strength. The mask-invariance theorem applies to full-board
inputs, but a separate general coverage/range theorem for every arbitrary
masked input is not claimed. The certificate covers every state in the actual
canonical generation walks. The conditional high-word theorem retains its
precondition rather than pretending all masks fit U32.

The array-equality theorem covers `Tables.fill` and `Tables.tables`, including
their metadata writes. It does not prove Array.get/set's general algebra, the
wrapper allocation capacity, that no later caller corrupts the table, or the
geometric correctness of `Tables.slider` itself. The existing build allocates
2^17 slots, larger than the checked used interval, and retains independent
complete-table and engine regression comparisons. No deeper perft or recurring
model export is added. Runtime proofs are not linked or executed in the engine.

The pinned fork's known upstream strict-TypeScript diagnostics remain distinct
from this source-proof gate. No diagnostics, token caps or numerical tolerances
were relaxed to obtain a proof or a runtime pass.

## Separate opt-in native qualification

With Clang, Python 3.13, NumPy 2.2.6 and python-chess 1.999 installed outside the
engine, the existing comparisons can be repeated without a model dependency:

```sh
bash native/bend_engine/standalone/qualify_toolchain.sh \
  /tmp/new-slider-qualification /path/to/pinned/bend
```

This orchestration reuses the existing oracles unchanged: all 108,160 used table
entries, UCI/legal children, unchanged perft depths, draw/history rules, and the
112-plane diagnostic encoder. It tests generic, forced-portable, native, UBSan,
and static builds, with the static engine inside an otherwise empty chroot.
The chroot check requires `sudo`. `reports/` contains compact results; binaries,
full tables and generated C stay in the explicitly requested output directory.
This script is opt-in, not part of the recurring source-proof workflow.
