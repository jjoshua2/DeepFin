# Exact U32 slider-index correspondence

## Scope and acceptance

Parent: PR #819 at `40803892f3204a54a33a87e2e7b25fa48fed2e1a`.
Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
with the unchanged 84-input fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This record was written after local constructive proof development and before
hosted qualification. It is not a backdated preregistration. The accepted target
is exact equality between actual U32 slider lookup indices and full-width PEXT,
then composition with the completed recurrence ordinal. In-range indices alone
do not establish identity. No production code or earlier statement is altered.

The six public laws quantify over their stated symbolic widths, valid chess keys,
Nat indices and arbitrary U64 occupancies. Population certificates needed by helper
functions are discharged by the existing imported layout proof, not added as new
assumptions to the public contracts. The detailed statements and commands live in
`native/bend_engine/standalone/proofs/exact_index/`.

The source gate must retain all 48 parent laws and 55 parent rejection controls,
check all six new laws and the importing consumer, and reject all 15 additional
mutations/policy violations. Exactly `All terms check.` plus successful execution
is required. The native probe must compare actual recurrence states, full PEXT and
U32 lookup results for every chess-mask subset, plus the all-irrelevant-bits-set
contrast, against independent coordinate geometry and compact-bit deposition.

Checks are opt-in and bounded. No deeper perft, full-engine/model build, GPU,
training, benchmark, permanent workflow, merge, deployment or live-process change.
One native compiler at a time. Self-review only unless separately recorded.

## Local source provenance and construction

The local workspace was reconstructed from the saved PR #807 source archive and
the exact saved slider-mask patch. All 35 inherited proof-graph input hashes match
PR #819's hosted `layout-proofs.json`; the compiler's full fingerprint also matches.
This is a hash-qualified source snapshot, not a freshly cloned full current Git
history. Direct network Git access was unavailable; hosted qualification uses a
complete checkout of the exact parent and the published candidate commit.

An exploratory direct 2^32-bound proof triggered the checker's existing deep
normalization limit. The retained structural lemma keeps k symbolic, for every
k <= 32, and derives chess-mask applicability from the existing population proof.
The checker and theorem domain were not weakened. An initial full-import helper
check exceeded its 180-second exploratory timeout without a verdict. Helpers were
then organized around explicit proof certificates, with all certificate producers
retained in the final aggregate import graph. Intermediate ordinary type/import
errors were corrected in the new proof code. None is counted as a passing check.

## Interpretation

The exact lookup index reconstructs **masked occupancy**, not the full board.
Collision soundness requires a valid chess mask; the reverse implication
(equal masked occupancies imply equal indices) holds for every U64 mask. A 33-bit mask counterexample is retained:
low-U32 indexing can discard the 33rd compact bit. The native boundary witnesses
also check that the supported 32-bit projection itself preserves all 32 bits.

This increment does not prove prefix offsets or arithmetic of `Tables.tables`,
the writes of `Tables.fill`, initialized/disjoint Array regions, or attack equality
with independent blocker rays. Those remain the next P2 acceptance target. The
source checker/Base, native lowering, affine runtime, effects/ABI, C compiler and
hardware remain separate trust boundaries. Existing compiler diagnostic/strict
TypeScript limitations are not suppressed. Python export, production/data control
and training, and C++/LibTorch/AOTI inference remain migration dependencies.

## Local results

The complete aggregate source gate passed: **54 laws** (six new plus 48 inherited)
and **70 rejection controls** (15 new plus 55 inherited), including the importing
consumer. It completed with exit zero in 445.00 seconds. The unmodified local
source report SHA-256 is
`168c52047083dccd2b131c762a9578b0a1b1bd0bc9eb36f887264f7b2880a079`.
The runtime-mutant checks reject wrong in-range indices in independent closed
source witnesses; manifest/policy rejections are distinguished in the report.

The first native compilation failed because the new probe matched a computed
parser result directly. A separate parameter-matching helper fixes that harness
syntax, without changing any law, proof helper, oracle or acceptance criterion.
The corrected native gate passed generic, forced-portable, native-target and
UBSan builds: **107,648 enumerated states, 215,296 indexed occupancies, 128 cycle
endpoints and six rejected requests per mode**. All modes have output SHA-256
`37cd7ab0da1fc36e2292fb00dc49fcd563ee0024c4656d55e47b8f182a6995da`.
The same fixtures are repeated in four environments, not disjoint datasets.
Bun 1.4.2 and Clang 17.0.0 were used; one compiler ran at a time.

The first source report fingerprints the pre-correction native probe, which it
hashes but does not execute or import as proof input. Every actual proof input
and the proof gate remain unchanged. The original source report is retained
unchanged; final native evidence identifies the corrected probe. Hosted
qualification must rerun the complete source gate on the final candidate and
verify its final file identities rather than relabel the earlier report.

Whole-repository lint is reserved for the complete locked hosted checkout; it
was not run on the reconstructed local snapshot. No hosted result or clean
branch publication is claimed until the follow-up record below exists.
