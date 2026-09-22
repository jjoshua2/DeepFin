# Exact U32 lookup-index correspondence

## Acceptance and isolated baseline

Before implementation, the continuation recorded its acceptance in a local
`acceptance.md`: preserve the pinned compiler and old laws/controls; prove exact
low32/full-width correspondence, actual chess lookup equality, and the connection
to imported recurrence order/coverage; reject an in-range constant-zero lookup.
No production change, model work, perft increase, merge or deployment was authorized.
The budget is one compiler at a time, focused native/source gates, original pin/source
checks, and one whole-repository lint attempt. The retained acceptance text is copied
into this record's evidence directory after execution.

The base is PR #819, `40803892f3204a54a33a87e2e7b25fa48fed2e1a`, branch
`feat/bend-slider-mask-bounds-20260921`. Its complete source tree was reconstructed
from retained source/patch/qualification artifacts and the final published metadata.
All entries, including tracked data, scratchpad, file modes, CI repairs and final
hosted documentation, match Git tree `50269344ff41534586f485a1b5aa70875e465a46`.
This is a full source-tree identity check, not an assumption from a few snippets.
The unsigned base commit object was also reconstructed byte-for-byte (including
its recorded timezone and message termination) and independently hashes to the
published commit SHA. The isolated local repository is shallow at that commit;
no remote history or live checkout was rewritten.

Compiler: `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
84 source inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The compiler branch was refreshed and still names that revision. Current repository
instructions and the existing proof inventory guided the isolated continuation.

## Eight new contracts

For a mathematical width `k <= 32`, a U64 whose value is below `2^k` retains its
entire value when projected through actual `U64.low`. Applying the existing PEXT
range theorem gives exact actual `Sliders.pext_index` for every mask of population
at most 32. The earlier mask-population proof supplies that bound for all 128 chess
keys without assuming the desired count as a new user precondition.

| Accepted law | Guarantee and explicit domain |
| --- | --- |
| `low_projection_exact` | Actual low32 projection preserves value for arbitrary U64 below mathematical `2^k`, `k <= 32`. |
| `compact_index_exact` | Actual U32 lookup value equals full PEXT for any occupancy and mask with population <=32. |
| `chess_lookup_exact` | Same exactness for each valid U32 chess key (<128) and arbitrary occupancy. |
| `chess_lookup_word` | Widening the actual lookup result reproduces the full PEXT U64 word. |
| `chess_sequence_ordinal` | The actual U32 lookup of the existing recurrence state at Nat `i` equals `i` below capacity. |
| `chess_sequence_coverage` | A mask-contained state is recovered by the recurrence at its actual U32 lookup value. |
| `chess_lookup_injective` | Equal actual indices imply equal mask-contained states. |
| `chess_lookup_redeposit` | Depositing the widened lookup value recovers the original occupancy masked to relevant bits. |

`Projection.bend` uses structural Word induction, not enumerated U64 operands.
`Correspondence.bend` imports the actual functions and composes earlier extraction,
ordinal and representation proofs. Its helpers receive explicit count certificates
so small helper iterations avoid rechecking all finite geometry cases. The public
`PROOF.bend` imports the old layout `PROOF.bend` and obtains those certificates from
its proved law. No accepted public law introduces an assumed count or weakened domain.

The consumer includes arbitrary-key/occupancy uses, empty masks, bit63 and cross-half
examples. A 33-selected-bit counterexample shows why the general width restriction
is necessary: the full PEXT word can have bit32 set while the U32 lookup is zero.
A starting-square occupancy outside a rook mask shows why recovering the unmasked
original state requires membership rather than only knowing the index.

## Development corrections and observed limitations

An initial helper signature expanded concrete `Bits.power(32n)` too eagerly and
triggered a checker stack overflow. The implemented structural theorem keeps the
width symbolic, including 32, and does not edit the checker or lower the accepted
width. This is a proof-construction/resource correction, not an inferred counterexample
to the arithmetic. The literal-expansion limitation is not claimed fixed in the checker.

During helper construction, an opaque Type alias could not be duplicated as Data;
explicit equality-certificate types resolved that ownership issue. The native probe
also needed a Bool helper rather than matching a computed scrutinee. An exploratory
consumer check was stopped after an incorrectly ordered `U64.from_parts` example was
noticed; `from_parts` takes high then low, and the corrected examples were checked in
the final consumer. Earlier exploratory starts are not counted as successful gates.

The completed candidate changes no existing implementation or accepted law. It adds
no axiom, unsafe dependency, foreign equality witness or proof hole. New rejection
controls distinguish semantic checker failures from manifest/import-policy failures.

## Historical local execution readout

The final local candidate completed its bounded source and native qualification.
The aggregate source gate reports **56 accepted laws**, including eight new laws
and all 48 prior laws, plus **72 rejection controls**: 17 new and 55 inherited.
The unchanged parent gate runs first; the actual new importing consumer then checks
all new obligations with the prior count/ordinal proofs in its import graph.
All semantic mutations receive expected/observed proof rejection, not a crash or timeout.
The unsafe projection mutation has raw CLI status 0
and is rejected by the import policy and exact-output guard.

| Executed check | Outcome |
| --- | --- |
| Aggregate source laws and controls | PASS: 56 laws, 72 controls, importing consumer. |
| Native generic / forced-portable / native-target / UBSan | PASS: 2,070 fixture rows and five malformed/budget rejections per mode. |
| Original compiler source suite | PASS: 16 laws and seven controls, including cyclic-template rejection. |
| Original compiler-pin contracts | PASS: 12 passed, zero failed. |
| Whole-repository `scripts/lint.sh` | UNQUALIFIED (command failed). |

Native coverage is 1,536 chess-key rows across every key 0..127 and 534 raw-mask rows;
mask populations 0..64 are represented. There are 28 population-32 rows, 253 wider-mask
rows and 218 explicit truncation counterexamples outside the exactness domain.
The probe also checks 1,906 first-cycle rows and 164 at/after-cycle rows using the
actual imported recurrence, with mathematical direct-deposition reference states.
These are the same fixtures repeated in four environments, not disjoint datasets
or exhaustive native testing of arbitrary U64 operands. The shared output SHA-256 is
`51fb92afe49cb0895f6c8125e05a9a0a8e50c877c43f2ee3577df88d131d8caa`.

Toolchain: Bun 1.4.2; clang version 17.0.0 (https://github.com/swiftlang/llvm-project.git 10999b6d034fe318f3d56c83bddb6572593a8bb0); Linux x86-64.
External execution recording used standard Python 3.13.5; no Python controller runs
inside the native candidate. All candidate source hashes and the 84-input compiler
identity are checked before/after gate execution and again before packaging.
Source gate elapsed 446.41 seconds; native gate
elapsed 5.59 seconds. These are command/resource
records, not a comparative benchmark or engine-runtime performance claim.

The unchanged whole-repository lint attempt exited 1. Its original output is retained. Ruff, Basedpyright and Vulture were not installed. This is a missing-tool environment failure, not a passing lint check or an established source finding. No lint requirement was bypassed.

The read-only GitHub actions available during this continuation do not expose source
publication. Plugin discovery found no usable alternate publishing action, and the
local runtime has no usable authenticated GitHub CLI path. Consequently this record
is retained in a **local commit and patch**, not a new remote commit/PR or hosted
qualification. PR #819's previous hosted pass belongs to its baseline, not this new
candidate. No branch protection, required checks or existing workflows changed.

The `evidence/bend-exact-slider-indices/` directory retains both gate reports,
all execution outcomes, original stdout/stderr losslessly encoded as JSON strings
with SHA-256 identities, baseline provenance and the pre-implementation acceptance.
No executable, compiler distribution, model data or private inference trace is committed.

## Interpretation and next acceptance

Exact correspondence is stronger than an index bound: replacing the lookup with zero
would remain in range yet select the wrong state. The new semantic control requires
rejection of that implementation, and the laws connect the precise lookup value to
the previously proved enumeration ordinal rather than merely repeating range checks.

This increment does not prove prefix-offset addition, initialized/disjoint Array
regions, affine writes/reads, actual `Chess.slide` table accesses, full mask geometry,
or blocker-ray lookup equality. The next acceptance is the prefix and storage
connection using these exact indices, followed by the independent ray specification.
Existing native table comparisons remain useful but are not that refinement theorem.

Self-review only; no independent reviewer was available. Source laws trust the pinned
checker and Base semantics. Native lowering/runtime, effects/ABI, toolchain, libraries,
OS and hardware remain distinct trust boundaries. The fork's inherited strict
TypeScript failures were not addressed, rerun or suppressed here.

No additional responsibility moved from Python into Bend. Export, external references,
production/data orchestration and training remain dependencies; C++/LibTorch/AOTI
remains the explicitly transitional inference backend. No full-engine/model build,
model forward, GPU, training, perft, benchmark, strength or speedup result is implied.


## Hosted qualification and publication

Hosted qualification **35706095107** on September 22, 2026 passes all 56 laws, 72 rejection controls, four native modes, original compiler source/pin checks and whole-repository lint. The original saved commit `53b6e78f3052a0ed273f7fc31be47b66b32707b9` and complete tree `fc6400fbefe04deded9d6e1c9ac3ff54ef883221` were recovered exactly, not rewritten.

Temporary workflow commit `7a906f4a2e32b1d564dd747a8e6b2eeefb5658dd` checks all four transport fragment identities, original patch SHA-256 `decd7ad1380771334ca0014ad474e9e7e4f578537ee97a34304621cf6d24b2c6`, full source tree, original commit and every recorded candidate source hash. The reconstructed commit retains its original author/committer metadata. Only this subsequent documentation/evidence commit is new.

The hosted source report is byte-identical to the saved local report. Native results and all source hashes match; only the separately reported C compiler identity may differ. Each mode checks 2,070 rows, all 128 chess keys, 28 population-32 rows, and 218 explicit wider-mask truncation counterexamples; five invalid requests are rejected. Repeated modes are not disjoint data sets or exhaustive arbitrary-U64 tests.

Toolchain: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1), locked Python 3.13 CPU environment and uv 0.12.10. The unchanged whole-repository Ruff/Basedpyright/Vulture gate passes. This resolves the historical missing-tool lint gap; original failed local logs remain unchanged. The compiler fork's separately documented strict-TypeScript and diagnostic-depth limitations are not fixed or suppressed here.

A parallel existing branch was found during refresh: feat/bend-exact-slider-index-20260921 at 5c05c5c281212e313a630f077ab466fbd650985d, with six laws in proofs/exact_index and its own hosted record 35684436097. It shares this parent but is not this eight-law proofs/address candidate. That branch and all its evidence are preserved untouched. Its test counts are not attributed to this candidate, and none of this candidate's eight accepted statements are dropped. Future consolidation must map contracts and preserve both sets before removing duplicate helpers.

Publication creates `feat/bend-exact-slider-indices-20260921` only after qualification. No merge, force push, deployment or live-process operation. The temporary workflow and transport payloads are absent from the feature tree. Compact hosted evidence is committed; full gate logs remain in the 30-day artifact exact-slider-indices-qualification.

Self-review only, not independent review. No production code, earlier laws, compiler inputs, existing tests or routine test budgets changed. No new full-engine/model build, perft, GPU, training or benchmark result. Python export/data/control/training and transitional C++/LibTorch/AOTI computation remain dependencies.

Exact scalar lookup/ordinal/redeposit claims are now qualified on this source. Actual prefix offsets, initialized/disjoint affine writes/reads and equality to independent blocker rays remain the next acceptance; this publication is not that storage proof.
