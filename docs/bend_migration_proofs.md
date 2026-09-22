# Bend migration and proof inventory

## Latest hosted storage qualification

Hosted run **35711968651**, temporary workflow commit `ae80e948fdd5ef12011a8d9f5d9ed98462df0968`, passes all **64 accepted laws**, **90 rejection controls** (18 new plus 72 retained), four native modes, original compiler source/pin checks and unchanged whole-repository lint on exact parent `290fc01a10517374518e505f9855c7d4a9a5d956`. The dated table-storage record and committed reports retain exact identities, original local limitations and remaining trust boundaries. Prefix/nonaliasing/full-lookup geometry remain open; these are eight new actual-buffer laws, not completion of P2.

## Affine-storage increment: September 22, 2026

On PR #820 `290fc01a10517374518e505f9855c7d4a9a5d956`, the new
[actual-buffer suite](experiments/2026-09-22-bend-table-storage.md) adds eight
source contracts without changing production or previous laws. Its final
aggregate retains all 56 earlier laws; qualification status is in the dated record.

| New law | Actual functions and assumptions | Meaning / remaining limit |
| --- | --- | --- |
| `storage/read_preserves_storage` | Actual `Array.get`; every affine U64 array and U32 index. | Returned storage is exactly unchanged; no address-bound assertion. |
| `storage/write_read_same_location` | Actual `Array.set/get`; arbitrary array, value and same index. | Full updated array and written value recovered; different indices may alias. |
| `storage/initialized_read` | Actual `Array.new/get`; source Nat depth, U64 seed and U32 index. | Initial read returns seed and same array; not an allocation-success or computed-attack theorem. |
| `storage/write_preserves_shape` | Actual `Array.set`, arbitrary affine array. | Complete constructor topology preserved, not contents or bounds. |
| `storage/fill_preserves_shape`, `tables_preserve_shape`, `extras_preserve_shape` | Actual `Tables` functions, all source Nat counts and scalar inputs. | Structural induction over the real loops; no interval/overflow or geometry guarantee. |
| `storage/fill_one_read` | Actual `Tables.fill(1n,...)/Array.get`. | First write stores actual slider result and returns full updated storage; independent slider correctness still open. |

All actual-array premises are supplied by live structural reification, not assumed
buffer certificates. Proof images never run in the application. Source laws trust
the unchanged checker/Base. Native fixtures use complete arrays, not all arbitrary
source constructor shapes. No new Python responsibility, model/GPU or performance
result follows. The P2 next target remains prefix bounds and nonaliasing/other-slot
preservation, then complete initialized writes and geometric lookup refinement.

Earlier status sections below retain their original revision/qualification context.

## Current qualification update: September 22, 2026

Hosted qualification **35706095107** on September 22, 2026 passes all 56 laws, 72 rejection controls, four native modes, original compiler source/pin checks and whole-repository lint. The original saved commit `53b6e78f3052a0ed273f7fc31be47b66b32707b9` and complete tree `fc6400fbefe04deded9d6e1c9ac3ff54ef883221` were recovered exactly, not rewritten. Historical local-only/lint-unqualified statements below retain their original execution context; the hosted result above supersedes those status gaps for this exact candidate. No prefix/affine-storage theorem follows from this update. The parallel six-law exact_index branch remains preserved and separate. See the dated exact-slider-indices readout and hosted evidence.

Status snapshot: September 21, 2026. This inventory covers published PR #819
(`40803892f3204a54a33a87e2e7b25fa48fed2e1a`) plus the locally qualified
[exact U32 index increment](experiments/2026-09-21-bend-exact-slider-indices.md).
The new source/native gates pass locally; whole-repository lint and hosted
qualification/publication are reported separately in that readout. PR #819's
hosted result remains historical baseline evidence, not a new candidate pass.
The older `feat/bend-native-leaf-evaluation@975cc8f6...` continuation is preserved,
not silently substituted for the validated PR #804 implementation. Prior probes
and unpublished branches are references, not evidence of standalone integration.

## Evidence labels

**Implemented**, **Source-checked**, **Law proved**, **Native-tested**,
**Target-model/GPU-tested**, and **Benchmarked** are different statuses. A native
fixture pass does not discharge a quantified law. Source typing alone is not a
correctness theorem. Blank proof/GPU/performance coverage below is unfinished work,
not an implication that a check was performed or a feature is impossible.

The unchanged standalone compiler pin is
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 plus U64; all 84 inputs have
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The compiler fork's 16 U64 laws are separate from engine laws. Its source-proof,
native/reference and repository-budget gates have recorded passes; inherited
strict TypeScript diagnostics remain separately failing. Nothing here changes or
suppresses that gate, adopts a new compiler, merges a branch or deploys an engine.

## Application ownership / remaining conversion

Paths in this table are under `native/bend_engine/` unless stated otherwise.

| Responsibility / actual source | Standalone ownership and evidence | Remaining conversion / acceptance |
| --- | --- | --- |
| `legal_probe/Chess.bend`, `standalone/{Tables,Position,Text,Protocol}.bend` | Board, legal moves, table construction, FEN/replay/history and bounded UCI are Bend-authored; inherited native/reference checks. The subset step now calls `Subsets.next`. | Full protocol/time controls and required optional draw-claim choices; general legality/completeness and parser-state proofs. |
| `standalone/{Rules,SearchHistory}.bend` | Bend reconstructs actual selected paths and applies automatic draw rules; inherited native tests include repetition/history and mate precedence. | Optional current/prospective claims, intended-move witnesses, exhaustive dead positions; history-safe reuse/transpositions. |
| `standalone/{HistoryEncoding,ClassicalEncoding,ModelInput,Policy,EvaluationInput}.bend` | Complete 146/175-plane input and exact legal policy entries are paired from one Game; native Python/C comparisons. | Quantified buffer initialization, bounds, perspective, history and mapping refinement. Array capacity is not logical tensor width. |
| `standalone/{NativeEvaluation,LogitReply,main}.bend`, `session_probe/Search.bend` | Actual leaf integration and legal-only policy/WDL normalization are Bend-authored; PR #804's untrained CPU fixture passes actual selected-leaf traces. | Trained checkpoint and target CUDA qualification; production Gumbel behavior, batching and responsive scheduling. Diagnostic PUCT is not production-search parity. |
| `standalone/{model_call.c,model_bridge.cpp}` | Bounded effect/ABI and LibTorch/AOTI execution are an explicitly transitional native model backend. Model remains loaded between searches. | Bend-authored tensor/model computation. Python-free deployment does not establish Python-free export/training. |
| Asynchronous worker/batcher/multi-game lifecycle | Existing earlier probes are references; the standalone model call is synchronous. | Bounded queues, unique IDs/epochs, real-row/padding accounting, deadlines, cancellation/storage ownership, model-update boundaries and blocked-evaluator readiness/stop. |
| Python production pipeline (`chess_anti_engine/selfplay/`, `worker*.py`, `replay/`, `train/`, `model/`) | Production control, data tooling, model definition/export and training remain separate migration dependencies, not claims of Bend implementation. | Inventory enabled source/config/checkpoint contracts before porting RNG, records/shards, retries/recovery, losses/targets, optimizers, augmentation, schedules, SWA, resume, precision and synchronization. `policy_sf` predicts the opponent reply, not the current move. |
| Practical viability | New tests are correctness checks, not benchmarks. PR #804's 9.22-GiB peak is a build measurement. | Compare equivalent model/precision/batching/history/search/hardware; separate startup, compilation, encoding, inference, search, scheduling and runtime memory. |

No new Python responsibility moves into Bend in the proof increments: this arithmetic
was already Bend-owned. The new successor/ordinal suite changes no production source;
it proves the actual helper and imported recurrence, with separate native checks.

## Accepted laws / assumptions / revision links

The initial eight engine laws live in `standalone/proofs/LAWS.bend`, import the actual `Subsets.bend`
used by `Tables.fill`, and are discharged by `PROOF.bend`. Their exact application
and proof source blobs are retained in the dated source report. `Mask.bend` uses
structural Word induction. The original fork U64 proofs are retained byte-for-byte
under `proofs/u64/`, with source provenance and import enforcement.

| Law(s) | Actual function / specification | Preconditions and trust | Status / related tests |
| --- | --- | --- | --- |
| `step_in_mask` | `Subsets.next`: applying its mask again changes nothing. | Any two U64 values; no chess-only or empty-domain condition. Pinned checker/Base. | **Law proved**; missing-mask mutation rejected; all native states checked separately. |
| `sequence_in_mask` | `Laws.at`: Nat recurrence repeatedly calls `Subsets.next` from zero. | Every Nat index and U64 mask; no machine-counter bound. Pinned checker/Base. | **Law proved**; connecting recurrence steps to affine table writes remains open. |
| `sequence_extract_deposit` | `U64.pdep(U64.pext(Laws.at(n,m),m),m)` recovers that state. | Every Nat index/mask; reuses inherited public U64 roundtrip, not an assumed new law. | **Law proved**; does not establish successive compact indices or coverage. |
| `empty_mask_sequence` | `Laws.at(n, U64.zero()) == U64.zero()`. | Every Nat index. Pinned checker/Base. | **Law proved**; empty-mask native case. |
| `cross_low_half`, `cross_bit63`, `mask_cycle_end`, `full_word_wrap` | Four closed equalities of actual `Subsets.next` at cross-half/high-bit/wrap boundaries. | Concrete inputs only; no universal period/order claim. | **Law proved (closed)**; reversed/truncated/stuck-zero implementation mutations fail intended law. |
| `index/capacity_matches_popcount`, `index/extraction_bound`, `index/sequence_index_bound` | Actual popcount, PEXT and imported production-step recurrence; unbounded Nat capacity. | Every U64 mask/input, and every Nat recurrence index; pinned checker/Base. | **Laws proved**; compact range, including population 64; not ordinal equality. |
| `index/compact_projection`, `index/compact_roundtrip` | Actual PDEP then PEXT; low-k-bit projection or exact bounded inverse. | Projection unconditional; inverse requires strict `value(x) < 2^popcount(mask)`. | **Laws proved**; inclusive/unrestricted variants rejected; four native modes test valid and invalid-domain values. |
| `index/deposit_in_mask`, `index/deposit_injective`, `index/masked_extract_injective`, `index/compact_coverage` | Bijection between bounded compact U64 words and mask-contained bitboards. | Injectivity/coverage use the stated bounds/membership; no enumerator-order assumption. | **Laws proved**; constructive witness is actual PDEP, not a proof of sequence coverage. |
| `successor/subtraction_refinement` | Actual split-U32 `U64.sub` equals full-width `Word.sub`. | All U64 operands, including low-half borrow and equal-low-half cases. | **Law proved**; three corrupt-Base controls specifically fail the new bridge; native borrow/non-borrow checks. |
| `successor/step_successor` | PEXT value after actual `Subsets.next` is piecewise compact successor with wrap. | State is mask-contained; compact value bound/capacity equality are already proved. | **Law proved**; unmasked-state counterexample retained; actual step native checks. |
| `successor/sequence_ordinal`, `successor/sequence_nonduplicating`, `successor/sequence_coverage` | Imported `at(i,mask)` reaches each masked state exactly once in its first mathematical capacity states, in PEXT-index order. | Nat indices strictly below mathematical capacity; coverage requires mask membership. | **Laws proved**; stuck actual recurrence and false inclusive bounds rejected; no enumeration-count assumption. |
| `successor/cycle_endpoint`, `successor/sequence_periodic` | Imported recurrence returns to zero at capacity and repeats after capacity for every Nat index. | All U64 masks, including empty/full; no machine-power overflow. | **Laws proved**; derived from successor/ordinal and numeric-observation injectivity. |
| `layout/mask_population`, `layout/tight_population_bound` | Actual Tables.slider mask and independent file/rank count. | Every U32 key with Nat value below 128; exhaustive source case split. | **Laws proved; hosted recheck passed**; rook <=12, bishop <=9; not complete geometric mask equality. |
| `layout/shift_in_range`, `layout/size_positive`, `layout/tight_size_bound`, `layout/size_matches_capacity` | Exact inline U32 size expression observed through Spec; mathematical capacity from prior laws. | Same 128-key domain; source-link guard, unchanged checker/Base. | **Laws proved; hosted recheck passed**; nonzero exact size <=4096/512, shift <32; offsets/Array writes remain separate. |
| `layout/full_index_bound`, `layout/lookup_index_bound` | Full PEXT and actual Sliders.pext_index. | Every valid key and arbitrary U64 occupancy. | **Laws proved; hosted recheck passed**; index below block size; the address increment below supplies exact low-projection equality, while buffer refinement remains explicit work. |
| `address/low_projection_exact`, `address/compact_index_exact` | Actual low32 projection and Sliders.pext_index equal the full numeric value. | Generic mathematical width k<=32 and the stated value bound; compact theorem derives the bound from extraction. | **Laws proved locally**; empty/32-bit domains included; wider-mask counterexamples retained. |
| `address/chess_lookup_exact`, `address/chess_lookup_word` | Actual Sliders.pext_index equals full PEXT as Nat and after actual U32-to-U64 extension. | Every valid key (<128), arbitrary occupancy; earlier population proof imported, not assumed. | **Laws proved locally**; in-range constant-zero and high-half mutations fail the new helper. |
| `address/chess_sequence_ordinal`, `address/chess_sequence_coverage` | Actual U32 lookup is the exact imported recurrence ordinal; indexing with it recovers a masked state. | Ordinal index strictly below capacity; recovery requires mask membership. | **Laws proved locally**; actual recurrence unchanged; affine table writes/reads remain separate. |
| `address/chess_lookup_injective`, `address/chess_lookup_redeposit` | Actual indices distinguish masked states and recover occupancy projection through PDEP. | Injectivity requires membership; redeposit applies to arbitrary occupancies at valid keys. | **Laws proved locally**; native scalar/reference checks are separate bounded evidence. |
| 16 existing U64 obligations | Public U64 operations, full-width representation and Word lemmas from pinned fork. | Original assumptions unchanged; source checker/Base remain trusted. | **Inherited laws reused**, not 16 new engine laws; original files fingerprinted and all imports checked. |

Both source gates require successful process plus exactly `All terms check.`. The
index aggregate invokes the unchanged initial eight-law/eleven-control gate, then
its nine-law/fifteen-control gate and importing consumer. Controls include a
dependency that emits an unsafe warning despite raw exit zero.
The fork's own negative suite additionally retains the cyclic-template regression.
No proof holes, unsafe dependencies, foreign witnesses or new axioms are accepted.

## Open proof obligations (P1–P7)

| Target | Current evidence / partial progress | Still required; do not relabel as proved |
| --- | --- | --- |
| P1: subset enumeration / slider indices | General ordinal/coverage/period, chess-mask/size bounds, and now exact actual U32/full-PEXT and ordinal/coverage correspondence are source-proved. The address increment is locally qualified with all 56 laws. | Actual prefix-offset arithmetic and its no-overflow/region bounds remain; connect exact indices and the recurrence to affine writes/reads. Scalar correspondence is not total Array-region safety. |
| P2: table/lookup refinement | Native `Tables.build` matches all 108,160 C-reference logical entries; independent geometric rays check slider values. | Independent square/rank/file/ray specification, step boundaries, blockers, mask correctness, initialized regions, disjoint offsets/no overflow, affine writes and actual `Chess.bend` lookup refinement. |
| P3: board / moves / perft | Existing legality/special-move/perft reference tests; no new perft budget or depth. | Named orthodox-chess specification and FIDE edition, board invariants, special moves and legal-move soundness/completeness/no duplicates; legal-tree recurrence separate from draw pruning and machine-counter overflow. |
| P4: history / rules / parser / UCI | Native reconstructed paths and draw/parser/protocol checks, including transactional rejection. | Quantified history reconstruction, EP/repetition/windows/clocks/mate precedence, sufficient material-case soundness, claim witnesses, parsing bounds/replay and pure controller safety. Responsiveness also needs scheduling/progress assumptions. |
| P5: input / policy | Native complete input and paired policy tests for supported corrected root layouts. | Logical bounds/initialization, perspective/temporal repetition, feature formulas, valid-domain forward/reverse mappings, sentinel exclusion and one-Game binding with affine observations. No unrestricted Full4672/compact1858 bijection. |
| P6: search / replies / scheduling | Diagnostic PUCT and synchronous neural leaf tests; finite/shape/move-alignment rejection. | Arena/parent/ticket/reset safety, no partial mutation, exactly-once completion, sign/terminal conventions; batching/cancellation ownership and progress assumptions. F32 is not exact real arithmetic; no exact IEEE-softmax-sum theorem. |
| P7: model / training / native trust | Transitional native CPU model fixture only. | Bend tensor/model/gradient/update/RNG/serialization/resume/data-provenance contracts as components migrate; justified numerical model and source-to-native refinement/translation validation. |

The next decisive source acceptance is the **P2 storage/lookup connection**:
connect the now-proved exact scalar indices and ordinal law to actual prefix
offsets, initialized/disjoint affine writes and reads, and independent blocker-ray
geometry. Per-mask size safety and exact U32/full-PEXT correspondence are proved;
do not relabel them as the complete Array or lookup theorem. Trained-checkpoint
and target-CUDA qualification remain separate inference targets.


The successor aggregate adds seven laws to the previous 33, checking **40 accepted
laws** in total and retaining 26 prior negative controls plus 16 new controls. The
compiler fork's independent source gate retains its cyclic-template regression.

## Trust and retained dependencies

The checker/normalizer and Base implementation are trusted. Source proofs do not
verify C lowering, affine-array lifetime, handwritten effects/ABI, the C/C++ compiler,
model package, LibTorch/GPU libraries, operating system or hardware. Continue native
differential tests, sanitizer/instruction checks and actual target qualification.
Native tests do not turn those components into an end-to-end theorem.

The material product retains its one-static-executable runtime test. The neural
product legitimately needs its executable, native loader/libraries, bound package,
CPU description and scratch. Python is outside the tested neural runtime but remains
in export, external references and the unported production/training pipeline. There
is no claim of a bug-free engine, complete migration, strength or speedup.


## PR #805 qualification record

The initial subset increment has renewed hosted source/native, whole-repository lint,
full inherited U64-regression, and static material normal/empty-runtime evidence.
See the dated readout for exact source and report identities. The first hosted
workflow failed only its overstrict comparison of measured stop times; original
reports and the corrected behavioral comparison are preserved. No trained-model,
CUDA, neural-executable rebuild, new formal ordering theorem or performance
improvement follows from these material-engine tests.


## Historical compact-index increment after PR #805

[Compact-index bijection](experiments/2026-09-21-bend-compact-index-bijection.md)
adds nine universal contracts under `standalone/proofs/index/`. It leaves all
production, prior accepted laws, the previous gate and the compiler pin unchanged.
The new aggregate gate retains all 33 source laws (nine new, eight prior engine,
16 inherited U64) and runs the eleven parent plus fifteen new negative controls.

P1 now has **proved exact extraction bounds and the bounded reverse PEXT/PDEP
identity**, an unbounded low-k-bit projection, masked/bounded injectivity and PDEP
coverage of bounded compact words. Capacity is proved equal to mathematical
`2^toNat(popcount(mask))`; 64-bit masks do not overflow that specification.
`sequence_index_bound` applies to the imported actual-step recurrence, but does
not establish its ordinal.

The **full P1 ordering/coverage theorem remains open**, including the actual
U64 two-limb subtraction/borrow-to-compact-successor bridge. A bijection between
representations is not proof that a particular enumerator visits each one.
P2 affine table-write/offset/lookup and geometry obligations are unchanged.
No application responsibility has newly moved into Bend in this proof increment.
The dated readout distinguishes source proofs, bounded native tests, historical
engine/model results and remaining trust/application dependencies.

Hosted compact-index qualification run **35643399130** passed the additive source gate, all four native modes, original compiler source/pin checks and whole-repository lint on the exact candidate. Compact reports/source identities are committed in the dated record. This does not add runtime-engine, model, GPU or enumeration-order evidence.

## Successor / ordinal qualification

The dated successor readout records the newly executed proof/native gates and exact
source identities. General enumeration order, coverage, nonduplication and period
are no longer listed as open; no table-buffer, whole-engine, model or performance
result is implied by that update. Earlier dated evidence below/above remains historical.

Hosted successor/ordinal qualification run **35655691011** passed the additive source gate, four native modes, original source/pin checks, and whole-repository lint on the exact candidate. Reports and source identities are committed in the dated record. Full P1 recurrence order/coverage/period is source-proved; actual affine table initialization, offsets and lookup geometry still require P2 refinement. No runtime-engine/model/GPU/performance requalification is implied.

## Historical local slider-mask qualification

The layout gate checks 48 accepted laws and retains the unchanged prior gate.
Its dated record distinguishes all 13 new rejection controls from 42 inherited
controls, source proofs from four-mode native tests, and the observed diagnostic
stack overflow from a proper proof rejection. Whole-repository lint is unavailable
in this verification environment, and no new hosted CI or publication occurred.
The patch preserves the newer CI/test repairs in the refreshed PR #807 head.


## Slider-mask hosted publication

Run **35681951103** qualifies the unchanged saved eight-law layout increment on #807 `ef3ded0b652a09918abb1d79467b1ac56a8e1ce5`: 48 total laws, 55 controls, four native modes, inherited source/pin checks and whole-repository lint pass. This resolves the local-only publication/lint gap. The dated slider-mask readout and committed hosted reports retain exact identities, failures and limits. No additional migration, offset/affine-table theorem, model/GPU result or benchmark is implied.

Initial clean qualification commit: `9cbb49aba36af12c133bea430920ce454dff4d69`, branch `feat/bend-slider-mask-bounds-20260921`. The follow-up matrix clarification changes documentation only; all proof, native-test and production source identities remain those checked in run 35681951103.


## Exact U32 indices: local continuation after PR #819

The additive `standalone/proofs/address/` suite proves eight new universal contracts
about actual low32 projection, Sliders.pext_index, U32 zero extension and imported
recurrence order/recovery. The aggregate retains all 48 prior laws and 55 controls:
**56 accepted laws and 72 controls pass locally**. Four bounded native modes pass
2,070 rows each, including explicit >32-bit truncation counterexamples outside the
exactness domain. Original compiler source and pin contracts pass separately.

Whole-repository lint could not run because its three required tools are absent;
hosted qualification and publication are still outstanding. There is no new hosted
candidate pass or remote PR in this continuation.
The baseline source tree and unsigned commit object both hash exactly to published
PR #819. All runtime code, older proofs, compiler inputs and CI repairs are preserved.
No additional Python migration, prefix/affine-table proof, GPU/model execution,
performance result, merge or deployment is implied.
