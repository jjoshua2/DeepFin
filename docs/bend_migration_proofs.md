# Bend migration and proof inventory

Status snapshot: September 21, 2026. This inventory covers PR #819 at
`40803892f3204a54a33a87e2e7b25fa48fed2e1a` plus the [exact U32 slider-index increment](experiments/2026-09-21-bend-exact-slider-index.md),
qualified in run 35684436097. Earlier dated records remain historical.
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
| `layout/full_index_bound`, `layout/lookup_index_bound` | Full PEXT and actual Sliders.pext_index. | Every valid key and arbitrary U64 occupancy. | **Laws proved; hosted recheck passed**; index below block size; low-projection equality and actual buffer refinement remain explicit work. |
| `exact_index/low_projection_exact`, `exact_index/lookup_index_exact` | Actual low U32 observation and Sliders.pext_index equal the full compact Nat value. | Generic symbolic width <=32 with strict bound; valid chess keys and arbitrary occupancies for actual masks. Prior population certificates are proved, not assumed. | **Laws proved**; final hosted aggregate and independent boundary mutations pass. |
| `exact_index/lookup_sequence_ordinal`, `exact_index/lookup_state_recovery` | Actual U32 lookup index identifies the exact state of the imported production-step recurrence. | Nat indices below actual block size; valid key and arbitrary occupancy. Recovery yields masked occupancy, not complete board. | **Laws proved**; native probe exhausts all relevant chess-mask states; actual Array writes remain separate. |
| `exact_index/lookup_collision_sound`, `exact_index/lookup_collision_complete` | Actual indices agree exactly when relevant occupancies agree for chess masks. | Soundness uses valid chess keys. Equal masked occupancies imply equal indices for any U64 mask, even wider than 32 bits. | **Laws proved**; >32-bit-mask counterexample prevents unrestricted inverse claim. |
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
| P1: subset enumeration / slider indices | General ordinal/coverage/period, tight actual chess-mask populations, safe block sizes, and now exact U32/full-PEXT identity, U32 ordinal, masked-occupancy recovery and both collision directions are source-proved. | Actual prefix-offset arithmetic and its connection to initialized/disjoint affine writes and reads remain. Exact scalar correspondence is not a table-buffer theorem. |
| P2: table/lookup refinement | Native `Tables.build` matches all 108,160 C-reference logical entries; independent geometric rays check slider values. | Independent square/rank/file/ray specification, step boundaries, blockers, mask correctness, initialized regions, disjoint offsets/no overflow, affine writes and actual `Chess.bend` lookup refinement. |
| P3: board / moves / perft | Existing legality/special-move/perft reference tests; no new perft budget or depth. | Named orthodox-chess specification and FIDE edition, board invariants, special moves and legal-move soundness/completeness/no duplicates; legal-tree recurrence separate from draw pruning and machine-counter overflow. |
| P4: history / rules / parser / UCI | Native reconstructed paths and draw/parser/protocol checks, including transactional rejection. | Quantified history reconstruction, EP/repetition/windows/clocks/mate precedence, sufficient material-case soundness, claim witnesses, parsing bounds/replay and pure controller safety. Responsiveness also needs scheduling/progress assumptions. |
| P5: input / policy | Native complete input and paired policy tests for supported corrected root layouts. | Logical bounds/initialization, perspective/temporal repetition, feature formulas, valid-domain forward/reverse mappings, sentinel exclusion and one-Game binding with affine observations. No unrestricted Full4672/compact1858 bijection. |
| P6: search / replies / scheduling | Diagnostic PUCT and synchronous neural leaf tests; finite/shape/move-alignment rejection. | Arena/parent/ticket/reset safety, no partial mutation, exactly-once completion, sign/terminal conventions; batching/cancellation ownership and progress assumptions. F32 is not exact real arithmetic; no exact IEEE-softmax-sum theorem. |
| P7: model / training / native trust | Transitional native CPU model fixture only. | Bend tensor/model/gradient/update/RNG/serialization/resume/data-provenance contracts as components migrate; justified numerical model and source-to-native refinement/translation validation. |

The next decisive source acceptance is the **P2 storage/lookup connection**:
connect exact scalar indices and the ordinal law to prefix offsets, initialized
and disjoint affine writes/reads, and independent blocker-ray geometry. Per-mask
population and size bounds are now proved; do not relabel them as the complete
Array or lookup theorem. Trained-checkpoint and target-CUDA qualification remain
separate inference targets.


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

## Exact U32 slider-index qualification

Run **35684436097** passes 54 accepted laws (six new plus 48 inherited) and 70 controls (15 new plus 55 inherited), four native modes, original source/pin checks and whole-repository lint. Exact candidate `f6fd6e349eea48af36a8fa4e14c2d40a0049f329` changes no production source. The dated record and committed reports distinguish the corrected probe hash from unchanged proof inputs. Prefix-offset/affine-storage and independent blocker-ray refinement remain the next P2 work; no new application migration or model/performance result follows.
