# Bend migration and proof inventory

Status snapshot: September 21, 2026. This inventory is specific to the standalone
stack through PR #805 (`d6503d8d6a0c2b8fa938477b755e8303c0af49f1`) plus the
[compact-index increment](experiments/2026-09-21-bend-compact-index-bijection.md).
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

No new Python responsibility moves into Bend in the subset increment: its arithmetic
was already Bend-owned. The concrete change is attaching checked properties to the
actual production helper, with independent native checks for its table integration.

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
| P1: subset enumeration / slider indices | Eight initial laws plus nine compact-index laws: exact extraction bounds, bounded inverse and representation bijection. Prior native exhaustive chess-mask enumeration remains separate. | For `k = popcount(mask)` and every `i < 2^k`, prove `toNat(pext(at(i,mask),mask)) = i`; derive sequence completeness/nonduplication. Bridge actual U64 subtraction/carry/borrow to compact successor. Prove chess-mask population bounds for actual U32 size/offset operations. Mathematical capacity, including k=64, is now proved separately. |
| P2: table/lookup refinement | Native `Tables.build` matches all 108,160 C-reference logical entries; independent geometric rays check slider values. | Independent square/rank/file/ray specification, step boundaries, blockers, mask correctness, initialized regions, disjoint offsets/no overflow, affine writes and actual `Chess.bend` lookup refinement. |
| P3: board / moves / perft | Existing legality/special-move/perft reference tests; no new perft budget or depth. | Named orthodox-chess specification and FIDE edition, board invariants, special moves and legal-move soundness/completeness/no duplicates; legal-tree recurrence separate from draw pruning and machine-counter overflow. |
| P4: history / rules / parser / UCI | Native reconstructed paths and draw/parser/protocol checks, including transactional rejection. | Quantified history reconstruction, EP/repetition/windows/clocks/mate precedence, sufficient material-case soundness, claim witnesses, parsing bounds/replay and pure controller safety. Responsiveness also needs scheduling/progress assumptions. |
| P5: input / policy | Native complete input and paired policy tests for supported corrected root layouts. | Logical bounds/initialization, perspective/temporal repetition, feature formulas, valid-domain forward/reverse mappings, sentinel exclusion and one-Game binding with affine observations. No unrestricted Full4672/compact1858 bijection. |
| P6: search / replies / scheduling | Diagnostic PUCT and synchronous neural leaf tests; finite/shape/move-alignment rejection. | Arena/parent/ticket/reset safety, no partial mutation, exactly-once completion, sign/terminal conventions; batching/cancellation ownership and progress assumptions. F32 is not exact real arithmetic; no exact IEEE-softmax-sum theorem. |
| P7: model / training / native trust | Transitional native CPU model fixture only. | Bend tensor/model/gradient/update/RNG/serialization/resume/data-provenance contracts as components migrate; justified numerical model and source-to-native refinement/translation validation. |

The next decisive source acceptance test is P1's **universal compact-index/order
theorem**, not another successful bounded occupancy test. For inference, the next
separate model acceptance is a representative trained-checkpoint run and then the
specified target-CUDA qualification; neither is established by the CPU fixture.

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


## Compact-index representation increment after PR #805

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
