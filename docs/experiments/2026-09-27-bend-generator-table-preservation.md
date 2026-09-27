# Whole-table preservation through actual move generation

## Scope and acceptance

Exact parent is PR #904 at1915e4f283b1e3d5daf7e26683d5a6c2ef48a121. This bounded,
opt-in increment targets the previously missing table-preservation link through
ordinary generation, both castling sides and the optimized final filter. Acceptance
requires the complete public source consumer, intended corruption controls, four
native modes, pinned compiler checks and configured repository lint. No perft,
training,model/GPU,benchmark or production operation is added.

## New source result

Five related public contracts prove that actual attack queries, arbitrary ordinary
scans, each castling-side call, prepared filtering and complete Chess.legal_moves
return exactly the original source-array value. They require no initialization,
shape,correct-content,Board-consistency,single-king or valid-side assumption.
The real functions and all callback/loop bodies are connected using existing
storage.Read, rather than assuming query preservation or invoking a shadow generator.

This generic result can now carry any previously established initialized contents
through the whole generator. It does not itself prove a new geometric move-decision
or accepted-castling theorem. Reverse/forward attack equivalence,history-valid rights,
whole-generator legality and completeness remain distinct obligations.

The proof uses the actual affine reification certificate solely inside proofs.
Native code imports no proof model. Raw U32 dispatch covers153 disjoint Boolean
clauses over2^32 values rather than narrowing the public kind argument. Ragged arrays
are in the source domain; complete bounded arrays are the native domain.

## Local executions before publication

The complete five-law consumer and all17 controls passed:8 semantic/refinement,
8 manifest/import-policy and1 synthetic warning-output unit. The full native verifier
passed generic,portable,native-target andUBSan with173 distinct tuples per mode,
2,313,088 whole-cell comparisons (4,626,176 U32 limbs),including17 complete131072-cell
buffers and156 smaller buffers. Seven malformed inputs per mode were rejected.

Actual scanning,unchecked-filter and queen-combination mutations overwrite unused
slot131071. All eight initialized full-generator move outputs remain unchanged,
while the full-buffer comparisons reject the altered state. These three mutations
compile and execute under generic flags. They are deliberate regressions,not
production defects. Raw nonuniform initialization is independently checked; initialized
contents are compared with the pre-operation snapshot,not separately re-proved by a
new native mask oracle. No move-set correctness claim follows from preserved buffers.

Original16-law/seven-control compiler suite and12 pin tests passed locally. Local
configured lint failed because Ruff,Basedpyright,Vulture were absent; its original log
is retained. One native invocation was interrupted after two clean modes; a serial
rerun completed all four modes and mutations. Earlier proof-projection/ownership/raw
scalar-elimination and probe callback/parser/size destructuring failures are retained
as development failures,not qualifying negative controls. Final proof statements and
all older source files were not weakened to resolve them.

## Publication status

Local checks complete; fresh hosted execution will be recorded separately before
claiming final source/native/lint qualification. Parent modular193/623 plus this5/17
would be198/640; the complete aggregate wrapper is not being run. These five are
related public contracts and their composition,not five independent discoveries.

Self-review only; a separate reviewer was not available. Source checker/Base,native
lowering/storage,ABI,toolchain,OS and hardware remain trust boundaries. Whole-cell
observations do not establish native allocation/lifetime or pointer identity. No
production function,previous accepted proof,compiler input,permanent workflow,Python
application responsibility,model/GPU,training,search or perft budget changed.
