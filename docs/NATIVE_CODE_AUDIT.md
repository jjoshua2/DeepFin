# Native C Audit

Fresh line-by-line review of the repository's complete native-code surface.
This audit is intentionally independent of the earlier component-level review
recorded in `docs/REVIEW_BUG_HUNT.md`.

Started: 2026-07-09

Base: `origin/main` at `af6c0ae` (post-#138/#139). Audit started against `64b44b6`.

## Scope

The repository currently has no C++ translation units.

⚑ THIS TABLE IS THE 2026-07-09 AUDIT'S SCOPE, NOT A CURRENT INVENTORY. It was
written before the `nnue/` subsystem existed and has not tracked it: measured
2026-08-26, `git ls-files '*.c' '*.h'` totals **18,043** lines, of which
`chess_anti_engine/nnue/` alone is 5,654 in files this table does not list. The
"9,627 lines" this line used to claim was true when written and is now off by
nearly half. Read the rows below as "what that audit covered", and re-measure
before quoting a total. The slider row was added when the header landed, so that
one unit at least does not start out stale.

| Review unit | Files | Lines | Status |
|---|---|---:|---|
| Board core and binding | `encoding/_cboard_impl.h`, `encoding/_lc0_ext.c` | 2,455 | finding |
| Feature/relation core and binding | `encoding/_features_impl.h`, `encoding/_features_ext.c` | 1,546 | finding |
| Shared bitboard-plane encoder | `encoding/_bitboard_planes_impl.h` | 118 | finding |
| Table-backed sliding attacks | `encoding/_slider_attacks_impl.h` | 480 | pending |
| MCTS extension | `mcts/_mcts_tree.c` | 5,384 | finding |
| CBoard fuzz harness | `scripts/fuzz/cboard_libfuzzer.c` | 132 | deep-reviewed |
| Cross-extension contracts | all of the above | - | finding |

Status meanings: `pending`, `contract-mapped`, `deep-reviewed`,
`verified`, or `finding`.

## Review Rules

- Read public callers and tests before judging an implementation contract.
- Review every file for correctness, memory safety, ownership, concurrency,
  error cleanup, portability, performance, and avoidable duplication.
- Record the violated invariant, exact evidence, production effect, and smallest
  verification for every finding.
- Treat static-analyzer output as a lead, not as a finding until confirmed.
- Add a regression test or deterministic reproducer before fixing a serious
  issue whenever practical.
- Keep fixes separated by root cause; do not mix cosmetic cleanup into them.

## Baseline

| Check | Result | Notes |
|---|---|---|
| Strict extension build (`CAE_EXT_WERROR=1`) | passed | All three extensions rebuilt with warnings as errors. |
| Native focused tests | passed | 259 passed, 2 skipped. |
| Cppcheck | reviewed | Missing-return and negative-shift reports were false positives; boundary leads were confirmed independently with ASAN. |
| ASAN/UBSAN CBoard differential fuzz | passed | 500 lockstep games, seed `0xdeeff15`. |
| ASAN/UBSAN batch-encode differential fuzz | passed | 120 games in both repetition modes, seed `0xba7c4`. |
| libFuzzer CBoard run | passed | Clang ASAN/UBSAN coverage-guided run completed 264,174 inputs in 121 seconds (1,446 coverage edges, 9,078 features) with no sanitizer finding. |

## Review Checklist

### Board core and Python binding

- [x] Initialization and static lookup tables
- [x] Attack generation and policy mapping
- [x] Legal move generation and king-safety filtering
- [x] Board construction and validation
- [x] Move application, castling, promotion, en passant, and clocks
- [x] Hash/history/repetition state
- [x] Terminal queries and FEN/result serialization
- [x] Plane encoders and orientation
- [x] Python reference ownership, parsing, allocation, and errors

### Feature and relation core

- [x] Lookup initialization and attack generation
- [x] Pins, x-rays, pawn structure, SEE, and threat accumulation
- [x] Plane offsets, widths, initialization, and orientation
- [x] Relation tensor construction
- [x] Python/NumPy validation and ownership
- [x] Standalone/CBoard/batch-path parity

### MCTS extension

- [x] Tree allocation, growth, hash table, reset, and teardown
- [x] Selection, expansion, backup, virtual loss, and solved propagation
- [x] Stored batch state and capacity accounting
- [x] Gumbel sequential-halving state machine
- [x] Batch board processing and input encoding
- [x] Python/NumPy buffer validation, ownership, and error cleanup
- [x] Locks, OpenMP, GIL transitions, and concurrent mutation
- [x] Module API, ABI marker, and duplicated `PyCBoard` layout

### Fuzz and cross-extension contracts

- [x] Fuzz input construction reaches valid and invalid state boundaries
- [x] Fuzz queries cover all board-core safety-sensitive operations
- [x] Static globals duplicated across shared objects stay synchronized
- [x] Shared headers behave consistently in all three translation units
- [x] ABI/layout assumptions are detected rather than silently drifting

## Findings

| ID | Severity | Category | Component | Summary | Evidence | Verification | Status |
|---|---|---|---|---|---|---|---|
| N001 | High | Memory Safety / Reliability | Native Python APIs | Public extension entry points trust array sizes, node IDs, path lengths, legal counts/actions, and in one case a type name substring before raw struct casts. Malformed inputs can read/write outside buffers or segfault the interpreter. | `_lc0_ext.c:120-139` reads `n_steps*12` bitboards without dtype/size checks; `_features_ext.c:48-55` blindly reads six entries; `_mcts_tree.c:1843-1868` indexes an unchecked `node_id`; `walker_integrate_leaf` accepts an empty path and unchecked legal indices; `batch_integrate_leaves` trusts every fixed-stride buffer and can overrun `priors_stack[256]`; `batch_process_ply` indexes `POLICY_LUT` with an unchecked action. | ASAN: one-element `encode_piece_planes(..., n_steps=8)` reports heap-buffer-overflow at `_lc0_ext.c:134`; one-element `compute_extra_features` reports heap-buffer-overflow at `_features_ext.c:54`; `tree.expand(999999, ...)` segfaults at `_mcts_tree.c:450`; UBSAN reports `POLICY_LUT[-1]` from `batch_process_ply`. | fixed in #140 |
| N002 | High | Search Correctness / Concurrency | MCTS backprop | Concurrent GIL-released backprop atomically increments `N` but performs non-atomic `W += value`. Writers lose updates and all concurrent reads/writes are a C data race. Multi-GPU PUCV actively calls this path from several worker threads. | `tree_backprop` uses `__atomic_fetch_add` for `N` and plain `t->W[nid] += v`; `batch_integrate_leaves` releases the GIL; `MultiGpuPucvPool` invokes it from one thread per GPU. | Eight threads doing 409,600 identical `Q≈1` updates produced root Q `0.998642578125`, implying about 556 lost value updates while visits advanced. | fixed in #138 (atomic W writes); #140 added atomic N/W loads on concurrent select/score paths |
| N003 | Medium | Chess Correctness | CBoard move generation | Castling generation trusts castling-right flags without requiring the king on its home square and the corresponding rook on its corner. An inconsistent but accepted FEN can produce an illegal castle and `cboard_push` then synthesizes a rook. | `_cboard_impl.h:574-628` checks rights, empty transit squares, and attacks, but not king/rook presence. `from_board` copies raw `castling_rights`, while python-chess legal generation uses cleaned rights. | FEN `4k3/8/8/8/8/8/8/4K3 w K - 0 1` has no python-chess castle but CBoard includes `e1g1`. | fixed in #140 |
| N004 | Medium | API Correctness | Writable NumPy buffers | Writable-output helpers use coercing `PyArray_FROMANY`; non-contiguous or wrong-dtype outputs are copied, written internally, then discarded instead of updating the caller or rejecting the input. | `FROMANY_*_RW` and `parse_relations_buffer` request contiguity/writeability but do not require no-copy identity or writeback semantics. Affects batch encoders, relations, classification `done`, temperature actions, and PUCT output buffers. | Passing a strided `(1,146,8,8)` view to `batch_encode_146` returns successfully while every caller-visible element remains `-7`. | fixed in #140 |
| N005 | Low | Reference Ownership | CBoard construction | Every `CBoard.from_board` leaks one reference each to the `True` and `False` singletons by passing new `PyBool_FromLong` references directly into `PyObject_GetItem`. | `_lc0_ext.c:306-307`; `PyObject_GetItem` does not steal key references. | 10,000 constructions increase both singleton refcounts by exactly 10,000. No heap growth because the objects are singletons, but the ownership contract is wrong. | fixed in #140 |
| N006 | Medium | Tablebase Correctness | Gumbel pending leaves | `get_pending_tb_leaves` silently scans only the first 4,096 leaves even though `target_batch` and the encoding-buffer capacity are not capped to 4,096. Later eligible leaves miss Syzygy overrides. | `_mcts_tree.c:3940-3944` sets `n_scan=min(n,4096)` based on a comment that `n_leaves <= GSS_GPU_BATCH`; `target_batch` can override that default. | Static proof from the public API/state machine; requires a >4,096 pending-batch regression fixture. | fixed in #140 |
| N007 | Medium | Search Correctness | Forced-chain collapse | At the maximum path length, forced collapse pushes the board before checking path capacity, then returns with the tree leaf/path unchanged. The caller can evaluate a board one ply beyond the recorded leaf. | `_mcts_tree.c:780-784` calls `cboard_push_index` before `if (*path_len >= MCTS_MAX_PATH) return 0`. | Static control-flow proof; reachable only at the 512-node path cap. | fixed in #140 |
| N008 | Medium | Compatibility | Extension build | `_mcts_tree` always compiles with `-march=native`, although `setup.py` documents `CAE_EXT_NATIVE=1` as the opt-in for non-portable builds and the project can distribute worker wheels. A wheel can contain instructions unsupported by another worker CPU. | `setup.py:_mcts_compile_args` appends `-march=native` unconditionally; `_ext_compile_args` gates the same flag on `CAE_EXT_NATIVE`. | Static build-command evidence from the strict baseline. | fixed in #140 (`CAE_EXT_NATIVE` gate) |
| N009 | Medium | Build Correctness | Native feature extension | `CAE_EXT_NATIVE=1` enables the AVX2 feature-plane path in standalone `_features_ext`, but its `bitboard_to_plane_*` helpers existed only in `_cboard_impl.h`, which that translation unit does not include. The shared object linked with unresolved symbols and failed at import. | `_features_impl.h:feat_bb_to_plane` calls the helpers under `__AVX2__`; only `_cboard_impl.h` defined them. | Strict native build followed by import failed with `undefined symbol: bitboard_to_plane_black`. | fixed by extracting the converter into `_bitboard_planes_impl.h`, shared by both headers |

## Non-Finding Observations

Record reviewed concerns that were disproved or judged non-actionable here so
future audits do not repeatedly reopen the same question without new evidence.

- Cppcheck's negative-shift warning for `IS_LEGAL_MOVE(..., -1, ...)` is a
  false positive: `sq_bit(capture_sq)` is guarded by `capture_sq >= 0` in the
  macro's executed control flow.
- Cppcheck's missing-return reports on the history toggle and tree reset are
  macro-analysis false positives (`Py_RETURN_NONE` returns).
- Omitting en-passant from the repetition hash differs from python-chess's
  formal transposition key, but a legal game cannot revisit the same pawn
  placement after the one-time double push that creates the EP right. No
  reachable false repetition was found.
- The v3 feature families are implemented in the shared feature core but not
  accepted by live CBoard/MCTS encoders. The experiment configs explicitly
  declare them offline-sidecar-only, so this is a known gate rather than a new
  defect.
- The direct feature algorithms (pins, x-rays, pawn structure, SEE, threats,
  relations) retain exact Python/C parity in the focused tests; no independent
  semantic divergence was found in this pass.
- `CBoard` is roughly 1.6 KiB, so the header's historical “memcpy ~72 bytes”
  comment is stale. The large history/hash state and per-node cache are already
  accounted for by `memory_bytes()`; compacting them is a potential performance
  project, not a safe audit-time patch.

## PRs

| Finding | Branch/PR | Validation | Status |
|---|---|---|---|
| N002 (W write) | #138 | `tests/test_mcts_thread_safety.py::test_concurrent_backprop_w_sum_is_exact` | merged |
| N001–N008 (remaining + N002 reads) | #140 (`fix/native-code-audit`) | native validation suite + castling parity + thread-safety | merged |
| N009 | `codex/native-build-fix` | strict `CAE_EXT_NATIVE=1` build + native import + encoding parity | open |
