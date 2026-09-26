# BT4 root-output reuse foundation

The [shared BT4 legal-policy conversion](2026-09-23-bt4-output-adapter-foundation.md)
now has an in-memory evaluator seam for a future neural generator. For one
played position, `BT4OnnxEvaluator.evaluate_root` checks the board's actual
history against the configured input encoding, makes one named ONNX
policy-plus-WDL call, and returns a `BT4RootOutput`. That record keeps the
collector-compatible float32 T=1 legal policy and the named WDL output in its
native dtype and side-to-move order. It carries the input and source keys,
head names, model hash assertion, and input modes. A root-policy-sampling actor
can consume this observation without binding a search tree or evaluating leaves.
No production sampling temperature has been selected here.

Search is optional and separately priced. If a caller binds the exact
`MCTSTree` it passes to `run_gumbel_root_many_c`, the adapter uses the root's
already-computed search logits. For each real pending leaf, it verifies the C
board and encoded history, checks the C and Python legal-index sets, then
uses the same Leela-to-compact mapping before search expands to its 4672-slot
space. The ONNX call excludes bucket-padding rows. Root, leaf-call, and real
leaf-row counts remain separate. The retained teacher prior is distinct from
search-improved policy or move choice; target sharpening and calibration remain
later recipe decisions.

The supplied ONNX session and model SHA-256 are caller assertions. This slice
does not hash a model file, attest its provider, write a sidecar, or establish
complete publishable provenance. A future writer must enforce the raw BT4
collector's full output, precision, history, and source contract. Game outcome
handling, including six-man Syzygy adjudication and unresolved-game exclusion,
is not wired into this adapter. No generator command, corpus schema, GPU
inference, quality trial, or throughput measurement is part of this change.

The batched-root extension adds `evaluate_roots(boards, x_batch)` for multiple
game roots. It checks every board, encoded history, and source identity before
one session call and returns observations in input order. `evaluate_root` uses
the same batch path with one row. Root counters report submitted ONNX calls and
rows, including a call that raises; invalid preflight data leaves both at zero.
Leaf counters remain separate and count successful leaf batches and real rows.
This makes batching possible for a later scheduler, without measuring a batch
size, selecting an actor temperature, or implementing that scheduler.

Root-only consumers now retain just the float32 teacher prior and native WDL
arrays. `search_inputs()` derives independent compact policy and WDL logits on
demand, preserving the prior four-array record's returned bytes. In a three-root
CPU fixture with float64 WDL, the sum of retained `ndarray.nbytes` is 22,368
versus 44,700 for the prior layout, a 22,332-byte payload reduction. This
excludes Python object and allocator overhead, and it is not a throughput
measurement. Search still requests and receives its logits when used.

Repetition-plane encoding has a separate process-global switch. The evaluator
now requires an explicit boolean `history_rep_fix`, verifies that
`rep_fix.current()` matches it at construction and before every root or leaf
batch, and records the mode with each root observation. It refuses an unset or
changed mode even when a nonrepeating board's planes happen to match. The
evaluator never changes the switch: a future worker must configure it once
before constructing any CBoard and keep it fixed for the game's lifetime.

CPU fake-session tests cover reversed policy/WDL output order, both colors,
castling, promotion, en passant, stale root/leaf histories, mismatched legal
sets, exact input/source keys, native float16 and float64 WDL, extreme finite
logits, root-only sampling, and an actual C-search call with padded leaf batches.
The focused adapter and neighboring conversion/WDL tests pass (100 cases);
scoped Ruff and basedpyright report no findings. These are interface and
numerical-contract checks, not evidence of playing strength or speedup.
