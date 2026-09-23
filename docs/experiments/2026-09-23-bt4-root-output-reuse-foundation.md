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

CPU fake-session tests cover reversed policy/WDL output order, both colors,
castling, promotion, en passant, stale root/leaf histories, mismatched legal
sets, exact input/source keys, native float16 and float64 WDL, extreme finite
logits, root-only sampling, and an actual C-search call with padded leaf batches.
The focused adapter and neighboring conversion/WDL tests pass (88 cases);
scoped Ruff and basedpyright report no findings. These are interface and
numerical-contract checks, not evidence of playing strength or speedup.
