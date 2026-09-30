# Bend explicit checkpoint qualification

## Decision and hypothesis

Follow #779 (4ab0836c69f7d60ae41c84fa069527a8c731f047) with checkpoint and
transformer compatibility, not another perft optimization. Compiler remains
57bc84edc0df32780e2c4dde44e3a2ee1a500cc9. Bend chess/search, production model,
encoder, configuration and perft depths remain unchanged.

Hypothesis: strict self-describing checkpoints can drive the existing native
batched evaluator and Bend tree without more language/compiler features. Keep
shared architecture parsing and policy-output selection. Reject partial weight
loads, encoding guesses, stale file identity and ambiguous SWA selection.

## Confirmation plan (recorded before hosted execution)

An explicitly untrained saved transformer fixture: 2 layers, width 32, 4 heads,
Smolgen, relation basis and dynamic relations, root-legacy-meta/v2_threats,
repetition fix enabled. Actual checkpoint deserialization and strict weight load;
not TinyNet and not a representative trained production checkpoint.

All four Bend modes; each has one-real-row control, normal batched searches and
queued/in-flight cancellation/recovery across five roots. Existing group contracts
check native logits against singleton eager execution at unchanged CPU F32 atol
2e-6 / rtol 2e-5, per-reply statistics and complete final trees. Retain all previous
session, single-row and batching regressions after changing the native worker.

Bounded 15-minute hosted CPU job, one compiler worker and two inference threads.
No timing acceptance gate, GPU allocation, training, checkpoint download or live
restart. No additional search/inference/compilation in ordinary pytest. Recovery:
discard this isolated branch; earlier PRs and production are untouched.

## CUDA boundary

V3 explicitly records CPU/F32 or CUDA/BF16 and a device index. CPU builds compile
the generic ATen device-selection path; this is NOT execution on CUDA. The runner
requires a real requested GPU and explicit predeclared CUDA tolerances. It fails
instead of silently using CPU. Native output checks require matching device,
float32 tuple policy/WDL and fixed shapes. Existing CUDA-package parity is separate.

No trained checkpoint or GPU was available in the local working environment.
The ability to accept a user-supplied file does not establish its provenance or
compatibility: actual execution and a successful report are required. No code
claims this fixture's step zero or any arbitrary checkpoint step proves training.

## Local observations

CPU Torch 2.10.0 and Clang 17: saved-transformer round trips passed in all four
Bend build modes, including normal/control/cancellation groups and all five roots.
Maximum native/eager singleton logit difference was 5.960464477539062e-7 for the
batch-four package. A separate native-target batch-one run also passed, max error
5.364418029785156e-7. The original TinyNet single-row and batched neural checks
passed after the worker target extension.

The initial 108 cheap contracts passed. Review then found that the shared group
still assumed four rows and an eight-row reservation: a requested batch one/two
would fail dispatch, while batch sixteen would fail capacity validation. The
group now derives its row limit from the package and reserves max(8, batch);
normal batch-four behavior is unchanged. Ten cheap bucket/limit tests bring the
count to 118. Actual native-model execution qualifies buckets one and four,
not all supported bucket sizes or the production model topology.

## Hosted readout: PASS

[Run 35418537394](https://github.com/jjoshua2/DeepFin/actions/runs/35418537394),
job **105831745415**, passed every stage. Exact validated implementation was
recorded as `9d15f7731d8bbed71feebcfeb8ab4c790ba8d299`. Clean stacked commit
`726b30122de5a8b33b6a69d8ab5b29bb42bab00d` preserves those executable bytes and
the tested permanent workflow, removing only temporary development payloads and
workflow history. Native-probe blob hashes match the locally checked source.

Locked **Torch 2.14.0+cpu**, Bun 1.4.2, Python 3.13.15. The actual saved fixture
contains **5,043,005 parameters**. It is still an untrained two-layer transformer,
not a 32M-parameter production checkpoint or learned chess-strength demonstration.

| Executed checkpoint package | Bend modes | Search epochs | Real input rows | Native calls | Maximum absolute logit error |
| --- | --- | ---: | ---: | ---: | ---: |
| Batch four CPU F32 | generic, portable, native, UBSan | 68 | 500 | 336 | 5.364418029785156e-7 |
| Batch one CPU F32 | native | 17 | 125 | 125 | 7.152557373046875e-7 |

Totals: **85 epochs, 625 real rows and 461 native forwards**. These include ten
intentionally cancelled partial epochs; 75 epochs completed normally. All exact
board/tree/visit/best-move controls and cancellation recovery checks passed. Every
real output row is compared with an independently executed eager singleton. The
same physical batch package is used for each padded one-real-row control; native
call-count reductions are NOT a throughput speedup.

Selected final normal outcomes, identical across modes and recovery:

| Root | Completed | Nodes | Best private key |
| --- | ---: | ---: | ---: |
| Start | 8 | 182 | 1739 |
| Kiwipete | 12 | 545 | 139 |
| En passant | 4 | 24 | 324 |
| Black promotion | 6 | 68 | 12296 |
| Repeated pre-root history | 10 | 215 | 1025 |

Private keys are not policy-head indices or recommendations. Native-derived
policy/WDL, not hard-coded best moves, drove both the Bend tree and the diagnostic
reference. No production Gumbel-search equivalence is claimed.

Regressions preserved in the same hosted run:
- 38 original sessions per mode; 207 independent python-chess oracle positions.
- 40 original TinyNet singleton epochs / 640 native calls, 64 encoding comparisons
  plus distinct-history check, and six malformed native inputs rejected.
- Original TinyNet batching: 68 epochs / 500 real rows / 332 native batch calls;
  all row/control/cancellation checks and six malformed inputs passed.
- Focused Ruff, Basedpyright (zero errors/warnings) and **118 cheap tests** passed.
- An actual CUDA-requesting CLI on the CPU-only runner failed at preflight with
  `requested CUDA device is unavailable; no CPU fallback`. The failure report is
  retained. This is a negative safety test, NOT a CUDA qualification.

The first hosted attempt stopped on four type-check findings: Python 3.10 API
compatibility for file_digest and generic Module weight-attribute annotations in
two tests. Streaming SHA-256 and typed state_dict access corrected them without
suppressions. Numerical tolerances, strict weight loading and chess checks were
not relaxed. The rerun also qualifies the reviewed package-bucket correction.

## Final output-path safety follow-up

After the native confirmation, self-review added a pre-report guard: --report
must not name the input checkpoint, its trainer.pt within a directory, or a
hardlink/symlink alias. The check is deliberately outside the finally block that
writes failure reports, so it cannot clobber the source on rejection. Four local
cheap tests passed for these cases (122 total), alongside Ruff. This follow-up
changes only output-path validation, tests and this readout; the model exporter,
worker, Bend, numerical checks and batch coordinator are unchanged. The PR's
normal checks cover the final head; the earlier hosted run is not mislabeled as
having executed these four later cases.

## Evidence

Artifact **bend-checkpoint-confirmation**, ID **10576424294**, 30-day retention.
ZIP SHA-256: `9d0d76e6744cb1bfcb71dcb1c22d11e42a87af1a0c0da19e0665823d35f376a6`.
- Transformer report: `017d21e0e447c68fe7b5191bfd08e512fdfb79719cdd36a3db1ca7770a4d0e98`.
- Batch-one report: `00c7690b9d44b26bac583f55411917a165c738e88c1bd2bee60e39ebd0edf526`.
- Session regression: `07c74e0e61735f3b1949c1cf82268cdfcde1763517ebab6c8dba42be4b40ff0a`.
- TinyNet singleton: `351a0c87f0991e6bf9e9b2dd7018cdac64c8126073f7b913976d0ea10139d418`.
- TinyNet batching: `fde9e3129b9a4d665677ae71f6164236942b49f32f71df3d88bd39cf1f2e359d`.
- CUDA-unavailable negative: `1538e934fbd4caebfd009a4f0a0d44b7cfae73c7178742eb651078c250fa3c96`.
- Saved fixture checkpoint: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Batch-four package: `539ea242a6d65a7c63bf6e30132985e84f3a850831143a6e47a28f33d3c53a0f`.
- Batch-one package: `1765d8fe5a74a6819e7256f32a721568c26fd613678a86febcbbbad7484cacdc`.

Temporary untrained checkpoints/packages are not committed. Hashes identify the
executed artifacts, not a reproducible-archive-bit-identity guarantee. Only load
trusted checkpoints and native-code AOTI packages.

## Limits / next decision

No trained production checkpoint, real CUDA execution, throughput, playing
strength, full draw handling or root reuse is qualified. The new runner accepts
an explicit copied checkpoint and model/SWA selection and reports failures rather
than guessing architecture or replacing missing weights. It uses the existing
shared strict architecture parser and policy/policy_own helper. It does not make
an all-Bend inference/coordinator claim; Python and native C++ remain boundaries.

No compiler or perft changes. Only cheap tests enter ordinary pytest. The existing
path-scoped native-neural workflow is extended within its 15-minute limit; no
parallel duplicate workflow or recurring timing benchmark is added. Nothing is
merged/deployed. Self-review only, not independent review or a universal proof.
The next decisive run is an actual trusted checkpoint on the target CUDA host,
with predeclared tolerances and a safe compute window, not another smoke fixture.
