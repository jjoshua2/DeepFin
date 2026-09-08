# BT4 raw-label legal-move reuse

Recorded September 7 local time; execution receipts use September 8 UTC.

The candidate preserves the labeler's output while avoiding repeated legal-move
work. In one CPU-only comparison on 1,024 source-qualified positions, the exact
postprocessing loop took **0.744244 s before and 0.378945 s after** the change:
49.08% less wall time for this component. All float32 policy values and aggregate
statistics matched exactly. This is not a measured end-to-end labeling speedup.

## Why this change

The live labeler serializes JSON decoding, history reconstruction, encoding, GPU
inference, legal-policy projection and sidecar publication. Its `--threads` option
configures ORT threads; it does not parallelize Python preparation. A single
GPU-idle observation does not identify the limiting stage.

The authorized initial profile read the first 1,024 physical rows of the original
closed `run06_g10/w00-00000.jsonl.zst` once, using the actual wise-cloud functions.
The raw shard contains 8,236 rows and its banked SHA-256 is
`3c93d2d127bc6d11db9eb31425e0c795c96b417106d737632ef4b689569ae17d`.
The original input keys were reconstructed and verified, and the source's
file identity, size, mtime and ctime remained unchanged. The full source hash was
reused from the closed receipt and a completed earlier source check; the profile
did not repeat that whole-file hash.

| Profiled CPU stage | Wall seconds |
| --- | ---: |
| Legal mapping proxy, without logits | 1.100252 |
| History reconstruction, encoding and input-key checks | 0.971694 |
| JSON/decompression | 0.873926 |
| LC0 input conversion | 0.062427 |
| Position fingerprints | 0.011651 |

These cProfile measurements include instrumentation overhead. Nested timings show
that history reconstruction cost 0.552704 s, `encode_position` 0.183646 s and extra
feature encoding 0.030431 s. JSON `raw_decode` accounted for 0.808424 s. Thus the
composite history timer did not support blaming feature encoding alone.

The mapping proxy processed 27,425 legal moves but called the compact converter
54,850 times and reparsed 54,850 UCI strings. The candidate first removes this
repeated work without introducing a CPU pool, IPC or a different resource regime.

## Implementation and parity

A shared helper enumerates the complete legal `chess.Move` list internally and
computes the existing teacher distribution directly from those objects.
`legal_move_policy` retains its public UCI-returning interface. The raw labeler
reuses the Move objects and computes compact indices once. It accepts no
caller-supplied move subset or cross-position cache.

Teacher and compact mappings remain independent. The helper's complete internal
legal enumeration establishes coverage; the raw consumer retains count,
injectivity, range, finite/nonnegative and unit-mass checks. Float64 gathering and
softmax arithmetic retain their order, followed by the same float32 cast. Input
history, teacher identity, row order, receipt format and illegal zero mass are
unchanged. The repeated expected-set calculation used the same compact converter;
it was not a second independent mapping implementation.

The 62 focused tests passed, including both colors' castling, promotions and
underpromotions, en passant, pins, no-legal-move positions, nonfinite-logit handling,
malformed shapes and mapping failures. An actual multi-position `label_shard` with
mocked inference matched all five content arrays and their hashes, entropy/top1
sums and legal counts against the frozen legacy projection. It also verified one
compact conversion per legal move. A separate reviewer passed eight consequential
cases and found no code blocker.

Whole-repository Ruff, basedpyright and vulture passed under the canonical local
validator; lint took 161.51 s. No full-suite or live-runtime restart was performed.

## Single CPU A/B observation

After parity tests passed, the registered comparison used the same 1,024 positions,
fixed shared float32 logits spanning −4 to 3, one invocation of each implementation,
old first, and no cProfile. It extracted the exact helper and board-loop bodies
from pinned source files. Both arms ran under the same qualified wise-cloud
Python 3.10.12 / NumPy 1.26.2 runtime, on core 0 with two numerical threads, nice 19,
ionice 3 and GPU hidden. The bound was 115 seconds to TERM plus five seconds to KILL.
No model or ORT session was created; CUDA remained uninitialized.

| Exact numerical postprocessing | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| Original | 0.744244 | 0.736780 |
| Candidate | 0.378945 | 0.372441 |

The observed ratio was 1.964×. Both results had policy SHA-256
`5aa3050aea6923d13abb3f0937ad09c3fb53dd7fc43be001d97e86b2610b6811`,
entropy sum 2476.5665358603, top1 sum 209.22563568875194 and 27,425 legal moves.
The fixed-logit hash and all input/runtime identities are in the completed receipt.

One initial attempted run failed before reading rows because the new checkout had
only CPython 3.13 extensions. An import-only preparation check then found that
current-main's transitive MCTS imports needed a newer ABI than the frozen 3.10
runtime. Neither reached a timed numerical invocation. Their evidence is preserved.
The final comparison used source-extracted functions under the common frozen
runtime rather than borrowing an incompatible MCTS binary or rebuilding a running
checkout. Main's actual candidate was separately tested in its qualified 3.13
development environment.

There were no timing repetitions or confidence intervals. Artificial logits,
old-first ordering, cache effects and this single early prefix limit the result.
The A/B excludes JSON/history preparation, real inference, compression, writing,
full-bank metadata checks and session startup. It supports keeping this bounded
reuse change; it does not establish a 49% improvement for the whole labeler or any
playing-strength result. Live adoption remains a separate operation.

## Evidence

The [evidence manifest](evidence/bt4-label-legal-reuse/manifest.json) binds compact
records, exact commands/scripts, source-qualified input hashes and external bulk
artifacts. Start with the [CPU profile](evidence/bt4-label-legal-reuse/profile_completed.json),
[analysis](evidence/bt4-label-legal-reuse/profile_analysis.json),
[A/B plan](evidence/bt4-label-legal-reuse/ab_plan.json),
[A/B result](evidence/bt4-label-legal-reuse/ab_completed.json),
[author validation](evidence/bt4-label-legal-reuse/author_validation.json) and
[independent review](evidence/bt4-label-legal-reuse/independent_review.json).
The review's pending-at-creation lint/benchmark fields are preserved; the later
author receipt records their completion.

Baseline code is commit `bb01468b1a0eb0e6a97102e069525911ef04f76a`.
Candidate file hashes are bound by the author and independent receipts; the
benchmark preceded its publication commit. Full pstats, row identities and legal
mapping banks remain under
`scratchpad/bt4_joint20/label_preparation_profile_v1/`, with hashes in the manifest.
No complete raw corpus, teacher model or checkpoint is included in this record.
