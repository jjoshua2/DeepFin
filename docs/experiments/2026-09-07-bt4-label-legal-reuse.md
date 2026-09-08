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

## Qualified runtime and armed handoff — September 8

[PR #537](https://github.com/jjoshua2/DeepFin/pull/537) merged the reusable change.
An inactive deployment snapshot, `4e600902e1fd08727e1d83e009d6e241be0c7cba`,
starts from the existing frozen runtime and changes only the two reviewed Python
files. It retains **Python 3.10.12, ONNX Runtime 1.23.2, Torch 2.11.0+cu128 and
NumPy 1.26.2**, with the same four physical CPython 3.10 native binaries. No
packages were installed or live checkout changed.

The [runtime qualification](../../scratchpad/bt4_joint20/label_preparation_profile_v1/runtime_adoption_pr537/qualification.json)
passed actual imports, exact functional remap identity and both-color castling
probability checks. Existing sidecars from **both run06 G10 and run07 G10
companion4** passed ordinary source/model/provider/layout admission against their
banked receipts. Qualification kept CUDA hidden and created no ORT session;
installed providers alone do not prove a successful new GPU inference group.

**The graceful handoff was armed at 03:51:30 UTC; takeover remains unconfirmed.**
The [start receipt](../../scratchpad/bt4_g10_raw_legal_reuse_v1/started.json) and
[armed snapshot](../../scratchpad/bt4_g10_raw_legal_reuse_v1/armed.json) record new
driver 295443 waiting for the original driver lock. Only after confirming that
waiter was ready did the parent exclusively create the owned pause request. Old
driver 4251 and queued group 288651 were preserved; the group finishes normally.
After acquiring the lock, the replacement must verify the old paused proof and
actually consumed handoff request before continuing. It retains the original two sources, teacher, output tree,
writer/GPU leases, 1,024-row batches, 16 ORT threads, OMP/MKL/OpenBLAS limits of
two, and the 150 GiB free-space reserve. The observed old resource regime was
explicitly preserved: nice 19, CPU affinity 0–31 and I/O priority class 0 (none).
The CPU-only qualification limits were not substituted for the normal labeler
regime. Both original and new pause markers remain effective.
Qualification is rechecked before normal groups; no process kill is part of this
transition.

An [independent prelaunch review](../../scratchpad/bt4_g10_raw_legal_reuse_v1/independent_handoff_findings_v1.json)
reproduced a race in the first proposal: replacement of the pause request between
its hash check and move could consume an operator's changed pause. It also found
that the PATH-resolved interpreter alias was not bound to the qualified target.
The correction validates the actually consumed request in a private archive,
restores a changed request without overwriting a newer pause, stays paused on a
mismatch, and binds the resolved interpreter. The blocked review and original v1
files remain alongside the corrected proposal. The [final independent review](../../scratchpad/bt4_g10_raw_legal_reuse_v1/independent_handoff_review_v2.json)
passed eleven isolated helper/shell checks and closed both findings before arming.
The armed snapshot has **no confirmed takeover, new label group or throughput
result**; it establishes no new GPU labels or end-to-end speedup.

The [readiness and handoff evidence manifest](evidence/bt4-bootstrap/b100-readiness-labeler-handoff-manifest.json)
binds these runtime and operational records. Archived driver/verifier scripts are
host-specific evidence, not portable instructions to restart the labeler. Actual
adoption requires the new child/runtime identity, realized CUDA provider and first
new completed sidecar stamps; an ordinary completed group can then provide a
single descriptive throughput observation without timing repeats.

## Evidence

The [evidence manifest](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/manifest.json) binds compact
records, exact commands/scripts, source-qualified input hashes and external bulk
artifacts. Start with the [CPU profile](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/profile_completed.json),
[analysis](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/profile_analysis.json),
[A/B plan](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/ab_plan.json),
[A/B result](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/ab_completed.json),
[author validation](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/author_validation.json) and
[independent review](../../scratchpad/bt4_joint20/label_preparation_profile_v1/published_evidence/independent_review.json).
The review's pending-at-creation lint/benchmark fields are preserved; the later
author receipt records their completion.

Baseline code is commit `bb01468b1a0eb0e6a97102e069525911ef04f76a`.
Candidate file hashes are bound by the author and independent receipts; the
benchmark preceded its publication commit. Full pstats, row identities and legal
mapping banks remain under
`scratchpad/bt4_joint20/label_preparation_profile_v1/`, with hashes in the manifest.
No complete raw corpus, teacher model or checkpoint is included in this record.
