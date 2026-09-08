# BT4 adapter: collect identity records during verification

A single CPU comparison on 8,236 banked raw positions reduced verification plus
identity-cache preparation from **17.489 s to 11.524 s**, a **34.1% wall-time
reduction**. The complete identity arrays and verified receipt dictionaries matched
exactly. This measures one adapter component, not end-to-end preparation throughput.

## Motivation and change

The completed G10v3 batch prepared 528,545 common rows in 56.6 minutes. Its recorded
adapter stages consumed 1,423 seconds (41.9% of total wall time), ahead of derivation
(32.5%) and rank generation (19.4%). Those stage receipts did not separate the
adapter's internal costs.

The adapter called the full raw verifier, which reconstructed history and encoded
every input, then repeated that work to collect quantized history keys and worker
IDs. The change lets the verifier fill the adapter's already bounded identity
buffer from the planes and IDs it has checked. Default verifier callers receive
the same attributes and perform the same checks. Original/source fingerprints,
history keys, legal policy support, finite normalized policy, array digests, row
counts and source-stability checks remain in place. Cache publication follows full
successful verification and receipt matching; a partially filled buffer is not a
verified result.

## Preregistered observation and result

The [plan](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/plan.json)
selected only the original closed `run06_g10/w00-00000.jsonl.zst` and its existing
BT4 receipt. It specified exactly one baseline-first pair of actual
`RawInputs.get` calls, with empty private caches and imports/constructor setup
outside the timers. There was no inference, corpus census or complete adapter run.

The baseline is commit `f77101a59fb2b38fd8e68c4cf44111e869a54115`.
Both arms used Python 3.13.15, NumPy 2.2.6 and Torch 2.14 CPU, numeric/Blosc threads
limited to two, affinity 4,5, nice 19 and ionice class 3. CUDA was hidden and stayed
uninitialized. The bound was 890 seconds to TERM plus five seconds to KILL.

| Actual verification/cache preparation | Baseline | Candidate |
| --- | ---: | ---: |
| Wall seconds | 17.489448 | 11.524281 |
| CPU seconds | 17.465856 | 11.516664 |
| Time inside verifier | 9.610361 | 11.510297 |
| Remaining preparation time | 7.879087 | 0.013984 |

The candidate's verifier timer includes quantized-key collection, which previously
happened in the second pass. The overall component ratio was 1.518×. Both identity
arrays had SHA-256
`fac412df749e86ee1191eb0355a20f37687daab5ec52f0ce5844e4d58f82ca0a`.
All compact input pins and original raw/sidecar storage identities remained
unchanged.

The entire measurement exited successfully in 32.55 seconds, with a peak RSS of
353,348 KiB and 922,944 bytes of cache files. Each cache was limited to 1 MiB;
the aggregate temporary-file bound of 1 GiB was checked at completion. The per-file
size limit was not an aggregate filesystem quota.

This single old-first observation is sensitive to cache order and shared-host
contention. There were no repetitions or confidence intervals. It excludes policy
gathering, derived-input checks, output writing/readback, subsequent rank generation
and training. It supports retaining the duplicated-work removal without claiming
a 34.1% reduction for the full adapter or preparation pipeline. Runtime adoption
remains separate from publication.

## Validation and evidence

All 27 focused tests passed: nine new compatibility/integrity cases and 18 existing
adapter/raw-sidecar cases. They include exact legacy-cache equality, unchanged
default verifier attributes, malformed-buffer and corrupted-content refusals,
and actual shuffled three-shard publication with every raw row encoded once.
Independent review also exercised a final-digest failure and confirmed that no
identity cache was accepted or written. Final whole-repository Ruff, basedpyright
(zero errors) and vulture passed in 201.56 seconds. A separate review checked the
measurement's original baseline bytes and arithmetic.

The [archive manifest](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/manifest.json)
binds the [completed result](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/completed.json),
[measurement source](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/measure_once.py),
[exact command](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/command.txt),
[author validation](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/author_validation.json),
[independent code review](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/independent_review.json)
and [measurement review](../../scratchpad/bt4_joint20/adapter_single_pass_v1/published_evidence/performance_review.json).
Original machine paths are historical evidence, not portable launch instructions.
The two baseline program files can be recovered from the stated commit using the
manifest's filename mapping and hashes; raw data and cache arrays remain external.
