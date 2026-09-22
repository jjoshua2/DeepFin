# BT4 complete labeling pipeline screen — September 21

## Registered question and status

Prepared, CPU tested, independently reviewed and queued after the existing D
training, D-C match and D-B match complete successfully. Execution is pending.
Does projected JSON decoding improve the complete raw BT4 labeling path, and does
one-batch CPU prefetch add at least 5% beyond that improvement?

This is a throughput screen with exact target preservation. It does not change
policy/value mixtures, production settings, or BT4 inference batch size. Candidate
prefetch remains off by default in [PR #815](https://github.com/jjoshua2/DeepFin/pull/815),
stacked on [projected decoding PR #814](https://github.com/jjoshua2/DeepFin/pull/814).

## Fixed design

Six fresh processes, order **A B C C B A**, each label and deeply verify the same
8,236-row closed `run06_g10/w00-00000.jsonl.zst` shard at batch128 (64 full batches
and one44-row tail). Each process warms its own CUDA session with its first128rows
before timing, and banks a provider profile proving CUDA neural execution.

| Arm | Frozen code | Behavior |
| --- | --- | --- |
| A | `7939609c903f2a59a57bc52f282fc39ad60182fa` | Original serial decoder/preparation |
| B | `74a31a1d3ee18328d12c2e6cf3f519316eb61114` | Projected JSON decoder, serial preparation |
| C | `0c34941073cb6e4dceb3ca735285be457c18c7d7` | Same projected decoder, one CPU preparation future ahead |

The primary metric is median **producer time plus deep verification time**.
C passes the incremental screen only when `median(B) / median(C) >= 1 / 0.95`.
A/B improvement, producer-only time, session setup, inference calls and whole-worker
elapsed time are secondary descriptive measurements. Input/output SHA observation
cost is included in producer timing and may itself overlap with CPU preparation.
These are instrumented pipeline times, not an uninstrumented throughput estimate.

Every arm must reproduce the exact actual inference-input sequence, raw policy/WDL
outputs and all stored array values/dtypes/shapes. Each output also passes the real
producer's complete verification replay. Any failure yields an incomplete or
inconclusive screen; do not widen tolerances or selectively rerun unfavorable arms.
Two observations per arm on one cache-affected shard do not establish fleet or500M
throughput, nor do they qualify external-drive training performance.

## Runtime and bounded execution

The immutable local launch package is
`/home/josh/chess-artifacts/operations/bt4-pipeline-20260921/`.
Its plan pins every file and resolved file target in three fresh detached runtimes,
the closed source shard and manifest, BT4 ONNX weights, dependency command descriptors,
Python and installed numerical/codec package versions, and the exact harness.
Four existing native extensions are copied into each frozen runtime. Imports cannot
write bytecode into the pinned runtimes.

The registered command is:

```bash
/usr/bin/python3 /home/josh/chess-artifacts/operations/bt4-pipeline-20260921/run.py \
  --plan /home/josh/chess-artifacts/operations/bt4-pipeline-20260921/plan.json \
  --sha256 8ad936c13fd36b1e97fcae11d27b87423a00cea00b754e342f6d90bb5bd614dc --execute
```

Admission requires successful logged receipts for D, D-C and D-B, plus the hardware
GPU lease and an empty device. Each child inherits that lease. The internal cap is
19minutes, with the existing operator enforcing20minutes; owned worker groups are
terminated and reaped on failure. CPU affinity14/15, nice19 and two threads apply.
STOP paths,40GiB available RAM,150GiB free disk,2GiB new-output cap and24GiB GPU
memory cap are enforced through final comparison and pin validation. The ONNX CUDA
allocator limit is8GiB. No new GPU work runs concurrently with D or its matches.

## Validation

Nine focused CPU tests pass, including input/output parity rejection, dependency
receipt failure, owned-worker reaping, nonreplacement of receipts, and STOP during
final comparison. Independent review caught a missing final guard and confirmed its
fix. Three additional CPU-only worker smokes call each actual producer and verifier with
a fake ONNX session, confirming argument wiring and identical synthetic outputs.
Scoped lint passes for the final candidate. Whole-repository lint reports the
same14 host-environment type diagnostics in six unchanged tests as the banked main
baseline; Ruff and Vulture pass. No GPU result exists yet.

[Independent code review](evidence/bt4-pipeline-20260921/code-review.json).

[Compact registration and full-plan hash](evidence/bt4-pipeline-20260921/registration.json);
[registered command](evidence/bt4-pipeline-20260921/registered-command.json).

[Final admission review](evidence/bt4-pipeline-20260921/admission-review.json) and
[append-only queue registration](evidence/bt4-pipeline-20260921/queue-registration.json)
confirm all prior items and active state were preserved.
