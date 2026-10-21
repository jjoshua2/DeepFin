# Shared generation recovery: bounded BT4 qualification

This is the readout of two finite correctness attempts on the reconstructed
PR1110 runtime. It is not a speed comparison or a recovery of the lost final v8
artifact. The ordinary head14 queue remained CLOSED+PAUSED throughout.

## Registered question and limits

Can the reconstructed shared owner, pooled selfplay runner, native history path,
real BT4 CUDA evaluator, durable writer and owner checkpoint complete one private
128-game unit while preserving ordinary state? The reviewed packet used 64 live
actors, native microbatch 32, dispatcher target/max64, maximum 400 plies, and seed
7000000000002000000. Unit completion and the actual writer-to-owner-ACK-to-refill
trace were required; partial receipts cannot satisfy that rule.

The second attempt retained 300 seconds active plus 180 seconds cleanup, family
CPU360 seconds, child CPU 300 seconds, sampled RSS 8 GiB/PSS 6 GiB, and private output
512MiB. CUDA address space kept its inherited unlimited setting; the observed
virtual reservation was not treated as resident memory. Three exclusive canonical
leases, six evaluation STOPs, 564 controls and five stopped checkpoint component
sets were bound before launch. Return required unchanged ordinary state, released
custody and actual ECHILD. No output is admitted as training data.

## Exact source and model

- Production startup fix: `42027482a7091ddbd2c8bc98e266b3ac3afbb744`.
- Runtime-v3 closure: `1fce66e0409ca24449d0ea6c897f9a0a5457d35fc0982e6ee9ee87a137168f12`;
  609 payloads plus closure, not whole current Git-tree equivalence.
- ONNX SHA256: `809db9c3c8063683bf8e7c08de8fec92ea4e37ad8c68c8b1befbde4bd1958961`;
  FP16 internals with FP32 I/O, CUDAExecutionProvider, retained artifact-specific
  arithmetic acceptance, `use_tf32=1`, GPU arena limit 2 GiB. This test does not
  establish full 4,096-row numerical equivalence.
- Second parent packet: `2aec73112bdb246ef52cc276661090e72948c4d7cd5705e01fea439058b10c5b`;
  action review `4cde1a360936e4334f3e132b0c133566c1a951d796a8846a18754a0d4a886331`.

## Observations against the rule

The first attempt reached ONNX session construction but failed before inference:
the evaluator required repetition mode before the owner configured it. The
production fix moves the existing setter before recovery/evaluator construction.
Three actual-caller startup regressions passed with production Python3.10/native
origins, normal conftest and restored repetition state. An independent exited-child
accounting fix passed five controls. Original failures remain preserved.

The corrected second attempt passed startup, performed real CUDA inference and
entered pooled gameplay. It stopped at its accepted active deadline: parent
301.641880 seconds, native 302.967614 seconds. It failed the complete 128-game rule.
The child reported `accepted owner deadline`; no child exit code is inferred from
the native wrapper's exit1 or the closure's null child return code.

Read-only retained-artifact inspection found:

- First input shape `[32, 112, 8, 8]`; 644 CUDA-provider nodes including 195 neural
  nodes, zero CPU-provider nodes. This is first-call proof, not a call histogram.
- One completed game, ID 52, with a writer receipt and committed owner-checkpoint
  ACK for 22 rows. The unit checkpoint still says RUNNING, not complete.
- NPZ name/stat observed; its contents were not read or reverified. No payload,
  admission, complete-unit, actor-refill or playing-strength credit follows.
- No checkpoint WAL/SHM/journal existed. The bounded immutable read left the main
  identity/component set unchanged. No ordinary queue SQLite was opened by this
  progress inspection.
- Sampled family peaks: RSS 1,839,222,784 bytes and PSS1,820,335,104 bytes;
  conservative family CPU 204.992699 seconds. These are sampled observations,
  not hard family quotas or complete short-lived peaks.

Independent closure review accepted actual ECHILD, released canonical locks and
identical before/after ordinary snapshot
`7bc24eb410a8672f519312da639385deedde1cf757f1f9a27faf7ab99b7e222a`.
All controls, evaluation STOPs and stopped checkpoint main/sidecars stayed intact.
No model process or experimental custody remains. Ordinary generation stays paused.

## Evidence custody

Private native evidence lives under
`C:/Users/jjosh/Documents/Codex/2026-10-13/task-5/bt4-qualification-intake-root-20261021-v1`.
The private child output is
`/home/josh/chess-artifacts/operations/bt4-pr1110-correctness-private-20261021-v2`;
its parent output is the adjacent `pr1110-finite-one-unit-correctness-20261021-v2`.
No private outputs are committed or deleted.

| Evidence | SHA256 |
| --- | --- |
| Six-file terminal export | `656d7bba1f84d8a7a7cf619bb0f9d5f66ad7e891f3394202098f0c6976776141` |
| CLOSED-PAUSED receipt | `f9967e151c02493b775054133922c0743cbc4b23d2ceb01941620e3f5131a30b` |
| Independent failure/closure review | `30a7bb8b8f24686785c93497f8ae73e2e1882590c13a45e7ad662c91294a69f2` |
| Saved metadata observer output | `864b6a0fa753a6acdeb29db8c64fe357a1ff9d13abf7089da9785fdc3f49ee84` |
| Independent saved metadata review | `f17dd6c698beeb7c3412985325a129bc6c6f5993d42c9aed2181c1edcdfb3ed0` |
| First-call CUDA profile | `30316882b276edf759f24355e13736f52337c46efa26f7d2e82fe9195732df53` |
| Unit CUDA proof | `e58dea1a18a32dfa2fd40ab85d068b1840630eff8f8b8d26ba3a9f952e716c60` |
| Unit launch | `527ea53db012ab8f037ad750d66c0373bd9283d96c16233b83b2a68d04f27ca2` |
| Saved private checkpoint main | `0440a1b730ae2b0451849b7960fae529d39c049bb0d15da877dad39de377dde3` |

## Confounds and next decision

This qualification-only adapter explicitly used global Python call/return
profiling, including `frame.f_locals` access before filtering for witness methods.
The finite guard also scans the preserved control set on its hot path. Source
inspection establishes additional work, not its share of elapsed time. No clean
throughput baseline was run; dividing 22 rows by total wall time would not measure
steady generation throughput.

The runner's completion trace and final batch counters were only serialized after
normal return. They were lost on this exception; absent telemetry cannot establish
how many in-flight plies, calls or actor replacements occurred. Prepare narrowly
targeted, exception-safe diagnostics without relaxing custody, STOPs, preservation
or deadlines, then independently review the smallest distinct successor. No blind
retry, cap increase, arbitrary queue refill, model adoption or generation resume
is authorized by this failed readout. Matched performance qualification follows
correctness; target 320/340 remains a hypothesis.
