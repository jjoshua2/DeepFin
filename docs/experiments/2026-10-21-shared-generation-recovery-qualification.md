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

## Targeted diagnostic follow-up

The diagnostic helper now observes only the actual commit, owner ACK, actor
construction and evaluator-execute boundaries. It retains instrumentation when
callers are not quiescent, preserves the primary exception, and writes explicitly
partial evidence; a later quiescent observation uses a distinct output. It never
publishes a complete unit or grants custody.

The first ten-case CPU wrapper failed: eight exact cases passed, then its audit
hook rejected the positive SQLite control before the final negative case ran.
CPython 3.10 converts the database argument to bytes before emitting its audit
event; the wrapper recognized only the string memory sentinel. The failed run
and original nine-case JUnit remain preserved. The distinct correction accepts
only the string and bytes forms of that same exact sentinel; all external-file
restrictions and fixture bytes stayed unchanged.

Only the failed positive and previously unrun negative controls were selected
for a second, independently reviewed CPU gate. Both passed with the actual
worker factory/nested commit and retained checkpoint transaction. The saved audit
metadata observes the bytes form. Production imports, native images, normal
conftest, method/profiler/thread restoration, native repetition state and
CUDA-uninitialized checks passed. This does not retroactively make the first
wrapper successful or add product qualification credit.

[Diagnostic source and evidence manifest](../recovery/2026-10-21-targeted-diagnostics.json)
indexes a byte-preserving source archive, the two separate CPU outcomes and exact
reviews. The finite adapter still needs its actual caller/import binding and
independent action review before another model attempt. No failed attempt is
silently resumed or overwritten.

## Targeted-diagnostic finite attempt V3

After five distinct adapter CPU controls passed, one independently reviewed
successor used the copied runtime containing 613 payloads and 41 module pins.
It retained the same one-unit, 128-game, 64-actor geometry and 300-second active
plus 180-second cleanup limits. It returned native exit 1 after 303.016 seconds
at the active deadline; full-unit qualification failed.

The saved unit launch binds the original model and repetition fix. Its saved
first-root provider proof reports 644 CUDA nodes, including 195 neural nodes,
and zero CPU-provider nodes in the same child/session. This observation read the
proof JSON, not the profile payload; the reported provider counts were not
independently recounted. One completed writer receipt, game 52 with 22 rows,
was retained. NPZ payloads were not read. Checkpoint WAL and SHM files remained,
so the observer did not open SQLite and the owner ACK count is unknown. The
previous attempt's verified ACK must not be transferred to this run.

Neither targeted diagnostic snapshot was present. There is no batch histogram,
refill-order witness or full-unit result from this attempt. Source inspection
shows that deadline cleanup may signal the child before Python unwinds its
context managers; this is a plausible explanation, not a uniquely established
cause. CPU exception-retention tests do not prove persistence across process
termination. Before another model experiment, bounded telemetry must be saved
during work, without relying on terminal unwinding, while retaining every
original guard call, STOP, custody and deadline check. No retry or larger limit
is authorized by this failure record.

Independent saved-closure reconciliation accepted kernel `ECHILD`, released
canonical locks, and identical before/after ordinary-state digests: generation
stayed CLOSED+PAUSED with all 564 controls, six evaluation STOPs and five stopped
checkpoint component sets preserved. Sampled family peaks were 1,861,275,648 RSS
bytes and 1,842,258,944 PSS bytes; these are sampled observations, not hard family
quotas. Conservative family CPU accounting was 160.702 seconds. These figures
and one writer receipt do not establish throughput or explain the deadline.
There is no speedup, numerical-equivalence, adoption or training-credit claim.

The [V3 evidence index](../recovery/2026-10-21-bt4-diagnostic-attempt-v3.json)
retains exact failure, proof, writer, component and independent-review hashes.

## Distinct 60-second during-work diagnostic

The durability mechanism passed eleven standalone CPU observations, including two actual owned-child termination cases. Five further pure source-join checks passed before a separately reviewed 60-second model attempt. The source preregistration was published at `5c519c0`; none of the old 300-second attempts was replayed or relabeled.

This attempt reached its active deadline and returned native exit 1 after 62.991 seconds. Saved physical closure was independently accepted: kernel `ECHILD`, no held canonical locks, and identical CLOSED+PAUSED preservation digests with 564 controls, six evaluation STOPs and five stopped checkpoint sets. Ordinary generation remained paused. The saved child return code is null; it is not invented from the native exit.

Five immutable snapshots survived: entered, first-execute pending, first-execute post, and two later during-work records. At the last saved timestamp, the trace recorded 12 entered and returned executes, all with 32 rows, 64 actor admissions, no execute exception, and zero writer/ACK callbacks. These are historical counters, not final counts. The fixed receipt inventory observed zero completed writer JSONs; SQLite stayed unopened with WAL/SHM present, so the checkpoint ACK-row count remains unknown.

Across the same entered-to-last-saved interval of 53.716 seconds, completed original evaluator `execute` boundaries accumulated 0.966 seconds, about 1.8%. Most of that interval was outside the measured boundaries. This is not GPU utilization, a full-run time partition, or proof of a unique guard/host bottleneck. Startup versus steady gameplay was not cleanly separated.

The newly saved first-root proof reports 644 CUDA graph nodes, including 195 neural nodes, and zero CPU nodes; registered providers include CUDA and CPU. Those are saved first-call graph-node counts, not inference-call counts or a fresh profile recount. The proof links this attempt's launch and same live session. No numerical-equivalence, full-unit, throughput, speedup, adoption or training-credit claim follows.

The next narrow hypothesis is time spent in unchanged owner-side guard/control/proof and callback-service work. Any added timing must preserve every call, result, exception, STOP and custody check, pass CPU fault controls, and receive separate review before another model attempt. The [result index](../recovery/2026-10-21-bt4-diagnostic60-result.json) retains source, closure, snapshot and readout identities.