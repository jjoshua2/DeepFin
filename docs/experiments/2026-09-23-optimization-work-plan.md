# CPU optimization work plan while E uses the GPU

Date: 2026-09-23 UTC. Owner: primary agent; implementation workers use GPT-6 Sol
with xhigh reasoning, as requested. Three workers are active in the current session.
This is a work plan, not a new experiment admission or a production deployment.

| Owner | Current deliverable | Next action after primary review |
|---|---|---|
| `sol_hotload_runner` | Eight-shard ABBA completed: exact parity and 62.38% less measured hot-load time; now independently reviewing host-overlap fixes | Rebase reusable diagnostic tooling onto current origin/main, fix publication typechecks, then prepare trainer-level validation after E and its arena |
| `sol_host_overlap` | Executable raw/prepared OFF/ON CPU qualification producer plus bounded GPU comparison runner; exact runtime and checkpoint/optimizer contracts | Qualify the actual batch/augmentation stream in an admitted CPU slot, then prepare the GPU comparison after E and its queued paired arena |
| `sol_generation_readout` | Audited c4/c8 readout published; BT4 compact-policy helper now reused by collector with caller-owned output buffer | Publish independently reviewed helper and exact parity tests; future generator integration must retain history, value convention and six-man outcomes |

The primary agent owns prioritization, independent code review, resource admission,
result interpretation and publication readiness. A worker finishing one task reports
its evidence before proceeding to a materially different implementation. No worker
launches GPU workloads, changes E, or restores the held second training seed.

## Admission and evidence constraints

- Six-man Syzygy adjudication and per-game TT clearing remain in all generation
  experiments. Retained mappings do not change these rules.
- Current E input waits increased during concurrent CPU generation and recovered
  after it stopped. More heavy generation is held; unused cores do not by themselves
  establish safe concurrency with training. Small code tests keep GPU hidden,
  two numeric threads, low priority and bounded cores.
- The next generation candidate is the same-engine retention OFF/ON plan, version2,
  under `~/chess-artifacts/operations/sf-retained-generation-plan-20260923-v2/`.
  Its fixed cleanup and pre-cleanup cutoff receipt passed independent review and
  five small tests. It remains unlaunched pending an appropriate workload window.
- A failed measurement stays failed. The c8 bank can support descriptive nominal
  arithmetic, but missing exact cutoff timing and cleanup failure do not become a
  passing primary scaling result. See [the generation readout](https://github.com/jjoshua2/DeepFin/pull/837).
- Loader validation reuse must preserve decoded arrays, source identities and
  mutation rejection. Host overlap must preserve raw and augmented batch order,
  coverage, initialization and final optimizer/model state. Acceptance of a flag is
  insufficient evidence that it ran.
- Teacher-output reuse must match history, legal-policy mapping, side-to-move value,
  native/calibrated target convention, precision and provenance. Generation's own
  inference remains charged to the GPU budget; no free-generation assumption.

Related: [shared compute requirements](2026-09-22-scale-compute-budget.md) and
[live loader profile](2026-09-23-loader-profile.md). No new compute has been launched
by this plan.

## Completed small CPU measurement

The [actual-loader ABBA](2026-09-23-overlay-hotload-diagnostic.md) was admitted after a host check and completed with exact parity. Median loading time fell from 10.925 to 4.110 seconds. This establishes the next trainer candidate; it does not authorize GPU work or claim a 500M throughput result. The host-overlap independent review found false-PASS and input-binding gaps; revised tooling is under re-review before any full-stream run.

## Review and handoff continuation

After the user identified an idle handoff, the primary checked live agent status:
all three workers had completed and were awaiting reassignment. The primary
resumed review and reassigned each worker. Within an active parent turn, the
primary owns completion-message intake, independent diff/test review, publication
and the next bounded assignment; another user prompt should not be needed for
those handoffs. A final response does not establish an unattended restart loop.

Completed publications: [BT4 output conversion #841](https://github.com/jjoshua2/DeepFin/pull/841),
[host-overlap qualification #842](https://github.com/jjoshua2/DeepFin/pull/842), and
[revised loader tooling #843](https://github.com/jjoshua2/DeepFin/pull/843).

The next active assignments are:

- Generation worker: a CPU-testable BT4 evaluator retaining native root policy and
  value output, with correct board-dependent mapping and exact encoded history.
  No generator command or real inference is enabled by this slice.
- Outcome worker: opt-in six-man outcome policy matching SF generation's natural
  terminal and theoretical Syzygy convention; missing tables and unresolved caps
  must not silently produce draw labels. Legacy generator behavior stays intact.
- Loader worker: direct cached-loader mutation regression using tiny schema-2
  fixtures, covering changed targets, base data and qualification receipt.

The primary reviews each resulting diff before integration. GPU benchmarks and
full-stream CPU qualification remain held; neither publication nor unit-test
success authorizes their launch.

The direct cached-loader regression subsequently passed 15 author and independent parent tests and was published as [#844](https://github.com/jjoshua2/DeepFin/pull/844), stacked on #810. That worker moved to independent review of the two generation changes without another user prompt.

## Subsequent review cycles

The primary kept the parent turn active and completed multiple author/reviewer
handoffs. The following changes are published for review; none has been deployed
into the pinned E runtime:

| PR | Result |
|---|---|
| [#845](https://github.com/jjoshua2/DeepFin/pull/845) | Natural and six-man outcome helper; unresolved caps or unavailable tables discard games |
| [#846](https://github.com/jjoshua2/DeepFin/pull/846) | BT4 root policy/native value adapter with exact history and legal mapping checks |
| [#848](https://github.com/jjoshua2/DeepFin/pull/848) | One ONNX call for an ordered batch of validated roots |
| [#849](https://github.com/jjoshua2/DeepFin/pull/849) | Derive search inputs on demand; retain fewer root arrays |
| [#850](https://github.com/jjoshua2/DeepFin/pull/850) | Require explicit repetition mode and refuse drift before inference |

These are stacked foundations for generation with reusable teacher outputs, not a
measured generation-throughput gain or an enabled label writer. The next game
stepper review found and corrected two issues: unset repetition mode acceptance,
and loss of an earlier terminal event if a later root encoding failed. The revised snapshot passed independent review and all 56 combined focused tests and is published as [#851](https://github.com/jjoshua2/DeepFin/pull/851). The generation worker is
implementing strict session/model/input provenance with fake-session tests.

A read-only host diagnostic during E recorded 2,542 compaction stalls over ten
seconds (2,541 failed, one successful), no direct reclaim or swap I/O in that
interval, and memory PSI avg60 around 9.69 percent. THP enable and defrag settings
were both `madvise`. This is a candidate explanation to investigate, not a causal
finding. No kernel setting, process environment or live job was changed. A source
allocation audit is assigned before proposing a bounded quiet-slot comparison.

E remained live at window 1079/1290 (94,952/113,459 updates) during these checks.
Its paired E/D arena remains queued; the second seed remains held for the first
seed decision. Full-stream CPU and GPU optimization benchmarks are still held.

The [generation memory and loader THP audit](2026-09-23-generation-memory-loader-thp-audit.md) records the buffering floor, transient allocations, and the next proposed child-process comparison. Its numbers do not establish an RSS bound or a throughput gain.

The verified-session factory passed final independent review and 124 focused author tests; the parent also passed all 50 session/evaluator tests. It is published as [#852](https://github.com/jjoshua2/DeepFin/pull/852). Review closed sparse external-weight, realized GPU-cap, and batch-shape gaps. A future writer still needs physical row and serialized-input identity plus native/library provenance.

All three bounded worker assignments and their review handoffs are complete. The next [NumPy advice ABBA proposal](2026-09-23-numpy-thp-loader-proposal.md) is banked with fixed inputs and acceptance criteria; its runner revision and quiet-slot admission remain. This is a saved plan, not a scheduled parent-agent restart or an executed experiment.
