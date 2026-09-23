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
