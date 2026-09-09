# Near-term compute and space schedule

September 8, approximately 22:53 local time. This is a relative decision schedule,
not a launched queue. `schedule.json` binds the receipts and source inspected;
its estimates are not completion promises. No corpus/model scans or operations ran.

**Keep V50 first.** Finish the active original-SF WDL remainder and G20T05 archive
without interruption. Once full WDL completion is admitted and the archive releases
the preparation lock, start the reviewed V50 rewrite. Do not put the potentially
eight-hour tactical pass into a short gap immediately before V50 becomes ready.
V50 and tactical use the same nonblocking `preparation.lock` as the archive;
different CPU affinities do not make concurrent materializations admissible.

| Relative slot | Work and budget | Relevant interference |
| --- | --- | --- |
| Current | WDL remainder: GPU lease, CPUs 2,3, four-hour whole invocation. Archive: CPUs 0,1, preparation lock, four hours inclusive. | Preserve both and the waiting raw-label group. Remainder has only start receipts at observation; no full bank is inferred. |
| After both complete | V50 CPU rewrite: CPUs 2,3, two threads, 12 GiB address space, 32 GiB sampled output. | Six hours bounds the producer including kill; supervisor metadata pre/post is outside that timeout. No measured full V50 ETA. |
| Natural GPU boundary during rewrite | One already-reviewed Ceres probe: CPUs 0,1, 900 seconds inclusive, 128 MiB output. | Requires free lease and no competing GPU compute. ORT arena 8 GiB is distinct from sampled device12 GiB/RSS12 GiB limits; host available minimum16 GiB. |
| After V50 publication and prospective planning | V50 original training, then registered low100/protected400 package. | Training16200s plus two5400s arenas =27000 GPU-stage ceiling, not end-to-end. Original16+16 worker configuration remains. |
| During V50 training, if headroom permits | Tactical CPU rewrite after V50 prospective CPU planning finishes. Author proposes CPUs0,1/two threads, eight hours whole invocation,32 GiB output, no hard RSS quota. | Preparation lock is available after V50 rewrite. Training's own startup/planning and16 loaders can still compete for memory/I/O; defer if they cannot coexist. Preserve realized verifier CPU0,1 needs as training ends. |
| After actual tactical qualification/schedule | Separate tactical comparison against unchanged B100. | Value and policy interventions stay separate; tactical need not wait for V50 to win, and targets are not combined prematurely. |

## Ceres opportunity is real but conditional

Host observation confirmed raw-WDL driver **423973**, direct child **477385**, in
the preserved PR580 overlay. Its current policy+value invocation is already
waiting for the same GPU lease: batch1024, threads16, up to16 shards,24 GiB GPU
allowance. Source retries nonblocking lease acquisition; this is not a guaranteed
FIFO queue. After the current group exits the driver sleeps15 seconds before
starting another group. Ceres can fit at a genuinely idle boundary during the
longer CPU rewrite, but cannot assume priority immediately after the remainder.
Let already-queued useful labels finish. Its own busy-lease refusal is not a
reason to consume one-shot output state early, pause the driver, or retry a probe.
The owner/queue evidence is in `raw_label_owner_observation.json`; no pause or
process alteration was performed.

## Measured costs and SSD ledger

Related saved measurements: SoftSF rewrite **4.82h**, peakRSS **650.45 MiB**, output
**11.686 GiB**; SoftSF original training **2.64 GPUh**; G20T1 copy **49.63min**;
G50/B100's two arenas **41.06 GPUmin**. The remainder's previous publication-span
proxy was **2.87h**, excluding startup/wait/teardown. None establishes current
V50/tactical runtime or a guaranteed finish time. Tactical's author supplied its
proposed budget separately; no in-progress files were edited.

The latest confirmed parent host observation (about02:55 UTC) has **358 GiB free
on SSD and7.5 TiB external**, after G20T05 staging growth. This supersedes the
historical372 GiB excerpt and leaves208 GiB above the150 GiB floor. Charge full planning
allowances again, conservatively even where some bytes may already be present:
32 GiB V50 +32 GiB tactical +24 GiB archive staging +4 GiB full WDL physical
estimate +0.125 GiB Ceres output =**92.125 GiB**. This leaves roughly265.875 GiB
free,115.875 GiB above the floor, **before unbudgeted growth and caches**.
Each32 GiB recipe figure is a **sampled allocated-size cap**, not expected actual
output and not space already reserved or allocated. The full-cap charges are
conservative planning arithmetic. The WDL4 GiB allowance is a rounded estimate,
not a physical quota; its actual
2 GiB guard is logical size. Sampled caps can overshoot. Two B100-sized recipe
copies would be about27.18 GiB, but this is a compression proxy, not a measured
V50/tactical size. Credit no future G20T05 deletion and do not credit G20T1 twice.
Raw generation, labels, training checkpoints/caches, staging and other owners
still need headroom; this is not a100M capacity proof.

Retain original SF, B100, the original raw source needed for tactical joins,
complete WDL bank, selected G10 inputs and all checkpoint/summary/schedule/arena
proof. Keep candidate datasets through realized verification and finalist choice.
Continue cold-policy archival/reclamation with consumer checks; the26 already
archived pools are mostly restart/checkpoint material, not a wholesale deletion
list. Preserve retained summaries and restore identities.

## Protect the next scientific questions

After these two contrasts, prefer evidence that distinguishes transferability
from more local target tuning: a matched fresh-seed replication of the useful
finalist; a matched two-epoch comparison on the same qualified new runtime;
and one targeted same-checkpoint prior calibration using the existing fixed-N
reader. The compiled four-update probe and full two-epoch planner already pass,
but do not prove full training behavior. SF is the source/schedule anchor, not a
mandatory second horizon finalist. V10 is an optional dose question, not a ladder.

Advance the9,298,514 prepared G10 common rows toward an actual source/phase-qualified
transfer recipe and schedule with a staged100M storage budget. They are not yet a
finalist training corpus. A successful Ceres parity/cost probe can support a later
retained-SF teacher-diversity contrast, not an automatic bulk-label or strength
claim. These are conditional next choices, not a mandatory queue.
