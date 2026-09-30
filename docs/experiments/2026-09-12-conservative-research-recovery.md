# Conservative research recovery after the September 12 reboot

Status: guarded SF generation resumed at **19:07:34 UTC**, and the remaining
common-corpus native-BT4 WDL batch launched at **19:08:34 UTC**. The native-WDL batch subsequently completed and passed saved-output qualification for all 15 cohorts; SF generation remains a separately monitored job. No new training or playing result is claimed.

The [initial conservative restart](2026-09-12-sf-allmove-downside.md#september-12-reboot-and-conservative-restart)
preserved interrupted downside output; the reboot's cause remains unconfirmed.
[PR676](https://github.com/jjoshua2/DeepFin/pull/676) records that recovery,
[PR677](https://github.com/jjoshua2/DeepFin/pull/677) adds execution concurrency
control, and [PR678](https://github.com/jjoshua2/DeepFin/pull/678) records the
completed 790,282-row native-WDL unit. This entry consolidates the next launches.

## SF generation: fewer active workers, same scientific partitions

Before resuming, 34 open-tail/state files totaling 164,257,369 bytes were copied
and verified; original files were unchanged by preservation. The frozen runtime
`289693b142d3459f448d30b249ae19b342d8226a` adds only the execution-concurrency
change to the historical generator runtime. It does not adopt the separate
future-roster guard or alter worker identities, game partitions, seeds, search
settings or per-worker deduplication capacity.

Run06 retains 12 logical workers but runs at most two concurrently; run07 retains
four logical workers but runs at most one. Each worker retains its 2,000,000-entry
deduplication cap. An authenticated process-group monitor checks Linux available
memory and free disk every two seconds, requiring 32 GiB and 150 GiB respectively.
On low available memory, it signals both owned generator groups before waiting
for cleanup.
This is sampled headroom protection, not a hard RSS cap or a measured warmup peak.
Existing STOP markers remain authoritative. Completed new corpus rows have not
been counted in this launch record.

## One batch to finish native WDL for the accepted common corpus

All 15 remaining cohorts passed the existing metadata admission once. They contain
**11,065,464 positions in 1,358 derived shards**, disjoint from the already complete
5,338,629 direct-G10 positions. Completing the batch would cover all 16,404,093
accepted common positions. Existing raw-native labels without a derived-row join
are not silently credited toward that total.

The unchanged collector runs cohorts sequentially with batch 128, two threads,
CPUs 2–3, an 8 GiB ORT allowance and its mandatory GPU lease. Each cohort retains
one fresh complete output directory and its original source/teacher identity.
The supervisor retains 32 GiB available-memory and 150 GiB disk guards; the ORT
allowance is neither a total-device nor CPU RSS cap.

The single 7,200-second allocation includes cleanup and 300 seconds reserved for
one saved-output verification pass over actually completed cohorts. Child limits
are calculated from cohort size and remaining time. If the next complete cohort
cannot fit the launch allowance, stop between cohorts and report the completed
subset. First integrity or child failure stops the batch, preserving artifacts;
there is no automatic retry, dropped cohort or budget extension. All five saved
arrays are checked once for completed cohorts within the same overall deadline.
Separate provenance-manifest/consumer checks still precede any training admission.

Around 19:09 UTC, the parent's initial sample recorded about 90 GiB WSL memory
available, 7.3 GiB used, no swap use and GPU usage 5,573 MiB / 61%. Summed owned
process RSS was 4,707,528,704 bytes and can double-count shared mappings. These
are early warmup observations, not peaks or proof of safe steady-state usage.

[Compact evidence](evidence/conservative-research-recovery-20260912.json) retains
all launch, preservation, plan and independent-review pins, exact cohorts and
parent-owned handles. No live process polling or payload scans were needed to
publish this record. Results will be appended after actual completion.

## Completed native WDL for all accepted common G10 positions

The batch finished all **11,065,464 positions across 15 cohorts and 1,358 shards**.
Together with the previously qualified 5,338,629 positions, direct native-WDL
coverage now includes all **16,404,093 accepted common G10 positions** in 20
cohorts. This is the frozen accepted pool, not the full planned 100M corpus.

Collection took 5,412.17 seconds (90m12s). The automatic saved-output check then
passed before the original two-hour deadline; total time through qualification
was 5,508.47 seconds (91m48s). All five arrays were read once across every new
shard—708,189,696 decoded bytes—and passed source/teacher/row bindings, stored
hashes and finite float32 probability checks. Maximum unit-mass error was
1.47535e-7. Parent root83736/observer71 closed with exit zero. A subsequent
compact review checked all 15 invocation bindings and 1,358 proof records without
reading arrays again. No cohort was dropped or retried, and no further GPU batch
was automatically queued.

[Full cohort counts and compact completion evidence](evidence/common-g10-native-wdl-complete-20260912.json)
retain all outputs, original qualification identities, completion pins and runtime
costs. Nineteen source cohorts have one complete native-WDL output directory and
can follow the existing historical manifest route. The older run06 common-large
cohort instead spans four independently qualified output units; the current
reader requires one `wdl_dir` and matching invocation/attribute namespaces. A
small explicit multi-output admission extension is needed to reuse those four
units together without inventing a merged provenance identity.

No new manifest or writer admission was run for this completion record. Those
steps must preserve actual producer/source/model/head/selection bindings and the
existing content/feed checks. Completed teacher coverage does not establish a
new value target, qualified training recipe or playing-strength gain.
