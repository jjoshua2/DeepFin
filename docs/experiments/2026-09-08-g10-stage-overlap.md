# G10 adapter/rank overlap preparation

## Adoption status — September 8

Worker01 adopted the explicit overlap mode after derivation and its storage
snapshot. The attempt then stopped at rank-cache admission; adaptation, ranking
and common-input qualification did not complete. This establishes operational
adoption, **not a measured speedup**. The original failure and retained derivation
status are recorded in the [Worker01 readout](2026-09-07-g10-transfer-readiness.md#worker01-stopped-at-rank-cache-admission--september-8).

## Historical implementation and planning snapshot

The common-input runner now supports explicit `overlap_adapt_rank: true` after
derivation and the immutable storage snapshot. Raw-BT4 adaptation and phase-zero
d9 ranking read the same frozen inputs and write separate sidecar directories.
Final common-input qualification still waits for both producers and retains all
source, survivor-complement and unchanged-storage checks. Existing manifests
remain serial by default. This is implemented tooling, not an operational launch.

The [banked timing analysis](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_stage_overlap_readiness_v1/readiness.json)
measured adapter stages at 1,000.30/1,003.46 seconds and rank stages at
679.01/673.85 seconds in two completed source lanes. Each averaged approximately
one CPU core. Replacing their serial sum with the larger duration would shorten
those historical lane critical paths by approximately 25%, **if neither stage
slowed down under concurrent execution**. This is an idealized estimate, not a
measured speedup or a promise for different rows or host load.

The [independent dependency assessment](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_adapter_rank_overlap_design_v1.json)
found no producer dependency after the snapshot and adapter-manifest creation.
The pair shares its lane's existing two-core affinity, two numeric threads,
priority and GNU timeout process group. One polling loop owns both wrappers;
a failure cancels the sibling and propagates to existing lane-group cleanup.
Qualification cannot start after only one producer finishes. Per-stage logs,
resource receipts and the effective overlap setting remain explicit.

Summed historical adapter/rank peak RSS is approximately 2 GiB per lane, or 4 GiB
for two lanes. These sums are planning references, not measured simultaneous
peaks or RAM caps. The existing 8 GiB setting limits sampled output/cache bytes;
it does not limit resident memory. Shared-disk contention and memory interaction
remain unmeasured. No active runtime was changed and no new corpus processing,
teacher inference, policy mixing or training was launched for this change.

See the [runner guide](../common_input_batch.md) for the manifest and ownership
contract. A later separately frozen batch can opt in and record actual resource
costs; this note selects no recipe or GPU experiment.
