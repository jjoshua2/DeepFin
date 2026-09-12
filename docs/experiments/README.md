# Experiment records

Start here for experiment planning and readouts. [Project guidance](../../CLAUDE.md)
holds durable constraints; [evaluation](../eval_protocol.md) explains which
measurements support which claims.

## Records

| Record | Scope |
| --- | --- |
| [Ceres materializer batch screen](2026-09-12-ceres-materializer-batch-benchmark.md) | Real 8,192-row outputs identical at 128/512; observed timings confounded by cache/order, no production speedup claim |
| [Ceres source-read cache](2026-09-12-ceres-source-block-cache.md) | CPU component benchmark: identical batches,1.74s→0.12s reads; future collector cache/timers, no GPU throughput claim |
| [Small G10 Ceres pilot](2026-09-11-g10-ceres-pilot.md) | Full 262,079-row/32-shard G10 Ceres bank qualified; existing policy/value/Q-draw and constraint diagnostics remain first4 only, no strength claim |
| [Saved SF negative constraints](2026-09-11-sf-negative-constraints-screen.md) | Completed 262,079-row SF/BT4 diagnostic: >300cp flagged mass mostly unscored; no blanket T300 promotion or Ceres strength claim |
| [Saved G10 value pilot and coverage](2026-09-11-g10-value-pilot-and-coverage.md) | 1,574,952 matched saved-SF rows; original run06/run07 large native BT4 WDL complete (4,548,347 rows with increments); consumer admission still pending; no training/strength result |
| [Weighted Ceres bootstrap preparation](2026-09-11-ceres-weighted-bootstrap.md) | Both CeresV25 fixed-400 matches complete: +6.79 Elo vs B100 and +25.83 vs B100V50, intervals include zero; retain B100 incumbent/Ceres alternative; no automatic extra games |
| [All-move SF downside candidate](2026-09-12-sf-allmove-downside.md) | One fixed >300 cp half-weight recipe; 8,192-row pilot passed; full 18.91M CPU preparation launched September 12 12:32 UTC under a 12-hour cap; no completion or strength result |
| [SF tactical guidance for BT4 policy](2026-09-10-bt4-sf-tactical-training.md) | Completed 400-simulation match: tactical recipe −13.6 Elo [−52.4,+24.9]; unresolved, retain B100 |
| [Deeper SF value census](2026-09-11-deeper-sf-value-census.md) | Complete: saved SF values pass label-change thresholds; 3,247 single-move exclusions and two malformed rosters explained; opt-in selector selected, no strength claim |
| [Saved adaptive SF value selector](2026-09-11-adaptive-sf-value-selector.md) | Opt-in deriver implementation and independent review passed; no real-corpus rewrite or training result |
| [BT4 target temperature and horizon: results](2026-09-10-bt4-target-temperature-horizon-results.md) | Complete: all three fixed deep matches; relative horizon change +5.664 points [−2.734, +13.867], direction and placement unresolved; [registration record](2026-09-09-bt4-target-temperature-horizon.md) retained |
| [B100 search-prior calibration](2026-09-09-b100-prior-calibration.md) | Complete: 0.7 versus 1.0 gives +16.30 Elo [−20.83, +53.80]; no follow-up selected, common prior 1.0 retained |
| [Bootstrap storage capacity](2026-09-08-bootstrap-storage-capacity.md) | Ongoing SSD-to-external transfer and verified reclamation for 100M; G20T1/G20T05/G50T05 reclamation complete; run04/run05 external copies verified and 870 closed shards reclaimed; metadata and unlisted tails retained |
| [Value collection and horizon readiness](2026-09-08-value-collection-and-horizon-readiness.md) | Full 18.91M WDL bank and V50 training complete; V50 arena started, no playing result |
| [B100 tactical policy readiness](2026-09-08-b100-tactical-policy-readiness.md) | Producer/admission merged; corrected bounded preparation independently reviewed; V50-first schedule, no tactical rewrite/training |
| [Next value and policy contrasts](2026-09-08-bootstrap-next-value-and-policy-contrasts.md) | B100V50 training and realized schedule complete; registered B100 arena started, no playing result |
| [SF-anchored value bootstrap](2026-09-08-sf-anchored-value-bootstrap.md) | Matched 128-row WDL diagnostic complete; B100V50 is the first value contrast |
| [Ceres teacher readiness](2026-09-08-ceres-teacher-readiness.md) | C3 CPU/sample evidence retained; approximate fixed32 policy + primary-value collector implemented, 8,192-row integrated cost pilot prepared, not launched |
| [Faster recipe matches](2026-09-08-faster-recipe-matches.md) | First prospective SPRT plus fixed deep probe completed in46m35s; speculative scheduling bottleneck identified; no controlled speedup claim |
| [BT4 adapter identity collection](2026-09-08-bt4-adapter-single-pass.md) | Exact identity parity and 34.1% CPU component reduction in one closed-shard pair; full-pipeline benefit unmeasured |
| [Qualified Soft-SF training sample](2026-09-08-soft-sf-qualified-training-sample.md) | Complete: SoftSF10 loses first-look low SPRT and fixed400 probe (−162.99 Elo); no clear depth recovery |
| [BT4 target geometry](2026-09-08-bt4-target-geometry.md) | Fixed 128-row training sample: C support/maxima and raw-cp controls; no temperature or strength selection |
| [BT4 raw-label legal-move reuse](2026-09-07-bt4-label-legal-reuse.md) | Exact policy parity; first optimized group completed 132,467 rows, throughput descriptive only |
| [G10 adapter/rank overlap](2026-09-08-g10-stage-overlap.md) | Adopted in Worker01 attempt; cache failure, no measured speedup |
| [G10 qualified inventory](2026-09-11-g10-qualified-inventory.md) | 16,404,093 accepted common-input rows; native WDL coverage remains separate; transfer training/strength unestablished |
| [G10 transfer readiness and alignment](2026-09-07-g10-transfer-readiness.md) | Historical preparation and alignment protocol; later inventory linked above |
| [G50 versus B100 policy dose](2026-09-08-bt4-g50-b100-dose-comparison.md) | Complete: G50 loses shallow SPRT and fixed deep comparison (−50.57 Elo); B100 retained, depth interaction unresolved |
| [Pure BT4 policy endpoint](2026-09-08-bt4-pure-policy-endpoint.md) | B100 beats H20: shallow H1, deep +46.42 Elo [11.23,82.58]; search interaction unresolved, same-seed development |
| [Bootstrap training-horizon readiness](2026-09-08-bootstrap-training-horizon-readiness.md) | CPU and bounded compiled-CUDA checks passed; full SF/B100 planner passed; training comparison not launched |
| [BT4 hybrid and global endpoints](2026-09-07-bt4-hybrid-endpoints.md) | H20 package complete: C100 +45.07, G100 +85.78, C400 +34.86 Elo; aligned search interaction unresolved; no promotion |
| [External archive and restore record](2026-09-07-storage-archive.md) | 38 verified archives, qualified local cleanup, retained dependencies and restore instructions |
| [Fresh paired BT4 confirmation](2026-09-07-bt4-fresh-confirmation.md) | Completed fresh seed-one comparison: C beats E by 37.67 Elo [18.72, 56.85] at 100 sims |
| [Direct SF-close versus global sharpened BT4](2026-09-06-bt4-direct-close-global.md) | Completed 100-simulation comparison favors C over G20T05 |
| [Sharpened stored SF top ties versus SF-close](2026-09-06-bt4-sharpened-ties-screen.md) | Completed E0T05 seed-zero epoch and direct 100-simulation screen favoring C20T05 |
| [BT4 bootstrap results](2026-09-06-bt4-bootstrap-results.md) | Completed screens and confirmation; C is the selected next-stage baseline |
| [50% sharpened global BT4 preparation](2026-09-06-bt4-global-dose50-preparation.md) | Reviewed CPU target preparation for a larger teacher dose; no training or matches queued |
| [Global BT4 mixing and search scaling](2026-09-05-bt4-global-search-scaling.md) | Matched SF control, full-distribution mixtures and 25/100/400-simulation playing screen |
| [BT4 near-tie policy targets](2026-09-05-bt4-joint-targets.md) | Superseded six-arm plan and completed control diagnostic |
| [Varying-horizon online controller](varying_horizon_online_controller.md) | Preregistration and staged evaluation of online search continuation |
| [Value head architecture](value_head_arch.md) | Historical April 2026 experiments |
| [Historical ledger](../experiment_ledger.md) | Frozen July–September 2026 record, including yardsticks, gotchas and recovery snapshots |

These descriptions identify the records, not the state of a running experiment.
Check subsequent readouts and the actual process/artifacts before resuming work.

The [BT4 evidence catalog](evidence/bt4-bootstrap/README.md) provides published
registrations, machine-readable readouts, game banks and provenance receipts.

## New experiments and follow-ups

Create `YYYY-MM-DD-slug.md` here and add a link above. Keep the preregistration,
amendments and subsequent readouts in that record. Existing individual records keep
their filenames. For a follow-up to an archived experiment, create a new record
linking the original evidence; do not append to the frozen ledger.

Include the hypothesis, baseline/control, realized settings and revision, deciding
command/threshold, uncertainty method, budget, horizon, confounds, artifact identities
and recovery plan. Add the readout against the precommitted rule, with its limits and
next decision. Publish compact supporting evidence with the record: realized settings,
readouts, review receipts and manageable game banks, with a source/hash manifest.
Keep large corpora, checkpoints and transient logs external and document their
identities and retrieval or restore locations. Historical absolute paths inside
frozen evidence are provenance, not portable command defaults.

## Finding prior evidence

The old ledger remains large and unsplit to preserve its contents, anchors and evidence
references. Search it; read only the matching entries and their later corrections.

```bash
rg -n -i 'YOUR_TOPIC_OR_CONFIG_KEY' docs/experiments docs/experiment_ledger.md docs/rl_loop_audit.md
```

Search by experiment name, config key, artifact path or checkpoint identity. Historical
“LIVE” labels and operating instructions are not current authority. An artifact absent
from a worktree may still exist under its original run directory.

For durable knowledge, use [model heads](../model_heads.md),
[target rebuildability](../target_rebuildability.md),
[the loop audit](../rl_loop_audit.md) and [operations](../operations.md), following their
links to supporting experiments. The `experiment-readout` Skill covers the reusable
analysis workflow.
