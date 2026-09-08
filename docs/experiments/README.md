# Experiment records

Start here for experiment planning and readouts. [Project guidance](../../CLAUDE.md)
holds durable constraints; [evaluation](../eval_protocol.md) explains which
measurements support which claims.

## Records

| Record | Scope |
| --- | --- |
| [Faster recipe matches](2026-09-08-faster-recipe-matches.md) | First prospective SPRT plus fixed deep probe completed in46m35s; speculative scheduling bottleneck identified; no controlled speedup claim |
| [BT4 adapter identity collection](2026-09-08-bt4-adapter-single-pass.md) | Exact identity parity and 34.1% CPU component reduction in one closed-shard pair; full-pipeline benefit unmeasured |
| [Qualified Soft-SF training sample](2026-09-08-soft-sf-qualified-training-sample.md) | 4,096 rows across 64 shards; 10cp matches C entropy most closely among four registered candidates, no training/strength selection |
| [BT4 target geometry](2026-09-08-bt4-target-geometry.md) | Fixed 128-row training sample: C support/maxima and raw-cp controls; no temperature or strength selection |
| [BT4 raw-label legal-move reuse](2026-09-07-bt4-label-legal-reuse.md) | Exact policy parity; first optimized group completed 132,467 rows, throughput descriptive only |
| [G10 adapter/rank overlap](2026-09-08-g10-stage-overlap.md) | Adopted in Worker01 attempt; cache failure, no measured speedup |
| [G10 transfer readiness and alignment](2026-09-07-g10-transfer-readiness.md) | 5,276,543 common rows prepared after completed Worker01 recovery; original failure preserved; combined training qualification and strength remain unestablished |
| [G50 versus B100 policy dose](2026-09-08-bt4-g50-b100-dose-comparison.md) | Prospective: fresh seed0 original epoch,50% versus100% global BT4; qualification/schedule pending, no launch |
| [Pure BT4 policy endpoint](2026-09-08-bt4-pure-policy-endpoint.md) | B100 beats H20: shallow H1, deep +46.42 Elo [11.23,82.58]; search interaction unresolved, same-seed development |
| [Bootstrap training-horizon readiness](2026-09-08-bootstrap-training-horizon-readiness.md) | Real tiny two-epoch CPU path checked on the older Torch environment; CUDA/full-corpus qualification and finalist selection remain |
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
