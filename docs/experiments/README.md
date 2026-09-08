# Experiment records

Start here for experiment planning and readouts. [Project guidance](../../CLAUDE.md)
holds durable constraints; [evaluation](../eval_protocol.md) explains which
measurements support which claims.

## Records

| Record | Scope |
| --- | --- |
| [BT4 raw-label legal-move reuse](2026-09-07-bt4-label-legal-reuse.md) | Exact policy parity; one CPU postprocessing observation took 49.08% less time, with end-to-end and artificial-logit limits |
| [G10 transfer readiness and alignment](2026-09-07-g10-transfer-readiness.md) | 17 malformed policy rows and 2,850 missing results; value follow-up adds no exclusions; bounded fix and corrected preparation registered |
| [BT4 hybrid and global endpoints](2026-09-07-bt4-hybrid-endpoints.md) | H20 selected next; subsequent target, horizon and search tests chosen adaptively |
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
