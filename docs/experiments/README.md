# Experiment navigation

[CLAUDE.md](../../CLAUDE.md) holds project constraints;
[the evaluation protocol](../eval_protocol.md) defines how to choose measurements;
[the ledger](../experiment_ledger.md) records actual decisions. Dated entries, plans
and handoffs describe their own run and revision, not necessarily the current system.

Use targeted lookup before proposing or resuming work:

```bash
rg -n '^##|Protocol gotchas|Revert points' docs/experiment_ledger.md
rg -n -i 'YOUR_TOPIC_OR_CONFIG_KEY' docs/experiment_ledger.md docs/experiments docs/rl_loop_audit.md
```

Read the matching entries, their referenced evidence and subsequent readouts or
reverts. Search by experiment name, config key, artifact path or checkpoint identity;
a last-mentioned “LIVE” status alone is insufficient. Confirm current state against
the owning checkout, process and output manifests. An absent artifact in a worktree
may still exist under the original run's directory.

Useful entry points:

| Question | Reference |
| --- | --- |
| What was tried, judged or reverted? | [Ledger](../experiment_ledger.md), including its Protocol gotchas and Revert points |
| Does the measurement describe the intended pipeline? | [Loop audit](../rl_loop_audit.md), Method rules and relevant stage |
| Does a result establish target quality or strength? | [Evaluation protocol](../eval_protocol.md) |
| What did each head learn? | [Model heads](../model_heads.md) |
| How are target observations preserved for reanalysis? | [Target rebuildability](../target_rebuildability.md) |
| How should a live change be adopted or recovered? | [Operations](../operations.md) |

For a new entry, record hypothesis, source-qualified artifacts, baseline/control,
realized settings and revision, deciding command/threshold, uncertainty, resource
budget, horizon, confounds and recovery state. Add the readout to that record when
available. Keep raw observations outside git where appropriate and link their paths;
keep enough provenance to recover their meaning without rerunning the experiment.

The `experiment-readout` Skill supplies a reusable workflow. Root `prompt.md`,
`spec.md` and `future_ideas.md` retain early design history; consult current source
before treating their architecture or proposed next steps as applicable.
