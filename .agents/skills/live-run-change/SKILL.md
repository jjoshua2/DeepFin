---
name: live-run-change
description: Plan and carry out a configuration change, deployment, pause, or recovery for an existing long-running job while preserving its state and neighboring work.
---

# Live run change

Read the project's operational constraints and the relevant operating procedure.
Identify the actual process, owner checkout, loaded revision, configuration, output
locations, monitors and stop markers. A branch name or saved PID alone is not identity.
Inspect active work before deciding it is stalled or replaceable.

Make the intended change concrete in an isolated checkout or configuration copy.
Trace each changed setting through parsing, validation, propagation and the running
consumer. Determine whether adoption is live, restart-only, or unsupported; check
removal semantics as well as additions. A parser-only dry run cannot prove runtime
safety or that a value is used.

Choose recovery state according to what persists: weights, optimizer/controller,
replay, manifests and configuration may all matter. Record the snapshot and its
identity outside locations managed by pruning. For populated generation jobs, retain
frozen resume settings; prefer a separately identified companion when scaling would
change the existing corpus's interpretation.

Use the project's graceful pause/stop mechanism and respect deliberate stops. Keep
neighboring jobs and monitors intact. Perform actions already authorized for the task;
ask only for a genuinely new scope, resource commitment or destructive action.
A deployment approval, when needed, should concern a prepared, reviewable change.

After adoption, verify effective values and behavior from the running job, not merely
from the edited file. Record the loaded revision, resumed state, observation and any
recovery action. Stop escalating when the agreed resource or error limit is reached;
a failed recovery attempt does not authorize broad process cleanup.
