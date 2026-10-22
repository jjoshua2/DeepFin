Read the repository [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md),
and [durable-ledger procedure](../../docs/agent_execution.md#durable-backlog-handoff-audit).

These fixtures exercise `scripts/backlog_handoff.py` using standard-library unittest.
Keep them synthetic: no project/model imports, native builds, WSL, live ledger
mutation, workload launch or scheduler. Run only within coordinated CPU/IO bounds.
Preserve denial, timeout and failure fixtures and verify report refusal as well as
success. Record tests actually executed separately from source review. A synthetic
report consumer is not evidence that the parent workflow invokes the hook.
