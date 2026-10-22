Project guidance lives in [CLAUDE.md](CLAUDE.md). Read it for every task.
Keep this file as a pointer so shared guidance cannot drift between agents.

For backlog handoffs and reports, follow the durable-ledger procedure in
[execution cadence](docs/agent_execution.md#durable-backlog-handoff-audit).
Run `python scripts/backlog_handoff.py <existing-ledger.json>` at these boundaries;
an audit failure requires reconciliation, and a clean audit grants no authority.
Generate local crosswalk reports with `scripts/current_backlog_report.py` using a
captured input hash and new output directory. Pending reports retain every ID;
manual parent chat and hosted Atlas reports remain outside this consumer's gate.
