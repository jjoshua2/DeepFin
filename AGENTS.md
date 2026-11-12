Project guidance lives in [CLAUDE.md](CLAUDE.md). Read it for every task.
Keep detailed shared guidance in CLAUDE.md; the tablebase requirement below is
explicit here so every agent sees it.

Backlog handoffs and reports follow the
[durable-ledger procedure](docs/agent_execution.md#durable-backlog-handoff-audit).

## Required tablebase use

Use available tablebases wherever supported in search, self-play, labeling and
evaluation, including diagnostic and teacher-reference runs. The owner's current
inventory is six-piece Syzygy, not seven-piece. Keep probing enabled for eligible
descendants: a root with 7-10 or more pieces can reach six-piece positions during
search; root piece count is not a reason to disable tablebases.

Before a new run, verify the actual readable WDL/DTZ inventory, effective engine
path and probe settings, and that each applicable engine/worker receives them.
Check a known covered position to establish successful probing, then retain actual
probe/hit evidence with the run settings. Zero hits in a particular search may be
legitimate; do not infer disabled probing from zero alone or claim coverage from a
configured path alone. Respect WDL/DTZ and fifty-move-rule semantics.

Do not silently omit tablebases for convenience. Record concrete technical
exceptions (missing required files, unsupported position/format or an engine path
that cannot probe), their affected scope and the resulting evidence limitation.
An intentional tablebase-off control needs explicit experiment authorization and
must be labeled as such, not substituted for the tablebase-enabled reference.
Do not change live jobs or frozen historical evidence to satisfy this rule:
coordinate any runtime adoption with the existing owner under the normal
authorization and preservation rules in CLAUDE.md.
