# Bounded bootstrap queue operator

`bootstrap_experiment_operator.py` replaces the host-local autonomous-loop operator;
`supervise_bootstrap_queue.sh LOOP RUNTIME` chains terminal receipts without model
polling. Adoption is explicit: stop only the old supervisor, preserve its running
child, and pass the existing loop directory to the replacement. Existing running
arenas can be harvested; their runtime is not changed by replacing the supervisor.

The queue and STATE.json retain their existing schemas. Each launch records intent
atomically before spawning. A lost wrapper or interrupted launch blocks further
work for recovery; it does not authorize restarting or killing the child. The
supervisor never group-kills a PID copied from a potentially stale file. A failed
GPU query blocks launching. Every launch includes two-thread numeric and compiler
limits and reserves its whole horizon plus TERM/KILL grace before the global
deadline. Queued UCI comparisons require `qualified_uci_profile: true` following
actual profile qualification. Held items are ignored.

A registered existing collector uses an item with `kind: registered_command`,
`command_file`, `command_sha256`, `out`, and `max_seconds`. Its hashed JSON is:

```json
{
  "argv": ["/bin/bash", "/absolute/reviewed-command.sh"],
  "cwd": "/absolute/working-directory",
  "env": {},
  "pins": [{"path": "/absolute/reviewed-command.sh", "sha256": "..."}],
  "completion": {"path": "/absolute/qualification.json", "status_key": "status", "expected": "PASS"}
}
```

Pins are small command/manifest files, limited to 8 MiB each; model files are not
rehash inputs here. The descriptor hash binds exact argv, cwd, environment and
expected qualification. Existing completion files are refused. `max_seconds` is
seconds until TERM, followed by at most 30 seconds until KILL; a 3,600-second task
therefore uses 3,570. Admission additionally allows 10 seconds to write receipts.
Collectors must themselves preserve their registered internal resource and
ownership limits. The outer process group controls the directly launched process
and inherited group, not arbitrary separately detached jobs.

Arena success requires terminal exit zero, nontruncated result, exact requested
actual game and pair counts, complete color swaps, matching pair openings and
matching score. UCI success also checks the actual unique game indices and W/D/L
counts. Registered collectors require exit zero and exact qualification status.
Partial work and failed artifacts are retained for inspection.
