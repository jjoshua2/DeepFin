# Live cohort recovery and current-stack reconciliation

## Preregistration — September 25, 2026

Continue PR5d from the archived draft on #863. New parent is #874 at
`42d4c78ebfca102e146720c818f5dba9db61fd57`. Preserve its FIFO ordering,
completion notification, arena configuration and source-only tests. No merge,
live training, deployment, trained checkpoint or GPU use.

Hypothesis: persistent generation-tagged roots can safely reuse one bounded model
batch slot across add/replace/remove/cancel/deadline transitions. New generations
must not receive old control actions, values, or accounting; unaffected roots
must match the fixed-cohort serial reference. Invalid replacement must not cancel
its old root. Stop/quit abandon pending lifecycle changes; EOF drains them.

Use the exact pinned compiler without changing generated C. First qualify fresh
full native generation on a hosted runner, then normal/UBSan held-callback tests
at 146/175 planes and batch four, including generation reuse, EOF, failure, shared
batch replacement and complete-tree comparison. Inherit all existing expectations
and reference tolerances. Test nondefault arena capacity through actual live roots
and replacements, not only the environment parser. Ordinary fixed-cohort tests
remain independent. Reuse generated C only after checking its entire source hash.

Bound compute to one compiler job and two Torch threads, one source generation
per unchanged entrypoint plus small registry/mutation probes. No model export,
training or strength run in this initial recovery. CPU-model composition may be a
separate bounded follow-up, never inferred from callback success. Keep exact source,
commands, failures, reports and compact identities. Publish only tested source;
any outstanding gate is stated explicitly. Self-review unless an actual independent
review is obtained. No throughput, Elo or hard-real-time hypothesis is tested.

## Readout

Pending full native qualification of the reconciled candidate.
