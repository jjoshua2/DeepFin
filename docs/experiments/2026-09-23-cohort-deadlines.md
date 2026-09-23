# Per-root asynchronous cohort deadlines

## Preregistration — September 23, 2026

PR5c on #862 at `57bb9636e3e6fcfb6832bb2cd494279bacae4d38`.
No merge, deployment, live training, GPU use or production configuration changes.
The feature is an opt-in CPU cohort control, not dynamic root admission or a
complete adaptive/self-play scheduler. Synchronous behavior stays unchanged.

Hypothesis: per-root deadlines can stop admissions and suppress expired results
without cancelling unaffected roots, losing prior accepted work, or refunding
physical inference already admitted. Timers continue during pending callbacks,
partial/buffered commands and control EOF. No performance or Elo hypothesis.

Contract: `deadline ROOT MS` sets a relative timeout measured from command
processing, not process launch; MS is 0..3,600,000. Existing deadlines can only
shorten. Completed/stopped/expired roots cannot be revived. Deadline checks occur
before gathering, after gathering/row compaction, and before each result's
normalization/backup. Individual encoding/normalization/backup/output operations
remain synchronous; this is not a hard-real-time guarantee. Final cohort reports
still follow physical drain. Expired dispatched rows remain charged and wasted.

Acceptance: inherited no-deadline synchronous/async serial-reference checks pass;
explicitly held evaluator tests prove selective expiry before release, queued-root
zero dispatch, retention of prior accepted trees, EOF/backlog timer progress and
cancelled-output failure detection. Preadmission compaction must preserve every
surviving input value and row/ticket identity. Malformed/extension/reactivation
commands and contradictory v2 deadline reports must reject. Deadline-free legacy
report schemas remain readable; new timing/expiry fields are explicit, not inferred.
Use exact pinned compiler and inherited numerical tolerances, without weakening
oracles. Real-model no-deadline parity is a separate gate from deterministic
cancellation/expiry injection; do not conflate the two.

Budget: bounded hosted fresh C generation, normal/UBSan deterministic checks,
cheap Python regressions and two-thread CPU model validation. Reuse the saved
untrained 5,043,005-parameter CPU fixture and singleton package; at most one
batch-four export from identical checkpoint/encoding. No training, game matches,
real GPU, model architecture or new permanent heavy CI step. Retain compact
source/build/test identities and reports; no private model or raw neural traces.

Recovery: preserve failed reports, fix only the relevant layer, and requalify
changed source. Full local generation exceeded its 80-second bound after type
checking; that is not successful code generation. Hosted compilation is required.
Publish a separate reviewable branch only after explicit qualification. Independent
review is unavailable; all author checks must be labeled self-review, not proof.

## Readout

Pending compiled qualification. Local full Bend type checking has passed.
EOF/held-callback and actual-model behavior remain unverified until executed.
