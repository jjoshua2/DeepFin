# SF-free E admission and paired seed replication — September 22

E preparation completed all 58,090,688 rows, 7,108 shards and 35 cohorts. Its full
storage qualification is pinned to
`4325d6bb319d278d5c844b0e16a82a3211dab1224d9c6443d0bbc25b82af1f03`.
E preserves D policy and replaces equal-third SF/BT4/Ceres main value with half
native BT4 and half calibrated Ceres. Native BT4 is read directly from its bank.

Five jobs are frozen for the [next-day preregistration](https://github.com/jjoshua2/DeepFin/blob/89ed8c1d05b4dc9948ad3e3468429121a537adfc/docs/experiments/2026-09-22-factorial-readout-next24h.md):
E121, E121 versus D121, D122, E122, and E122 versus D122. Each training job keeps
the original model/config/runtime, batch512, one fresh epoch and 113,459 updates.
D122 establishes the seed122 initial tensor identity; E122 must match it.
The second pair is fixed regardless of the first match's outcome. Each arena is
256 games/128 opening pairs at 400 simulations and prior temperature 1.0; seeds
2026092101 and2026092201 respectively. Per-job schedule hashes must match their
realized hashes; cross-arm hashes include different corpus paths/content and
are not asserted equal.

Expected active work is 47,000 seconds per training and 1,800 per arena: 40.17 hours
for these five jobs, plus the separately scheduled storage retry. Each full
training job retains the 20-hour active cap, two-hour cumulative disk-pause budget
and 600-second wrapper allowance, with 79,800-second supervisor cap. Arenas have
3,660-second supervisor caps. These are safety bounds, not truncated epoch targets.
The parent scheduler owns enqueueing, deadline admission and launch. This source
publication by itself does not claim any E training or arena completed.

All training uses frozen `/tmp/deepfin-factorial58-runtime` at
`502cd02e072471c901255f3fdb580d6ea7b826d0`; neither packed storage nor loader reuse
is adopted. The frozen arena runtime is unchanged. New operation-local wrappers
mask termination signals while reaping owned children and writing receipts;
completion requires exact rows/updates, no repeated same-game rows within a batch,
matching planned/realized schedule digests and finite recorded training-window
losses. The actual initialization is checked after startup. The shared GPU lease,
STOP checks, reserve thresholds, hourly recovery and existing process ownership
rules remain in force. Frozen deployed runtime files were not edited.

The [independent final review](evidence/sffree-E-admission-20260922/factorial-next24h-final-admission-review-20260922.json)
approves all five final plans/descriptors. CPU-only validation passed E121's full
receipt admission, all five registered descriptor checks, and refusal of the four
future jobs before GPU access while dependencies are absent. Fourteen synthetic
replica admission checks cover correct target choice and matching initialization,
changed seed/command/runtime, incomplete prerequisites, failed terminals, repeated
games and schedule/count mismatches. They mock only the already separately exercised
E121 admission and use the actual paired-bank/hash validator. No training was
launched by those checks. No new GPU throughput or playing-strength result is claimed.

The [snapshot manifest](evidence/sffree-E-admission-20260922/manifest.json) publishes
exact operational runners, admission modules, plans, descriptors and test scripts.
These are historical frozen operation sources, not a general supported API. Shared
artifacts live at
`~/chess-artifacts/operations/factorial58-sffree-training-20260922/`.
The [reviewed queue metadata](evidence/sffree-E-admission-20260922/factorial-next24h-reviewed-queue-items-20260922.json)
uses the scheduler's `estimated_active_seconds` field; the earlier standalone
queue-item snapshots retain their original `estimated_seconds` spelling as provenance.

Documentation path and identity checks pass (46 tests, two CPU threads).
Whole-repository Ruff and Vulture pass; Basedpyright findings are adjudicated
against the unchanged source baseline: all 14 diagnostics exactly match the earlier
11+3 baseline logs. The [lint adjudication](evidence/sffree-E-admission-20260922/E-publication-lint-adjudication-20260922.json)
and complete validation logs are banked beside the source snapshot. Inference follows the published preregistration: report both seed
contrasts and their equal-weight score-margin average; concordant positive signs
favor E provisionally, concordant negative signs favor D, and mixed directions
remain unresolved. No outcome-driven extra games or automatic deployment.
