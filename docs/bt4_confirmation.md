# Fresh-seed BT4 confirmation launcher

`scripts/bt4_confirmation.py` prepares one direct confirmation match after a
separate experiment record identifies its candidate, reference and budget. It
selects no recipe and launches nothing by default. This lane uses the qualified
Python 3.10 / Torch 2.11 CUDA runtime, trainer and history-bearing confirmation
opening bank; it is not an environment upgrade.

Create an immutable preregistration outside the new state directory. Supply a
JSON manifest using the following shape. Angle-bracket values are deliberately
unselected placeholders, not executable defaults:

```json
{
  "schema": 1,
  "repository_root": "<absolute original repository root>",
  "state": "<absolute NEW confirmation state directory>",
  "training_seed": "<registered nonzero integer seed for BOTH arms>",
  "arena_seed": "<registered integer arena seed>",
  "sims": "<registered 25, 100 or 400>",
  "gpu_budget_seconds": "<registered total seconds>",
  "cpu_stage_caps_seconds": {"schedule": 1800, "readout": 120},
  "stage_caps_seconds": {
    "reference_train": "<seconds>",
    "candidate_train": "<seconds>",
    "arena": "<seconds>"
  },
  "reference": {
    "role": "<S0, E0, E0T05, G20T1, G20T05 or C20T05>",
    "corpus": "<absolute published source corpus>",
    "run": "<absolute NEW reference run directory>",
    "derive_sha256": "<derive_targets_summary.json SHA256>",
    "mix_sha256": "<bt4_policy_mix_summary.json SHA256; JSON null for S0>"
  },
  "candidate": {
    "role": "<different selected recipe role>",
    "corpus": "<absolute published candidate corpus>",
    "run": "<absolute NEW candidate run directory>",
    "derive_sha256": "<derive_targets_summary.json SHA256>",
    "mix_sha256": "<bt4_policy_mix_summary.json SHA256; JSON null for S0>"
  },
  "runtime_manifest": {
    "path": "<original repo>/scratchpad/bt4_joint20/sf_close_run02/sf-close/manifest.json",
    "sha256": "<independently verified runtime manifest SHA256>"
  },
  "preregistration": {"path": "<immutable record>", "sha256": "<record SHA256>"},
  "openings": "<original repo>/scratchpad/bt4_joint20/confirmation_openings_v1/openings.fen",
  "schedule_verifier": "<original repo>/scratchpad/bt4_joint20/confirmation_tools/verify_matched_schedule.py",
  "launcher_sha256": "<SHA256 of scripts/bt4_confirmation.py from this reviewed checkout>"
}
```

Replace numeric placeholders with JSON numbers. The total cap must cover all
three GPU stage caps, each greater than 30 seconds. Supply the CPU caps exactly as
shown; a manifest missing them is rejected. Every stage cap includes 30 seconds of
termination grace. The launcher binds each named role to its exact qualified
mix-summary hash (and S0 to its unmodified source-summary hash). E0 is the raw
BT4 stored-top-tie recipe; E0T05 sharpens BT4 to temperature 0.5 within the stored SF
maximum ties. These stored policy ties need not be ties in the underlying SF score.
C20T05 is the separately qualified sharpened SF-close recipe.
A different temperature, algorithm or source is a new recipe requiring separate
qualification, even if its top-level role/alpha fields match. The launcher verifies
runtime identities, qualified opening-bank bytes, and the preregistration hash.
The runtime root remains the original repository's frozen `.dev/worktree/wise-cloud`.
`CHESS_ANTI_ENGINE_LIVE_CONFIG` is set to that runtime's `configs/pbt2_small.yaml`,
overriding an inherited configuration path. The pinned completed C100 readout at
`scratchpad/bt4_joint20/sf_close_run02/completed_s100_readout.json` supplies the full
qualified search configuration; this is a required input alongside the opening bank
and metadata schedule verifier. These host artifacts are not bundled with the tool.
Keep this launcher checkout and all pinned inputs unchanged throughout execution.

```bash
# Prints exact commands, seeds, paths and budgets. Creates no state or outputs.
python scripts/bt4_confirmation.py --manifest /absolute/confirmation.json

# Only after the separate registration, scheduling decision and review:
python scripts/bt4_confirmation.py --manifest /absolute/confirmation.json --execute
```

Execution trains **both** roles from scratch with the same fresh seed and
qualified exact-epoch recipe. It rejects every preexisting state/run directory,
including apparently completed runs; there is no seed-zero adoption or automatic
partial restart. A CPU metadata verifier must establish both completed actual
staging chains, equal fresh-seed epoch counts and a common source-normalized
schedule before the arena can start. The seed-zero batch count is not reused.
Completion also requires 88-step training windows, their count derived from the fresh seed's planned batches, finite loss and
gradient norms, zero skipped or retried batches, and the exact last-checkpoint path
and hash.

GPU stages share `scratchpad/gpu0_experiment.lock`, wait for existing GPU users,
recheck pins after the wait, and inherit the lease into their owned process group.
CPU schedule verification and the final qualified readout release that lease.
The arena uses 1,000 games, 500 fixed history-bearing opening pairs, explicit
prior 1.0 on both sides, and `--openings-fen` with neither `--openings` nor
`--opening-plies`. Checkpoint hashes and the realized header are checked, including
equality of both sides' complete search settings to the qualified C100 settings.
The paired readout uses the explicitly named candidate and reference; E0T05 can
appear on either side. It reports a direct comparison at the chosen budget, without
inventing an S0 comparison or a search-scaling result.

The reader now defaults to prior temperature 1.0. Reading older softened-prior
screen banks requires explicit `--prior-temperature 1.5`; confirmation always
requires prior 1.0. The reader retains the separately qualified global, SF-close
and calibration profiles for those completed banks.

The launcher retains per-stage commands, stdout/stderr, completion evidence and
closed success/failure GPU charges. A state `STOP` file, SIGINT/SIGTERM, disk
reserve breach or elapsed stage cap stops its owned child group and preserves
partial files. The reserve is 150 GiB, with 10 GiB additional space required before
starting. GNU `timeout` supervises each GPU stage using its remaining cap and each
CPU stage using its registered cap. It sends TERM 30 seconds before that limit and
then KILL if needed. The supervisor and owned child group retain the inherited
lease even if the coordinator is killed, and the independent timeout bounds their
lifetime. Such a kill still leaves an unresolved charge: verify that the owned stage
has stopped and reconcile its actual resource use before separate recovery. Do not
erase partial state and rerun to make a failed attempt appear fresh.

A completed readout does not establish an optimal recipe or trigger promotion.
Apply the separately registered rule after independent provenance/readout review.
Opening-bank freshness remains conditional on the banked exclusions; account for
any additional screening openings before registering its use. Metadata schedule
verification is not full feature/target payload equality or a per-row trace.
