# CPU target preparation during teacher collection

`bootstrap_preparation_overlap.py` permits CPU preparation of already pinned
teacher cohorts while a GPU collection queue continues. It calls the existing
preparation worker, preserving all target construction and admission checks.

Both actors must adopt the ownership wrapper before starting overlap. Stage the
queued descriptor with `--mode queued`, its pinned config and wrapper source,
then update the registered queue hash under its existing supervisor lock. The
sidecar refuses to run unless that exact wrapper is present in the live queue.
No dataset, teacher recipe, frozen runtime, or preparation plan changes are needed.

The queued actor publishes a permanent handoff request and waits for the current
whole stage. The sidecar finishes its stage and starts no further stage. They
share one never-replaced flock inode. Every writer inherits the lock descriptor,
including the original coordinator's workers that start a new session. A killed
supervisor therefore cannot release ownership while its writer survives. Never
remove lock/request files to force a restart; inspect surviving processes first.

Use a measured stage timeout and an equal or longer handoff allowance. Increase
the outer queue allowance by the explicit handoff allowance, preserving the
original preparation wall budget. No unmeasured handoff ETA is implied. STOP,
resource exhaustion, or a wall deadline may leave a partial cohort: preserve it
for explicit recovery; never mark it complete or delete it automatically.

The sidecar uses two allowed CPUs, nice15, idle I/O priority, no GPU, and the
original RAM/disk limits. It processes the smallest already pinned cohorts first;
missing-teacher cohorts are sealed only. Existing final preparation revalidates
all completed work and performs final qualification. Do not relocate sealed base
roots or modify their contents: their canonical paths and storage identity are
part of the binding.

Before adoption, `bootstrap_preparation_probe.py` can time one complete cohort in
an independent shared-artifact output directory. It uses the same pinned worker
and producer, replacing only the probe's control/output locations. It reports
seal/build seconds and actual compressed/allocated output bytes. Its outputs are
measurement artifacts, not production admission receipts. Run it through the
bounded entry point, never its internal `--worker` entry point directly.

Config fields are explicit: pinned `prep_plan` and `prep_runner`, `python`,
`runtime`, child `env`, independent `state`, `queue_file` and `queue_id`, inherited
`stop_paths`, destination `disk_paths`, `memory_floor_gib`, `rss_cap_gib`,
`disk_floor_gib`, `stage_seconds`, `handoff_seconds`, `sidecar_seconds`, and
`queued_seconds`. Resource validation rejects relaxed original bounds; queued
seconds must equal original preparation seconds plus the handoff allowance.

A `max_build_rows` limit can leave unusually large cohort builds to the final
coordinator while still sealing every base. The sidecar starts no further stage
when its remaining wall allowance is shorter than the configured stage cap.

The initial 262,079-row/32-shard probe on two CPUs measured17.57s sealing and
119.68s building/verifying B/C/D, with106,098,688 allocated output bytes. These
measurements support a bounded initial overlap of ready cohorts up to2.1M rows,
a40-minute per-stage cap, and45-minute handoff allowance; they are extrapolation
inputs, not guaranteed full-corpus runtimes. Larger19M builds remain on the final
coordinator. Preserve the raw probe receipt with each concrete adoption record.
