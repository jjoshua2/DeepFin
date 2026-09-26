# Prepared overlay hot-load diagnostic

The pinned [plan](prepared-hotload-plan.json) and
[`scripts/benchmark_overlay_hotload.py`](../../../../scripts/benchmark_overlay_hotload.py)
are held for any future quiet CPU slot. No comparison has run with this revised
runner and plan. The `check` command
verifies the runner, Python and package versions, both clean source commits and
file hashes, qualification receipt hash, eight-shard roster and row count. It
does not open shard contents.

The original ABBA measurement, its exact executed runner and plan bytes, and
the readout are preserved in [PR #838](https://github.com/jjoshua2/DeepFin/pull/838).
This plan pins the later tooling revision for any future run; it is not the
executed plan from that measurement.

```bash
/usr/bin/python3 scripts/benchmark_overlay_hotload.py check \
  docs/experiments/evidence/loader-profile-20260923/prepared-hotload-plan.json
```

The `run` command requires a separate JSON admission receipt with `status` set
to `ADMITTED`, `plan_sha256` equal to the exact plan file, a nonempty
`quiet_workload_evidence` and `admitted_by`, and `admitted_unix_seconds` no more
than 15 minutes old. The plan itself stays `PREPARED_HELD_FOR_UNCONTENDED_CPU_SLOT`.
This makes the quiet-workload decision explicit and separate from preparation.
When admitted, use a new or empty output directory:

```bash
/usr/bin/python3 scripts/benchmark_overlay_hotload.py run \
  docs/experiments/evidence/loader-profile-20260923/prepared-hotload-plan.json \
  /tmp/overlay-hotload-admission.json \
  /home/josh/chess-artifacts/operations/overlay-hotload-diagnostic-20260923
```

The runner executes four children in control/candidate/candidate/control order.
Each rechecks the source and receipt pins, checks the selected content digests,
scans and plans genuine records, then times every actual
`GameAwareEpochBuffer._load_one` call. Every decoded field's shape, dtype and
byte hash, complete plan, record census, and ordered target hashes must match
across all arms. The synthetic census counts rows and does not represent the
trainer's real objective masks. The primary decision is a 10% or larger
reduction in the median summed per-record `_load_one` time with exact parity;
passing warrants a later trainer test, not deployment.

Each child uses cores 16 and 17 at nice 19, two numerical threads and no
visible CUDA devices. Per-arm ceilings are 180 wall seconds and 120 CPU
seconds. Admission requires 40 GiB available memory and 150 GiB free disk at
each gate. The four-arm wall ceiling is approximately 12 minutes plus small
setup and receipt overhead; total child CPU is capped at 480 CPU seconds.
Previous constructor measurements suggest a shorter run, but `_load_one`
timing under a quiet workload has not yet been observed. A STOP file in the
output directory or its parent prevents new work. The parent owns each child
process group and checks the PID start time before terminating it. Each arm
receipt is atomic and capped at 768 KiB; the summary is capped at 128 KiB and
all new output at 4 MiB. Failed admission, timeout, failed parity, and other
post-output-directory errors write `summary.json` with failure evidence.

This is a read-only CPU diagnostic on the first eight qualified B schema-2
shards (65,536 rows). It omits full-corpus qualification, actual trainer
objective masks, model training, GPU work, and E-corpus performance.
