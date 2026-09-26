# Held proposal: process-local NumPy huge-page advice A/B

**Status:** reviewable proposal only. No child, shard loader, training, GPU,
or system-setting run is authorized by this record. Reconsider only in a
separately admitted quiet CPU slot after the active E workload.

## Question and controlled comparison

Does turning off NumPy's `MADV_HUGEPAGE` advice reduce the actual
`GameAwareEpochBuffer._load_one` time on the same qualified eight-shard
schema-2 overlay fixture? Run four isolated child processes in A/B/B/A order:

| Arm | Runtime | Child-only environment |
| --- | --- | --- |
| A (0, 3) | candidate `7bafd7c3004a010356c214898c807d9d67b8b3f3` | `NUMPY_MADVISE_HUGEPAGE=1` |
| B (1, 2) | same exact candidate commit and source hashes | `NUMPY_MADVISE_HUGEPAGE=0` |

The supervisor must construct each child environment from one common inherited
environment and change **only** this key. Set it before interpreter startup
and NumPy import. Keep `/usr/bin/python3` 3.10.12, NumPy 1.26.2, Torch
2.11.0+cu128, numcodecs 0.13.1, Zarr 2.18.3, loader code, seed 121,
batch size 128, load workers 1, selected shard order, filesystem, and
cache-control procedure identical. Do not set `MALLOC_ARENA_MAX` in one arm
alone. The source's `target_overlay.py` SHA-256 is
`903d0c314b3c4627fe72da63d1fc6a33bd4638145bcc6fab0195472a2a8cb0f0`;
`game_epoch.py` is
`a84ad58fd68719e42513fa0411ca64f02ce848167cc1496a0a6add1ff4664b78`.
Recheck those and the clean source HEAD at admission and inside each child.

Reuse the immutable 65,536-row first-eight-B-shard roster, qualification
receipt SHA, per-shard content digests, eight-path order, and existing
measurement steps from
`/home/josh/projects/chess-worktrees/overlay-hotload-diagnostic-main-20260923/docs/experiments/evidence/loader-profile-20260923/prepared-hotload-plan.json`
(current SHA-256
`a7d406a794e366729c2c1d4f1ca2be330fc361472153a8cabebb05127f406d56`).
The prior qualified run's four-arm summary is at
`/home/josh/chess-artifacts/operations/overlay-hotload-diagnostic-20260923/summary.json`;
it established exact decoded-array parity for the PR810 source contrast and
reported 10.9253 s versus 4.1104 s median summed `_load_one` time. Those
timings are **not** a huge-page A/B result.

## Smallest compatible runner change before any admission

The existing `scripts/benchmark_overlay_hotload.py` cannot run this A/B
unchanged: its prepared plan pins *different* control/candidate source commits,
and its `subprocess.Popen` inherits one environment for all four children.
Prepare a new pinned plan that maps both arm names to the same candidate
source, retaining the original fixture and resource gates. In an isolated
copy/revision of that runner, set the per-arm child environment before
`Popen`, and have every child record and require both the requested env value
and `np.core.multiarray._get_madvise_hugepage()` (`True` for A, `False` for B)
immediately after NumPy import. Refuse missing/mismatched realization. Pin
the revised runner's exact hash in the new plan and independently review its
environment-only diff before it is eligible for admission. The old pinned
plan/runner and completed output remain immutable.

Retain the current child's selected-shard content rehash and receipt checks,
real `_scan_shards`/`_plan_epoch`/`_load_one` calls, and hashes of every
decoded field's shape/dtype/bytes plus the ordered target hashes. All four
arms must match the exact plan, roster, row count, output hashes, source
commit, and package versions. Record per-shard and summed `_load_one` wall
time, whole-child wall and user/system CPU, peak child RSS, realized NumPy
advice state, and relevant read-I/O counts. Global `/proc/vmstat` compaction
and memory-PSI deltas may be banked as **descriptive concurrent context**;
they cannot assign stalls to these children or prove a system-wide cause.
No cache dropping or global THP/sysctl change is permitted.

## Bounded admission and decision

Keep the reviewed runner's limits: four arms maximum; each child at most
180 s wall and 120 s CPU; cores 16–17, nice 19, two numerical threads,
CUDA hidden; require at least 40 GiB `MemAvailable` and 150 GiB free disk at
each gate; no more than 4 MiB new output. Use a fresh, separate `ADMITTED`
receipt pinning the new plan and quiet-workload evidence, and the existing
STOP/deadline/owned-child cleanup rules. Refuse missing inputs or changed
source/receipt/shard hashes. Keep the total nominal wall ceiling at 12 min
plus small setup/receipt overhead. Do not execute while E training or another
heavy CPU/storage job is active.

Eligibility requires exact output parity and four valid realized-mode
receipts. The proposed positive screen is at least **10% lower** median summed
`_load_one` wall time for B versus A, both adjacent A/B comparisons in the
same direction, and no more than **5% higher** median whole-child wall time.
Otherwise classify as no passing local loader benefit or invalid if a gate or
parity check fails. Even a positive screen warrants only a later separately
reviewed trainer-level test; it does not justify changing the active trainer,
global THP settings, or a deployment default. A null result applies only to
NumPy-advised allocations in this eight-shard fixture, not every allocator or
all THP behavior.
