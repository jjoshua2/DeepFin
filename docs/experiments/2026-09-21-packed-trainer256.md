# 256-shard paired actual-trainer storage benchmark

Status: corrected CPU qualification running; GPU pair unlaunched. Packing finished
in229.83 s, but the original CPU attempt stopped after626.29 s because PR #795
rejected the retained root `row_provenance.npz` on224 shards. That failed run and all
archive bytes are preserved. [PR #817](https://github.com/jjoshua2/DeepFin/pull/817),
stacked on #795, explicitly admits only that root provenance file as opaque hashed
bytes;33 packed-loader/CPU-trainer tests passed independently.

Before any corrected qualification, the operational budget was explicitly amended
to a fresh20 min qualification-only cap because the completed directory traversal
in the failed attempt took roughly396 s. The selection, archives, seed, and order
remain unchanged. This corrects representation eligibility; it does not reroll a
throughput result. The corrected runtime is
`2fc51585955044e85f09e27e704d86787c47e382`. Its supervisor/plan passed independent
review before launch and banks each completed stream atomically. No complete
paired qualification or GPU throughput result is claimed yet.
This continues the 2026-09-20 storage-loader record. The previous 32-shard,
262,144-row exact sampler passed full ordered tensor parity, but measured no
training. PR #795 supplies ordinary packed-Zarr loading; it is an explicit
unmerged dependency at `7f2c0ca55cbe0a12b08cc80ab107422f290880da`.
The benchmark tools are based separately on main `5d4f50ddc`.

## Frozen comparison

Use 256 unique V50 shards covering all 35 factorial source cohorts: **1,963,948
rows**, including legitimate partial endpoint shards. Reuse the existing 32
archives at `/mnt/e/chess_packed_exact32_20260920`; select 7 evenly spread shards
from cohorts 1–20 and 6 from cohorts 21–34. The modest sample per cohort is broad
coverage, not proportional sampling of the corpus. Group identities must retain
the original source-parent partition. Ordinary roots contain symlinks to existing
NVMe data; external roots retain cohort subdirectories. Do not copy NVMe data.

Pinned metadata plan: `/tmp/packed-trainer256-preparation-plan-20260921.json`.
Preparation code refuses a different selected roster or output binding. It packs
unchanged compressed Zarr members into ZIP_STORED and verifies every member
against its source, including reused archives. An incomplete output is preserved;
no automatic reroll or cleanup. Metadata inspection itself is not qualification.

Before training, require full ordered tensor-stream parity using the exact frozen
runtime's `qualify_packed_zarr_epoch.qualify`, batch512/seed121, 175 input planes,
`lc0_root_legacy_meta`, history-repetition fix and mirror augmentation enabled,
2 plan workers, 2 load workers, and 12 GiB working-set bound. Both coverage receipts
must include every selected row. Use `run_packed_trainer_preparation.py` around this qualification rather than
the older CLI's weaker32 GiB available-RAM guard. Its exact independent review
receipt is required at launch; its20 min deadline includes runtime preflight.

Both arms use the real `lc0_control_train.py`, D's frozen config
`/tmp/deepfin-factorial58-runtime/configs/lc0_positive_control.yaml`, SHA256
`413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2`, seed121,
batch512, steps0, one exact epoch, 88-step windows, 2 plan/load workers,
12 GiB working set. Pass `--allow-partial-corpus --allow-invalid-control` because
this is a deliberately subsetted throughput comparison without a strength-purity
receipt; never score its checkpoints as a strength experiment. Preserve all
architecture and realized loss guards. The external arm additionally requires
`--allow-packed-zarr`. Compilation configuration remains identical in both arms.

Run **external ZIP then NVMe directory**, each in a fresh process and output root,
through `packed_trainer_probe.py`. The wrapper records initial-model SHA256,
remapped game/ply batch order, per-call sampler time, first-batch latency and its
own observation cost. Full tensor hashes are computed outside timed training;
compact actual batch hashes still must agree between arms. Retain the driver's
full summary, train-window phase metrics and checkpoints. A successful driver
return alone is not a completed benchmark: require exact coverage, expected steps,
matching initial weights, matching actual batch order, finite losses, and all
qualification pins verified. Complete runtime/import/native-extension and config
pins are enforced by the parent's launch supervisor; the probe's driver hash is
only an additional check.

## Metrics and decisions

Primary: external/NVMe realized training rows-per-second ratio after excluding the
first two completed 88-step windows from **both** arms. A descriptive screen passes
at ratio >=0.90, provided every parity/coverage/resource gate passes. Report every
window and total timing, including cold process startup, first batch, excluded
windows, compilation and full-epoch wall time. Report `batch_prefetch_wait_s` and
other existing window phases, batch sampler time, and observer cost separately;
the trainer CPU phase fields partition train_time_s, but sampler observation,
GPU event spans and total wall timing overlap those fields and must not be added. Bank all raw
records and actual step counts, including the final partial window.

If external throughput is below90% with materially higher loader wait, this
storage comparison fails its screen and supports further loader investigation.
One cache-affected ordered pair does not identify disk access as the cause. If it passes, packed storage
has survived a larger actual-training screen, not a 500M capacity qualification.
If both arms are mostly compute-bound, the test bounds observed loader cost for
this workload only. One ordered pair gives no reliable between-run confidence
interval or causal estimate independent of order and cache state; do not bootstrap
windows as independent runs. Do not reroll an unfavorable pair.

## Budgets and unresolved limits

Initial CPU preparation **including full-stream qualification** had a20 min cap;
the explicitly amended qualification-only run has a separate20 min cap,
cores12,13, nice19, two numerical threads, new data <=10 GiB. Floors150 GiB free on
both devices and40 GiB available host RAM. The preparation supervisor enforces aggregate deadline,
STOP, and resource floors during all stages; in-process checks between ZIP members
are additional protection and do not replace bounded child termination. Reused
archives are read-only. Expected new ZIP payload is roughly1.2 GiB; actual bytes
are recorded, not inferred from this estimate. No global cache drop.

GPU work remains unlaunched, queued only after D, D_C, D_B and the BT4 pipeline
benchmark. Maximum45 min/arm,90 min aggregate; only the owned benchmark child may be
terminated. The observed D rate near1600rows/s suggests20.5 min/arm steady training
plus startup/compile; that estimate is not a completion guarantee. Required CPU
qualification warms both roots; the selected dataset also fits host memory. This
is explicitly a cache-affected paired trainer test, not sustained cold HDD I/O at
500M or a test of teacher quality. Outputs/checkpoints count toward the10 GiB
combined new-disk allocation and require a supervisor byte cap.

## Queued GPU supervisor

`run_packed_trainer_pair.py` requires a separately approved immutable plan and
independent review receipt. It authenticates the BT4 queue descriptor and requires
that predecessor's successful completion before CUDA admission. It additionally
refuses any existing CUDA compute PID rather than stopping another process. The
parent queue places it after D and both arenas, then the BT4 pipeline benchmark.

The pair supervisor authenticates the prepared source-member hashes and archive
hashes and both staged rosters, then launches each actual trainer through the
observer. TensorBoard, temporary files and separate TorchInductor caches reside
in each owned arm working directory; frozen runtime contents are checked again
after each arm. Host-batch overlap stays off, matching D's actual CLI. No compiler
or model setting is changed between arms. Aggregate90 min and per-arm45 min limits
reserve8 s for owned-process-group cleanup. An exited leader does not exempt its
descendants from cleanup. STOP, RAM/disk floors and combined10 GiB new-output cap
remain active. D and E are never terminated by this instrument.

A completed measurement can report `storage_screen_pass:false`; that is distinct
from an incomplete or parity-invalid run. The primary denominator is precisely
`sum(train_samples_seen)/sum(train_time_s)` over windows after the first two
88-step windows; it counts the actual final partial batch, not512 times step
count. Full process wall including startup and checkpoints remains an explicit
secondary, not a denominator selected after seeing the result.

Host dependency distribution versions are recorded by the CPU qualifier and
checked again before and after each GPU arm. This is a pinned repository/native
runtime with version-checked host packages, not a newly installed locked environment.
