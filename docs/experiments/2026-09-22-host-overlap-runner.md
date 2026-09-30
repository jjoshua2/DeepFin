# Host-overlap runner preparation

Status: executable companion tooling with small CPU tests; no full-stream CPU
qualification, approved launch plan, GPU admission, queue entry or launch.
The preregistration is `/tmp/host-overlap-next-gpu-prereg-20260922.md` (SHA-256
`419e88a51e137066e8d4c5f2273154b4bbef1af21a38c02e403dd87d50b2c66d`).

The runner `scripts/run_host_overlap_pair.py` implements the proposed fixed OFF
then ON comparison on the existing 256-shard NVMe bank and original factorial
runtime `502cd02e072471c901255f3fdb580d6ea7b826d0`. The only training argument
intervention is `--epoch-host-batch-overlap`; seed121, batch512, one full epoch,
88-step windows, two plan/load workers and 12GiB sampler cap are fixed. Companion
observer `scripts/host_overlap_probe.py` is distinct from immutable runtime code.
It records initial tensors/optimizer, final saved tensor/optimizer-state digests,
checkpoint bytes, actual overlap-iterator consumption, peak RSS and CUDA allocator
memory. No exact GPU-state parity assumption is waived: a mismatch is ineligible.

`scripts/qualify_host_overlap_cpu.py` is the full-stream producer. It imports the
frozen runtime and calls its actual `Trainer._iter_training_batches` in the same
88-step windows as training, with device CPU and no forward/backward. A host-only
Trainer shell uses the frozen constructor's config/default values for host
preparation and objective-mask counting; it never builds the 61M model. OFF samples
and prepares on the main thread; ON must use the frozen `exact-host` producer and
consume all 3,836 batches through `_iter_exact_overlapped_batches`. The producer
records the complete 3,752×512 plus 84×511 schedule, the ordered raw, prepared
and game/ply digests, separate plan hashes, realized receipts, producer threads,
reservation and peak planned/realized working set. The consumer verifies those
fields against the bank's 1,963,948 rows and the two streams against each other.
The OFF plan deliberately omits its reservation key, which the tooling reads as
zero; ON must reserve positive bytes.

The CPU producer takes `--plan PATH --plan-sha256 HEX --review PATH
--review-sha256 HEX`. Its immutable JSON plan must include `status` =
`APPROVED_HOST_OVERLAP_CPU_QUALIFICATION`, `runtime` =
`/tmp/deepfin-factorial58-runtime`, `runtime_commit` =
`502cd02e072471c901255f3fdb580d6ea7b826d0`, `root` =
`~/chess-artifacts/operations/packed-trainer256-20260921/directory`,
`config` = the frozen `configs/lc0_positive_control.yaml`, `preparation_receipt`
= `~/chess-artifacts/operations/packed-trainer256-20260921/receipt.json`,
`python` = the exact intended interpreter, `out` = a new output directory,
`cpu_qualification_seconds` ≤ 1,800, `runtime_files` = the exact `inventory(runtime)`
mapping, `cpu_producer_sha256` = the producer's own SHA-256, and `pins` =
path/SHA-256 pairs for the config, receipt, producer and
runtime driver. The review JSON must say `status` = `APPROVED` and bind the exact
`plan_sha256`; both files are authenticated before work. Resource checks reserve
32 GiB RAM and 150 GiB free disk, use cores 14/15 at nice 19, hide CUDA and set
two library threads. This separate prerequisite has its own 30-minute cap;
the pair supervisor's 10-minute nontraining cap starts later. A passing run creates `OFF.json`, `ON.json`,
`qualification.json` and `complete.json` under `out`.

No such full-stream receipt exists yet. The prior OFF-only packed-loader receipt
cannot replace it. The producer remains held while E training needs storage/CPU
resources. Future GPU admission also requires the specified E_D arena to be
successfully logged, its descriptor binding and outer exit.

Bounds are 75 minutes per arm, 160 minutes overall and 10 minutes aggregate
non-training admission/reporting, each with cleanup reserve. Shared GPU lease and
owned process-group cleanup are retained. STOP/deadline/RAM/free-disk checks stay
active on every guard call. Output accounting checks shallow checkpoint/log files
each 0.5-second supervisor poll and recursively samples only new output trees at
most every 15 seconds, plus every arm boundary and finalization. It counts the
greater of logical/allocated bytes and derives the preparation allocation from
the pinned source receipt. At 28 GiB it stops admitting a new arm; at a sampled
32 GiB it terminates owned work. This is a sampled stop rule, not a kernel disk
quota: an output can temporarily grow past 32 GiB between samples, especially
in nested compiler/cache trees. Admission requires 150 GiB free disk and 32 GiB
available RAM. Each arm uses isolated compiler/TensorBoard/temp output; compiler
and numerical-library thread counts are two. No source bytes are copied.

The GPU supervisor requires an independently reviewed plan with the pinned CPU
qualification and its `complete.json`, the E_D prerequisite descriptor/queue,
the exact runtime roster, dependency versions and path/SHA-256 pins. The reviewed
plan must include `runner_sha256` and immutable hashes for `qualification.json`, CPU `complete.json`,
the E_D queue, descriptor, completion receipt and outer terminal. It also
requires `driver_sha256` and `probe_sha256` for the immutable driver and external
observer. The fixed input bank, original config SHA-256
`413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2`
and original receipt SHA-256
`e42f319ccd2c10176979bb6dd6b5217b1c29510814df1ec42f263adb71b832f6`
are code-gated. The plan status must be `APPROVED_FOR_QUEUED_HOST_OVERLAP_PAIR`.
No such plan or queue item has been created.

The fixed primary is train time per update in windows 2–42 inclusive. All 44
windows remain in coverage, finite-metric and full-wall gates. The external
observer banks every individual update loss with one post-epoch device transfer,
the actual game/ply order, overlap iterator consumption, initial/final state and
checkpoint digests, and RSS/CUDA peaks. Passing needs exact state, loss, order and
coverage parity, at least 5% less primary time, and no more than 5% full process
wall regression. The terminal status distinguishes `COMPLETE_SCREEN_PASS` from
`COMPLETE_SCREEN_FAIL_RETAIN_OFF`. One fixed-order pair supplies no fleet
precision or authority to change live training.

Twenty-three focused CPU/fake-child tests pass on cores 14/15 at nice 19 with two Torch
threads. They exercise qualification drift, fixed-window/full-wall gates, exact
state comparison, observer saved-checkpoint bindings/hook restoration, byte cap,
and child failure/STOP cleanup. Ruff and scoped typechecking are checked before
review. These tests do not certify full-bank equivalence or measured GPU speed.
