# Live E loader profile and the next CPU comparison

Date: 2026-09-23 UTC. Status: descriptive profile; prepared comparison not admitted.

The existing validation-reuse patch warrants a hot-loader comparison before more
work on small batch-gather optimizations. A 20-second nonblocking Python profile
of the live E trainer sampled the repeated schema-2 target validation path inside
normal shard loading, not just constructor work. The profile was taken during
the eight-worker CPU generation arm, when input wait had increased; this is a
short, potentially unrepresentative interval, not an end-to-end speedup estimate.
No trainer settings or sources were changed.

## Profile evidence

[Raw folded stacks](evidence/loader-profile-20260923/E-main.raw) and
[summary/source hashes](evidence/loader-profile-20260923/profile-summary.json)
retain the sample. `py-spy record --nonblocking --rate 20 --duration 20 --format raw`
observed trainer PID2156939, frozen runtime `502cd02e072471c901255f3fdb580d6ea7b826d0`.
The profiler ran on cores16–17 at nice19. It reported1198 all-thread samples and
zero sampling errors. Only397 identified main-thread samples form the denominator
below; idle/compiler/empty stacks are excluded. Categories are nested and must
not be added together.

| Main-thread path | Samples | Fraction |
|---|---:|---:|
| Host sample/preparation wrapper |187|47.1%|
| Loading next shards |173|43.6%|
| Dense overlay validation (`_validate_local`) |109|27.5%|
| Batch gather |3|0.8%|
| Host augmentation/preparation |5|1.3%|
| Tensor collation |6|1.5%|

`GameAwareEpochBuffer._load_one` hashes the qualified overlay before and after
loading it. `load_shard_arrays` also checks its identity around decoding and obtains
array proxies. In the frozen runtime these operations reopen the schema-2 manifest
and repeat `_validate_local`, which reads target probabilities and legal support.
The code still performs useful checks; removing validation outright is not proposed.

[PR810](https://github.com/jjoshua2/DeepFin/pull/810) reuses a validated identity
within the caller's sealed context, while rechecking anchored file/tree/receipt
identities and rejecting mutation. Its previous **5.42×** result is for eight-shard
constructor stages only. The source path now observed during training gives a
reason to measure `_load_one` too. Neither27.5% nor5.42× predicts a full-epoch gain.

## Prepared CPU diagnostic

[Draft runner](evidence/loader-profile-20260923/benchmark_overlay_hotload.py) and
[plan](evidence/loader-profile-20260923/prepared-hotload-plan.json) extend the previous
banked eight-shard B constructor screen. They call the real `_load_one` method on
scanned/planned records with the same validated context; all decoded arrays' shapes,
dtypes and bytes, complete plans, counts and target hashes must agree. This isolates
loader work and does not instantiate a complete trainer. Its census callback counts
rows, not the trainer's real objective masks; decoded-array hashes cover all fields.
B has the same schema-2
loader mechanism but different targets from E.

The intended order is control/candidate/candidate/control, with no GPU, two low-priority
cores, two numeric threads,180seconds/120CPU-seconds per arm,40GiB available RAM and
150GiB free disk. The primary is median summed per-shard `_load_one` time, measured
after constructor work; semantic-validation counts explain the mechanism. A10% gain
with exact parity would justify a full-trainer comparison, not deployment.

This is **not launch-ready**: exact current source/fixture/environment pins, independent
runner review, aggregate supervision and quiet-workload admission remain necessary.
The draft's per-arm limits and fresh-output checks do not replace those steps. No
CPU stream or GPU comparison was launched. More generation work is held while we
check the observed input-wait slowdown and preserve the current E experiment.

Independent review reproduced every sample count, verified all three frozen source
hashes and the candidate's code path, and checked the draft loader initialization.
It confirmed the diagnostic scope and identified the missing expected source/receipt
pins; those remain explicitly required before admission.
