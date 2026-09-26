# SF-close corpus preparation overlap

## Prospective scheduling amendment

Prepare the already selected C20T05 corpus while the same-checkpoint prior
calibration and subsequent global arenas use the GPU. This changes only CPU
materialization timing. SF-close training and evaluation still wait for the global
screen and its reviewed scheduling decision. No result selects this recipe or
changes its dose. No training or arena is launched by this preparation wrapper.

The immutable parent record remains
`scratchpad/bt4_joint20/sf_close_run02/preregistration.md`, SHA-256
`c4c5a0b861efce888a62c9fc085d61509da0ab805790e108f0f293ea33d029b4`.
Its instruction to finish globals first is relaxed only for this one CPU copy and
policy rewrite. Preserve that snapshot and its audit admission; this amendment is
not an `--experiment-record` argument for the legacy audit.

## Fixed artifact and prerequisites

The recipe remains `sf-cp-window`, SF d9 rank cap 3, 20 effective-cp window,
alpha 1.0, BT4 temperature 0.5: original exact ties union eligible SF-close moves,
with selected SF mass redistributed using sharpened BT4. Source, BT4 and SF-rank
sidecars, and the existing C audit are the exact pinned full-bank inputs from the
parent record: 18,910,484 rows in 2,309 shards. No teacher inference is needed.
SF regret remains a descriptive proxy, not a strength verdict.

Output is `data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05`.
Both it and its `.writing` sibling must be absent at admission. The G20T05 corpus
must already be published, its `.writing` path absent, and the operator must verify
there is no other active materializer before authorizing execution. Serving
checkouts, generators, monitors, calibration and global jobs remain unchanged.

Use frozen mixer checkout `/tmp/deepfin-bt4-toolchain`, HEAD
`4394bd2cddc05ea780ce303a1f731730a1f4dfff`, script SHA-256
`5fe8202f1e9681b042caeea06614cdbe3b55cf41ec2100182ce3e58e95e377c7`.
The runtime launcher pins the interpreter, source/audit/sidecar summaries, parent
record, G20T05 publication and baseline source-footprint manifest. Its complete
command and environment are banked before its owned child starts.

## Resource budget and stopping

Preparation has a four-hour wall-clock limit, nice 19, a two-core CPU affinity,
two numerical threads and an explicit two-thread Numcodecs Blosc setting. Hiding
CUDA and invoking only the arithmetic `mix` subcommand excludes GPU computation.
The Python bootstrap is solely for enforcing these runtime limits before running
the unchanged CLI; it does not change mixer semantics.

The source footprint in the completed baseline manifest is 12,021,141,504 bytes
(about 11.20 GiB). At the reported 476.5 GiB free, a nominal copy leaves about
465.3 GiB. Compression differences and ongoing corpus generation can change that
estimate. Admission requires 150 GiB plus the recorded source footprint; the
launcher checks the 150 GiB floor every two seconds. This is a sampled guard, not
a filesystem quota. Keep the existing generator disk monitor running.

A single preparation reservation and lock prevent duplicate launches. The wrapper
checks preparation STOP, parent SF-close STOP and SF-close phase STOP. Interruption,
STOP, reserve breach, time limit or child failure stops only the owned child group,
waits 20 seconds, then escalates within that group and reaps its leader. Preserve
all partial output, receipts and logs. There is no automatic retry or deletion.
After wrapper SIGKILL or host reboot, the unresolved reservation requires explicit
recovery; do not infer success or launch a replacement.

## Completion and later use

Success requires exit zero and published output with the expected row/shard counts
and recipe. Bank summary hashes and the exit receipt. The existing frozen SF-close
driver later performs its complete published-corpus adoption validation before
training. Never start it while preparation is still running or `.writing` exists.
Bank source-normalized metadata schedule verification and an independent sampled
non-policy readback before training; a sample does not prove full-corpus byte equality.

Runtime artifacts live under
`scratchpad/bt4_joint20/sf_close_run02/preparation/`: `prepare_c20t05.py`, a frozen
copy of this amendment, `launch.json`, `exit.json`, and `mixer.log`. Execution
remains pending parent coordination and independent review. The default command
prints the plan only:

```bash
/usr/bin/python3 scratchpad/bt4_joint20/sf_close_run02/preparation/prepare_c20t05.py
```

Adding `--execute` starts only the reviewed CPU preparation. No playing-strength
claim or automatic follow-up is produced by successful materialization.
