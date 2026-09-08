# Common G10 input batch: prepared, not launched

The fixed registration selects the first32 closed receipt-backed shards from each
original source,531,412 physical rows total. Only derivation, adaptation and top3
ranking are selected. No audit rerun, C/H materialization, teacher call, GPU or
training is included. The64 closed receipts are immutable copies; their selected
names/counts/raw SHA values match the banked assessment. Registration preparation
read receipt/attribute metadata only. Raw payload hashes are checked under the
bounded execution before use, and again after the entire batch.

Runtime is the already qualified clean checkout `/tmp/deepfin-g10-pilot-tools` at
`ac4a246ba025ca54329d6e4dc380473f1ea4b728`, using the existing CPU development
interpreter. No runtime checkout or global environment is modified. `launch.json`
pins the registration, runtime qualification, runner, timer/supervisor binaries,
64 raw-sidecar attribute files, source manifests and receipt snapshots. Source
raw hashes are explicitly registered, not recomputed during preparation.

`run_common.py` keeps the pilot's reviewed owned-timeout and receipt cleanup
structure, with one explicit common-data worker. All stages are sequential.
Each real stage executes once under GNU time, recording wall/user/system time,
maxRSS KiB and filesystem input/output counters. These counters include child
work as reported by GNU time; filesystem counters are operation counters rather
than bytes, and maxRSS is not aggregate simultaneous process-tree RSS. No timing
rerun is scheduled. During derivation the runner observes only owned descendants
and requires exactly two distinct spawn worker PIDs; it does not infer realized
worker count from the requested flag alone.

The total cap is7200 seconds, including30 seconds termination grace and initial
coordinator checks. A surviving GNU timeout session contains every stage and its
spawn workers. SIGINT triggers coordinator cleanup; SIGTERM/SIGKILL can bypass
Python cleanup, while the surviving timeout retains the cap. Failure or STOP
prevents dependent stages; exclusive output/log creation refuses reuse. Missing
terminal receipts or partial outputs are failures, never completed inputs.

All descendants inherit CPU0,1, nice19, idle I/O and hidden GPU. Torch/numerical
thread environment variables request2. A real fresh-spawn probe found that
`BLOSC_NTHREADS=2` alone left Blosc at8 threads (Torch was2). The operation therefore
adds its **private pinned** `python_bootstrap/sitecustomize.py` at the front of
PYTHONPATH, followed by the unchanged frozen checkout. Python loads that hook at
startup, including fresh multiprocessing-spawn interpreters, before script imports.
It sets and verifies Blosc2, exiting78 if initialization fails instead of letting
Python silently continue after a sitecustomize exception. A second fresh-spawn
probe established Torch2/Blosc2. Both observations are retained in `author_fixture`.
The hook is only passed to this operation's supervisor environment; it is not
installed globally, and does not affect any live process.

The runner samples total state apparent bytes and free disk every second and at
stage boundaries:8GiB output/cache allowance,150GiB reserve. These are sampled
stop conditions, not filesystem quotas. Both per-source private adapter/rank
caches have hard64MiB limits. Spill, partial outputs, timing logs and fixture
artifacts count toward the allowance. No cleanup or retry is automatic.

Per-source commands use the registration verbatim: `uniform-d9`, phase0 policy,
latest-phase searched value, temperature0.0005, floor0, seed0,8192 rows/shard,
`--row-provenance --workers 2`, and the complete registered physical prefix limit.
After derivation, a metadata identity snapshot freezes all source output entries.
The adapter mapping is produced solely from completed derived summary pins and
registered source/teacher receipts. Adapter/rank counts and hashes come from that
completed summary. Ranks request top3 and64MiB cache, consuming the provenance-aware
phase0 path. Exact expanded commands, including predecessor hashes, are written
immediately after Popen inside the cleanup guard in each stage-start receipt.

Final qualification uses ordinary source-bound BT4 and rank admission functions
on every emitted row, verifies legal normalized BT4 mass and its published payload
hash, checks exact shard inventories, and rejects duplicate original physical rows
across output shards. It reports survivors per raw shard, all derivation exclusion
counters and the source-qualified input-sequence hash. Derived storage identity
must remain unchanged across adapter/rank/qualification, preserving every nonpolicy
field as well as SF targets. The original raw namespace, not varying output roots,
is the identity basis. This is shared-input qualification, not execution of a
training schedule or evidence of playing strength or complete G10 coverage.

Prepared invocation after separate review:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 19 ionice -c 3 taskset -c 0,1 \
  python3 run_common.py --execute --expected-plan-sha256 ACTUAL_LAUNCH_SHA
```

No execution has been performed. Author qualification used only a tiny2-worker
stdlib fixture for the new GNU-time/descendant measurement path and tiny actual
runtime spawn-thread probes. The completed pilot supplies prior pipeline evidence;
this larger batch is the next actual single measurement, not a timing repetition.
