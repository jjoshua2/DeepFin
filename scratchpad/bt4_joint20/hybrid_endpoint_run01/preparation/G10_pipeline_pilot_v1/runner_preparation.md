# Registered G10 CPU pilot runner preparation

Prepared only. No payload scan, derivation, adaptation, audit execution or GPU work
has been performed by this preparation. `run_pilot.py` is a fixed operation for
this directory's registration; it is not a repository launcher or a queue.

The parent must first merge the reviewed G10 implementation and select an inactive
clean checkout at that exact commit. Provide `runtime_qualification.json` with:

```json
{
  "status": "qualified",
  "checkout": "/absolute/frozen/checkout",
  "commit": "actual full merged SHA",
  "python": "/absolute/qualified/venv/bin/python",
  "pins": {"/absolute/imported/file/or/native.so": "actual SHA256"}
}
```

Use the existing qualified CPU development environment; bind its actual resolved
interpreter, native extensions, dependency lock/environment identity and imported
project runtime paths. This preparation deliberately does not invent a final
merged commit or runtime hashes. The freeze command validates the receipt and
pins, tracked clean HEAD, registration, fixed audit bank and selected BT4 attrs;
it produces an exclusive `launch.json` and prints its SHA. It does not run stages.

```bash
python3 run_pilot.py --freeze \
  --checkout /absolute/frozen/checkout --commit ACTUAL_MERGED_SHA \
  --python /absolute/qualified/venv/bin/python \
  --runtime-pins /absolute/runtime_qualification.json
```

Review the generated launch JSON, pinned runner and runtime qualification before
execution. The separately authorized invocation is:

```bash
CUDA_VISIBLE_DEVICES='' nice -n 19 ionice -c 3 taskset -c 0,1 \
  python3 run_pilot.py --execute --expected-plan-sha256 ACTUAL_LAUNCH_JSON_SHA
```

`--execute` creates a single GNU timeout supervisor session: the registered total
1800 seconds includes 30 seconds of kill grace and coordinator preflight time.
Children inherit the hidden GPU, two numerical/compression threads, CPU0,1,
nice19 and idle I/O. The timeout survives loss of the coordinator and owns only
this pilot's process group. Coordinator SIGINT triggers cleanup of that owned group. Coordinator SIGTERM or
SIGKILL can bypass Python cleanup; the surviving GNU timeout retains the deadline.
No existing job or GPU lease is touched. Any failed stage stops dependent stages,
leaves logs/partials, and cannot be automatically adopted/retried. A surviving
partial worker receipt is not a completed pilot. Only terminal exit0 plus the
final worker proof within1800 seconds produces `completed.json`.

The worker checks `STOP`, total pilot-state apparent bytes and free disk at every
stage boundary and every second during stages. The disk checks are sampled stop
conditions, not filesystem quotas: retain headroom when evaluating the8GiB cap.
The private adapter and rank caches each have their independent hard16MiB limit.
All pilot output is charged, including temporary `.writing` trees, copied C/H
corpora, logs and indexes; preexisting small registration files count too.

## Exact stage construction

The frozen runner contains the complete argv builder. Dynamic values are only
actual completed predecessor counts and SHA hashes, written in each exclusive
`*.started.json` at stage start, immediately after Popen inside the cleanup guard,
and repeated in its successful stage receipt.
No unknown count/hash is replaced by a guess or permissive bypass.

1. For each registered source, read the real resolved inventory, require selected
   `w00-00000` to be first and claimed8236/8243 respectively, hash its raw bytes
   against the registered SHA. This reads no other raw payload. Retain raw inode,
   size, mtime and ctime through the whole pilot and rehash at the end.
2. Reanalyze the unchanged saved deep-SF audit bank for C20T05 and H20T05 using
   `bt4_policy_mix.py audit`,10000 bootstrap replicates, seed20260903,
   descriptive mode and the pilot JSON as experiment record. Prior receipts bind
   other registrations and cannot be relabeled for ordinary admission. The exact
   three original bank files and hashes are in the runner; no teacher is invoked,
   and these are new contextual receipts for old observations, not new evidence.
3. Per source, derive with the registration's literal arguments and physical-row
   `--limit`. Write an adapter manifest from the actual derived summary, pinned
   original source/closed receipt and original teacher attrs. Adapt into ordinary
   BT4 sidecars with16MiB private identity allowance.
4. Build provenance-aware d9 ranks with top3, original raw parent, identical
   physical limit, rows-per-shard8192 and seed0; expected output counts and source
   SHA come from the completed derived summary. Its selected policy observation
   is phase0, inherited from that summary, independent of latest-phase value.
5. Ordinary C mixer: scope `sf-cp-window`, alpha1, BT4temperature0.5, rankcap3,
   cpwindow20. Ordinary H mixer: scope `c20-global`, alpha0.2, same BT4/rank/window,
   actual C parent with both completed C summary hashes. Both admit the exact
   adapted sidecar, original derived corpus and new descriptive audit receipt.
6. Compare every non-policy array in256-row chunks across original/C/H; require
   identical array sets and exact copied NPZ provenance per shard. Hash the full
   original source-qualified row-reference sequence, not the varying output
   directory identity. Record physical/emitted/excluded counts and full derivation
   exclusion statistics. Finally recheck all pins and original raw identities/SHA.

The final comparison is **input-schedule identity**. It proves the physical row
sequence and non-policy inputs match after mapping output parents to original
source namespaces. It does not execute a training planner or replace a later
prospective/realized training schedule qualification, nor establish playing
strength, all-G10 coverage or100M throughput.

Preparation validation: Python syntax compilation only. Actual runtime pin and
stage qualification, failure-path review, and operational execution remain pending.
