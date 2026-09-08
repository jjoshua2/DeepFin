# Prepared B100 handoff (not armed)

This one-use operational wrapper waits for H20's first registered C20T05/100-simulation
arena. It reads only process/control receipts and process identities, never game
scores. Default invocation prints preparation status and launches nothing.

After parent review, the exact arming command is:

```sh
/usr/bin/nice -n 19 /usr/bin/ionice -c 3 /usr/bin/taskset -c 0,1 /usr/bin/python3 /home/josh/projects/chess/scratchpad/bt4_joint20/B100_preparation_v1/arena_handoff_v1/wait_and_exec.py --arm 57f9be7bb543a9edd4b3028e1e3ce419a37a2ebb1a9bceaa8a4c18d81752117f
```

The wrapper pins the coordinator's PID/start ticks/cmdline/cwd/parent and boot ID,
H20 manifests and launcher source, runtime manifest, and the reviewed B100 command,
launch manifest and supervisor. Every poll rechecks these small pins. It requires
both original training handles absent, H20's qualified training completion and its
schedule hash, and a still-running first arena with the exact registered manifest,
command, working directory and coordinator → timeout → arena ancestry. It refuses
STOP, failure, completion, missing/reused coordinator identities, or a four-hour
wait timeout. Polling is every 15 seconds. A stale/missing first arena does not
fall through to a later comparison.

An exclusive `run/` directory allows only one attempt. `started.json` records the
waiting PID; `handoff.json` records the observed arena identities and exact command
immediately before `execv`, which replaces the wrapper under the same PID. Failures
preserve `failed.json` and require explicit reconciliation; there is no retry.
No other process is signaled. The arena may naturally finish after the observation;
this is a start-time gate, not a requirement that it overlap all materialization.
The existing B100 supervisor retains its own four-hour cap, disk/output/CPU guards,
exclusive preparation lease and publication checks. This wait does not consume its
materialization allowance.

`qualify_gate.py` passed six disposable gate cases: valid first arena, still-live
training refusal, wrong ancestry, wrong command, reused coordinator and STOP.
Process identities and receipts are synthetic; the actual pinned arena command
constructor is used. It calls neither the operational main nor exec, and is not an
operational qualification or arming event. No B100 materialization was launched.
