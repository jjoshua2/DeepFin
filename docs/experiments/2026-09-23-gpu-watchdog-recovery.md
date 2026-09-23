# GPU watchdog incident and recovery hold — 2026-09-23

**Status:** Recovery hold. No GPU qualification or loader benchmark has been run under this record. This note records the incident and recovery sequence without identifying a cause.

## Report and observed evidence

- The user reported two NVIDIA watchdog bluescreens today and said the driver was updated to the latest available version. The user requires **one GPU job at a time**: training, benchmark, labeling, and arena jobs are mutually exclusive. The coordinating agent is the sole GPU launcher; other agents do CPU-only work unless it explicitly grants an exclusive GPU slot.
- A subsequent host check reported an RTX 5090, NVIDIA driver `617.14`, no GPU compute applications, and current boot ID `92844ac4-013d-4c78-adaf-2a0fb56cdde6`. These observations establish the checked host state, not the cause or duration of either reported bluescreen.
- The frozen loader ABBA `plan_v5.json` pins earlier boot ID `26438fc2-a424-438c-9340-5ab88b699a1d`; `admission-host.json` was also on that boot. The current boot is a changed host environment, so the old admission does not qualify a new launch.
- `run01/complete.json` says `INCOMPLETE`, `arms: []`, `arm or lease deadline reached`, after `786.6776` seconds. The first control's `run01/arm1_control/training.log` shows compile/max-autotune activity and Triton candidates rejected for resource limits. `ptxas` received `SIGTERM` near the supervisor timeout. This is consistent with termination at the deadline; it is **not evidence of a driver crash**. No ABBA performance or correctness comparison can be read from this run. These files are preserved under `/home/josh/chess-artifacts/operations/gpu-loader-abba-20260923/`.

## Hold and proposed recovery

Preserve the complete run directory, plan, logs, caches, and host receipts as evidence. Do not restart the same ABBA plan on the new boot, alter watchdog settings, or tune driver, clocks, or voltage in response to this record. Keep GPU work paused while the coordinating agent reviews the host state and assigns the only GPU slot.

The next GPU action is one minimal, bounded **single-job** trainer qualification using eager execution without max-autotune. Disable Aurora optimizer CUDA graphs separately (`aurora_cuda_graphs: false`); eager/no-compile alone does not disable them. Register the exact workload, source/config identities, time limit, observed progress, and stop rule before launch. Record host boot/driver and any driver or kernel fault evidence alongside its terminal result. A timeout, reset, or bluescreen ends that qualification and returns the host to the hold; a clean completion qualifies only that bounded workload, not general GPU stability.

Only after that qualification is reviewed should a **separately registered** loader benchmark be considered. It needs a fresh current-boot admission, a new output directory, an explicit control/candidate comparison and decision rule, and exclusive GPU ownership for its full run. The incomplete ABBA result supplies no speedup claim and is not a restart point.
