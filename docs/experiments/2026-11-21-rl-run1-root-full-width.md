# 2026-11-21: RL run 1 — full-width root round (`--root-full-width`): cost, target audit, search strength, training test (pre-registration)

**Status: registered, not launched.**
- **Stages 0–2** need two things before they start: this record is reviewed and merged, and the GPU queue has
  released the card. Every launcher refuses until both queue markers exist (see [the GPU guard](#the-gpu-guard)).
- **Stage 3 is authorised by the owner's standing statement of 2026-11-22: "You dont have to wait for me to approve gpu use it dont have time ill just tell you to stop if necessary" (addendum 2026-11-22).** It is launched on the operator's release only
  after Stages 0–2 have passed by their registered rules and been read in-session. The coordinator records that
  in the bank as `stage3/OWNER_GO`, and one command then runs the whole stage. Nothing else about the gates
  changes.

**Queue position:** after the value-anchor screen (#1475) and the sibling-ordering screen (#1481), unless the owner
reorders.

**Recovery:** nothing live changes. No yaml, no live tree and no deployed code is involved. Every output goes to a
new bank.

## Why

The owner's hypothesis, verbatim:
- "perhaps if we force our gumbel to eval all nodes at root that we can focus on correct ordering?"
- "if we just do all it makes batching way easier and gpu will load better so we could also increase node count
  from 64 to like 74 and probably still come out ahead"
- "just this change should help some i believe with value head"

The knob exists on main since #1477 (merge `fd8c0ce08`) as `GumbelConfig.root_full_width`, exposed as the collect
flag `--root-full-width`. It is default off.

**What it does:**
- **Round 0** visits every searched root child that has 0 visits, once each, before sequential halving starts.
  - The top-k candidates are then re-chosen by gumbel + log prior + σ(completed q).
  - The halving budget is unchanged.
  - Round 0 is skipped on fast plies.
- **Accounting.** The round-0 simulations are spent **on top of** `--sims`.
  - They are stored per row as `root_prelude_visits`.
  - The stats field is `mean_root_prelude_visits`.

**The real trade, at equal cost.** Let p be the round-0 visits per position. At equal GPU cost the knob spends
about `S_eq` + p ≈ 64 evaluations, of which ≈ 64 − p go to sequential halving.
- So full width moves ≈ p of the 64 evaluations away from depth on the top-16 candidates, and gives one visit to
  every other legal move instead.
- What the test asks is whether that buys a better target, and a better player, than the depth it costs.

**Spec deviation, stated once and carried everywhere.** The owner pictured round 0 as one wide batch. The
implementation drives the unchanged C `start_gumbel_sims` state machine instead. It packs round 0 first-fit into
calls of at most the existing leaf buffer (`len(enc_buf)`, 4,096 rows at 3 × 128) and at most `GSS_MAX_CANDS` = 64
children per board per call.

**Is the 4,096-row buffer binding?** Measured on CPU from the pilot's own roots (`sizing/results/round0_fit.json`):
- The searched support is 27.43 children per root (median 29, p90 43, max 68). Only 12 of 752,885 rows exceed 64.
- **A typical-call estimate under independent sampling**: sum the supports of 128 rows drawn at random from the
  whole collect, ignoring carried roots. The mean is 3,510, SD 141 and p99.9 3,939. P(> 4,096) is 2e-5.
- **This is not an upper bound.** All 128 slots start together from 16-ply openings, so early in a run the boards
  are in phase, at middlegame plies where the support is widest. The review's computation (not banked) found:
  - with 128 boards at ply 30 and full supports, the sum averages 4,985 and exceeds 4,096 every time;
  - with the same boards but carried roots (only unvisited children are evaluated), the mean is 3,692 and
    P(> 4,096) = 0.0004.
- **So the buffer splits a round-0 call rarely: mostly in the synchronised opening phase, and even there in about 1
  call in 2,500 once carried roots are counted.** A split costs one extra GPU call on that ply. Enlarging the buffer
  is not part of this test.
- **Stage 0 sees more of the synchronised phase than a real collect does.** A run is 384 games per process, about
  three waves of 128, against about 21 waves in an 8,000-game collect.

**What the CPU smoke already shows, so the owner's cost guess is checked rather than assumed.** On 32 cold rl-audit
roots (`stage1/smoke_cpu/`; a smoke, not a result):
- Knob on at 64 sims spends **94.1** root visits per position, against 64.0 with it off: round 0 visited 29.2
  children.
- In the collect, carried roots need fewer round-0 visits.
- "64 → 74" therefore cannot be assumed to cost about the same. **The equal-cost sims count `S_eq` may be below 64.**
  Stage 0 measures it.

## Prior evidence (grepped first; [evidence log](2026-11-21-rl-loop-evidence-log.md), bank `INDEX.md`)

- **The pilot's searches leave about half the root unvisited.**
  - At 64 sims, 14.53 of 27.43 searched children are visited.
  - The prior mass on unvisited children is small: mean 0.0041, median 0.00021. It exceeds 0.05 on 2.1% of rows.
  - So round 0 mostly buys q-values for moves the prior already rates near zero. Its candidate re-selection matters
    only where one of those moves is actually good.
- **Search-ceiling and search-gain banks.**
  - Search gains from the SL base are large: 64 vs 1 is +849, 256 vs 64 is +272.
  - **A net's own deeper search is NOT an admissible ruler for its target.** In
    `self_search_ruler_inverts_vs_independent_refs`, the stored 64-sim target looked +8.4 pp better by the net's own
    R800 but only +0.7 by BT4.
  - So Stage 1 uses deep SF 8M and the searched Ceres ruler, never the SL base's own deeper search.
- **Policy-target zeros (`policy-target-zeros/`).**
  - The stored target's near-zeros are mostly right on deep SF.
  - Search does push some SF near-equal alternatives below 1e-4: 9.2% [7.5, 10.8] of them, vs 1.6% for the prior.
  - The full-width round could change exactly this. It is a Stage 1 secondary measure.
- **Searched ordering vs deep SF (`sibling-target-scoping/`).**
  - On clear pairs (gap ≥ 50 cp), R64n q orders children better than the raw net: +0.0274 [+0.0180, +0.0365].
  - On near-ties it is a coin flip.
  - The knob gives every child at least one visit, which widens the set this applies to. Stage 1 measures that.
- **Collect throughput (`collect-throughput/`, [simple_rl_loop.md](../simple_rl_loop.md)).**
  - 3 × 128 at 64 sims: 5,021 games/h, 9,262 forwarded positions/s, GPU 87%, 21.9 GB.
  - The repeat agreed within 0.5% (4,993.6 vs 5,021.4).
- **Pilot collect timing.**
  - The pilot's `stats.json` wall fields are broken: `wall_seconds = active_seconds = 2.33e-06`. That attempt was
    a merge-only reseal after the seal step was OOM-killed.
  - Collect time is therefore taken from the per-process walls (5,512 / 5,534 / 5,471 s) and the stamps:
    - `collect.started` 06:41:24;
    - first attempt exit 137 at 08:26:46;
    - reseal 09:51:00 → 09:54:42.
  - **Play time is 1.54 h, the slowest process's wall (5,534 s), giving 5,204 games/h.** The pilot was not paused.
    #1483 adds `slowest_process_wall_seconds` and `slowest_process_active_seconds` for exactly this reading.
  - The stats bug still existed on main and is fixed separately in #1483, with a test.

## The GPU guard

**Addendum (2026-11-22): the guard as first written could never pass on this machine.** v1–v6 refused unless
`memory.used` < 1,024 MiB. With every queue finished and nothing of ours running, this card reads **3,382 MiB**
(30 one-second samples, 04:38–04:39 on 2026-11-22: min = max = 3,382 MiB, utilisation mean 0.0%, max 1%; between
the anchor cells earlier that night it read 3,430 and 3,452 MiB). That memory belongs to the Windows side of WSL2
(desktop compositor, browser), not to a job. So the old bar refused always, and nobody could have seen it pass,
because it had only ever been exercised against stubs. v7 replaces it (below), and the guard test now runs the
guard's two probe commands against the REAL `nvidia-smi` once (read-only), asserting that both parse to integers.

**Memory and utilisation cannot see a loading job.** A reproduction cell started at 04:54:28 read 3,448–3,463 MiB
at 0–1% utilisation for **six minutes** (until 05:00:15), then 5.5 GB, then 22 GB
(`~/chess-artifacts/operations/rl-run1-value-anchor-repro/arm_Zq_k11.3.gpu.txt`, 5 s samples). During a job's loading
phase the process scan (`gpu_scan.sh`, below) and the operator's release file are the only defences; the memory
band is a backstop.

**Which launchers are guarded.** `gpu_guard.sh` v7 (sha in the table below; earlier versions kept as `.vN`) is sourced by:
- `stage0/run_stage0_all.sh` and each `stage0/run_stage0.sh` it starts;
- `stage1/run_stage1.sh`;
- `stage2/run_stage2.sh` and each `run_rfw_match.sh` it starts;
- `stage3/launch_stage3.sh`. It sources the guard at its start, and again immediately before **each** GPU step: the
  collect, each training cell, Stage A, and each match seat.

**One child is not guarded itself.** Stage 3's matches run at the PIN through the follow-up bank's sha-pinned
`run_followup_match.sh` (`7c8fd065…`). That script has no guard and is not changed. The re-check in
`launch_stage3.sh` immediately before each of its calls is what guards those matches. A re-check runs the full list
in the same shell, so it takes the nested-lock path. It waits up to 60 s (`RFW_MEM_WAIT_S`) for memory to
return to ≤ `idle_mib` + 512 MiB after the previous step exits; it never relaxes the bar. Guard (d)'s `check_load.py` runs with
`CUDA_VISIBLE_DEVICES` empty, on the CPU.

The guard refuses unless **all** of these hold:
- **Memory, against the operator's measured idle baseline (v7).** The release line carries it:
  `stage0 idle_mib=3382 2026-11-22T09:00`. The guard refuses (exit 8) if the line has no `idle_mib=<int>`; if
  `idle_mib` exceeds a 4,096 MiB sanity cap (nobody may declare a busy card idle); if `memory.used`, the maximum over
  **every** GPU line, exceeds `idle_mib` + 512 MiB (something else holds the card); or if it is below `idle_mib` −
  512 MiB (the baseline is stale: re-measure it). The ±512 MiB band:
  - observed idle drift over six hours is about 70 MiB;
  - our smallest GPU job holds well over 512 MiB once its context and net are loaded. Banked evidence: in
    `~/chess-artifacts/operations/rl-claude-20261119/engine-speed-profile/idle/*.gpu` (25 runs, 1 s samples, our net
    in one engine process), every sample from the 6th second on sat **≥ 911 MiB** above the pre-start idle (median
    +1,151 MiB for the smallest configuration). One run's first 1 s sample was only +27 MiB: a process in its first
    second, before its CUDA context exists, is inside the band. No banked sampler of a 4-game arena exists; the
    engine profile is the closest banked measurement of our smallest job;
  - idle itself moves between sessions (1,575–2,981 MiB on 2026-11-19/20, 3,382 now), which is why the baseline is
    measured at release time and goes stale.
  - **It fails closed.** If `nvidia-smi` exits non-zero, or prints anything but an integer, the guard refuses with
    exit 11. `RFW_MEM_WAIT_S` must be a non-negative integer (exit 12).
  - **Not `--query-compute-apps`.** Under WSL2 that list cannot be relied on to see another process tree. During
    the incident below it listed this bank's three collects in 62 of 64 samples, but it never listed match 8's
    PID 2337486, which was using the card the whole time.
- **Utilisation (v7).** Five samples of `utilization.gpu`, one second apart, every GPU line: the maximum must be
  below 10% (idle here reads 0–1%; a foreign job, or the owner gaming, reads more, and when he games the GPU is not
  ours). A non-integer or a failed `nvidia-smi` refuses with exit 11.
- **No GPU job of this user (v7: `gpu_scan.sh`).** One file, byte-identical in this bank, the generation-2 bank and
  the sibling bank (same sha; this bank owns it). It reads `/proc` only. A same-uid process (Uid from
  `/proc/<pid>/status`) outside the caller's own ancestors and descendants refuses if:
  - its argv (`/proc/<pid>/cmdline`, NUL-separated) is a GPU entry point: `basename(argv[0])` is listed (`lc0`,
    `Ceres`, `deepfin`, a script run directly); or `argv[0]` is `python*` and the first non-option argument has a
    listed basename, or a `-m` names a module starting `chess_anti_engine`; or `argv[0]` is `bash`/`sh`/`dash`/`zsh`
    and the first non-option argument has a listed basename; or `argv[0]` is `dotnet` and the first argument that
    is neither an option nor `exec` has a listed basename. Options are parsed as the programs parse them (scan v5,
    after the first outside review): python's short options are CLUSTERS (`-um mod`, `-bm mod`, `-uB X.py`,
    `-Wignore X.py`; an `m` or `c` takes the rest of the cluster or the next word as its value and ends the options;
    `-X`/`-W` take a value); a shell's long options are single words (`--norc` is not `-c`; `--rcfile`/`--init-file`
    take a value) and `-o`/`-O`/`+O` take a value. A shell that only mentions a script (`bash -c "sha256sum
    …/arena_standard.py"`) is not a job: v6's `ps -o args` token scan refused it, v7 does not. **Stated limits of
    the argv clause:** `python -c "runpy.run_path(…)"`, `python -m cProfile X.py` (any runner module) and
    `--check-hash-based-pycs <mode> X.py` are not matched, nor is a wrapper that forks a python child (`timeout`,
    `uv run`) at its own argv (the child is scanned as itself). The release file covers these before a launch, and
    the device clause once CUDA initialises. Processes of other uids are not scanned; the memory band is their
    backstop.
    **CPU-only exemption (argv clause only):** a matched process is exempt iff its exec environment
    (`/proc/<pid>/environ`) is readable and holds the exact entry `CUDA_VISIBLE_DEVICES=` with an empty value and no
    other `CUDA_VISIBLE_DEVICES` entry. Such a process cannot reach the card: the rule every agent works under, so
    agents' pytest children, fake-HOME launcher copies and harnesses do not block the operator. The variable absent,
    any value, a confusable name, two entries, or an unreadable environ → not exempt. A NOTE names each exempted pid.
    **Honest limit:** `environ` is the environment at exec; a process can still set the variable itself before
    importing torch. The device clause catches that only once its CUDA context exists, so the release file and the
    launch discipline remain the primary defence;
  - it holds the GPU device open, whatever its name. This clause runs for EVERY non-excluded same-uid process,
    independently of the argv clause: an empty or unreadable cmdline skips only the argv clause (v4 skipped such a
    process before the device check; the outside review's surviving mutant). The device is identified by rdev, not by name: the scan stats
    its own `/dev/dxg` once (it must be a character device, else it refuses: "cannot identify the GPU device"), and
    a process holds the GPU iff one of its fds, by `stat -L` on `/proc/<pid>/fd/<n>`, is a character device with
    that major:minor. Every fd is checked, so a holder through another path (a second device node, a bind mount) is
    caught. A file whose link merely reads "/dev/dxg" but is another file is not a holder (a NOTE names it): v1–v3
    matched the name, and another agent's namespaced test harness, holding a stand-in, made the real guard refuse
    (processes of a child pid namespace are visible in the parent's `/proc`, and `readlink` prints the path string
    whatever it resolves to in that mount namespace). That would have aborted a multi-hour stage at its between-step
    re-guard. Cost of checking every fd: one scan took 0.26–0.75 s wall on 2026-11-22 with agents working (v4: 62
    same-uid processes / 561 fds; v5: 0.31–0.32 s for 95 processes / 617 fds);
  - its fds cannot be stat'ed, so the device check cannot be made for it. Readability is decided by actually
    reading every fd (`find` lists them, `stat -L` reads each target; exit status and error text kept), never by
    `access(2)`: access
    to a proc fd entry also goes through the ptrace check, so `access()` can say yes while every readlink is denied
    (a non-dumpable process in a user namespace, scanned by a capability-free shell; `gpu_scan.sh` v1 scanned such a
    `/dev/dxg` holder clean). Listing the directory is not enough either: there `readdir` succeeds while `readlink`
    is denied. A lister failure whose every error is "No such file or directory" means the process exited mid-scan
    and is skipped; any other failure, including one with no error text, refuses.
  - **Allowlist, exactly two:** `argv[0]` equal to `/usr/bin/ssh-agent` or `/usr/bin/gpg-agent` (full path). These
    are same-uid but non-dumpable by design, so their fd links are unreadable; v1 refused permanently on this
    machine because of `/usr/bin/ssh-agent` (pid 40). An allowlisted process is skipped for the fd checks only, and
    the scan prints a NOTE naming it. A read-only check of the real machine on 2026-11-22 found exactly one
    unreadable same-uid process: `/usr/bin/ssh-agent`.
  - **The list** is derived, not chosen: every `scripts/*.py` at main `fd756be84` matching `git grep -lE
    "cuda|--device"` (97 files); every `*.sh` / `*.py` in the follow-up, value-anchor, value-anchor-repro,
    generation-2, sibling, root_full_width and pilot-value-diagnosis banks that uses cuda, `--device cuda`,
    `torch.load`, `arena_standard.py`, `simple_rl_loop.py`, `match_vs_uci.py`, lc0 or Ceres (minus the guards and
    test files), plus the launchers those start; every console script of the package (`[project.scripts]`,
    identical at the pilot PIN, `ARENA_SHA` and main: `deepfin = chess_anti_engine.uci.cli:main`); and the lc0 and
    Ceres binaries. 128 names; the derivation is in the file's header.
  - **Excluded:** the caller's ancestors (the launcher that sourced the guard, and its parents) and the descendants
    of the caller's shell, so a launcher's between-step re-check does not trip on itself or its own child. Process
    groups are deliberately NOT excluded: a non-interactive shell starts background jobs in its own group, so a
    sibling copy of the same launcher started from the same shell line would share it. A sibling is always refused.
  - Seeing no process of this user at all refuses (the scan must see its own shell).
  - **Harness rule.** A guard harness whose cases run a listed name WITHOUT an empty `CUDA_VISIBLE_DEVICES` is
    visible from the host through its child pid namespace, so a real scan refuses while it lives. Such harnesses
    (this bank's `guard_test.sh` and its mutant runner among them) must not run while a registered stage relies on a
    re-guard. Both wait, before starting a pass, while the coordinator's `GPU_CHECK_HOLD` file exists (a 5 s file
    test). The converse trap is in the harness itself: since the exemption, a "refused" or "excluded" case whose
    listed process runs WITH the empty entry passes whatever the scan does. Every argv-dependent case therefore
    reads `/proc/<pid>/environ` of the process it started and fails as VACUOUS if the exemption would apply.
  - **Known cosmetic defect (v5, left for the next batch to spare three re-pins).** A launcher log can show
    `gpu_scan.sh: line 217: desc: bad array subscript`. It is not a failure. Cause: a process exiting mid-scan
    leaves its PPid empty, and the descendant loop then indexes `desc[""]`. The test fails, so that process is NOT
    excluded (the safe direction), and the scan's result is unaffected; it is non-fatal under `set -euo pipefail`
    (verified). The one-line fix for the next batch: store `pp[$p]` only when the PPid read is non-empty
    (`[ -n "$pa" ] && pp[$p]=$pa`).

**Between-step re-guards retry through a transient GPU user (addendum 2026-11-22).** At about 07:50 on 2026-11-22
the scan refused, correctly, on an agent's pytest process holding the real device. `torch.cuda.is_available()`
opens and holds `/dev/dxg` for the process's lifetime even with `CUDA_VISIBLE_DEVICES=` empty, because WSL2's
`cuInit` opens the adapter before the mask applies. A transient holder like that must delay a registered multi-hour
stage, not abort it. So (`launch_lib.sh` `guard_retry` / `rfw_top_guard`):
  - every between-step re-check re-runs the WHOLE guard (release line, memory band, utilisation, process scan, lock)
    about every 5 s, until it passes or `RFW_REGUARD_WAIT_S` expires (default 900 s; it must be a non-negative
    integer, else exit 12). Each distinct refusal reason is logged once, with a timestamp, to the stage's
    `guard.log`, and so is a pass after a wait. The re-check refuses only if the LAST attempt still refuses, with
    that attempt's reason and a deadline line in the log. A refusal by the operator's release line (revoked,
    missing, wrong stage) is the operator's intent and refuses at once, never retried;
  - Stage 3's `reguard` between its steps is this re-check;
  - the TOP guard of a launcher started by its registered parent (`run_stage0_all.sh` → `run_stage0.sh`,
    `run_stage2.sh` → `run_rfw_match.sh`) is this re-check too. That covers the gap between Stage 0 runs and between
    Stage 2 seats. "Started by its registered parent" needs all of the following, and anything else gets the
    single-shot guard as before (an operator's direct start, or a stray start with a leftover variable):
    - `RFW_PARENT_PID`, set by the parent on its start line, equals the child's own PPID;
    - that parent's script fd (`/proc/<pid>/fd/255`) is the registered launcher, and its argv names it;
    - the child's fd 9 and the parent's fd 9 are both this bank's lock;
    - a fresh open of the lock cannot take it, while `flock -n 9` on the child's fd succeeds, so the child holds the
      parent's own inherited lock;
  - the first guard of a stage stays single-shot;
  - an agent test touching the device during a stage therefore delays that step by up to the bounded wait, and the
    delay is logged.

  **Stage 0 timing.** The wait happens BEFORE a run starts: `run_stage0.sh` takes `t0` only after its top guard has
  passed, so a run never begins while a holder is present and no wait enters a run's timing. The wait (seconds
  until the passing attempt began; 0 when the first attempt passes) and the guard mode go into the run's summary
  row (`guard_wait_s`, `guard_mode`). Inside the ABBA block a wait lies in the gap `t1(previous leg) → t0(this leg)`,
  which the existing `GAP_MAX_S` = 900 s rule (the block's drift tolerance: one leg) already bounds: a gap
  including a wait above 900 s means FRESH_ABBA_REQUIRED. So the bound on an in-block wait is `GAP_MAX_S` itself,
  less the normal inter-leg overhead. A wait is never mixed in silently: `stage0_decide.py` v6 reports every leg's
  wait (`abba_guard_wait_s`) and sets `abba_block_has_guard_wait` iff leg 2, 3 or 4 waited (leg 1's wait comes
  before the block), and it prints a NOTE naming those legs.

**GPU memory of the full-width collect (addendum 2026-11-22).** The only banked peak at Stage 0's shape is with the
knob OFF: K=3, C=128, 64 sims, `eval_max_batch_realised` 4096, peak `memory.used` 22,104 MiB in total
(`~/chess-artifacts/operations/rl-claude-20261119/collect-throughput/logs/sweep.txt`, `mp3_c128_s64_prof`). No
measurement with the knob ON existed, and a CPU smoke does not measure GPU memory. So the coordinator measured it
as an operator diagnostic outside this bank, a paired OFF-then-ON collect at Stage 0's exact shape (3 × 128 games,
64 sims, the arena pin, all processes rc 0, `eval_max_batch_realised` 4096 everywhere):
`~/chess-artifacts/operations/rl-run1-root-full-width.opdiag/mem_20261122/` — `README.md` [`3d00d75f`] (written
before the run), `RESULT.md` [`0a062442`], `SHA256SUMS` [`d5251224`], `gpu_{off,on}.txt` (1 s samples), and
per-process stats and logs. Idle was 1,138 MiB before each run.
- Peak `memory.used`, knob OFF: 20,566 MiB (peak − idle 19,428).
- Knob ON: **28,320 MiB** (peak − idle 27,182; median 26,454).
- The knob adds **7,754 MiB** at this shape. The prereg's expectation of "a similar peak" was wrong by 7.8 GB,
  which is why it had to be measured.
- Take effect, read from the consumer: `mean_root_prelude_visits` 25.13 / 25.31 / 24.83 with the knob on, absent
  with it off.

**Rule:** Stage 0 is released only if the knob-ON peak ≤ `memory.total` − 1,024 MiB (31,583 MiB) at the measured
idle. With the measured +27,182 MiB, that holds for idle ≤ 4,401 MiB: 305 MiB inside at the guard's 4,096 idle cap,
and 30,564 MiB total at the usual idle of 3,382. PASS, but thin at the cap.

**Per-leg memory check (manual; addendum 2026-11-22).** `run_stage0.sh` already samples `memory.used` every 500 ms
into `runs/<tag>/logs/gpu.csv` and writes the run's maximum to its summary row as `gpu_mem_max_mib`. Nothing reads it
automatically. Before reading any Stage 0 block, the operator compares each knob-ON leg's `gpu_mem_max_mib` with
`memory.total` − 512 MiB. `memory.total` is read once at the release and recorded in the release line's note. A leg
above that bound voids the block and holds the stage: a spill into shared memory, which on WSL2 slows a run before
it fails, must not silently contaminate a timed leg. The coordinator makes this check by hand and records it in the
stage's readout. The automated version (`run_stage0.sh` recording `memory.total`, `stage0_decide.py` refusing with
`MEMORY_SPILL_HOLD`) is on the next-batch list, beside the `gpu_scan.sh` `desc` fix.

**Where the 7.8 GB comes from (read from the code and the logs, not changed).** The collect never uses the 2-group
eval pipeline, in either arm. It reuses trees, and `gumbel_c.py:1342` turns the pipeline off whenever a persistent
tree is supplied; all six diagnostic logs print that warning, and only that one. Both arms therefore evaluate all 128
boards in one call per rep, one leaf per active candidate:
- **knob OFF:** the widest call is the first halving round, 16 candidates × 128 = 2,048 rows (the banked
  collect-throughput histogram at this shape has no bucket above 2,048);
- **knob ON:** round 0 adds about 27 unvisited children × 128 ≈ 3,500 rows, packed first-fit up to the 4,096-row
  buffer (`_full_width_round0`).

CUDA graphs cover only ≤ 512 rows, so these calls run eager. The caching allocator keeps the eager activation
high-water mark, about 2.6 GB more per process × 3. A lever not taken: cap round-0 calls at 2,048 rows (about two
calls per ply instead of one). The −26% rows/s of the drain runs is the extra evaluations themselves: 89 against 64
evaluations predicts ×0.72 at a constant evaluation rate; the reading is ×0.743, at ×1.03 evaluations per second.
Stage 0's `S_eq` prices exactly that. In the arena (Stage 2, cold trees) the off seat does pipeline and the knob seat
does not, but Stage 2 (b) is played at fixed sims, so this changes only the reported timings.
  - **Measured on the real machine (read-only, 2026-11-22):** `gpu_scan.sh` v2 refused because of another agent's
    pytest child (`python -u -m chess_anti_engine.uci … --device cpu`), which is why v3 added the CPU-only
    exemption. v4 returns 0 there with agents working, printing only NOTE lines: an exempted `launch_stage3.sh` (a
    test copy, run with `CUDA_VISIBLE_DEVICES=`), and the allowlisted `/usr/bin/ssh-agent`.
- **The queue has released the card.** Both markers exist: `rl-run1-followup/chain.exit` and
  `rl-run1-value-anchor/queue2.exit`.
- **The operator released THIS stage.** The bank holds a file `GPU_RELEASED_BY_OPERATOR` with a line whose first
  word is the stage the launcher requests (`stage0` … `stage3`), with the measured idle baseline and a date, e.g.
  `stage0 idle_mib=3382 2026-11-22T09:00`.
  - **Queue order is enforced by that file, which the operator writes. It is never inferred.**
  - The markers cannot do this job. The sibling screen (#1481) runs after queue 2 and has no marker this guard
    could know about. And between a chain's steps, memory falls back to idle.

It then takes a non-blocking `flock` on the bank's own `.gpu_launch.lock`; if the lock is held it exits 9. Since
v6 the exit-9 message also names the case where no launcher is running: a child of a SIGKILLed launcher
inherits fd 9, so an orphan can hold the lock until it exits. The message gives the commands to find the holder
(`fuser -v <lock>`, or the lock file's inode in `/proc/locks`, whose line carries the PID), and says to stop that
exact PID, never a pattern match. The flock so two of this bank's
launches cannot overlap. It never opens `~/projects/chess/scratchpad/gpu0_experiment.lock`. A launcher started by
another launcher of this bank (`run_stage0_all.sh` → `run_stage0.sh`, `run_stage2.sh` → `run_rfw_match.sh`) inherits
the parent's locked descriptor and re-locks that, so nesting works; anything else opens the file and must get the lock.
That inheritance is new in v3. Since v4 the guard reads `/proc/$BASHPID/fd/9` rather than `/proc/$$/fd/9`, so it
is also correct if sourced inside a subshell. fd 9 open on any other file is replaced by a fresh open of the lock
file, which must then win the flock.

**Launcher hardening (2026-11-22).** Another agent's guard test reached the GPU because its launcher had no
`set -e`, so a failed `source` of a missing guard fell through to the real launch. This bank's launchers had the
same hole: `run_stage0.sh`, `run_stage1.sh` and `run_rfw_match.sh` sourced the guard without checking the result.
Every launcher now starts with these lines, before any sha check or use of a path:

```bash
set -euo pipefail
GUARD=$HOME/chess-artifacts/operations/rl-run1-root-full-width/gpu_guard.sh
[ -f "$GUARD" ] || { echo "REFUSED: no GPU guard at $GUARD"; exit 8; }
RFW_STAGE=<stage> source "$GUARD" || exit 8
```

Under `set -e`, a command whose exit code is read on purpose is written `cmd && rc=0 || rc=$?`. After the guard,
every launcher also sources `launch_lib.sh` (`9bae510f71b0…`, new), with the same existence check and `|| exit 8`.

**Children are stopped before a marker is written (new).** Each long-running child is started in its own session
and process group with `setsid`; `launch_lib.sh`'s `spawn` / `runkid` do this. The launcher then waits for it with
`wait`, never as a foreground child, so INT and TERM reach the launcher's trap at once. INT exits 130, TERM exits
143. The children are:
- each collect process;
- each training cell and Stage A step;
- each arena seat, including the follow-up bank's `run_followup_match.sh` together with its arena python;
- each nested launcher.

Before writing its marker, the EXIT trap calls `stop_kids`:
1. it sends TERM to each child's process group, by group id (never a pattern kill);
2. it waits until every member is gone: up to 30 s for leaf steps, 60 s for the parents `run_stage0_all.sh` and
   `run_stage2.sh`, whose children have their own 30 s;
3. it sends KILL to whatever is left.

So a marker never reads "ended" while GPU work still runs. Before this, a TERM wrote 130 while the three collects
kept going.

**What the guard test does and does not prove.** It does not prove that `set -e` is present. The explicit
existence check and the `|| exit 8` are what stop a launcher, and the mutant below shows that removing them is
caught.

**The exit marker is written from an EXIT trap.** Once a launcher has passed its checks, every exit goes through an
EXIT trap that writes its marker: a success, a failed step, a `set -e` abort, or SIGINT/SIGTERM. The markers are:
- `run_stage0.sh`: `run.exit` and `REMOVED_ROW_DATA.txt`;
- `run_stage0_all.sh`: `stage0_all.exit`;
- `run_stage1.sh`: `stage1.exit`;
- `run_rfw_match.sh`: the look's `.exit`;
- `run_stage2.sh`: `stage2.exit`;
- `launch_stage3.sh`: `stage3.exit`, beside the per-step `.exit` files.

A missing marker can then only mean SIGKILL or the host dying, never "failed".

The review found a real case of the defect this fixes. In `run_stage0.sh` v4, the shard cleanup after the GPU work
ran `find` on every process directory. Under `set -e`, a process that died before creating its directory made that
`find` abort the launcher, so neither `run.exit` nor `REMOVED_ROW_DATA.txt` was written. The cleanup now runs inside
the trap, and only on directories that exist. I checked every launcher for the same class: a command after the GPU
work whose failure skips a marker. Stage 3's Stage A steps (`|| exit 1` before `stage_a.exit`) were the other
instance; `stage3.exit` now covers them.

**Relaunch after a refusal: resume, never re-roll, never lock out (new).** The review found that Stage 3 could not
be relaunched after a refusal that followed its collect. Its `[ ! -e stage3/collect ] || exit 95` turned each of
the ≈ 12 legitimate re-check refusals after the 8,000-game collect into a lock-out. The multi-step parents had the
same lock-out:
- `run_stage2.sh` restarted at its first seat, and `run_rfw_match.sh` then refused with 95 because the game log
  existed;
- `run_stage0_all.sh` skipped the rest of the ABBA block once `summary.jsonl` was non-empty, and reused a run
  index whose directory existed.

All three now resume by the same rules, and every decision is appended to `resume.log` with its evidence:
- **A finished step is skipped and its outputs reused.** Finished means its `.exit` reads 0 **and** its completion
  check passes:

  | Step | What it consumed (`.inputs`, must equal now) | Its own outputs |
  |---|---|---|
  | Stage 3 collect | launcher sha, PIN, `ARENA_SHA`, seed, `S_eq`, the collect argument string, the environment | seal (the collect's sealing `manifest.json` + `stats.json`), the take-effect read run again, and (v7) every output the collect's manifest records (shards tree, games, starts) re-hashed against it by `verify_outputs.py`, with the repo's own digest: one read of the collect, 6.3 s wall and 112 MB RSS on the pilot's 1.9 GB collect, page cache warm (measured; a cold cache is disk-bound) |
  | Stage 3 training cell | sha of the source collect's `manifest.json`, the cell's argument string, `ARENA_SHA`, PIN, `S_eq`, launcher sha, the environment | seal (`model.pt` + the train `manifest.json`), plus a cross-check: the cell's own record of the collect it read (`gen_0001/collect/manifest.json` → `inputs.source_collect_manifest.sha256`) names that same manifest |
  | Stage A | the 3 nets' and the SL base's shas, the environment | a seal over its outputs |
  | Arena seat (stages 2 and 3) | seat, seed, n, both models' shas (stage 2: its seat arguments, i.e. sims and knob, plus the SL base), the opening file's sha, the arena commit or PIN, the match launcher's sha, the environment | `seat_state.py` v3 DONE (below) |
  | Stage 0 run | `s0_inputs`: the run's sims and knob, plus the shas of `run_stage0.sh` (its constants, seeds, `ARENA_SHA`), `count_wrap.py`, `steady_rate.py`, `stage0_decide.py`, the SL base, the pilot bank's sums, the environment | a row in `summary.jsonl` |

  "The environment" (addendum, `launch_lib.sh` v4 `envf`) is the variables production code actually READS on these
  paths, as one sorted line of shell-quoted `KEY=value` words. v3 fingerprinted whole prefixes (`CUDA_`, `PYTORCH_`,
  `TORCH`, `OMP_`), so `OMP_NUM_THREADS=3` on a relaunch blocked a whole Stage 3 resume although nothing on these
  paths reads it, and for Stage 0 the printed remedy threw away a valid run and tripped FRESH_ABBA_REQUIRED. The set,
  from `git grep` for `os.environ` / `getenv` at `ARENA_SHA` `fd8c0ce08` and PIN `0c795fb7a` (no `getenv` in C):
  - every `CAE_*`, `CHESS_*` and `DEEPFIN_*` name: the families the repo uses for its own knobs, e.g.
    `CAE_COMPILED_LEGAL_POLICY` / `CAE_COMPILED_LEGAL_ROWS_POLICY` (`chess_anti_engine/inference.py:191`, `:200`),
    `CAE_DUP_STATS` (`mcts/gumbel_c.py:113`), `CAE_BUCKET_HIST` (`inference.py:1526`), `CAE_ARRIVAL_TRACE`
    (`inference.py:1581`), `CAE_NEURAL_WORK` (`inference.py:2572`), `CAE_INFERENCE_COMPILE`
    (`tune/distributed_runtime.py:1207`; `:1206` at the PIN), `CAE_HEAP_CENSUS_SECONDS` (`heap_census.py:66`),
    `CAE_SYZYGY_PATH` (`utils/syzygy.py:46`), `CAE_STOCKFISH` (`utils/engine_discovery.py:41`),
    `CAE_RESEARCH_COMPILE_CACHE_DIR` (`scripts/arena_standard.py:2044`; `:2025` at the PIN), `CHESS_LOG_LEVEL`
    (`tune/trainable.py:106`), `CHESS_ANTI_ENGINE_LIVE_CONFIG` (`eval/production_shape.py:66`),
    `CHESS_LIVE_PRODUCTION_CONFIG` (`eval/lc0_control_arch.py:122`);
  - `CUDA_VISIBLE_DEVICES` (`simple_rl/collect.py:582`) and `CUDA_DEVICE_ORDER` (`simple_rl/collect.py:618`; the
    PIN does not read it);
  - `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED`, which the arena's own durable record names as run identity
    (`eval/arena_durable.py:64`): cuBLAS determinism and hash order.

  The full prefix set (`CAE_`, `CHESS_`, `DEEPFIN_`, `CUDA_`, `CUBLAS_`, `PYTORCH_`, `TORCH`, `TRITON_`, `OMP_`,
  `MKL_`, `OPENBLAS_`, `PYTHON`, `LD_`) is written, report-only and never compared, to an `env.observed` file beside
  every step's `.inputs` (stage 3 `<step>.env.observed`, each seat's `<label>.env.observed`, each stage-0 run's
  `env.observed`), so a behaviour difference can still be diagnosed. `clear_step.sh` moves it with its step.

  **An environment difference now says what differs and how to restore it.** Every 87 refusal prints each
  fingerprinted variable that differs (`KEY: recorded=<value|unset>  now=<value|unset>`) and the lines that restore
  the recorded environment (`export KEY=…` / `unset KEY`). If nothing but the environment differs it says so:
  restore and relaunch, never clear. For a stage-0 row that matters most: a clear discards a valid run, and a
  cleared ABBA leg forces a fresh block.

  **A step's completion check covers what it CONSUMED, not only what it wrote** (new in this revision). The review
  reproduced the defect without this: collect GEN-A and a cell are sealed; the collect is cleared and redone as
  GEN-B; the cell trained on GEN-A was then SKIPPED and reused beside GEN-B. Now:
  - every step writes `<step>.inputs` when it starts, and on resume a finished step whose `.inputs` differs from
    now is refused with **exit 87** (never skipped);
  - a Stage 3 run has one `stage3/inputs.fingerprint`: the launcher sha, `ARENA_SHA`, PIN, the PIN worktree's
    actual HEAD, `S_eq`, the `OWNER_GO` sha, the sha of this bank's `SHA256SUMS`, and the environment. Stage 3 also
    checks the bank's sums, as stages 0 and 2 do, and (v7) refuses with 96 unless the PIN worktree is at PIN with
    no tracked edits (`git status --porcelain --untracked-files=no`), as it already did for the arena worktree.
    The fingerprint is written on the first launch, and any relaunch whose fingerprint differs refuses with 87.
    Step outputs without a fingerprint also refuse with 87 (v6 tested this with one `ls -d` over five globs, which
    fails whenever any one glob matches nothing, so the refusal never fired once a run had started; v7 tests each
    path, and a case covers it);
  - **a changed `OWNER_GO` after a first launch that refused before any step ran** (so a fingerprint exists but
    no output does): `clear_step.sh stage3-fingerprint` moves the fingerprint aside, and only while NO stage-3 step
    output exists (exit 2 otherwise); the next launch writes a new one. Once any output exists, the fingerprint is
    part of its record and a changed `OWNER_GO` is an owner decision;
  - `run_stage0_all.sh` reuses an existing `decision.json` only while every row it read still has unchanged inputs
    (87 otherwise).

  `seat_state.py` v3 reads DONE only if all of these hold:
  - `.exit` is 0;
  - there is exactly one header, and its `settings.seed` is the registered seed;
  - there are exactly n game lines, and every `(pair_id, half)` is unique;
  - every line is a JSON object;
  - there is exactly one results row with its label, and that row's `candidate_sha256` / `reference_sha256` equal
    the model files as they are now, and its seed matches.

  If everything holds except the model shas, the seat reads **STALE** (v3): a complete seat played by a net that
  has since changed. Its results row exists, so it is not a clear candidate; both launchers refuse it with **87**
  and say so (owner decision; if a training cell was retrained, `clear_step.sh stage3 <cell> --finished
  --cascade …` moves that cell's match directories with it). v2 read this as PARTIAL → 88 and printed a `seat`
  clear command that itself refuses (exit 3). Anything else, including an empty `.exit` or a non-object line, reads
  PARTIAL; it never raises a traceback.

  **Reusing a finished, sealed step is not a re-roll.** Nothing is redrawn; its outputs are the registered ones.
  Guard (d) (CPU) and the pools (read-only) are simply redone on every launch.
- **A finished step whose inputs differ from now is refused, exit 87** (above): an owner decision, never a skip.
- **A finished step whose check fails is refused, exit 89.** For example, the collect's seal sha differs, or its
  take-effect read now fails. This means tampered or damaged, and it is an owner decision.
- **An interrupted or failed step is refused, exit 88.** That covers a `.started` or the step's directory with no
  `.exit`, a non-zero `.exit`, a Stage 0 run directory without a summary row, and a PARTIAL seat. It is never
  silently rerun or blended with a rerun. The refusal prints the one documented recovery command,
  `clear_step.sh` v3 (sha in the table below; v2 `dd07013a9641…`, v1 `4ace011c377f…`):
  - the forms are `stage3 <step> [--finished] [--cascade a,b]`, `stage0 <tag> [--finished]`, `stage0-block
    [--finished]` (summary, runs, decision and both series files: a fresh ABBA block), `seat <arena_dir> <label>`, and (v3)
    `stage3-fingerprint` (above).
  - **Names are validated, never used as paths.** A stage-3 step must be one of the five registered names. A tag
    must look like `r01_off64`. A label is `[A-Za-z0-9_.-]+`, and not `.` or `..`. A seat's arena directory must
    not be a symlink, and must resolve (realpath) inside this bank's `stage2/` or `stage3/`. v1 had a path
    traversal hole, and the review used it three ways: `stage3 ../stage1` moved all of `stage1/`, `stage0
    ../../../rl-run1-pilot` moved the (fake) pilot bank, and `seat <pilot>/arenas/m1 lab` moved a file inside it.
  - **It never deletes or overwrites.** v1's "never deletes" was false in one case: two clears in the same second
    shared one `aborted_<ts>/`, so the second `mv` overwrote the first's games file, and a directory clear could
    fail half-moved with no log line. v2 changes three things:
    - each invocation gets its own `mktemp -d aborted_<ts>.XXXX`;
    - it moves with `mv -n`, then checks that every source is gone and every destination exists;
    - `resume.log` gets `CLEAR-BEGIN` before the moves and `CLEAR-DONE` or `CLEAR-FAILED` after (`CLEAR-FAILED`
      exits 4).
  - **A FINISHED step needs `--finished`** (exit 5 without it). That is a stage-3 step whose `.exit` reads 0, or a
    stage-0 tag with a summary row. The refusal prints what would be moved (v3: `show()` returns 0, so a finished
    step missing its last companion file, such as a v5-finished step without `.inputs`, now exits 5 instead of
    aborting with 1 under `set -e`).
  - **A stage-0 `--finished` clear removes the ROW first** (v3), then the run directory. The row is matched on the
    parsed `tag` field exactly, never a text prefix (`r01_off64` vs `r01_off640`); the old summary is kept in the
    aborted directory and the removed row is logged verbatim in a `CLEAR-ROW` line. An unparsable summary line or a
    failed rewrite refuses (2 or `CLEAR-FAILED` 4) and leaves `summary.jsonl` untouched; an empty or partial
    summary is never installed. A kill after the row goes leaves a run directory without a row: an ordinary 88 that
    the plain `clear_step.sh stage0 <tag>` recovers. A row whose run directory is already gone (a kill inside a v2
    clear, which moved the directory first) is accepted by `--finished` and cleared alone.
  - **A stage-3 step with downstream outputs needs `--cascade`** (exit 6 without it). The downstream steps are:
    - collect → both rfw cells, Stage A, and its four match directories;
    - a cell → Stage A and its two match directories;
    - `arm_P0_ctl` → Stage A.

    `--cascade` must name every downstream step that has outputs. They then move in the given order into the same
    aborted directory.
  - It takes the bank lock, so it cannot run beside a launcher.
  - It refuses a seat whose results row already exists (exit 3).

  Then relaunch.
- **An interrupted arena seat is replayed whole, from a clean slate, with its registered seed.** The arena's own
  resume is not used: `run_rfw_match.sh` and `run_followup_match.sh` both refuse to touch a look whose game log
  exists. A resume that keeps an interrupted seat's finished games over-represents short games, because the games
  finished before the stop are biased short unless every game's randomness is fixed by its deal index. Replaying
  the whole seat discards the partial sample entirely. The replacement is a complete draw of the same deal, so the
  stop cannot select on game length or outcome. The arena appends its results row only at the very end, so a
  PARTIAL seat normally has no row.
- **An interrupted collect or training cell is cleared and rerun whole**, under the same command and seed. It is
  not resumed by the collect's own resume path.
  - There is one window where a finished step is lost: the seal is written, then the `.exit`. A kill between the two
    leaves a `.started` without an `.exit`. The step must then be cleared and rerun, which costs that step but never
    blends it.

Shas (addendum 2026-11-22). Every row is a citation in the evidence verifier's stamp form (prefix `RFW/`, declared
in the evidence log's prefix table), so drift is machine-checked: `CUDA_VISIBLE_DEVICES= python3
scripts/verify_evidence_stamps.py --quiet docs/experiments/2026-11-21-rl-run1-root-full-width.md` must exit 0.

| File | Current version | Previous versions (each kept as `.vN`) |
|---|---|---|
| `RFW/gpu_guard.sh` [`4324d33990f6`] | v7 | `RFW/gpu_guard.sh.v6` [`c8697d517544`], `RFW/gpu_guard.sh.v5` [`30d2226f6a78`], `RFW/gpu_guard.sh.v4` [`b3ffd88cf5f7`], `RFW/gpu_guard.sh.v3` [`ee073f841a24`], `RFW/gpu_guard.sh.v2` [`997969c87bf3`], `RFW/gpu_guard.sh.v1` [`a0a2f869a719`] |
| `RFW/launch_lib.sh` [`bbdcdb37ed6e`] | v4 | `RFW/launch_lib.sh.v3` [`b9723e9355cc`], `RFW/launch_lib.sh.v2` [`218d215dd305`], `RFW/launch_lib.sh.v1` [`9bae510f71b0`] |
| `RFW/seat_state.py` [`15022d1b3aa5`] | v3 | `RFW/seat_state.py.v2` [`730a1c958edf`], `RFW/seat_state.py.v1` [`1577deaa9a97`] |
| `RFW/clear_step.sh` [`e6a3f324e61e`] | v4 | `RFW/clear_step.sh.v3` [`e2df58427ed4`], `RFW/clear_step.sh.v2` [`dd07013a9641`], `RFW/clear_step.sh.v1` [`4ace011c377f`] |
| `RFW/verify_outputs.py` [`50ca4fa4b77c`] | v1 | — |
| `RFW/stage0/run_stage0.sh` [`7d4ff2022a2d`] | v8 | `RFW/stage0/run_stage0.sh.v7` [`4f851a9c9c4f`], `RFW/stage0/run_stage0.sh.v6` [`8aafea169066`], `RFW/stage0/run_stage0.sh.v5` [`eb1dd4d3163b`], `RFW/stage0/run_stage0.sh.v4` [`205bd4f0db30`], `RFW/stage0/run_stage0.sh.v3` [`d0e3bcb6952b`], `RFW/stage0/run_stage0.sh.v2` [`39303c3d7620`], `RFW/stage0/run_stage0.sh.v1` [`a84f4c7cfa63`] |
| `RFW/stage0/run_stage0_all.sh` [`89a04df9ae8f`] | v6 | `RFW/stage0/run_stage0_all.sh.v5` [`5163ea29e2cb`], `RFW/stage0/run_stage0_all.sh.v4` [`09c9eea07472`], `RFW/stage0/run_stage0_all.sh.v3` [`0d9a81de48f5`], `RFW/stage0/run_stage0_all.sh.v2` [`98a9bea18970`], `RFW/stage0/run_stage0_all.sh.v1` [`51482410ec7c`] |
| `RFW/stage0/stage0_decide.py` [`d2ae12874881`] | v6 | `RFW/stage0/stage0_decide.py.v5` [`cf5e1cbb397f`], `RFW/stage0/stage0_decide.py.v4` [`f61952c2b32a`], `RFW/stage0/stage0_decide.py.v3` [`4771354e2117`], `RFW/stage0/stage0_decide.py.v2` [`4c02fc7565dd`], `RFW/stage0/stage0_decide.py.v1` [`fb099bed9a35`] |
| `RFW/stage0/fab_test/run_fab.py` [`5360bc2ceb99`] | v3 | `RFW/stage0/fab_test/run_fab.py.v2` [`3abfd33405ac`], `RFW/stage0/fab_test/run_fab.py.v1` [`93e32555a306`] |
| `RFW/stage1/run_stage1.sh` [`b419265e881d`] | v5 | `RFW/stage1/run_stage1.sh.v4` [`b9ab2fae46ed`], `RFW/stage1/run_stage1.sh.v3` [`0539b8e1e4f4`], `RFW/stage1/run_stage1.sh.v2` [`3778d06ddad6`], `RFW/stage1/run_stage1.sh.v1` [`d83c8601b3a4`] |
| `RFW/run_rfw_match.sh` [`29aedb0040c9`] | v6 | `RFW/run_rfw_match.sh.v5` [`e8e85a601c32`], `RFW/run_rfw_match.sh.v4` [`91ef5a8f87ed`], `RFW/run_rfw_match.sh.v3` [`b24a435b24e5`], `RFW/run_rfw_match.sh.v2` [`0487d857e379`], `RFW/run_rfw_match.sh.v1` [`0c192e052082`] |
| `RFW/stage2/run_stage2.sh` [`49c17f057987`] | v8 | `RFW/stage2/run_stage2.sh.v7` [`a9d05768c5c9`], `RFW/stage2/run_stage2.sh.v6` [`d6839c0ca884`], `RFW/stage2/run_stage2.sh.v5` [`e831f6274f47`], `RFW/stage2/run_stage2.sh.v4` [`174495f8c6eb`], `RFW/stage2/run_stage2.sh.v3` [`ba70d73eb7b7`], `RFW/stage2/run_stage2.sh.v2` [`c07fe428aa39`], `RFW/stage2/run_stage2.sh.v1` [`5ba517c5ae0e`] |
| `RFW/stage3/launch_stage3.sh` [`f154bdbe7bda`] | v8 | `RFW/stage3/launch_stage3.sh.v7` [`2880311d902c`], `RFW/stage3/launch_stage3.sh.v6` [`86494309a5f5`], `RFW/stage3/launch_stage3.sh.v5` [`b4a0cf0e46f8`], `RFW/stage3/launch_stage3.sh.v4` [`f5ea61297307`], `RFW/stage3/launch_stage3.sh.v3` [`8bfe478f0db8`], `RFW/stage3/launch_stage3.sh.v2` [`a519b39f2200`], `RFW/stage3/launch_stage3.sh.v1` [`95c8a66dbc59`] |
| `RFW/gpu_scan.sh` [`8b4261b0a653`] | v5 | `RFW/gpu_scan.sh.v4` [`949408e727e7`], `RFW/gpu_scan.sh.v3` [`23a504a858bc`], `RFW/gpu_scan.sh.v2` [`38c6db9bbf69`], `RFW/gpu_scan.sh.v1` [`ec611f4b3a5d`] |
| `RFW/guard_test/guard_test.sh` [`83a3b679efe0`] | v11 | `RFW/guard_test/guard_test.sh.v10` [`78d1b3f40d22`], `RFW/guard_test/guard_test.sh.v9` [`167b79eeee93`], `RFW/guard_test/guard_test.sh.v8` [`c835c5280f1c`], `RFW/guard_test/guard_test.sh.v7` [`ac4196580a48`], `RFW/guard_test/guard_test.sh.v6` [`275c91d526df`], `RFW/guard_test/guard_test.sh.v5` [`d9b84ec2eba5`], `RFW/guard_test/guard_test.sh.v4` [`d4ad57efc6f3`], `RFW/guard_test/guard_test.sh.v3` [`b40b08d3d30c`], `RFW/guard_test/guard_test.sh.v2` [`730d515388e3`], `RFW/guard_test/guard_test.sh.v1` [`621a57547b22`] |
| `RFW/guard_test/run_mutants_v11.py` [`d3136aea234f`] | v1 | — |
| `RFW/SHA256SUMS` [`1bd3bec3f25f`] | v5 | `RFW/SHA256SUMS.v4` [`5ebf76b14621`], `RFW/SHA256SUMS.v3` [`5905c14b95f4`], `RFW/SHA256SUMS.v2` [`5816a4677421`], `RFW/SHA256SUMS.v1` [`ac86aa6cbfe9`] |

**Addendum changes beyond the guard (2026-11-22):**
- **Stage-0 series files.** `clear_step.sh stage0-block` now moves `decision.series.json` and
  `decision.fresh_abba.series.json` too (v9 left the series behind, and the next decide refused with no clear form).
  When FRESH_ABBA_REQUIRED fires, `stage0_decide.py` banks the discarded block's series as
  `decision.fresh_abba.series.json` (written once, never overwritten), so a discarded block still feeds the
  re-derivation of the 900 s threshold. `gpu.csv` samples now carry the sampler's own timestamp (`run_stage0.sh` v8
  queries `timestamp,utilization.gpu,memory.used`); v7 placed sample i at t0 + 0.5 i, but the sampler starts about
  1 s before t0, so the whole curve was shifted. `decision.series.json` says which placement each run used
  (`gpu_t_source`: `sampler`, or `t0+0.5i` for rows without timestamps).
- **One aborted directory per stage-0 `--finished` clear** (v3 made two: one for the old summary, one for the run).
- **`verify_outputs.py` is pinned to the repo's digest by a test.** The guard test writes a stage with nested
  shard files in the fake HOME, has the repo's own `write_manifest` (the arena worktree at `ARENA_SHA`, imported
  read-only by the venv interpreter with CUDA hidden) seal it, and requires `verify_outputs.py` to verify it (0),
  then to refuse one changed shard byte (1). The reviewer's surviving mutant (the digest omits the relative file
  names) is now killed (M46).

**Tested** (`guard_test/guard_test.sh` v11, run on the staging copy; output `guard_test/guard_test.txt`; v10 with
its output `guard_test.v10.txt`, and v9 with `guard_test.v9.txt`, kept). All 191 checks pass. The whole test runs in its own user + pid namespace: a fresh `/proc` (the scan sees only
the test's own processes, never another agent's runs); `/dev/full` (a real character device, rdev 1:7) bind-mounted
over `/dev/dxg`, so no case can open the real node (a:102) while the scan's rdev match is still exercised; every
capability dropped for good; SIGINT / SIGQUIT / SIGTERM reset to default (`unshare` leaves them ignored in its child).
The harness asserts each of these before any case runs. It waits, before starting, while the coordinator's
`GPU_CHECK_HOLD` file exists (see the harness rule above). Every case whose expected result depends on the argv
clause reads `/proc/<pid>/environ` of the process it started and fails as VACUOUS if the CPU-only exemption would
apply; only the exemption cases themselves expect it. New cases:
- the idle-baseline rule: no `idle_mib` → 8; `idle_mib` 5000 → 8; idle + 600 → 8; idle − 600 → 8; idle + 100 with
  utilisation 0 → passes; idle + 512 exactly → passes; one utilisation sample of 35% in five → 8; at most 9% →
  passes; utilisation `N/A` or nothing → 11; two GPU lines, `0` and `N/A` → 11 (a non-integer line is never
  skipped); `RFW_MEM_WAIT_S` waits through idle + 600 twice, then passes;
- the REAL `nvidia-smi` (read-only, before the namespace): both probe commands print integers and exit 0;
- the scan (v5): 23 argv unit cases (python short-option clusters `-um` / `-bm` / `-uB` / `-Wignore` / `-W x` /
  `-X dev` / `-mmod`; `deepfin` as argv[0] and as `python …/deepfin`; `bash --norc` / `-O x` / `+O x` / `-o x` /
  `--rcfile x`; `dotnet exec --depsfile x`; and, unmatched, `-uc '…'`, `-c '… # …/arena_standard.py'`, `bash -lc`,
  `python -`, and the stated limits `-m cProfile` and runpy via `-c`); an enumeration that sees no process of this
  user → refused; a lister error list mixing ENOENT with EACCES → refused; a real single-argument device holder
  (argv `['cat']`) and a holder whose cmdline reads empty → refused by the device clause; a LISTED parent launcher
  (not exempt) starting a guarded child → the child passes (ancestors excluded);
- the scan (v4 and earlier): listed scripts under `python3 -u`, `python3 -I -X dev`, `python3 -m chess_anti_engine…`, `lc0` as
  argv[0], `dotnet Ceres.dll`, `count_wrap.py`, a sibling copy of a bank launcher → each refused (all started
  WITHOUT `CUDA_VISIBLE_DEVICES`); a name only as a substring, `bash -c '…sha256sum …/arena_standard.py'`, and
  `python3 -c '…' …/arena_standard.py` → each passes; a foreign holder of the device → refused; a holder of the same
  device through another path (`/dev/full`) → refused; a process holding a REGULAR file named `/dev/dxg` in its own
  mount namespace → NOTE, passes; the scanner's own `/dev/dxg` missing or not a character device → refused; a
  non-dumpable device holder (access says yes, readlink denied) → refused; `/usr/bin/ssh-agent` / `/usr/bin/gpg-agent` (non-dumpable) → NOTE, passes; the same process as plain
  `ssh-agent` → refused; three race cases with an interposed lister (ENOENT → skipped; Permission denied → refused;
  no error text → refused); a launcher's own re-check beside its own running child → passes, while a sibling copy
  started from the same shell line → refused;
- the CPU-only exemption: a matched script started with `CUDA_VISIBLE_DEVICES=` (empty) → NOTE, passes; with `=0`,
  with only `CUDA_VISIBLE_DEVICES_X=` or `XCUDA_VISIBLE_DEVICES=`, or with two entries (`=` then `=0`) → refused; an
  exempt matched process that holds the device → refused (the exemption is argv-only); an environ that cannot be
  read (interposed reader) → refused, with the real reader as control → exempt;
- the environment: `CAE_COMPILED_LEGAL_POLICY=1` on a Stage 3 relaunch → 87 naming it, with the `unset` line and the
  ONLY-env verdict; `OMP_NUM_THREADS=3`, `TORCH_LOGS`, `PYTORCH_CUDA_ALLOC_CONF` → not refused, every step and seat
  skipped; the fingerprint holds no `OMP_` name and each step's `env.observed` exists; a stage-0 row differing only
  by `CAE_DUP_STATS` → 87 naming it, "Do NOT clear"; a row whose SL base changed → no ONLY-env verdict;
- series and clears: `stage0-block --finished` moves the decision and both series files; a stage-0 `--finished`
  clear leaves one aborted directory; `run_stage0.sh` writes 3-field `gpu.csv` rows and an `env.observed`;
- `verify_outputs.py` against the repo's own `write_manifest` (0), and after one changed shard byte (1);
- the between-step re-guard (`guard_retry`, against a stub guard): a holder gone after 10 s → proceeds, with the
  reason logged once; a persistent holder with `RFW_REGUARD_WAIT_S=12` → refused at the deadline, with a deadline
  line; a revoked release → refused at once (bounded at `RFW_REGUARD_WAIT_S=12`, so a mutant that retries it is
  caught in seconds rather than hanging the harness); `RFW_REGUARD_WAIT_S=abc` → 12; only the memory probe failing → 11;
- the nested top guard (a second fake bank: the real `launch_lib.sh` / `gpu_guard.sh` / `gpu_scan.sh`, the real top
  of `run_stage0.sh`, a minimal parent at the registered path, and a holder appended to that bank's scan copy): a
  nested child plus a holder that goes away → proceeds after the wait, reason and "passed after" logged; plus a
  persistent holder → refused at the deadline; a direct start plus a holder → single-shot refusal at once; four
  forgeries → single-shot refusal at once (a hand-set marker, an unregistered parent script, the lock fd closed, a
  grandchild whose PPID differs); a no-holder control → nested with a 0 s wait. The real `run_stage0_all.sh` and
  `run_stage2.sh` runs through stubs assert that every child ran its top guard nested (0 s wait), and the stage-0
  rows carry `"guard_mode": "nested", "guard_wait_s": 0`.

`stage0/fab_test/run_fab.py` v3: 17 cases, 0 WRONG (new: FRESH_ABBA_REQUIRED banks the discarded block's series and
writes no decision; `gpu.csv` placed by the sampler's own timestamps, the first sample 1.25 s before t0; guard waits
reported per leg, with the in-block flag set only for legs 2–4).

**Mutants v11** (`guard_test/run_mutants_v11.py`). Each mutant edits a fresh copy of the staging bank, and
`guard_test.sh` runs against the copy, 4 at a time. The runner asserts the expected count (`--expect N`) and exits
non-zero unless the control survives with 0 WRONG and every other mutant is CAUGHT: a survivor, a mutant that does
not apply, a run without a total line, or a timeout all fail it.
- **Scan series** (M0 + G1–G33, on `gpu_scan.sh` v5): every G mutant CAUGHT. Two needed a harness fix first, and
  each fix was rerun solo with the control: G19 (rdev compared by its major number only) first aborted the harness on
  an unbound variable when it read a hang pid; the read is now guarded, and G19 is CAUGHT with 27 WRONG. G32 (`-c`
  inside a cluster does not end the scan) first survived, because no case ended the command string with a listed
  path; two such cases were added, and G32 is CAUGHT with 2 WRONG. G10, G11 and G33 are each killed by their own
  case, not only by a neighbour's.
- **M-series** (M1–M68; M5 and M9 retired): `MUTANTS RAN=67 EXPECTED=67`. 63 CAUGHT in the series; three were not:
  - M3 (`run_stage0.sh` without its guard prologue) survived, because its pattern still named the guard line that
    the restructure moved into `rfw_top_guard`. It was retargeted to replace `rfw_top_guard` with the single-shot
    guard: CAUGHT, 20 WRONG.
  - M54 (a non-integer utilisation line skipped: fails open) survived, because no case printed an integer line
    beside a non-integer one. v11 adds that case: CAUGHT, 1 WRONG.
  - M57 (a revoked release line retried) timed out. The revoked-release case ran at the default 900 s retry,
    which the mutant spent in full. v11 bounds that case at 12 s: CAUGHT, 5 WRONG.
  The harness gained those two cases mid-series. Both are additive (a new case cannot un-catch a mutant), and M0,
  M3, M54 and M57 were rerun solo on the final v11: `MUTANTS RAN=4 EXPECTED=4 NOT_OK=0`, control 191/191.
  Outputs: `guard_test/guard_mutants_v11_scan_v5.txt`, `…_G19_v5.txt`, `…_G32_v5.txt`,
  `…_mseries_final.txt`, `…_rerun4.txt`.


**Tested, previous round** (`guard_test/guard_test.sh` v9 `167b79eeee93…`, output `guard_test/guard_test.v9.txt`; v8 `c835c5280f1c…`, v1 `621a57547b22…`,
v2 `730d515388e3…`, v3 `b40b08d3d30c…`, v4 `d4ad57efc6f3…`, v5 `d9b84ec2eba5…`, v6 `275c91d526df…` and v7
`ac4196580a48…`, with their outputs, are kept as `.v1` … `.v8`). Since v6 the test **exits non-zero if any case is WRONG**. v5 exited 0 even with WRONG lines, so a
mutant harness must key on the exit code.
Nothing in the test can reach the GPU:
- `CUDA_VISIBLE_DEVICES` is empty;
- HOME is a fake;
- `nvidia-smi`, `ps` and `python` are shadowed by stubs first on PATH;
- the launchers' venv python is a stub too.

Every launcher runs through one wrapper. Inside the environment it is about to run in, the wrapper asserts that
`$HOME` is the fake and that `command -v` of `python`, `nvidia-smi` and `ps` gives the stubs. If any of this fails, the
whole test aborts before any case runs. For a collect call the stub records its arguments and fabricates a tiny
finished collect: step-synchronous fake play in real time, 600 games, 128 slots. For the launcher's summary step it
runs the real python. All 100 checks pass (`guard_test.sh` v9; 80 in v8):
- memory 10,971 MiB, with every other check clear → refused for memory (the test greps "GPU memory in use");
- `nvidia-smi` printing nothing, printing `N/A`, or exiting 2 while printing 500 → 11;
- `ps` exiting 1, or printing nothing → 10;
- a missing queue marker → 8;
- each of the three script names running → 8;
- the name present only as a substring → passes;
- all clear → passes;
- no operator release file → 8;
- a release file without the requested stage → 8;
- a release file with the requested stage → passes;
- a stage name outside stage0–3 → 8;
- the lock held → 9;
- fd 9 open on `/dev/null` while an unrelated process holds the lock → 9;
- fd 9 opened on the lock file by the launcher itself (not inherited), with an unrelated holder → 9;
- `run_stage0.sh` end to end with the stub:
  - three collect calls, each carrying `--root-full-width`;
  - exit 0;
  - a summary row whose `steady_rows_per_s` comes from `steady_rate.py` on the real progress log and the stub's
    first-forward time. It must lie within **[−3%, +2%]** of the stub's own measured full-concurrency rate. The stub
    records its real step times, because sleep overhead makes the nominal 128 / 0.1 s an overstatement. In v4 the
    stub plays 600 games per process, about 15 s of full concurrency.
    - The bar was tightened from ±5% after 5 consecutive runs (`guard_test/tolerance_5runs.txt`). Their ratios were
      0.9966, 0.9963, 0.9954, 0.9959 and 0.9942.
    - The final v6 run read 0.9903 (3,792 against 3,829). The readings skew low: mean 0.9950, minimum 0.9903,
      never above 1. A symmetric ±2% would leave about 2× margin on the low side but would waste the high side.
    - So v7 uses the asymmetric bar [−3%, +2%] the reviewer recommended. Low-side noise gets about 3× the observed
      worst miss, and an estimator that starts reading HIGH is still caught at +2%. A high reading would be the
      direction that flatters the knob, given the Confounds lean.
    - The final v7 run reads 0.9962.
    - v3 allowed ±25% on a 6 s run. That read 3,990 against the nominal 3,840.
    - This case found a real bug: the progress sampler could miss the final games. The launcher now takes one last
      sample after every process exits.
- `run_stage0.sh` with memory busy → 8, and the stub was never called;
- **the guard file missing** → 8 with no stub call, for each of `run_stage0.sh`, `run_stage0_all.sh`,
  `run_stage1.sh`, `run_rfw_match.sh` and `launch_stage3.sh`;
- **the guard present but failing to `source`** (a syntax error) → 8 with no stub call, for each of
  `run_stage0.sh`, `run_rfw_match.sh` and `launch_stage3.sh`;
- a nested launch, where the parent holds the bank lock → the child passes;
- `launch_stage3.sh` without the operator's stage3 line → 8, and with it but no `OWNER_GO` → 90;
- `run_rfw_match.sh` writing into the frozen pilot bank → 94;
- `run_stage0.sh` where process 2 exits 3 before creating its run directory: the launcher exits non-zero,
  `run.exit` holds that same code, `REMOVED_ROW_DATA.txt` is written, and `p2.exit` holds 3;
- `launch_stage3.sh` with every prerequisite faked: the stub collect revokes the stage3 release mid-run. The launcher
  refuses before the first training cell, with exit 8. `collect.exit` is 0, `stage3.exit` is 8, and no training cell
  has started;
- **Stage 3 resume** (v7), continuing from that refusal:
  - restore the release and relaunch: the collect is SKIPPED (no collect call); cells 1 and 2 run; the stub revokes
    the release during cell 2, so the launcher refuses before cell 3 (8);
  - restore and relaunch: only cell 3 runs. Guard (d) then needs the PIN worktree, which the fake HOME does not
    have, so the run exits 1. `resume.log` holds the 4 skips;
  - remove cell 3's `.exit`, as if it were SIGKILLed after starting → 88, with the `clear_step.sh stage3 arm_P0_ctl`
    command, and nothing runs;
  - `clear_step.sh stage3 arm_P0_ctl` → the directory and its `.started` are in `aborted_<ts>/`, and nothing is left
    in place;
  - tamper with the finished collect's `stats.json` → 89, and nothing runs;
- **Stage 2 resume**: the release is revoked during smoke seat A, so seat B refuses (`run_stage2.sh` 1, one seat
  played). The relaunch skips seat A, plays the other 7, and writes `STAGE2_DONE`;
- **Stage 0 parent resume**: the release is revoked during r02, so r03 refuses (1); r01 and r02 are summarised. The
  relaunch skips r01 and r02 (no calls for them) and runs r03 and r04;
- **TERM** to `run_stage1.sh` while its child ignores TERM for 2 s: the marker reads 143, its mtime is after the
  child's exit, and the child is gone;
- **W2**, the reviewer's case: memory goes 9,000 → 500 during the wait while a foreign `arena_standard.py` appears
  → 8 (GPU work running);
- `RFW_MEM_WAIT_S=abc` with memory busy → 12, with no endless wait; `RFW_MEM_WAIT_S=-5` → 12;
- two GPU lines: 500 and 9,000 → 8; 500 and `N/A` → 11;
- `launch_lib.sh` missing → `run_stage0.sh` refuses (8), and nothing runs;
- **v8 cases**:
  - a finished cell whose own record names another source collect → 89 (the cross-check), and nothing runs;
  - the reviewer's GEN-A/GEN-B repro, with the collect moved by hand → the collect is redone, then the cell refuses
    87 instead of being skipped;
  - `clear_step stage3 collect` without `--finished` → 5;
  - with `--finished` → 6, naming `--cascade arm_P0_rfw,arm_Zq_rfw`;
  - with one of the two named → 6;
  - with both named → everything moves into one aborted directory, and nothing is left;
  - `OWNER_GO` edited → the fingerprint refuses (87);
  - the three traversal reproductions, plus a symlink out of `stage3/` → each refused (2) and the target intact;
  - two clears in the same second (time faked) → two aborted directories, and both files kept;
  - the SL base changed → `run_stage0_all.sh` refuses 87 on r01's reused row, and 87 again when `decision.json`
    exists;
  - the openings changed → a DONE stage-2 seat refuses 87;
  - `clear_step stage0 r01_off64` → 5; with `--finished` → the run moves aside, its row is removed, and the old
    summary is kept;
  - `seat_state.py`: a complete seat → DONE; duplicate `(pair_id, half)`, two headers, the wrong header seed, an
    empty `.exit`, or a non-object line → PARTIAL with no traceback (5 cases); a changed candidate file → STALE;
  - `count_wrap.py` running → 8;
  - the lock held → the exit-9 message carries the `fuser -v` diagnosis;
  - `run_fab.py` (the ABBA gap rule, below) → 0 WRONG;
  - the TERM case now also asserts that the stub's whole process GROUP is gone (`kill -0 -- -<pgid>` fails).
  - **New in v9:**
    - the fake HOME now holds a fake PIN worktree (a one-commit throwaway git repo whose HEAD is the fake
      `PIN.txt`) and a fake Stage A, so a Stage 3 run goes through guard (d), Stage A and all 8 match seats on
      stubs, and a third relaunch skips every step and seat (the collect re-hashed) with no call made;
    - a tracked edit in the PIN worktree → 96; `CAE_COMPILED_LEGAL_POLICY` set → the fingerprint refuses 87;
    - an output the collect's manifest records but the seal does not cover, changed → the SKIP's re-hash refuses 89;
    - step outputs without `inputs.fingerprint` → 87 (v6 missed this, above);
    - `clear_step stage3 arm_P0_ctl` with Stage A's outputs → 6 naming `--cascade stage_a`; collect's cascade now
      names all four match directories;
    - a finished step without `.inputs` → `clear_step` exits 5, not 1;
    - a complete stage-2 seat whose SL base changed → STALE, 87 (not 88); `seat_state.py` on a changed candidate →
      STALE;
    - `stage0_decide.py` changed → a stage-0 row is not reused (87);
    - stage-0 clears: `r02_on64 --finished` beside a row tagged `r02_on640` removes only the exact row and logs it
      verbatim in `CLEAR-ROW`; a row whose run directory is gone → `--finished` recovers it; a run directory without
      its row → a plain clear recovers it; an unparsable summary line → refused 2 with the summary unchanged;
    - `clear_step stage3-fingerprint`: refused 2 while any step output exists; moves the fingerprint when none does;
    - the four traversal cases and the two fingerprint-clear cases run on a THROWAWAY copy of the fake HOME, so a
      regression that moves a directory cannot break the cases after it (M17's v8 run had no total line for this
      reason);
    - **Stage 3's `seat()` tested directly.** It is not Stage 2's code: its `want` string and its arguments
      (candidate and reference nets, the follow-up match launcher) differ. The harness extracts the block between
      the `# >>> seat()` / `# <<< seat()` markers in `launch_stage3.sh` and runs it with stubbed `reguard` /
      `runkid` (the stub writes a complete seat): ABSENT → plays; a rerun → skipped; the match launcher changed → 87;
      the candidate net changed → STALE 87; a truncated game log → PARTIAL 88 with the `seat` clear command.

**Mutants v9** (`guard_test/run_mutants_v9.py`, output `guard_test/guard_mutants_v9.txt`; shas in the table below).
- **Method.** As v8: each mutant edits a fresh copy of the bank, `guard_test.sh` runs with `GT_SRC` at the copy, 4
  at a time, on the FINAL files. New in v9: a run that prints no `TOTAL PASS=… WRONG=…` line is reported as a
  HARNESS FAILURE and never counted as caught.
- **Control.** M0, no mutation, exits 0 with 100/100.
- **All 38 mutants are caught** (exit 1, each with a total line; no harness failure). M1–M27 are v8's (M17 now
  reports a total line: 99/100, the `../stage1` case WRONG). M22's v8 pattern did not apply to `seat_state.py` v3,
  so the full run reported it NOT APPLIED; it was rewritten (the state line ignores the shas) and rerun alone:
  3 WRONG. New in v9:
  - M28, `seat_state` never reports STALE → the stage-2 STALE and `seat()` STALE cases fail;
  - M29, the stage-0 row removal matches the tag by prefix → the `r02_on64` / `r02_on640` case fails;
  - M30, `show()` returns 1 again → the finished-without-`.inputs` case fails;
  - M31, the PIN-worktree tracked-edit check removed;
  - M32, the environment left out of the Stage 3 fingerprint;
  - M33, a collect SKIP without the output re-hash;
  - M34, Stage 3 `seat()` reuses a DONE seat without comparing its inputs (caught by the direct `seat()` case);
  - M35, the outputs-without-fingerprint check back to v6's `ls` over globs;
  - M36, `stage0_decide.py` does not bank the series (`run_fab.py` fails);
  - M37, `clear_step stage3-fingerprint` allowed while outputs exist;
  - M38, `stage0_decide.py` left out of `s0_inputs`.

**Mutants v8** (superseded by v9 above; `guard_test/run_mutants_v8.py` `c841aecdad5b…`, output `guard_test/guard_mutants_v8.txt`).
- **Method.** Every mutant was run on the FINAL files of this revision. Each mutant edits a fresh COPY of the bank,
  never the bank itself, and `guard_test.sh` runs with `GT_SRC` pointing at that copy, 4 at a time. The verdict is
  keyed on the exit code.
- **Control.** M0, no mutation on a copy, exits 0 with 80/80.
- **All 27 mutants are caught** (exit 1). Mutants carried over from earlier rounds:
  - M1, the v4 cleanup;
  - M2, any open fd 9 accepted;
  - M3, the other agent's hole;
  - M4, the Stage 3 re-check made a no-op;
  - M5, `ps` failing open;
  - M6, the `nvidia-smi` exit code ignored;
  - M7, a `.started` counted as done;
  - M8, a finished collect rerun;
  - M9, the reviewer's W2 mutant;
  - M10, `stop_kids` without the wait;
  - M11, `stop_kids` a no-op;
  - M12, `RFW_MEM_WAIT_S` not validated;
  - M13, the first GPU line only.
- **New in this revision:**
  - M14, Stage 3 step inputs not compared → the GEN-A/GEN-B case fails;
  - M15, the run fingerprint not compared;
  - M16, `--cascade` not required;
  - M17, the stage-3 step name not validated → `../stage1` is moved;
  - M18, the seat directory not confined;
  - M19, a fixed aborted directory plus `mv -f` → the same-second clears collide;
  - M20, a finished step cleared without `--finished`;
  - M21, `(pair_id, half)` uniqueness not required;
  - M22, the results-row shas not compared;
  - M23, `stop_kids` signalling the pid, not the group (the reviewer's mutant) → the group-gone assertion fails;
  - M24, the stage-0 row inputs not compared;
  - M25, the stage-2 seat inputs not compared;
  - M26, the ABBA gap rule off;
  - M27, the cell's source-collect cross-check off.
- **One note.** M17's run printed no total line: moving the fake `stage1/` broke a later case, and the test exited
  1 there. The `../stage1` case itself is WRONG in that run's output. v8's runner counted that as CAUGHT; v9's
  reports a run without a total line as a HARNESS FAILURE, and runs the traversal cases on a throwaway copy.

**Mutants v7** (superseded by v8 above; `guard_test/run_mutants_v7.py` `eaae8995aa30…`, output `guard_test/guard_mutants_v7.txt`). The
verdict is keyed on the test's exit code, and each restore is verified by md5. **All 13 are caught** (exit 1):
- M1, the v4 cleanup → the process-2 case fails;
- M2, any open fd 9 accepted → the `/dev/null` case fails;
- M3, the other agent's hole → 2 WRONG;
- M4, the Stage 3 re-check made a no-op → 3 WRONG;
- M5, `ps` failing open → 2 WRONG;
- M6, the `nvidia-smi` exit code ignored → 1 WRONG;
- M7, a step skipped without checking its `.exit` (a `.started` counts as done) → the 88 case fails;
- M8, a finished collect rerun → 3 WRONG;
- M9, the reviewer's W2 mutant (the re-check skips the `ps` scan once it has waited) → the W2 case fails;
- M10, `stop_kids` sends TERM but does not wait → the TERM case fails;
- M11, `stop_kids` a no-op → 2 WRONG;
- M12, `RFW_MEM_WAIT_S` not validated → 2 WRONG;
- M13, only the first GPU line read → both two-GPU cases fail.

M12 first survived on the `abc` case: the pattern `rc=12` also matched the `rc=124` of `timeout`. The 11 exit-code
patterns are now anchored (`rc=12$`). M12 was rerun against the anchored test, with both cases WRONG
(`guard_mutants_v7_M12_rerun.txt`). Anchoring can only add WRONG lines, so the other twelve verdicts stand.

**Mutants v6** (`guard_test/run_mutants_v6.sh`, output `guard_test/guard_mutants_v6.txt`). The verdict is keyed on
the test's exit code, and each restore is by copy, checked by md5. All six are caught (exit 1):
- M1, the v4 cleanup (`find` on a missing directory, inside the trap) → the process-2 case fails;
- M2, the reviewer's mutant (any open fd 9 accepted) → the `/dev/null` case fails;
- M3, the other agent's hole re-created in `run_stage0.sh` → 2 WRONG;
- M4, `launch_stage3.sh`'s re-check made a no-op → the revoke case fails;
- M5, the `ps` scan failing open → both `ps` cases fail;
- M6, the `nvidia-smi` exit code ignored → the exit-2 case fails.

**Earlier mutants** (`guard_test/guard_mutants.txt`; the restore was checked by md5):
- with the memory check disabled, the two memory cases fail;
- with the release-file stage check disabled, the "without the stage" case fails;
- with the other agent's hole re-created in `run_stage0.sh` (no `set -euo pipefail`, no existence check, no
  `|| exit 8`), both the guard-missing case and the guard-unloadable case for that launcher fail. This is in
  `guard_test/guard_mutants_v5.txt`.

## Stage 0 — cost (GPU)

**Question.** What sims count `S_eq` makes knob-on cost the same wall per stored position as knob-off at 64?

**Design.**
- The pilot's collect flags (`--full-prob 1.0 --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450`, production
  Syzygy pair, the pilot's SF and openings) and the SL base (`a39f900c…`).
- Main `fd8c0ce08`, in the detached worktree `rl-run1-rfw-arena-fd8c0ce08`.
- **One run** is three independent collects × 128 games in flight × 384 games each. This is the collect-throughput
  bench's method (`mp_sims.sh`), and the doc's table says it is what `--collect-procs 3` launches.
  - Forward rows are counted by `count_wrap.py` v2 (`f1528df4eb34…`): that bench's counter (v1, byte copy,
    `fcb2861aa1a9…`) plus the time of the first forward.
  - nvidia-smi samples every 500 ms.
- **Every run uses the same seeds (2026112121/22/23) and the same openings.** Only `--sims` and the knob change.
- **Measured per run** (`stage0/summary.jsonl`):
  - **full-concurrency stored positions/s, the registered cost quantity** (`stage0/steady_rate.py` `41f5006bb850…`, v3).
    - **Window.** A process plays G games, 128 at a time, refilling a freed slot with the next dole index. So game d
      starts at play start if d < 128, and otherwise at the (d − 127)-th finish. The process's full-concurrency
      window runs from its first finish to finish #(G − 128).
    - **Rate.** The rate is 128 × (Σ rows / Σ durations) over the games that start and finish inside that window,
      and the three processes are summed. Because the search is batched, every in-flight game advances one stored
      row per network step, so at full concurrency the process generates 128 × one game's row speed.
    - **Inputs.**
      - Finish times come from `logs/games_progress.txt`: each process's `games.jsonl` line count, sampled every
        second, plus one final sample after the processes exit.
      - Play start is the process's first network forward, written by `count_wrap.py` v2. That is the only change
        from the bench's counter, which is kept as `.v1`.
    - **No per-row timestamps exist.** `games.jsonl` carries none, and neither does the move journal. So spreading
      each game's rows uniformly over its life is the only per-row alternative. It is reported, not registered,
      because it over-counts when the tail runs faster per game.
    - **Rows over the whole wall** are reported as a secondary. They include model load and the low-concurrency
      tail, which pull the ratio toward 1.
    - **Why v3 replaces v2.** v2 took the rows of games finishing between the 10% and 90% game counts. That window
      ends at finish #346 of 384, so about its last 30% is the falling-concurrency tail. And it counts rows of games
      that FINISH in the window, not rows GENERATED in it.
      - In simulation (`stage0/sim_estimator.py`; rows per game drawn from the pilot's 8,000 games; step-synchronous
        play; 200 runs per case; output `sim_estimator.json`), v2 read +4.3% above the
        planted full-concurrency rate when per-board speed is constant, and +17.1% when total
        throughput is constant.
      - It would cancel in the on/off ratio only if both arms shared the throughput-vs-concurrency profile, which
        the knob changes.
      - v3 reads +0.1% and +0.2% in the same two cases. The uniform-spread variant reads
        +0.0% and +3.3%.
    - **Tests** (`stage0/fab_test/test_steady_rate.py`, 8/8 pass). On fabricated progress logs with known per-game
      timing, v3 returns the planted rate within 1% in both models at G = 384 and 512, and excludes 300 s of model
      load. Too few games are refused. **The mutant test shows the v2 estimator missing the planted rate by more
      than 1%.**
    - **Run size, re-derived from the estimator's own variance** (same simulation, 15% per-step timing jitter). The
      SD of a run's estimate (3 processes) is 0.78% at G = 384, 0.59% at 512 and 0.45% at 768.
      - With two runs per configuration, the ratio's SD at 384 is about 0.78%, so its 95% interval is
        about ±1.5%, inside ±3%. **384 games per process is kept.**
      - Real run-to-run noise beyond this model is caught by the 2% repeat gate.
    - **Rows ≠ plies.** A forced ply or a tablebase ply advances a game without storing a row. So a game's row speed
      at full concurrency varies by about 1% from game to game, around 128 × rows per step. This is the same in both
      arms, so it does not move the on/off ratio.
  - forwarded positions/s;
  - forwarded evaluations per stored position;
  - evaluations per position computed as sims + mean prelude, from the search's own `mean_root_prelude_visits`;
  - mean eval batch size;
  - mean and peak GPU utilisation and memory;
  - `buffer_full_flushes`.

**Take-effect checks.** Every run must pass all three, or it fails:
- the search's own `mean_root_prelude_visits` is > 0 when the knob is on and absent when it is off (read from what
  the consumer recorded, not from the flag passed);
- the manifest reads back `root_full_width`;
- `buffer_full_flushes` = 0 and `eval_max_batch_realised` = 4,096 on every process.

**The rule** (`stage0_decide.py` v3 `4771354e2117…`, applied mechanically; v1 `fb099bed9a35…` and v2 `4c02fc7565dd…` are kept;
v3 changes only the docstring's description of the cost quantity, which `run_stage0.sh` computes, unchanged since v3 `d0e3bcb6952b…`). Cost per stored position is
t = 1 / steady-state rows/s.

**ABBA integrity** (`stage0_decide.py` v5, sha in the table below; v4 `f61952c2b32a…`; pre-registered here).
- **The rule.** No decision is written, and a fresh ABBA block is required (`clear_step.sh stage0-block`), if either
  holds:
  - the largest gap between consecutive ABBA legs exceeds **900 s**;
  - any ABBA leg was cleared and rerun inside the block, i.e. a `CLEAR-DONE` line in `resume.log` timestamped
    inside it.
- **The mechanics.** Every summary row now carries `t0_unix` / `t1_unix` (`run_stage0.sh` v7). `decision.json`
  records the gaps, the largest gap, and the rerun list.
- **The derivation.** ABBA cancels a linear drift, and it cancels a step change between legs 2 and 3. A step change
  δ after leg 1 or after leg 3 leaves a bias of δ/2 on the on/off ratio. Keeping that bias within half the ±3%
  precision needs δ < 3%.
- **The assumption.** This bank has no series of collect rate versus time over hours: the collect-throughput
  benches are single runs, and the repeat gate measures back-to-back runs, within 2%. So the threshold rests on a
  stated assumption: the card's state (thermals, host load, driver) is stable within one leg's length and not
  beyond. A leg is ≈ 13–15 min (384 games per process), and legs normally follow each other within seconds. 900 s
  is one leg.
- **900 s is a conservative assumption, not a measurement, and it will be re-derived.** The first block banks its
  own series: each run's `logs/games_progress.txt` (1 s samples, games finished per process) and `logs/gpu.csv`
  (500 ms samples of utilisation and memory, no timestamps), stitched onto absolute time by the row's `t0_unix` /
  `t1_unix`. `stage0_decide.py` v5 writes that stitched rate-versus-time series beside the decision
  (`decision.series.json`, one entry per run; a run whose logs are missing gets a `missing` field rather than a
  silent gap). The threshold for any later block is re-derived from that series: the largest gap over which the
  rate's step change stays under 3%. It is not changed for the first block itself.
- **Tests.** `stage0/fab_test/run_fab.py` v2, output `fab_test.txt`, runs 13 cases with 0 WRONG:
  - the original 8, with timestamps added (the pre-v4 fixtures are kept as `*.pre-v4.jsonl`);
  - a 2 h gap → FRESH_ABBA_REQUIRED;
  - a leg cleared inside the block → FRESH_ABBA_REQUIRED;
  - a block clear before the block → not flagged;
  - rows without timestamps → refused;
  - (v2) a final decision banks the series beside `decision.json`, one entry per run.
1. **Step A.** Run off@64, on@64, on@64, off@64, in that ABBA order to cancel drift.
2. **Repeat gate.** Two runs of one configuration that differ by > 2% get exactly one more run, and the
   configuration's t is then the mean of three.
   - Two runs within 2% put each mean within about ±1%, so the ratio is known to about ±3%. The bench's own repeat
     agreed within 0.55%.
3. **Step B.** Compute S1 = round((64 + p) · t_off / t_on64 − p), where p is the mean prelude at on@64, clamped to
   [16, 128]. That is where knob-on would cost t_off if a position's cost were proportional to its evaluations.
   - Run on@S1 once. If |t_onS1 / t_off − 1| ≤ 0.015 after that one run, then **S_eq = S1**.
   - Otherwise run on@S1 a second time. If the mean of the two runs is within ±0.03, then **S_eq = S1**.
4. **Step C**, only if step B misses. One step: S2 is the point where the straight line through (64, t_on64) and
   (S1, t_onS1) reaches t_off. Run on@S2 twice. **S_eq = S2**, final, with its measured ratio reported.
5. If S1 = 64, then S_eq = 64.

**Why ±3% is precise enough.** The SL base gains ≈ 136 Elo per doubling of search, measured above 64 sims: 256 vs 64 is +272 over two
doublings.
- A 3% cost error is therefore worth ≈ 136 × log2(1.03) ≈ 5.8 Elo of search.
- A 6% error is worth ≈ 11 Elo, about the size of Stage 2's −10 kill margin.
- The measured ratio at `S_eq` (`ratio_at_S_eq`) is reported next to Stage 2 (b).

**Tests.** `stage0/fab_test/fab_test.txt` (8/8 pass) covers each branch of the rule, including step B's one-run early
stop, and the flush gate. `driver_test.txt` runs the full driver against a fake `run_stage0.sh` and stops after one
step-B run. The v1 outputs are kept as `.v1.txt`.

**Report-only.** Mean eval batch, GPU utilisation and evaluations per position, knob on vs off. These answer the
owner's batching claim directly.

**Cost.** One run is ≈ 15–18 min: the bench's 3 × 512 took 18 min, and the knob adds round 0. Five to eight runs
make **≈ 1.3–2.4 GPU-h**.
- **This deviates from the coordinator's ≈ 15 min.** One run per arm cannot show that the ratio is known to ±3%;
  the repeat gate needs at least two runs per configuration.

## Stage 1 — audit first (CLAUDE.md rule 5; GPU ≈ 0.15 h, then CPU)

**Rows.** The 2,433 leak-free rl-audit rows from 190 games, scored. The search runs on all 3,080.

**Arms.** All are searched with the SL base at main `fd8c0ce08` by `stage1_search.py` (`b928707dee79…`), which is
adapted from the rl-audit `search_bank.py` that produced R64n. These are kept verbatim from it:
- positions;
- the rl-audit protocol's search/game/serving config;
- root logits pre-computed from the stored x;
- the strict rule50 TB probe;
- 16 boards per call on a cold tree, noise on, seed 1.

The arms:

| Arm | Sims | Knob | Role |
|---|---|---|---|
| `M64off` | 64 | off | control (same code and seed as the knob arms) |
| `M64on` | 64 | on | report-only |
| `MSEQon` | `S_eq` | on | **the registered arm** (the Stage 3 arm) |
| `R64n_banked` | 64 | off | the banked rl-audit target; the reproduction gate |

Main's mapping reads two fields that the rl-audit protocol predates: c_visit 50 and q rescale on. These are the
values the GumbelConfig defaults gave R64n, and the script exits unless they took. The knob goes through the
production `SearchConfig` field. The script logs each arm's `GumbelConfig`, which is the consumer's own object. It
exits if knob-on records no prelude visits, or if knob-off records any.

**CPU smoke** (32 positions, `stage1/smoke_cpu/smoke32_check.txt`):
- `M64off` agrees with banked R64n on top-1 for 32 of 32 positions; L1 0.026, from fp32 vs bf16.
- Knob on visits 30.1 children per row vs 14.8.

**Scoring.** `stage1_score.py` (`b1e62b3ab729…`), on CPU. Every row-level measure is banked per row with its row
index and game id (rule 4). Paired differences use a game-clustered bootstrap (2,000 resamples, seed 20261121).
- **Deep SF 8M.** `plan-review-1465/sf8m_table.npz`, with `policy-target-zeros/code/part23_audit.py`'s definitions
  exec'd verbatim:
  - **expected cp loss, lower bound (`exp_cp_lb`): the PRIMARY measure**;
  - SF-best probability;
  - top-1 SF-best;
  - cross-entropy to SF-best;
  - rows giving the SF-best set < 1e-4;
  - mass on unscored moves;
  - the share of SF near-equal alternatives (≤ 10 cp, excluding the target's top-1) pushed below 1e-4.
- **Independent net ruler.** "Ceres C3 (Mish-rewritten net, 98.5% best-move agreement with the original)" @ 256,
  from `ceres-ruler/raw`, with `wide_table_v3.py`'s formulas: top-1, argmax loss, and expected loss over covered
  mass.
- **Root-child VALUE quality.**
  - Mean |q_child − q_SF(child)| over SF-scored children, where q_SF is W − L of the production cp-logistic (slope
    0.006, draw width 120). It is reported over each arm's own visited set and over the set every arm visited.
  - Clear-pair ordering (SF gap ≥ 50 cp, not both mate scores), as in `searched_order_vs_sf.py`:
    - (a) on the pairs `M64off` visited, with every arm scored on the same pairs;
    - (b) on **all** SF-scored legal pairs, each arm on the pairs it visited.
  - A knob-on child with one visit has the raw net's value. So (b) mixes search quality and coverage, and the pair
    counts are reported beside it.

**Scorer validation.** `stage1/dryrun_banked/` scores banked R64n as "off" and R64n_seed2 as "on". It reproduces
the seed-noise numbers below exactly, and the reproduction gate reads top-1 1.000 with paired 0.0. It is a
validation, not a result, and the decision script refuses it.

**The rule** (`stage1_decide.py` `baafe67e25f6…`; `stage1/fab_test/fab_test.txt` 7/7 branches pass):
1. **Code-base control.** `M64off` must reproduce banked R64n: top-1 agreement ≥ 0.95, and a paired
   `exp_cp_lb` CI that contains 0. Otherwise **UNREAD**, stop.
2. **Coverage.** The paired mean of mass on moves deep SF did not score, `MSEQon − M64off`, must be ≤ +0.01.
   Otherwise **UNREAD** until an SF top-up amendment. The reason: `exp_cp_lb` gives unscored moves the 10th-best
   regret as a lower bound, so extra unscored mass would flatter the knob.
   - The banked level is 0.018 for R64n and 0.051 for the prior. The review estimates ≈ 0.0084 for both arms, so the
     gate is not expected to fire.
   - **If it does fire, the top-up procedure is this.** Every move with ≥ 1e-3 mass in any Stage 1 arm that is
     missing from `sf_rows.json` is labelled with `search-compose/scripts/sf_label.py`'s protocol: 8M nodes, cold TT,
     production Syzygy, `searchmoves`. The result goes into a copy of the SF table, and the arms are rescored. It is
     a new amendment, never a silent rerun.
3. **Kill: STOP iff** the paired `MSEQon − M64off` on `exp_cp_lb` has LB95 > 0 **and** its estimate is above
   +1.2 cp.
   - 1.2 cp is the 95% half-width of the paired difference between two 64-sim targets that differ only in the
     search seed: R64n − R64n_seed2 = +0.05 [−1.10, +1.20] (`sizing/results/seed_noise.json`).
4. Otherwise **PROCEED**.

**What the verdicts license.** "Worse" licenses stopping. "Better" licenses only proceeding, never an Elo claim:
there are three fit→Elo transfer failures on record (v020, ft595, F-only) and the ladder's sharp-target inversion.
`M64on` and every secondary measure are report-only.

## Stage 2 — search strength (GPU ≈ 3.0–3.6 h; report-only for the training question, one kill)

**Arena commit.**
- The knob is not at the pilot's PIN, so Stage 2 runs at **main `fd8c0ce08`**, from the detached worktree
  `rl-run1-rfw-arena-fd8c0ce08`, through the follow-up launcher's main-pinned twin `run_rfw_match.sh`
  (v5 `e8e85a601c32…`; the version history is in the GPU guard section's table; v1 `0c192e052082…`).
- The twin keeps the follow-up launcher's flags byte for byte, and its guards:
  - refusal to write into a frozen bank, exit 94 (now the pilot bank **and** the follow-up bank);
  - never resume a look whose game log exists, exit 95. `run_stage2.sh` skips a finished seat and refuses a partial
    one (88) before ever calling the twin;
  - worktree at the pinned commit with no tracked edits, exit 96;
  - `cd` failure, exit 97;
  - a `.exit` stamp;
  - plus the GPU guard.
- Exits 94, 95 and 97 were exercised (`fab_test/launcher_guards.txt`).
- **Main's training shape is the PIN's.** A CPU dry resolution at `fd8c0ce08` prints the same realised gumbel
  fields as match 5's banked row, plus `root_full_width`, and the in-tree yaml has the same sha256 (`8275f2b84d0e`)
  at both commits.

**Design.**
- SL base vs itself, knob on vs off.
- `--search-shape training`, no noise, `--compile off`, the production Syzygy pair for both sides, the pilot's
  `eval.pgn` at 16 opening plies, `--max-plies 600 --on-max-plies abort`.
- Seat-balanced: the knob seat is the candidate in run A (`--cand-gumbel root_full_width=1`) and the reference in
  run B (`--ref-gumbel root_full_width=1`). 3,200 games per run, 6,400 per match.
- The two matches:
  - **(a) `rfw_matched`**, matched sims 64 vs 64, seed **2026112152**.
    - **This match is NOT at equal cost.** On the arena's cold trees, the knob seat gets ≈ 46% more root
      evaluations: 94 vs 64 root visits per position in the CPU smoke.
  - **(b) `rfw_eqtime`**, equal GPU time, `S_eq` vs 64, seed **2026112153**.
    - The equal-cost point is the one measured in the collect, where carried roots are reused. On the arena's cold
      trees the knob seat still evaluates somewhat more, so (b) leans slightly toward the knob.
    - Stage 0's `ratio_at_S_eq` is reported beside this match.
  - Neither seed appears under `docs/` or in any bank.

**Seat proof.**
- `check_shape.py` (`95ee428f58f9…`) reads each run's result row and requires:
  - the knob seat realised `root_full_width=True` and the other seat `False`;
  - every other realised gumbel field, plus `vloss_weight`, `target_batch`, `tree_reuse` and shape, on both seats
    equal to match 5's banked row (which ran at the PIN);
  - the registered sims on each seat;
  - no value hybrid, and no noise.
- Fabricated-row tests pass 6/6 (`fab_test/fab_test.txt`): knob on the wrong seat, a changed c_scale, wrong sims,
  a missing knob and a warm tree all fail.
- **Take effect, observed in the search** (`stage2/run_stage2.sh` v7 `a9d05768c5c9…`; history in the GPU guard section). gumbel_c logs once per process "disabling
  the 2-group eval pipeline for this call because root_full_width is on". It does so on a call of ≥ 64 boards, and
  only the knob seat's config can trigger it.
  - After each real run, the knob run's log is grepped for that line, and `TAKE_EFFECT_{A,B}.txt` records OBSERVED
    or UNOBSERVED.
  - A real run without it makes Stage 2 **"unread: take effect unobserved"**. No kill fires, and Stage 3 refuses.
  - The arena does not expose per-seat forward counts. Banking them would need a code change, so this record uses
    the log line.
  - The 4-game smoke has too few boards per call to show the line, so it is not read there.
- A 4-game smoke pass of both matches, with the pool, runs first. The real pass runs only if every smoke step exits
  0. That includes the pool's seat-swap check on a pair whose two seats are the same file.

**Rules.**
- Kill/void as the follow-up record's "Common" section: the launch record's max-plies rule, any non-zero arena
  exit, and any failed readback.
- Read on `pool.json` → `result.pooled.elo_ci95`, with the arm being the knob seat.
- **Kill: STOP iff match (b)'s UB95 < −10.** At equal GPU time the knob would then make the agent clearly weaker:
  by more than the −10 adoption margin every match here uses, at 95%.
  - Why −10 and not 0: an arena measures the PLAY regime on a cold tree (the
    `gumbel_cscale_miscalibrated` trap), and the training temperature exceeds the strength-optimal one
    (`training_temp_exceeds_strength_optimal_temp`). So a small loss in play does not speak to target quality.
  - Only a clear loss at equal cost means the selfplay games themselves come from a weaker player for the same
    compute.
- Match (a) and every other reading are **report-only**. This search is not tuned toward the play optimum
  (production c_scale 0.1 is chosen for target quality).

**Power.** At the measured match-scale SE (≈ 3.3 per 6,400 games), the 95% interval is ≈ ±6.5 Elo.

**Cost.** Match 5 took 78 min (38.5 + 39.5) at 64 vs 64 with the 2-group pipeline on both seats. In these matches
the knob seat loses the pipeline and evaluates more: (a) ≈ +46% at the root, (b) less.
- Estimate: ≈ 1.6–1.9 GPU-h per match, ≈ 3.0–3.6 GPU-h with the smokes.
- The arena's own timings are reported.

## Stage 3 — the training test (authorised by the owner's standing statement of 2026-11-22; launched only after Stages 0–2 pass and are read)

**One command:** `stage3/launch_stage3.sh` (v7 `2880311d902c…`; v1 `95c8a66dbc59…`; history in the GPU guard section).
- It refuses without `stage3/OWNER_GO` (exit 90, tested), a Stage 1 PROCEED, Stage 2 (b) UB95 ≥ −10, Stage 2's
  take effect observed on both (b) runs, and the operator's `stage3` release line.
- It checks every instrument's sha, sources the GPU guard and holds the bank lock to the end. It re-runs the guard
  before each GPU step.
- It resumes after a refusal, skipping finished and sealed steps. It refuses an interrupted step (88) or a damaged
  one (89) until `clear_step.sh` moves it aside; see the GPU guard section.

**The registered arm: knob on at `S_eq`.** The owner's claim is that the knob comes out ahead at equal GPU cost.
Knob-on at 64 would confound the knob with ≈ 1.3–1.5× the evaluations.
- If `S_eq` ≠ 64, the arm changes the root's width and its sims together. That pairing is the owner's proposal
  itself ("increase node count … and still come out ahead"), registered as ONE data-affecting change; see Confounds.

**Steps**, each stopping the run on failure:
1. **Collect at `fd8c0ce08`**, identical to the pilot's command (same openings, seed 2026112100, 8,000 games, 3 ×
   128, same flags) except `--root-full-width --sims S_eq`. Take effect: the collect's `mean_root_prelude_visits`
   > 0 and 0 buffer flushes.
2. **Train at `fd8c0ce08`.** The pilot's `train_cell.sh` flags verbatim, plus `--w-sibling 0` (main's default, made
   explicit). Three cells:
   - `P0_rfw`: `--w-wdl 0`, on the new collect;
   - `Zq_rfw`: 0.5 outcome / 0.5 stored search WDL, on the new collect;
   - `P0_ctl`: P0's flags on the **pilot's** collect, the code-base control.
   - Between the sibling screen's training commit `febed2042` and `fd8c0ce08`, the trainer path differs only in the
     new optional field `root_prelude_visits` (checked with `git diff`).
3. **Guard (d).** Each main-written checkpoint loads and evaluates identically at the PIN, checked with the sibling
   screen's `check_load.py` (`e8a870764bb5…`).
4. **Stage A instruments by sha**, at the PIN: the value-anchor screen's banked `run_fits.sh` (`5017cea31e6e…`),
   `collect_logits.py` (`956c238f88c7…`), `analyze_holdout.py` (`3291449af64b…`), `supp_outcome_distance.py`
   (`2ad7f272d00a…`) and `sibling_order.py` (`17eeb159f691…`). The bank's `SHA256SUMS` is checked first.
   - The hold-out rows are the pilot collect's hold-out games. Neither knob-on cell trained on them, so for those
     cells this is a check on a different collection.
   - **Veto,** as the anchor screen: a cell whose clear-pair sibling-ordering UB95 (cell − base) is < 0 gets no
     match.
   - **Code-base control** (the sibling screen's guard (c) rule): |P0_ctl − banked P0| ≤ √2 × the banked P0's 95%
     CI half-width on the clear-pair difference, `KL_base_net_raw` and `share_of_gap_closed`. A failure makes every
     Stage 3 match **unread**.
5. **Sibling sidecar re-materialised** on the new collect (a separate CPU command, below; ≈ 30 min, ≈ 63 GB) with
   `scripts/simple_rl_sibling_pairs.py` at defaults (K 2, min visits 2). It is for later screens only; no cell here
   trains on it. See Confounds.
6. **Matches vs the SL base at the PIN**, using the follow-up launcher (`7c8fd065…`) with output root `stage3/`.
   - Plain seats, seat-balanced, 6,400 games each.
   - Seeds **2026112162** (`P0_rfw`) and **2026112163** (`Zq_rfw`).
   - The smoke pass runs first.
   - Match 5's rules: damage iff UB95 < 0; adoptable iff LB95 > −10; "improves on the base" iff LB95 > 0.
   - The seat proof is the pool's row-sha256 cross-check, as for match 5's plain seats.

**Dose, reported.** Stage 3 fixes 8,000 games, as the pilot did, not rows or GPU time. If the knob changes game
length, the arm gets more or fewer rows and GPU-h than the pilot's 761,979 rows and 1.54 h of play.
- Both are reported: rows and rows per game, played GPU-h, and the training updates they yield.
- See Confounds.

**Comparison with the pilot's banked arms.**
- Banked P0 as a single net (match 5): +2.2 [−4.2, +8.6], SE ≈ 3.3.
- Banked Zq (match 1, the gate): −10.0 [−23.2, +3.2], SE 6.74 at 1,600 games.
- Recalibrated Zq (match 8): +9.61 [+3.27, +15.96], which by its rule "improves on the base". This is a nomination
  only, not an adoption ([follow-up results](2026-11-21-rl-run1-followup-results.md)).
- S's value on the base policy (match 6, 6b and 6a): unread by its rules (same results record).

**Power.** The SE of the difference of two independent 6,400-game matches is √2 × 3.3 ≈ 4.7 Elo.
- The 95% half-width of the difference is ≈ 9.2 Elo.
- The difference detectable at 80% power (two-sided 5%) is ≈ 13 Elo.
- Against the 1,600-game Zq gate the difference SE is ≈ 7.5, so detectable at 80% is ≈ 21 Elo.
- **Promise:** "the knob helps training" only if `P0_rfw` − banked P0 has LB95 > 0. A true effect below ≈ 13 Elo is
  likely to read as unread. Nothing smaller is claimed.

## Confounds

- **Code base.**
  - Stages 0–2 and Stage 3's collect and training run at main; every banked comparator ran at the PIN.
  - Bridging Stage 1: `M64off`'s reproduction of R64n.
  - Bridging Stage 2: the realised-shape readback.
  - Bridging Stage 3: `P0_ctl` and guard (d).
- **Width and sims move together** in the registered arm when `S_eq` ≠ 64. This is unavoidable at equal cost. The
  matched-sims report (`M64on`, Stage 2 (a)) separates them descriptively.
- **Sibling sidecar min-visits.** With the knob on, a candidate eliminated after the first halving round has 2
  visits (round 0 + round 1). So `min_visits=2` admits children that the knob-off collect would never pair: pairs
  per row, the children's visit mix and dq noise all shift.
  - A sibling cell trained on this sidecar is not comparable to one trained on the pilot's sidecar without
    re-deriving the strengths.
- **Cold trees in Stages 1–2 vs tree reuse in Stage 3** (reused-root share 0.646). Round 0 evaluates fewer children
  on carried roots, so Stages 1–2 overstate the per-position round-0 work. Stage 0 measures the collect's real
  share.
- **Dose.** Stage 0's unit is cost per stored position. Stage 3's dose is 8,000 games. If the knob changes game
  length or the decisive/draw mix, rows, training updates and GPU-h all differ from the pilot's, and the training
  comparison carries that difference. Both are reported; neither is equalised after the fact.
- **The cost estimator leans slightly toward the knob.** Round-0 cost is proportional to support width. Early in a
  run the 128 slots are in phase, at middlegame plies where the support is widest, and the first wave's games are
  excluded from the estimator's full-concurrency games. So knob-on's measured cost per position reads slightly low.
  The review's adversarial model puts the lean of the on/off ratio at +0.5% to +0.8% toward the knob, which is
  inside the ±3% precision. It is stated here and not corrected.
- **Stage 1 search config** is the rl-audit protocol's (gumbel scale 0.75, serving max batch 512), chosen to match
  banked R64n. Stage 3's is the pilot's.

## Cost and order

| Item | GPU-h |
|---|---|
| Stage 0: 5–8 runs × ≈ 15–18 min | ≈ 1.3–2.4 |
| Stage 1: three searches on 3,080 rows (R64n took 91 s; the knob adds ≈ 40%) + model load | ≈ 0.15 |
| Stage 2: two 6,400-game matches (match 5: 78 min; the knob seat loses the pipeline and evaluates more) + smokes | ≈ 3.0–3.6 |
| Stage 3: collect (equal cost to the pilot's 1.54 h of play, the slowest process's wall, + seal) | ≈ 1.6 |
| Stage 3: training P0 / Zq / P0_ctl (pilot: 11 / 14 / 11 min) | ≈ 0.6 |
| Stage 3: Stage A, 3 nets (fits + logits + sibling order) | ≈ 0.3 |
| Stage 3: two 6,400-game matches + smokes (no knob in these matches) | ≈ 2.7 |
| **Total** | **≈ 4.5–6.2 (Stages 0–2) + ≈ 5.2 (Stage 3)** |

CPU: Stage 1 scoring takes ≈ 10 min, the Stage 3 sidecar ≈ 30 min. Disk: ≈ 63 GB for the sidecar plus ≈ 30 GB for
the collect; ≈ 800 GB is free.

## Option, not registered and not built: a budget-neutral round 0

This is the owner's idea in a form that adds no evaluations.
- **What it is.** Round 0 spends the first halving round's budget on every legal child, instead of adding a
  prelude, so the total stays at `--sims`. With k children and a first-round budget below k, the remaining rounds
  are shortened to fit.
- **Code it would take.** `_full_width_round0` in `mcts/gumbel_c.py` would charge its visits against the halving
  schedule instead of adding to it. The C `start_gumbel_sims` state machine would then need to start from a reduced
  budget with a pre-chosen candidate set, or take a per-board budget offset. Tests would need to pin the total root
  visits at `--sims`.
- **Under a cost-proportional model, it is the same search as the registered arm.**
  - The additive form at `S_eq` spends ≈ `S_eq` + p ≈ 64 evaluations, and halves with ≈ 64 − p.
  - The budget-neutral form at 64 also spends p on round 0, and also halves with ≈ 64 − p.
  - So a low `S_eq` does not by itself argue for building it.
- **When it would be worth building.** Only if a round-0 evaluation costs measurably MORE than a halving one, or the
  per-call overhead differs, so that the additive form loses evaluations to cost the budget-neutral form would not.
  - Stage 0 can show this. Compare the measured cost per forwarded evaluation (forwarded rows/s and mean batch,
    knob on vs off) against the ratio predicted from evaluations per position (sims + prelude).
  - A knob-on cost per evaluation more than ≈ 3% above knob-off's (the ratio's precision) would make this option
    worth registering.
- **Caveat on the Elo-per-doubling figure.** The ≈ 136 Elo per doubling quoted above was measured above 64 sims
  (256 vs 64). Below 64 the rate is not measured here.

## Commands

```bash
B=$HOME/chess-artifacts/operations/rl-run1-root-full-width
AW=$HOME/projects/chess-worktrees/rl-run1-rfw-arena-fd8c0ce08; PY=$HOME/projects/chess/.venv/bin/python
(cd $B && sha256sum -c --quiet SHA256SUMS) || exit 93
cd $AW && [ "$(git rev-parse HEAD)" = fd8c0ce08e1a0829e051cf1a6fef07fd6e83c244 ] || exit 96
$PY scripts/build_production_extensions.py || exit 1          # the worktree's .so files, at this commit
# Every GPU stage first needs the OPERATOR's line in $B/GPU_RELEASED_BY_OPERATOR, e.g. "stage0 2026-11-22T09:00".
# Stage 0 (writes stage0/decision.json)
$B/stage0/run_stage0_all.sh || exit 1
# Stage 1 (GPU search, then CPU score and verdict)
$B/stage1/run_stage1.sh || exit 1
(cd $AW && PYTHONPATH=. $PY $B/stage1/stage1_score.py $B/stage1/stage1_bank.npz $B/stage1/scores) || exit 1
$PY $B/stage1/stage1_decide.py $B/stage1/scores/scores.json $B/stage1/verdict.json || exit 1
# Stage 2 (refuses unless Stage 1 is PROCEED)
$B/stage2/run_stage2.sh || exit 1
# Stage 3: authorised by the owner's standing statement (2026-11-22). After Stages 0-2 pass by their rules and
# are read in-session, the coordinator writes $B/stage3/OWNER_GO and the operator's stage3 release line, then:
$B/stage3/launch_stage3.sh
```

The sidecar step, after Stage 3's collect (CPU):

```bash
(cd $AW && PYTHONPATH=. /usr/bin/time -v $PY scripts/simple_rl_sibling_pairs.py \
  --collect $B/stage3/collect/gen_0001/collect --out $B/stage3/sibling/gen_0001) > $B/stage3/sibling.log 2>&1
```

## Provenance

- **Bank:** `~/chess-artifacts/operations/rl-run1-root-full-width/`. `README.md` lists every file; `SHA256SUMS`
  covers every script, and the short shas above are its first 12 hex digits.
- **Sizing measurements:** `sizing/results/`, code head `fd8c0ce08` in `code_head.txt`. They were added as two
  rows to Table 2 of the evidence log.
- **Incident, 2026-11-21 21:38:44–21:40:56 EST.**
  - What happened: a guard test of v1 of `run_stage0.sh` passed its GPU-idle check. v1 trusted
    `nvidia-smi --query-compute-apps`, which never listed match 8's PID 2337486, although it did list this bank's
    own collects in 62 of 64 samples. Three collect processes (SL base, 64 sims, 128 in flight each) then ran for
    ≈ 126 s on the GPU beside the registered match 8 (`zq_recal_vs_base_1_A`).
  - Peak memory was 30,406 MiB, at about 21:40:49.
  - They were killed by exact PID. The match survived: no OOM, no CUDA error. It is matched-sims, so its games are
    not clock-dependent. The overlap is recorded in [match 8's results](2026-11-21-rl-run1-followup-results.md#match-8-recalibrated-zq-vs-the-sl-base-seed-2026112191).
  - Their partial outputs, all under this bank, were deleted. `stage0/INCIDENT_guardtest_20261121/` keeps only the
    logs and `INCIDENT.txt`; it is not a measurement.
  - The guard was rewritten as above. From now on, launchers are tested only with the GPU command stubbed and a
    fake `nvidia-smi`.
- **Versions.** Scripts changed after the first review are kept beside their v2 as `.v1`, and `SHA256SUMS.v1` is the
  first sum file.
