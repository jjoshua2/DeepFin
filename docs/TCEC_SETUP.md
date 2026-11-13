# DeepFin — TCEC setup & operation guide

DeepFin is a **GPU neural-network MCTS engine** (transformer policy+value net,
Gumbel MCTS) speaking standard **UCI**. It runs as a Python process, not a
compiled binary — think Lc0-style deployment. This guide is written for the
TCEC ops workflow (cutechess-fork GUI, UCI registration) and includes a
~5–10 minute autotune script to find the best nps configuration for your
hardware, including a multi-GPU test.

## 1. Requirements

- Linux x86-64, NVIDIA GPU (Ampere or newer recommended), an NVIDIA driver
  supporting CUDA 13 (the install below uses the `cu130` torch wheel; minimum
  driver version to be confirmed on the target machine with
  `deepfin --selfcheck --device cuda:0`).
- Python **3.10.12–3.13**, `gcc` (five small C extensions build during install).
- ~12 GB disk: PyTorch (~5 GB), the network file (~700 MB), torch compile
  cache (~2–4 GB grows on first runs).
- One GPU is the quality-validated configuration (classic Gumbel). 2+ GPUs
  use multi-GPU PUCV search (throughput path; quality vs Gumbel is measured
  separately). Multi-GPU automatically disables CUDA-graph compile modes
  (see section 4 / multi-GPU notes) — inductor still fuses without graphs.

## 2. Install

For the install roles, a torch build that supports current NVIDIA cards, and
`deepfin --selfcheck`, see [install and verify](install.md).

```bash
# from a repository checkout (or a `git archive` of one; sdists are not supported):
cd deepfin
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # see install.md (cu124 has no RTX 50-series build)
pip install -e .
python3 setup.py build_ext --inplace     # builds the five C extensions in place

# sanity check (should print a UCI id block and exit):
printf 'uci\nquit\n' | PYTHONPATH=. python3 -m chess_anti_engine.uci \
  --checkpoint /path/to/trainer.pt --device cuda:0
```

The network is a single file, `trainer.pt` (~700 MB), supplied alongside the
source. Any path works; it is passed on the command line.

## 3. Engine registration (cutechess fork)

Register a UCI engine with:

| Field | Value |
|---|---|
| Command | `python3 -m chess_anti_engine.uci --checkpoint /path/to/trainer.pt --device cuda:0 --walkers 1 --chunk-sims 512 --max-batch 1024 --compile-mode max-autotune` |
| Working directory | the repo root (imports resolve from there; set `PYTHONPATH=.` in the environment if your fork supports env vars) |
| Protocol | UCI |

If the autotune (section 6) finds a better batching/multi-GPU configuration,
we will send an updated command line — only the flags change, never the
workflow.

## 4. IMPORTANT: one-time compile warm-up (before any clocked game)

The first process after install triggers a one-time `torch.compile`
(budget **≤5–10 minutes** cold under `max-autotune` / multi-GPU
`max-autotune-no-cudagraphs`). **Caching is automatic and persistent — no
Triton/Inductor cache paths to configure**: the engine sets up a shared
TorchInductor + Triton cache at `~/.cache/deepfin/worker_cache` by itself
(override with the `DEEPFIN_COMPILE_CACHE` env var or `--compile-cache-dir`
if home is small — it grows to a few GB). Because it lives under `$HOME`, it
survives reboots.

**Tournament timing targets after the disk cache is warm:**

| Phase | Budget | What happens |
|---|---|---|
| Offline prewarm (once) | ≤ 5–10 min | Cold compile + search-path warmup |
| Per-game process `isready` | ≤ ~30 s single-GPU; ≤ ~90 s dual-GPU on mid-range cards | Load weights + re-capture process-local CUDA graphs (graphs are not on disk) |
| Mid-game moves | no compile stalls | Search only |

Two rules:

1. **After install, run the prewarm once** (populates the disk cache):
   ```bash
   # single GPU
   PYTHONPATH=. python3 scripts/uci_prewarm.py --checkpoint /path/to/trainer.pt
   # multi-GPU (recommended before any multi-device tournament registration)
   PYTHONPATH=. python3 scripts/uci_prewarm.py --checkpoint /path/to/trainer.pt \
       --devices cuda:0,cuda:1 --multi-gpu-pucv --vl-gather 256 --repeat 2
   ```
   Equivalent manual one-liner (single GPU):
   ```bash
   printf 'uci\nisready\nposition startpos\ngo nodes 3000\nquit\n' | \
     PYTHONPATH=. python3 -m chess_anti_engine.uci --checkpoint /path/to/trainer.pt --device cuda:0
   ```
2. **Give the engine a generous start/`isready` allowance** in the GUI
   (≥60 s recommended even warm; cold first launch can be minutes), or
   play one unrated warm-up game per session. The engine's own clock
   management then keeps it safe (it self-calibrates a per-move safety margin
   from observed GPU batch times — `ClockBatchMarginSigmas`, default 2 — and
   has never flagged in our TC testing at 30′+10″ and 60″+1″ after warm-up).

### Multi-GPU compile note (cudagraphs enabled)

Multi-GPU workers are ordinary Python threads, but PyTorch only auto-stashes
Inductor cudagraph tree TLS onto *autograd*-spawned threads. The engine
bootstraps that TLS on each pool worker and **serializes first-graph
capture** across devices, then runs concurrent replay for the NPS win.
`reduce-overhead` / `max-autotune` therefore work on multi-GPU.

If a device's cudagraph warmup fails, that worker retries once with
`max-autotune-no-cudagraphs` so a single bad capture cannot kill the
process. Force a mode with `CAE_UCI_MULTI_GPU_COMPILE_MODE`
(`default`, `max-autotune-no-cudagraphs`, or empty for eager).

## 5. UCI options

| Option | Recommended | Notes |
|---|---|---|
| `Hash` | most of the machine's RAM | MCTS tree memory (MB), measured against the tree's resident footprint (~3.3 KB per visit, so 16384 ≈ 5M visits ≈ 5 min at ~17k nps). The tree is the memory consumer; when it fills, the search stops early with `info string tree memory full …`. Provision about **Hash + 150 MB + the eval cache** (`EvalCacheEntries`) of RAM: per-search buffers and allocator slack are outside Hash. A cross-move compaction briefly holds new child arrays beside the old ones; it is only chosen while that transient keeps the tree within Hash + 32 MB, so the provisioning above still holds. One tree addresses at most ~2.1 billion nodes (int32 ids, about 130–280 GB of footprint); Hash above that is unused and the stop says `tree node-id space full`. |
| `SyzygyPath` | your TB path | 3–7 man WDL/DTZ supported. |
| `MoveOverheadMs` | your standard | Added per-move safety on top of the internal margin. |
| `ClockBatchMarginSigmas` | `2` (default) | Self-calibrating clock margin in units of observed GPU-batch-time sigma. Leave at default. |
| `MultiPV` | supported | For broadcast eval display. |
| `Ponder` | `false` | Not supported meaningfully; leave off. |
| `Threads` | any (ignored) | Accepted because GUIs send it, and answered with an `info string` saying it has **no effect**: DeepFin is a GPU engine and `Threads` never changes the search. The search path is chosen only on the command line (`--walkers`, default 1 = classic Gumbel; `--devices`) or by the operator-only `SearchWalkers` option. |
| `MaxBatch` / `MinibatchSize` | leave default | Overridden by the autotuned command line if needed. |

## 6. The 5–10 minute autotune (the "light testing" ask)

One command, prints per-configuration nps (sims/s) and a summary; please run
it on an idle GPU and send us the produced `tcec_autotune_*.log`:

```bash
# single GPU:
bash scripts/tcec_autotune.sh --checkpoint /path/to/trainer.pt
# multi-GPU test (any 2+ device list):
bash scripts/tcec_autotune.sh --checkpoint /path/to/trainer.pt --devices cuda:0,cuda:1
```

Phases: (A) a batching grid over (chunk-sims × candidates × GPU batch), (B) a
search-parallelism sweep (`--walkers 1/2/4/8`), (C) if 2+ devices are given,
multi-GPU PUCV, swept at 2, 4, and all N GPUs so we see the scaling curve.
The first configuration includes the one-time compile, which is inside the
time budget. Prefer `scripts/uci_prewarm.py` first on multi-GPU hosts so
phase C measures steady-state nps rather than cold compile. From the log we
will reply with the exact command line to register — typically within the
same day.

Example multi-GPU registration shape (after prewarm + autotune; numbers from
your log):

```text
python3 -m chess_anti_engine.uci --checkpoint /path/to/trainer.pt \
  --devices cuda:0,cuda:1 --multi-gpu-pucv --vl-gather 256 \
  --chunk-sims 512 --max-batch 1024 --compile-mode max-autotune
```

(Multi-GPU uses cudagraph TLS bootstrap; see section 4.)

A note on what we will and won't register from the results: the section-3
command line (`--walkers 1` = classic **Gumbel** search) is our
quality-validated configuration — all of the engine's search tuning was done
on it, and raw nps is not playing strength. Phases B/C measure whether the
alternative parallel-search modes buy throughput on your hardware; if they
do, we validate their play quality on our side before recommending any
change. You will never be asked to register a config we haven't
quality-checked.

Reference point: our development machine is a single RTX 5090 — the same
silicon as your 8×5090 box — sustaining ~17.6k sims/s per GPU with the
section-3 command line, so per-GPU numbers should transfer directly and
phase C tells us what the additional seven GPUs are worth.

## 7. Known behaviors / troubleshooting

- **Startup failure exits the process.** If model load, compile or the
  search warmup fails, the engine prints `info string engine startup failed: ...`
  and `deepfin: engine startup failed: ...` on stderr and exits non-zero
  *before* `readyok`, so the GUI reports a failed engine start instead of
  playing a game on an unsearched fallback.
- **A failed search never loses on time.** The move comes from, in order: the
  network's cached root policy for that exact position (`source=root_policy`);
  one fresh root evaluation of the position, bounded by min(150 ms, a tenth of
  the remaining clock) (`source=fresh_root_eval`); or, when that is skipped
  (another thread may still be inside the evaluator) or times out, the first
  legal move (`source=first_legal`). Nothing is rebuilt during a game. Each kind
  of unsearched move has its own line; any of them is worth a report:
  - a search that raised: `info string search error: ...`, then
    `info string bestmove_fallback_used=N source=... move=... exception=...
    phase=main` and `info string WARNING search failed on N consecutive
    move(s); this move is UNSEARCHED (fallback)` (a failed ponder search says
    `WARNING ponder search failed; ...` and counts only if `stop` publishes it);
  - a `go` that could not start a search, or that arrived while a stuck search
    still holds the worker: `info string error handling CmdGo: ...` or
    `info string command rejected (CmdGo): ...`, then `info string search not
    started; bestmove_fallback_used=N source=... move=...`;
  - a `stop` that the search ignored for 1 s: `info string search did not stop
    in time; answering unsearched bestmove_fallback_used=N source=... move=...`;
  - a ponder transition past its deadline or cancelled: `info string search
    incomplete reason=ponder_transition_deadline|ponder_transition_cancelled
    move=... source=...`;
  - a root proved solved whose proof names no move: `info string search
    incomplete reason=solved_root_without_move_witness ...
    witness_unavailable_roots=N`. This is a property of the position, not a
    fault, and does NOT count toward the unhealthy streak below.
- **Unhealthy state.** 3 counted fallback moves in a row (every kind above except
  the solved-root one), or a failed search-path re-warm at `isready`, print
  `info string WARNING engine unhealthy: ...` once. The engine keeps answering
  every `go`. A game boundary is `ucinewgame`, or, for a GUI that never sends
  it, `position startpos` with no moves; it only marks the boundary. The checks
  run at the next `isready` after it (never inside `ucinewgame` / `position`,
  because a `go` may follow at once): it announces `engine unhealthy (...):
  rebuilding the evaluator and rerunning the strict startup check`, rebuilds
  the evaluator synchronously, and reruns the strict startup warmup. Outcomes:
  `engine health check passed; unhealthy state cleared`; or `WARNING engine
  health check passed on a DEGRADED path (realized path=... but the options
  named ... before the rebuild)` (see "One bad GPU"); or `engine health check
  failed ...` plus `deepfin: engine health check failed` on stderr and an exit
  with status 1 *before* `readyok`. Whether a given GUI then restarts a fresh
  process for the next game is unverified (see the PR's GPU checks).
  **A GUI that never sends `isready` between games gets no health re-check at
  all**: the engine stays unhealthy, keeps answering every `go`, and every
  unsearched move stays loud as listed above.
- **`isready` mid-game** never rebuilds, re-checks or exits. "Mid-game" starts
  at the first `go` after a boundary, however that `go` was answered (a search,
  or an unsearched reply while a stuck search still held the worker). It
  answers at once when a search (or a stuck one) holds the worker. Otherwise,
  if an option rebuild left the search path cold, it first re-warms it with one
  search on the start position, no deadline, of:
  - classic Gumbel, walker or single-device paths: max(256, `ChunkSims`) nodes;
  - multi-GPU PUCV pool: max(256, `ChunkSims`, max(`ChunkSims`, G) x number
    of devices) nodes, so every device captures its own graph;
  - root-parallel Gumbel (`SearchParallel gumbel`): max(256, `ChunkSims`,
    max(`ChunkSims`, 1 + 2 x G) x number of groups) nodes;

  where G = min(`VLGather`, `MaxBatch`) is the pools' gather and `ChunkSims` is
  the UCI spin option of that name (default 2048; the startup flag
  `--chunk-sims` sets its initial value).

  That takes milliseconds when the graphs are already captured and a few
  seconds (about 3-4 s measured for a cold Gumbel capture, per device) right
  after a rebuild. No clock runs between `bestmove`
  and the next `go`. A failed re-warm prints `search-path re-warm failed`,
  marks the engine unhealthy, keeps the path cold for the next `isready`, and
  still answers `readyok`.
- **Options mid-game.** A `setoption` that rebuilds an evaluator or a pool
  (e.g. `MaxBatch`, `SearchWalkers`, `UseMultiGpuPUCV`) is applied at once,
  synchronously, between moves; if the rebuild fails the previous settings and
  evaluator stay. While a search that ignored `stop` still holds the worker,
  `MultiPV` and `UCI_ShowWDL` still apply (they only change what the next info
  line reports; the engine says `applied while a stuck search still holds the
  worker`). Every other `setoption` -- including `Hash`, which resizes the tree
  the stuck search is still using -- is rejected at once with `info string
  command rejected (CmdSetOption): ...` and the previous value stays.
- **One bad GPU on a multi-GPU host.** There is no degrade-to-the-healthy-GPUs
  path. If a multi-GPU pool cannot be (re)installed -- after a `setoption`, or
  during the boundary health rebuild -- the engine first restores the previous
  pool (`... failed; previous configuration kept: path=pucv_pool`). If that
  also fails it falls back to classic Gumbel on ONE GPU and says so
  (`previous configuration could not be restored ...; fell back to the
  single-evaluator classic Gumbel path`); after a health rebuild the result
  line then reads `engine health check passed on a DEGRADED path`, never plain
  `passed`. It stays on one GPU until the process is restarted (or the GUI
  sends `UseMultiGpuPUCV` / `SearchParallel` again): a later boundary rebuild
  rebuilds what the options name, which is then the single-GPU path. A pool
  that fails at startup fails the strict startup check and the engine exits
  before `readyok`.
- **A search that ignores `stop`.** After 1 s the engine answers that `go` from
  the fallback; the stuck search is parked and every later `go` answers at
  once (within a tenth of its clock) for the board the GUI sent, until the
  stuck search ends. A parked search prints nothing more: its info lines and
  warnings are dropped, so they cannot appear in the middle of another move.
  If it is still stuck at `quit`, the engine waits up to 3 s, then exits; the
  last lines are `info string quit: a search that ignored stop, or a fallback
  eval stuck in the evaluator, is still running; exiting with status N without
  waiting for it` (and the same on stderr). N is 1 when the stuck thread is a
  search.
- **Long first move of a fresh process** (~15–30 s even warm-cached for
  `isready` / first search): expected; see section 4. Never happens mid-game
  after a successful `isready`.
- **Multi-GPU first `go` returns nodes=0 / search error**: usually means
  prewarm failed or an old build without the no-cudagraph multi-GPU path.
  Re-run `scripts/uci_prewarm.py` and check stderr for `engine load failed`
  / `CUDA graphs` info strings.
- **GPU contention**: the engine assumes exclusive GPU use during play;
  sharing the device with another GPU process degrades nps ~15–20× and risks
  time trouble.
- The engine resigns/claims nothing on its own; adjudication is the GUI's.
- Logs: pass `--log-level DEBUG` on the command line (or UCI `LogFile`) if
  you need diagnostics for a report; default output is UCI-clean.
- Anything unclear or broken: send us the console output and the autotune
  log; turnaround is usually same-day.
