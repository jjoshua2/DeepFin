# DeepFin — TCEC setup & operation guide

DeepFin is a **GPU neural-network MCTS engine** (transformer policy+value net,
Gumbel MCTS) speaking standard **UCI**. It runs as a Python process, not a
compiled binary — think Lc0-style deployment. This guide is written for the
TCEC ops workflow (cutechess-fork GUI, UCI registration) and includes a
~5–10 minute autotune script to find the best nps configuration for your
hardware, including a multi-GPU test.

## 1. Requirements

- Linux x86-64, NVIDIA GPU (Ampere or newer recommended), recent NVIDIA
  driver (CUDA 12.x capable).
- Python **3.10+**, `gcc` (two small C extensions build during install).
- ~12 GB disk: PyTorch (~5 GB), the network file (~700 MB), torch compile
  cache (~2–4 GB grows on first runs).
- One GPU is the quality-validated configuration (classic Gumbel). 2+ GPUs
  use multi-GPU PUCV search (throughput path; quality vs Gumbel is measured
  separately). Multi-GPU automatically disables CUDA-graph compile modes
  (see section 4 / multi-GPU notes) — inductor still fuses without graphs.

## 2. Install

```bash
# from the provided source archive / repository checkout:
cd deepfin
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # any CUDA 12.x build
pip install -e .
python3 setup.py build_ext --inplace     # builds the two C extensions in place

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
| `Hash` | `16384` | MCTS tree memory (MB). The tree is the memory consumer; small Hash degrades long searches. |
| `SyzygyPath` | your TB path | 3–7 man WDL/DTZ supported. |
| `MoveOverheadMs` | your standard | Added per-move safety on top of the internal margin. |
| `ClockBatchMarginSigmas` | `2` (default) | Self-calibrating clock margin in units of observed GPU-batch-time sigma. Leave at default. |
| `MultiPV` | supported | For broadcast eval display. |
| `Ponder` | `false` | Not supported meaningfully; leave off. |
| `Threads` | leave default | GPU engine — CPU threads are not the scaling axis; parallelism is configured via `--walkers`/`--devices` on the command line. |
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
