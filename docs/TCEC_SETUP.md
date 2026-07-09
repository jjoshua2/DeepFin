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
- One GPU is the validated configuration; 2+ GPUs are optionally used by an
  experimental parallel-search mode the autotune will evaluate (section 6).

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

The first search after install triggers a one-time `torch.compile`
(~2 minutes). It is cached on disk afterwards, but **each engine process**
still pays ~15 s on its very first search. Two rules:

1. **After install, run the warm-up once** (populates the disk cache):
   ```bash
   printf 'uci\nisready\nposition startpos\ngo nodes 3000\nquit\n' | \
     PYTHONPATH=. python3 -m chess_anti_engine.uci --checkpoint /path/to/trainer.pt --device cuda:0
   ```
2. **Give the engine a generous start/`isready` allowance** in the GUI, or
   play one unrated warm-up game per session. The engine's own clock
   management then keeps it safe (it self-calibrates a per-move safety margin
   from observed GPU batch times — `ClockBatchMarginSigmas`, default 2 — and
   has never flagged in our TC testing at 30′+10″ and 60″+1″ after warm-up).

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
the experimental multi-GPU parallel-search mode. The first configuration
includes the one-time compile, which is inside the time budget. From the log
we will reply with the exact command line to register — typically within the
same day.

Reference point: on a single RTX 5090 the engine sustains ~17.6k sims/s with
the section-3 command line; a batching-saturated data-center GPU should land
in a similar range or above, and phase C tells us whether a second GPU adds
anything on your hardware.

## 7. Known behaviors / troubleshooting

- **Long first move of a fresh process** (~15 s even warm-cached): expected;
  see section 4. Never happens mid-game.
- **GPU contention**: the engine assumes exclusive GPU use during play;
  sharing the device with another GPU process degrades nps ~15–20× and risks
  time trouble.
- The engine resigns/claims nothing on its own; adjudication is the GUI's.
- Logs: pass `--log-level DEBUG` on the command line (or UCI `LogFile`) if
  you need diagnostics for a report; default output is UCI-clean.
- Anything unclear or broken: send us the console output and the autotune
  log; turnaround is usually same-day.
