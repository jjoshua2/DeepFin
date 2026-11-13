# Installing and verifying the engine

This page is for running DeepFin as a UCI engine (for example under a tournament
manager). Training and selfplay installs are described in the README and
[development](development.md).

## Supported platform

- Linux x86-64. That is the only platform the engine is built and tested on.
- CPython **3.10.12 – 3.13**, standard (GIL-enabled) build. 3.13 is what CI and
  the development lock use; 3.10 is what the production host runs. Newer
  interpreters are untested and `deepfin --selfcheck` reports them as FAIL.
- A C compiler (`gcc`) and the Python headers, to build the five C extensions.
- For GPU play: an NVIDIA driver plus a PyTorch build whose CUDA version supports
  the card. An RTX 50-series (Blackwell, `sm_120`) card needs a CUDA 12.8 or newer
  build; the `cu124` builds stop at torch 2.6.0 and contain no `sm_120` kernels.

## Install roles

| Command | Installs | For |
|---|---|---|
| `pip install .` or `pip install ".[match]"` | torch, numpy, python-chess | the UCI engine only |
| `pip install ".[client]"` (alias `.[worker]`) | + pyyaml, zarr, requests | a selfplay worker |
| `pip install ".[server]"` | + trainer, Ray Tune, HTTP server | the training server |
| `pip install ".[dev]"` | everything, plus test and lint tools | development and CI |

The engine imports nothing outside the `match` set; a test enforces it.

## Match-only install

Install torch first from the PyTorch wheel index for your CUDA version, so pip does
not pull a default build. The project's CUDA profile pins torch 2.14.0 from the
`cu130` index (`pyproject.toml`, `[tool.uv.index]`):

```bash
cd deepfin                                   # the source archive / checkout
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install ".[match]"                       # builds the C extensions with the stock gcc
```

For a CPU-only host use `--index-url https://download.pytorch.org/whl/cpu`.
Use `pip install -e ".[match]"` instead if you want to edit or rebuild in place;
that also lets the self-check compare each built extension against its sources.

The production host builds the extensions with
`scripts/build_production_extensions.py` (gcc 15.3, `-march=native`, LTO). An
operator does not need it: `pip install` builds a portable variant with the stock
compiler. The self-check's encoder and move-generation comparisons apply to either
build.

## Verify

```bash
deepfin --selfcheck --checkpoint /path/to/trainer.pt --syzygy /path/to/syzygy
deepfin --selfcheck --json ...                    # same results as JSON
deepfin --selfcheck --devices cuda:0,cuda:1 ...   # multi-GPU host: name every GPU
```

`python -m chess_anti_engine.uci --selfcheck` is equivalent. The run takes a few
seconds on a CPU. Each component prints `PASS`, `FAIL` or `SKIP` with a reason,
then `SELFCHECK PASS` or `SELFCHECK FAIL`; the exit status is 0 only when nothing
failed. Multi-GPU hosts pass `--devices` to both the self-check and the engine.

| Component | What a FAIL means |
|---|---|
| `python` | Interpreter outside 3.10.12–3.13, or a free-threaded build. Use a supported CPython. |
| `torch` | torch is missing, a requested CUDA device is not visible, or this torch build has no kernels for the GPU (the message names the GPU's `sm_XY`). Reinstall torch from a newer `cuXXX` index. |
| `extensions` | A required C extension (`_lc0_ext`, `_features_ext`, `_mcts_tree`) is missing, lacks a symbol, or has an older ABI than the Python code needs. Rebuild: `pip install .` again (or `-e .`). |
| `freshness` | A built extension is older than its C sources (source checkouts only). Rebuild. SKIP on a regular `pip install .`, which has no sources beside the package. |
| `encoder` | The C board encoder produced all-zero feature planes or differs from the Python reference encoder. Rebuild; if it persists, report it. |
| `movegen` | C move generation disagrees with python-chess or perft. Rebuild; report if it persists. |
| `search` | A short C Gumbel search on a random tiny net did not complete in the C tree. |
| `checkpoint` | The network failed to load, has missing or mis-shaped tensors, or a forward pass gave NaN/inf. SKIP without `--checkpoint`. |
| `syzygy` | A listed directory does not exist, holds no `.rtbw` tables, or a known 3-man position probed to the wrong result. SKIP without `--syzygy`. |
| `uci` | The engine, run as a subprocess, did not answer `uciok`/`readyok` or returned an illegal `bestmove` for `go nodes 64`. It uses `--checkpoint` when given, otherwise a random-weight tiny net, and `--no-compile`. |

The self-check does not run `torch.compile`, which the engine uses by default, and
its search check runs on the CPU.

## Startup refusal

The engine checks the same extension facts on every start and exits with status 2,
before printing `uciok`, if a required extension is missing, ABI-incompatible,
stale or produces all-zero feature planes. The message names the problem and the
fix. For development only, `DEEPFIN_ALLOW_UNVERIFIED_EXTENSIONS=1` downgrades the
refusal to a warning.
