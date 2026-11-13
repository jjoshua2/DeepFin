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
  the card. An RTX 50-series (Blackwell, `sm_120`) card needs a build made with
  CUDA 12.8 or newer; the `cu124` wheel index stops at torch 2.6.0. The self-check's
  `torch` line reads the installed build's architecture list and names the
  mismatch, whatever the card.

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
deepfin --selfcheck ... --walkers 2               # any other flag goes to the engine
```

`python -m chess_anti_engine.uci --selfcheck` behaves the same, including on an
install whose C extensions are missing. Each component prints `PASS`, `FAIL` or
`SKIP` with a reason; the last line is `SELFCHECK PASS` or `SELFCHECK FAIL`
followed by which engine configuration the UCI exchange used and which
components were skipped, e.g.
`SELFCHECK PASS (uci: default play path, torch.compile on; 2 skipped: checkpoint, syzygy)`.
A SKIP is not a pass for that component. The exit status is 0 only when nothing
failed.

**The UCI exchange runs the engine exactly as you will:** the same entry point,
your `--checkpoint`, `--device`/`--devices`, and every other engine flag you add
to the command line, with `torch.compile` on unless you pass `--no-compile`. Give
the self-check the flags you will give the GUI. Multi-GPU hosts pass `--devices`
to both. Because compilation is part of the default play path, a cold compile runs
inside the check. It takes about 15 s for the CPU tiny-net case; with a real network
on a GPU, a cold `max-autotune` compile can take several minutes. Replies are
awaited for up to 900 s (`--uci-timeout` changes this). `--fast` adds
`--no-compile` for a check in seconds; the summary then says the default compiled
path was **not** covered. The engine subprocess uses a private temporary compile
cache unless you pass `--compile-cache-dir`, so the check never writes into a
shared cache. Passing your real cache directory makes the check warm and fast, at
the cost of writing to it.

| Component | What a FAIL means |
|---|---|
| `python` | Interpreter outside 3.10.12–3.13, or a free-threaded build. Use a supported CPython. |
| `torch` | torch is missing, a requested CUDA device is not visible, or this torch build has no kernels for the GPU (the message names the GPU's `sm_XY`). Reinstall torch from a newer `cuXXX` index. |
| `extensions` | A required C extension (`_lc0_ext`, `_features_ext`, `_mcts_tree`) is missing, lacks a symbol, or has an older ABI than the Python code needs. Rebuild: `pip install .` again (or `-e .`). |
| `freshness` | (Source checkouts only; SKIP on a regular `pip install .`.) An extension's sources changed since it was built. The build writes a hash of every source beside each extension, and the check compares contents, so a timestamp-only change (branch switch and back, `cp` without `-p`) does not fail it. An extension built before these records existed falls back to comparing timestamps; the message then says a checkout or copy can trigger it. Either way `pip install -e .` rebuilds and clears it. |
| `encoder` | The C board encoder or the search tree's leaf encoder produced all-zero feature planes, or differs from the Python reference encoder. Rebuild; if it persists, report it. |
| `movegen` | C move generation disagrees with python-chess or perft. Rebuild; report if it persists. |
| `search` | A short C Gumbel search on a random tiny net did not complete in the C tree. |
| `checkpoint` | The engine's own loader refused the network, left a model tensor at random initialisation, or a forward pass gave NaN/inf. Migrations the loader applies (a `torch.compile` key prefix, fused q/k/v, zero-padded v2 input columns) are listed as notes on a PASS. SKIP without `--checkpoint`. |
| `syzygy` | A listed directory does not exist, holds no `.rtbw` tables, or a known position probed to the wrong result. 3-man positions are always probed; a 5-man and a 6-man position are probed when those tables are in the path. SKIP without `--syzygy`. |
| `uci` | The engine refused or failed to start (the line quotes its reason and the root cause from its log, e.g. a compiler or libstdc++ mismatch inside `torch.compile`), did not answer `uciok`/`readyok`/`bestmove`, played an illegal move, reported an unsearched or degraded move, or exited non-zero after `quit`. |

The `search` component runs on the CPU; the `uci` component uses your devices.

## Startup refusal

Both `deepfin` and `python -m chess_anti_engine.uci` check the extension facts on
every start. They exit with status 2, printing nothing on stdout, if a required
extension is missing, ABI-incompatible, stale or produces all-zero feature planes.
The message on stderr names the problem and the fix. For development only,
`DEEPFIN_ALLOW_UNVERIFIED_EXTENSIONS=1` downgrades the refusal to a warning.
A failed warmup at startup (for example a broken `torch.compile` toolchain) makes
the engine exit 1 with `engine startup failed:` before `readyok`.
