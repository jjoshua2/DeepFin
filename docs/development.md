# Development

Shared project constraints are in [CLAUDE.md](../CLAUDE.md). This page covers local
setup and verification; [operations](operations.md) covers an existing live run.

## Environment and commands

Use standard, GIL-enabled Python 3.13 for development and CI, with a current patch
release. `.python-version` is the shared version selector. Install the uv version
specified by `tool.uv.required-version` in `pyproject.toml` (currently 0.12.10), then
create the locked CPU development environment in a fresh worktree:

```bash
uv sync --locked --extra dev --extra cpu
. .venv/bin/activate
```

The `dev` extra supplies the server, training, tuning, ONNX and test dependencies.
`uv.lock` records their complete resolved dependency graph and wheel hashes;
`--locked` fails if the project requirements and lock disagree. CI uses the same
command. The lock targets Linux x86-64 with Python 3.13, our development/CI platform.
Other platforms and older workers retain the ordinary package installation path,
such as `python -m pip install -e '.[worker]'`; those installs do not use this lock.
Build dependencies are constrained in `pyproject.toml`, and native sources, headers
and build flags participate in uv's build cache key.

Use a separate checkout/environment for each accelerator or native build variant:
editable installations share the checkout's in-place extensions. Avoid syncing an
environment or rebuilding extensions used by active jobs. After activating, use
`python` directly (or `uv run --no-sync`) so a later command cannot silently resync
away the selected extras.

The package's older Python floor is retained for existing workers; it is not the
recommended development interpreter. Python 3.14 needs separate dependency work:
the pinned NumPy 2.2.6 and Zarr 2's Numcodecs 0.15.1 have no CPython 3.14 wheels.
The July 2026 experiment in the [historical ledger](experiment_ledger.md) tested
**free-threaded** 3.14, which additionally lacked compatible Ray and project native
extensions. Its failure does not rule out a newer standard Python interpreter.

An interpreter upgrade requires a fresh environment and native-extension rebuild.
Exercise checkpoint restoration and Ray in disposable state before moving an existing
run; pickled runtime state and compiled artifacts are not an interpreter migration
contract. Changing the development default does not migrate running jobs.

For CUDA development on the RTX 5090, replace `--extra cpu` with `--extra cu130`.
These extras are mutually exclusive in uv and select the official PyTorch index
explicitly. Generic dependencies still come from PyPI. CI checks the installed Torch
variant as well. This follows [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).

To update a dependency, change its constraint if pinned, run
`uv lock --upgrade-package PACKAGE`, sync the relevant profile with `--locked`, and
validate the affected paths. Commit the lock with the change. A broad `uv lock
--upgrade` is an intentional dependency upgrade, not a routine setup step. The weekly
lint canary starts from the lock and deliberately upgrades its selected tools outside
it; a canary result is not evidence that the locked environment changed.

Validate Torch upgrades in a separate environment with model/loss/optimizer tests and
an actual GPU forward/backward and compiled-inference check. Existing AOT packages and
compiled caches are version-specific; rebuild and validate them before production adopts
a different Torch build. Installing a candidate does not upgrade a running process.

From the repository root:

```bash
python -m pytest tests/test_param_count.py       # example focused CPU check
python scripts/validate.py cpu                  # ordinary CI suite, GPU hidden
python scripts/validate.py capped               # worker tests preserve the local cap
./scripts/lint.sh tests/test_param_count.py      # focused static feedback
python scripts/validate.py lint                 # same whole-repo gate as CI
```

`tests/conftest.py` caps torch at two threads by default. The terminal summary reports
the regime, including under `-q`. Use `CAE_TEST_THREADS=auto` only on a dedicated
runner or during a confirmed pause with no competing work; an explicit thread count
is also supported. Invalid values retain the cap. In CI YAML, quote string values.

The shared validator prints the interpreter, Torch build, requested thread count,
native backend, elapsed time and failing command. CPU suites hide GPUs even in a CUDA
environment; `--require-cpu-wheel` additionally verifies CI's actual dependency variant.
The `capped` suite always uses two threads, including when CI inherits `auto`, and
checks the cap after in-process worker tests. Direct pytest retains its permissive
thread-setting aliases; the validator accepts a positive count or `auto`.

The PEXT suite needs a separate checkout built with `CFLAGS=-mbmi2`:

```bash
CFLAGS=-mbmi2 uv sync --locked --extra dev --extra cpu
. .venv/bin/activate
python scripts/validate.py pext
```

The validator checks native freshness and the actual backend before pytest; it never
rebuilds an extension while another process may be using it. Keep the same `CFLAGS`
on later syncs of that checkout. CI covers both the portable magic build and this
PEXT build. Both local and CI lint run `scripts/lint.sh`, including vulture.

## Runtime upgrade smokes

Run these explicitly when qualifying Python, Torch or Ray changes, after budgeting
resources alongside existing work:

```bash
python scripts/runtime_smoke.py ray --report /tmp/deepfin-ray-smoke.json
# In the CUDA profile, with available GPU capacity:
python scripts/runtime_smoke.py gpu --report /tmp/deepfin-gpu-smoke.json
```

The Ray check starts its own disposable one-CPU, zero-GPU local cluster, restores a
persisted Tuner and continues a checkpoint. The GPU check uses a tiny project model
for BF16 backward, fused AdamW, model/optimizer roundtrip and compiled/eager parity.
Its default allocator fraction is 0.08, with two CPU threads and one compiler worker;
the fraction neither reserves memory nor guarantees safety beside a live workload.
Both commands report the failing stage and exit nonzero on failure. They are opt-in,
not part of ordinary pytest. For a basic CUDA multiprocessing diagnostic, the existing
`scripts/cuda_sanity_check.py` remains available.

These synthetic checks qualify the exercised runtime paths. They do not prove that
an existing live Ray directory, production checkpoint or compiled artifact migrates;
validate a copied artifact separately before adopting an upgrade.

Scripts that import the package need installation or `PYTHONPATH=.`. CLI modes are
`train`, `tune` and `salvage`. `--mode train` is a single distributed trial, including
the server and worker, not a separate local selfplay implementation. Use
`scripts/train.sh` and the operations procedure to manage an existing production run.

## Validation by change scope

- Documentation and instruction changes: inspect links, commands, discovery paths and
  consistency against source. Run relevant tests only if the change affects them.
- Behavior changes: exercise the changed behavior and a meaningful failure case.
  Prefer deterministic tests in encoding, replay, MCTS and target construction.
- Distributed/selfplay wiring: include `tests/test_e2e_smoke.py`; for a search-only
  change the `-k gumbel_selfplay_smoke` selection exercises the real chain.
- Code changes: run focused checks while iterating, then the whole-repo
  `./scripts/lint.sh` gate once on the final candidate. Expand to the ordinary CI suite
  when impact crosses components; select slow tests for the paths that need them.

The lint script includes ruff, basedpyright and other checks. Only explicit paths
narrow basedpyright; `--changed` still uses its whole-repo scope. Address introduced
findings. If a failure is environmental or already present on the base, establish and
report that evidence rather than weakening a rule or changing unrelated code.
Basedpyright suppressions use `# pyright: ignore[reportRuleName]`, not mypy rule names.

## Native extensions

Normal installs build portable extensions. For deployment on the production host,
`python scripts/build_production_extensions.py` forces a native/LTO rebuild using the
validated GCC 15 compiler (`CAE_GCC15_CC` can select it). Rebuild after relevant C or
header changes: incremental dependency detection can miss headers. Keep distributed
wheels portable unless all target CPUs are known to match. Building a file does not
replace an extension already loaded in a running process.

## Code and evidence map

| Task | Starting points |
| --- | --- |
| Input/history and policy indices | `chess_anti_engine/encoding/`, `moves/`, `moves/torch_maps.py` |
| Network, target and loss semantics | `chess_anti_engine/model/`, `chess_anti_engine/train/`, [model heads](model_heads.md) |
| Search and game generation | `mcts/`, `selfplay/`, `stockfish/` under `chess_anti_engine/` |
| Ingest, replay and distributed config | `replay/`, `server/`, `worker.py`, `tune/` under `chess_anti_engine/` |
| UCI and NNUE work | `chess_anti_engine/uci/`, [NNUE design](nnue_native_eval.md), [incremental evaluation](nnue_incremental.md) |
| Offline generation and training | `scripts/generate_bootstrap.py`, `scripts/offline_replay_epoch.py` |
| Prior experiments and measurements | [Experiment navigation](experiments/README.md), [loop audit](rl_loop_audit.md) |

Use checkpoint architecture metadata for comparisons and the resolved YAML for current
values. For tablebases, inspect both `syzygy_path` and `stockfish_syzygy_path`: they can
intentionally differ. Record each engine's actual paths and coverage; directory names
alone do not prove which WDL/DTZ tables are installed.
