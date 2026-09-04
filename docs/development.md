# Development

Shared project constraints are in [CLAUDE.md](../CLAUDE.md). This page covers local
setup and verification; [operations](operations.md) covers an existing live run.

## Environment and commands

Use Python compatible with `pyproject.toml` and install into an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The `dev` extra supplies the server, training, tuning, ONNX and test dependencies.
`.[worker]` is the lighter client install. Dependency pins and build requirements live
in `pyproject.toml`; don't infer them from another checkout's environment. An old
setuptools without PEP 660 can fail editable installation; build isolation uses the
project's build requirements. Avoid upgrading an environment used by active jobs.

From the repository root:

```bash
python -m pytest tests/test_param_count.py       # example focused CPU check
python -m pytest -m 'not slow'                   # ordinary CI suite
./scripts/lint.sh tests/test_param_count.py      # focused static feedback
./scripts/lint.sh                               # whole-repo static gate
```

`tests/conftest.py` caps torch at two threads by default. The terminal summary reports
the regime, including under `-q`. Use `CAE_TEST_THREADS=auto` only on a dedicated
runner or during a confirmed pause with no competing work; an explicit thread count
is also supported. Invalid values retain the cap. In CI YAML, quote string values.

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
