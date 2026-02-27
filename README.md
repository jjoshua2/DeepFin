# chess-anti-engine
A Python/PyTorch self-play training loop for a transformer chess model that trains primarily vs Stockfish ("anti-engine" style), with LC0-style 4672-move policy encoding.

## Status: how close is this to a self-improving pipeline?
The core **self-improving loop is working end-to-end**:
1) generate games (`selfplay`) using MCTS + the current network vs Stockfish
2) write positions into a replay buffer
3) train the network for a fixed number of gradient steps
4) repeat for N outer iterations, checkpointing along the way

It also includes several spec-critical stabilizers:
- LC0-style `POLICY_SIZE=4672` move encoding (+ legal-move masking)
- soft-target losses for Stockfish WDL + MultiPV-derived soft policy targets
- adaptive Stockfish difficulty PID controller (changes SF node budget by winrate)
- opening diversification: optional opening book (`.bin` / `.pgn` / `.pgn.zip`) and/or random-start plies
- AlphaZero/LC0-style temperature sampling from MCTS visits, with optional linear decay schedule
- ONNX export + parity smoke test

What’s still missing for a more "production" self-improving pipeline:
- richer evaluation policies (currently a simple **latest vs best** arena gate is available in distributed mode)
- distributed hardening (quotas, manifests/versioning, arenas vs multiple baselines, etc.)

See `future_ideas.md` for longer-term ideas like opponent mixing / league training.

## Setup
```bash
pip install -e .
pip install -e ".[dev]"      # pytest
pip install -e ".[tune]"     # ray[tune] + optuna (default harness)
pip install -e ".[onnx]"     # onnx + onnxruntime (optional)
```

## Quickstart: single run (selfplay + train loop)
You must point at a Stockfish binary.

```bash
python -m chess_anti_engine.run \
  --mode single \
  --stockfish-path /path/to/stockfish \
  --iterations 10
```

Outputs:
- checkpoints: `--work-dir` (default `runs/`)
- tensorboard logs: `<work_dir>/tb`

## Quickstart: YAML config
`run.py` does a two-pass parse so YAML provides defaults and CLI overrides them.

1) Copy and edit `configs/default.yaml` (at minimum set `stockfish.path`).
2) Run:

```bash
python -m chess_anti_engine.run --config configs/default.yaml --mode single
```

## Ray Tune harness (Optuna + ASHA)
```bash
python -m chess_anti_engine.run \
  --config configs/default.yaml \
  --mode tune \
  --num-samples 8
```

## Opening diversification
In YAML:
- `selfplay.opening_book_path`: path to `.bin` (polyglot), `.pgn`, or `.pgn.zip` / `.zip`
- `selfplay.random_start_plies`: play N random legal half-moves from startpos

Example (Stockfish official book): set `opening_book_path` to `2moves_v2.pgn.zip`.

## Temperature schedule (LC0-ish)
By default, the YAML config uses a simple LC0-like linear decay:
- start at `temperature=1.0`
- start decaying at move 20 over 60 moves
- clamp at `temperature_endgame=0.6`

Knobs:
- `selfplay.temperature`
- `selfplay.temperature_decay_start_move`
- `selfplay.temperature_decay_moves`
- `selfplay.temperature_endgame`

## Distributed pipeline (server + learner + many workers)
This repo now supports a simple server/client setup:
- the **learner** trains continuously by ingesting uploaded selfplay shards from disk
- the **HTTP server** serves the latest model + opening book and accepts shard uploads
- many **workers** run selfplay locally vs Stockfish and upload shards back

### Install
On the server/learner machine:
```bash
pip install -e ".[server]"
```
On worker machines:
```bash
pip install -e ".[worker]"
```

### 1) Create upload users
The upload endpoint uses HTTP Basic auth (username/password) backed by `server/users.json`.

```bash
python -m chess_anti_engine.server.manage_users --users-db server/users.json add alice
```

### 2) Run the learner
The learner watches `server/inbox/` for `*.npz` shards, trains, and publishes:
- `server/publish/latest_model.pt`
- `server/publish/manifest.json`

By default, the learner runs an **arena gate** (latest vs best) and only promotes `best_model.pt` when the challenger scores above the configured threshold.

Optional worker artifacts you can publish:
- `--stockfish-binary-path /path/to/stockfish` publishes Stockfish for workers to download.
- `--worker-wheel-path /path/to/chess_anti_engine.whl` publishes a worker wheel so workers can self-update.

```bash
python -m chess_anti_engine.learner \
  --server-root server \
  --work-dir server/work \
  --stockfish-binary-path /path/to/stockfish \
  --worker-wheel-path /path/to/chess_anti_engine.whl
```

### 3) Run the HTTP server
```bash
python -m chess_anti_engine.server.run_server \
  --server-root server \
  --host 0.0.0.0 \
  --port 8000
```

### 4) Run a worker
Workers poll the manifest, download the latest model (and opening book if advertised), generate selfplay, and upload shards.
By default, workers use the server-provided `recommended_worker` settings so volunteers usually don’t need to set any MCTS/SF strength knobs.
The project server is intended to control **Stockfish nodes/MultiPV and MCTS simulations** to keep the training distribution consistent.

```bash
python -m chess_anti_engine.worker \
  --server-url http://SERVER_HOST:8000 \
  --username alice \
  --stockfish-path /path/to/stockfish

# Or: download Stockfish from the server (if published in the manifest)
python -m chess_anti_engine.worker \
  --server-url http://SERVER_HOST:8000 \
  --username alice \
  --stockfish-from-server

# Optional: allow self-update from a server-published wheel
# (If the server blocks this worker as too old, the worker will fetch /v1/update_info to find the wheel.)
python -m chess_anti_engine.worker \
  --server-url http://SERVER_HOST:8000 \
  --username alice \
  --stockfish-from-server \
  --self-update

# Optional: calibrate once, then future runs reuse <work_dir>/worker.yaml
python -m chess_anti_engine.worker \
  --server-url http://SERVER_HOST:8000 \
  --username alice \
  --stockfish-path /path/to/stockfish \
  --calibrate

# Optional: continuously auto-tune games_per_batch (throughput only)
python -m chess_anti_engine.worker \
  --server-url http://SERVER_HOST:8000 \
  --username alice \
  --stockfish-path /path/to/stockfish \
  --auto-tune

# By default, the worker writes <work_dir>/worker.yaml and stores the password there.
# Opt out with:
#   --no-save-config
#   --no-save-password
python -m chess_anti_engine.worker --help | grep save

# Debug only: override server-managed strength knobs (not recommended)
# Requires an explicit env var:
#   CHESS_ANTI_ENGINE_DEBUG_OVERRIDES=1
python -m chess_anti_engine.worker --help | grep allow-overrides
```

Notes:
- For v1 we do **not** distribute Stockfish binaries from the server; each worker provides its own `--stockfish-path`.
- Server-side moderation can be done by disabling a user: `manage_users disable alice`.
- Invalid shard uploads are quarantined under `server/quarantine/invalid/` (and the server returns HTTP 200 with `rejected=true` so workers don’t retry forever).
- `server/publish/manifest.json` includes `recommended_worker` settings and (if configured) `opening_book.sha256` so workers refresh cached books when they change.

## Run tests
```bash
pytest
```
