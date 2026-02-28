#!/usr/bin/env bash
# End-to-end distributed smoke test: server + learner + worker.
# Exercises the Gumbel path by setting recommended_worker.mcts=gumbel.
#
# Usage:
#   ./scripts/e2e_distributed_smoke_gumbel.sh
#
# Environment overrides (optional):
#   SERVER_ROOT=/tmp/chess_e2e_server_gumbel
#   WORKER_DIR=/tmp/chess_e2e_worker_gumbel
#   SF_BIN=/path/to/stockfish
#   TIMEOUT_S=180
#
# Notes:
# - Do NOT use --calibrate for the worker here; calibrate exits before writing/uploading a shard.

set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/tmp/chess_e2e_server_gumbel}"
WORKER_DIR="${WORKER_DIR:-/tmp/chess_e2e_worker_gumbel}"
SF_BIN="${SF_BIN:-/home/josh/projects/chess/e2e_server/publish/stockfish}"
TIMEOUT_S="${TIMEOUT_S:-180}"

PORT="${PORT:-$(python - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)}"
URL="http://127.0.0.1:${PORT}"

printf "Server root: %s\n" "$SERVER_ROOT"
printf "Worker dir:  %s\n" "$WORKER_DIR"
printf "Stockfish:   %s\n" "$SF_BIN"
printf "URL:         %s\n" "$URL"
printf "Timeout:     %ss\n" "$TIMEOUT_S"

rm -rf "$SERVER_ROOT" "$WORKER_DIR"
mkdir -p "$SERVER_ROOT/publish" "$SERVER_ROOT/inbox" "$SERVER_ROOT/quarantine" "$SERVER_ROOT/work" "$WORKER_DIR"

python - <<PY
from pathlib import Path
from chess_anti_engine.server.auth import ensure_user
ensure_user(Path("$SERVER_ROOT") / "users.json", username="alice", password="pw")
print("created user alice")
PY

python - <<PY
import hashlib, json
from pathlib import Path
import torch
from chess_anti_engine.model import ModelConfig, build_model

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

root = Path("$SERVER_ROOT")
pub = root / "publish"

mc = ModelConfig(kind="tiny", embed_dim=64, num_layers=2, num_heads=4, ffn_mult=2, use_smolgen=False, use_nla=False)
model = build_model(mc)
out = pub / "latest_model.pt"
torch.save({"model": model.state_dict()}, out)
sha = sha256_file(out)

manifest = {
  "server_time_unix": 0,
  "trainer_step": 0,
  "model": {"sha256": sha, "endpoint": "/v1/model", "filename": "latest_model.pt", "format": "torch_state_dict"},
  "model_config": {
    "kind":"tiny","embed_dim":64,"num_layers":2,"num_heads":4,"ffn_mult":2,
    "use_smolgen":False,"use_nla":False,"gradient_checkpointing":False
  },
  "recommended_worker": {
    "games_per_batch": 1,
    "max_plies": 20,

    # Force Gumbel at the root.
    "mcts": "gumbel",

    # Keep budgets modest for a fast smoke.
    "mcts_simulations": 32,
    "playout_cap_fraction": 1.0,
    "fast_simulations": 8,

    "opening_book_prob": 0.0,
    "random_start_plies": 0,

    "sf_nodes": 30,
    "sf_multipv": 2,
    "sf_policy_temp": 0.25,
    "sf_policy_label_smooth": 0.05,

    "temperature": 1.0,
    "temperature_decay_start_move": 20,
    "temperature_decay_moves": 60,
    "temperature_endgame": 0.6
  }
}
(pub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
print("bootstrapped publish with sha", sha)
PY

(timeout "${TIMEOUT_S}s" python -m chess_anti_engine.server.run_server \
  --server-root "$SERVER_ROOT" --host 127.0.0.1 --port "$PORT" \
  >"$SERVER_ROOT/server.log" 2>&1) &
SERVER_PID=$!

(timeout "${TIMEOUT_S}s" python -m chess_anti_engine.learner \
  --server-root "$SERVER_ROOT" \
  --work-dir "$SERVER_ROOT/work" \
  --min-replay-size 1 \
  --train-steps 2 \
  --batch-size 1 \
  --model tiny --embed-dim 64 --num-layers 2 --num-heads 4 --ffn-mult 2 --no-smolgen \
  \
  --recommended-mcts gumbel \
  --recommended-mcts-simulations 32 \
  --recommended-playout-cap-fraction 1.0 \
  --recommended-fast-simulations 8 \
  --recommended-games-per-batch 1 \
  --recommended-max-plies 20 \
  --recommended-sf-nodes 30 \
  --recommended-sf-multipv 2 \
  --recommended-opening-book-prob 0.0 \
  --sleep-seconds 0.5 \
  --max-shards-per-iter 64 \
  >"$SERVER_ROOT/learner.log" 2>&1) &
LEARNER_PID=$!

for i in $(seq 1 80); do
  if curl -fsS "$URL/v1/manifest" >/dev/null 2>&1; then
    echo "server up"
    break
  fi
  sleep 0.25
done

(timeout 90s python -m chess_anti_engine.worker \
  --server-url "$URL" \
  --username alice --password pw \
  --stockfish-path "$SF_BIN" \
  --work-dir "$WORKER_DIR" \
  --poll-seconds 0.5 \
  --auto-tune --target-batch-seconds 5 \
  >"$WORKER_DIR/worker.log" 2>&1) || true

sleep 2

echo
echo "=== server log tail ==="
tail -n 80 "$SERVER_ROOT/server.log" || true

echo
echo "=== learner log tail ==="
tail -n 160 "$SERVER_ROOT/learner.log" || true

echo
echo "=== worker log tail ==="
tail -n 160 "$WORKER_DIR/worker.log" || true

echo
echo "=== inbox files ==="
find "$SERVER_ROOT/inbox" -maxdepth 3 -type f -print | sed -e "s#^$SERVER_ROOT/##" | head -n 200

echo
echo "=== processed files ==="
find "$SERVER_ROOT/processed" -maxdepth 3 -type f -print 2>/dev/null | sed -e "s#^$SERVER_ROOT/##" | head -n 200 || true

kill "$SERVER_PID" "$LEARNER_PID" >/dev/null 2>&1 || true
wait "$SERVER_PID" "$LEARNER_PID" >/dev/null 2>&1 || true

python - <<PY
from pathlib import Path
root = Path("$SERVER_ROOT")
inbox = list((root/"inbox"/"alice").glob("*.npz")) if (root/"inbox"/"alice").exists() else []
proc = list((root/"processed"/"alice").glob("*.npz")) if (root/"processed"/"alice").exists() else []
print("Inbox shards:", len(inbox))
print("Processed shards:", len(proc))
assert inbox or proc, "no shards uploaded/processed"
print("OK")
PY
