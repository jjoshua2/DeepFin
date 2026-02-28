#!/usr/bin/env bash
# 1-hour training run on RTX 5090.
# Uses the distributed server+learner+worker pipeline (same arch as smoke test).
#
# Usage:
#   ./scripts/run_hour.sh
#
# Watch training in a second terminal:
#   tensorboard --logdir /tmp/chess_hour/work/tb
#
# What to look for after an hour:
#   - WDL loss trending down from ~1.1 (random) toward ~0.7-0.9
#   - Policy loss trending down
#   - TensorBoard difficulty/sf_nodes rising over time (network getting stronger)
#   - TensorBoard difficulty/pid_ema_winrate hovering near 0.52 (PID working)
#   - Arena win rate vs checkpoint > 50% (model improving on itself)
#

set -euo pipefail

# Auto-number log file: /tmp/chess_hour_run1.log, run2.log, ...
_next_run=1
while [ -f "/tmp/chess_hour_run${_next_run}.log" ]; do
  _next_run=$((_next_run + 1))
done
RUN_LOG="${RUN_LOG:-/tmp/chess_hour_run${_next_run}.log}"

# Redirect all script output to the run log (tee so terminal still gets it).
exec > >(tee -a "$RUN_LOG") 2>&1

SERVER_ROOT="${SERVER_ROOT:-/tmp/chess_hour/server}"
WORK_DIR="${WORK_DIR:-/tmp/chess_hour/work}"
WORKER_DIR="${WORKER_DIR:-/tmp/chess_hour/worker}"  # base dir; each worker gets _N suffix
SF_BIN="${SF_BIN:-/home/josh/projects/chess/e2e_server/publish/stockfish}"
DURATION_S="${DURATION_S:-3600}"
# WORKER_LOG_FILE is set after workers launch (points to worker_1)
WORKER_LOG_FILE=""

# Opponent difficulty defaults: "Stockfish floor" + bootstrap settings.
# Override via env if you want a different curriculum.
SF_SKILL_LEVEL="${SF_SKILL_LEVEL:-0}"          # 0..20
SF_NODES_START="${SF_NODES_START:-100}"        # starting node budget advertised to workers
SF_NODES_FLOOR="${SF_NODES_FLOOR:-50}"         # PID won't go below this
SF_MULTIPV="${SF_MULTIPV:-3}"

# Opponent random-move bootstrap (PID-controlled).
# Start very high so the net can learn legality/tactics, then PID reduces it toward 0.
OPP_RANDOM_START="${OPP_RANDOM_START:-1.0}"
OPP_RANDOM_STAGE_END="${OPP_RANDOM_STAGE_END:-0.50}"

# Opening book (optional):
# If OPENING_BOOK_PATH is unset but the Stockfish official 2-move book is present in-repo,
# use it by default.
OPENING_BOOK_PATH="${OPENING_BOOK_PATH:-}"
if [ -z "$OPENING_BOOK_PATH" ] && [ -f "data/opening_books/2moves_v2.pgn.zip" ]; then
  OPENING_BOOK_PATH="data/opening_books/2moves_v2.pgn.zip"
fi

# Probability of using the opening book for the starting position.
# If a book is configured and OPENING_BOOK_PROB isn't set, default to 1.0.
OPENING_BOOK_PROB="${OPENING_BOOK_PROB:-}"
if [ -z "$OPENING_BOOK_PROB" ]; then
  if [ -n "$OPENING_BOOK_PATH" ]; then
    OPENING_BOOK_PROB="1.0"
  else
    OPENING_BOOK_PROB="0.0"
  fi
fi

export OPENING_BOOK_PATH
export OPENING_BOOK_PROB

# PID ladder bounds.
PID_MIN_NODES="${PID_MIN_NODES:-$SF_NODES_FLOOR}"
PID_MAX_NODES="${PID_MAX_NODES:-50000}"

PORT="${PORT:-$(python3 - <<'PY'
import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()
PY
)}"
URL="http://127.0.0.1:${PORT}"

echo "============================================================"
echo "  Chess anti-engine — 1-hour training run"
echo "  Run log:   $RUN_LOG"
echo "  Server:    $SERVER_ROOT"
echo "  Learner:   $WORK_DIR"
echo "  SF:        $SF_BIN"
echo "  URL:       $URL"
echo "  Duration:  ${DURATION_S}s"
echo "  Workers:    ${N_WORKERS:-12} processes, target=${TARGET_BATCH_S:-60}s/batch each"
echo "  SF:         skill=${SF_SKILL_LEVEL} nodes_start=${SF_NODES_START} nodes_floor=${SF_NODES_FLOOR} multipv=${SF_MULTIPV}"
echo "  Opp rand:   start=${OPP_RANDOM_START} stage_end=${OPP_RANDOM_STAGE_END}"
echo "  Opening:    path=${OPENING_BOOK_PATH:-<none>} prob=${OPENING_BOOK_PROB}"
echo "  PID:        min_nodes=${PID_MIN_NODES} max_nodes=${PID_MAX_NODES}"
echo "============================================================"

# ── Prior checkpoint (set by autorun loop after healthy runs) ─────
# /tmp/chess_hour_best_prior.pt is written by chess_autorun.sh only
# when the previous run passed the health check. Never carry forward
# weights from a collapsed run.
_PRIOR_CKPT="/tmp/chess_hour_best_prior.pt"

# ── Clean slate ──────────────────────────────────────────────────
rm -rf "$SERVER_ROOT" "$WORK_DIR"
# Remove all previous worker dirs (worker_1, worker_2, ...)
for _old in "${WORKER_DIR}"_*; do rm -rf "$_old"; done
mkdir -p "$SERVER_ROOT/publish" "$SERVER_ROOT/inbox" \
         "$SERVER_ROOT/quarantine" "$SERVER_ROOT/work" \
         "$WORK_DIR"

# ── Restore trainer + PID state from prior healthy run ───────────
# trainer.pt: optimizer momentum/scheduler carry forward so AdamW
#             doesn't overshoot on the first step of a new run.
# pid_state.json: rand/nodes resume where the last run left off.
if [ -f "/tmp/chess_hour_prior_trainer.pt" ]; then
    cp -f "/tmp/chess_hour_prior_trainer.pt" "$WORK_DIR/trainer.pt"
    echo "Restored optimizer state from prior run"
fi
if [ -f "/tmp/chess_hour_prior_pid.json" ]; then
    cp -f "/tmp/chess_hour_prior_pid.json" "$WORK_DIR/pid_state.json"
    echo "Restored PID state from prior run"
fi

# ── Create server user ───────────────────────────────────────────
python3 - <<PY
from pathlib import Path
from chess_anti_engine.server.auth import ensure_user
ensure_user(Path("$SERVER_ROOT") / "users.json", username="worker1", password="pw")
print("Created user worker1")
PY

# ── Bootstrap initial model ──────────────────────────────────────
# 6-layer 128-dim transformer: big enough to learn chess patterns,
# fast enough to train many iterations in an hour on a 5090.
python3 - <<PY
import hashlib, json, os, shutil
from pathlib import Path
import torch
from chess_anti_engine.model import ModelConfig, build_model

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

root = Path("$SERVER_ROOT")
pub  = root / "publish"

use_qk = str(os.environ.get("USE_QK_RMSNORM", "0")).lower() in ("1", "true", "yes", "y")

opening_book_src = str(os.environ.get("OPENING_BOOK_PATH", "") or "").strip()
opening_book_prob = float(os.environ.get("OPENING_BOOK_PROB", "0.0") or 0.0)
opening_book_rec = None
if opening_book_src:
    sp = Path(opening_book_src)
    if sp.exists() and sp.is_file():
        dst = pub / sp.name
        shutil.copy2(sp, dst)
        opening_book_rec = {
            "endpoint": "/v1/opening_book",
            "filename": dst.name,
            "sha256": sha256_file(dst),
        }
    else:
        print(f"WARNING: opening book path not found: {sp}")

mc = ModelConfig(
    kind="transformer",
    embed_dim=256,
    num_layers=8,
    num_heads=8,
    ffn_mult=2,
    use_smolgen=True,
    use_nla=False,
    use_qk_rmsnorm=bool(use_qk),
    use_gradient_checkpointing=False,
)
model = build_model(mc)

# Load prior checkpoint weights if available (carry forward between runs).
prior_ckpt = Path("$_PRIOR_CKPT")
if prior_ckpt.exists():
    ckpt = torch.load(prior_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded prior checkpoint weights from {prior_ckpt}")
else:
    print("No prior checkpoint found — starting from random init")

out = pub / "latest_model.pt"
torch.save({"model": model.state_dict()}, out)
sha = sha256_file(out)

manifest = {
  "server_time_unix": 0,
  "trainer_step": 0,
  "model": {
    "sha256": sha,
    "endpoint": "/v1/model",
    "filename": "latest_model.pt",
    "format": "torch_state_dict",
  },
  "model_config": {
    "kind": "transformer",
    "embed_dim": 256,
    "num_layers": 8,
    "num_heads": 8,
    "ffn_mult": 2,
    "use_smolgen": True,
    "use_nla": False,
    "use_qk_rmsnorm": bool(use_qk),
    "gradient_checkpointing": False,
  },
  "recommended_worker": {
    # Gumbel at the root: guaranteed policy improvement even at 32 sims.
    # Falls back to PUCT at non-root nodes automatically (manager.py logic).
    "mcts": "gumbel",
    "mcts_simulations": 64,    # Full search budget
    "playout_cap_fraction": 0.25,  # 25% get full, 75% get fast (KataGo-style)
    "fast_simulations": 16,

    # Game length + temperature
    "games_per_batch": 4,      # auto-tune will override this
    "max_plies": 200,
    "temperature": 1.0,
    "temperature_decay_start_move": 20,
    "temperature_decay_moves": 60,
    "temperature_endgame": 0.6,

    # Opening diversification
    "random_start_plies": 2,
    "opening_book_prob": float(opening_book_prob),

    # Stockfish difficulty (bootstrap start). PID can ramp nodes/skill upward from here.
    "sf_nodes": int("$SF_NODES_START"),
    "sf_multipv": int("$SF_MULTIPV"),
    "sf_policy_temp": 0.25,
    "sf_policy_label_smooth": 0.05,
    "sf_skill_level": int("$SF_SKILL_LEVEL"),

    # PID-controlled opponent random-move probability.
    "opponent_random_move_prob": float("$OPP_RANDOM_START"),
  },
}
if opening_book_rec is not None:
  manifest["opening_book"] = opening_book_rec
(pub / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
mc_params = sum(p.numel() for p in model.parameters())
print(f"Bootstrapped: {mc_params:,} params, sha={sha[:12]}...")
PY

# ── Server ───────────────────────────────────────────────────────
BOOK_PUBLISH_PATH=""
if [ -n "${OPENING_BOOK_PATH:-}" ]; then
  BOOK_PUBLISH_PATH="$SERVER_ROOT/publish/$(basename "$OPENING_BOOK_PATH")"
fi

SERVER_BOOK_ARG=()
if [ -n "$BOOK_PUBLISH_PATH" ]; then
  SERVER_BOOK_ARG=(--opening-book-path "$BOOK_PUBLISH_PATH")
fi

env PYTHONUNBUFFERED=1 python3 -m chess_anti_engine.server.run_server \
  --server-root "$SERVER_ROOT" \
  --host 127.0.0.1 --port "$PORT" \
  "${SERVER_BOOK_ARG[@]}" \
  >"$SERVER_ROOT/server.log" 2>&1 &
SERVER_PID=$!

# ── Learner ──────────────────────────────────────────────────────
# 128-dim/6-layer transformer on 5090: ~1-2M params, trains in <1s per 200-step iter.
# Min replay 500 so first training happens quickly after a few worker batches.
QK_ARG=""
if [ "${USE_QK_RMSNORM:-0}" != "0" ]; then
  QK_ARG="--use-qk-rmsnorm"
fi

LEARNER_BOOK_ARG=()
if [ -n "${BOOK_PUBLISH_PATH:-}" ]; then
  LEARNER_BOOK_ARG=(--opening-book-path "$BOOK_PUBLISH_PATH")
fi

env PYTHONUNBUFFERED=1 python3 -m chess_anti_engine.learner \
  --server-root "$SERVER_ROOT" \
  --work-dir "$WORK_DIR" \
  --device cuda \
  $QK_ARG \
  "${LEARNER_BOOK_ARG[@]}" \
  \
  --model transformer \
  --embed-dim 256 \
  --num-layers 8 \
  --num-heads 8 \
  --ffn-mult 2 \
  \
  --replay-capacity 200000 \
  --min-replay-size 30000 \
  --train-steps 20 \
  --batch-size 512 \
  --accum-steps 1 \
  \
  --lr 3e-4 \
  --warmup-steps 500 \
  --zclip-z-thresh 2.5 \
  --zclip-alpha 0.97 \
  --zclip-max-norm 1.0 \
  --optimizer nadamw \
  --feature-dropout-p 0.3 \
  --w-volatility 0.05 \
  --w-soft 0.15 \
  --w-sf-wdl 1.0 \
  --w-sf-wdl-end 0.1 \
  \
  --max-shards-per-iter 32 \
  --sleep-seconds 1 \
  \
  --arena \
  --arena-games-target 100 \
  --arena-accept-winrate 0.50 \
  --arena-every-n-steps 1000 \
  --arena-mcts-simulations 1 \
  --no-recommended-progressive-mcts \
  \
  --recommended-mcts gumbel \
  --recommended-mcts-simulations 64 \
  --recommended-playout-cap-fraction 0.25 \
  --recommended-fast-simulations 16 \
  --recommended-games-per-batch 4 \
  --recommended-max-plies 200 \
  --max-plies-ramp-steps 0 \
  --recommended-sf-nodes "$SF_NODES_START" \
  --recommended-sf-multipv "$SF_MULTIPV" \
  --recommended-sf-policy-temp 0.25 \
  --recommended-sf-skill-level "$SF_SKILL_LEVEL" \
  --recommended-opponent-random-move-prob "$OPP_RANDOM_START" \
  --recommended-random-start-plies 2 \
  --recommended-opening-book-prob "$OPENING_BOOK_PROB" \
  \
  --pid \
  --pid-target-winrate 0.53 \
  --pid-ema-alpha 0.05 \
  --pid-min-nodes "$PID_MIN_NODES" \
  --pid-max-nodes "$PID_MAX_NODES" \
  --pid-random-move-prob-start "$OPP_RANDOM_START" \
  --pid-random-move-prob-min 0.0 \
  --pid-random-move-prob-max 1.0 \
  --pid-random-move-stage-end "$OPP_RANDOM_STAGE_END" \
  --pid-max-rand-step 0.01 \
  --pid-min-games-between-adjust 100 \
  >"$WORK_DIR/learner.log" 2>&1 &
LEARNER_PID=$!

# ── Wait for server ──────────────────────────────────────────────
echo "Waiting for server..."
for i in $(seq 1 40); do
  if curl -fsS "$URL/v1/manifest" >/dev/null 2>&1; then
    echo "Server up on port $PORT"
    break
  fi
  sleep 0.5
done

# ── Workers (multi-process) ───────────────────────────────────────
# Run N_WORKERS independent worker processes so CPU encoding is truly parallel
# across cores (bypasses Python GIL that serializes encoding within one process).
# Each process runs a small batch so its GPU calls interleave with others'.
# Total SF processes = N_WORKERS × SF_WORKERS_PER_WORKER.
N_WORKERS="${N_WORKERS:-12}"
TARGET_BATCH_S="${TARGET_BATCH_S:-60}"
MIN_GAMES_PER_WORKER="${MIN_GAMES_PER_WORKER:-4}"
MAX_GAMES_PER_WORKER="${MAX_GAMES_PER_WORKER:-32}"
SF_WORKERS_PER_WORKER="${SF_WORKERS_PER_WORKER:-1}"

WORKER_PIDS=()
for _wi in $(seq 1 "$N_WORKERS"); do
  _WDIR="${WORKER_DIR}_${_wi}"
  _WLOG="${_WDIR}/worker_debug.log"
  mkdir -p "$_WDIR"
  : >"$_WLOG" || true
  (timeout "${DURATION_S}s" env PYTHONUNBUFFERED=1 python3 -m chess_anti_engine.worker \
    --server-url "$URL" \
    --username worker1 --password pw \
    --stockfish-path "$SF_BIN" \
    --work-dir "$_WDIR" \
    --device cuda \
    --sf-workers "$SF_WORKERS_PER_WORKER" \
    --poll-seconds 2 \
    --auto-tune \
    --target-batch-seconds "$TARGET_BATCH_S" \
    --min-games-per-batch "$MIN_GAMES_PER_WORKER" \
    --max-games-per-batch "$MAX_GAMES_PER_WORKER" \
    --seed "$_wi" \
    --log-file "$_WLOG" \
    --log-level info \
    >"$_WDIR/worker.out" 2>&1) &
  WORKER_PIDS+=($!)
done
# WORKER_LOG_FILE points to worker_1 for monitoring (same as before)
WORKER_LOG_FILE="${WORKER_DIR}_1/worker_debug.log"

# ── Monitor ──────────────────────────────────────────────────────
echo
echo "Running ${N_WORKERS} worker processes for ${DURATION_S}s. Watch progress with:"
echo "  tail -f $WORK_DIR/learner.log"
echo "  tail -f $WORKER_LOG_FILE   (worker 1 of ${N_WORKERS})"
echo "  tensorboard --logdir $WORK_DIR/tb"
echo

# Print a summary every 5 minutes while any worker is alive
ELAPSED=0
STATUS_INTERVAL_S=${STATUS_INTERVAL_S:-300}

_any_worker_alive() {
  for _pid in "${WORKER_PIDS[@]}"; do
    kill -0 "$_pid" 2>/dev/null && return 0
  done
  return 1
}

while _any_worker_alive && [ $ELAPSED -lt $DURATION_S ]; do
  REMAIN=$((DURATION_S - ELAPSED))
  SLEEP_FOR=$STATUS_INTERVAL_S
  if [ $REMAIN -lt $SLEEP_FOR ]; then
    SLEEP_FOR=$REMAIN
  fi
  if [ $SLEEP_FOR -le 0 ]; then
    break
  fi

  sleep $SLEEP_FOR
  ELAPSED=$((ELAPSED + SLEEP_FOR))
  MINS=$((ELAPSED / 60))
  echo "--- ${MINS}m elapsed ---"

  # Print current published state from manifest (trainer_step, recommended sf strength).
  python3 - <<PY
import json
from pathlib import Path
mf = Path("$SERVER_ROOT") / "publish" / "manifest.json"
try:
    d = json.loads(mf.read_text())
    rw = d.get("recommended_worker") or {}
    print("manifest: trainer_step=", d.get("trainer_step"), "model_sha=", (d.get("model") or {}).get("sha256", "")[:12])
    print("manifest: sf_nodes=", rw.get("sf_nodes"), "sf_skill_level=", rw.get("sf_skill_level"), "mcts=", rw.get("mcts"), "sims=", rw.get("mcts_simulations"))
except Exception as e:
    print("manifest: (unavailable)", e)
PY

  # Last 2 lines from each worker log
  for _wi in $(seq 1 "$N_WORKERS"); do
    _WLOG="${WORKER_DIR}_${_wi}/worker_debug.log"
    _last=$(tail -n 1 "$_WLOG" 2>/dev/null || true)
    [ -n "$_last" ] && echo "  worker${_wi}: $_last"
  done
done

for _pid in "${WORKER_PIDS[@]}"; do
  wait "$_pid" 2>/dev/null || true
done

# ── Summary ──────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  Run complete. Summary:"
echo "============================================================"
echo
echo "=== Final learner log (last 40 lines) ==="
tail -40 "$WORK_DIR/learner.log" || true

echo
echo "=== Worker batch logs (last 5 lines each) ==="
for _wi in $(seq 1 "$N_WORKERS"); do
  _WLOG="${WORKER_DIR}_${_wi}/worker_debug.log"
  echo "--- worker${_wi} ---"
  tail -5 "$_WLOG" 2>/dev/null || true
done

echo
echo "=== Worker stdout/stderr (last 10 lines each) ==="
for _wi in $(seq 1 "$N_WORKERS"); do
  _WOUT="${WORKER_DIR}_${_wi}/worker.out"
  _last=$(tail -3 "$_WOUT" 2>/dev/null || true)
  [ -n "$_last" ] && echo "--- worker${_wi} ---" && echo "$_last"
done

echo
python3 - <<PY
from pathlib import Path
root = Path("$SERVER_ROOT")
proc = list((root / "processed").rglob("*.npz")) if (root / "processed").exists() else []
print(f"Total shards processed: {len(proc)}")
try:
    import json
    manifest = json.loads((root / "publish" / "manifest.json").read_text())
    step = manifest.get("trainer_step", "?")
    print(f"Final trainer step: {step}")
except Exception:
    pass
PY

kill $SERVER_PID $LEARNER_PID 2>/dev/null || true
wait $SERVER_PID $LEARNER_PID 2>/dev/null || true
echo "Done."
