#!/usr/bin/env bash
# Deferred matched_time arena for threat_planes — run AFTER the matched_sims
# clarity check. ARENA_GAMES env sets game count (default 600).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
games="${ARENA_GAMES:-600}"
b=runs/arena_threat_20260615/baseline_warm/aurora_mlp_out/trainer.pt
t=runs/arena_threat_20260615/threat_warm/aurora_mlp_out/trainer.pt
PYTHONPATH=. .venv/bin/python scripts/arena_standard.py \
  --candidate "$t" --reference "$b" --games "$games" \
  --mode matched_time --ms-per-move 100 --device cuda \
  --out runs/arena_results.jsonl --label threat_v2_vs_v1_matched_time
