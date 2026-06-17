#!/usr/bin/env bash
# Safe launcher for long offline_replay_epoch.py sidecar experiments.
#
# Examples:
#   scripts/sidecar_replay.sh smoke --out-dir runs/sidecar/foo -- \
#     --config configs/pbt2_small.yaml --replay-dir runs/pbt2_small/.../replay_shards
#
#   scripts/sidecar_replay.sh launch --name foo_sidecar --out-dir runs/sidecar/foo -- \
#     --config configs/pbt2_small.yaml --replay-dir runs/pbt2_small/.../replay_shards \
#     --batch-size 512 --report-every-shards 25
#
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${SIDECAR_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/sidecar_replay.sh smoke  --out-dir DIR [--smoke-shards N] -- OFFLINE_REPLAY_ARGS...
  scripts/sidecar_replay.sh launch --name SESSION --out-dir DIR [--smoke-shards N] [--no-smoke] [--force] -- OFFLINE_REPLAY_ARGS...
  scripts/sidecar_replay.sh status --name SESSION [--out-dir DIR]
  scripts/sidecar_replay.sh tail   --out-dir DIR [--lines N]

launch runs a small foreground smoke first, then starts offline_replay_epoch.py
inside tmux with PYTHONUNBUFFERED=1 and logs to DIR/run.log.

OFFLINE_REPLAY_ARGS are passed to scripts/offline_replay_epoch.py. The wrapper
appends --out-dir DIR; for smoke it also appends --max-shards N.
EOF
}

shell_quote_args() {
  local out="" arg
  for arg in "$@"; do
    printf -v arg "%q" "${arg}"
    out+="${arg} "
  done
  printf "%s" "${out}"
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for durable sidecar launch" >&2
    return 1
  fi
}

split_wrapper_args() {
  name=""
  out_dir=""
  log_path=""
  smoke_shards=2
  no_smoke=0
  force=0
  lines=40
  replay_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        name="${2:?missing value for --name}"
        shift 2
        ;;
      --out-dir)
        out_dir="${2:?missing value for --out-dir}"
        shift 2
        ;;
      --log)
        log_path="${2:?missing value for --log}"
        shift 2
        ;;
      --smoke-shards)
        smoke_shards="${2:?missing value for --smoke-shards}"
        shift 2
        ;;
      --no-smoke)
        no_smoke=1
        shift
        ;;
      --force)
        force=1
        shift
        ;;
      --lines)
        lines="${2:?missing value for --lines}"
        shift 2
        ;;
      --)
        shift
        replay_args=("$@")
        return 0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown wrapper argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

require_out_dir() {
  if [[ -z "${out_dir}" ]]; then
    echo "--out-dir is required" >&2
    exit 2
  fi
}

require_replay_args() {
  if [[ "${#replay_args[@]}" -eq 0 ]]; then
    echo "offline replay args are required after --" >&2
    exit 2
  fi
}

run_smoke() {
  require_out_dir
  require_replay_args
  local smoke_dir="${out_dir%/}/_smoke"
  mkdir -p "${smoke_dir}"
  echo "Running sidecar smoke: out=${smoke_dir} max_shards=${smoke_shards}"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "${python_bin}" scripts/offline_replay_epoch.py \
    "${replay_args[@]}" \
    --out-dir "${smoke_dir}" \
    --max-shards "${smoke_shards}" \
    --report-every-shards 1
  echo "Smoke passed: ${smoke_dir}"
}

launch_sidecar() {
  require_tmux
  require_out_dir
  require_replay_args
  if [[ -z "${name}" ]]; then
    echo "--name is required for launch/status" >&2
    exit 2
  fi
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "tmux session already exists: ${name}" >&2
    exit 1
  fi
  if [[ -e "${out_dir%/}/results.jsonl" && "${force}" -eq 0 ]]; then
    echo "results already exist in ${out_dir}; pass --force to launch anyway" >&2
    exit 1
  fi

  if [[ "${no_smoke}" -eq 0 ]]; then
    run_smoke
  fi

  mkdir -p "${out_dir}"
  if [[ -z "${log_path}" ]]; then
    log_path="${out_dir%/}/run.log"
  fi

  local -a command_args
  local command_str root_q log_q
  command_args=("${python_bin}" scripts/offline_replay_epoch.py "${replay_args[@]}" --out-dir "${out_dir}")
  command_str="$(shell_quote_args "${command_args[@]}")"
  printf -v root_q "%q" "${repo_root}"
  printf -v log_q "%q" "${log_path}"

  tmux new-session -d -s "${name}" \
    "cd ${root_q} && export PYTHONPATH=. PYTHONUNBUFFERED=1 && ${command_str} 2>&1 | tee -a ${log_q}"

  echo "Launched ${name}"
  echo "  out: ${out_dir}"
  echo "  log: ${log_path}"
  echo "  attach: tmux attach -t ${name}"
}

status_sidecar() {
  if [[ -z "${name}" ]]; then
    echo "--name is required for status" >&2
    exit 2
  fi
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "Running: ${name}"
  else
    echo "Not running: ${name}"
  fi
  if [[ -n "${out_dir}" ]]; then
    if [[ -s "${out_dir%/}/results.jsonl" ]]; then
      echo "Results: ${out_dir%/}/results.jsonl"
    fi
    if [[ -f "${out_dir%/}/run.log" ]]; then
      echo "Log: ${out_dir%/}/run.log"
      tail -n 5 "${out_dir%/}/run.log"
    fi
  fi
}

tail_sidecar() {
  require_out_dir
  local log="${out_dir%/}/run.log"
  if [[ ! -f "${log}" ]]; then
    echo "missing log: ${log}" >&2
    exit 1
  fi
  tail -n "${lines}" "${log}"
}

cmd="${1:-}"
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
  usage
  exit 0
fi
shift

split_wrapper_args "$@"

case "${cmd}" in
  smoke) run_smoke ;;
  launch) launch_sidecar ;;
  status) status_sidecar ;;
  tail) tail_sidecar ;;
  *)
    echo "unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
