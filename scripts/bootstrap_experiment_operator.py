#!/usr/bin/env python3
"""Bounded bootstrap arena queue; supports existing launch receipts without restarting jobs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tempfile
from contextlib import contextmanager
from pathlib import Path

LOOP = Path("/home/josh/projects/chess/scratchpad/bt4_joint20/autonomous_loop_20260914")
RUNTIME = Path("/tmp/deepfin-ordered-arena-lookahead-runtime")
BOOK = "/home/josh/projects/chess/data/opening_books/8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip"
CERES = Path("/home/josh/projects/chess/scratchpad/bt4_joint20/g10_ceres_batchv3_run07_complete_v1")
LOCK = LOOP / "gpu.lock"


def append_results(line: str) -> None:
    path = LOOP / "RESULTS.md"
    with path.open("a") as f:
        f.write(line.rstrip() + "\n")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def dump(path: Path, obj: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(obj, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def log(event: str, **kwargs) -> None:
    rec = {"unix": time.time(), "event": event, **kwargs}
    with (LOOP / "log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def mem_avail_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def disk_free_gib() -> float:
    return shutil.disk_usage("/home/josh/projects/chess").free / 2**30


def gpu_apps() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def harvest_results_jsonl(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last = path.read_text().splitlines()[-1]
    d = json.loads(last)
    return {
        "elo": d.get("elo"),
        "elo_ci95": d.get("elo_ci95"),
        "score": d.get("score"),
        "pentanomial": d.get("pentanomial"),
        "pairs": d.get("pairs"),
        "games": d.get("games"),
        "duration_s": d.get("duration_s"),
        "truncated": d.get("truncated"),
        "seed": d.get("seed"),
    }


def validate_arena_bank(item: dict, result: dict) -> None:
    expected = int(item["games"])
    if result.get("truncated") is not False or result.get("games") != expected:
        raise ValueError("truncated or wrong result game count")
    if result.get("pairs") != expected // 2 or expected % 2:
        raise ValueError("wrong result pair count")
    games = []
    with (Path(item["out"]) / "arena.games.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "game":
                games.append(record)
    pairs = {}
    for game in games:
        key = game["pair_id"]
        pairs.setdefault(key, []).append(game)
    if len(games) != expected or len(pairs) != expected // 2:
        raise ValueError("incomplete actual game bank")
    for pair in pairs.values():
        if len(pair) != 2 or {g["half"] for g in pair} != {0, 1}:
            raise ValueError("missing or duplicate pair half")
        if {g["a_is_white"] for g in pair} != {True, False}:
            raise ValueError("pair is not color swapped")
        if len({g["opening_fen"] for g in pair}) != 1:
            raise ValueError("pair opening mismatch")
    score = sum(g["score_candidate"] for g in games) / expected
    if abs(score - float(result["score"])) > 0.00001:
        raise ValueError("bank score differs from result")


def launch_budget(item: dict) -> int:
    # Full declared horizon plus 30 seconds TERM grace and 10 seconds receipt slack.
    budget = int(float(item["max_seconds"]))
    deadline = float(load(LOOP / "STATE.json")["deadline_unix"])
    if budget <= 0 or time.time() + budget + 40 > deadline:
        raise ValueError("full experiment and termination grace do not fit deadline")
    return budget


def harvest_arena(item: dict, state: dict) -> None:
    if item.get("status") != "running":
        return
    out = Path(item["out"])
    terminal = out / "parent_outer_terminal.json"
    if not terminal.exists():
        pid_file = out / "parent_outer.pid"
        if item["status"] == "running" and pid_file.exists():
            pid = int(pid_file.read_text())
            if not pid_alive(pid):
                item["status"] = "needs_recovery"
                log("arena_wrapper_missing_terminal", id=item["id"], pid=pid)
        return
    term = json.loads(terminal.read_text())
    result = harvest_results_jsonl(out / "arena.results.jsonl")
    if term.get("returncode") != 0 or result is None:
        item["status"] = "failed"
        log("arena_failed", id=item["id"], terminal=term, result=result)
        return
    try:
        validate_arena_bank(item, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        item["status"] = "failed"
        log("arena_invalid_completion", id=item["id"], reason=str(exc))
        return
    item["status"] = "logged"
    item["result"] = result
    state["completed"].append(item["id"])
    log("arena_logged", id=item["id"], **result)
    ci = result.get("elo_ci95") or [None, None]
    append_results(
        f"- {item['id']}: {result.get('games')} games seed {result.get('seed')} "
        f"**{result.get('elo'):+.1f} Elo** [{ci[0]}, {ci[1]}] "
        f"score {result.get('score')} pentanomial {result.get('pentanomial')}"
    )


def gpu_busy() -> str | None:
    try:
        apps = gpu_apps()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"probe_failed:{exc}"
    if apps:
        return f"compute_apps:{apps.splitlines()[0][:120]}"
    # A missing wrapper does not establish that its detached children stopped.
    for item in load(LOOP / "queue.json")["items"]:
        if item.get("status") in {"running", "launching", "needs_recovery"}:
            return f"unresolved_job:{item['id']}"
    return None


def start_wrapper(item: dict, wrapper: Path, out: Path):
    launch_budget(item)
    # Persist intent first: a crash between Popen and PID publication must not
    # cause another operator to duplicate-launch an unrecorded child.
    queue = load(LOOP / "queue.json")
    persisted = next(x for x in queue["items"] if x["id"] == item["id"])
    persisted["status"] = "launching"
    dump(LOOP / "queue.json", queue)
    with (out / "launch_parent.wrapper.log").open("w") as handle:
        return subprocess.Popen(
            ["/bin/bash", str(wrapper)], stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )


def launch_arena(item: dict) -> None:
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "arena.games.jsonl").exists():
        raise SystemExit(f"{item['id']} games already exist")
    if not RUNTIME.is_dir():
        raise SystemExit(f"arena runtime missing: {RUNTIME}")
    games = str(item["games"])
    sims = str(item["sims"])
    seed = str(item["seed"])
    timeout_s = str(launch_budget(item))
    max_s = timeout_s
    games_out = out / "arena.games.jsonl"
    results_out = out / "arena.results.jsonl"
    if item.get("openings_fen"):
        openings_args = ["--openings-fen", item["openings_fen"]]
    else:
        openings_args = ["--openings", BOOK, "--opening-plies", "16"]
    cmd = [
        "/usr/bin/python3",
        "-m",
        "scripts.arena_standard",
        "--candidate", item["candidate"],
        "--reference", item["reference"],
        "--games", games,
        "--mode", "matched_sims",
        "--sims", sims,
        "--seed", seed,
        *openings_args,
        "--max-plies", "300",
        "--temperature", "0.1",
        "--search-shape", "training",
        "--cand-gumbel", "policy_temp=1.0",
        "--ref-gumbel", "policy_temp=1.0",
        "--compile", "on",
        "--device", "cuda",
        "--max-concurrent-games", "128",
        "--eval-max-batch", "4096",
        "--max-seconds", max_s,
        "--syzygy-max-pieces", "0",
        "--games-out", str(games_out),
        "--out", str(results_out),
    ]
    argv_path = out / "command.json"
    dump(argv_path, {"cwd": str(RUNTIME), "timeout_s": int(timeout_s), "command": cmd})
    wrapper = out / "launch_parent.sh"
    wrapper.write_text(
        f"""#!/bin/bash
set -euo pipefail
cd {RUNTIME}
unset PYTHONOPTIMIZE PYTHONHOME LD_PRELOAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONPATH={RUNTIME}${{PYTHONPATH:+:$PYTHONPATH}}
mapfile -t CMD < <(python3 -c "import json; print('\\n'.join(json.load(open('{argv_path}'))['command']))")
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s {timeout_s}s "${{CMD[@]}}" > {out / 'arena.log'} 2>&1
rc=$?
set -e
python3 - <<PY
import json, time
from pathlib import Path
p = Path({str(out)!r})
started = json.loads((p / "parent_outer_started.json").read_text())
term = {{
    "returncode": {0}+int("$rc"),
    "ended_unix": time.time(),
    "elapsed_seconds": time.time() - float(started["started_unix"]),
}}
(p / "parent_outer_terminal.json").write_text(json.dumps(term) + "\\n")
PY
exit "$rc"
"""
    )
    wrapper.chmod(0o755)
    started = {
        "started_unix": time.time(),
        "id": item["id"],
        "command": cmd,
        "cwd": str(RUNTIME),
        "seed": item["seed"],
        "games": item["games"],
        "sims": item["sims"],
        "free_disk_gib": disk_free_gib(),
        "mem_avail_gib": mem_avail_gib(),
        "gpu_apps_before": gpu_apps().splitlines(),
    }
    dump(out / "parent_outer_started.json", started)
    launch_budget(item)  # Recheck after preparation and GPU probes.
    # launch in new session
    still_busy = gpu_apps()
    if still_busy:
        raise SystemExit(f"refusing launch {item['id']}: GPU already has {still_busy.splitlines()[0][:120]}")
    proc = start_wrapper(item, wrapper, out)
    (out / "parent_outer.pid").write_text(str(proc.pid) + "\n")
    item["status"] = "running"
    item["pid"] = proc.pid
    log("arena_launched", id=item["id"], pid=proc.pid, seed=item["seed"], out=str(out))


def registered_spec(item: dict) -> dict:
    path = Path(item["command_file"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["command_sha256"]:
        raise ValueError("registered command file hash mismatch")
    spec = json.loads(raw)
    if not spec["argv"] or not all(isinstance(x, str) for x in spec["argv"]):
        raise ValueError("registered argv must be a nonempty string list")
    for pin in spec["pins"]:
        source = Path(pin["path"])
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("pin must be a small command or manifest, not a model")
        if hashlib.sha256(source.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError(f"registered input pin mismatch: {source}")
    if Path(spec["completion"]["path"]).exists():
        raise ValueError("registered completion already exists")
    return spec


def launch_registered(item: dict) -> None:
    spec = registered_spec(item)
    budget = launch_budget(item)
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "parent_outer_started.json").exists():
        raise ValueError("registered launch already has an attempt")
    command = {"spec": spec, "max_seconds": budget, "out": str(out)}
    dump(out / "command.json", command)
    dump(out / "parent_outer_started.json", {"started_unix": time.time(), "id": item["id"],
        "command_sha256": item["command_sha256"], "max_seconds": budget})
    wrapper = out / "launch_parent.sh"
    # Only fixed Python runner and shell-quoted filesystem paths enter the wrapper.
    import shlex
    wrapper.write_text("#!/bin/bash\nexec /usr/bin/python3 " + shlex.quote(str(Path(__file__).resolve()))
        + " --run-registered " + shlex.quote(str(out / "command.json")) + "\n")
    if gpu_apps():
        raise ValueError("GPU became busy before registered launch")
    proc = start_wrapper(item, wrapper, out)
    (out / "parent_outer.pid").write_text(str(proc.pid) + "\n")
    item.update(status="running", pid=proc.pid)
    log("registered_launched", id=item["id"], pid=proc.pid)


def terminate_owned_group(proc, grace: float = 30) -> None:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + grace
    # The leader can exit before a TERM-ignoring grandchild. Test the group,
    # not just Popen.wait(), throughout its already-budgeted grace period.
    while time.monotonic() < deadline:
        proc.poll()  # Reap the leader when possible.
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            proc.wait()
            return
        time.sleep(min(.1, max(0, deadline-time.monotonic())))
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_registered(path: Path) -> int:
    command = load(path)
    out = Path(command["out"])
    spec = command["spec"]
    started = time.monotonic()
    env = os.environ.copy()
    env.update(spec.get("env", {}))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        env[key] = "2"
    rc = 1
    try:
        with (out / "arena.log").open("w") as handle:
            proc = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=float(command["max_seconds"]))
            except subprocess.TimeoutExpired:
                terminate_owned_group(proc)
                rc = 124
    finally:
        dump(out / "parent_outer_terminal.json", {"returncode": rc,
            "ended_unix": time.time(), "elapsed_seconds": time.monotonic()-started})
    return rc


def harvest_registered(item: dict, state: dict) -> None:
    if item.get("status") != "running":
        return
    out = Path(item["out"])
    terminal = out / "parent_outer_terminal.json"
    if not terminal.exists():
        return  # Absence never proves its child exited.
    term = load(terminal)
    try:
        completion = load(out / "command.json")["spec"]["completion"]
        result = load(Path(completion["path"]))
        if term["returncode"] != 0 or result[completion["status_key"]] != completion["expected"]:
            raise ValueError("registered command or qualification failed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        item["status"] = "failed"
        log("registered_failed", id=item["id"], reason=str(exc))
        return
    item.update(status="logged", result=result)
    state["completed"].append(item["id"])
    log("registered_completed", id=item["id"], terminal=term)


def parse_match_log(text: str) -> dict | None:
    """Parse scripts/match_vs_uci.py's stdout summary. Trinomial (no pairs)."""
    m = re.search(r"\[match\] (\d+) games in ([\d.]+)s", text)
    w = re.search(r"A wins\s*:\s*(\d+)", text)
    d = re.search(r"draws\s*:\s*(\d+)", text)
    l = re.search(r"A losses\s*:\s*(\d+)", text)
    s = re.search(r"A score\s*:\s*([\d.]+)", text)
    if not (m and w and d and l and s):
        return None
    ci = re.search(r"Score 95% CI:\s*\[([\d.]+),\s*([\d.]+)\]", text)
    elo = re.search(r"Elo \(A - B\) ≈ ([+-]?\d+)", text)
    elo_ci = re.search(r"Elo 95% CI\s*:\s*\[([+-]?\d+),\s*([+-]?\d+)\]", text)
    return {
        "games": int(m.group(1)),
        "duration_s": float(m.group(2)),
        "wins": int(w.group(1)),
        "draws": int(d.group(1)),
        "losses": int(l.group(1)),
        "score": float(s.group(1)),
        "score_ci95": [float(ci.group(1)), float(ci.group(2))] if ci else None,
        "elo_crude": int(elo.group(1)) if elo else None,
        "elo_ci95_crude": [int(elo_ci.group(1)), int(elo_ci.group(2))] if elo_ci else None,
        "truncated": False,
    }


def validate_match_bank(item: dict, result: dict) -> None:
    with (Path(item["out"]) / "match.games.jsonl").open() as handle:
        games = [row for line in handle if (row := json.loads(line)).get("kind") == "game"]
    expected = int(item["games"])
    if len(games) != expected or {g["game_index"] for g in games} != set(range(expected)):
        raise ValueError("incomplete or duplicated UCI game bank")
    wins = sum(g["score_a"] == 1 for g in games)
    draws = sum(g["score_a"] == .5 for g in games)
    losses = sum(g["score_a"] == 0 for g in games)
    if (wins, draws, losses) != (result["wins"], result["draws"], result["losses"]):
        raise ValueError("UCI game bank disagrees with summary")


def harvest_match(item: dict, state: dict) -> None:
    if item.get("status") != "running":
        return
    out = Path(item["out"])
    terminal = out / "parent_outer_terminal.json"
    if not terminal.exists():
        pid_file = out / "parent_outer.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if not pid_alive(pid):
                item["status"] = "needs_recovery"
                log("match_wrapper_missing_terminal", id=item["id"], pid=pid)
        return
    term = json.loads(terminal.read_text())
    result = parse_match_log((out / "arena.log").read_text() if (out / "arena.log").exists() else "")
    if (term.get("returncode") != 0 or result is None
            or result.get("games") != item["games"]
            or sum(result.get(k, 0) for k in ("wins", "draws", "losses")) != item["games"]):
        item["status"] = "failed"
        log("match_failed", id=item["id"], terminal=term, result=result)
        return
    try:
        validate_match_bank(item, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        item["status"] = "failed"
        log("match_invalid_completion", id=item["id"], reason=str(exc))
        return
    item["status"] = "logged"
    item["result"] = result
    state["completed"].append(item["id"])
    log("match_logged", id=item["id"], **result)
    ci = result.get("score_ci95") or [None, None]
    append_results(
        f"- {item['id']}: {item.get('label_a')} vs {item.get('label_b')} "
        f"{result.get('games')} games **W{result.get('wins')}/D{result.get('draws')}/L{result.get('losses')}** "
        f"score {result.get('score')} [{ci[0]}, {ci[1]}] crude Elo ~{result.get('elo_crude')} "
        f"(trinomial; common-opponent delta is the signal)"
    )


def launch_match(item: dict) -> None:
    """Net-vs-UCI-engine match (scripts/match_vs_uci.py). Same wrapper/terminal protocol."""
    if not item.get("qualified_uci_profile"):
        raise ValueError("UCI profile has not been qualified")
    launch_budget(item)
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    games_out = out / "match.games.jsonl"
    if games_out.exists():
        raise SystemExit(f"{item['id']} games already exist")
    if not RUNTIME.is_dir():
        raise SystemExit(f"arena runtime missing: {RUNTIME}")
    timeout_s = str(launch_budget(item))
    cmd = [
        "/usr/bin/python3",
        "-m",
        "scripts.match_vs_uci",
        "--engine-a", item["engine_a"],
        "--engine-b", item["engine_b"],
        "--label-a", item.get("label_a", "A"),
        "--label-b", item.get("label_b", "B"),
        "--games", str(item["games"]),
        "--openings", item["openings"],
        "--games-out", str(games_out),
        "--move-log-out", str(out / "match.moves.csv"),
    ]
    for opt in item.get("option_a", []):
        cmd += ["--option-a", opt]
    for opt in item.get("option_b", []):
        cmd += ["--option-b", opt]
    if item.get("time_ms_a") is not None:
        cmd += ["--time-ms-a", str(item["time_ms_a"])]
    if item.get("time_ms_b") is not None:
        cmd += ["--time-ms-b", str(item["time_ms_b"])]
    if item.get("nodes_a") is not None:
        cmd += ["--nodes-a", str(item["nodes_a"])]
    if item.get("nodes_b") is not None:
        cmd += ["--nodes-b", str(item["nodes_b"])]
        cmd += ["--enforce-nodes-b"]
    argv_path = out / "command.json"
    dump(argv_path, {"cwd": str(RUNTIME), "timeout_s": int(timeout_s), "command": cmd})
    wrapper = out / "launch_parent.sh"
    wrapper.write_text(
        f"""#!/bin/bash
set -euo pipefail
cd {RUNTIME}
unset PYTHONOPTIMIZE PYTHONHOME LD_PRELOAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONPATH={RUNTIME}${{PYTHONPATH:+:$PYTHONPATH}}
mapfile -t CMD < <(python3 -c "import json; print('\\n'.join(json.load(open('{argv_path}'))['command']))")
set +e
/usr/bin/timeout --signal=TERM --kill-after=30s {timeout_s}s "${{CMD[@]}}" > {out / 'arena.log'} 2>&1
rc=$?
set -e
python3 - <<PY
import json, time
from pathlib import Path
p = Path({str(out)!r})
started = json.loads((p / "parent_outer_started.json").read_text())
term = {{
    "returncode": {0}+int("$rc"),
    "ended_unix": time.time(),
    "elapsed_seconds": time.time() - float(started["started_unix"]),
}}
(p / "parent_outer_terminal.json").write_text(json.dumps(term) + "\\n")
PY
exit "$rc"
"""
    )
    wrapper.chmod(0o755)
    started = {
        "started_unix": time.time(),
        "id": item["id"],
        "command": cmd,
        "cwd": str(RUNTIME),
        "games": item["games"],
        "free_disk_gib": disk_free_gib(),
        "mem_avail_gib": mem_avail_gib(),
        "gpu_apps_before": gpu_apps().splitlines(),
    }
    dump(out / "parent_outer_started.json", started)
    still_busy = gpu_apps()
    if still_busy:
        raise SystemExit(f"refusing launch {item['id']}: GPU already has {still_busy.splitlines()[0][:120]}")
    proc = start_wrapper(item, wrapper, out)
    (out / "parent_outer.pid").write_text(str(proc.pid) + "\n")
    item["status"] = "running"
    item["pid"] = proc.pid
    log("match_launched", id=item["id"], pid=proc.pid, out=str(out))


@contextmanager
def acquire_operator_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU_BUSY operator_lock")
            raise SystemExit(0) from None
        yield


def main() -> int:
    state = load(LOOP / "STATE.json")
    queue = load(LOOP / "queue.json")
    for item in queue["items"]:
        if item.get("kind") == "arena":
            harvest_arena(item, state)
        elif item.get("kind") == "match_uci":
            harvest_match(item, state)
        elif item.get("kind") == "registered_command":
            harvest_registered(item, state)

    now = time.time()
    if now >= float(state["deadline_unix"]):
        log("deadline_reached")
        state["current_gpu"] = "stopped_deadline"
        dump(LOOP / "STATE.json", state)
        dump(LOOP / "queue.json", queue)
        print("DEADLINE")
        return 0

    dump(LOOP / "queue.json", queue)
    busy = gpu_busy()
    if busy:
        dump(LOOP / "STATE.json", state)
        dump(LOOP / "queue.json", queue)
        print(f"GPU_BUSY {busy}")
        return 0

    # GPU idle: launch first queued GPU work
    for item in queue["items"]:
        if item["kind"] == "harvest_ceres" and item["status"] not in {"logged", "failed"}:
            dump(LOOP / "STATE.json", state)
            dump(LOOP / "queue.json", queue)
            print("WAIT_CERES")
            return 0
        if item["kind"] in {"arena", "match_uci", "registered_command"} and item["status"] == "queued":
            if disk_free_gib() < 150:
                log("disk_below_floor", free_gib=disk_free_gib())
                dump(LOOP / "STATE.json", state)
                dump(LOOP / "queue.json", queue)
                print("DISK_FLOOR")
                return 2
            if mem_avail_gib() < 24:
                log("ram_low", mem_avail_gib=mem_avail_gib())
                dump(LOOP / "STATE.json", state)
                dump(LOOP / "queue.json", queue)
                print("RAM_LOW")
                return 2
            try:
                launch_budget(item)
            except ValueError as exc:
                log("budget_refusal", id=item["id"], reason=str(exc))
                dump(LOOP / "queue.json", queue)
                print("DEADLINE_INSUFFICIENT")
                return 0
            if item["kind"] == "match_uci":
                launch_match(item)
            elif item["kind"] == "registered_command":
                launch_registered(item)
            else:
                launch_arena(item)
            state["current_gpu"] = item["id"]
            dump(LOOP / "STATE.json", state)
            dump(LOOP / "queue.json", queue)
            print(f"LAUNCHED {item['id']} pid={item['pid']}")
            return 0

    dump(LOOP / "STATE.json", state)
    dump(LOOP / "queue.json", queue)
    print("QUEUE_IDLE")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registered", type=Path)
    parser.add_argument("--loop-dir", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.run_registered:
        raise SystemExit(run_registered(args.run_registered))
    if args.loop_dir is None or args.runtime is None:
        parser.error("--loop-dir and --runtime required for queue operation")
    LOOP, RUNTIME = args.loop_dir.resolve(), args.runtime.resolve()
    LOCK = LOOP / "gpu.lock"
    with acquire_operator_lock():
        raise SystemExit(main())
