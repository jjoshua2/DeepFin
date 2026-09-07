#!/usr/bin/env python3
"""Plan or execute one separately registered fresh-seed BT4 confirmation.

Usage: python scripts/bt4_confirmation.py --manifest confirmation.json [--execute]

The JSON manifest supplies schema=1, repository_root, state, training_seed (>0),
arena_seed, sims (25/100/400), gpu_budget_seconds, cpu_stage_caps_seconds
(schedule=1800/readout=120), and stage_caps_seconds with
reference_train/candidate_train/arena. Both candidate and reference contain role,
corpus, run, derive_sha256 and mix_sha256 (null for S0). Roles are S0, E0, E0T05, G20T1,
G20T05 or C20T05. runtime_manifest and preregistration each contain path/sha256.
openings and schedule_verifier are absolute paths to the qualified artifacts;
launcher_sha256 pins this file. No recipe, seed, budget or output path is defaulted.

Outputs must be new. There is deliberately no training adoption or restart mode.
A failed invocation preserves its state and partial runs for explicit recovery.
All stage caps include 30 seconds of TERM-to-KILL grace. Each child has an
independently surviving GNU timeout; abrupt coordinator loss leaves an unfinished
charge receipt requiring reconciliation after its owned stage has stopped.
The frozen runtime manifest supplies Python/native/build identities; the immutable
preregistration supplies the scientific decision, which this launcher never makes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

QUALIFIED_HEAD = "7ec261509fb7345cf1ca0ad73809193fc2749bb1"
OPENINGS_SHA = "e0d13b2ea70c0ac278570a0e463c3c1c3030a18256522bcba864db23cdc07c98"
VERIFIER_SHA = "b212d4b220387dcd121b155d1f7d49ba7c1ea2e94c4e6c1d9a31ff6035bea9f4"
READER_SHA = "49cd6ceab88e1b7d225b18a166fbf7a004309d58bc1f0bf9ae2afe859606f16c"
SOURCE_SHA = "391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef"
ROWS = 18_910_484
SHARDS = 2309
RESERVE = 150 * 1024**3
OUTPUT_ALLOWANCE = 10 * 1024**3
ROLES = ("S0", "E0", "E0T05", "G20T1", "G20T05", "C20T05")
QUALIFIED_SEARCH_SHA = "353d95c0f94cad1c1be1f9b23f6449d8cad50c6a41523c2b9ab5415f5cd6c12e"
CPU_STAGE_CAPS = {"schedule": 1800, "readout": 120}
KILL_GRACE_SECONDS = 30
QUALIFIED_MIX_SHA = {
    "E0T05": "a85ba1403c2477018bdc59b622311094027ca17cf045dc78090f6c0ee9f5463d",
    "E0": "7bbb7df4e42cbb61fc937fd210bb7f3c6ed66a04644d02fd85b8a32f64c55222",
    "G20T1": "d0848a2357f8d1f8cd22850c6ba71cfe58ae57bf2a99ac63b1f91de4f1544fcb",
    "G20T05": "ff50d6b3f4c8ab38233de723c68924980d745335a374a71d6cfc3eac71c248ab",
    "C20T05": "5bf8502a12af0b9ce938a39ddfd9d95df4bfb2b1a0f80292d80d04109cce7100",
}
STAGES = ("reference_train", "candidate_train", "arena")
READER = Path(__file__).with_name("bt4_joint_readout.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected):
    require(sha(path) == expected, f"identity changed: {path}")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def absolute(value):
    path = Path(value)
    require(path.is_absolute(), f"absolute path required: {value}")
    return path.resolve()


def absent(path):
    require(not path.exists() and not path.is_symlink(), f"existing output requires explicit recovery: {path}")


def validate_manifest(m):
    keys = {"schema", "repository_root", "state", "training_seed", "arena_seed", "sims",
            "gpu_budget_seconds", "stage_caps_seconds", "cpu_stage_caps_seconds", "candidate", "reference",
            "runtime_manifest", "preregistration", "openings", "schedule_verifier", "launcher_sha256"}
    require(set(m) == keys and m["schema"] == 1, "unsupported or incomplete confirmation manifest")
    for key in ("training_seed", "arena_seed"):
        require(type(m[key]) is int and 0 <= m[key] < 2**32, f"invalid {key}")
    require(m["training_seed"] > 0, "fresh confirmation cannot adopt or retrain seed zero")
    require(type(m["sims"]) is int and m["sims"] in (25, 100, 400), "unsupported simulation budget")
    caps = m["stage_caps_seconds"]
    require(set(caps) == set(STAGES), "supply all three GPU stage caps")
    for value in (m["gpu_budget_seconds"], *caps.values()):
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid GPU budget")
    require(all(value > KILL_GRACE_SECONDS for value in caps.values()), "GPU stage caps must include termination grace")
    require(m["cpu_stage_caps_seconds"] == CPU_STAGE_CAPS, "register schedule=1800/readout=120 CPU stage caps")
    require(sum(caps.values()) <= m["gpu_budget_seconds"], "GPU budget cannot cover registered stage caps")
    outputs = [absolute(m["state"])]
    inputs = [absolute(m[k]) for k in ("openings", "schedule_verifier")]
    inputs.append(absolute(m["repository_root"]) / ".dev/worktree/wise-cloud")
    for name in ("runtime_manifest", "preregistration"):
        require(set(m[name]) == {"path", "sha256"}, f"invalid {name} identity")
        inputs.append(absolute(m[name]["path"]))
    for side in ("reference", "candidate"):
        arm = m[side]
        require(set(arm) == {"role", "corpus", "run", "derive_sha256", "mix_sha256"}, f"invalid {side}")
        require(arm["role"] in ROLES, f"unsupported {side} role")
        outputs.append(absolute(arm["run"]))
        inputs.append(absolute(arm["corpus"]))
    require(m["candidate"]["role"] != m["reference"]["role"], "confirmation requires distinct recipe roles")
    require(absolute(m["candidate"]["corpus"]) != absolute(m["reference"]["corpus"]), "recipes share a corpus")
    for i, out in enumerate(outputs):
        for other in outputs[i + 1:] + inputs:
            require(out != other and out not in other.parents and other not in out.parents,
                    f"overlapping input/output paths: {out}, {other}")
    absolute(m["repository_root"])


def check_qualified_mix(arm, corpus):
    # These are fixed named recipes, including the legacy raw E0 whose summary
    # omits temperature. Shape-only checks confuse it with the existing T=.5 tie arm.
    expected = QUALIFIED_MIX_SHA[arm["role"]]
    require(arm["mix_sha256"] == expected, "role does not identify its qualified published recipe")
    pin(corpus / "bt4_policy_mix_summary.json", expected)
    require(read(corpus / "derive_targets_summary.json")["policy_target_postprocess"]
            == read(corpus / "bt4_policy_mix_summary.json"), "corpus lineage differs")


def commands(m):
    """Both trainers use the identical explicit fresh seed and qualified recipe."""
    runtime = read(m["runtime_manifest"]["path"])["runtime"]
    python = runtime["executable"]
    state = absolute(m["state"])
    result = {}
    for side in ("reference", "candidate"):
        arm = m[side]
        result[side + "_train"] = [python, "scripts/lc0_control_train.py", "--config",
            "configs/lc0_positive_control.yaml", "--shards", str(absolute(arm["corpus"])),
            "--out-dir", str(absolute(arm["run"])), "--steps", "0", "--batch-size", "512",
            "--sampling-mode", "game_epoch", "--epoch-plan-workers", "16", "--epoch-load-workers", "16",
            "--seed", str(m["training_seed"]), "--device", "cuda", "--train-window-steps", "88",
            "--allow-invalid-control"]
    result["schedule"] = [python, str(absolute(m["schedule_verifier"])), "--seed", str(m["training_seed"])]
    for side in ("reference", "candidate"):
        result["schedule"] += ["--run", f"{side}={absolute(m[side]['run'])}"]
    result["schedule"] += ["--output", str(state / "matched_schedule.json")]
    checkpoints = {side: str(absolute(m[side]["run"]) / "checkpoint.pt") for side in ("reference", "candidate")}
    result["arena"] = [python, "scripts/arena_standard.py", "--candidate", checkpoints["candidate"],
        "--reference", checkpoints["reference"], "--games", "1000", "--mode", "matched_sims",
        "--search-shape", "training", "--cand-gumbel", "policy_temp=1.0", "--ref-gumbel", "policy_temp=1.0",
        "--sims", str(m["sims"]), "--seed", str(m["arena_seed"]), "--openings-fen", str(absolute(m["openings"])),
        "--max-plies", "300", "--temperature", "0.1", "--max-concurrent-games", "128",
        "--eval-max-batch", "4096", "--compile", "on", "--no-rolling", "--label", "bt4_confirmation",
        "--games-out", str(state / "arena.games.jsonl"), "--out", str(state / "arena.results.jsonl")]
    result["readout"] = [python, str(READER), "--profile", "confirmation", "--prior-temperature", "1.0",
        "--candidate", checkpoints["candidate"], "--reference", checkpoints["reference"],
        "--candidate-role", m["candidate"]["role"], "--reference-role", m["reference"]["role"],
        "--confirmation-sims", str(m["sims"]), "--confirmation-openings", str(absolute(m["openings"])),
        "--confirmation-sha256", OPENINGS_SHA, "--seed", str(m["arena_seed"]),
        "--cell", f"{m['candidate']['role']}:{m['sims']}={state / 'arena.games.jsonl'}"]
    return result


def training_complete(m, side):
    run = absolute(m[side]["run"])
    summary = read(run / "summary.json")
    sampling = summary["sampling"]
    batches = sampling["batches_planned"]  # A fresh seed needs its own realized planner proof.
    require(type(batches) is int and batches > 0, "invalid planned batches")
    required = {"mode": "game_epoch", "complete": True, "seed": m["training_seed"], "batch_size": 512,
                "rows_planned": ROWS, "rows_realized": ROWS, "batches_realized": batches,
                "same_game_repeats_max": 0, "decoded_rows_resident": 0, "shards": SHARDS}
    require(all(sampling.get(k) == v for k, v in required.items()), f"{side}: incomplete fresh epoch")
    require(sampling.get("plan_sha256") and sampling["plan_sha256"] == sampling.get("realized_sha256"),
            f"{side}: realized schedule differs")
    require(summary["seed"] == m["training_seed"] and summary["batch_size"] == 512
            and summary["steps_realized"] == summary["compute_loss_calls"] == batches
            and summary["corpus"]["shard_dirs"] == [str(absolute(m[side]["corpus"]))],
            f"{side}: training settings or corpus differ")
    windows = summary["train_window_metrics"]
    require(summary["train_window_steps"] == 88 and summary["train_windows"] == len(windows) == (batches + 87) // 88
            and all(w["grad_nonfinite_skip_rate"] == 0 and w["transient_cuda_retry_batches"] == 0
                    and math.isfinite(w["loss"]) and math.isfinite(w["grad_norm_mean"]) for w in windows),
            f"{side}: incorrect, skipped/retried or nonfinite training windows")
    checkpoint = sha(run / "checkpoint.pt")
    require(any(item.get("role") == "last" and item.get("path") == str(run / "checkpoint.pt")
                and item.get("sha256") == checkpoint for item in summary["checkpoints"]),
            f"{side}: summary checkpoint differs")
    return {"checkpoint_sha256": checkpoint, "summary_sha256": sha(run / "summary.json"),
            "seed": m["training_seed"], "batches": batches, "rows": sampling["rows_realized"],
            "physical_plan_sha256": sampling["plan_sha256"]}


def validate_schedule(m, completions):
    report = read(absolute(m["state"]) / "matched_schedule.json")
    require(report["verifier_sha256"] == VERIFIER_SHA and report["seed"] == m["training_seed"]
            and report["batch_size"] == 512 and report["runtime"]["numpy"] == "1.26.2", "schedule verifier identity differs")
    require(set(report["arms"]) == {"reference", "candidate"}, "missing matched training proof")
    require(completions["candidate"]["batches"] == completions["reference"]["batches"]
            == report["source_plan"]["batches_planned"]
            and completions["candidate"]["rows"] == completions["reference"]["rows"] == ROWS,
            "fresh arms have different realized epoch counts")
    for side, arm in report["arms"].items():
        require(arm["training_completion_verified"] is True and arm["staging"] == "verified actual"
                and arm["metadata_matches_source"] is True and arm["corpus"] == str(absolute(m[side]["corpus"]))
                and arm["summary_sha256"] == completions[side]["summary_sha256"]
                and arm["physical_plan_sha256"] == completions[side]["physical_plan_sha256"]
                and arm["canonical_plan_sha256"] == report["source_plan"]["plan_sha256"],
                f"{side}: completed matched schedule not established")


def qualified_search(m):
    path = absolute(m["repository_root"]) / "scratchpad/bt4_joint20/sf_close_run02/completed_s100_readout.json"
    pin(path, QUALIFIED_SEARCH_SHA)
    settings = read(path)["cells"]["C20T05:100"]["settings"]
    require(settings["search_candidate"] == settings["search_reference"], "qualified search sides differ")
    return settings["search_candidate"]


def validate_header(m, header):
    settings = header["settings"]
    require(header.get("kind") == "header" and header.get("driver") == "arena_standard" and header.get("version") == 1,
            "unsupported arena header")
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    require(header["fingerprint"] == fingerprint, "arena fingerprint differs")
    expected = {"mode": "matched_sims", "games": 1000, "seed": m["arena_seed"], "openings_kind": "fen",
                "opening_plies": None, "openings": str(absolute(m["openings"])), "max_plies": 300,
                "temperature": 0.1, "gumbel_add_noise": True, "sims_candidate": m["sims"], "sims_reference": m["sims"]}
    for side in ("candidate", "reference"):
        expected[side] = str(absolute(m[side]["run"]) / "checkpoint.pt")
    require(all(settings.get(k) == v for k, v in expected.items()) and "opening_plies" in settings,
            "off-protocol confirmation arena")
    search = settings["search_candidate"]
    require(search == settings["search_reference"] == qualified_search(m),
            "confirmation requires the full qualified prior-1 training search")


class Runner:
    def __init__(self, manifest):
        validate_manifest(manifest)
        self.m = manifest
        self.state = absolute(manifest["state"])
        self.repo = absolute(manifest["repository_root"])
        self.runtime = self.repo / ".dev/worktree/wise-cloud"
        self.stop = threading.Event()
        self.deadline = None
        self.lock_fd = -1
        self.used = 0.0
        self.child_reaped = True

    def guard(self, extra_disk=0):
        require(not self.stop.is_set() and not (self.state / "STOP").exists(), "stop requested; preserving partial outputs")
        require(self.deadline is None or time.monotonic() < self.deadline, "GPU stage budget exhausted")
        for path in (self.repo, self.state.parent, *(absolute(self.m[s]["run"]).parent for s in ("reference", "candidate"))):
            while not path.exists():
                path = path.parent
            require(shutil.disk_usage(path).free >= RESERVE + extra_disk, f"disk below 150 GiB reserve: {path}")

    def check_pins(self):
        m = self.m
        pin(Path(__file__), m["launcher_sha256"])
        pin(READER, READER_SHA)
        pin(m["openings"], OPENINGS_SHA)
        pin(m["schedule_verifier"], VERIFIER_SHA)
        qualified_search(m)
        for name in ("runtime_manifest", "preregistration"):
            pin(m[name]["path"], m[name]["sha256"])
        frozen = read(m["runtime_manifest"]["path"])
        require(frozen["identities"]["heads"].get(str(self.runtime)) == QUALIFIED_HEAD, "unqualified frozen runtime")
        require(subprocess.check_output(["git", "-C", str(self.runtime), "rev-parse", "HEAD"], text=True).strip() == QUALIFIED_HEAD,
                "frozen runtime checkout moved")
        require(not subprocess.check_output(["git", "-C", str(self.runtime), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
                "tracked changes in frozen runtime")
        runtime = frozen["runtime"]
        modules = {"chess_anti_engine.encoding._features_ext", "chess_anti_engine.encoding._lc0_ext",
                   "chess_anti_engine.mcts._mcts_tree", "chess_anti_engine.nnue._nnue_ext"}
        require(set(runtime["native_extensions"]) == modules
                and set(runtime["native_extension_sha256"]) == set(runtime["native_extensions"].values()),
                "incomplete native runtime pins")
        require(runtime["python"].startswith("3.10.12") and runtime["torch"] == "2.11.0+cu128" and runtime["cuda"] == "12.8",
                "unqualified runtime versions")
        for path, digest in runtime["native_extension_sha256"].items():
            pin(path, digest)
        for side in ("reference", "candidate"):
            arm = m[side]
            corpus = absolute(arm["corpus"])
            pin(corpus / "derive_targets_summary.json", arm["derive_sha256"])
            require(not corpus.name.endswith(".writing"), "unpublished corpus")
            if arm["role"] == "S0":
                require(arm["derive_sha256"] == SOURCE_SHA and arm["mix_sha256"] is None, "S0 must be the unmodified SF corpus")
            else:
                check_qualified_mix(arm, corpus)

    def check_runtime(self):
        expected = read(self.m["runtime_manifest"]["path"])["runtime"]
        probe = ("import importlib,json,sys,torch,numpy; "
                 f"modules={list(expected['native_extensions'])!r}; "
                 "print(json.dumps(dict(python=sys.version,executable=sys.executable,torch=torch.__version__,"
                 "cuda=torch.version.cuda,numpy=numpy.__version__,"
                 "native_extensions={m:importlib.import_module(m).__file__ for m in modules})))")
        actual = json.loads(subprocess.check_output([expected["executable"], "-c", probe], cwd=self.runtime,
                                                    env=self.environment(False), text=True, timeout=60))
        require(actual["numpy"] == "1.26.2", "matched NumPy runtime differs")
        require(actual == {k: v for k, v in expected.items() if k != "native_extension_sha256"}, "actual frozen runtime differs")
        return actual

    def environment(self, gpu):
        return {**os.environ, "PYTHONPATH": str(self.runtime), "CUDA_VISIBLE_DEVICES": "0" if gpu else "",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "BLOSC_NTHREADS": "2",
                "CHESS_ANTI_ENGINE_LIVE_CONFIG": str(self.runtime / "configs/pbt2_small.yaml")}

    @contextmanager
    def gpu(self, stage):
        self.guard()
        cap = self.m["stage_caps_seconds"][stage]
        require(self.used + cap <= self.m["gpu_budget_seconds"], "insufficient remaining GPU budget")
        with (self.repo / "scratchpad/gpu0_experiment.lock").open("a") as lease:
            while True:
                self.guard()
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.stop.wait(2)
                    continue
                if not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    break
                fcntl.flock(lease, fcntl.LOCK_UN)
                self.stop.wait(2)
            started = time.monotonic()
            self.deadline = started + min(cap, self.m["gpu_budget_seconds"] - self.used)
            charge = {"stage": stage, "complete": False, "started_unix": time.time(), "owner_pid": os.getpid()}
            path = self.state / f"{stage}.gpu-charge.json"
            write(path, charge)
            outcome = "failed"
            try:
                self.check_pins()  # Opening/runtime bytes may have changed during a long lease wait.
                self.check_runtime()
                self.guard()
                yield lease.fileno()
                self.guard()
                outcome = "succeeded"
            finally:
                seconds = time.monotonic() - started
                self.used += seconds
                self.deadline = None
                write(path, {**charge, "complete": self.child_reaped, "outcome": outcome, "seconds": seconds})

    def run(self, stage, command, gpu_fd=None, output=None):
        self.guard()
        require(gpu_fd is None or self.deadline is not None, "GPU child requires an active stage deadline")
        seconds = (self.deadline - time.monotonic()) if self.deadline is not None else self.m["cpu_stage_caps_seconds"][stage]
        require(seconds > KILL_GRACE_SECONDS, "insufficient stage time including termination grace")
        supervisor = ["/usr/bin/timeout", "--signal=TERM", f"--kill-after={KILL_GRACE_SECONDS}s",
                      f"{seconds - KILL_GRACE_SECONDS}s", *command]
        with (self.state / f"{stage}.log").open("x") as log, (output or self.state / f"{stage}.stdout").open("x") as stdout:
            child = subprocess.Popen(supervisor, cwd=self.runtime, env=self.environment(gpu_fd is not None),
                                     stdout=stdout, stderr=log, start_new_session=True,
                                     pass_fds=tuple(fd for fd in (self.lock_fd, gpu_fd) if fd is not None and fd >= 0))
            self.child_reaped = False
            try:
                write(self.state / f"{stage}.process.json", {"pid": child.pid, "command": command, "supervisor_command": supervisor,
                      "cap_seconds_at_launch": seconds, "cwd": str(self.runtime),
                      "live_config": self.environment(False)["CHESS_ANTI_ENGINE_LIVE_CONFIG"]})
                while child.poll() is None:
                    self.guard()
                    if stage == "arena":
                        self.check_arena_header(required=False)
                    self.stop.wait(2)
                self.guard()
                require(child.returncode == 0, f"{stage} exited {child.returncode}; inspect {stage}.log/.stdout")
            finally:
                # The newly created session contains only this stage and its descendants.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.child_reaped = True

    def check_arena_header(self, *, required):
        bank = self.state / "arena.games.jsonl"
        if not bank.exists():
            require(not required, "arena bank missing")
            return
        with bank.open() as stream:
            first = stream.readline()
        if not first.endswith("\n"):
            require(not required, "incomplete arena header")
            return
        validate_header(self.m, json.loads(first))

    def execute(self):
        self.guard(OUTPUT_ALLOWANCE)
        self.check_pins()
        absent(self.state)
        for side in ("reference", "candidate"):
            absent(absolute(self.m[side]["run"]))
        self.state.mkdir(parents=True, exist_ok=False)
        with (self.state / "driver.lock").open("x") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd = lock.fileno()
            write(self.state / "manifest.json", self.m)
            (self.state / "preregistration.md").write_bytes(Path(self.m["preregistration"]["path"]).read_bytes())
            plan = commands(self.m)
            pin(self.state / "preregistration.md", self.m["preregistration"]["sha256"])
            try:
                write(self.state / "launch.json", {"pid": os.getpid(), "commands": plan, "runtime": self.check_runtime(),
                      "cpu_stage_caps_seconds": self.m["cpu_stage_caps_seconds"], "termination_grace_seconds": KILL_GRACE_SECONDS,
                      "qualified_search_sha256": QUALIFIED_SEARCH_SHA})
                completions = {}
                for side in ("reference", "candidate"):
                    stage = side + "_train"
                    with self.gpu(stage) as lease_fd:
                        absent(absolute(self.m[side]["run"]))
                        self.run(stage, plan[stage], lease_fd)
                        completions[side] = training_complete(self.m, side)
                        write(self.state / f"{stage}.complete.json", completions[side])
                self.check_pins()
                self.run("schedule", plan["schedule"])
                validate_schedule(self.m, completions)
                with self.gpu("arena") as lease_fd:
                    for side in ("reference", "candidate"):
                        require(training_complete(self.m, side) == completions[side], "trained checkpoint changed before arena")
                    self.run("arena", plan["arena"], lease_fd)
                    self.check_arena_header(required=True)
                    for side in ("reference", "candidate"):
                        require(training_complete(self.m, side) == completions[side], "trained checkpoint changed during arena")
                self.check_pins()
                self.run("readout", plan["readout"], output=self.state / "readout.json")
                readout = read(self.state / "readout.json")
                require(readout["match_complete"] is True, "confirmation match incomplete")
                write(self.state / "complete.json", {"training": completions, "games_sha256": sha(self.state / "arena.games.jsonl"),
                      "schedule_sha256": sha(self.state / "matched_schedule.json"), "readout_sha256": sha(self.state / "readout.json"),
                      "gpu_seconds": self.used, "promotion": "NONE; apply the separately registered rule"})
            except BaseException as error:
                write(self.state / "failed.json", {"error": str(error), "gpu_seconds": self.used, "partial_outputs_preserved": True})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="execute the supplied preregistered confirmation; otherwise print commands only")
    args = parser.parse_args()
    try:
        m = read(args.manifest)
        runner = Runner(m)
        if not args.execute:
            print(json.dumps({"execute": False, "manifest": m, "commands": commands(m)}, indent=2))
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda _signum, _frame: runner.stop.set())
        runner.execute()
    except (ValueError, TypeError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"confirmation failed: {error}\n")


if __name__ == "__main__":
    main()
