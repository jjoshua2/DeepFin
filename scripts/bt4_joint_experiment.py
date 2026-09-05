#!/usr/bin/env python3
"""Run the registered global BT4 screen, preserving the frozen training runtime.

This is an opt-in experiment driver, not a test or a live deployment command.
Select --phase baseline to train S0, then --phase treatments for global arms.
Without --execute this prints the stage plan only. STOP in the state directory requests
termination of owned jobs. Partial corpora/training are preserved and need explicit
recovery; arenas resume through arena_standard's own fingerprint checks.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


LIVE = Path("/home/josh/projects/chess")
TRAIN = LIVE / ".dev/worktree/wise-cloud"
MIX = Path("/tmp/deepfin-bt4-toolchain")
MIX_PYTHON = str(MIX / ".venv/bin/python")
PYTHON = "/usr/bin/python3"
OPS = LIVE / "scratchpad/bt4_joint20"
ROOT_STATE = OPS / "global_run01"
STATE = ROOT_STATE / "baseline"
PREREGISTRATION = ROOT_STATE / "preregistration.md"
PHASE = "baseline"
SOURCE = LIVE / "data/nnue_derived/armB/qtemp_0.0005_hist_20m"
SIDECAR = LIVE / "data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m"
E0 = LIVE / "runs/armB/qtemp_0.0005_hist_20m_bt4_toptie_a100_epoch_v3"
OPENINGS = LIVE / (
    "data/opening_books/"
    "8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip"
)
ROWS, SHARDS, BATCHES = 18_910_484, 2309, 36_935
ARMS = [("S0", 0.0, 1.0), ("G20T1", 0.2, 1.0), ("G20T05", 0.2, 0.5)]
BUDGETS = (25, 100, 400)
GPU_BUDGET_SECONDS = 30 * 3600
# Conservative admission estimates; actual process time is charged on exit.
STAGE_ALLOWANCE = {"train": 4 * 3600, 25: 1800, 100: 3600, 400: 4 * 3600}
SOURCE_SHA = "391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef"
PINS = {
    TRAIN
    / "scripts/lc0_control_train.py": "52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337",
    TRAIN
    / "configs/lc0_positive_control.yaml": "413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2",
    TRAIN
    / "chess_anti_engine/replay/game_epoch.py": "621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3",
    SOURCE / "derive_targets_summary.json": SOURCE_SHA,
    E0
    / "checkpoint.pt": "c7d0bb38f952150db004b29699e0509437bcec7928d79cd44f69125ffa5fa817",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def names(arm):
    letter, _, _ = arm
    base = f"qtemp_0.0005_hist_20m_bt4_global_{letter}"
    return (
        SOURCE if letter == "S0" else SOURCE.parent / base,
        LIVE / "runs/armB" / (base + "_epoch_v1"),
        ROOT_STATE / f"audit_{letter}.json",
    )


def training_complete(run):
    summary = read(run / "summary.json")
    sampling = summary["sampling"]
    required = {
        "mode": "game_epoch",
        "complete": True,
        "rows_planned": ROWS,
        "rows_realized": ROWS,
        "batches_planned": BATCHES,
        "batches_realized": BATCHES,
        "same_game_repeats_max": 0,
        "decoded_rows_resident": 0,
        "plan_workers": 16,
        "load_workers": 16,
    }
    require(
        all(sampling.get(k) == v for k, v in required.items()),
        f"{run}: exact-epoch completion gate failed",
    )
    require(
        sampling.get("plan_sha256")
        and sampling["plan_sha256"] == sampling.get("realized_sha256"),
        f"{run}: mismatched schedule hashes",
    )
    require(
        summary.get("steps_realized") == BATCHES
        and summary.get("seed") == 0
        and summary.get("batch_size") == 512,
        f"{run}: training settings differ",
    )
    require((run / "checkpoint.pt").is_file(), f"{run}: checkpoint missing")
    return {
        "summary_sha256": sha(run / "summary.json"),
        "checkpoint_sha256": sha(run / "checkpoint.pt"),
    }


class Driver:
    def __init__(self, lock_fd):
        self.lock_fd = lock_fd
        self.stop = threading.Event()
        self.manifest = {}
        self.gpu_deadline = None

    def guard(self):
        require(
            self.gpu_deadline is None or time.monotonic() < self.gpu_deadline,
            "GPU screen budget exhausted; preserving partial outputs",
        )
        require(
            not self.stop.is_set() and not (STATE / "STOP").exists(),
            "stop requested; preserving partial outputs",
        )
        require(
            shutil.disk_usage(LIVE).free >= 150 * 1024**3,
            "free space below 150 GiB reserve; preserving partial outputs",
        )

    def status(self, stage, status, **extra):
        write(
            STATE / f"{stage}.status.json",
            dict(
                status=status,
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **extra,
            ),
        )

    def identities(self):
        heads = {}
        for root in (TRAIN,) if PHASE == "baseline" else (TRAIN, MIX):
            require(
                not subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    text=True,
                ).strip(),
                f"tracked changes in runtime checkout {root}",
            )
            heads[str(root)] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
        paths = [
            *list(PINS),
            SIDECAR / "bt4_policy_sidecar_summary.json",
            E0 / "summary.json",
            OPENINGS,
            PREREGISTRATION,
            Path(__file__).resolve(),
        ]
        if PHASE == "treatments":
            paths += [names(arm)[2] for arm in ARMS[1:]]
            paths += [
                MIX / "scripts/bt4_policy_mix.py",
                names(ARMS[0])[1] / "checkpoint.pt",
            ]
        hashes = {str(path): sha(path) for path in paths}
        for path, expected in PINS.items():
            require(hashes[str(path)] == expected, f"identity mismatch: {path}")
        return {"heads": heads, "hashes": hashes}

    def check_code(self):
        # The complete checkout heads pin the imported runtime, not just entry points.
        for path, expected in self.manifest["runtime"][
            "native_extension_sha256"
        ].items():
            require(sha(Path(path)) == expected, f"native runtime changed: {path}")
        for root, head in self.manifest["identities"]["heads"].items():
            require(
                subprocess.check_output(
                    ["git", "-C", root, "rev-parse", "HEAD"], text=True
                ).strip()
                == head,
                f"runtime checkout moved: {root}",
            )
            require(
                not subprocess.check_output(
                    [
                        "git",
                        "-C",
                        root,
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    text=True,
                ).strip(),
                f"tracked runtime changes: {root}",
            )

    def run(self, stage, command, cwd, gpu_fd=None):
        self.guard()
        self.check_code()
        self.status(stage, "running", command=list(map(str, command)), cwd=str(cwd))
        start = time.monotonic()
        env = dict(
            os.environ,
            PYTHONPATH=str(cwd),
            CUDA_VISIBLE_DEVICES="0" if gpu_fd is not None else "",
            PYTHONUNBUFFERED="1",
            BLOSC_NTHREADS="4",
        )
        fds = (self.lock_fd,) if gpu_fd is None else (self.lock_fd, gpu_fd)
        with (STATE / f"{stage}.log").open("a") as log:
            child = subprocess.Popen(
                list(map(str, command)),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=fds,
            )
            try:
                write(
                    STATE / f"{stage}.process.json",
                    {"pid": child.pid, "command": command},
                )
                while child.poll() is None:
                    self.guard()
                    self.stop.wait(5)
                require(child.returncode == 0, f"{stage} exited {child.returncode}")
            except BaseException as error:
                self.status(
                    stage,
                    "failed",
                    error=str(error),
                    elapsed_seconds=time.monotonic() - start,
                )
                raise
            finally:
                # Kill the owned process group even if its leader already exited.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self.status(stage, "command_complete", elapsed_seconds=time.monotonic() - start)

    @contextmanager
    def gpu(self, stage, allowance):
        self.status(stage, "waiting_for_gpu")
        with (LIVE / "scratchpad/gpu0_experiment.lock").open("a") as lock:
            while True:
                self.guard()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.stop.wait(10)
                    continue
                pids = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    text=True,
                ).strip()
                if not pids:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
                self.stop.wait(10)
            used = 0.0
            for charge in ROOT_STATE.glob("*/*.gpu-charge.json"):
                record = read(charge)
                require(
                    record.get("complete") is True,
                    f"{charge}: interrupted GPU charge needs reconciliation",
                )
                used += float(record["seconds"])
            require(
                used + allowance <= GPU_BUDGET_SECONDS,
                f"{stage}: remaining GPU budget needs prospective amendment",
            )
            started = time.monotonic()
            charge_path = STATE / f"{stage}.{time.time_ns()}.gpu-charge.json"
            charge = {
                "stage": stage,
                "complete": False,
                "started_unix": time.time(),
                "owner_pid": os.getpid(),
            }
            write(charge_path, charge)
            self.gpu_deadline = started + GPU_BUDGET_SECONDS - used
            try:
                yield lock.fileno()
            finally:
                self.gpu_deadline = None
                # A killed coordinator leaves an explicit unresolved charge.
                write(
                    charge_path,
                    dict(charge, complete=True, seconds=time.monotonic() - started),
                )
                fcntl.flock(lock, fcntl.LOCK_UN)

    def mix(self, arm):
        letter, alpha, temperature = arm
        corpus, _, audit = names(arm)
        stage = f"{letter}.mix"
        require(letter != "S0", "the control uses the untouched source corpus")
        if not corpus.exists():
            require(
                shutil.disk_usage(LIVE).free
                >= 150 * 1024**3 + self.manifest["source_footprint_bytes"],
                f"{letter}: insufficient space for a corpus copy plus reserve",
            )
            require(
                not corpus.with_name(corpus.name + ".writing").exists(),
                f"partial corpus requires explicit recovery: {corpus}",
            )
            self.run(
                stage,
                [
                    "nice",
                    "-n",
                    "19",
                    MIX_PYTHON,
                    "scripts/bt4_policy_mix.py",
                    "mix",
                    "--shards",
                    str(SOURCE),
                    "--sidecar",
                    str(SIDECAR),
                    "--out",
                    str(corpus),
                    "--scope",
                    "global",
                    "--alpha",
                    str(alpha),
                    "--bt4-temperature",
                    str(temperature),
                    "--expected-rows",
                    str(ROWS),
                    "--expected-shards",
                    str(SHARDS),
                    "--expected-source-summary-sha256",
                    SOURCE_SHA,
                    "--audit-receipt",
                    str(audit),
                    "--sf-audit-mode",
                    "descriptive",
                    "--experiment-record",
                    str(PREREGISTRATION),
                ],
                MIX,
            )
        summary = read(corpus / "bt4_policy_mix_summary.json")
        expected = {
            "kind": "global",
            "alpha": alpha,
            "bt4_temperature": temperature,
            "rows": ROWS,
            "shards": SHARDS,
            "source_dir": str(SOURCE),
            "source_derive_summary_sha256": SOURCE_SHA,
            "sidecar_dir": str(SIDECAR),
            "sf_rank_sidecar_dir": None,
            "mutated_arrays": ["policy_target"],
        }
        require(
            all(summary.get(k) == v for k, v in expected.items()),
            f"{letter}: published corpus treatment differs",
        )
        for field, path in [
            ("audit_receipt", audit),
            ("sidecar_summary", SIDECAR / "bt4_policy_sidecar_summary.json"),
        ]:
            require(
                summary[field]["sha256"]
                == self.manifest["identities"]["hashes"][str(path)],
                f"{letter}: published {field} identity differs",
            )
        require(
            len(list(corpus.glob("shard_*.zarr"))) == SHARDS,
            f"{letter}: published shard count differs",
        )
        require(
            read(corpus / "derive_targets_summary.json").get(
                "policy_target_postprocess"
            )
            == summary,
            f"{letter}: derived provenance differs from mix receipt",
        )
        self.status(
            stage,
            "complete",
            summary_sha256=sha(corpus / "bt4_policy_mix_summary.json"),
        )

    def train(self, arm):
        corpus, run, _ = names(arm)
        stage = f"{arm[0]}.train"
        receipt = STATE / f"{stage}.complete.json"
        if receipt.exists():
            require(
                read(receipt) == training_complete(run),
                f"{stage}: output identity drift",
            )
            return
        require(
            not run.exists(), f"{run}: existing training requires explicit recovery"
        )
        with self.gpu(stage, STAGE_ALLOWANCE["train"]) as gpu_fd:
            self.run(
                stage,
                [
                    PYTHON,
                    "scripts/lc0_control_train.py",
                    "--config",
                    "configs/lc0_positive_control.yaml",
                    "--shards",
                    str(corpus),
                    "--out-dir",
                    str(run),
                    "--steps",
                    "0",
                    "--batch-size",
                    "512",
                    "--sampling-mode",
                    "game_epoch",
                    "--epoch-plan-workers",
                    "16",
                    "--epoch-load-workers",
                    "16",
                    "--seed",
                    "0",
                    "--device",
                    "cuda",
                    "--train-window-steps",
                    "88",
                    "--allow-invalid-control",
                ],
                TRAIN,
                gpu_fd,
            )
        result = training_complete(run)
        write(receipt, result)
        self.status(stage, "complete", **result)

    def arena(self, arm, sims):
        run = E0 if arm[0] == "E0" else names(arm)[1]
        reference = names(ARMS[0])[1]
        stage = f"{arm[0]}.s{sims}.arena"
        games = STATE / f"{stage}.games.jsonl"
        receipt = STATE / f"{stage}.complete.json"
        if receipt.exists():
            require(
                read(receipt)["games_sha256"] == sha(games),
                f"{stage}: bank identity drift",
            )
            return
        command = [
            PYTHON,
            "scripts/arena_standard.py",
            "--candidate",
            str(run / "checkpoint.pt"),
            "--reference",
            str(reference / "checkpoint.pt"),
            "--games",
            "1000",
            "--mode",
            "matched_sims",
            "--search-shape",
            "training",
            "--sims",
            str(sims),
            "--seed",
            "42",
            "--openings",
            str(OPENINGS),
            "--opening-plies",
            "16",
            "--max-plies",
            "300",
            "--temperature",
            "0.1",
            "--max-concurrent-games",
            "128",
            "--eval-max-batch",
            "4096",
            "--compile",
            "on",
            "--no-rolling",
            "--label",
            f"bt4_global_{arm[0]}_vs_S0_s{sims}",
            "--games-out",
            str(games),
            "--out",
            str(STATE / f"{stage}.results.jsonl"),
        ]
        if games.exists():
            command.append("--resume")
        with self.gpu(stage, STAGE_ALLOWANCE[sims]) as gpu_fd:
            self.run(stage, command, TRAIN, gpu_fd)
        # Use the producing arena's resume semantics, preserving its append-only bank.
        sys.path.insert(0, str(TRAIN))
        try:
            from chess_anti_engine.utils.game_log import (
                latest_rows_by_key,
                read_game_log,
            )
        finally:
            sys.path.pop(0)
        bank = read_game_log(games)
        raw_pairs = {}
        for row in bank.games:
            raw_pairs.setdefault(row["pair_id"], []).append(row)
        require(
            all(
                len(pair) >= 2
                and {r["half"] for r in pair[-2:]} == {0, 1}
                and len({r["half"] for r in pair[:-2]}) <= 1
                and len({r["opening_fen"] for r in pair}) == 1
                for pair in raw_pairs.values()
            ),
            f"{stage}: unsupported complete-pair replay or unfinished pair",
        )
        rows = list(
            latest_rows_by_key(bank.games, lambda r: (r["pair_id"], r["half"])).values()
        )
        require(
            not bank.truncated_tail and len(rows) == 1000, f"{stage}: incomplete bank"
        )
        settings = bank.settings
        for key, value in {
            "mode": "matched_sims",
            "games": 1000,
            "seed": 42,
            "sims_candidate": sims,
            "sims_reference": sims,
            "candidate": str(run / "checkpoint.pt"),
            "reference": str(reference / "checkpoint.pt"),
            "opening_plies": 16,
            "max_plies": 300,
            "temperature": 0.1,
        }.items():
            require(settings.get(key) == value, f"{stage}: protocol differs: {key}")
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair_id"], []).append(row)
        require(
            len(pairs) == 500
            and all(
                len(pair) == 2
                and {r["half"] for r in pair} == {0, 1}
                and {r["a_is_white"] for r in pair} == {True, False}
                and pair[0]["opening_fen"] == pair[1]["opening_fen"]
                for pair in pairs.values()
            ),
            f"{stage}: incomplete or unpaired openings",
        )
        result = {
            "games_sha256": sha(games),
            "checkpoint_sha256": sha(run / "checkpoint.pt"),
            "reference_sha256": sha(reference / "checkpoint.pt"),
            "readout": "unread",
            "raw_rows": len(bank.games),
            "replaced_orphan_rows": len(bank.games) - len(rows),
        }
        write(receipt, result)
        self.status(stage, "complete", **result)

    def execute(self):
        self.guard()
        identities = self.identities()
        training_complete(E0)
        if PHASE == "treatments":
            require(
                read(ROOT_STATE / "baseline/S0.train.complete.json")
                == training_complete(names(ARMS[0])[1]),
                "S0 completion identity differs",
            )
        runtime = subprocess.check_output(
            [
                PYTHON,
                "-c",
                "import importlib,json,sys,torch; "
                "modules=['chess_anti_engine.encoding._features_ext',"
                "'chess_anti_engine.encoding._lc0_ext',"
                "'chess_anti_engine.mcts._mcts_tree',"
                "'chess_anti_engine.nnue._nnue_ext']; "
                "print(json.dumps(dict(python=sys.version,"
                "executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,"
                "native_extensions={m:importlib.import_module(m).__file__ for m in modules})))",
            ],
            cwd=TRAIN,
            text=True,
        ).strip()
        info = json.loads(runtime)
        info["native_extension_sha256"] = {
            path: sha(Path(path)) for path in info["native_extensions"].values()
        }
        require(
            info["torch"] == "2.11.0+cu128" and info["python"].startswith("3.10.12"),
            f"E0-compatible runtime changed: {info}",
        )
        footprint = max(
            int(
                subprocess.check_output(
                    ["du", "--summarize", "--block-size=1", *flags, str(SOURCE)],
                    text=True,
                ).split()[0]
            )
            for flags in ([], ["--apparent-size"])
        )
        manifest = {
            "identities": identities,
            "runtime": info,
            "arms": ARMS,
            "phase": PHASE,
            "budgets": BUDGETS,
            "source_footprint_bytes": footprint,
        }
        # JSON normalization makes tuples compare consistently across restarts.
        manifest = json.loads(json.dumps(manifest))
        path = STATE / "manifest.json"
        if path.exists():
            require(
                read(path) == manifest,
                "run manifest drift; do not mix experiment identities",
            )
        else:
            write(path, manifest)
        self.manifest = manifest
        try:
            if PHASE == "baseline":
                self.train(ARMS[0])
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.mix, ARMS[1])
                    try:
                        for sims in BUDGETS:
                            self.arena(("E0", 0.0, 1.0), sims)
                        for index, arm in enumerate(ARMS[1:], start=1):
                            future.result()
                            if index + 1 < len(ARMS):
                                future = executor.submit(self.mix, ARMS[index + 1])
                            self.train(arm)
                            for sims in BUDGETS:
                                self.arena(arm, sims)
                    finally:
                        self.stop.set()
        except BaseException as error:
            self.status("driver", "failed", error=str(error))
            raise
        self.status(
            "driver", "complete", readout="unread; apply registered paired comparison"
        )


def main():
    global PHASE, STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("baseline", "treatments"), default="baseline"
    )
    parser.add_argument(
        "--execute", action="store_true", help="launch the registered phase"
    )
    args = parser.parse_args()
    PHASE = args.phase
    STATE = ROOT_STATE / PHASE
    if not args.execute:
        print(
            json.dumps(
                {
                    "state": str(STATE),
                    "plan": [
                        {
                            "arm": arm,
                            "corpus": str(names(arm)[0]),
                            "run": str(names(arm)[1]),
                        }
                        for arm in (ARMS[:1] if PHASE == "baseline" else ARMS[1:])
                    ],
                },
                indent=2,
            )
        )
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        driver = Driver(lock.fileno())
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: driver.stop.set())
        write(
            STATE / "driver.process.json",
            {"pid": os.getpid(), "script_sha256": sha(Path(__file__))},
        )
        driver.execute()


if __name__ == "__main__":
    main()
