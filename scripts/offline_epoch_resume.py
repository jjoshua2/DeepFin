"""Durable, exact cursor commits for ``offline_replay_epoch`` fixed epochs.

The checkpoint and its JSON cursor are a two-file transaction: the cursor is
published last and names the checkpoint by SHA-256.  A checkpoint without a
cursor is an uncommitted orphan, never an invitation to guess where to resume.
"""

from __future__ import annotations

import copy
from collections.abc import Generator, Mapping
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys
import time
import uuid
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.utils.atomic import atomic_write_text


MAX_COMMIT_SECONDS = 1800.0
MAX_INTERVAL_SECONDS = 1200.0
KEEP_COMMITTED_GENERATIONS = 2


@contextmanager
def candidate_lock(run_dir: Path) -> Generator[None]:
    """Only one process may advance a candidate's durable generation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".resume.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"resumable candidate is already active: {run_dir}") from exc
        yield


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path: Path, *, deadline: float | None = None) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("source admission exceeded 30 minutes; supply a qualified"
                                   " --resume-source-manifest for a long corpus")
            digest.update(block)
    return digest.hexdigest()


def _tree_receipt(path: Path, *, deadline: float | None = None) -> dict[str, Any]:
    """Hash all bytes and relative names of one immutable Zarr or NPZ shard."""
    root = path.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"resume shard is a symlink: {path}")
    entries = [root] if root.is_file() else []
    if root.is_dir():
        for entry in root.rglob("*"):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("source admission exceeded 30 minutes; supply a qualified"
                                   " --resume-source-manifest for a long corpus")
            entries.append(entry)
    if any(p.is_symlink() for p in entries):
        raise ValueError(f"resume shard is empty or contains a symlink: {path}")
    files = sorted(p for p in entries if p.is_file())
    if not files:
        raise ValueError(f"resume shard is empty: {path}")
    digest = sha256()
    total = 0
    for file in files:
        rel = file.name if root.is_file() else file.relative_to(root).as_posix()
        info = file.stat()
        before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        content_sha = _sha_file(file, deadline=deadline)
        after = file.stat()
        if before != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"resume shard changed while hashing: {file}")
        digest.update(_json([rel, info.st_size, content_sha]).encode() + b"\n")
        total += info.st_size
    return {"path": str(root), "sha256": digest.hexdigest(), "bytes": total}


def _rng_state(rng: np.random.Generator, mask_rng: np.random.RandomState) -> dict[str, Any]:
    py = random.getstate()
    return {
        "numpy": copy.deepcopy(rng.bit_generator.state),
        "mask": mask_state(mask_rng),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
        "python": [py[0], list(py[1]), py[2]],
    }


def _set_numpy(rng: np.random.Generator, state: Mapping[str, Any]) -> None:
    rng.bit_generator.state = copy.deepcopy(dict(state))


def _set_mask(mask_rng: np.random.RandomState, state: list[Any]) -> None:
    mask_rng.set_state((state[0], np.asarray(state[1], dtype=np.uint32),
                        int(state[2]), int(state[3]), float(state[4])))


def mask_state(mask_rng: np.random.RandomState) -> list[Any]:
    state = cast(tuple[str, np.ndarray, int, int, float],
                 cast(object, mask_rng.get_state()))
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def restore_mask(mask_rng: np.random.RandomState, state: list[Any]) -> None:
    _set_mask(mask_rng, state)


def restore_numpy(rng: np.random.Generator, state: Mapping[str, Any]) -> None:
    _set_numpy(rng, state)


def _restore_rng(state: dict[str, Any], rng: np.random.Generator,
                 mask_rng: np.random.RandomState) -> None:
    _set_numpy(rng, state["numpy"])
    _set_mask(mask_rng, state["mask"])
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    cuda = state["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("resume CUDA RNG device count changed")
    if cuda:
        torch.cuda.set_rng_state_all([torch.tensor(raw, dtype=torch.uint8) for raw in cuda])
    py = state["python"]
    random.setstate((int(py[0]), tuple(int(x) for x in py[1]), py[2]))


class EpochResume:
    """One candidate's frozen sources and committed trainer/cursor generation."""

    def __init__(self, run_dir: Path, *, science: dict[str, Any],
                 train_paths: list[Path], eval_paths: list[Path],
                 interval_seconds: float, resume: bool,
                 source_manifest: Path | None = None):
        if not 0 < interval_seconds <= MAX_INTERVAL_SECONDS:
            raise ValueError("resume checkpoint interval must be in (0, 1200] seconds")
        self.run_dir = run_dir
        self.interval = interval_seconds
        self.science_sha = sha256(_json(science).encode()).hexdigest()
        self.train_paths = [str(path.resolve(strict=True)) for path in train_paths]
        self.eval_paths = [str(path.resolve(strict=True)) for path in eval_paths]
        self.spec_path = run_dir / "resume_spec.json"
        self.manifest_path = run_dir / "resume_sources.json"
        self.last_commit_time = time.monotonic()
        self.last_commit_step = 0
        self.spec: dict[str, Any]
        self.manifest: dict[str, Any]
        if resume:
            if not self.spec_path.is_file() or not self.manifest_path.is_file():
                raise ValueError("--resume requires an existing exact run spec and source manifest")
            self.spec = json.loads(self.spec_path.read_bytes())
            if self.spec.get("science_sha256") != self.science_sha:
                raise ValueError("resume scientific configuration or source code changed")
            if self.spec.get("manifest_sha256") != _sha_file(self.manifest_path):
                raise ValueError("resume source manifest changed")
            self.manifest = json.loads(self.manifest_path.read_bytes())
            if ([x["path"] for x in self.manifest["train"]] != self.train_paths
                    or [x["path"] for x in self.manifest["eval"]] != self.eval_paths):
                raise ValueError("resume ordered shard inventory changed")
        else:
            if any(path.name != ".resume.lock" for path in run_dir.iterdir()):
                raise ValueError("new resumable candidate requires an empty output directory")
            if source_manifest is None:
                deadline = time.monotonic() + MAX_COMMIT_SECONDS
                receipts: dict[str, dict[str, Any]] = {}

                def receipt(path: Path) -> dict[str, Any]:
                    key = str(path.resolve(strict=True))
                    if key not in receipts:
                        receipts[key] = _tree_receipt(path, deadline=deadline)
                    return receipts[key]

                self.manifest = {
                    "schema": "offline_epoch_sources_v1",
                    "train": [receipt(path) for path in train_paths],
                    "eval": [receipt(path) for path in eval_paths],
                }
            else:
                self.manifest = json.loads(source_manifest.read_bytes())
                if (self.manifest.get("schema") != "offline_epoch_sources_v1"
                        or [x["path"] for x in self.manifest["train"]] != self.train_paths
                        or [x["path"] for x in self.manifest["eval"]] != self.eval_paths):
                    raise ValueError("qualified resume source manifest inventory differs")
            atomic_write_text(self.manifest_path, _json(self.manifest) + "\n")
            self.spec = {"schema": "offline_epoch_resume_spec_v1",
                         "science": science,
                         "science_sha256": self.science_sha,
                         "manifest_sha256": _sha_file(self.manifest_path)}
            atomic_write_text(self.spec_path, _json(self.spec) + "\n")
        self.by_path = {item["path"]: item for section in ("train", "eval")
                        for item in self.manifest[section]}
        if self.manifest.get("schema") != "offline_epoch_sources_v1":
            raise ValueError("resume source manifest schema changed")
        for section, ordered in (("train", self.train_paths), ("eval", self.eval_paths)):
            items = self.manifest[section]
            if [item["path"] for item in items] != ordered:
                raise ValueError("resume source manifest ordered inventory changed")
            for item in items:
                if (not isinstance(item["bytes"], int) or item["bytes"] < 0
                        or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
                    raise ValueError("resume source manifest receipt is invalid")
                if self.by_path[item["path"]] != item:
                    raise ValueError("resume source manifest has conflicting duplicate path")
        commits = self._commits()
        self.next_commit_sequence = (int(commits[-1].name[7:19]) + 1) if commits else 0

    def verify_shard(self, path: Path) -> None:
        key = str(path.resolve(strict=True))
        if (key not in self.by_path
                or _tree_receipt(path, deadline=self.last_commit_time + MAX_COMMIT_SECONDS)
                != self.by_path[key]):
            raise ValueError(f"resume shard bytes changed: {path}")

    def _commits(self) -> list[Path]:
        commits = sorted(path for path in self.run_dir.glob("resume-*.json")
                         if path.name != "resume_result.json")
        if any(re.fullmatch(r"resume-[0-9]{12}-[0-9]{12}-[0-9a-f]{32}\.json",
                            path.name) is None for path in commits):
            raise ValueError("unexpected resume commit filename")
        if len(commits) != len({path.name[7:19] for path in commits}):
            raise ValueError("ambiguous resume commit sequence")
        return commits

    def _latest_sidecar(self) -> dict[str, Any]:
        commits = self._commits()
        if not commits:
            raise ValueError("--resume requires an existing committed trainer and cursor")
        sidecar = json.loads(commits[-1].read_bytes())
        if (sidecar.get("schema") != "offline_epoch_resume_commit_v1"
                or sidecar.get("science_sha256") != self.science_sha
                or sidecar.get("manifest_sha256") != self.spec["manifest_sha256"]
                or int(sidecar.get("sequence", -1)) != int(commits[-1].name[7:19])
                or int(sidecar["cursor"]["steps"]) != int(commits[-1].name[20:32])
                or sidecar.get("payload_sha256") != sha256(_json({
                    "cursor": sidecar["cursor"], "rng": sidecar["rng"],
                }).encode()).hexdigest()):
            raise ValueError("resume cursor identity changed")
        checkpoint = self.run_dir / sidecar["checkpoint_name"]
        if (checkpoint.name != sidecar["checkpoint_name"]
                or _sha_file(checkpoint) != sidecar["checkpoint_sha256"]):
            raise ValueError("resume trainer checkpoint missing or changed")
        cursor = sidecar["cursor"]
        required = ("epoch", "shard_pos", "batch_start", "steps", "positions",
                    "skipped_shards", "vr_seen", "vr_dropped", "shard_i",
                    "base_trainer_step", "epoch_rng_before", "shard_rng_before",
                    "mask_rng_before", "order_sha256", "keep_sha256")
        if not isinstance(cursor, dict) or any(key not in cursor for key in required):
            raise ValueError("resume cursor is incomplete")
        if (any(int(cursor[key]) < 0 for key in required[:10])
                or int(cursor["epoch"]) >= max(1, int(self.spec["science"]["settings"]["epochs"]))
                or int(cursor["shard_pos"]) > len(self.train_paths)
                or (int(cursor["batch_start"]) > 0
                    and (cursor["shard_rng_before"] is None
                         or cursor["mask_rng_before"] is None
                         or cursor["order_sha256"] is None))):
            raise ValueError("resume cursor is out of range or lacks current-shard state")
        return sidecar

    def load(self, trainer: Any, rng: np.random.Generator,
             mask_rng: np.random.RandomState) -> dict[str, Any]:
        sidecar = self._latest_sidecar()
        checkpoint = self.run_dir / sidecar["checkpoint_name"]
        trainer.load(checkpoint, exact_resume=True)
        if trainer.step != sidecar["trainer_step"]:
            raise ValueError("resume optimizer/scheduler step differs from cursor")
        _restore_rng(sidecar["rng"], rng, mask_rng)
        self.last_commit_time = time.monotonic()
        self.last_commit_step = int(sidecar["cursor"]["steps"])
        return sidecar["cursor"]

    def commit(self, trainer: Any, *, cursor: dict[str, Any],
               rng: np.random.Generator, mask_rng: np.random.RandomState) -> None:
        step = int(cursor["steps"])
        sequence = self.next_commit_sequence
        generation = f"{sequence:012d}-{step:012d}-{uuid.uuid4().hex}"
        checkpoint = self.run_dir / f"resume-trainer-{generation}.pt"
        trainer.save(checkpoint)
        rng_state = _rng_state(rng, mask_rng)
        sidecar = {
            "schema": "offline_epoch_resume_commit_v1",
            "science_sha256": self.science_sha,
            "manifest_sha256": self.spec["manifest_sha256"],
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": _sha_file(checkpoint),
            "trainer_step": trainer.step,
            "sequence": sequence,
            "cursor": cursor,
            "rng": rng_state,
            "payload_sha256": sha256(_json({
                "cursor": cursor, "rng": rng_state,
            }).encode()).hexdigest(),
        }
        atomic_write_text(self.run_dir / f"resume-{generation}.json", _json(sidecar) + "\n")
        self.next_commit_sequence += 1
        elapsed = time.monotonic() - self.last_commit_time
        self.last_commit_time = time.monotonic()
        self.last_commit_step = step
        self._prune_commits()
        if elapsed > MAX_COMMIT_SECONDS:
            raise RuntimeError(f"resume checkpoint window exceeded 30 minutes: {elapsed:.1f}s")

    def _prune_commits(self) -> None:
        """Retain two committed generations after the newest sidecar is durable."""
        commits = self._commits()
        for old in commits[:-KEEP_COMMITTED_GENERATIONS]:
            sidecar = json.loads(old.read_bytes())
            checkpoint = self.run_dir / sidecar["checkpoint_name"]
            if checkpoint.parent != self.run_dir or not checkpoint.name.startswith("resume-trainer-"):
                raise ValueError("resume sidecar names an unsafe checkpoint path")
            old.unlink()
            checkpoint.unlink(missing_ok=True)
        # An orphan has no cursor and is never selected. Remove only files in
        # the private generation namespace, after a valid commit exists.
        referenced = {json.loads(path.read_bytes())["checkpoint_name"]
                      for path in self._commits()}
        for checkpoint in self.run_dir.glob("resume-trainer-*.pt"):
            if checkpoint.name not in referenced:
                checkpoint.unlink()

    def due(self) -> bool:
        return time.monotonic() - self.last_commit_time >= self.interval

    def result(self, value: dict[str, Any]) -> None:
        final = self.run_dir / "trainer.pt"
        record = {"science_sha256": self.science_sha,
                  "manifest_sha256": self.spec["manifest_sha256"],
                  "trainer_sha256": _sha_file(final), "value": value}
        atomic_write_text(self.run_dir / "resume_result.json", _json(record) + "\n")

    def completed_result(self) -> dict[str, Any] | None:
        path = self.run_dir / "resume_result.json"
        if not path.exists():
            return None
        self._latest_sidecar()
        record = json.loads(path.read_bytes())
        if (record.get("science_sha256") != self.science_sha
                or record.get("manifest_sha256") != self.spec["manifest_sha256"]
                or record.get("trainer_sha256") != _sha_file(self.run_dir / "trainer.pt")):
            raise ValueError("completed resume result or final trainer changed")
        return record["value"]


def source_identity(script: Path, helper: Path, package: Path, config: Path,
                    init_checkpoint: Path | None) -> dict[str, Any]:
    """Pin local implementation and donor bytes without hashing corpus on restart."""
    package_files = sorted(p for p in package.rglob("*")
                           if p.is_file() and p.suffix in (".py", ".so"))
    package_hash = sha256()
    for path in package_files:
        package_hash.update(_json([path.relative_to(package).as_posix(), _sha_file(path)]).encode() + b"\n")
    return {"runner_sha256": _sha_file(script), "helper_sha256": _sha_file(helper),
            "package_sha256": package_hash.hexdigest(),
            "config_sha256": _sha_file(config),
            "init_sha256": _sha_file(init_checkpoint) if init_checkpoint else None,
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
