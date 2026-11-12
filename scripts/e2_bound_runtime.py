"""Synthetic-qualified strict-loader/finite-bank process wrapper.

NO real experiment route is enabled. CPU16GiB/120s and180s wall are synthetic
qualification ceilings, not an estimate for the full donor or GPU permission.
Caller must externally pin plan/checkpoint/bank; digests do not authenticate origin.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/e2_bound_runtime.py",
        "scripts/e2_bank_gate.py",
        "scripts/stored_wdl_checkpoint.py",
        "chess_anti_engine/train/e2_runtime.py",
        "chess_anti_engine/train/e2_batch_adapter.py",
        "chess_anti_engine/train/e2_objective.py",
        "chess_anti_engine/train/trainer.py",
        "chess_anti_engine/train/losses.py",
        "chess_anti_engine/replay/shard.py",
    )
    return {name: sha(root / name) for name in names}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, expected, cap=2 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("bounded regular receipt required")
    raw = path.read_bytes()
    if len(raw) > cap or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("receipt identity mismatch")
    return json.loads(raw)


def within(root, name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("bank path escape")
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("bank path escape")
    return result


def load_trace(plan):
    import torch
    from chess_anti_engine.train.e2_batch_adapter import (
        load_paired_batch,
        PairedBatch,
        tensor_hash,
    )

    root = Path(plan["bank_manifest"]).parent
    bank = read_json(plan["bank_manifest"], plan["bank_sha256"])
    if bank["status"] != "FINITE_READBACK_BANK_NOT_EXPERIMENT_READY":
        raise ValueError("not committed finite readback bank")
    attempts = bank["attempts"]
    selected = bank["selected_start_ordinals"]
    if not 1 <= len(attempts) <= 160 or [r["ordinal"] for r in attempts] != list(
        range(len(attempts))
    ):
        raise ValueError("complete ordered attempt accounting required")
    eligible = [r["ordinal"] for r in attempts if r["exclusion"] is None]
    if not 1 <= len(selected) <= 128 or selected != eligible[: len(selected)]:
        raise ValueError("selection differs from first eligible ordinals")
    if plan["synthetic"] is not True or len(selected) > 2:
        raise ValueError("real bank execution not resource-qualified")
    for row in attempts:
        obs = read_json(
            root / f"{row['ordinal']}.observation.json", row["observation_sha256"]
        )
        if (obs["ordinal"], obs["game_id"], obs["exclusion"], obs["run_id"]) != (
            row["ordinal"],
            row["game_id"],
            row["exclusion"],
            bank["run_id"],
        ):
            raise ValueError("attempt observation join mismatch")
    trace = plan["trace"]
    if not 1 <= len(trace) <= 64:
        raise ValueError("bounded trace required")
    used = set()
    pairs = []
    for segments in trace:
        if not 1 <= len(segments) <= len(selected):
            raise ValueError("bounded batch segments required")
        parts = []
        count = 0
        for segment in segments:
            ordinal = segment["ordinal"]
            indices = segment["indices"]
            if (
                type(ordinal) is not int
                or ordinal not in selected
                or not 1 <= len(indices) <= 4
                or any(type(i) is not int or i < 0 for i in indices)
            ):
                raise ValueError("trace refers to nonselected game or invalid rows")
            used.add(ordinal)
            row = attempts[ordinal]
            count += len(indices)
            if count > 4:
                raise ValueError("synthetic batch cap")
            parts.append(
                load_paired_batch(
                    shard=within(root, row["shard"]),
                    candidate=within(root, row["candidate"]),
                    qualified=within(root, row["qualified"]),
                    expected_manifest_sha256=row["readback_sha256"],
                    indices=tuple(indices),
                    expected_indices_sha256=hashlib.sha256(
                        json.dumps(indices, separators=(",", ":")).encode()
                    ).hexdigest(),
                    categorical_bins=32,
                    categorical_sigma=0.04,
                )
            )
            if sha(within(root, row["candidate"])) != row["candidate_sha256"]:
                raise ValueError("bank candidate pin mismatch")
        keys = set(parts[0].search_only)
        if any(set(p.search_only) != keys or set(p.outcome20) != keys for p in parts):
            raise ValueError("batch field mismatch")
        a = {k: torch.cat([p.search_only[k] for p in parts], dim=0) for k in keys}
        b = {k: torch.cat([p.outcome20[k] for p in parts], dim=0) for k in keys}
        pairs.append(
            PairedBatch(
                a,
                b,
                {"bank_sha256": plan["bank_sha256"], "segments": segments},
                tuple((k, tensor_hash(a[k]), tensor_hash(b[k])) for k in sorted(keys)),
            )
        )
    if used != set(selected):
        raise ValueError("trace omits selected games")
    return pairs


def child(plan, output):
    from scripts.stored_wdl_checkpoint import _qualify
    from chess_anti_engine.train.e2_objective import FrozenObjective, canonical
    from chess_anti_engine.train.e2_runtime import (
        run_verified_pair,
        digest,
        trace_digest,
    )

    if plan["source_sha256"] != source_hashes():
        raise ValueError("helper source identity mismatch")
    if plan["checkpoint"].get("operation", "qualify") != "qualify":
        raise ValueError("load-only checkpoint request required")
    pairs = load_trace(plan)
    objective = FrozenObjective(plan["objective_json"], plan["objective_sha256"])
    pins = json.loads(objective.payload_json)["pins"]
    request = plan["checkpoint"]
    if (
        pins["donor"] != request["sha256"]
        or pins["architecture"] != digest(json.loads(request["architecture_json"]))
        or pins["batch_trace"] != trace_digest(pairs)
    ):
        raise ValueError("checkpoint/architecture/trace objective binding mismatch")

    def consume(model, report):
        if (
            report["sha256"] != pins["donor"]
            or digest(report["architecture"]) != pins["architecture"]
        ):
            raise ValueError("realized strict loader identity mismatch")
        return run_verified_pair(
            model,
            objective,
            objective.sha256,
            pairs,
            trainer_overrides=plan["trainer_overrides"],
            seed=plan["seed"],
            synthetic=True,
        )

    result = _qualify(request, verified_model_callback=consume)
    if plan["source_sha256"] != source_hashes():
        raise ValueError("source changed during runtime")
    result["source_sha256"] = plan["source_sha256"]
    result["synthetic_training_performed"] = True
    result["bank_manifest_sha256"] = plan["bank_sha256"]
    result["wrapper_scope"] = (
        "synthetic strict-loaded runtime; not real experiment qualified"
    )
    Path(output).write_text(canonical(result) + "\n")


def run_synthetic(plan_path, expected_plan_sha256, output_root):
    plan = read_json(plan_path, expected_plan_sha256)
    if plan.get("synthetic") is not True:
        raise ValueError("synthetic-only wrapper")
    output = Path(tempfile.mkdtemp(prefix="e2-bound-", dir=output_root))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                str(Path(plan_path).resolve()),
                expected_plan_sha256,
                str(output / "candidate.json"),
            ],
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=180)
        except BaseException:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
            raise
    if code != 0:
        raise ValueError(
            "bounded child failed: " + (output / "stderr.log").read_text()[-2000:]
        )
    result = read_json(
        output / "candidate.json", sha(output / "candidate.json"), 8 * 1024**2
    )
    if (
        result["source_sha256"] != source_hashes()
        or result["bank_manifest_sha256"] != plan["bank_sha256"]
        or sha(plan_path) != expected_plan_sha256
    ):
        raise ValueError("post-run binding mismatch")
    from scripts.e2_bank_gate import atomic_exclusive

    atomic_exclusive(
        output / "SYNTHETIC-RUNTIME-COMMITTED.json",
        (output / "candidate.json").read_bytes(),
    )
    return output


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    child(read_json(sys.argv[1], sys.argv[2]), sys.argv[3])
