"""Finite one-game Zarr readback gate. No generation, model, optimizer or training.

Trusted local caller/filesystem only. A qualified game is not a complete E2 bank:
128-game selection, immutable shared batch trace, optimizer/LR identity and TB
execution receipts remain separate experiment gates. Target files are explicit
adapter inputs; existing trainers do not automatically consume this overlay.
Consumers must revalidate all manifest hashes at load time; this does not make
external files immutable or authenticate historical producer/dependency state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

MAX_BYTES = 128 * 1024**2
SOURCE_FILES = (
    "scripts/e2_bank_gate.py",
    "chess_anti_engine/natural_outcome.py",
    "chess_anti_engine/replay/shard.py",
    "chess_anti_engine/train/targets.py",
    "chess_anti_engine/moves/encode.py",
    "chess_anti_engine/encoding/lc0.py",
    "chess_anti_engine/encoding/_lc0_ext.cpython-313-x86_64-linux-gnu.so",
    "chess_anti_engine/encoding/_features_ext.cpython-313-x86_64-linux-gnu.so",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: digest(root / name) for name in SOURCE_FILES}


def digest(path):
    with Path(path).open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def inventory(root):
    files = sorted(Path(root).rglob("*"))
    if len(files) > 4096 or any(p.is_symlink() for p in files):
        raise ValueError("unbounded or symlink shard")
    files = [p for p in files if p.is_file()]
    if sum(p.stat().st_size for p in files) > MAX_BYTES:
        raise ValueError("shard byte cap")
    return {str(p.relative_to(root)): digest(p) for p in files}


def atomic_exclusive(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    linked = False
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
        linked = True
        d = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    except BaseException:
        if linked:
            path.unlink()  # Never leave an apparent committed marker on failure.
        raise
    finally:
        os.unlink(tmp)


def qualify_child(shard, candidate, expected, output):
    # Only the isolated worker changes the native repetition switch.
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    plan = json.loads((Path(output) / "frozen-plan.json").read_text())
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source identity changed")
    import chess
    import numpy as np
    from chess_anti_engine.encoding import _lc0_ext
    from chess_anti_engine.encoding.lc0 import c_input_history_mode
    from chess_anti_engine.moves import move_to_index_for_encoding
    from chess_anti_engine.natural_outcome import (
        NaturalOutcome,
        admit_rows,
        require_search_labels,
    )
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target

    candidate = Path(candidate)
    if candidate.is_symlink() or candidate.stat().st_size > 2 * 1024**2:
        raise ValueError("candidate bound")
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("external candidate identity mismatch")
    c = json.loads(raw)
    if c["status"] != "CAPTURE_CANDIDATE_NOT_BANK_COMMITTED":
        raise ValueError("unsupported candidate")
    natural = c["natural"]
    receipt = NaturalOutcome(
        tuple(natural["moves"]),
        natural["result"],
        natural["termination"],
        natural["final_fen"],
        natural["trace_sha256"],
    )
    rows = c["rows"]
    if not 1 <= len(rows) <= 1600 or c["post_rescore_result"] != receipt.result:
        raise ValueError("natural row/result gate")
    before = inventory(shard)
    arrays, meta = load_shard_arrays(shard, validate=True)
    if (
        meta.get("history_rep_fix") is not True
        or meta.get("input_history_encoding") != "lc0_root_legacy_meta"
    ):
        raise ValueError("encoding identity mismatch")
    n = len(rows)
    x = arrays["x"]
    if x.shape != (n, 175, 8, 8) or x.dtype != np.dtype(np.float16):
        raise ValueError("stored175 shape/dtype")
    for flag in (
        "has_policy",
        "has_search_wdl",
        "has_game_id",
        "has_ply_index",
        "has_is_network_turn",
        "is_network_turn",
        "has_is_selfplay",
        "is_selfplay",
    ):
        if (
            flag not in arrays
            or arrays[flag].shape != (n,)
            or not np.all(arrays[flag] == 1)
        ):
            raise ValueError("missing required mask: " + flag)
    plies = [int(v) for v in arrays["ply_index"]]
    turns = [not bool(v[108, 0, 0]) for v in x]
    labels = [int(v) for v in arrays["wdl_target"]]
    for start in range(0, n, 256):
        admit_rows(
            receipt,
            expected_trace_sha256=receipt.trace_sha256,
            plies=plies[start : start + 256],
            stm_white=turns[start : start + 256],
            outcomes=labels[start : start + 256],
            tb_rescored=False,
        )
    if len(set(plies)) != n:
        raise ValueError("duplicate ply")
    for i, row in enumerate(rows):
        if (
            row["game_id"] != int(arrays["game_id"][i])
            or row["ply"] != plies[i]
            or row["stm_white"] != turns[i]
            or row["natural_wdl_class"] != labels[i]
            or row["stored_float16_x_sha256"]
            != hashlib.sha256(x[i].astype("<f2").tobytes()).hexdigest()
        ):
            raise ValueError("emitted row crosswalk mismatch")
    _lc0_ext.set_history_rep_fix(True)
    board = chess.Board()
    wanted = {p: i for i, p in enumerate(plies)}
    policy = arrays["policy_target"]
    if policy.shape != (n, 1858) or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("policy declaration")
    for ply, uci in enumerate(receipt.moves):
        if ply in wanted:
            i = wanted[ply]
            encoded = (
                _lc0_ext.CBoard.from_board(board)
                .encode_full(c_input_history_mode("lc0_root_legacy_meta"), 63)
                .astype(np.float16)
            )
            if not np.array_equal(encoded, x[i]):
                raise ValueError("full175 replay mismatch")
            legal = {
                move_to_index_for_encoding(m, board, policy_encoding="lc0_1858")
                for m in board.legal_moves
            }
            if (
                any(float(v) != 0 for j, v in enumerate(policy[i]) if j not in legal)
                or abs(float(policy[i].astype(np.float64).sum()) - 1) > 1e-3
            ):
                raise ValueError("legal policy mismatch")
        board.push_uci(uci)
    search_raw = arrays["search_wdl"]
    if (
        search_raw.shape != (n, 3)
        or search_raw.dtype != np.dtype(np.float16)
        or not np.isfinite(search_raw).all()
        or np.any(search_raw < 0)
        or np.any(search_raw > 1)
    ):
        raise ValueError("search declaration")
    search = search_raw.astype(np.float64)
    sums = search.sum(axis=1)
    if np.max(np.abs(sums - 1)) > 1e-3:
        raise ValueError("search quantization tolerance")
    search /= sums[:, None]
    for start in range(0, n, 256):
        require_search_labels(
            search[start : start + 256].tolist(), [True] * min(256, n - start)
        )
    onehot = np.eye(3)[labels]
    wdl_a, wdl_b = search, 0.8 * search + 0.2 * onehot
    cat_a = []
    cat_b = []
    for i, label in enumerate(labels):
        outcome = (1.0, 0.0, -1.0)[label]
        for frac, dest in ((1.0, cat_a), (0.8, cat_b)):
            scalar = categorical_target_value(
                outcome,
                None,
                blend_frac=0.0,
                search_wdl=search[i],
                search_blend_frac=frac,
            )
            dest.append(hlgauss_target(scalar, num_bins=32, sigma=0.04))
    if (
        before != inventory(shard)
        or digest(candidate) != expected
        or source_hashes() != plan["source_sha256"]
    ):
        raise ValueError("source changed during readback")
    out = Path(output)
    with (out / "arm-targets.npz").open("xb") as f:
        np.savez(
            f,
            wdl_a=wdl_a.astype(np.float32),
            wdl_b=wdl_b.astype(np.float32),
            categorical_a=np.array(cat_a),
            categorical_b=np.array(cat_b),
        )
        f.flush()
        os.fsync(f.fileno())
    result = {
        "status": "GAME_READBACK_QUALIFIED_NOT_EXPERIMENT_READY",
        "candidate_sha256": expected,
        "source_sha256": plan["source_sha256"],
        "source_inventory": before,
        "rows": n,
        "trace_sha256": receipt.trace_sha256,
        "all175_replayed": True,
        "target_origin": "reconstructed automatic natural outcome; TB never substituted",
        "tb_rescore_enabled_at_capture": c["tb_rescore_enabled"],
        "search_raw_sha256": hashlib.sha256(search_raw.tobytes()).hexdigest(),
        "search_max_sum_error": float(np.max(np.abs(sums - 1))),
        "search_max_normalization_change": float(
            np.max(np.abs(search - search_raw.astype(np.float64)))
        ),
        "arm_targets_sha256": digest(out / "arm-targets.npz"),
        "categorical": {
            "bins": 32,
            "sigma": 0.04,
            "method": "scalar Q blend then HL-Gauss",
        },
        "remaining_gates": [
            "whole-bank selection and immutable batch trace",
            "optimizer/LR/control identity",
            "TB execution receipt admission",
            "trainer overlay integration",
        ],
        "source_identity": "trusted local observed code; not historical authentication",
    }
    atomic_exclusive(
        out / "verified-candidate.json", json.dumps(result, sort_keys=True).encode()
    )


def qualify_game(
    shard: Path, candidate: Path, expected_sha256: str, output_root: Path
) -> Path:
    """No overwrite; bounded owned child; failed directories remain uncommitted."""
    output = Path(tempfile.mkdtemp(prefix="e2-game-", dir=output_root))
    plan = {
        "source_sha256": source_hashes(),
        "candidate_sha256": expected_sha256,
        "limits": {
            "AS_bytes": 8 * 1024**3,
            "CPU_seconds": 60,
            "wall_seconds": 90,
            "threads": 1,
        },
    }
    atomic_exclusive(
        output / "frozen-plan.json", json.dumps(plan, sort_keys=True).encode()
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(shard),
            str(candidate),
            expected_sha256,
            str(output),
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _stdout, stderr = proc.communicate(timeout=90)
        if proc.returncode != 0:
            raise RuntimeError(
                "bank qualification failed: " + stderr[-4000:].decode(errors="replace")
            )
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    if (output / "verified-candidate.json").stat().st_size > 2 * 1024**2:
        raise ValueError("receipt bound")
    result = (output / "verified-candidate.json").read_bytes()
    parsed = json.loads(result)
    if (
        parsed["candidate_sha256"] != expected_sha256
        or parsed["source_sha256"] != plan["source_sha256"]
        or digest(output / "arm-targets.npz") != parsed["arm_targets_sha256"]
    ):
        raise ValueError("child receipt binding")
    if inventory(shard) != parsed["source_inventory"] or digest(candidate) != expected_sha256:
        raise ValueError("source changed before game commit")
    atomic_exclusive(output / "GAME-READBACK-COMMITTED.json", result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=4, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    qualify_child(*args.child)
