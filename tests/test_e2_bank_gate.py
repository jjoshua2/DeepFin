import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.e2_bank_gate import atomic_exclusive, qualify_game


@pytest.fixture
def bank(tmp_path):
    # Real production encoder/writer in a separate synthetic-only process.
    source = r"""
import sys
from pathlib import Path
import chess,numpy as np
from chess_anti_engine.encoding import _lc0_ext
from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.moves import move_to_index_for_encoding
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import save_local_shard_arrays,samples_to_arrays
from chess_anti_engine.natural_outcome import capture_natural_outcome
from chess_anti_engine.natural_outcome_sidecar import NaturalOutcomeSidecar
root=Path(sys.argv[1]);b=chess.Board();samples=[]
_lc0_ext.set_history_rep_fix(True)
for i,u in enumerate(['f2f3','e7e5','g2g4','d8h4']):
 if i<2:
  x=_lc0_ext.CBoard.from_board(b).encode_full(c_input_history_mode('lc0_root_legacy_meta'),63)
  policy=np.zeros(1858,dtype=np.float32);policy[move_to_index_for_encoding(next(iter(b.legal_moves)),b,policy_encoding='lc0_1858')]=1
  samples.append(ReplaySample(x=x,policy_target=policy,wdl_target=2 if b.turn else 0,game_id=123,ply_index=i,input_history_encoding='lc0_root_legacy_meta',history_rep_fix=True,search_wdl=np.array([.25,.5,.25],np.float32),is_network_turn=True,is_selfplay=True))
 b.push_uci(u)
r=capture_natural_outcome(b,resolved_result='0-1',adjudicated=False,tb_adjudicated=False,resigned=False,censored=False)
NaturalOutcomeSidecar(root/'candidates')(r,samples,post_rescore_result='0-1',tb_rescore_enabled=True)
save_local_shard_arrays(root/'game.zarr',arrs=samples_to_arrays(samples),meta={'history_rep_fix':True,'input_history_encoding':'lc0_root_legacy_meta','policy_encoding':'lc0_1858'})
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)], check=True, env=env, timeout=30
    )
    candidate = next((tmp_path / "candidates").glob("*.candidate.json"))
    return (
        tmp_path / "game.zarr",
        candidate,
        hashlib.sha256(candidate.read_bytes()).hexdigest(),
    )


def test_real_zarr_readback_full175_and_scalar_targets(bank, tmp_path):
    import numpy as np

    shard, candidate, digest = bank
    out = qualify_game(shard, candidate, digest, tmp_path)
    receipt = json.loads((out / "GAME-READBACK-COMMITTED.json").read_text())
    assert receipt["rows"] == 2
    assert receipt["all175_replayed"] is True
    assert receipt["tb_rescore_enabled_at_capture"] is True
    with np.load(out / "arm-targets.npz", allow_pickle=False) as targets:
        np.testing.assert_allclose(targets["wdl_b"][0], [0.2, 0.4, 0.4])
        assert not np.array_equal(targets["categorical_a"], targets["categorical_b"])


def test_external_candidate_pin_mismatch_never_commits(bank, tmp_path):
    shard, candidate, _ = bank
    with pytest.raises(RuntimeError, match="candidate identity"):
        qualify_game(shard, candidate, "0" * 64, tmp_path)
    assert not list(tmp_path.glob("e2-game-*/GAME-READBACK-COMMITTED.json"))


def test_full175_corruption_fails_after_valid_row_crosswalk(bank, tmp_path):
    shard, candidate, _ = bank
    # Candidate mismatch must fail before pretending plane replay qualified.
    value = json.loads(candidate.read_text())
    value["rows"][0]["stored_float16_x_sha256"] = "0" * 64
    candidate.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="crosswalk mismatch"):
        qualify_game(
            shard,
            candidate,
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            tmp_path,
        )
    assert not list(tmp_path.glob("e2-game-*/GAME-READBACK-COMMITTED.json"))


def test_commit_marker_removed_on_directory_sync_failure(tmp_path, monkeypatch):
    real = os.fsync
    calls = 0

    def fail_second(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory failure")
        return real(fd)

    monkeypatch.setattr(os, "fsync", fail_second)
    with pytest.raises(OSError, match="directory failure"):
        atomic_exclusive(tmp_path / "GAME-READBACK-COMMITTED.json", b"{}")
    assert not list(tmp_path.iterdir())


def test_all175_replay_rejects_rebound_corrupt_input(bank, tmp_path):
    from chess_anti_engine.replay.shard import (
        load_shard_arrays,
        save_local_shard_arrays,
    )

    shard, candidate, _ = bank
    arrays, meta = load_shard_arrays(shard)
    arrays["x"][0, 174, 0, 0] = 1 - arrays["x"][0, 174, 0, 0]
    changed = tmp_path / "changed.zarr"
    save_local_shard_arrays(changed, arrs=arrays, meta=meta)
    value = json.loads(candidate.read_text())
    value["rows"][0]["stored_float16_x_sha256"] = hashlib.sha256(
        arrays["x"][0].astype("<f2").tobytes()
    ).hexdigest()
    candidate.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="full175 replay mismatch"):
        qualify_game(
            changed,
            candidate,
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            tmp_path,
        )
    assert not list(tmp_path.glob("e2-game-*/GAME-READBACK-COMMITTED.json"))


def test_owned_child_interrupt_cleanup(bank, tmp_path, monkeypatch):
    import scripts.e2_bank_gate as gate

    shard, candidate, digest = bank
    calls = []

    class Child:
        pid = 987654321

        def __init__(self):
            self.first = True

        def communicate(self, timeout=None):
            assert timeout in (None, 90)
            if self.first:
                self.first = False
                raise KeyboardInterrupt()
            calls.append("reaped")
            return b"", b""

        def poll(self):
            return None

    monkeypatch.setattr(gate.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(gate.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        qualify_game(shard, candidate, digest, tmp_path)
    assert calls == [(987654321, gate.signal.SIGKILL), "reaped"]
    assert not list(tmp_path.glob("e2-game-*/GAME-READBACK-COMMITTED.json"))
