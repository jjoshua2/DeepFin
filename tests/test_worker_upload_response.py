from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import ShardMeta, save_local_shard_arrays
from chess_anti_engine.worker import WorkerSession, _upload_response_allows_pending_delete


@dataclass
class _Resp:
    status_code: int
    body: Any

    def json(self) -> Any:
        if isinstance(self.body, BaseException):
            raise self.body
        return self.body


def test_upload_response_deletes_after_stored_true():
    assert _upload_response_allows_pending_delete(
        _Resp(200, {"stored": True, "positions": 10}),
    )


def test_upload_response_deletes_after_deduped_not_rejected():
    assert _upload_response_allows_pending_delete(
        _Resp(200, {"stored": False, "sha256": "abc"}),
    )


def test_upload_response_keeps_rejected_200_response():
    assert not _upload_response_allows_pending_delete(
        _Resp(200, {"stored": False, "rejected": True, "reason": "protocol mismatch"}),
    )


def test_upload_response_keeps_non_200_or_invalid_body():
    assert not _upload_response_allows_pending_delete(_Resp(503, {"stored": True}))
    assert not _upload_response_allows_pending_delete(_Resp(200, ValueError("bad json")))
    assert not _upload_response_allows_pending_delete(_Resp(200, ["not", "a", "dict"]))


class _Requests:
    def __init__(self, response: _Resp) -> None:
        self.response = response
        self.calls = 0

    def post(self, *_args, **_kwargs) -> _Resp:
        self.calls += 1
        return self.response


def _minimal_session_for_arena_upload(tmp_path, response: _Resp) -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.server = "http://server"
    session.trial_api_prefix = "/v1"
    session.leased_trial_id = "trial_a"
    session.fixed_trial_id = ""
    session._auth = ("u", "p")
    session._requests = _Requests(response)
    session.log = logging.getLogger("test.worker_upload_response")
    session.arena_pending_dir = tmp_path / "arena" / "pending"
    session.arena_uploaded_dir = tmp_path / "arena" / "uploaded"
    session.arena_pending_dir.mkdir(parents=True)
    session.arena_uploaded_dir.mkdir(parents=True)
    return session


def _minimal_session_for_shard_upload(tmp_path, response: _Resp) -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.server = "http://server"
    session.trial_api_prefix = "/v1/trials/trial_b"
    session.leased_trial_id = "trial_b"
    session.fixed_trial_id = ""
    session.lease_id = "lease"
    session.machine_id = "machine"
    session.args = SimpleNamespace(username="u", password="p")
    session._requests = _Requests(response)
    session.log = logging.getLogger("test.worker_upload_response")
    session.pending_dir = tmp_path / "shards" / "pending"
    session.pending_dir.mkdir(parents=True)
    session.last_successful_send_s = 0.0
    return session


def _write_tagged_shard(path, *, run_id: str) -> None:
    policy = np.zeros((1, 4672), dtype=np.float32)
    policy[0, 0] = 1.0
    arrs = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.zeros((1,), dtype=np.int64),
        "priority": np.ones((1,), dtype=np.float32),
        "has_policy": np.ones((1,), dtype=np.uint8),
    }
    save_local_shard_arrays(
        path,
        arrs=arrs,
        meta=ShardMeta(username="u", run_id=run_id, positions=1, model_sha256="abc", model_step=1),
    )


def test_arena_upload_keeps_rejected_200_response(tmp_path):
    session = _minimal_session_for_arena_upload(
        tmp_path,
        _Resp(200, {"stored": False, "rejected": True, "reason": "protocol mismatch"}),
    )
    pending = session.arena_pending_dir / "result.json"
    pending.write_text('{"games": 1}', encoding="utf-8")

    session._upload_pending_arena_results()

    assert pending.exists()


def test_arena_upload_deletes_after_stored_true(tmp_path):
    session = _minimal_session_for_arena_upload(
        tmp_path,
        _Resp(200, {"stored": True, "sha256": "abc"}),
    )
    pending = session.arena_pending_dir / "result.json"
    pending.write_text('{"games": 1}', encoding="utf-8")

    session._upload_pending_arena_results()

    assert not pending.exists()


def test_arena_upload_skips_other_trial_pending_result(tmp_path):
    session = _minimal_session_for_arena_upload(
        tmp_path,
        _Resp(200, {"stored": True, "sha256": "abc"}),
    )
    pending = session.arena_pending_dir / "result.json"
    pending.write_text('{"games": 1, "trial_id": "trial_b"}', encoding="utf-8")

    session._upload_pending_arena_results()

    assert pending.exists()
    assert session._requests.calls == 0


def test_shard_upload_skips_other_trial_pending_shard(tmp_path):
    session = _minimal_session_for_shard_upload(
        tmp_path,
        _Resp(200, {"stored": True, "sha256": "abc"}),
    )
    pending = session.pending_dir / "old_trial.zarr"
    _write_tagged_shard(pending, run_id="trial_a")

    uploaded_at = session._upload_pending_shards(default_elapsed_s=0.0)

    assert uploaded_at is None
    assert pending.exists()
    assert session._requests.calls == 0
