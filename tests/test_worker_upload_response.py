from __future__ import annotations

import logging
import threading
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
    session._pending_upload_lock = threading.Lock()
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


def test_shard_upload_quarantines_rejected_200_response(tmp_path):
    session = _minimal_session_for_shard_upload(
        tmp_path,
        _Resp(200, {"stored": False, "rejected": True, "reason": "invalid shard"}),
    )
    pending = session.pending_dir / "bad.zarr"
    _write_tagged_shard(pending, run_id="trial_b")

    uploaded_at = session._upload_pending_shards(default_elapsed_s=0.0)

    assert uploaded_at is None
    assert not pending.exists()
    quarantined = list((tmp_path / "shards" / "corrupt").glob("bad.zarr*"))
    assert len([p for p in quarantined if p.is_dir()]) == 1
    reason_files = [p for p in quarantined if p.name.endswith(".reason.txt")]
    assert len(reason_files) == 1
    assert "invalid shard" in reason_files[0].read_text(encoding="utf-8")
    assert session._requests.calls == 1


def test_shard_upload_sends_locally_invalid_shard_for_server_quarantine(tmp_path):
    session = _minimal_session_for_shard_upload(
        tmp_path,
        _Resp(200, {"stored": False, "rejected": True, "reason": "invalid shard"}),
    )
    pending = session.pending_dir / "zero.zarr"
    pending.mkdir()
    (pending / ".zgroup").write_bytes(b"")

    uploaded_at = session._upload_pending_shards(default_elapsed_s=0.0)

    assert uploaded_at is None
    assert not pending.exists()
    quarantined = list((tmp_path / "shards" / "corrupt").glob("zero.zarr*"))
    assert len([p for p in quarantined if p.is_dir()]) == 1
    reason_files = [p for p in quarantined if p.name.endswith(".reason.txt")]
    assert len(reason_files) == 1
    assert "invalid shard" in reason_files[0].read_text(encoding="utf-8")
    assert session._requests.calls == 1


# ── #344 review finding A, worker half: a stalled poll must not be silent ────


def _poll_session(responses: list[_Resp]) -> Any:
    """A `WorkerSession` reduced to what `_poll_manifest`'s non-200 path reads.

    Built with `object.__new__` (the `_bare_session` idiom already used in
    `test_reco_coverage.py`) because a real session needs a server, a model and
    a GPU. The branch under test runs before any of that.
    """
    from chess_anti_engine.worker import WorkerSession

    session: Any = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_poll")
    session.args = SimpleNamespace(poll_seconds=0.0)
    session._manifest_poll_failures = 0
    session.manifest_state = "active"
    session.manifest_state_elapsed_s = None
    session.worker_id = "w"
    session.lease_id = ""
    session.fixed_trial_id = "trial_00000"
    session.leased_trial_id = None
    session.trial_api_prefix = "/v1/trials/trial_00000"
    session.inference_client = object()
    session.machine_id = "m"
    session.cfg = {}
    session._upload_pending_shards = lambda **_kw: None
    session._upload_pending_arena_results = lambda: None
    session._server_url_for = lambda p: "http://server" + str(p)
    session._requests = SimpleNamespace(get=lambda *_a, **_k: responses.pop(0))
    return session


def test_a_failed_manifest_poll_is_logged(caplog) -> None:
    """The server answers an undecidable compat gate with 503 so the worker
    keeps its shard and retries. That is correct — but this branch used to
    `time.sleep` and return in COMPLETE silence, so a fleet stalled behind a
    corrupt manifest was invisible from the worker end as well as the server
    end. A worker that cannot poll is a worker doing nothing.
    """
    from chess_anti_engine.worker import WorkerSession

    session = _poll_session([_Resp(503, {"detail": "cannot read manifest"})])
    with caplog.at_level(logging.INFO, logger="test.worker_poll"):
        assert WorkerSession._poll_manifest(session) is None

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("manifest poll returned HTTP 503" in m for m in msgs), msgs
    assert any("no selfplay runs until it succeeds" in m for m in msgs), msgs


def test_repeated_poll_failures_do_not_flood_but_keep_reporting(caplog) -> None:
    """Same cadence as the server side: first, then every 100th. One line per
    poll would be a flood; one line ever would be the defect."""
    from chess_anti_engine.worker import WorkerSession

    session = _poll_session([_Resp(503, {}) for _ in range(250)])
    with caplog.at_level(logging.INFO, logger="test.worker_poll"):
        for _ in range(250):
            assert WorkerSession._poll_manifest(session) is None

    failures = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and "manifest poll returned" in r.getMessage()
    ]
    assert len(failures) == 3, len(failures)   # 1, 100, 200
    assert "consecutive failures: 200" in failures[-1], failures[-1]


def test_a_recovered_poll_says_so(caplog) -> None:
    """Without this, the log shows a fleet going down and never coming back —
    the operator cannot tell a resolved incident from an ongoing one."""
    from chess_anti_engine.worker import WorkerSession

    session = _poll_session([
        _Resp(503, {}),
        _Resp(200, {"training_iteration": 1}),
    ])
    session._check_manifest_compat = lambda _m: SimpleNamespace(
        protocol_mismatch=False, version_too_old=False,
        req_proto=None, min_worker_version=None,
    )
    session._maybe_self_update_from_manifest = lambda _m, _c: None
    session._check_pause_selfplay = lambda _m: False

    with caplog.at_level(logging.INFO, logger="test.worker_poll"):
        assert WorkerSession._poll_manifest(session) is None
        assert WorkerSession._poll_manifest(session) is not None

    msgs = [r.getMessage() for r in caplog.records]
    assert any("recovered after 1 consecutive failure" in m for m in msgs), msgs
    assert session._manifest_poll_failures == 0


def test_a_healthy_poll_logs_nothing(caplog) -> None:
    """Negative control. Every worker polls this route constantly; a line on
    the success path would be the flood the cadence exists to avoid, and would
    make the failure lines above unfindable."""
    from chess_anti_engine.worker import WorkerSession

    session = _poll_session([_Resp(200, {"training_iteration": 1}) for _ in range(5)])
    session._check_manifest_compat = lambda _m: SimpleNamespace(
        protocol_mismatch=False, version_too_old=False,
        req_proto=None, min_worker_version=None,
    )
    session._maybe_self_update_from_manifest = lambda _m, _c: None
    session._check_pause_selfplay = lambda _m: False

    with caplog.at_level(logging.DEBUG, logger="test.worker_poll"):
        for _ in range(5):
            assert WorkerSession._poll_manifest(session) is not None
    assert [r.getMessage() for r in caplog.records] == []
