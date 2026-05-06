from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

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
    session._auth = ("u", "p")
    session._requests = _Requests(response)
    session.log = logging.getLogger("test.worker_upload_response")
    session.arena_pending_dir = tmp_path / "arena" / "pending"
    session.arena_uploaded_dir = tmp_path / "arena" / "uploaded"
    session.arena_pending_dir.mkdir(parents=True)
    session.arena_uploaded_dir.mkdir(parents=True)
    return session


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
