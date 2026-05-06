from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chess_anti_engine.worker import _upload_response_allows_pending_delete


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
