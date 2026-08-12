"""The WORKER half of the upload digest.

⚑ WHY THIS FILE EXISTS. The server-side tests prove the server VERIFIES the
header; they say nothing about whether the worker COMPUTES it correctly, and
that is the half that can take the fleet down. A worker that hashes the wrong
bytes 422s on every upload, forever, for every worker in the fleet -- and the
worker's own handling of a 422 is silent: `_upload_response_allows_pending_delete`
returns False, `_upload_response_rejection_reason` returns None, and the loop
`break`s WITHOUT logging. The only signal anywhere is a server-side WARNING,
while the pending dir grows without bound and ingest goes to zero. Silent upload
wedges are this project's recurring outage class.

So the assertion that matters is not "a header was sent" but "the server
ACCEPTED it", which is only true when the two digests agree over the same bytes.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

from chess_anti_engine.replay.shard import (
    ShardMeta,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.version import UPLOAD_CONTENT_SHA256_HEADER
from chess_anti_engine.worker import WorkerSession

from .test_server_upload_security import _build_client, _sample, _seed_user


def _worker_stub(work_dir: Path, client, seen: list[dict]):
    """Bind the real unbound method to the minimum state it touches.

    Constructing a whole `WorkerSession` would boot selfplay; the upload loop
    only reads these attributes, so this exercises the PRODUCTION method rather
    than a reimplementation of it.
    """
    pending = work_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    class _Requests:
        @staticmethod
        def post(url, files=None, auth=None, headers=None, timeout=None, json=None):  # noqa: ARG004  # pyright: ignore[reportUnusedParameter]
            seen.append(dict(headers or {}))
            path = str(url).replace("http://server", "")
            if json is not None:
                return client.post(path, json=json, auth=auth, headers=headers)
            return client.post(path, files=files, auth=auth, headers=headers)

    w = types.SimpleNamespace(
        pending_dir=pending,
        leased_trial_id="",
        fixed_trial_id="",
        trial_api_prefix="/v1",
        lease_id="",
        machine_id="m1",
        log=logging.getLogger("test_worker_upload_digest"),
        last_successful_send_s=0.0,
        args=types.SimpleNamespace(username="u", password="p"),
        _requests=_Requests,
    )
    w._server_url_for = lambda p: "http://server" + p
    w._report_bad_pending_shard = lambda payload: None
    return w


def _write_pending_shard(w, n: int = 2) -> Path:
    zp = w.pending_dir / "shard_000001.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(n)]),
        meta=ShardMeta(
            username="u", games=1, positions=n, model_sha256="abc1234567", model_step=0,
        ),
    )
    return zp


def test_worker_digest_is_accepted_by_the_server(tmp_path) -> None:
    """End to end through the real upload method: worker hashes, server agrees.

    MUTATION THIS CATCHES: hash the wrong bytes in `worker.py` -- e.g.
    `payload.getbuffer()[:-1]` -- and the shard stays pending because the server
    422s. Every server-side test still passes under that mutation, and so does
    repo-wide lint.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    seen: list[dict] = []
    w = _worker_stub(tmp_path / "w", client, seen)
    shard = _write_pending_shard(w)

    WorkerSession._upload_pending_shards_locked(w, default_elapsed_s=1.0)  # pyright: ignore[reportArgumentType]

    assert seen, "worker never POSTed"
    sent = seen[0]
    assert UPLOAD_CONTENT_SHA256_HEADER in sent, (
        f"worker did not send the digest header: {sorted(sent)}"
    )
    # ⚑ THE LOAD-BEARING ASSERTION. "a header was sent" passes with a wrong
    # digest; the pending shard only disappears when the server ACCEPTED the
    # upload, which requires the two digests to agree over the same bytes.
    assert not shard.exists(), (
        "pending shard was not accepted -- worker and server disagree on the digest"
    )


def test_worker_digest_covers_the_exact_bytes_uploaded(tmp_path) -> None:
    """The digest must be over what is SENT, not over a re-read of the shard.

    `pack_shard_for_upload` returns a fresh BytesIO each call and tars a
    directory, so a digest computed from a second pack would differ if tar
    metadata (mtimes) moved -- and would then be wrong exactly when it matters.
    """
    import hashlib

    from chess_anti_engine.replay.shard import pack_shard_for_upload

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    seen: list[dict] = []
    w = _worker_stub(tmp_path / "w", client, seen)
    shard = _write_pending_shard(w)
    _, payload = pack_shard_for_upload(shard)
    expected = hashlib.sha256(payload.getvalue()).hexdigest()
    payload.close()

    WorkerSession._upload_pending_shards_locked(w, default_elapsed_s=1.0)  # pyright: ignore[reportArgumentType]

    assert seen[0][UPLOAD_CONTENT_SHA256_HEADER] == expected
    # And the server independently agreed -- it echoes the sha it computed.
    assert not shard.exists()
