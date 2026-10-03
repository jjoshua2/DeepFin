"""Upload retries must stay idempotent after compaction and server restart."""

from __future__ import annotations
import hashlib
import json
import io
import shutil
from pathlib import Path
from typing import Any
from fastapi.testclient import TestClient
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.server.app import create_app
from chess_anti_engine.server.auth import UserRecord, hash_password, save_users
from chess_anti_engine.version import PROTOCOL_VERSION, UPLOAD_CONTENT_SHA256_HEADER
from tests.test_server_upload_security import _sample

TRIAL = "trial_00000"


def _payload(tmp_path: Path, *, name: str, generated_at: int) -> tuple[str, bytes]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard = tmp_path / f"{name}.zarr"
    save_local_shard_arrays(
        shard,
        arrs=samples_to_arrays([_sample(1), _sample(2)]),
        meta=ShardMeta(
            username="u",
            run_id=TRIAL,
            games=1,
            positions=2,
            model_sha256="f" * 64,
            model_step=7,
            generated_at_unix=generated_at,
        ),
    )
    filename, stream = pack_shard_for_upload(shard)
    try:
        return filename, stream.getvalue()
    finally:
        stream.close()


def _seed_user(root: Path) -> None:
    salt, pw_hash, iterations = hash_password("fixture-only")
    save_users(
        root / "users.json",
        {
            "u": UserRecord(
                username="u", salt_b64=salt, hash_b64=pw_hash, iterations=iterations
            )
        },
    )


def _post(
    client: TestClient,
    filename: str,
    payload: bytes,
    *,
    declared_sha: str | None = None,
):
    digest = declared_sha or hashlib.sha256(payload).hexdigest()
    return client.post(
        f"/v1/trials/{TRIAL}/upload_shard",
        auth=("u", "fixture-only"),
        files={"file": (filename, io.BytesIO(payload), "application/x-tar")},
        headers={
            "X-CAE-Worker-Version": "0.0.0",
            "X-CAE-Protocol-Version": str(PROTOCOL_VERSION),
            UPLOAD_CONTENT_SHA256_HEADER: digest,
        },
    )


def _accepted_positions(root: Path) -> int:
    trial_root = root / "trials" / TRIAL
    shards = list((trial_root / "inbox" / "_compacted").glob("*.zarr"))
    shards += list((trial_root / "processed" / "_compacted").glob("*.zarr"))
    return sum(int(load_shard_arrays(path)[0]["x"].shape[0]) for path in shards)


def test_retry_after_compaction_and_restart_dedupes_payload_not_rows(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    filename, original = _payload(
        tmp_path / "original", name="original", generated_at=100
    )
    filename2, same_rows_new_upload = _payload(
        tmp_path / "different", name="different", generated_at=101
    )
    assert (
        hashlib.sha256(original).digest()
        != hashlib.sha256(same_rows_new_upload).digest()
    )
    app_args: dict[str, Any] = {
        "server_root": root,
        "users_db": "users.json",
        "upload_compact_shard_size": 2,
        "upload_compact_max_age_seconds": 3600,
    }
    from chess_anti_engine.server import app as server_app

    real_atomic_write_text = server_app.atomic_write_text
    receipt_failures = 2

    def fail_receipt_twice(path, text, **kwargs):
        nonlocal receipt_failures
        if Path(path).parent.name == "_upload_receipts" and receipt_failures:
            receipt_failures -= 1
            raise OSError("injected receipt fsync failure")
        return real_atomic_write_text(path, text, **kwargs)

    monkeypatch.setattr(server_app, "atomic_write_text", fail_receipt_twice)
    with TestClient(create_app(**app_args)) as client:
        # Worker identity is SHA-256 of the exact packed tar bytes. A payload
        # altered in transit while retaining the old digest must be rejected.
        conflict = _post(
            client,
            filename2,
            same_rows_new_upload,
            declared_sha=hashlib.sha256(original).hexdigest(),
        )
        assert conflict.status_code == 422, conflict.text
        assert _accepted_positions(root) == 0
        first = _post(client, filename, original)
        assert first.status_code == 200, first.text
        assert first.json()["stored"] is True
        assert receipt_failures == 1
        assert _accepted_positions(root) == 2
        inbox_compacted = root / "trials" / TRIAL / "inbox" / "_compacted"
        processed_compacted = root / "trials" / TRIAL / "processed" / "_compacted"
        processed_compacted.mkdir(parents=True, exist_ok=True)
        for shard in inbox_compacted.glob("*.zarr"):
            shutil.move(str(shard), processed_compacted / shard.name)
        assert _accepted_positions(root) == 2
    with TestClient(create_app(**app_args)) as restarted:
        assert receipt_failures == 0
        retry = _post(restarted, filename, original)
        assert retry.status_code == 200, retry.text
        assert retry.json()["stored"] is False
        assert _accepted_positions(root) == 2
    monkeypatch.setattr(server_app, "atomic_write_text", real_atomic_write_text)
    with TestClient(create_app(**app_args)) as recovered:
        assert _accepted_positions(root) == 2
        distinct = _post(recovered, filename2, same_rows_new_upload)
        assert distinct.status_code == 200, distinct.text
        assert distinct.json()["stored"] is True
        assert _accepted_positions(root) == 4


def test_receipt_expiry_uses_fixed_six_hour_wall_clock_window(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _seed_user(root)
    filename, payload = _payload(
        tmp_path / "original", name="original", generated_at=100
    )
    app_args: dict[str, Any] = {
        "server_root": root,
        "users_db": "users.json",
        "upload_compact_shard_size": 2,
        "upload_compact_max_age_seconds": 3600,
    }
    with TestClient(create_app(**app_args)) as client:
        accepted = _post(client, filename, payload)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["stored"] is True

    from chess_anti_engine.server import app as server_app

    receipt = (
        root
        / "trials"
        / TRIAL
        / "inbox"
        / "_upload_receipts"
        / f"{hashlib.sha256(payload).hexdigest()}.json"
    )
    assert receipt.is_file()
    receipt.write_text('{"seen_at_unix": 1000.0}', encoding="utf-8")

    # Retry just before the existing six-hour expiry: still a duplicate, and
    # retries do not extend the original receipt timestamp.
    monkeypatch.setattr(server_app.time, "time", lambda: 1000.0 + 6.0 * 3600.0 - 1.0)
    with TestClient(create_app(**app_args)) as before_expiry:
        retry = _post(before_expiry, filename, payload)
        assert retry.status_code == 200, retry.text
        assert retry.json()["stored"] is False
    assert json.loads(receipt.read_text(encoding="utf-8"))["seen_at_unix"] == 1000.0

    # Wall-clock rollback extends the suppression window because this matches
    # the pre-existing time.time-based cache; a forward jump beyond six hours
    # expires the receipt at next startup, matching that same contract.
    monkeypatch.setattr(server_app.time, "time", lambda: 900.0)
    with TestClient(create_app(**app_args)) as rolled_back:
        retry = _post(rolled_back, filename, payload)
        assert retry.status_code == 200, retry.text
        assert retry.json()["stored"] is False
    assert receipt.is_file()

    monkeypatch.setattr(server_app.time, "time", lambda: 1000.0 + 6.0 * 3600.0 + 1.0)
    with TestClient(create_app(**app_args)) as after_expiry:
        assert not receipt.exists()
        retry = _post(after_expiry, filename, payload)
        assert retry.status_code == 200, retry.text
        assert retry.json()["stored"] is True
        assert _accepted_positions(root) == 4
