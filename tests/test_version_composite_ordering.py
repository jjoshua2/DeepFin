"""RS4 subset: nested PEP 440 stages, without changing malformed-input policy."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from chess_anti_engine.utils.versioning import parse_version, version_lt


# PyPA's nested-stage ladder, restricted to the syntax this helper already
# accepts. This does not add epochs, local-label ordering or extra components.
_LADDER = [
    "1.dev0", "1.0.dev456", "1.0a1", "1.0a2.dev456", "1.0a12.dev456",
    "1.0a12", "1.0b1.dev456", "1.0b2", "1.0b2.post345.dev456",
    "1.0b2.post345", "1.0rc1.dev456", "1.0rc1", "1.0",
    "1.0.post456.dev34", "1.0.post456", "1.0.15", "1.1.dev1",
]


def test_all_pairs_in_composite_stage_ladder() -> None:
    for i, older in enumerate(_LADDER):
        for j, newer in enumerate(_LADDER):
            assert version_lt(older, newer) is (i < j), (older, newer)


@pytest.mark.parametrize(("older", "newer"), [
    ("1.0a1", "1.0b1.dev1"), ("1.0", "1.0.post1.dev1"),
    ("1.0rc1", "1.0rc1.post1"), ("1.0a1", "1.0a2.dev0"),
    ("1.0rc1.dev99", "1.0rc1"), ("1.0rc1.post1.dev99", "1.0rc1.post1"),
    ("1.0rc1.post1", "1.0rc1.post2.dev0"), ("1.0.post1.dev99", "1.0.post1"),
    ("1.0.post1", "1.0.post2.dev0"), ("1.0rc1.post999", "1.0rc2.dev0"),
    ("1.0.post1.dev1", "1.0.post1.dev2"), ("1.0rc1.dev1", "1.0rc1.dev2"),
])
def test_nested_stage_order(older: str, newer: str) -> None:
    assert version_lt(older, newer) is True
    assert version_lt(newer, older) is False
    assert version_lt(older, older) is False


@pytest.mark.parametrize(("a", "b"), [
    ("v1.0_rc1.post_2.dev-3", "1.0rc1.post2.dev3"),
    ("1.0beta.post.dev", "1.0b0.post0.dev0"),
    ("1.0preview1-rev2-dev3", "1.0rc1.post2.dev3"),
    ("1.0rc1.post2.dev3+g123", "1.0rc1.post2.dev3+other456"),
])
def test_existing_aliases_and_ignored_local_labels(a: str, b: str) -> None:
    assert parse_version(a) == parse_version(b) == (1, 0, 0)
    assert version_lt(a, b) is False
    assert version_lt(b, a) is False


# Malformed digit extraction remains intentionally UNFIXED. Capture actual
# old release triples, rather than accidentally normalizing these to zero.
_FALLBACKS = [
    ("garbage9.9.9", (9, 9, 9)), ("build 99", (99, 0, 0)),
    ("1.2.3junk", (1, 2, 3)), ("1..2", (1, 2, 0)),
    ("1.0-", (1, 0, 0)), ("1.0rc1oops9", (1, 0, 1)),
    ("1!0", (1, 0, 0)), ("garbage", (0, 0, 0)), ("", (0, 0, 0)),
]


def test_malformed_fallback_matrix_is_unchanged_for_plain_releases() -> None:
    samples = [*_FALLBACKS, ("0.0.0", (0, 0, 0)), ("0.0.2", (0, 0, 2)),
               ("1.0.0", (1, 0, 0)), ("9.9.9", (9, 9, 9))]
    for a, release_a in samples:
        assert parse_version(a) == release_a
        for b, release_b in samples:
            assert version_lt(a, b) is (release_a < release_b), (a, b)


@pytest.mark.parametrize("composite", [
    "1.0b1.dev1", "1.0.post1.dev1", "1.0rc1.post1", "1.0rc2.dev0",
])
def test_mixed_composite_order_uses_the_same_existing_fallback_release(composite: str) -> None:
    for malformed, release in _FALLBACKS:
        fallback = ".".join(map(str, release))
        assert version_lt(composite, malformed) == version_lt(composite, fallback)
        assert version_lt(malformed, composite) == version_lt(fallback, composite)


@pytest.mark.parametrize(("actual", "minimum", "too_old"), [
    ("1.0", "1.0.post1.dev1", True), ("1.0rc1", "1.0rc1.post1", True),
    ("1.0b1.dev1", "1.0a1", False), ("1.0.post1.dev1", "1.0", False),
    ("1.0.post1.dev1", "garbage1.0", False),
    ("garbage1.0", "1.0.post1.dev1", True),
    # Unfixed malformed-input behavior is deliberate, not claimed safe.
    ("garbage9.9.9", "1.0", False), ("garbage", "0.0", False),
    ("0.0.2", "garbage9.9.9", True), ("0.0.2", "garbage", False),
    ("1.0", "1.0", False), ("0.0.0", "0.0.0", False),
])
def test_real_server_and_worker_compat_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    actual: str, minimum: str, too_old: bool,
) -> None:
    from fastapi.testclient import TestClient

    import chess_anti_engine.worker as worker
    from chess_anti_engine.server.app import create_app

    publish = tmp_path / "publish"
    publish.mkdir()
    (publish / "manifest.json").write_text(
        json.dumps({"min_worker_version": minimum}), encoding="utf-8",
    )
    # In-process request only: no server, worker session, engine, or model starts.
    client = TestClient(create_app(server_root=str(tmp_path), users_db="users.json"))
    response = client.get("/v1/manifest", headers={"X-CAE-Worker-Version": actual})
    assert response.status_code == (426 if too_old else 200), response.text
    monkeypatch.setattr(worker, "PACKAGE_VERSION", actual)
    compat = worker.WorkerSession._check_manifest_compat(
        object.__new__(worker.WorkerSession), {"min_worker_version": minimum},
    )
    assert compat.version_too_old is too_old


@pytest.mark.parametrize(("installed", "wheel", "required", "update"), [
    ("1.0", "1.0.post1.dev1", False, True),
    ("1.0b1.dev1", "1.0a1", False, False),
    ("1.0rc1", "1.0rc1.post1", False, True),
    ("1.0", "1.0", False, False),
    ("0.0.2", "garbage9.9.9", False, True),
    ("0.0.2", "garbage", False, False),
    ("0.0.2", "garbage", True, True),
])
def test_real_wheel_update_decision_without_installing(
    monkeypatch: pytest.MonkeyPatch, installed: str, wheel: str,
    required: bool, update: bool,
) -> None:
    import chess_anti_engine.worker as worker

    session = object.__new__(worker.WorkerSession)
    install = Mock()
    monkeypatch.setattr(worker, "PACKAGE_VERSION", installed)
    monkeypatch.setattr(session, "_self_update_enabled", lambda: True)
    monkeypatch.setattr(session, "_install_worker_wheel", install)
    manifest = {"worker_wheel": {"version": wheel, "endpoint": "/not-requested", "sha256": "stub"}}
    compat = worker._ManifestCompat(False, required, None, None)
    session._maybe_self_update_from_manifest(manifest, compat)
    assert install.call_count == int(update)
