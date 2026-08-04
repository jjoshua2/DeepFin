"""A8: the seed-dole rearm must leave evidence of every outcome.

Before this suite, `_consume_rearm_unlocked` returned a bare ``False`` on all
four non-match paths and unlinked the file in a ``finally`` regardless, so
"blind-spot seeding fired normally" and "blind-spot seeding never fired once"
produced byte-identical (empty) evidence, and a malformed rearm destroyed the
only copy of the bytes that would have explained why.

Each test below names the path it covers and asserts on the observable
surface (log record + on-disk residue + gate counters), not on internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.server.app import (
    SEED_DOLE_REARM_FILENAME,
    _SeedDoleGate,
    consume_seed_dole_rearm,
)

_LOGGER = "chess_anti_engine.server"


def _pub(tmp_path: Path) -> Path:
    pub = tmp_path / "publish"
    pub.mkdir()
    return pub


def _write_rearm(pub: Path, payload: str) -> Path:
    path = pub / SEED_DOLE_REARM_FILENAME
    path.write_text(payload, encoding="utf-8")
    return path


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno >= level
    ]


def _quarantined(pub: Path) -> list[Path]:
    return sorted(pub.glob(SEED_DOLE_REARM_FILENAME + ".bad-*"))


# ── path 1: consumed (the happy path) ────────────────────────────────────────


def test_consumed_logs_info_with_iteration(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    pub = _pub(tmp_path)
    _write_rearm(pub, json.dumps({"training_iteration": 5}))
    gate = _SeedDoleGate()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(pub, 5, trial_key="t1") is True

    msgs = _messages(caplog, logging.INFO)
    assert any("rearm CONSUMED" in m for m in msgs), msgs
    assert any("iteration=5" in m and "trial=t1" in m for m in msgs), msgs
    assert gate.counters() == {
        "dole_rearm_consumed": 1, "dole_rearm_skipped": 0, "dole_rearm_bad": 0,
    }
    # Still one-shot: the file is gone and nothing is quarantined.
    assert not (pub / SEED_DOLE_REARM_FILENAME).exists()
    assert _quarantined(pub) == []


# ── path 2: absent (by far the most common; must stay quiet) ─────────────────


def test_absent_is_silent_and_uncounted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Every poll with no rearm file hits this; a log line here would be one
    per worker per poll and would bury the four interesting outcomes."""
    gate = _SeedDoleGate()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(_pub(tmp_path), 5) is False
    assert _messages(caplog, logging.DEBUG) == []
    assert gate.counters() == {
        "dole_rearm_consumed": 0, "dole_rearm_skipped": 0, "dole_rearm_bad": 0,
    }


# ── path 3: stale iteration ──────────────────────────────────────────────────


def test_stale_iteration_warns_with_both_numbers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    pub = _pub(tmp_path)
    _write_rearm(pub, json.dumps({"training_iteration": 9}))
    gate = _SeedDoleGate()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(pub, 5, trial_key="t1") is False

    warnings = _messages(caplog, logging.WARNING)
    assert any("rearm SKIPPED as stale" in m for m in warnings), warnings
    # Both numbers, or the operator cannot tell which side is ahead.
    assert any("iteration=9" in m and "iteration=5" in m for m in warnings), warnings
    assert gate.counters()["dole_rearm_skipped"] == 1
    assert not (pub / SEED_DOLE_REARM_FILENAME).exists()


def test_claim_carries_the_trial_key_into_the_rearm_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The trial key must survive the hop from `claim` into the consumer.

    `claim` reaches `_consume_rearm_unlocked` through `run_in_threadpool` and a
    `functools.partial`, so `trial_key` is passed at a seam the direct-call
    tests above bypass: they hand it in themselves. Drop `trial_key=trial_key`
    from that partial and every rearm line silently loses its `trial=` field --
    on a multi-trial server that is the difference between an actionable
    warning and an unattributable one. Only this test fails when it is dropped.
    """
    pub = _pub(tmp_path)
    gate = _SeedDoleGate()
    assert asyncio.run(gate.claim("trial_00007", 5, publish_dir=pub)) is True
    _write_rearm(pub, json.dumps({"training_iteration": 5}))

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert asyncio.run(gate.claim("trial_00007", 5, publish_dir=pub)) is True

    consumed = [m for m in _messages(caplog, logging.INFO) if "rearm CONSUMED" in m]
    assert len(consumed) == 1, consumed
    assert "trial=trial_00007" in consumed[0], consumed[0]


def test_stale_rearm_does_not_lose_a_later_dole(tmp_path: Path) -> None:
    """The stale branch discards a re-arm, never a claim: `claim` still grants
    a strictly newer iteration through its ordinary `iteration > last` path.
    Asserted so the WARNING above is not read as a dropped dole."""
    pub = _pub(tmp_path)
    gate = _SeedDoleGate()
    assert asyncio.run(gate.claim("t", 5, publish_dir=pub)) is True
    _write_rearm(pub, json.dumps({"training_iteration": 3}))  # stale
    assert asyncio.run(gate.claim("t", 5, publish_dir=pub)) is False  # no re-open
    assert asyncio.run(gate.claim("t", 6, publish_dir=pub)) is True  # next iter fine


# ── path 4: malformed (the destroyed-evidence case) ──────────────────────────


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        ("{not json", "truncated-json"),
        ('{"training_iteration": "twelve"}', "wrong-type"),
        ("[1, 2, 3]", "wrong-shape"),
    ],
)
def test_malformed_is_quarantined_not_destroyed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: str, label: str,
) -> None:
    pub = _pub(tmp_path)
    _write_rearm(pub, payload)
    gate = _SeedDoleGate()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(pub, 5, trial_key="t1") is False

    warnings = _messages(caplog, logging.WARNING)
    assert any("rearm MALFORMED" in m for m in warnings), (label, warnings)
    assert gate.counters()["dole_rearm_bad"] == 1

    # The original name is cleared (so the next poll is not stuck on it) but
    # the BYTES survive -- this is the whole point of the change.
    assert not (pub / SEED_DOLE_REARM_FILENAME).exists()
    kept = _quarantined(pub)
    assert len(kept) == 1, (label, sorted(p.name for p in pub.iterdir()))
    assert kept[0].read_text(encoding="utf-8") == payload
    # And the log says where it went, so the operator does not have to guess.
    assert any(kept[0].name in m for m in warnings), warnings


def test_quarantine_is_still_one_shot(tmp_path: Path) -> None:
    """A malformed file must not be re-read forever: keeping the bytes must not
    resurrect the pre-#209 double-dole risk."""
    pub = _pub(tmp_path)
    _write_rearm(pub, "{not json")
    gate = _SeedDoleGate()
    assert gate._consume_rearm_unlocked(pub, 5) is False
    assert gate._consume_rearm_unlocked(pub, 5) is False
    assert gate.counters()["dole_rearm_bad"] == 1  # second call saw nothing


def test_quarantine_failure_falls_back_to_delete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the quarantine rename itself fails the file must not be left behind
    under `.consuming`, where the next poll would never see it and never clean
    it up either."""
    pub = _pub(tmp_path)
    _write_rearm(pub, "{not json")
    real_rename = pathlib.Path.rename

    def _fail_on_quarantine(self: Path, target: Any) -> Path:
        if ".bad-" in str(target):
            raise OSError(28, "No space left on device")
        return real_rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", _fail_on_quarantine)
    gate = _SeedDoleGate()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(pub, 5) is False

    warnings = _messages(caplog, logging.WARNING)
    assert any("quarantine failed" in m for m in warnings), warnings
    assert gate.counters()["dole_rearm_bad"] == 1
    assert sorted(p.name for p in pub.iterdir()) == []


# ── path 5: unreadable (claim rename fails for a reason that is not "gone") ──


def test_unreadable_is_distinguished_from_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permissions/ENOSPC failure to claim the file used to return the same
    bare False as "there is no rearm file", so a rearm that could never be
    consumed looked exactly like a rearm that was never written."""
    pub = _pub(tmp_path)
    _write_rearm(pub, json.dumps({"training_iteration": 5}))
    real_rename = pathlib.Path.rename

    def _fail_on_claim(self: Path, target: Any) -> Path:
        if str(target).endswith(".consuming"):
            raise PermissionError(13, "Permission denied")
        return real_rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", _fail_on_claim)
    gate = _SeedDoleGate()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert gate._consume_rearm_unlocked(pub, 5) is False

    warnings = _messages(caplog, logging.WARNING)
    assert any("rearm UNREADABLE" in m for m in warnings), warnings
    assert any("PermissionError" in m for m in warnings), warnings
    assert gate.counters()["dole_rearm_bad"] == 1
    # Not consumed: the file is untouched and a later poll can still win it.
    assert (pub / SEED_DOLE_REARM_FILENAME).exists()


# ── the standalone helper shares the implementation ──────────────────────────


def test_helper_and_gate_share_one_implementation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """`consume_seed_dole_rearm` was a verbatim copy of the gate's body; the
    two could drift so that the tested path and the served path disagreed."""
    pub = _pub(tmp_path)
    _write_rearm(pub, "{not json")
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert consume_seed_dole_rearm(pub, 5) is False
    assert any("rearm MALFORMED" in m for m in _messages(caplog, logging.WARNING))
    assert len(_quarantined(pub)) == 1


# ── the grant itself: the line that proves seeding fired ─────────────────────


def test_manifest_grant_logs_iteration_and_seed_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """End of the chain: an operator greps ONE line to learn that trial X was
    doled N seeds at iteration I. The rearm payload carries only
    `training_iteration` (writer: `distributed_runtime._publish_...`), so the
    count has to come from the reco at the grant site, not from the file.

    Helpers are imported from the sibling dole suite rather than duplicated:
    the publish path they drive is exactly the one production uses.
    """
    from chess_anti_engine.server.app import create_app
    from chess_anti_engine.worker import _manifest_poll_headers
    from tests.test_distributed_selfplay_backpressure import (
        _DOLE_SEED_FEN,
        _poll_app_n,
        _publish_dole_trial,
    )

    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=11, dole=3, fen_path=fen_path)
    app = create_app(server_root=tmp_path)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        polls = _poll_app_n(
            app, "/v1/trials/trial_00000/manifest",
            headers=_manifest_poll_headers(worker_id="w"), n=3,
        )

    assert [p["dole_fen_seeds"] for p in polls] == [True, False, False]
    granted = [m for m in _messages(caplog, logging.INFO) if "dole GRANTED" in m]
    # Exactly one line per iteration per trial -- not one per poll.
    assert len(granted) == 1, granted
    assert "iteration=11" in granted[0], granted[0]
    assert "seeds=3" in granted[0], granted[0]
    assert "trial=trial_00000" in granted[0], granted[0]
    # Running totals ride along so one grep answers "and how has rearm behaved".
    assert "dole_rearm_consumed=0" in granted[0], granted[0]


def test_manifest_declines_are_not_logged_as_grants(tmp_path: Path,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    """A trial that declines at the FIRST guard logs nothing at all.

    What this pins, precisely: `opening_fen_dole_per_iter == 0` returns before
    `claim` is ever called, so an operator who greps a trial and finds no
    `dole GRANTED` line can read that as "doling is off for this trial", not as
    "the grant site is reached but silent".

    It does NOT kill an `if granted:` -> `if True:` mutation -- these polls
    never reach that line. That mutation is killed by the `len(granted) == 1`
    assertion in the test above: with doling ON, polls 2 and 3 DO reach the
    line with `granted=False`, so an unconditional log would emit three.
    """
    from chess_anti_engine.server.app import create_app
    from chess_anti_engine.worker import _manifest_poll_headers
    from tests.test_distributed_selfplay_backpressure import (
        _DOLE_SEED_FEN,
        _poll_app_n,
        _publish_dole_trial,
    )

    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=12, dole=0, fen_path=fen_path)
    app = create_app(server_root=tmp_path)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        polls = _poll_app_n(
            app, "/v1/trials/trial_00000/manifest",
            headers=_manifest_poll_headers(worker_id="w"), n=3,
        )

    assert [p["dole_fen_seeds"] for p in polls] == [False, False, False]
    assert [m for m in _messages(caplog, logging.INFO) if "dole GRANTED" in m] == []
