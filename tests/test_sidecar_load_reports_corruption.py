"""A corrupt resume sidecar must not look exactly like a missing one.

`load_optional_json` backs every piece of restored trial state: PID state, RNG
state, trial metadata, the gate counter and the best-loss record. All eight
call sites do the same thing with `None` -- start again from defaults. So a
file that is present but unreadable silently resets the very state a resume
exists to preserve, and reads as "this checkpoint predates the sidecar".

The case that motivates this is `pid_state.json`. Operators edit it by hand --
pinning `wdl_regret` below the config floor is a documented procedure -- and a
stray comma parses as nothing. The PID controller then starts from defaults and
the resulting curriculum shift is indistinguishable from the controller working
normally.

Absence stays silent: it is routine on an old checkpoint or a fresh start.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_anti_engine.tune._utils import SIDECAR_PID_STATE, load_optional_json


def test_a_missing_sidecar_stays_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Routine — an old checkpoint, or a fresh start. Must not cry wolf."""
    assert load_optional_json(tmp_path / SIDECAR_PID_STATE) is None
    assert capsys.readouterr().out == ""


def test_a_valid_sidecar_is_returned_unchanged_and_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / SIDECAR_PID_STATE
    state = {"wdl_regret": 0.1003, "integral": -2.5, "stage": 2}
    path.write_text(json.dumps(state))

    assert load_optional_json(path) == state
    assert capsys.readouterr().out == ""


def test_a_hand_edit_typo_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live footgun: a trailing comma from a manual regret pin."""
    path = tmp_path / SIDECAR_PID_STATE
    path.write_text('{"wdl_regret": 0.05, "stage": 2,}')

    assert load_optional_json(path) is None

    out = capsys.readouterr().out
    assert SIDECAR_PID_STATE in out
    assert "WITHOUT it" in out, "the message must state the consequence"
    assert str(path) in out, "and name the file, so the operator can go fix it"


def test_a_truncated_sidecar_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / SIDECAR_PID_STATE
    path.write_text('{"wdl_regret": 0.10')

    assert load_optional_json(path) is None
    assert SIDECAR_PID_STATE in capsys.readouterr().out


def test_an_empty_file_is_reported_rather_than_treated_as_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero bytes is a failed write, not a missing file."""
    path = tmp_path / SIDECAR_PID_STATE
    path.write_text("")

    assert load_optional_json(path) is None
    assert SIDECAR_PID_STATE in capsys.readouterr().out


def test_a_non_object_root_names_the_type_it_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid JSON, wrong shape — e.g. an editor that saved a list."""
    path = tmp_path / SIDECAR_PID_STATE
    path.write_text(json.dumps([{"wdl_regret": 0.1}]))

    assert load_optional_json(path) is None

    out = capsys.readouterr().out
    assert "list" in out, "naming the type is what makes this fixable at a glance"
    assert "WITHOUT it" in out


def test_an_unreadable_file_is_reported_not_swallowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory at the sidecar path raises OSError on read.

    Uses a real filesystem condition rather than a patched `read_text`, so the
    OSError branch is exercised the way it would actually occur.
    """
    path = tmp_path / SIDECAR_PID_STATE
    path.mkdir()

    assert load_optional_json(path) is None
    assert SIDECAR_PID_STATE in capsys.readouterr().out


def test_the_return_contract_is_unchanged_for_every_failure_mode(
    tmp_path: Path,
) -> None:
    """Callers do `load_optional_json(...) or {}` and `.get(...)` on the result.

    Reporting must not have turned any failure into an exception or a
    non-dict — the eight call sites all assume `dict | None`.
    """
    cases = [
        tmp_path / "absent.json",
        tmp_path / "bad.json",
        tmp_path / "list.json",
    ]
    cases[1].write_text("{nope")
    cases[2].write_text("[1, 2, 3]")
    (good := tmp_path / "good.json").write_text('{"a": 1}')

    for p in cases:
        assert load_optional_json(p) is None
    assert load_optional_json(good) == {"a": 1}


@pytest.mark.parametrize("raw", [b'\xff', b'{"name":"\xc3"}', b'{"name":"\xe2\x82'])
def test_invalid_utf8_sidecar_reports_corruption(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw: bytes,
) -> None:
    path = tmp_path / SIDECAR_PID_STATE
    path.write_bytes(raw)
    assert load_optional_json(path) is None
    out = capsys.readouterr().out
    assert str(path) in out
    assert "could not be read" in out
    assert "WITHOUT it" in out
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw", [b'\xff', b'{"matches":\xff}', b'{"matches":'])
def test_corrupt_own_gate_does_not_adopt_pool_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw: bytes,
) -> None:
    from chess_anti_engine.tune.trainable import _startup_gate_state
    from chess_anti_engine.tune.trial_config import RestoreResult

    path = tmp_path / "gate_state.json"
    path.write_bytes(raw)
    pool = {"matches": 41, "holds": 3}
    restore = RestoreResult(restored_gate_state=pool)
    assert _startup_gate_state(gate_state_path=path, restore=restore) == {}
    assert restore.restored_gate_state == pool
    out = capsys.readouterr().out
    assert "EMPTY window" in out
    assert "seeded from the salvage pool" not in out
    assert path.read_bytes() == raw


def test_valid_utf8_preserves_non_ascii_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / SIDECAR_PID_STATE
    value = {"label": "échec ♟", "matches": 7}
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    assert load_optional_json(path) == value
    assert capsys.readouterr().out == ""


def test_mandatory_yaml_invalid_encoding_still_raises(tmp_path: Path) -> None:
    from chess_anti_engine.utils.config_yaml import load_yaml_file

    path = tmp_path / "config.yaml"
    path.write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        load_yaml_file(path)


def test_optional_json_does_not_hide_unrelated_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / SIDECAR_PID_STATE
    path.write_text("{}")

    def fail_read(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        load_optional_json(path)
