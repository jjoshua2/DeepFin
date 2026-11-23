"""A snapshot whose resume files cannot be read back is refused with a named message, with the sibling term OFF as
well as on; nothing is deleted.

Covered (through ``train_generation`` on the tiny real collect):
- a truncated ``resume_state.pt``;
- a ``resume_state.pt`` that loads but is not a snapshot state: a list, an int, and (#1514 review) a str / list /
  tuple / set that CONTAINS the eight field names, term off AND term on;
- a dict missing each one of the eight required fields;
- a truncated ``resume_trainer.pt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.test_simple_rl_sibling import Killed, _fresh_run, _kill_after, _settings
from tests.test_simple_rl_sibling import run as run  # the module-scoped collect fixture


def _phase(run: dict[str, Path], name: str, monkeypatch: pytest.MonkeyPatch,
           term: bool = False) -> tuple[Path, Any, Path]:
    """A phase (term off, or on) killed after its update-3 snapshot; returns (run dir, settings, that snapshot's dir)."""
    from chess_anti_engine.simple_rl import learn as learn_mod
    from chess_anti_engine.simple_rl.learn import SNAPSHOTS, train_generation

    on = {"w_sibling": 1.0, "sibling_rows": 8, "sibling_dir": str(run["side"])} if term else {}
    s = _settings(run, snapshot_updates=(3,), **on)
    assert (float(s.w_sibling) > 0.0) == term
    d = _fresh_run(run, name)
    real_snap = _kill_after(monkeypatch, 3)
    with pytest.raises(Killed):
        train_generation(run_dir=d, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None, settings=s)
    monkeypatch.setattr(learn_mod, "write_snapshot", real_snap)
    snap = d / "gen_0001" / "train" / SNAPSHOTS / "step_3"
    assert snap.is_dir()
    return d, s, snap


def _train(run: dict[str, Path], d: Path, s: Any) -> dict[str, Any]:
    from chess_anti_engine.simple_rl.learn import train_generation

    return train_generation(run_dir=d, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                            settings=s)


def _files(snap: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in sorted(snap.iterdir())}


def test_the_phase_resumes_from_an_intact_snapshot_with_the_term_off(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d, s, _snap = _phase(run, "intact", monkeypatch)
    assert _train(run, d, s)["resumed_from_snapshot"] == 3   # the control: the cases below fail only on the damage


def test_the_phase_resumes_from_an_intact_snapshot_with_the_term_on(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    d, s, _snap = _phase(run, "intact_on", monkeypatch, term=True)
    assert _train(run, d, s)["resumed_from_snapshot"] == 3


_REFUSED = r"snapshot step_3: resume_state\.pt could not be restored"


@pytest.mark.parametrize("term", [False, True], ids=["term_off", "term_on"])
@pytest.mark.parametrize("kind", ["str", "list", "tuple", "set"])
def test_a_non_dict_that_contains_the_field_names_is_refused_by_name(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch, kind: str, term: bool) -> None:
    from chess_anti_engine.simple_rl.learn import _STATE_KEYS, RESUME_STATE

    d, s, snap = _phase(run, f"names_{kind}_{int(term)}", monkeypatch, term=term)
    payload = {"str": " ".join(_STATE_KEYS), "list": list(_STATE_KEYS), "tuple": tuple(_STATE_KEYS),
               "set": set(_STATE_KEYS)}[kind]
    assert all(k in payload for k in _STATE_KEYS)   # the trap: `in` alone would accept it
    torch.save(payload, str(snap / RESUME_STATE))
    before = _files(snap)
    with pytest.raises(RuntimeError, match=_REFUSED + r" \(TypeError: not a snapshot state"):
        _train(run, d, s)
    assert _files(snap) == before


@pytest.mark.parametrize("field", ["cursor", "served", "served_sl", "drawn_rows", "realised_sl_rows", "sampler_rng",
                                   "rng", "chunks"])
def test_a_state_missing_any_required_field_is_refused_by_name(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    from chess_anti_engine.simple_rl.learn import _STATE_KEYS, RESUME_STATE

    assert field in _STATE_KEYS
    d, s, snap = _phase(run, f"missing_{field}", monkeypatch)
    f = snap / RESUME_STATE
    state = torch.load(str(f), map_location="cpu", weights_only=False)
    del state[field]
    torch.save(state, str(f))
    before = _files(snap)
    with pytest.raises(RuntimeError, match=_REFUSED + rf" \(KeyError: .*missing \['{field}'\]"):
        _train(run, d, s)
    assert _files(snap) == before


def test_a_term_on_state_missing_a_field_is_refused_by_name(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.simple_rl.learn import RESUME_STATE

    d, s, snap = _phase(run, "missing_chunks_on", monkeypatch, term=True)
    f = snap / RESUME_STATE
    state = torch.load(str(f), map_location="cpu", weights_only=False)
    del state["chunks"]
    torch.save(state, str(f))
    with pytest.raises(RuntimeError, match=_REFUSED):
        _train(run, d, s)


@pytest.mark.parametrize("damage", ["truncated", "a_list", "an_int", "missing_fields"])
def test_a_damaged_resume_state_is_refused_by_name_with_the_term_off(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    from chess_anti_engine.simple_rl.learn import RESUME_STATE

    d, s, snap = _phase(run, f"state_{damage}", monkeypatch)
    f = snap / RESUME_STATE
    if damage == "truncated":
        f.write_bytes(f.read_bytes()[:100])
    elif damage == "a_list":
        torch.save([1, 2, 3], str(f))
    elif damage == "an_int":
        torch.save(5, str(f))
    else:
        state = torch.load(str(f), map_location="cpu", weights_only=False)
        del state["sampler_rng"]
        torch.save(state, str(f))
    before = _files(snap)
    with pytest.raises(RuntimeError, match=f"snapshot step_3: {RESUME_STATE} could not be restored"):
        _train(run, d, s)
    assert _files(snap) == before


def test_a_truncated_resume_trainer_is_refused_by_name_with_the_term_off(
        run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.simple_rl.learn import RESUME_TRAINER

    d, s, snap = _phase(run, "trainer_truncated", monkeypatch)
    f = snap / RESUME_TRAINER
    f.write_bytes(f.read_bytes()[:100])
    before = _files(snap)
    with pytest.raises(RuntimeError, match=f"snapshot step_3: {RESUME_TRAINER} could not be restored"):
        _train(run, d, s)
    assert _files(snap) == before
