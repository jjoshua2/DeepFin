"""AOT constant rebinding must not leave a package on stale weights (audit I4).

The broker (and ``AOTEvaluator``) build ONE ``load_constants`` payload from ONE
package's ``get_constant_fqns()`` and load it into every package with
``check_full_update=False``. ``build_aot_constants`` fails loud on a missing FQN
— but only against package 0's list. Nothing compared package *i*'s FQN set to
package 0's, and ``check_full_update=False`` explicitly tells AOTInductor not to
complain about constants the payload does not cover. So a package built from a
different architecture revision kept its build-time weights across every model
publish, while the comment three lines above promised the opposite.

``load_aot_packages`` silently skips missing files and ``should_use_aot_forward``
silently falls back to eager for uncovered buckets, so a partial rebuild — the
exact output of a bucket-ladder change — is how this arms.

No GPU: the packages here are fakes that mimic the constant surface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from chess_anti_engine import inference as inf
from chess_anti_engine.inference import (
    AOT_CHECK_FULL_UPDATE_ENV,
    aot_check_full_update_enabled,
    assert_uniform_constant_fqns,
)

_BUILD_TIME_WEIGHT = -999.0


class _FakePackage:
    """Mimics an AOTInductor package's constant surface."""

    def __init__(self, name: str, fqns: list[str]) -> None:
        self.name = name
        self._fqns = list(fqns)
        self.constants = {f: torch.full((2,), _BUILD_TIME_WEIGHT) for f in fqns}

    def get_constant_fqns(self) -> list[str]:
        return list(self._fqns)

    def load_constants(self, constants, *, check_full_update: bool = True) -> None:
        if check_full_update:
            missing = [f for f in self._fqns if f not in constants]
            if missing:
                raise RuntimeError(f"{self.name}: missing constants {missing}")
        for k, v in constants.items():
            if k in self.constants:
                self.constants[k] = v


def _load_fakes(packages: dict[int, _FakePackage], monkeypatch) -> dict[int, object]:
    tmp = Path(tempfile.mkdtemp(prefix="aot_bind_"))
    for b in packages:
        (tmp / inf.aot_package_filename(b)).write_bytes(b"stub")
    monkeypatch.setattr(
        inf,
        "_aoti_load_package",
        lambda path: packages[int(Path(path).stem.removeprefix("chess_b"))],
    )
    return inf.load_aot_packages(tmp, buckets=tuple(sorted(packages)))


def test_uniform_fqn_sets_are_accepted_and_returned(monkeypatch) -> None:
    fqns = ["trunk.w", "head.b"]
    models = _load_fakes(
        {128: _FakePackage("chess_b128.pt2", fqns),
         256: _FakePackage("chess_b256.pt2", list(reversed(fqns)))},
        monkeypatch,
    )
    # Order must not matter; the set must.
    assert sorted(assert_uniform_constant_fqns(models)) == sorted(fqns)


def test_heterogeneous_package_set_is_refused_at_load(monkeypatch) -> None:
    """The exact I4 setup: bucket 256 needs a constant bucket 128 never lists."""
    models = _load_fakes(
        {128: _FakePackage("chess_b128.pt2", ["trunk.w", "head.b"]),
         256: _FakePackage("chess_b256.pt2", ["trunk.w", "head.b", "extra.buf"])},
        monkeypatch,
    )
    with pytest.raises(RuntimeError, match="not uniform"):
        assert_uniform_constant_fqns(models)


def test_check_full_update_default_is_on_and_env_can_disable_it() -> None:
    assert aot_check_full_update_enabled({}) is True
    assert aot_check_full_update_enabled({AOT_CHECK_FULL_UPDATE_ENV: "1"}) is True
    assert aot_check_full_update_enabled({AOT_CHECK_FULL_UPDATE_ENV: ""}) is True
    for off in ("0", "false", "no", "OFF"):
        assert aot_check_full_update_enabled({AOT_CHECK_FULL_UPDATE_ENV: off}) is False


def test_full_update_check_catches_a_package_the_payload_cannot_cover() -> None:
    """With the flip, a publish into an under-covered package raises.

    Without it the package silently kept ``extra.buf`` at its build-time value
    and every batch that rounded to that bucket was answered by a hybrid of new
    and stale weights.
    """
    pkg = _FakePackage("chess_b256.pt2", ["trunk.w", "head.b", "extra.buf"])
    payload = {
        "trunk.w": torch.full((2,), 7.0),
        "head.b": torch.full((2,), 7.0),
    }

    # Pre-fix behaviour, for contrast: no error, and the stale constant survives.
    pkg.load_constants(payload, check_full_update=False)
    assert float(pkg.constants["extra.buf"][0]) == pytest.approx(_BUILD_TIME_WEIGHT)

    with pytest.raises(RuntimeError, match="missing constants"):
        pkg.load_constants(payload, check_full_update=True)


def test_aot_evaluator_load_weights_refuses_a_mismatched_package(monkeypatch) -> None:
    """End-to-end through the worker-local AOT path (inference.py AOTEvaluator).

    Constructing it now refuses the heterogeneous set outright, which is the
    earliest possible failure point.
    """
    packages = {
        16: _FakePackage("chess_b16.pt2", ["trunk.w", "head.b"]),
        32: _FakePackage("chess_b32.pt2", ["trunk.w", "head.b", "extra.buf"]),
    }
    tmp = Path(tempfile.mkdtemp(prefix="aot_eval_"))
    for b in packages:
        (tmp / inf.aot_package_filename(b)).write_bytes(b"stub")
    monkeypatch.setattr(
        inf,
        "_aoti_load_package",
        lambda path: packages[int(Path(path).stem.removeprefix("chess_b"))],
    )
    with pytest.raises(RuntimeError, match="not uniform"):
        inf.AOTEvaluator(tmp, device="cpu", max_batch=32, input_planes=8)


def test_aot_evaluator_load_weights_names_the_bucket_that_rejected(monkeypatch) -> None:
    """A uniform set still gets check_full_update; the error names the bucket."""
    fqns = ["trunk.w", "head.b"]
    packages = {
        16: _FakePackage("chess_b16.pt2", fqns),
        32: _FakePackage("chess_b32.pt2", fqns),
    }
    tmp = Path(tempfile.mkdtemp(prefix="aot_eval_ok_"))
    for b in packages:
        (tmp / inf.aot_package_filename(b)).write_bytes(b"stub")
    monkeypatch.setattr(
        inf,
        "_aoti_load_package",
        lambda path: packages[int(Path(path).stem.removeprefix("chess_b"))],
    )
    ev = inf.AOTEvaluator(tmp, device="cpu", max_batch=32, input_planes=8)
    ev.load_weights({f: torch.full((2,), 7.0) for f in fqns})
    for pkg in packages.values():
        assert float(pkg.constants["trunk.w"][0]) == pytest.approx(7.0)

    # Now make one package start wanting a constant nobody supplies, as a
    # mid-life rebuild would, and prove the rebind raises with the bucket named.
    packages[32]._fqns.append("extra.buf")
    packages[32].constants["extra.buf"] = torch.full((2,), _BUILD_TIME_WEIGHT)
    with pytest.raises(RuntimeError, match="AOT bucket 32 rejected"):
        ev.load_weights({f: torch.full((2,), 8.0) for f in fqns})


def test_env_kill_switch_restores_the_permissive_rebind(monkeypatch) -> None:
    """CAE_AOT_CHECK_FULL_UPDATE=0 must actually reach the load_constants call."""
    fqns = ["trunk.w", "head.b"]
    packages = {
        16: _FakePackage("chess_b16.pt2", fqns),
        32: _FakePackage("chess_b32.pt2", fqns),
    }
    tmp = Path(tempfile.mkdtemp(prefix="aot_eval_off_"))
    for b in packages:
        (tmp / inf.aot_package_filename(b)).write_bytes(b"stub")
    monkeypatch.setattr(
        inf,
        "_aoti_load_package",
        lambda path: packages[int(Path(path).stem.removeprefix("chess_b"))],
    )
    ev = inf.AOTEvaluator(tmp, device="cpu", max_batch=32, input_planes=8)
    packages[32]._fqns.append("extra.buf")
    packages[32].constants["extra.buf"] = torch.full((2,), _BUILD_TIME_WEIGHT)

    monkeypatch.setenv(AOT_CHECK_FULL_UPDATE_ENV, "0")
    ev.load_weights({f: torch.full((2,), 9.0) for f in fqns})
    assert float(packages[32].constants["trunk.w"][0]) == pytest.approx(9.0)
    assert float(packages[32].constants["extra.buf"][0]) == pytest.approx(
        _BUILD_TIME_WEIGHT
    )
