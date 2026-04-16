from __future__ import annotations

from pathlib import Path

from chess_anti_engine.learner import (
    _compute_max_plies,
    _resolve_trial_subdir,
    _sf_bootstrap_ramp_frac,
)


def test_compute_max_plies_clamps_and_interpolates() -> None:
    assert _compute_max_plies(0, start=40, end=100, ramp_steps=1000) == 40
    assert _compute_max_plies(500, start=40, end=100, ramp_steps=1000) == 70
    assert _compute_max_plies(9999, start=40, end=100, ramp_steps=1000) == 100


def test_compute_max_plies_degenerate_returns_end() -> None:
    assert _compute_max_plies(0, start=100, end=100, ramp_steps=0) == 100
    assert _compute_max_plies(500, start=200, end=100, ramp_steps=1000) == 100


def test_sf_bootstrap_ramp_frac_endpoints() -> None:
    assert _sf_bootstrap_ramp_frac(0.8, ramp_start=0.8, ramp_end_threshold=0.1) == 0.0
    assert _sf_bootstrap_ramp_frac(0.1, ramp_start=0.8, ramp_end_threshold=0.1) == 1.0
    mid = _sf_bootstrap_ramp_frac(0.45, ramp_start=0.8, ramp_end_threshold=0.1)
    assert 0.49 < mid < 0.51


def test_sf_bootstrap_ramp_frac_degenerate_span_returns_one() -> None:
    assert _sf_bootstrap_ramp_frac(0.1, ramp_start=0.1, ramp_end_threshold=0.1) == 1.0


def test_resolve_trial_subdir_with_trial_id(tmp_path: Path) -> None:
    server_root = tmp_path / "server"
    trial_root = server_root / "trials" / "t1"
    got = _resolve_trial_subdir("publish", "t1", server_root, trial_root)
    assert got == trial_root / "publish"


def test_resolve_trial_subdir_fallback_strips_server_prefix(tmp_path: Path) -> None:
    server_root = tmp_path / "server"
    trial_root = server_root / "trials" / "t1"
    got = _resolve_trial_subdir("server/work", "t1", server_root, trial_root, fallback_name="work")
    assert got == trial_root / "work"


def test_resolve_trial_subdir_absolute_path_passes_through(tmp_path: Path) -> None:
    server_root = tmp_path / "server"
    absolute = tmp_path / "elsewhere" / "publish"
    got = _resolve_trial_subdir(str(absolute), "t1", server_root, server_root / "trials" / "t1")
    assert got == absolute


def test_resolve_trial_subdir_no_trial_id_uses_server_root(tmp_path: Path) -> None:
    server_root = tmp_path / "server"
    got = _resolve_trial_subdir("publish", "", server_root, server_root)
    assert got == server_root / "publish"
