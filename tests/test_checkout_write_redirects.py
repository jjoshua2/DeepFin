"""``tests/conftest.py`` keeps checkout-relative defaults out of the checkout.

The redirect must catch only the omitted-``log_dir`` default and leave an
explicit ``log_dir`` exactly where the caller put it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_anti_engine.train.trainer import Trainer


def _writer_dir(trainer: Trainer) -> Path:
    return Path(trainer.writer.log_dir).resolve()


def test_an_omitted_log_dir_lands_outside_the_working_directory() -> None:
    trainer = Trainer(torch.nn.Linear(1, 1), device="cpu", lr=1e-3)
    try:
        where = _writer_dir(trainer)
        assert where != (Path.cwd() / "tb").resolve()
        assert where.name == "trainer_default_tb"
    finally:
        trainer.writer.close()


def test_an_explicit_log_dir_is_untouched(tmp_path: Path) -> None:
    trainer = Trainer(torch.nn.Linear(1, 1), device="cpu", lr=1e-3, log_dir=tmp_path / "mine")
    try:
        assert _writer_dir(trainer) == (tmp_path / "mine").resolve()
    finally:
        trainer.writer.close()


def test_a_lazily_imported_trainer_is_patched_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The conftest finder patches the module on (re-)import, not only at session start."""
    import importlib
    import sys

    import chess_anti_engine.train as train_pkg

    name = "chess_anti_engine.train.trainer"
    original = sys.modules[name]
    monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(train_pkg, "trainer", original, raising=False)
    fresh = importlib.import_module(name)
    try:
        assert fresh is not original
        assert getattr(fresh.SummaryWriter, "_cae_default_tb_redirect", False)
    finally:
        sys.modules[name] = original


def test_the_arena_game_log_default_is_outside_the_checkout() -> None:
    import scripts.arena_standard as arena

    repo = Path(__file__).resolve().parents[1]
    assert not arena.DEFAULT_GAME_LOG_DIR.resolve().is_relative_to(repo)
    assert arena.DEFAULT_GAME_LOG_DIR.name == "arena_games"
