import importlib


def test_targets_import_without_trainer_extras() -> None:
    # Regression test: train package should not eagerly import Trainer,
    # because Trainer has optional runtime deps (e.g. zclip).
    targets = importlib.import_module("chess_anti_engine.train.targets")
    assert hasattr(targets, "hlgauss_target")
