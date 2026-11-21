"""Shared fixtures for the simple_rl tests: a tiny net and a quiet fake engine."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tests.selfplay_helpers import FakeStockfish

TINY_ARCH: dict[str, Any] = {
    "embed_dim": 32, "num_layers": 1, "num_heads": 2, "ffn_mult": 1.0,
    "smolgen_hidden_channels": 8, "smolgen_hidden_sz": 32, "smolgen_gen_sz": 32,
    "enable_policy_sf_head": False,
}


def tiny_model_config() -> Any:
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
    from chess_anti_engine.uci.model_loader import model_config_from_arch

    return model_config_from_arch({"_schema_version": ARCH_SCHEMA_VERSION, **asdict(ModelConfig()), **TINY_ARCH})


def tiny_model(seed: int = 0) -> torch.nn.Module:
    from chess_anti_engine.model import build_model

    torch.manual_seed(seed)
    return build_model(tiny_model_config()).eval()


def save_tiny_checkpoint(path: Path, seed: int = 0) -> Path:
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer

    torch.manual_seed(seed)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu"), log_dir=None)
    trainer.save(Path(path))
    return Path(path)


class QuietStockfish(FakeStockfish):
    """Fixed-answer engine with the ``close`` the collector calls."""

    def close(self) -> None:
        return None


def quiet_stockfish(_settings: Any) -> QuietStockfish:
    """Module-level engine factory (picklable by name, so spawned collect
    processes can use it where a lambda could not be sent)."""
    return QuietStockfish([0.0, 1.0, 0.0])


class _ScriptedStockfish(QuietStockfish):
    """QuietStockfish whose ``search`` first runs a hook (wait / die / hang)."""

    def __init__(self, hook: Any) -> None:
        super().__init__([0.0, 1.0, 0.0])
        self._hook = hook

    def search(self, fen: str, **kwargs: Any) -> Any:
        self._hook()
        return super().search(fen, **kwargs)


def scripted_stockfish(_settings: Any) -> QuietStockfish:
    """Engine factory driven by environment variables, for spawned collect processes.

    ``SIMPLE_RL_TEST_SF_SCRIPT`` is a JSON object keyed by collect process index
    (``SIMPLE_RL_COLLECT_PROC``, set by the child): ``{"1": {"after": <path>,
    "do": "die" | "hang", "pid_dir": <dir>}}``. A process with an entry waits on
    its first query until ``after`` exists (if given), then raises or blocks
    forever. With ``pid_dir`` the factory records its own pid and starts a
    sentinel ``sleep`` child exactly the way ``stockfish.uci`` starts engines
    (own session, parent-death signal), recording that pid too.
    """
    import json
    import os
    import subprocess
    import time
    from functools import partial

    script = json.loads(os.environ.get("SIMPLE_RL_TEST_SF_SCRIPT", "{}"))
    entry = script.get(os.environ.get("SIMPLE_RL_COLLECT_PROC", ""), {})
    if entry.get("pid_dir"):
        from chess_anti_engine.stockfish.uci import _child_reap_guard

        pid_dir = Path(entry["pid_dir"])
        sentinel = subprocess.Popen(
            ["sleep", "600"], start_new_session=True,
            preexec_fn=partial(_child_reap_guard, os.getpid()))  # noqa: PLW1509 - mirrors stockfish.uci
        (pid_dir / f"collect_{os.getpid()}").write_text(str(os.getpid()))
        (pid_dir / f"sentinel_{sentinel.pid}").write_text(str(sentinel.pid))
    if not entry.get("do"):
        return QuietStockfish([0.0, 1.0, 0.0])

    def hook() -> None:
        after = entry.get("after")
        while after and not Path(after).exists():
            time.sleep(0.05)
        if entry["do"] == "die":
            raise RuntimeError("scripted engine death")
        while True:
            time.sleep(1.0)

    return _ScriptedStockfish(hook)
