from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text


def load_worker_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}

    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load worker config files.") from e

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config root must be a mapping/dict, got {type(data).__name__}")
    return data


def save_worker_config(path: str | Path, cfg: dict[str, Any]) -> None:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to save worker config files.") from e

    p = Path(path)
  # ⚑ The mode is passed to atomic_write_text so the tmp is CREATED at 0600
  # before the secret is written into it. chmod-ing the destination afterwards
  # -- what this did until now -- left the password world-readable for the
  # whole write-and-fsync window, on a file whose default path is under a
  # work_dir the operator did not necessarily make private.
    has_password = bool(cfg.get("password"))
    atomic_write_text(
        p,
        yaml.safe_dump(cfg, sort_keys=True),
        mode=0o600 if has_password else None,
    )

  # Backstop for the destination: atomic_write_text re-applies `mode` after the
  # writer returns, but an already-existing worker.yaml written by an older
  # version keeps its own permissions through os.replace on some filesystems.
    if has_password:
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass  # Windows / non-POSIX filesystem — chmod is best-effort
