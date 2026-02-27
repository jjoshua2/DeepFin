from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_worker_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load worker config files.") from e

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config root must be a mapping/dict, got {type(data).__name__}")
    return data


def save_worker_config(path: str | Path, cfg: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to save worker config files.") from e

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    tmp.replace(p)

    # Best-effort: if config contains a password, lock down permissions.
    try:
        if "password" in cfg and cfg.get("password"):
            os.chmod(p, 0o600)
    except Exception:
        pass
