"""Non-optional validation and atomic receipt output for opt-in proof gates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def require(condition: bool, message: str) -> None:
    """Keep evidence checks enabled even with -O or PYTHONOPTIMIZE."""
    if not condition:
        raise AssertionError(message)


def write_report(path: Path, report: dict) -> None:
    """A report path contains either the old complete JSON or the new complete JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def begin_report(path: Path, gate: str) -> None:
    """Invalidate an older PASS before this invocation starts external checks.

    An interrupted or failed attempt leaves NOT_COMPLETED, never an older PASS at
    the requested output path. Existing committed historical receipts are not read
    or changed. Callers should use separate output paths for concurrent runs.
    """
    write_report(path, {
        gate: "NOT_COMPLETED",
        "reason": "This invocation has not completed all required checks.",
        "pid": os.getpid(),
    })
