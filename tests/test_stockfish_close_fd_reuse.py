from __future__ import annotations

import os
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from chess_anti_engine.stockfish.uci import StockfishUCI


@pytest.mark.parametrize("repeat_closes", [1, 2])
def test_repeated_close_preserves_reused_owned_descriptor(
    tmp_path: Path, repeat_closes: int,
) -> None:
    """An old owner must not write to or close a newly acquired descriptor."""
    old_fd = os.open(tmp_path / "old", os.O_RDWR | os.O_CREAT, 0o600)
    owner = StockfishUCI.__new__(StockfishUCI)
    owner._lock = threading.Lock()
    owner._tty_fd = old_fd
    owner._pgid = -1
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    owner.proc = cast(subprocess.Popen[bytes], process)
    owner.close()

    fresh_fd = os.open(tmp_path / "fresh", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert fresh_fd == old_fd, "fixture must exercise descriptor reuse"
        os.write(fresh_fd, b"fresh")
        for _ in range(repeat_closes):
            owner.close()
        os.fstat(fresh_fd)
        os.lseek(fresh_fd, 0, os.SEEK_SET)
        assert os.read(fresh_fd, 100) == b"fresh"
        process.kill.assert_not_called()
    finally:
        with suppress(OSError):
            os.close(fresh_fd)
