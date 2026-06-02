from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_root_tactics_import_does_not_require_cboard_encode() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCBoardEncode(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "chess_anti_engine.encoding.cboard_encode":
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockCBoardEncode())
        import chess_anti_engine.mcts.root_tactics as root_tactics
        assert root_tactics.immediate_mate_move is not None
        """
    )
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        str(repo)
        if not env.get("PYTHONPATH")
        else f"{repo}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
