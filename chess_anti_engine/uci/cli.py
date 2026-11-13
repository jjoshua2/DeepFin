"""``deepfin`` console-script entry point.

Kept import-light on purpose: ``uci/__main__.py`` imports torch and the C
extensions at module level, so a missing or broken extension used to end
``deepfin`` with a bare ImportError traceback — and made ``deepfin
--selfcheck`` unusable on exactly the install it exists to diagnose. Here the
self-check runs without importing the engine, and an engine import failure is
reported as a refusal with the fix.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--selfcheck" in args:
        from chess_anti_engine.uci.selfcheck import main as selfcheck_main

        return selfcheck_main(args)
    try:
        # By name: uci/__main__.py hands `python -m` runs to this module, and a
        # static import back would be an import cycle.
        engine_main = importlib.import_module("chess_anti_engine.uci.__main__").main
    except ImportError as exc:
        from chess_anti_engine.uci.selfcheck import REBUILD_HINT

        print(
            f"deepfin: refusing to start: the engine cannot be imported ({exc}).\n"
            f"Fix: {REBUILD_HINT}. Diagnose with `deepfin --selfcheck`.",
            file=sys.stderr,
        )
        return 2
    if argv is not None:
        sys.argv = [sys.argv[0], *args]
    return engine_main()


if __name__ == "__main__":
    raise SystemExit(main())
