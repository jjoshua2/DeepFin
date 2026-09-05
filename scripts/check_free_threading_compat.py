"""Check whether the worker/match import graph keeps CPython's GIL disabled."""
from __future__ import annotations

import argparse
import importlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="also import Ray, the main training-runtime compatibility gate",
    )
    args = parser.parse_args()

    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        raise SystemExit("FAIL: this interpreter cannot report free-threading status")

    gil_initial = bool(is_gil_enabled())
    print(f"python={sys.version.split()[0]} gil_initial={gil_initial}")
    if gil_initial:
        raise SystemExit(
            "FAIL: the GIL is enabled at startup; this diagnostic requires a "
            "free-threaded interpreter starting with the GIL disabled"
        )

    modules = [
        "numpy",
        "torch",
        "chess",
        "yaml",
        "requests",
        "zarr",
        "chess_anti_engine.encoding._features_ext",
        "chess_anti_engine.encoding._lc0_ext",
        "chess_anti_engine.mcts._mcts_tree",
        "chess_anti_engine.nnue._nnue_ext",
    ]
    if args.include_training:
        modules.append("ray")

    failures: list[str] = []
    gil_was_enabled = gil_initial
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment diagnostic
            failures.append(f"{module_name}: import failed: {exc!r}")
            print(f"module={module_name} import=FAIL error={exc!r}")
            continue
        gil_enabled = bool(is_gil_enabled())
        already_enabled = gil_was_enabled and gil_enabled
        suffix = " (already enabled; module not assessed)" if already_enabled else ""
        print(f"module={module_name} import=ok gil_enabled={gil_enabled}{suffix}")
        if gil_enabled and not gil_was_enabled:
            failures.append(f"{module_name}: import enabled the GIL")
        gil_was_enabled = gil_enabled

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS: all requested imports left the GIL disabled")


if __name__ == "__main__":
    main()
