#!/usr/bin/env python3
"""Build host-optimized in-place C extensions for local production runs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections.abc import Mapping


_TRANSIENT_BUILD_ENV = (
    "AR",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "LDSHARED",
    "RANLIB",
    "CAE_EXT_SANITIZE",
    "CAE_EXT_WERROR",
)


def select_compiler(*, env: Mapping[str, str], home: Path) -> str:
    """Prefer an explicit compiler, then the validated local GCC 15 install."""
    candidates = (
        env.get("CAE_GCC15_CC", ""),
        str(home / ".local/gcc-15.3/bin/gcc"),
    )
    for candidate in candidates:
        path = Path(candidate).expanduser() if candidate else None
        if path is not None and path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise RuntimeError(
        "validated GCC 15 compiler not found; install ~/.local/gcc-15.3/bin/gcc "
        "or set CAE_GCC15_CC (use setup.py directly only for portable builds)",
    )


def build_environment(*, compiler: str, base: Mapping[str, str]) -> dict[str, str]:
    env = dict(base)
    for name in _TRANSIENT_BUILD_ENV:
        env.pop(name, None)
    env.update({
        "CC": str(compiler),
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    })
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the build without executing it.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = select_compiler(env=os.environ, home=Path.home())
    # Always force: setuptools does not discover transitive .h dependencies,
    # and a stale portable/sanitized object must never be certified as a
    # production build merely because its top-level .c mtime is unchanged.
    command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
    print(
        f"production extension build: CC={compiler} native=1 lto=1 "
        f"command={shlex.join(command)}",
    )
    if args.dry_run:
        return 0
    subprocess.run(
        command,
        cwd=root,
        env=build_environment(compiler=compiler, base=os.environ),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
