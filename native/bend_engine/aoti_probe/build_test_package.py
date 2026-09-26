from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def _resolve_package_cxx() -> str:
    """Pick a C++ compiler whose libstdc++ matches a normal native process.

    Inductor compiles the wrapper from ``CXX``. A user-local PATH ``g++`` can
    be newer than the system libstdc++ the probe loads. Honor only
    ``BEND_AOTI_PACKAGE_CXX``, then ``/usr/bin/g++`` — not a leftover ``CXX``.
    """
    explicit = os.environ.get("BEND_AOTI_PACKAGE_CXX", "").strip()
    if explicit:
        return explicit
    usr = Path("/usr/bin/g++")
    if usr.is_file() and os.access(usr, os.X_OK):
        return str(usr)
    found = shutil.which("g++")
    if found:
        return found
    raise SystemExit("error: no g++ for AOTInductor package compile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Must land before torch/inductor import: config.cpp.cxx is filled from CXX
    # at import, and a PATH g++ that is newer than system libstdc++ produces a
    # wrapper this process cannot dlopen.
    cxx = _resolve_package_cxx()
    os.environ["CXX"] = cxx
    print(f"aoti package cxx: {cxx}", file=sys.stderr)

    torch = importlib.import_module("torch")
    inductor_config = importlib.import_module("torch._inductor.config")
    inductor_config.cpp.cxx = (cxx,)

    class ProbeModel(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            # Exact binary arithmetic for the integer-valued probe inputs.
            return x * 2.0 + 1.0

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    model = ProbeModel().eval()
    example = torch.zeros((2, 4), dtype=torch.float32)
    with torch.no_grad():
        exported = torch.export.export(model, (example,))
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=str(out),
        )

    print(out)


if __name__ == "__main__":
    main()
