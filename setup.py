"""Build C extensions (called by setuptools via pyproject.toml).

Environment variables:
  CAE_EXT_NATIVE=1     — add -march=native (non-portable wheels)
  CAE_EXT_SANITIZE=X   — add -fsanitize=X -fno-omit-frame-pointer -g
                         e.g. CAE_EXT_SANITIZE=address,undefined
                         Requires LD_PRELOAD=$(gcc -print-file-name=libasan.so)
                         when running Python with ASAN.
  CAE_EXT_WERROR=1     — promote warnings to errors (strict builds)
"""
import os

from setuptools import setup, Extension
import numpy as np


def _warning_flags() -> list[str]:
    """Baseline warnings for all C extensions.

    -Wall/-Wextra: standard + extended warnings
    -Wshadow: reused variable names in nested scopes (common MCTS helper bugs)
    -Wformat=2: strict printf/scanf format-string checks
    -Wstrict-prototypes: catches K&R-style declarations
    -Wno-unused-parameter: Python C-API callbacks take (self, args) even when
      one is unused — would otherwise swamp signal on every Py function
    -Wno-cast-function-type: CPython macros require this for METH_VARARGS etc.
    """
    flags = [
        "-Wall", "-Wextra", "-Wshadow", "-Wformat=2", "-Wstrict-prototypes",
        # CPython / shared-header patterns — these fire on legitimate code:
        "-Wno-unused-parameter",  # Py_METH callbacks take (self, args)
        "-Wno-cast-function-type",  # METH_VARARGS etc. require the cast
        "-Wno-unused-function",  # _cboard_impl.h has helpers used by some .c files but not others
        "-Wno-missing-field-initializers",  # PyModuleDef has m_slots we leave zero
    ]
    if os.environ.get("CAE_EXT_WERROR", "").strip().lower() in {"1", "true", "yes"}:
        flags.append("-Werror")
    return flags


def _sanitizer_flags() -> tuple[list[str], list[str]]:
    """Return (compile_args, link_args) for the requested sanitizers, if any."""
    san = os.environ.get("CAE_EXT_SANITIZE", "").strip()
    if not san:
        return [], []
    flags = [f"-fsanitize={san}", "-fno-omit-frame-pointer", "-g", "-O1"]
    return flags, flags


def _ext_compile_args() -> list[str]:
    args = ["-O3"] + _warning_flags()
    # Keep default wheels/source builds portable across machines. Opt in to
    # host-specific tuning only when the builder explicitly requests it.
    if os.environ.get("CAE_EXT_NATIVE", "").strip().lower() in {"1", "true", "yes"}:
        args.append("-march=native")
    args += _sanitizer_flags()[0]
    return args


def _mcts_compile_args() -> list[str]:
    """MCTS tree extension gets OpenMP + native SIMD for encoding throughput."""
    args = ["-O3"] + _warning_flags() + ["-fopenmp", "-march=native"]
    args += _sanitizer_flags()[0]
    return args


def _ext_link_args(*, openmp: bool = False) -> list[str]:
    args = list(_sanitizer_flags()[1])
    if openmp:
        args.append("-fopenmp")
    return args


features_ext = Extension(
    "chess_anti_engine.encoding._features_ext",
    sources=["chess_anti_engine/encoding/_features_ext.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=_ext_compile_args(),
    extra_link_args=_ext_link_args(),
)

lc0_ext = Extension(
    "chess_anti_engine.encoding._lc0_ext",
    sources=["chess_anti_engine/encoding/_lc0_ext.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=_ext_compile_args(),
    extra_link_args=_ext_link_args(),
)

mcts_tree_ext = Extension(
    "chess_anti_engine.mcts._mcts_tree",
    sources=["chess_anti_engine/mcts/_mcts_tree.c"],
    include_dirs=[np.get_include(), "chess_anti_engine/encoding"],
    extra_compile_args=_mcts_compile_args(),
    extra_link_args=_ext_link_args(openmp=True),
)

setup(ext_modules=[features_ext, lc0_ext, mcts_tree_ext])
