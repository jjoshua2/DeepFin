"""Build C extensions (called by setuptools via pyproject.toml)."""
import os

from setuptools import setup, Extension
import numpy as np


def _ext_compile_args() -> list[str]:
    args = ["-O3"]
    # Keep default wheels/source builds portable across machines. Opt in to
    # host-specific tuning only when the builder explicitly requests it.
    if os.environ.get("CAE_EXT_NATIVE", "").strip().lower() in {"1", "true", "yes"}:
        args.append("-march=native")
    return args


def _mcts_compile_args() -> list[str]:
    """MCTS tree extension gets OpenMP + native SIMD for encoding throughput."""
    args = _ext_compile_args() + ["-fopenmp", "-march=native"]
    return args

features_ext = Extension(
    "chess_anti_engine.encoding._features_ext",
    sources=["chess_anti_engine/encoding/_features_ext.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=_ext_compile_args(),
)

lc0_ext = Extension(
    "chess_anti_engine.encoding._lc0_ext",
    sources=["chess_anti_engine/encoding/_lc0_ext.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=_ext_compile_args(),
)

mcts_tree_ext = Extension(
    "chess_anti_engine.mcts._mcts_tree",
    sources=["chess_anti_engine/mcts/_mcts_tree.c"],
    include_dirs=[np.get_include(), "chess_anti_engine/encoding"],
    extra_compile_args=_mcts_compile_args(),
    extra_link_args=["-fopenmp"],
)

setup(ext_modules=[features_ext, lc0_ext, mcts_tree_ext])
