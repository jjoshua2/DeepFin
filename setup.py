"""Build C extensions (called by setuptools via pyproject.toml).

Environment variables:
  CAE_EXT_NATIVE=1     — add -march=native (non-portable wheels). This is also
                         what selects the BMI2 PEXT slider backend, on the CPU
                         families where PEXT is fast; the family exclusion and
                         its reasoning live at the gate in
                         encoding/_slider_attacks_impl.h. A portable build gets
                         the magic backend, which is correct and expected.
  CAE_EXT_LTO=1        — add -flto at compile and link time
  CAE_EXT_SANITIZE=X   — add -fsanitize=X -fno-omit-frame-pointer -g
                         e.g. CAE_EXT_SANITIZE=address,undefined
                         Requires LD_PRELOAD=$(gcc -print-file-name=libasan.so)
                         when running Python with ASAN.
  CAE_EXT_WERROR=1     — promote warnings to errors (strict builds)
"""
import os

from setuptools import setup, Extension
import numpy as np


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


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
    if _env_enabled("CAE_EXT_WERROR"):
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
    args = ["-O3", *_warning_flags()]
    # Keep default wheels/source builds portable across machines. Opt in to
    # host-specific tuning only when the builder explicitly requests it.
    if _env_enabled("CAE_EXT_NATIVE"):
        args.append("-march=native")
    if _env_enabled("CAE_EXT_LTO"):
        args.append("-flto")
    if _env_enabled("CAE_EXT_NATIVE") and _env_enabled("CAE_EXT_LTO"):
        # Preserve the effective native architecture and LTO codegen recipe
        # in the ELF so the production startup guard can reject a plain GCC15
        # portable rebuild. GCC expands -march=native in this section.
        args.append("-frecord-gcc-switches")
    args += _sanitizer_flags()[0]
    return args


def _mcts_compile_args() -> list[str]:
    """MCTS tree extension gets OpenMP; host-native SIMD is opt-in like other exts.

    Default builds stay portable for worker wheels. Set CAE_EXT_NATIVE=1 for
    local/production machines that want -march=native (same gate as
    _ext_compile_args).
    """
    args = ["-O3", *_warning_flags(), "-fopenmp"]
    if _env_enabled("CAE_EXT_NATIVE"):
        args.append("-march=native")
    if _env_enabled("CAE_EXT_LTO"):
        args.append("-flto")
    if _env_enabled("CAE_EXT_NATIVE") and _env_enabled("CAE_EXT_LTO"):
        args.append("-frecord-gcc-switches")
    args += _sanitizer_flags()[0]
    return args


def _ext_link_args(*, openmp: bool = False) -> list[str]:
    args = list(_sanitizer_flags()[1])
    if _env_enabled("CAE_EXT_LTO"):
        args.append("-flto")
    if openmp:
        args.append("-fopenmp")
    return args


# _cboard_impl.h defines the original ray walkers before its shared
# bitboard-plane include boundary. These aliases retain those definitions as
# an exhaustive oracle; _slider_attacks_impl.h undefines the aliases at that
# boundary and installs table-backed PEXT/magic helpers for all subsequent
# move-generation and search code.
#
# ⚑ THESE MACROS ARE PER-EXTENSION AND THEY DO NOT TRAVEL. The headers are
# header-only statics, so every .so that includes _cboard_impl.h compiles its
# OWN slider code, and only the extensions listed below get the fast one. In
# particular the C tree does NOT hand its sliders to the NNUE search: a value
# provider reaches the tree through a PyCapsule, never an #include
# (chess_anti_engine/mcts/_value_provider.h), so qsearch, the check resolver,
# FastQ and FastQ's SEE x-ray loop all execute the copy compiled into
# _nnue_ext.so. Give the macros to every extension that includes
# _cboard_impl.h; verify per-.so with
#   objdump -d <so> | grep -c pext        # native/BMI2 build
# and treat an unchanged count as "the macros did not reach this extension".
# The annotation is load-bearing for the type gate: setuptools declares
# define_macros as list[tuple[str, str | None]] (None means a bare -D), and a
# bare list literal infers as list[tuple[str, str]], which is invariant and so
# not assignable.
_CBOARD_FAST_SLIDER_MACROS: list[tuple[str, str | None]] = [
    ("DEEPFIN_FAST_SLIDERS", "1"),
    ("init_attack_tables", "init_attack_tables_reference"),
    ("slider_attacks", "slider_attacks_reference"),
    ("bishop_attacks", "bishop_attacks_reference"),
    ("rook_attacks", "rook_attacks_reference"),
    ("queen_attacks", "queen_attacks_reference"),
    ("is_attacked_by", "is_attacked_by_reference"),
]


# ⚑ NO fast-slider macros here, deliberately. _features_ext.c includes
# _features_impl.h ALONE and never _cboard_impl.h, so _CBOARD_IMPL_H is
# undefined and it compiles the standalone feat_rook_attacks/feat_bishop_attacks
# ray walkers under their own names — names these macros do not rename and
# _slider_attacks_impl.h does not define. Defining DEEPFIN_FAST_SLIDERS here
# would not redirect one call: on a portable build it is inert (nothing pulls
# _slider_attacks_impl.h in), and on a native build __AVX2__ makes
# _features_impl.h include _bitboard_planes_impl.h, which would then pull
# _slider_attacks_impl.h into a translation unit with no slider_attacks_reference
# and no RAY_DF/PAWN_ATTACKS — a build failure, not a speedup. Making this
# extension table-backed means giving _features_impl.h's standalone branch its
# own tables, which is a separate change with its own oracle; it is also not on
# the search hot path (these sliders serve v3_xray/attack-map plane encoding,
# once per encoded position, not per search node).
features_ext = Extension(
    "chess_anti_engine.encoding._features_ext",
    sources=["chess_anti_engine/encoding/_features_ext.c"],
    include_dirs=[np.get_include(), "chess_anti_engine/encoding"],
    extra_compile_args=_ext_compile_args(),
    extra_link_args=_ext_link_args(),
)

lc0_ext = Extension(
    "chess_anti_engine.encoding._lc0_ext",
    sources=["chess_anti_engine/encoding/_lc0_ext.c"],
    include_dirs=[np.get_include()],
    define_macros=_CBOARD_FAST_SLIDER_MACROS,
    extra_compile_args=_ext_compile_args(),
    extra_link_args=_ext_link_args(),
)

mcts_tree_ext = Extension(
    "chess_anti_engine.mcts._mcts_tree",
    sources=["chess_anti_engine/mcts/_mcts_tree.c"],
    # The tree includes the encoding headers directly (_cboard_impl.h,
    # _features_impl.h, and the eval-plugin seam _value_provider.h, which itself
    # includes _cboard_impl.h for the CBoard type in the eval() signature), so it
    # needs the encoding include dir.
    #
    # ⚑ It does NOT compile the NNUE evaluator in. A provider reaches the tree
    # through a PyCapsule, never an #include — see the "HOW A PROVIDER REACHES
    # THE TREE" contract in mcts/_value_provider.h, which exists precisely
    # because a header-only evaluator duplicated into the tree would give the
    # tree a second copy of the evaluator's kernel flag and weight cache. So the
    # macros below make the TREE's own movegen table-backed and reach nothing
    # inside _nnue_ext.so; that extension is given them separately.
    include_dirs=[np.get_include(), "chess_anti_engine/encoding"],
    define_macros=_CBOARD_FAST_SLIDER_MACROS,
    extra_compile_args=_mcts_compile_args(),
    extra_link_args=_ext_link_args(openmp=True),
)

# ⚑ THE DEFAULT (PORTABLE) BUILD IS SCALAR-ONLY. The AVX2 kernels compile in
# only under -march=native, i.e. when CAE_EXT_NATIVE is set — as
# scripts/build_production_extensions.py does and CI does not. So on a portable
# build _nnue_ext.HAVE_AVX2 is 0, set_simd(True) raises, and there is one kernel
# rather than two. Anything that flips kernels must branch on HAVE_AVX2; the
# evaluator's own numbers are identical either way, since the two kernels are
# gated against Stockfish precisely to keep that true.
nnue_ext = Extension(
    "chess_anti_engine.nnue._nnue_ext",
    sources=["chess_anti_engine/nnue/_nnue_ext.c"],
    include_dirs=[np.get_include(), "chess_anti_engine/encoding"],
    # ⚑ THIS IS THE EXTENSION PR-S2 EXISTS FOR. _nnue_ext.c includes
    # _cboard_impl.h and compiles in qsearch, the recursive check resolver,
    # FastQ and FastQ's SEE x-ray loop (_fastq_see.h calls bishop_attacks/
    # rook_attacks directly). None of that is reachable from _mcts_tree.so's
    # copy — the tree calls a provider through a capsule, not an #include — so
    # without these macros the whole NNUE search ray-walks no matter what the
    # other two extensions were built with.
    define_macros=_CBOARD_FAST_SLIDER_MACROS,
    # OpenMP: the throughput benchmark measures the multi-thread scaling the
    # native-generator gate is decided on, so it has to actually run threaded.
    extra_compile_args=_mcts_compile_args(),
    extra_link_args=_ext_link_args(openmp=True),
)

setup(ext_modules=[features_ext, lc0_ext, mcts_tree_ext, nnue_ext])
