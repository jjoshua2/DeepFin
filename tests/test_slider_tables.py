"""The table-backed sliders, checked on the binaries that actually ship them.

⚑⚑ THE DEFECT THIS FILE EXISTS FOR. The fast sliders are header-only statics
selected by `-D` flags listed per-extension in `setup.py`. Every `.so` that
includes `_cboard_impl.h` therefore compiles its OWN copy — its own tables, its
own ray walker, its own backend choice — and an extension left off that list
silently keeps ray-walking. That is not hypothetical: the first version of this
change gave the macros to `_lc0_ext` and `_mcts_tree` only, so `_nnue_ext` —
which is where qsearch, the recursive check resolver, FastQ and FastQ's SEE
x-ray loop live, i.e. the entire search the change was written to speed up —
kept the slow path. Around 230 tests across 13 movegen/SEE/qsearch files passed
against that build, because every one of them asks whether the answer is right
and none of them asks which implementation produced it.

So the tests here ask the second question. `SLIDER_BACKEND` is read off each
imported module rather than off `setup.py`, because the build system's intent is
exactly what was already wrong.

⚑ ON THE INIT-TIME "EXHAUSTIVE VERIFICATION". Each module verifies its tables
against its own ray walker at import over the Carry-Rippler enumeration of each
square's relevant mask. That check is circular with respect to the MASK: it
cannot construct an occupancy with a bit outside the mask, so it cannot see a
wrong one. A popcount-preserving mask edit passes it, passes the table-size
assertion, and (on the PEXT arm, which has no collision check by construction)
imports perfectly cleanly — while producing wrong attack sets on real boards.
`slider_selftest` closes that hole from Python: full-board occupancies, off-mask
bits set, arm against arm.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import chess
import pytest

from chess_anti_engine.encoding import _features_ext, _lc0_ext
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.nnue import _nnue_ext

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every extension that compiles CBoard's move generator, and must therefore
#: have been given the fast-slider macros.
MOVEGEN_MODULES = [
    ("chess_anti_engine.encoding._lc0_ext", _lc0_ext),
    ("chess_anti_engine.mcts._mcts_tree", _mcts_tree),
    ("chess_anti_engine.nnue._nnue_ext", _nnue_ext),
]

TABLE_BACKED = {"pext", "magic"}


@pytest.mark.parametrize(
    ("name", "module"), MOVEGEN_MODULES, ids=[row[0] for row in MOVEGEN_MODULES]
)
def test_every_movegen_extension_is_table_backed(name: str, module: object) -> None:
    """⚑ THE REGRESSION GUARD. This is the test that fails on the shipped defect.

    `_nnue_ext` reporting "rays" while the others report "pext" is precisely the
    state this change was sent back to fix, and nothing else in the suite can
    see it: the ray walkers and the tables agree on every answer, so only the
    question "which one ran" separates them.
    """
    backend = getattr(module, "SLIDER_BACKEND", None)
    assert backend in TABLE_BACKED, (
        f"{name} reports SLIDER_BACKEND={backend!r}; it compiles CBoard's move "
        "generator and must be built with _CBOARD_FAST_SLIDER_MACROS. 'rays' "
        "means this extension silently kept the pre-change ray walkers."
    )


def test_the_movegen_extensions_all_chose_the_same_backend() -> None:
    """One build, one backend. A split means the macro list drifted per-target."""
    backends = {name: module.SLIDER_BACKEND for name, module in MOVEGEN_MODULES}
    assert len(set(backends.values())) == 1, backends


def test_features_ext_is_deliberately_ray_based() -> None:
    """Pins a DECISION, so that changing it has to be deliberate.

    `_features_ext.c` includes `_features_impl.h` alone and never
    `_cboard_impl.h`, so it carries standalone `feat_rook_attacks` /
    `feat_bishop_attacks` walkers under names the macros do not rename and
    `_slider_attacks_impl.h` does not define. Handing it the macros does not
    redirect a single call: on a portable build nothing pulls the slider header
    in at all, and on a native build the header lands in a translation unit with
    no `slider_attacks_reference` and no `RAY_DF`, which is a compile error. Its
    sliders also serve per-position plane encoding, not per-node search. See the
    comment above `features_ext` in setup.py.
    """
    assert _features_ext.SLIDER_BACKEND == "rays"


def test_the_built_backend_is_the_one_the_build_recipe_asked_for() -> None:
    """Lets CI (and the operator) PIN the arm instead of accepting whatever came out.

    Unset, this test still checks the build is coherent. Set
    `CAE_EXPECT_SLIDER_BACKEND=pext|magic|rays` and it becomes an assertion that
    the recipe produced the intended arm — which is the only way a "we built the
    fast one" claim is checkable after the fact.
    """
    expected = os.environ.get("CAE_EXPECT_SLIDER_BACKEND")
    if not expected:
        pytest.skip("set CAE_EXPECT_SLIDER_BACKEND to pin the expected arm")
    assert expected in TABLE_BACKED | {"rays"}, f"bad expectation {expected!r}"
    actual = {name: module.SLIDER_BACKEND for name, module in MOVEGEN_MODULES}
    assert set(actual.values()) == {expected}, (
        f"expected every movegen extension to be built with the {expected!r} "
        f"slider backend, got {actual}"
    )


@pytest.mark.parametrize(
    ("name", "module"), MOVEGEN_MODULES, ids=[row[0] for row in MOVEGEN_MODULES]
)
def test_slider_tables_agree_with_the_ray_walker_on_full_board_occupancies(
    name: str, module: object
) -> None:
    """The non-circular differential: off-mask bits set, arm against arm.

    Run per module, not once, because each `.so` holds its own tables and its own
    walker — asking `_lc0_ext` proves nothing about `_nnue_ext`.

    Measured on the mask mutant this test was written against (a
    popcount-preserving edit to `deepfin_slider_relevant_mask` that keeps d4's
    north edge square and drops the nearest): the PEXT build imports cleanly, the
    init-time exhaustive check passes, and this reports **11,284** mismatches at
    20,000 samples.
    """
    backend = getattr(module, "SLIDER_BACKEND", None)
    # ⚑ Non-vacuity: in a "rays" build `slider_attacks` IS the reference, so the
    # differential would compare the walker with itself and pass for free.
    assert backend in TABLE_BACKED, (
        f"{name} is {backend!r}: this differential is vacuous unless the module "
        "is table-backed"
    )
    selftest = getattr(module, "slider_selftest")
    # 20,000 samples x 64 squares x 2 kinds = 2,560,000 comparisons, ~0.2 s.
    assert selftest(0x243F6A8885A308D3, 20_000) == 0
    # A second, disjoint stream: one seed passing is one sample of the space.
    assert selftest(0xB5026F5AA96619E9, 20_000) == 0


def _see_corpus(games: int = 40, plies: int = 60, seed: int = 20260826) -> list[str]:
    """Deterministic self-play FENs.

    Moves are sorted by UCI before sampling so the corpus depends only on the
    seed and the rules of chess — not on python-chess's internal generation
    order, which is free to change under us and would silently redefine the
    checksum below.
    """
    rng = random.Random(seed)
    positions: list[str] = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over():
                break
            positions.append(board.fen())
            board.push(rng.choice(sorted(board.legal_moves, key=lambda m: m.uci())))
    return positions


#: Sum of SEE over every capture in the corpus, and the capture count.
#: ⚑ Backend-INDEPENDENT by construction: PEXT and magic index the same tables
#: with the same masks, so this number is identical on both arms and a change to
#: it means the slider results changed, not that the build did. Verified equal
#: on a PEXT build and a portable magic build of the same tree.
SEE_CHECKSUM = 167_690
SEE_CAPTURE_COUNT = 8_396


def test_see_checksum_is_pinned_across_slider_backends() -> None:
    """Drives `_nnue_ext`'s OWN sliders over ~8.4k real captures.

    `cae_see_capture` re-derives `bishop_attacks`/`rook_attacks` against a
    mutating occupancy at every swap ply — that is how it finds x-rays — so this
    is a dense, search-shaped exercise of exactly the code path that was still
    ray-walking in the shipped version of this change. `_nnue_ext.see()` is the
    same function FastQ orders and gates with, not a test reimplementation.
    """
    total = 0
    captures = 0
    for fen in _see_corpus():
        board = chess.Board(fen)
        cboard = CBoard.from_board(board)
        for move in sorted(board.legal_moves, key=lambda m: m.uci()):
            if not board.is_capture(move):
                continue
            total += _nnue_ext.see(
                cboard, move.from_square, move.to_square, move.promotion or 0
            )
            captures += 1
    assert captures == SEE_CAPTURE_COUNT
    assert total == SEE_CHECKSUM


def _setup_py_macro_defines() -> list[str]:
    """The `-D` flags setup.py hands the CBoard extensions, as clang would see them."""
    source = (REPO_ROOT / "setup.py").read_text()
    block = re.search(
        r"_CBOARD_FAST_SLIDER_MACROS.*?=\s*\[(.*?)\]", source, re.DOTALL
    )
    assert block is not None, "could not find _CBOARD_FAST_SLIDER_MACROS in setup.py"
    pairs = re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', block.group(1))
    assert pairs, "found the macro list but parsed no entries"
    return sorted(f"-D{name}={value}" for name, value in pairs)


def _fuzz_script_defines() -> list[str]:
    """The `-D` flags the libFuzzer harness compiles CBoard with."""
    source = (REPO_ROOT / "scripts/fuzz/run_fuzz.sh").read_text()
    block = re.search(
        r"CBOARD_FAST_SLIDER_DEFINES=\((.*?)\)", source, re.DOTALL
    )
    assert block is not None, "could not find CBOARD_FAST_SLIDER_DEFINES in run_fuzz.sh"
    return sorted(re.findall(r"-D\S+", block.group(1)))


def test_the_fuzz_harness_compiles_the_same_sliders_production_does() -> None:
    """⚑ Two hand-maintained copies of one list, in two languages, pinned together.

    `scripts/fuzz/run_fuzz.sh` builds `cboard_libfuzzer.c` with clang directly,
    bypassing setup.py, so it needs its own copy of the macro list. Before this
    change it had none — meaning the C fuzz entry point exercised the LEGACY ray
    walkers while reporting coverage of "the CBoard C implementation", and the
    table-backed generator production runs was fuzzed by nothing at all.
    """
    assert _fuzz_script_defines() == _setup_py_macro_defines()
