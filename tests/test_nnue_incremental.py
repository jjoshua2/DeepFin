"""Parity gates for the make-on-copy incremental NNUE qsearch path.

The production provider ``nnue-qsearch`` carries an NNUE accumulator forward
across the CBoard copies qsearch already makes.  ``nnue-qsearch-refresh`` keeps
the old full-refresh-at-every-node implementation as an explicit oracle.  The
two providers must return exactly the same values and walk exactly the same
search tree; only evaluator work is allowed to differ.

⚑⚑ THE FIXTURE IS THE GATE. A parity test is worth exactly what its weights
make observable, and the obvious synthetic pack makes almost nothing
observable: with a zero feature transformer and a zero FC net the accumulator
never reaches the output, so the two providers agree on every position no
matter how the incremental row updates behave. ``cae_nnue_inc_sub_row_i16``
could subtract nothing at all and CI would stay green.

So the pack here is built backwards from the arithmetic that has to be
exercised:

* ``ft_bias`` sits every accumulator lane in the transform's sensitive band —
  clipped to ``[0, 255]`` and multiplied pairwise, a lane pinned at either rail
  is as blind as a zero weight;
* ``ft_weight`` (int16) and ``threat_weight`` (int8) are non-zero on EVERY lane,
  so both accumulator row helpers move the value;
* ``fc0``'s forward-skip output carries weight 1 on every transformed lane. It
  is the only linear route from the transformer to the evaluation, and without
  it the FC net swallows the accumulator whole.

``test_the_parity_fixture_is_sensitive_to_both_accumulator_paths`` is the
non-vacuity gate for all of that, and it is the test to read first if this file
ever starts passing suspiciously.

⚑ WHAT THIS GATE CANNOT SEE, MEASURED RATHER THAN ASSUMED. The evaluation
truncates twice — ``/512`` inside the transform and ``/CAE_NNUE_OUTPUT_SCALE``
at the end — so a K-lane accumulator error of magnitude ``d`` moves the final
integer by roughly ``K * d * s / 6900``, where ``s`` is the partner lane's
level. This fixture's lanes measure mean 126, 1st-99th percentile 112-138, and
NEVER touch either clip rail, which puts the scale at:

    whole missed row (K=512)   ~9.3 cp     killed by mutant, below
    half a row       (K=256)   ~4.7 cp     killed by mutant, below
    one AVX2 block   (K= 16)   ~0.3 cp     SURVIVES — verified, it survives
    one lane         (K=  1)   ~0.02 cp    far below an integer evaluation

Raising the feature weights does not rescue the bottom two: the accumulator has
to stay inside the transform's ``[0, 255]`` clip, so lifting ``d`` lowers ``s``,
and the product peaks only ~1.4x above the operating point chosen here.

⇒ this file polices whole-row and sign errors in ``cae_nnue_inc_sub_row_i16`` /
``cae_nnue_inc_sub_row_i8`` and the HalfKA/threat diff plumbing, and it is
mutation-verified against exactly those, under BOTH kernels. A loop-BOUND error
inside one row helper is NOT observable end to end at any fixture — a deliberate
16-lane AVX2 truncation was run and passed — and catching that class needs the
helper exported and checked directly. Do not read a green run here as covering
it.
"""

from __future__ import annotations

import ast
import os
import random
import re
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_check_resolver import ROOK_TO_BACK_RANK, SCHOLAR_MATE
from tests.test_nnue_native_eval import BIG, POSITIONS, write_synthetic_pack

NNUE_DIR = Path(__file__).resolve().parents[1] / "chess_anti_engine" / "nnue"
ARM_PROVIDERS_H = NNUE_DIR / "_arm_providers.h"
NNUE_PROVIDER_H = NNUE_DIR / "_nnue_provider.h"

#: fc0 emits ``l2 + 1`` outputs per layer stack. Outputs ``0..l2-1`` feed the
#: squared/clipped pair fc1 consumes; output ``l2`` is the forward-skip term
#: ``cae_nnue_propagate`` scales straight into the result.
SKIP_OUTPUT = BIG.l2
FC0_OUTPUTS = BIG.l2 + 1
#: ``pack_layout`` pads fc0's input to ``pad(l1)``, which is l1 exactly at 1024.
FC0_PADDED_IN = BIG.l1

#: Feature-transformer bias on every accumulator lane.
#:
#: ⚑ THIS NUMBER IS THE FIXTURE'S SENSITIVITY. ``cae_nnue_transform`` clips each
#: lane to ``[0, 255]`` and forms ``clip(acc[j]) * clip(acc[j + half]) / 512``,
#: so the value's response to one accumulator unit is the PARTNER lane over 512.
#: Near zero the product vanishes and near 255 both lanes saturate; either rail
#: makes an incremental-update bug invisible. The row weights below are signed,
#: so they random-walk around this centre rather than climbing away from it:
#: measured over the sample boards the lanes sit at mean 126, 1st-99th
#: percentile 112-138, min 100 / max 151 — no position touches either rail.
FT_BIAS = 128


# ===========================================================================
# The synthetic pack — see the module docstring for why it is shaped this way
# ===========================================================================


@pytest.fixture(scope="module")
def accumulator_weights() -> tuple[np.ndarray, np.ndarray]:
    """Small HalfKA (int16) and threat (int8) feature rows, SIGNED PER ROW.

    Magnitudes are small so the arithmetic is exact: the accumulator sums one
    row per active feature — about 32 HalfKA rows and some tens of threat
    relations — and ``{0, 1, 2}`` / ``{0, 1}`` keep every lane inside the
    transform's ``[0, 255]`` clip around the ``FT_BIAS`` centre, where a clipped
    lane would be blind.

    ⚑⚑ THE SIGN IS PER ROW, AND THAT IS THE WHOLE POINT — MEASURED, NOT
    ARGUED. With every weight non-negative, a dropped subtract can only push an
    accumulator UP, so every corrupted child looks BETTER for the side to move
    and therefore WORSE once negamax negates it. Quiescence takes a MAX over
    children, so it discards all of them and returns the leaf's own stand-pat —
    which came from a full refresh and is correct. The mutant survives with
    byte-identical values AND byte-identical search stats: it was run, and the
    gate was blind to it.

    The sign is per ROW, not per weight: all 1024 lanes of one feature share it,
    so a dropped row still moves the evaluation coherently (~9cp, see the module
    docstring) instead of cancelling into a random walk — while WHICH way it
    moves now depends on the feature, so some children rise, others fall, and the
    max actually changes. A one-directional error under a max filter is
    invisible; this is what makes the gate able to fail at all.
    """
    rng = np.random.default_rng(20260825)
    halfka = rng.integers(0, 3, size=(nnue_parse.HALFKA_DIMS, BIG.l1), dtype=np.int16)
    halfka *= (rng.integers(0, 2, size=(nnue_parse.HALFKA_DIMS, 1), dtype=np.int16) * 2 - 1)
    threats = rng.integers(0, 2, size=(nnue_parse.THREAT_DIMS, BIG.l1), dtype=np.int8)
    threats *= (rng.integers(0, 2, size=(nnue_parse.THREAT_DIMS, 1), dtype=np.int8) * 2 - 1)
    return halfka, threats


def _ft_bias_blob() -> list[tuple[int, np.ndarray]]:
    return [(0, np.full(BIG.l1, FT_BIAS, dtype=np.int16))]


def _skip_row_blob() -> list[tuple[int, np.ndarray]]:
    """fc0's forward-skip output: ``+1`` on the side-to-move half, ``-1`` on the other.

    ⚑⚑ THE SIGN SPLIT IS WHAT MAKES THE GATE ABLE TO FAIL, AND IT WAS MEASURED
    THE HARD WAY. ``cae_nnue_transform`` writes perspective p0 (side to move)
    into ``ft[0:half]`` and p1 into ``ft[half:]``. Weight both halves ``+1`` and
    the evaluation is a large POSITIVE number for whoever is to move — about
    +2418 here — instead of an antisymmetric one. Quiescence is fail-soft
    negamax: it compares the node's stand-pat against ``-child``, so with a
    non-antisymmetric evaluation every child comes back around -2400, loses the
    max to the parent's own +2400, and the root just returns the leaf's
    stand-pat. That stand-pat is built by ``cae_nnue_state_init``, a FULL
    REFRESH — so no child accumulator reaches the output at all.

    Measured on this fixture: with both halves ``+1``, zeroing the entire child
    accumulator in ``cae_nnue_state_make`` left every value and every search
    counter byte-identical. The gate could not have failed.

    Weighting the halves oppositely makes the skip term antisymmetric, centres
    the evaluation on 0, and puts child values back in contention, which is what
    carries an incremental-update error into the result. The ceiling is still
    ``l1 * 126``, so ``cae_nnue_propagate``'s int32 forward-skip multiply cannot
    wrap.
    """
    row = np.ones(BIG.l1, dtype=np.int8)
    row[BIG.l1 // 2 :] = -1
    return [
        ((stack * FC0_OUTPUTS + SKIP_OUTPUT) * FC0_PADDED_IN, row)
        for stack in range(nnue_parse.LAYER_STACKS)
    ]


def _psqt_blobs() -> dict[str, list[tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(20260826)
    halfka = rng.integers(
        -32, 33, size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS, dtype=np.int32
    )
    threats = rng.integers(
        -32, 33, size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS, dtype=np.int32
    )
    return {"ft_psqt": [(0, halfka)], "threat_psqt": [(0, threats)]}


@pytest.fixture(scope="module")
def dense_pack(
    tmp_path_factory: pytest.TempPathFactory,
    accumulator_weights: tuple[np.ndarray, np.ndarray],
) -> Path:
    """The parity fixture: every feature reaches the value, by both routes."""
    halfka, threats = accumulator_weights
    path = tmp_path_factory.mktemp("nnue-incremental") / "dense.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_bias": _ft_bias_blob(),
            "ft_weight": [(0, halfka)],
            "threat_weight": [(0, threats)],
            "fc0_weight": _skip_row_blob(),
            **_psqt_blobs(),
        },
    )
    return path


@pytest.fixture(scope="module")
def bias_only_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """PSQT-free and feature-free: the accumulator is the bias for every board.

    The control for the sensitivity gate. Its evaluation cannot depend on the
    position through the transform at all, so any board where a feature pack
    disagrees with it is a board where the accumulator reached the output.
    """
    path = tmp_path_factory.mktemp("nnue-incremental") / "bias-only.pack"
    write_synthetic_pack(
        path, blobs={"ft_bias": _ft_bias_blob(), "fc0_weight": _skip_row_blob()}
    )
    return path


@pytest.fixture(scope="module")
def halfka_only_pack(
    tmp_path_factory: pytest.TempPathFactory,
    accumulator_weights: tuple[np.ndarray, np.ndarray],
) -> Path:
    """int16 HalfKA rows only — no PSQT, no threat rows."""
    halfka, _ = accumulator_weights
    path = tmp_path_factory.mktemp("nnue-incremental") / "halfka-only.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_bias": _ft_bias_blob(),
            "ft_weight": [(0, halfka)],
            "fc0_weight": _skip_row_blob(),
        },
    )
    return path


@pytest.fixture(scope="module")
def threat_only_pack(
    tmp_path_factory: pytest.TempPathFactory,
    accumulator_weights: tuple[np.ndarray, np.ndarray],
) -> Path:
    """int8 FullThreat rows only — no PSQT, no HalfKA rows."""
    _, threats = accumulator_weights
    path = tmp_path_factory.mktemp("nnue-incremental") / "threat-only.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_bias": _ft_bias_blob(),
            "threat_weight": [(0, threats)],
            "fc0_weight": _skip_row_blob(),
        },
    )
    return path


def _sample_boards() -> list[chess.Board]:
    """Deterministic natural positions, including checks and tactical leaves."""
    out = [chess.Board(fen) for fen in POSITIONS]
    out.extend([chess.Board(SCHOLAR_MATE), chess.Board(ROOK_TO_BACK_RANK)])

    rng = random.Random(20260825)
    for _game in range(5):
        board = chess.Board()
        for ply in range(28):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply in {7, 11, 15, 19, 23, 27}:
                out.append(board.copy(stack=True))
    return out


def _run(provider: str, pack: Path, boards: list[chess.Board]) -> tuple[list[int], dict[str, int]]:
    cboards = [CBoard.from_board(board) for board in boards]
    return _nnue_ext.arm_eval(provider, str(pack), cboards)


@pytest.fixture(autouse=True)
def _bounded_qsearch() -> Iterator[None]:
    """Keep the parity gate tactical but cheap; quiet checks are a separate cost axis."""
    _nnue_ext.set_arm_config(12, 3, 0)
    yield
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH,
        _nnue_ext.QSEARCH_MAX_PLY,
        _nnue_ext.QSEARCH_CHECK_PLIES,
    )


@pytest.fixture(params=[True, False], ids=["simd", "scalar"])
def kernel(request: pytest.FixtureRequest) -> Iterator[bool]:
    """Run a gate under BOTH accumulator kernels.

    ⚑ ``cae_nnue_simd_enabled`` defaults to ``CAE_NNUE_HAVE_AVX2``, so a
    single-run parity test exercises the AVX2 branch of
    ``cae_nnue_inc_sub_row_i16`` / ``cae_nnue_inc_sub_row_i8`` and NEVER their
    scalar fallbacks. Both branches are new code on the production path — the
    scalar one is what non-AVX2 workers run — and an unexercised branch is an
    unchecked one.
    """
    enabled = bool(request.param)
    if enabled and not _nnue_ext.HAVE_AVX2:
        pytest.skip("build has no AVX2 kernels")
    _nnue_ext.set_simd(enabled)
    yield enabled
    _nnue_ext.set_simd(bool(_nnue_ext.HAVE_AVX2))


# ===========================================================================
# Is the fixture able to see an accumulator bug at all?
# ===========================================================================


def test_the_parity_fixture_is_sensitive_to_both_accumulator_paths(
    bias_only_pack: Path, halfka_only_pack: Path, threat_only_pack: Path
) -> None:
    """⚑⚑ THE NON-VACUITY GATE FOR EVERY OTHER TEST IN THIS FILE.

    Both ablation packs have ZERO PSQT tables, so the only surviving route from
    a feature to the evaluation is accumulator row -> ``cae_nnue_transform`` ->
    fc0's forward-skip term. Disagreeing with the feature-free control is
    therefore proof that the route is live, per weight dtype:

    * ``halfka_only`` exercises ``cae_nnue_add_row_i16`` / ``cae_nnue_inc_sub_row_i16``
    * ``threat_only`` exercises ``cae_nnue_add_row_i8`` / ``cae_nnue_inc_sub_row_i8``

    Written as a majority of boards rather than "at least one" on purpose: one
    differing board is also what a fixture that has decayed to almost-blind
    produces, and the difference between the two is the whole point of the
    file.
    """
    boards = _sample_boards()
    bias, _ = _run("nnue-qsearch", bias_only_pack, boards)
    halfka, _ = _run("nnue-qsearch", halfka_only_pack, boards)
    threats, _ = _run("nnue-qsearch", threat_only_pack, boards)

    halfka_live = sum(a != b for a, b in zip(halfka, bias, strict=True))
    threats_live = sum(a != b for a, b in zip(threats, bias, strict=True))
    assert halfka_live > len(boards) // 2, (
        f"the int16 HalfKA accumulator lanes changed only {halfka_live}/{len(boards)} "
        "evaluations — the parity gate is close to blind"
    )
    assert threats_live > len(boards) // 2, (
        f"the int8 threat accumulator lanes changed only {threats_live}/{len(boards)} "
        "evaluations — the parity gate is close to blind"
    )


# ===========================================================================
# Incremental vs the refresh oracle
# ===========================================================================


def test_incremental_qsearch_is_exactly_the_refresh_search(
    dense_pack: Path, kernel: bool
) -> None:
    assert _nnue_ext.simd_active() is kernel
    boards = _sample_boards()
    inc_values, inc_stats = _run("nnue-qsearch", dense_pack, boards)
    ref_values, ref_stats = _run("nnue-qsearch-refresh", dense_pack, boards)

    assert inc_values == ref_values
    # Accumulator maintenance is not allowed to alter a cutoff, terminal, or
    # resolver decision. These stats describe the search tree, not the NNUE
    # implementation, so they must remain byte-for-byte equivalent as integers.
    assert inc_stats == ref_stats


def test_parity_fixture_really_exercises_qsearch(dense_pack: Path) -> None:
    boards = _sample_boards()
    static_values, _ = _run("nnue-static", dense_pack, boards)
    q_values, q_stats = _run("nnue-qsearch", dense_pack, boards)

    assert q_stats["qnodes"] > len(boards)
    assert any(a != b for a, b in zip(static_values, q_values, strict=True))


# ===========================================================================
# Which provider is wired to which implementation
# ===========================================================================

#: ``CaeValueProvider``'s fields, in declaration order. The initializers are
#: positional, so this list is what gives a parsed field its meaning.
PROVIDER_FIELDS = ("name", "init", "eval", "retain", "destroy", "kernel_name")

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_PROVIDER_STRUCT = re.compile(
    r"static\s+const\s+CaeValueProvider\s+(?P<symbol>\w+)\s*=\s*\{(?P<body>[^{}]*)\}\s*;",
    re.DOTALL,
)
_REGISTRY_ARRAY = re.compile(
    r"CAE_VALUE_PROVIDERS\[\]\s*=\s*\{(?P<body>[^{}]*)\}\s*;", re.DOTALL
)
_QSEARCH_WRAPPER = re.compile(
    r"cae_arm_qsearch_eval_(?P<mode>incremental|refresh)\s*\([^)]*\)\s*\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)
_MODE_CALL = re.compile(r"cae_arm_qsearch_eval_mode\s*\((?P<args>[^)]*)\)", re.DOTALL)


def _uncommented(path: Path) -> str:
    return _BLOCK_COMMENT.sub("", path.read_text(encoding="utf-8"))


def _provider_initializers(*paths: Path) -> dict[str, dict[str, str]]:
    """Parse every ``CaeValueProvider`` initializer into ``name -> field map``.

    ⚑⚑ THE TEST THIS REPLACES COULD NOT FAIL, IN EITHER DIRECTION. It asserted
    the source literal ``'"nnue-qsearch",\\n    cae_arm_qsearch_eval_incremental,'``
    — the provider name immediately followed by the eval callback. But ``eval``
    is the THIRD field and ``init`` is the second, so the literal does not occur
    in a correctly wired file and would not occur in a swapped one either. It
    was a tripwire strung across a corridor nobody walks down.

    Reading the initializer positionally is what makes a swap observable, and
    the field-count assertion is what stops the parse from quietly meaning
    something else after a struct change.
    """
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        for match in _PROVIDER_STRUCT.finditer(_uncommented(path)):
            symbol = match.group("symbol")
            fields = [f.strip() for f in match.group("body").split(",") if f.strip()]
            assert len(fields) == len(PROVIDER_FIELDS), (
                f"{symbol} has {len(fields)} initializer fields, expected "
                f"{len(PROVIDER_FIELDS)} {PROVIDER_FIELDS} — CaeValueProvider changed, "
                "so re-read the struct before trusting anything this parse says"
            )
            entry: dict[str, str] = dict(zip(PROVIDER_FIELDS, fields, strict=True))
            entry["symbol"] = symbol
            out[ast.literal_eval(entry["name"])] = entry
    return out


def test_production_provider_is_wired_to_incremental_and_oracle_to_refresh() -> None:
    """The struct field that decides which implementation each name gets."""
    providers = _provider_initializers(ARM_PROVIDERS_H)
    assert set(providers) == {"nnue-static", "nnue-qsearch", "nnue-qsearch-refresh"}

    assert providers["nnue-qsearch"]["eval"] == "cae_arm_qsearch_eval_incremental"
    assert providers["nnue-qsearch-refresh"]["eval"] == "cae_arm_qsearch_eval_refresh"
    assert providers["nnue-static"]["eval"] == "cae_arm_static_eval"


def test_the_two_qsearch_wrappers_pass_opposite_incremental_flags() -> None:
    """One frame below the struct, and invisible to the parity gate.

    Both providers route through ``cae_arm_qsearch_eval_mode``; only the trailing
    ``incremental`` argument separates them. Wire the struct correctly onto two
    wrappers that both pass 1 and the parity test compares the incremental path
    against itself — it agrees perfectly, and the oracle has quietly stopped
    being an oracle.
    """
    flags: dict[str, str] = {}
    for match in _QSEARCH_WRAPPER.finditer(_uncommented(ARM_PROVIDERS_H)):
        call = _MODE_CALL.search(match.group("body"))
        assert call is not None, (
            f"cae_arm_qsearch_eval_{match.group('mode')} no longer delegates to "
            "cae_arm_qsearch_eval_mode; this test can no longer read the flag"
        )
        flags[match.group("mode")] = call.group("args").split(",")[-1].strip()

    assert flags == {"incremental": "1", "refresh": "0"}


def test_the_registry_the_binary_reports_is_the_one_this_source_declares() -> None:
    """⚑ WHAT MAKES THE SOURCE PARSES ABOVE EVIDENCE ABOUT THE RUNNING BUILD.

    Every wiring test in this file reads ``_arm_providers.h`` off disk, which
    says nothing about the extension actually imported — an unrebuilt ``.so``
    would let all of them pass while the tree ran different code. Resolving the
    registry array's symbols back through the initializers gives the provider
    order this SOURCE declares; ``provider_names()`` gives the order the BINARY
    holds. Requiring them to be equal is the join.
    """
    providers = _provider_initializers(ARM_PROVIDERS_H, NNUE_PROVIDER_H)
    by_symbol = {entry["symbol"]: name for name, entry in providers.items()}

    match = _REGISTRY_ARRAY.search(_uncommented(ARM_PROVIDERS_H))
    assert match is not None, "CAE_VALUE_PROVIDERS[] initializer not found"
    declared = tuple(
        by_symbol[token.strip().lstrip("&").strip()]
        for token in match.group("body").split(",")
        if token.strip() and token.strip() != "NULL"
    )
    assert declared == _nnue_ext.provider_names()


@pytest.mark.skipif(not os.environ.get("CAE_NNUE_TEST_PACK"), reason="needs real NNUE pack")
def test_real_net_incremental_qsearch_matches_refresh() -> None:
    pack = Path(os.environ["CAE_NNUE_TEST_PACK"])
    boards = _sample_boards()[:20]
    inc_values, inc_stats = _run("nnue-qsearch", pack, boards)
    ref_values, ref_stats = _run("nnue-qsearch-refresh", pack, boards)
    assert inc_values == ref_values
    assert inc_stats == ref_stats
