"""Parity gates for the make-on-copy incremental NNUE qsearch path.

The production provider ``nnue-qsearch`` carries an NNUE accumulator forward
across the CBoard copies qsearch already makes.  ``nnue-qsearch-refresh`` keeps
the old full-refresh-at-every-node implementation as an explicit oracle.  The
two providers must return exactly the same values and walk exactly the same
search tree; only evaluator work is allowed to differ.

The synthetic pack deliberately has dense, non-zero PSQT feature weights.  A
zero/bucket-only pack would let a broken accumulator update pass because moving
a piece would not change the value.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_check_resolver import ROOK_TO_BACK_RANK, SCHOLAR_MATE
from tests.test_nnue_native_eval import POSITIONS, write_synthetic_pack


@pytest.fixture(scope="module")
def dense_psqt_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A cheap synthetic net where every active feature can affect the value."""
    rng = np.random.default_rng(20260825)
    halfka = rng.integers(
        -32,
        33,
        size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    threats = rng.integers(
        -32,
        33,
        size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    path = tmp_path_factory.mktemp("nnue-incremental") / "dense-psqt.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_psqt": [(0, halfka)],
            "threat_psqt": [(0, threats)],
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


def test_incremental_qsearch_is_exactly_the_refresh_search(dense_psqt_pack: Path) -> None:
    boards = _sample_boards()
    inc_values, inc_stats = _run("nnue-qsearch", dense_psqt_pack, boards)
    ref_values, ref_stats = _run("nnue-qsearch-refresh", dense_psqt_pack, boards)

    assert inc_values == ref_values
    # Accumulator maintenance is not allowed to alter a cutoff, terminal, or
    # resolver decision. These stats describe the search tree, not the NNUE
    # implementation, so they must remain byte-for-byte equivalent as integers.
    assert inc_stats == ref_stats


def test_parity_fixture_really_exercises_qsearch(dense_psqt_pack: Path) -> None:
    boards = _sample_boards()
    static_values, _ = _run("nnue-static", dense_psqt_pack, boards)
    q_values, q_stats = _run("nnue-qsearch", dense_psqt_pack, boards)

    assert q_stats["qnodes"] > len(boards)
    assert any(a != b for a, b in zip(static_values, q_values, strict=True))


def test_production_provider_is_wired_to_incremental_and_oracle_to_refresh() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "chess_anti_engine"
        / "nnue"
        / "_arm_providers.h"
    ).read_text(encoding="utf-8")
    assert '"nnue-qsearch",\n    cae_arm_qsearch_eval_incremental,' in source
    assert '"nnue-qsearch-refresh",\n    cae_arm_qsearch_eval_refresh,' in source


@pytest.mark.skipif(not os.environ.get("CAE_NNUE_TEST_PACK"), reason="needs real NNUE pack")
def test_real_net_incremental_qsearch_matches_refresh() -> None:
    pack = Path(os.environ["CAE_NNUE_TEST_PACK"])
    boards = _sample_boards()[:20]
    inc_values, inc_stats = _run("nnue-qsearch", pack, boards)
    ref_values, ref_stats = _run("nnue-qsearch-refresh", pack, boards)
    assert inc_values == ref_values
    assert inc_stats == ref_stats
