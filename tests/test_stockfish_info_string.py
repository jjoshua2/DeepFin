"""UCI info strings are opaque debug text, not structured search fields."""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.stockfish.uci import _parse_info_fields, _SearchInfoAccumulator


@pytest.mark.parametrize('debug', ['score cp 999', 'score mate 1', 'wdl 0 0 1000', 'nodes 999 depth 999 multipv 2', 'lowerbound', 'upperbound', 'pv a1a8'])
def test_info_string_preserves_structured_fields(debug: str) -> None:
    parts = ('info depth 12 nodes 37 score cp 25 wdl 300 400 300 string ' + debug).split()
    assert _parse_info_fields(parts) == (None, 37, 12, 25, None, (300, 400, 300), None)
    acc = _SearchInfoAccumulator()
    acc.consume(parts)
    result = acc.result('e2e4')
    assert result.cp == 25
    assert result.mate is None
    assert result.depth == 12
    assert result.nodes == 37
    assert result.wdl is not None
    np.testing.assert_allclose(result.wdl, [0.3, 0.4, 0.3])
    assert result.pvs == []


def test_debug_only_line_preserves_last_snapshot() -> None:
    acc = _SearchInfoAccumulator()
    acc.consume(['info', 'depth', '9', 'nodes', '20', 'score', 'cp', '11'])
    acc.consume(['info', 'string', 'depth', '999', 'nodes', '999', 'score', 'mate', '1', 'pv', 'a1a8'])
    result = acc.result('e2e4')
    assert result.cp == 11
    assert result.mate is None
    assert result.depth == 9
    assert result.nodes == 20
    assert result.pvs == []


def test_debug_bound_after_pv_does_not_suppress_exact_score() -> None:
    acc = _SearchInfoAccumulator()
    acc.consume(['info', 'depth', '12', 'nodes', '37', 'score', 'cp', '25', 'pv', 'e2e4', 'string', 'lowerbound', 'upperbound'])
    result = acc.result('e2e4')
    assert result.cp == 25
    assert result.pvs[0].move_uci == 'e2e4'


def test_structured_bound_still_preserves_previous_exact_score() -> None:
    acc = _SearchInfoAccumulator()
    acc.consume(['info', 'depth', '12', 'nodes', '37', 'score', 'cp', '25'])
    acc.consume(['info', 'depth', '13', 'nodes', '38', 'score', 'cp', '60', 'lowerbound', 'string', 'score', 'cp', '999'])
    result = acc.result('e2e4')
    assert result.cp == 25
    assert result.depth == 13
    assert result.nodes == 38
