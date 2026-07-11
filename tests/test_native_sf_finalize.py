from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import (
    prepare_sf_multipv,  # pyright: ignore[reportAttributeAccessIssue]
)
from chess_anti_engine.moves import (
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_ENCODING_LC0_1858,
)
from chess_anti_engine.selfplay.finalize import (
    _prepare_sf_multipv_native,
    _prepare_sf_multipv_python,
)


@pytest.mark.parametrize("want_regret", [False, True])
def test_native_sf_finalize_matches_python_randomized(want_regret: bool) -> None:
    policy_encoding = POLICY_ENCODING_LC0_1858
    rng = np.random.default_rng(8127)
    valid_moves = np.asarray(COMPACT_TO_FULL_POLICY, dtype=np.int16)
    for n_rows in (0, 1, 7, 40, 55):
        rows = np.zeros((n_rows, 5), dtype=np.int16)
        if n_rows:
            rows[:, 0] = rng.choice(valid_moves, size=n_rows, replace=False)
            rows[:, 1] = rng.integers(-2500, 2501, size=n_rows, dtype=np.int16)
            rows[:, 3] = rng.integers(0, 1001, size=n_rows, dtype=np.int16)
            rows[:, 4] = rng.integers(0, 1001, size=n_rows, dtype=np.int16)
        if n_rows >= 1:
            rows[0] = (-2, -32768, 0, -1, -1)
        if n_rows >= 2:
            rows[1, 1:3] = (-32768, 0)
        if n_rows >= 3:
            rows[2, 2] = 5
        if n_rows >= 4:
            rows[3, 2] = -7

        expected_padded, expected_regret = _prepare_sf_multipv_python(
            rows, policy_encoding=policy_encoding, want_regret=want_regret,
        )
        actual_padded, actual_regret = _prepare_sf_multipv_native(
            rows, policy_encoding=policy_encoding, want_regret=want_regret,
        )

        np.testing.assert_array_equal(actual_padded, expected_padded)
        if expected_regret is None:
            assert actual_regret is None
        else:
            assert actual_regret is not None
            np.testing.assert_array_equal(actual_regret, expected_regret)


def test_native_sf_finalize_is_thread_safe() -> None:
    rows = np.zeros((40, 5), dtype=np.int16)
    rows[:, 0] = np.asarray(COMPACT_TO_FULL_POLICY[:40], dtype=np.int16)
    rows[:, 1] = np.arange(200, 160, -1, dtype=np.int16)
    expected = _prepare_sf_multipv_native(
        rows, policy_encoding=POLICY_ENCODING_LC0_1858, want_regret=True,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _idx: _prepare_sf_multipv_native(
                rows, policy_encoding=POLICY_ENCODING_LC0_1858, want_regret=True,
            ),
            range(64),
        ))

    for padded, regret in results:
        np.testing.assert_array_equal(padded, expected[0])
        assert regret is not None
        assert expected[1] is not None
        np.testing.assert_array_equal(regret, expected[1])


def test_native_sf_finalize_validates_input_contract() -> None:
    with pytest.raises(ValueError, match=r"shape \(K, 5\)"):
        prepare_sf_multipv(
            np.zeros((2, 4), dtype=np.int16),
            FULL_TO_COMPACT_POLICY,
            1858,
            48,
            -32768,
            1000.0,
            True,
        )

    rows = np.zeros((1, 5), dtype=np.int16)
    rows[0, 0] = np.int16(COMPACT_TO_FULL_POLICY[0])
    bad_map = FULL_TO_COMPACT_POLICY.copy()
    bad_map[int(rows[0, 0])] = 1858
    with pytest.raises(ValueError, match="out-of-range entry"):
        prepare_sf_multipv(rows, bad_map, 1858, 48, -32768, 1000.0, True)
