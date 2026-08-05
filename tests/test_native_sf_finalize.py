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
        # The C mirror of `mate_to_effective_cp` hardcodes the band constants
        # (base 100000, step 100/ply, plies floored at 500). The regret vector
        # is a DIFFERENCE capped at SF_OWN_REGRET_CAP_CP = 1000cp, so it can
        # only see constants that change a gap of less than 1000cp:
        #   * mate-vs-cp gaps always saturate to 1.0, so they pin mate
        #     PRECEDENCE (row 4's cp 19998 outranked the mate under the old
        #     +-1500..2480 band) but NOT the base;
        #   * a mate PAIR 7 plies apart has a 700cp gap -> pins the step;
        # The ply floor cannot be pinned in THIS fixture — the mate-in-2 row
        # dominates it, so a floored and an unfloored long mate both saturate
        # to 1.0. `test_native_sf_finalize_pins_the_ply_floor` gives it a row
        # set where it is the best score and therefore observable.
        # The base is invisible here by construction: a shared offset cancels
        # in every difference. `test_mate_score_single_home.py::
        # test_c_mirror_literals_match_the_python_constants` pins it at source
        # level instead.
        if n_rows >= 7:
            rows[4] = (rows[4, 0], 19998, 0, -1, -1)
            rows[5] = (rows[5, 0], -32768, 2, -1, -1)
            rows[6] = (rows[6, 0], -32768, 9, -1, -1)

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


def test_native_sf_finalize_pins_the_ply_floor() -> None:
    """The C mirror's `plies > 500` floor, in a fixture that can SEE it.

    Regret is a difference capped at 1000cp, so the floor is only observable
    when the floored pair holds the best score: mate-in-500 and mate-in-600
    both map to 50000 once floored (gap 0 -> regret 0.0), and to 50000 vs
    40000 without the floor (gap 10000 -> regret saturates at 1.0). Deleting
    the floor from the C branch turns the second row 0.0 -> 1.0 here, while it
    is invisible in the randomized fixture above.
    """
    policy_encoding = POLICY_ENCODING_LC0_1858
    rows = np.zeros((2, 5), dtype=np.int16)
    rows[0] = (COMPACT_TO_FULL_POLICY[0], -32768, 500, -1, -1)
    rows[1] = (COMPACT_TO_FULL_POLICY[1], -32768, 600, -1, -1)

    expected_padded, expected_regret = _prepare_sf_multipv_python(
        rows, policy_encoding=policy_encoding, want_regret=True,
    )
    actual_padded, actual_regret = _prepare_sf_multipv_native(
        rows, policy_encoding=policy_encoding, want_regret=True,
    )
    np.testing.assert_array_equal(actual_padded, expected_padded)
    assert expected_regret is not None
    assert actual_regret is not None
    np.testing.assert_array_equal(actual_regret, expected_regret)
    # both mates are past the floor -> identical scores -> zero regret apart
    assert float(expected_regret[FULL_TO_COMPACT_POLICY[rows[0, 0]]]) == 0.0
    assert float(expected_regret[FULL_TO_COMPACT_POLICY[rows[1, 0]]]) == 0.0


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
