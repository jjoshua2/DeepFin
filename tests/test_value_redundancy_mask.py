"""Unit tests for the value-redundancy keep-mask (offline data-selection screen).

Covers the four vr-modes, the policy_kl veto, missing-signal safety, and the
soft-mode effective drop fraction. The mask is a pure function over shard
arrays, so no model / C-extension is needed.
"""
from __future__ import annotations

import numpy as np

from scripts.offline_replay_epoch import _value_redundancy_keep_mask


def _arrs(n: int, *, q, sfe_search_v, sf_v, kl, has_policy=None) -> dict[str, np.ndarray]:
    """Build a minimal shard-arrays dict.

    sf_value_err = |E[search_wdl] - E[sf_wdl]|; we set W=P(win) columns so that
    (W - L) equals the requested expected value with D=0.
    """
    def wdl(vals):
        vals = np.asarray(vals, dtype=np.float64)
        w = np.clip((1.0 + vals) / 2.0, 0.0, 1.0)
        return np.stack([w, np.zeros_like(w), 1.0 - w], axis=1).astype(np.float32)

    hp = np.ones(n, dtype=bool) if has_policy is None else np.asarray(has_policy, dtype=bool)
    return {
        "x": np.zeros((n, 1), dtype=np.float32),
        "has_policy": hp,
        "priority_q_delta": np.asarray(q, dtype=np.float32),
        "has_priority_q_delta": np.ones(n, dtype=bool),
        "priority_policy_kl": np.asarray(kl, dtype=np.float32),
        "has_priority_policy_kl": np.ones(n, dtype=bool),
        "search_wdl": wdl(sfe_search_v),
        "has_search_wdl": np.ones(n, dtype=bool),
        "sf_wdl": wdl(sf_v),
        "has_sf_wdl": np.ones(n, dtype=bool),
    }


def _mask(
    arrs,
    mode,
    *,
    q_thresh: float = 0.05,
    sfe_thresh: float = 0.05,
    kl_veto: float = 3.0,
    soft_keep_prob: float = 0.5,
    mask_rng: np.random.RandomState | None = None,
):
    return _value_redundancy_keep_mask(
        arrs,
        mode=mode,
        q_thresh=q_thresh,
        sfe_thresh=sfe_thresh,
        kl_veto=kl_veto,
        soft_keep_prob=soft_keep_prob,
        mask_rng=mask_rng if mask_rng is not None else np.random.RandomState(0),
    )


def test_control_keeps_everything():
    a = _arrs(4, q=[0, 0, 0, 0], sfe_search_v=[0, 0, 0, 0], sf_v=[0, 0, 0, 0], kl=[0, 0, 0, 0])
    assert _mask(a, "control").all()


def test_hard_drops_value_converged_only():
    # row0: value-converged (low q, sfe=0, low kl) -> drop
    # row1: high q -> keep
    # row2: high sfe (search=0.5 vs sf=0.0) -> keep
    # row3: kl veto (kl high) -> keep
    a = _arrs(
        4,
        q=[0.01, 0.20, 0.01, 0.01],
        sfe_search_v=[0.0, 0.0, 0.5, 0.0],
        sf_v=[0.0, 0.0, 0.0, 0.0],
        kl=[1.0, 1.0, 1.0, 9.0],
    )
    keep = _mask(a, "hard")
    assert list(keep) == [False, True, True, True]


def test_sf_only_ignores_q():
    # sf_only drops on sfe+veto only; the high-q row0 is still dropped.
    a = _arrs(
        3,
        q=[0.99, 0.99, 0.99],
        sfe_search_v=[0.0, 0.5, 0.0],
        sf_v=[0.0, 0.0, 0.0],
        kl=[1.0, 1.0, 9.0],
    )
    keep = _mask(a, "sf_only")
    assert list(keep) == [False, True, True]  # row0 dropped, row1 high-sfe kept, row2 vetoed


def test_kl_veto_protects_policy_hard_rows():
    a = _arrs(2, q=[0.0, 0.0], sfe_search_v=[0.0, 0.0], sf_v=[0.0, 0.0], kl=[1.0, 5.0])
    keep = _mask(a, "hard", kl_veto=3.0)
    assert list(keep) == [False, True]


def test_non_full_policy_rows_are_kept():
    a = _arrs(2, q=[0.0, 0.0], sfe_search_v=[0.0, 0.0], sf_v=[0.0, 0.0], kl=[1.0, 1.0],
              has_policy=[True, False])
    keep = _mask(a, "hard")
    assert list(keep) == [False, True]  # row1 (no policy) kept


def test_missing_signal_is_noop():
    a = _arrs(2, q=[0.0, 0.0], sfe_search_v=[0.0, 0.0], sf_v=[0.0, 0.0], kl=[1.0, 1.0])
    del a["search_wdl"]  # cannot compute sf_value_err
    assert _mask(a, "hard").all()


def test_soft_drops_about_half_of_selected():
    n = 20000
    a = _arrs(
        n,
        q=np.full(n, 0.01),
        sfe_search_v=np.zeros(n),
        sf_v=np.zeros(n),
        kl=np.ones(n),
    )
    keep = _mask(a, "soft", soft_keep_prob=0.5, mask_rng=np.random.RandomState(1))
    dropped = n - int(keep.sum())
    assert 0.45 * n < dropped < 0.55 * n  # ~half of the all-selected rows


def test_soft_uses_dedicated_rng_not_main():
    # Two calls with the SAME mask_rng seed give identical masks (reproducible).
    a = _arrs(1000, q=np.full(1000, 0.01), sfe_search_v=np.zeros(1000),
              sf_v=np.zeros(1000), kl=np.ones(1000))
    m1 = _mask(a, "soft", mask_rng=np.random.RandomState(7))
    m2 = _mask(a, "soft", mask_rng=np.random.RandomState(7))
    assert np.array_equal(m1, m2)
