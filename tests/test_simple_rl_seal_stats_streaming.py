"""The collect seal's row statistics stream shard by shard and chunk by chunk.

The whole-generation load in ``generation_stats`` OOM-killed the pilot's
762k-row seal (~84 GB RSS). The streamed version must give the same numbers:
every statistic is a pooled mean / share / count, so chunking may change only
float summation order. The reference below is the pre-streaming
``policy_stats`` verbatim, run on the concatenated rows.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays
from chess_anti_engine.simple_rl import collect as collect_mod
from tests.test_simple_rl_root_stats import _sample


def _entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=-1)


def _reference_policy_stats(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """The pre-streaming collect.policy_stats, verbatim (whole-generation arrays)."""
    target = np.asarray(arrays["policy_target"], dtype=np.float64)
    target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-30)
    policy_rows = np.asarray(arrays.get("has_policy", np.ones(len(target))), dtype=bool)
    out: dict[str, float] = {
        "target_entropy": float(_entropy(target[policy_rows]).mean()) if policy_rows.any() else math.nan,
        "policy_rows": int(policy_rows.sum()),
    }
    has = np.asarray(arrays.get("has_root_log_prior", np.zeros(len(target))), dtype=bool)
    out["root_stats_rows"] = int(has.sum())
    out["rows_without_root_stats"] = int((~has).sum())
    if not has.any():
        return out
    for name in ("root_net_value", "root_survivor_q", "root_node_q"):
        if name in arrays:
            vals = np.asarray(arrays[name], dtype=np.float64)[has]
            out[f"mean_{name}"] = float(np.nanmean(vals))
    if "root_survivor_q" in arrays and "root_node_q" in arrays:
        gap = np.asarray(arrays["root_survivor_q"], np.float64)[has] - np.asarray(arrays["root_node_q"], np.float64)[has]
        out["mean_survivor_minus_node_q"] = float(np.nanmean(gap))
    if "root_start_visits" in arrays:
        sv = np.asarray(arrays["root_start_visits"], dtype=np.float64)[has]
        out.update(mean_root_start_visits=float(sv.mean()), reused_root_share=float((sv > 0).mean()))
    has &= policy_rows
    if not has.any():
        return out
    support = np.asarray(arrays["root_child_visits"])[has] >= 0
    logp = np.where(support, np.asarray(arrays["root_log_prior"], dtype=np.float64)[has], -np.inf)
    logp = logp - logp.max(axis=1, keepdims=True)
    prior = np.where(support, np.exp(logp), 0.0)
    prior /= prior.sum(axis=1, keepdims=True)
    t = target[has]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(t > 0, t * (np.log(np.where(t > 0, t, 1.0)) - np.log(np.where(prior > 0, prior, 1e-300))), 0.0).sum(axis=1)
    out.update(prior_entropy=float(_entropy(prior).mean()), kl_target_prior=float(kl.mean()),
               top1_agreement=float((t.argmax(axis=1) == prior.argmax(axis=1)).mean()))
    return out


def _shard(dirpath: Path, idx: int, rng: np.random.Generator, *, n: int, root: bool,
           policy: np.ndarray | None = None) -> None:
    arrs = samples_to_arrays([_sample(rng, with_root=root) for _ in range(n)])
    if policy is not None:
        arrs["has_policy"] = policy.astype(np.uint8)
    save_local_shard_arrays(dirpath / f"shard_{idx:05d}.zarr", arrs=arrs)
    np.savez(dirpath / f"shard_{idx:05d}.meta.npz", game_id=np.full(n, idx, dtype=np.int64),
             ply_index=np.arange(n, dtype=np.int32), opening_id=np.asarray([f"o{idx}"] * n),
             tb_policy_override=np.zeros(n, dtype=np.bool_), x_hash=rng.integers(0, 2**63, n).astype(np.uint64))


@pytest.fixture
def collect(tmp_path: Path) -> Path:
    d = tmp_path / "gen_0001" / "collect"
    (d / "shards").mkdir(parents=True)
    rng = np.random.default_rng(7)
    _shard(d / "shards", 0, rng, n=7, root=True, policy=np.array([1, 1, 0, 1, 1, 0, 1]))
    _shard(d / "shards", 1, rng, n=5, root=False)                          # no root-stats rows
    _shard(d / "shards", 2, rng, n=6, root=True, policy=np.zeros(6))       # no policy rows
    _shard(d / "shards", 3, rng, n=3, root=True, policy=np.ones(3))
    _shard(d / "shards", 4, rng, n=2, root=False)        # the LAST shard lacks the root fields: union, zero-filled
    return d


def _assert_same(got: dict, want: dict) -> float:
    assert set(got) == set(want), set(got) ^ set(want)
    worst = 0.0
    for k, w in want.items():
        g = got[k]
        if isinstance(w, int):
            assert g == w, k
        elif math.isnan(w):
            assert math.isnan(g), k
        else:
            worst = max(worst, abs(g - w) / max(abs(w), 1e-300))
            assert g == pytest.approx(w, rel=1e-9, abs=0.0), k
    return worst


@pytest.mark.parametrize("chunk", [1, 4, 5, 8192])
def test_streamed_generation_stats_equal_the_whole_generation_reference(collect: Path, chunk: int) -> None:
    arrays, meta = collect_mod.load_generation_rows(collect, keys=("x", *collect_mod.GENERATION_STATS_KEYS))
    want = _reference_policy_stats(arrays)
    rs, meta_s = collect_mod.stream_generation_row_stats(collect, chunk_rows=chunk)
    assert rs["rows"] == arrays["x"].shape[0] == 23
    _assert_same(rs["policy"], want)
    hb = np.asarray(arrays["has_search_budget"], bool)
    b = np.asarray(arrays["search_budget"])[hb]
    assert rs["rows_by_search_budget"] == {str(int(v)): int((b == v).sum()) for v in np.unique(b)}
    for k in meta:
        np.testing.assert_array_equal(meta_s[k], meta[k])


@pytest.mark.parametrize("chunk", [1, 3, 10**9])
def test_policy_stats_chunking_is_pooled_not_a_mean_of_means(collect: Path, chunk: int) -> None:
    """Shards refuse NaN, so the nanmean path is exercised on the arrays: NaNs in two chunks,
    one 3-row chunk (rows 18-20) all NaN."""
    arrays, _ = collect_mod.load_generation_rows(collect, keys=collect_mod.GENERATION_STATS_KEYS)
    arrays["root_node_q"] = np.asarray(arrays["root_node_q"], dtype=np.float32).copy()
    arrays["root_node_q"][[1, 4, 18, 19, 20]] = np.nan
    _assert_same(collect_mod.policy_stats(arrays, chunk_rows=chunk), _reference_policy_stats(arrays))


def test_all_nan_and_no_root_rows_follow_the_reference() -> None:
    acc = collect_mod.PolicyStatsAccumulator()
    t = np.full((2, 4), 0.25)
    acc.add({"policy_target": t, "has_root_log_prior": np.zeros(2), "has_policy": np.zeros(2)})
    out = acc.result()
    assert math.isnan(out["target_entropy"])
    assert out["policy_rows"] == 0
    assert out["rows_without_root_stats"] == 2
    assert "mean_root_node_q" not in out
    acc = collect_mod.PolicyStatsAccumulator()
    acc.add({"policy_target": t, "has_root_log_prior": np.ones(2), "has_policy": np.zeros(2),
             "root_node_q": np.array([np.nan, np.nan])})
    assert math.isnan(acc.result()["mean_root_node_q"])


def test_the_seal_never_loads_x(collect: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []
    real = collect_mod.gather_rows

    def spy(groups: Any, *, keys: tuple[str, ...] | None = None) -> Any:
        seen.append(tuple(keys or ("<all>",)))
        return real(groups, keys=keys)

    monkeypatch.setattr(collect_mod, "gather_rows", spy)
    collect_mod.stream_generation_row_stats(collect, chunk_rows=2)
    assert seen
    assert all("x" not in k and "<all>" not in k for k in seen)
    assert len(seen) == 5                                               # one load per shard


@pytest.mark.parametrize("pick", ["some", "none_in_a_shard", "nothing"])
def test_masked_generation_load_equals_the_whole_load_then_select(collect: Path, pick: str) -> None:
    """fit_value_recal --collect-dir and the model gates' hold-out read only the selected rows."""
    def select(m: dict[str, np.ndarray]) -> np.ndarray:
        gid = np.asarray(m["game_id"])
        if pick == "nothing":
            return np.zeros(len(gid), dtype=bool)
        return (np.asarray(m["ply_index"]) % 2 == 0) & (gid != 1 if pick == "none_in_a_shard" else True)

    keys = ("x", "policy_target", "has_policy")
    whole, meta = collect_mod.load_generation_rows(collect, keys=keys)
    mask = select(meta)
    got, got_meta = collect_mod.load_generation_rows_where(collect, select, keys=keys)
    for k in keys:
        np.testing.assert_array_equal(got[k], np.asarray(whole[k])[mask], err_msg=k)
    for k in meta:
        np.testing.assert_array_equal(got_meta[k], meta[k][mask], err_msg=k)
