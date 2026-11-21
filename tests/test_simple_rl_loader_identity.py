"""The streaming row loader is value- and dtype-identical to the concatenating one.

``collect.gather_rows`` / ``learn.load_window`` replaced "load every shard, then
concatenate" (peak about twice the output, plus every target-source key). The
frozen copies below ARE that old loader; each target configuration -- stored
search WDL, survivor Q with the net's or a zero draw, the actor prior mix, and
a retargeted (no-rescale) collect -- must give the same training rows after
``prepare_targets``, the same held-out rows and the same presentation ids, on a
real two-generation multi-shard collect.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.simple_rl import learn
from chess_anti_engine.simple_rl.collect import (
    concat_arrays,
    generation_shards,
    load_generation_rows,
    resolve_collect_dir,
    shard_row_meta,
)
from chess_anti_engine.simple_rl.learn import TARGET_SOURCE_KEYS, TRAIN_KEYS, TrainSettings
from chess_anti_engine.simple_rl.retarget import retarget_collect
from scripts import simple_rl_loop
from tests.test_simple_rl_resume import _argv, _setup


# ---- the pre-change loader, frozen -------------------------------------------------------------
def _old_load_generation_rows(collect_dir: Path, *, keys: tuple[str, ...] | None = None) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray]]:
    from chess_anti_engine.replay.shard import load_shard_arrays

    shard_dir = resolve_collect_dir(collect_dir) / "shards"
    parts, metas = [], []
    for shard in sorted(shard_dir.glob("shard_*.zarr")):
        arrays, _ = load_shard_arrays(shard, lazy=False, validate=True)
        if keys is not None:
            arrays = {k: v for k, v in arrays.items() if k in keys or str(k).startswith("_")}
        parts.append(arrays)
        with np.load(shard.with_name(shard.name.replace(".zarr", ".meta.npz"))) as z:
            metas.append({k: z[k] for k in z.files})
    return concat_arrays(parts), {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}


def _old_load_window(run_dir: Path, generation: int, settings: TrainSettings) -> tuple[Any, ...]:
    gens = learn.phase_generations(generation, settings)
    train_parts, ids, holdout = [], [], None
    for g in gens:
        arrays, meta = _old_load_generation_rows(Path(run_dir) / f"gen_{g:04d}" / "collect",
                                                 keys=(*TRAIN_KEYS, *TARGET_SOURCE_KEYS))
        held = learn.holdout_mask(meta, settings.holdout_frac)
        train_parts.append(learn._select(arrays, ~held))
        rows = np.flatnonzero(~held)
        ids.append(np.stack([np.full(rows.size, g, dtype=np.int64), rows.astype(np.int64)], axis=1))
        if g == generation:
            holdout = learn._select(arrays, held)
    assert holdout is not None
    return concat_arrays(train_parts), holdout, gens, np.concatenate(ids, axis=0)
# -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("loader")
    mp = pytest.MonkeyPatch()
    try:
        openings, model_config = _setup(tmp_path, mp)
        run = tmp_path / "run"  # 2 generations x 4 games of 30 rows, two games per shard
        argv = _argv(run, openings, model_config)
        argv[argv.index("--rows-per-shard") + 1] = "50"
        assert simple_rl_loop.main(argv) == 0
        out = tmp_path / "retargeted"
        retarget_collect(source_run_dir=run, generation=1, out_run_dir=out)
    finally:
        mp.undo()
    return run, out


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert list(a) == list(b)
    for k in a:
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        assert va.dtype == vb.dtype, k
        assert va.shape == vb.shape, k
        assert np.array_equal(va, vb, equal_nan=va.dtype.kind == "f"), k


SETTINGS = {
    "stored_search_wdl": TrainSettings(presentations=2, holdout_frac=0.25, value_search_weight=0.5,
                                       value_search_source="stored_search_wdl"),
    "survivor_q_net_draw": TrainSettings(presentations=2, holdout_frac=0.25, value_search_weight=0.5,
                                         value_search_source="survivor_q", value_draw_source="net"),
    "survivor_q_zero_draw": TrainSettings(presentations=2, holdout_frac=0.25, value_search_weight=0.5,
                                          value_search_source="survivor_q", value_draw_source="zero"),
    "actor_prior_mix": TrainSettings(presentations=2, holdout_frac=0.25, policy_prior_mix=0.6,
                                     prior_mix_source="actor"),
    "defaults": TrainSettings(presentations=2, holdout_frac=0.25),
}


@pytest.mark.parametrize("source", sorted(SETTINGS))
def test_the_streaming_window_equals_the_concatenating_one(runs: tuple[Path, Path], source: str) -> None:
    run, _ = runs
    settings = SETTINGS[source]
    shards = generation_shards(run / "gen_0002" / "collect")
    assert len(shards) > 1  # several shards per generation
    held = [learn.holdout_mask(shard_row_meta(s), settings.holdout_frac) for s in shards]
    assert any(0 < h.sum() < h.size for h in held)  # a shard split between train and held-out rows
    old = _old_load_window(run, 2, settings)
    new = learn.load_window(run, 2, settings)
    assert old[2] == new[2] == [1, 2]
    assert np.array_equal(old[3], new[3])
    old_train, old_info = learn.prepare_targets(old[0], settings)
    new_train, new_info = learn.prepare_targets(new[0], settings)
    assert old_info == new_info
    _same(old_train, new_train)
    _same({k: v for k, v in old[1].items() if k not in TARGET_SOURCE_KEYS},
          {k: v for k, v in new[1].items() if k not in TARGET_SOURCE_KEYS})
    if source == "survivor_q_net_draw":
        assert new_info["value_from_survivor_q"] > 0  # the source keys really were loaded and used
    if source == "actor_prior_mix":
        assert new_info["prior_mixed_rows"] > 0


def test_a_retargeted_collect_loads_identically(runs: tuple[Path, Path]) -> None:
    _, retargeted = runs
    for settings in (TrainSettings(presentations=1, holdout_frac=0.25),
                     TrainSettings(presentations=1, holdout_frac=0.25, value_search_weight=0.5)):
        old = _old_load_window(retargeted, 1, settings)
        new = learn.load_window(retargeted, 1, settings)
        assert np.array_equal(old[3], new[3])
        _same(learn.prepare_targets(old[0], settings)[0], learn.prepare_targets(new[0], settings)[0])
        _same({k: v for k, v in old[1].items() if k not in TARGET_SOURCE_KEYS},
              {k: v for k, v in new[1].items() if k not in TARGET_SOURCE_KEYS})


@pytest.mark.parametrize("keys", [None, TRAIN_KEYS], ids=["all_keys", "train_keys"])
def test_load_generation_rows_equals_the_concatenating_one(runs: tuple[Path, Path], keys: Any) -> None:
    run, _ = runs
    old_rows, old_meta = _old_load_generation_rows(run / "gen_0001" / "collect", keys=keys)
    new_rows, new_meta = load_generation_rows(run / "gen_0001" / "collect", keys=keys)
    _same(old_rows, new_rows)
    _same(old_meta, new_meta)


def test_only_the_configured_target_source_keys_are_loaded() -> None:
    assert learn.target_source_keys(TrainSettings()) == ()
    assert learn.target_source_keys(SETTINGS["stored_search_wdl"]) == ()
    assert set(learn.target_source_keys(SETTINGS["survivor_q_net_draw"])) == {
        "root_survivor_q", "has_root_survivor_q", "root_net_wdl", "has_root_net_wdl"}
    assert set(learn.target_source_keys(SETTINGS["survivor_q_zero_draw"])) == {
        "root_survivor_q", "has_root_survivor_q"}
    assert set(learn.target_source_keys(SETTINGS["actor_prior_mix"])) == {
        "has_root_log_prior", "root_log_prior", "root_child_visits"}


def test_a_field_missing_from_one_shard_is_zero_filled_as_before(runs: tuple[Path, Path], tmp_path: Path) -> None:
    """Mixed-schema shards: concat_arrays zero-fills an absent field with the schema default."""
    import shutil

    from chess_anti_engine.replay.shard import load_shard_arrays, save_local_shard_arrays

    run, _ = runs
    src = generation_shards(run / "gen_0001" / "collect")[:2]
    out = tmp_path / "gen_0001" / "collect" / "shards"
    out.mkdir(parents=True)
    for i, shard in enumerate(src):
        arrays, meta = load_shard_arrays(shard, lazy=False, validate=True)
        if i == 1:  # the second shard lacks an optional field pair
            arrays = {k: v for k, v in arrays.items() if k not in ("search_wdl", "has_search_wdl")}
        save_local_shard_arrays(out / shard.name, arrs={k: v for k, v in arrays.items()
                                                        if not str(k).startswith("_")}, meta=meta)
        meta_npz = shard.with_name(shard.name.replace(".zarr", ".meta.npz"))
        shutil.copy2(meta_npz, out / meta_npz.name)
    collect = tmp_path / "gen_0001" / "collect"
    old_rows, _ = _old_load_generation_rows(collect, keys=TRAIN_KEYS)
    new_rows, _ = load_generation_rows(collect, keys=TRAIN_KEYS)
    assert "search_wdl" in new_rows
    _same(old_rows, new_rows)


@pytest.mark.parametrize("source", ["actor", "reference"])
def test_the_chunked_prior_mix_equals_the_whole_window_one(runs: tuple[Path, Path], source: str,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """prepare_targets' prior mix runs in row chunks; a chunk larger than the window
    is the pre-change whole-window computation, and every row is row-local."""
    run, _ = runs
    settings = TrainSettings(presentations=2, holdout_frac=0.25, policy_prior_mix=0.6, prior_mix_source=source)
    rows = learn.load_window(run, 2, settings)[0]
    if source == "reference":
        rows = {**rows, **_old_load_window(run, 2, settings)[0]}

    def reference(x: np.ndarray, legal: np.ndarray) -> np.ndarray:  # a deterministic row-local prior
        w = np.abs(np.asarray(x, dtype=np.float64).reshape(len(x), -1)[:, :1858]) + 1.0
        return np.where(legal, w, 0.0)

    ref = reference if source == "reference" else None
    monkeypatch.setattr(learn, "PRIOR_MIX_CHUNK_ROWS", 10**9)
    whole, whole_info = learn.prepare_targets(rows, settings, ref)
    monkeypatch.setattr(learn, "PRIOR_MIX_CHUNK_ROWS", 7)
    chunked, chunked_info = learn.prepare_targets(rows, settings, ref)
    assert whole_info == chunked_info
    assert whole_info["prior_mixed_rows"] > 7  # several chunks really ran
    _same(whole, chunked)


def test_gather_rows_fills_every_group_at_its_own_offsets(runs: tuple[Path, Path]) -> None:
    """Two groups drawing interleaved, partial row masks from the same shards each
    equal the concatenation of their masked per-shard arrays."""
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.simple_rl.collect import gather_rows

    run, _ = runs
    shards = [*generation_shards(run / "gen_0001" / "collect"), *generation_shards(run / "gen_0002" / "collect")]
    loaded = [load_shard_arrays(s, lazy=False, validate=True)[0] for s in shards]
    rng = np.random.default_rng(0)
    masks = [rng.random(int(a["x"].shape[0])) < 0.5 for a in loaded]
    groups = [[(s, m) for s, m in zip(shards, masks, strict=True)],
              [(s, ~m) for s, m in zip(shards, masks, strict=True)][::-1]]
    got = gather_rows(groups, keys=TRAIN_KEYS)
    for group, out in zip(groups, got, strict=True):
        parts = [learn._select({k: v for k, v in loaded[shards.index(s)].items()
                                if k in TRAIN_KEYS or str(k).startswith("_")}, m) for s, m in group]
        _same(concat_arrays(parts), out)
