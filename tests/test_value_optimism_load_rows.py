"""Shard-reading tests for ``scripts/value_optimism.py::_load_rows``.

The scorer's arithmetic is covered by ``tests/test_value_optimism.py``. This
file covers the part that decides WHICH ROWS the arithmetic runs on, over real
zarr shards, because every defect that has actually bitten this measurement
lived there rather than in the maths: the one-ply shift picking up a row that is
not the predecessor, a pair forming across a shard boundary, a shard whose SF
labels are detached being scored anyway, and — the one that inverted a
conclusion — the paired set silently being a single population.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from chess_anti_engine.eval.value_optimism import SF_LABEL_ATTACHMENT_MIN
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from scripts.value_optimism import _load_rows

SLOPE = 0.0060
DRAW_WIDTH = 120.0
ENCODING = "lc0_root_legacy_meta"
N_PLANES = 175


def _write_shard(
    path: Path, *, game_id: list[int], ply_index: list[int],
    is_selfplay: list[int] | None = None, mate: list[int] | None = None,
    scramble_labels: bool = False, encoding: str = ENCODING,
    history_rep_fix: bool = True, seed: int = 3,
) -> np.ndarray:
    """Write a synthetic shard; return each row's record-POV material balance.

    Rows are built so the SF label is genuinely ATTACHED to its own position:
    ``sf_label_meta`` cp is the eval AFTER the row's move (opponent POV, so the
    negation of the row's material) and ``sf_wdl`` is that same eval flipped
    back to record POV. That is the production relation the attachment gate
    checks, so a shard from this helper passes the gate unless
    ``scramble_labels`` deliberately breaks it.
    """
    n = len(game_id)
    x = np.zeros((n, N_PLANES, 8, 8), dtype=np.float16)
    # A deterministic material LADDER, not random pieces: random counts tie at
    # small n, rank_corr is then undefined, and the attachment gate rejects the
    # shard for a reason that has nothing to do with what the test is about.
    material = np.array([((i + seed) % 9) - 4 for i in range(n)], dtype=np.float64)
    for i in range(n):
        white = int(max(material[i], 0.0))
        black = int(max(-material[i], 0.0))
        for k in range(white):
            x[i, 0, 1, k] = 1.0          # white pawns on plane 0
        for k in range(black):
            x[i, 6, 6, k] = 1.0          # black pawns on plane 6

    mates = list(mate) if mate is not None else [0] * n
    meta = np.zeros((n, 6), dtype=np.int32)
    meta[:, 0] = 698000
    meta[:, 1] = 12
    sf_wdl = np.zeros((n, 3), dtype=np.float16)
    for i in range(n):
        if mates[i]:
            meta[i, 2] = -32768
            meta[i, 3] = mates[i]
            # Record POV is the flip of the opponent-POV mate score.
            probs = cp_to_wdl(None, mates[i], slope=SLOPE, draw_width_cp=DRAW_WIDTH)[::-1]
        else:
            meta[i, 2] = round(-material[i] * 100.0)
            probs = cp_to_wdl(
                float(material[i] * 100.0), None, slope=SLOPE, draw_width_cp=DRAW_WIDTH,
            )
        sf_wdl[i] = probs
    if scramble_labels:
        perm = np.random.default_rng(99).permutation(n)
        meta = meta[perm]
        sf_wdl = sf_wdl[perm]

    g = zarr.open_group(str(path), mode="w")
    g.create_dataset("x", data=x, overwrite=True)
    g.create_dataset("game_id", data=np.asarray(game_id, dtype=np.int64), overwrite=True)
    g.create_dataset("ply_index", data=np.asarray(ply_index, dtype=np.int32), overwrite=True)
    g.create_dataset("sf_label_meta", data=meta, overwrite=True)
    g.create_dataset("has_sf_label_meta", data=np.ones(n, dtype=np.uint8), overwrite=True)
    g.create_dataset("sf_wdl", data=sf_wdl, overwrite=True)
    g.create_dataset("has_sf_wdl", data=np.ones(n, dtype=np.uint8), overwrite=True)
    g.create_dataset(
        "search_wdl", data=np.tile(np.array([0.3, 0.4, 0.3], dtype=np.float16), (n, 1)), overwrite=True,
    )
    g.create_dataset("has_search_wdl", data=np.ones(n, dtype=np.uint8), overwrite=True)
    g.create_dataset("wdl_target", data=np.zeros(n, dtype=np.int8), overwrite=True)
    sp = np.ones(n, dtype=np.uint8) if is_selfplay is None else np.asarray(is_selfplay, np.uint8)
    g.create_dataset("is_selfplay", data=sp, overwrite=True)
    g.attrs["input_history_encoding"] = encoding
    g.attrs["history_rep_fix"] = history_rep_fix
    return material


def _load(paths: list[Path], **kw) -> dict[str, np.ndarray]:
    args: dict[str, object] = {
        "max_rows": 0, "slope": SLOPE, "draw_width": DRAW_WIDTH,
        "history_encoding": ENCODING, "history_rep_fix": True,
        "attachment_min": SF_LABEL_ATTACHMENT_MIN,
    }
    args.update(kw)
    return _load_rows(paths, **args)  # pyright: ignore[reportArgumentType]


def test_pairing_requires_the_immediately_preceding_ply(tmp_path: Path) -> None:
    """A gap of more than one ply must NOT pair.

    Without this the shift silently reads an eval of a position several plies
    back, which still looks plausible and is wrong.
    """
    p = tmp_path / "shard_000001.zarr"
    material = _write_shard(p, game_id=[7] * 8, ply_index=[1, 2, 4, 5, 9, 10, 20, 30])
    out = _load([p])
    # Rows at index 1, 3 and 5 follow their predecessor by exactly one ply.
    assert out["ply_index"].tolist() == [2, 5, 10]
    # And the P0 ruler is the PREDECESSOR's stored eval, taken WITHOUT a flip:
    # that eval is already in the point of view of the side to move after the
    # predecessor's move, which is this row's mover. The all-rows arm below
    # negates instead, and the two together pin both conventions.
    expected = [round(-material[i] * 100.0) for i in (0, 2, 4)]
    assert out["sf_label_meta"].tolist() == pytest.approx(expected)


def test_pairs_never_form_across_a_shard_boundary(tmp_path: Path) -> None:
    """Contiguous game/ply either side of a boundary must still not pair.

    Shards are independent files that need not be adjacent in game order, so a
    cross-file pair would join two unrelated positions.
    """
    a = tmp_path / "shard_000001.zarr"
    b = tmp_path / "shard_000002.zarr"
    _write_shard(a, game_id=[4] * 4, ply_index=[10, 11, 12, 13])
    _write_shard(b, game_id=[4] * 4, ply_index=[14, 15, 16, 17], seed=5)
    out = _load([a, b])
    # Pairs form inside each shard, but ply 13 -> ply 14 spans the boundary and
    # must not, even though the game and ply numbers are contiguous.
    assert sorted(out["ply_index"].tolist()) == [11, 12, 13, 15, 16, 17]


def test_shard_with_detached_labels_is_rejected_entirely(tmp_path: Path) -> None:
    """The 2026-07-31 defect must remove the shard from BOTH arms."""
    good = tmp_path / "shard_000001.zarr"
    bad = tmp_path / "shard_000002.zarr"
    _write_shard(good, game_id=[1] * 8, ply_index=list(range(1, 9)))
    _write_shard(bad, game_id=[2] * 40, ply_index=list(range(1, 41)), scramble_labels=True)
    out = _load([good, bad])
    assert int(out["_shards_rejected"][0]) == 1
    assert set(np.unique(out["game_id"]).tolist()) == {1}
    # The rejected shard must not leak into the model-free arm either.
    assert out["_all_game_id"].size == 8


def test_encoding_mismatch_skips_the_shard(tmp_path: Path) -> None:
    p = tmp_path / "shard_000001.zarr"
    q = tmp_path / "shard_000002.zarr"
    _write_shard(p, game_id=[1] * 8, ply_index=list(range(1, 9)))
    _write_shard(q, game_id=[2] * 8, ply_index=list(range(1, 9)), encoding="legacy")
    out = _load([p, q])
    assert set(np.unique(out["game_id"]).tolist()) == {1}
    r = tmp_path / "shard_000003.zarr"
    _write_shard(r, game_id=[3] * 8, ply_index=list(range(1, 9)), history_rep_fix=False)
    out2 = _load([p, r])
    assert set(np.unique(out2["game_id"]).tolist()) == {1}


def test_all_rows_arm_keeps_rows_the_paired_arm_drops(tmp_path: Path) -> None:
    """The arm that can see curriculum must not inherit the pairing filter.

    This is the property whose absence made a 100%-selfplay sample look like
    "the training distribution".
    """
    p = tmp_path / "shard_000001.zarr"
    material = _write_shard(
        p, game_id=[1, 1, 1, 1, 2, 2], ply_index=[1, 2, 5, 9, 14, 20],
        is_selfplay=[1, 1, 1, 1, 0, 0],
    )
    out = _load([p])
    assert out["game_id"].size == 1                 # only ply 1 -> 2 pairs
    assert out["_all_game_id"].size == 6            # every labelled row survives
    assert out["_all_is_selfplay"].astype(bool).tolist() == [True] * 4 + [False] * 2
    # The all-rows ruler is the row's OWN eval negated into the scored side's POV.
    assert out["_all_ruler_cp"].tolist() == pytest.approx(
        [round(m * 100.0) for m in material], abs=1e-6,
    )


def test_mate_scores_fold_into_the_ruler(tmp_path: Path) -> None:
    """A mate row carries no cp; dropping it would bias the losing tail."""
    p = tmp_path / "shard_000001.zarr"
    _write_shard(p, game_id=[1] * 8, ply_index=list(range(1, 9)), mate=[3] + [0] * 7)
    out = _load([p])
    # Row 0's label is "mate in 3 for the side to move AFTER row 0's move" — and
    # that side IS row 1's mover, so row 1's ruler is a large POSITIVE cp. The
    # POV is the whole subtlety of the shift; a sign slip here would invert the
    # losing tail.
    assert out["ply_index"][0] == 2
    assert out["sf_label_meta"][0] > 1000.0


def test_max_rows_truncates_and_stops(tmp_path: Path) -> None:
    a = tmp_path / "shard_000001.zarr"
    b = tmp_path / "shard_000002.zarr"
    _write_shard(a, game_id=[1] * 8, ply_index=list(range(1, 9)))
    _write_shard(b, game_id=[2] * 8, ply_index=list(range(1, 9)), seed=5)
    out = _load([a, b], max_rows=3)
    assert out["game_id"].size == 3
    assert set(np.unique(out["game_id"]).tolist()) == {1}


def test_no_usable_rows_is_a_hard_exit_naming_the_gate(tmp_path: Path) -> None:
    bad = tmp_path / "shard_000001.zarr"
    _write_shard(bad, game_id=[1] * 40, ply_index=list(range(1, 41)), scramble_labels=True)
    with pytest.raises(SystemExit, match="attachment gate"):
        _load([bad])
