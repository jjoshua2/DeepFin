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

from chess_anti_engine.eval.value_optimism import (
    SF_DESYNC_MAX,
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
)
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from scripts.value_optimism import _load_rows

SLOPE = 0.0060
DRAW_WIDTH = 120.0
ENCODING = "lc0_root_legacy_meta"
N_PLANES = 175


def _write_shard(
    path: Path, *, game_id: list[int], ply_index: list[int],
    is_selfplay: list[int] | None = None, mate: list[int] | None = None,
    scramble_labels: bool = False, desync_frac: float = 0.0, multipv_miss: float = 0.0,
    encoding: str = ENCODING, history_rep_fix: bool = True, seed: int = 3,
    material_override: list[float] | None = None,
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
    material = (np.array(material_override, dtype=np.float64) if material_override is not None
                else np.array([((i + seed) % 9) - 4 for i in range(n)], dtype=np.float64))
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
    # sf_move_index / sf_legal_mask feed the desync axis of the gate. A sound
    # row points at a move that is NOT the first legal one; a desynced row falls
    # back to legal_indices[0], which is what the detector counts.
    legal = np.zeros((n, 1858), dtype=np.uint8)
    legal[:, 10] = 1
    legal[:, 40] = 1
    legal[:, 900] = 1
    move_idx = np.full(n, 40, dtype=np.int32)
    n_desync = round(desync_frac * n)
    move_idx[:n_desync] = 10                      # == first legal index
    g.create_dataset("sf_move_index", data=move_idx, overwrite=True)
    g.create_dataset("has_sf_move", data=np.ones(n, dtype=np.uint8), overwrite=True)
    g.create_dataset("sf_legal_mask", data=legal, overwrite=True)
    g.create_dataset("has_sf_legal_mask", data=np.ones(n, dtype=np.uint8), overwrite=True)
    n_miss = round(multipv_miss * n)
    hmv = np.ones(n, dtype=np.uint8)
    hmv[:n_miss] = 0
    g.create_dataset("has_sf_multipv_raw", data=hmv, overwrite=True)
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
        "attachment_min": SF_LABEL_ATTACHMENT_MIN, "desync_max": SF_DESYNC_MAX,
        "multipv_miss_max": SF_MULTIPV_MISS_MAX,
    }
    args.update(kw)
    return _load_rows(paths, **args)  # pyright: ignore[reportArgumentType]


def test_pairing_requires_the_immediately_preceding_ply(tmp_path: Path) -> None:
    """A gap of more than one ply must NOT pair.

    Without this the shift silently reads an eval of a position several plies
    back, which still looks plausible and is wrong. (Shards are >= 30 rows
    throughout this file: below that the desync detector cannot estimate a rate
    and correctly refuses to clear the shard.)
    """
    p = tmp_path / "shard_000001.zarr"
    plies = [1, 2, 4, 5, 9, 10] + [20 + 10 * i for i in range(34)]
    material = _write_shard(p, game_id=[7] * len(plies), ply_index=plies)
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
    _write_shard(a, game_id=[4] * 40, ply_index=list(range(10, 50)))
    _write_shard(b, game_id=[4] * 40, ply_index=list(range(50, 90)), seed=5)
    out = _load([a, b])
    plies = sorted(out["ply_index"].tolist())
    # Every row but the first of each shard pairs...
    assert plies == list(range(11, 50)) + list(range(51, 90))
    # ...and ply 50 does NOT, even though game and ply are contiguous with 49.
    assert 50 not in plies


def test_shard_with_detached_labels_is_rejected_entirely(tmp_path: Path) -> None:
    """The 2026-07-31 defect must remove the shard from BOTH arms."""
    good = tmp_path / "shard_000001.zarr"
    bad = tmp_path / "shard_000002.zarr"
    _write_shard(good, game_id=[1] * 40, ply_index=list(range(1, 41)))
    _write_shard(bad, game_id=[2] * 40, ply_index=list(range(1, 41)), scramble_labels=True)
    out = _load([good, bad])
    assert int(out["_shards_rejected"][0]) == 1
    assert set(np.unique(out["game_id"]).tolist()) == {1}
    # The rejected shard must not leak into the model-free arm either.
    assert out["_all_game_id"].size == 40


def test_encoding_mismatch_skips_the_shard(tmp_path: Path) -> None:
    p = tmp_path / "shard_000001.zarr"
    q = tmp_path / "shard_000002.zarr"
    _write_shard(p, game_id=[1] * 40, ply_index=list(range(1, 41)))
    _write_shard(q, game_id=[2] * 40, ply_index=list(range(1, 41)), encoding="legacy")
    out = _load([p, q])
    assert set(np.unique(out["game_id"]).tolist()) == {1}
    r = tmp_path / "shard_000003.zarr"
    _write_shard(r, game_id=[3] * 40, ply_index=list(range(1, 41)), history_rep_fix=False)
    out2 = _load([p, r])
    assert set(np.unique(out2["game_id"]).tolist()) == {1}


def test_all_rows_arm_keeps_rows_the_paired_arm_drops(tmp_path: Path) -> None:
    """The arm that can see curriculum must not inherit the pairing filter.

    This is the property whose absence made a 100%-selfplay sample look like
    "the training distribution".
    """
    p = tmp_path / "shard_000001.zarr"
    plies = [1, 2] + [5 + 4 * i for i in range(38)]
    material = _write_shard(
        p, game_id=[1, 1] + [2] * 38, ply_index=plies,
        is_selfplay=[1, 1] + [0] * 38,
    )
    out = _load([p])
    assert out["game_id"].size == 1                 # only ply 1 -> 2 pairs
    assert out["_all_game_id"].size == 40           # every labelled row survives
    assert out["_all_is_selfplay"].astype(bool).tolist() == [True] * 2 + [False] * 38
    # The all-rows ruler is the row's OWN eval negated into the scored side's POV.
    assert out["_all_ruler_cp"].tolist() == pytest.approx(
        [round(m * 100.0) for m in material], abs=1e-6,
    )


def test_mate_scores_fold_into_the_ruler(tmp_path: Path) -> None:
    """A mate row carries no cp; dropping it would bias the losing tail."""
    p = tmp_path / "shard_000001.zarr"
    _write_shard(p, game_id=[1] * 40, ply_index=list(range(1, 41)), mate=[3] + [0] * 39)
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
    _write_shard(a, game_id=[1] * 40, ply_index=list(range(1, 41)))
    _write_shard(b, game_id=[2] * 40, ply_index=list(range(1, 41)), seed=5)
    out = _load([a, b], max_rows=3)
    assert out["game_id"].size == 3
    assert set(np.unique(out["game_id"]).tolist()) == {1}


def test_no_usable_rows_is_a_hard_exit_naming_the_gate(tmp_path: Path) -> None:
    bad = tmp_path / "shard_000001.zarr"
    _write_shard(bad, game_id=[1] * 40, ply_index=list(range(1, 41)), scramble_labels=True)
    with pytest.raises(SystemExit, match="integrity gate"):
        _load([bad])


def test_multipv_gate_rejects_what_the_attachment_axis_cannot(tmp_path: Path) -> None:
    """The 2026-07-30/31 UCI desync: labels on the RIGHT rows, wrong position.

    The shard is built so its material<->label correlation is perfect — the
    attachment axis sees nothing wrong, exactly as it saw nothing wrong in three
    of the four live episodes. Only the MultiPV-miss rate separates it, which is
    why the gate carries two enforced axes.
    """
    good = tmp_path / "shard_000001.zarr"
    desynced = tmp_path / "shard_000002.zarr"
    _write_shard(good, game_id=[1] * 40, ply_index=list(range(1, 41)))
    _write_shard(
        desynced, game_id=[2] * 40, ply_index=list(range(1, 41)), multipv_miss=0.6, seed=4,
    )
    # The attachment axis alone would pass BOTH shards.
    z = zarr.open_group(str(desynced), mode="r")
    from chess_anti_engine.eval.value_optimism import sf_label_attachment_corr
    reading = sf_label_attachment_corr(
        np.asarray(z["x"][:, 0:12]), np.asarray(z["sf_wdl"][:]).astype(float),
    )
    assert reading.usable
    assert reading.value > SF_LABEL_ATTACHMENT_MIN
    out = _load([good, desynced])
    assert int(out["_shards_rejected"][0]) == 1
    assert set(np.unique(out["game_id"]).tolist()) == {1}
    assert set(np.unique(out["_all_game_id"]).tolist()) == {1}


def test_multipv_gate_passes_a_sound_shard(tmp_path: Path) -> None:
    """Sound shards read exactly 0.000000 live; the gate must not eat them."""
    p = tmp_path / "shard_000001.zarr"
    _write_shard(p, game_id=[1] * 400, ply_index=list(range(1, 401)), multipv_miss=0.005)
    out = _load([p])
    assert int(out["_shards_rejected"][0]) == 0
    assert out["_all_game_id"].size == 400


def test_a_missing_field_is_a_reject_named_as_such(tmp_path: Path) -> None:
    """A gate that cannot evaluate a shard has not cleared it — and the reason
    must not share a string with "genuinely corrupt", or a pre-schema shard is
    swallowed wholesale under it."""
    p = tmp_path / "shard_000001.zarr"
    _write_shard(p, game_id=[1] * 40, ply_index=list(range(1, 41)))
    g = zarr.open_group(str(p), mode="a")
    del g["has_sf_multipv_raw"]
    with pytest.raises(SystemExit, match="integrity gate"):
        _load([p])


def test_attachment_nan_is_a_reject_not_a_pass(tmp_path: Path) -> None:
    """Pins the attachment axis's own unevaluable path.

    Previously only the second axis pinned this, so a mutation that let an
    unevaluable attachment reading through escaped the suite.
    """
    p = tmp_path / "shard_000001.zarr"
    good = tmp_path / "shard_000002.zarr"
    # Constant material on every row -> undefined rank correlation.
    _write_shard(p, game_id=[1] * 40, ply_index=list(range(1, 41)),
                 material_override=[3.0] * 40)
    _write_shard(good, game_id=[2] * 40, ply_index=list(range(1, 41)))
    out = _load([p, good])
    assert int(out["_shards_rejected"][0]) == 1
    assert set(np.unique(out["_all_game_id"]).tolist()) == {2}


def test_desync_axis_is_diagnostic_by_default(tmp_path: Path) -> None:
    """It has no honest threshold, so it must not silently reject.

    Sound max 0.1496 vs corrupt min 0.1505 over the live trial: any cut sits in
    a 0.0009 gap. The value is reported; enforcement is opt-in.
    """
    p = tmp_path / "shard_000001.zarr"
    good = tmp_path / "shard_000002.zarr"
    _write_shard(p, game_id=[1] * 40, ply_index=list(range(1, 41)), desync_frac=0.9)
    _write_shard(good, game_id=[2] * 40, ply_index=list(range(1, 41)), seed=6)
    assert int(_load([p, good])["_shards_rejected"][0]) == 0   # default: not enforced
    enforced = _load([p, good], desync_max=0.15)
    assert int(enforced["_shards_rejected"][0]) == 1
    assert set(np.unique(enforced["_all_game_id"]).tolist()) == {2}
