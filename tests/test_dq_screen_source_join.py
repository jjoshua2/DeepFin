"""EXP-01: child recovery joins logical replay sources, not global game IDs."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import cast_probe, dq_free_dataset_screen as screen
from scripts import tail_censor_screen as tail


def _arrays(plies: list[int], markers: list[int], *, history: str = "history-a") -> dict[str, Any]:
    n = len(plies)
    return {
        "sf_multipv_raw": np.tile(np.array([[i, 100 - i * 10, 0] for i in range(6)]), (n, 1, 1)),
        "has_sf_multipv_raw": np.ones(n, dtype=bool),
        "legal_mask": np.ones((n, 8), dtype=bool),
        "policy_target": np.full((n, 8), 0.125),
        "x": np.array(markers).reshape(n, 1),
        "game_id": np.full(n, 7), "ply_index": np.array(plies),
        "has_is_selfplay": np.ones(n, dtype=bool), "is_selfplay": np.ones(n, dtype=bool),
        "_input_history_encoding": np.array(history),
        "_policy_encoding": np.array("policy-a"),
    }


@pytest.mark.parametrize("variant", ["different_root", "different_trial", "same_source",
                                    "identical_duplicate", "conflicting_duplicate",
                                    "different_encoding", "foreign_encoding_first",
                                    "foreign_only", "legacy_same_source"])
def test_collected_row_recovers_only_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    parent = tmp_path / "a" / "shard_000.zarr"
    child = tmp_path / "a" / "shard_001.zarr"
    other = tmp_path / ("b" if variant in {"different_root", "foreign_encoding_first", "foreign_only"} else "a") / "shard_002.zarr"
    shards = [parent, child]
    data = {parent: (_arrays([0, 1], [0, 1]), {"run_id": "trial-a"}),
            child: (_arrays([2], [11]), {"run_id": "trial-a"})}
    if variant not in {"same_source", "legacy_same_source"}:
        marker = 11 if variant == "identical_duplicate" else 22
        data[other] = (_arrays([2], [marker]), {
            "run_id": "trial-b" if variant == "different_trial" else "trial-a"})
        shards.append(other)
    if variant == "different_encoding":
        data[child] = (_arrays([2], [11], history="history-b"), {"run_id": "trial-a"})
        shards = [parent, child]
    if variant == "foreign_encoding_first":
        data[other] = (_arrays([2], [22], history="foreign-history"), {"run_id": "trial-a"})
        shards = [other, parent, child]
    if variant == "foreign_only":
        shards = [parent, other]
    if variant == "legacy_same_source":
        data = {path: (arrays, {}) for path, (arrays, _) in data.items()}
    monkeypatch.setattr(tail, "load_shard_arrays", lambda p: data[p])
    monkeypatch.setattr(screen, "load_shard_arrays", lambda p: data[p])
    monkeypatch.setattr(tail, "sf_eval_pv_orphan_flags", lambda a: (
        np.zeros(len(a["game_id"])), np.zeros(len(a["game_id"]))))
    scan: dict[str, Any] = dict.fromkeys([
        "desync_checked", "desync_orphaned", "rows_scanned", "rows_selfplay",
        "skipped_not_selfplay", "desync_rows_rejected", "coverage_sum", "coverage_n",
        "surfaced_not_legal", "unscored_mass_sum"], 0)
    rows, planes = tail.collect([parent], scan, 2)
    assert len(rows) == 1
    calls = []

    def recover(parent_x, child_x, **encodings):
        calls.append((int(parent_x[0]), encodings))
        return int(child_x[0])

    monkeypatch.setattr(cast_probe, "recover_played_move", recover)
    moves, stats = screen.recover_search_moves(shards, rows, planes)
    if variant == "foreign_only":
        assert moves == [None]
        assert stats == {"recovered": 0, "ambiguous": 0, "no_child": 1}
        assert calls == []
    elif variant in {"conflicting_duplicate", "different_encoding"}:
        assert moves == [None]
        assert stats == {"recovered": 0, "ambiguous": 1, "no_child": 0}
        assert calls == []
    else:
        assert moves == [11]
        assert stats == {"recovered": 1, "ambiguous": 0, "no_child": 0}
        assert calls == [(1, {"input_history_encoding": "history-a",
                              "policy_encoding": "policy-a"})]
