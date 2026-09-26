from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from scripts import sf_d9_rank_sidecar as tool


def raw_row(*, stm: str = "w") -> dict[str, object]:
    moves = (
        [[1, "e2e4", 20.0, 100], [2, "d2d4", 13.0, 100], [3, "g1f3", 3.0, 100]]
        if stm == "w"
        else [[1, "e7e5", 20.0, 100], [2, "d7d5", 13.0, 100], [3, "g8f6", 3.0, 100]]
    )
    return {
        "game_id": 7,
        "ply": 4,
        "stm": stm,
        "phases": [
            {
                "per_depth": [
                    {"depth": 8, "complete": True, "lines": moves},
                    {"depth": 9, "complete": True, "lines": moves},
                ],
            }
        ],
    }


@pytest.mark.parametrize("stm", ["w", "b"])
def test_rank_observation_banks_top_indices_and_cp_gaps(stm: str) -> None:
    observed = tool.rank_observation(raw_row(stm=stm), top_k=3)
    assert observed.game_id == 7
    assert observed.ply == 4
    assert observed.count == 3
    assert observed.indices.dtype == np.uint16
    assert len(set(map(int, observed.indices))) == 3
    np.testing.assert_array_equal(observed.gaps_cp, [0.0, 7.0, 17.0])


def test_rank_observation_pads_short_width() -> None:
    row: Any = raw_row()
    row["phases"][0]["per_depth"][1]["lines"] = [[1, "e2e4", 20.0, 100]]
    observed = tool.rank_observation(row, top_k=3)
    assert observed.count == 1
    np.testing.assert_array_equal(
        observed.indices[1:],
        np.asarray([tool.INVALID_INDEX, tool.INVALID_INDEX], dtype=np.uint16),
    )
    assert bool(np.all(np.isinf(observed.gaps_cp[1:])))


def test_d9_lines_reads_only_the_full_width_phase_zero_block() -> None:
    row: Any = raw_row()
    expected = row["phases"][0]["per_depth"][1]["lines"]
    row["phases"].append(
        {
            "per_depth": [
                {
                    "depth": 9,
                    "complete": True,
                    "lines": [[1, "a2a3", 99.0, 50]],
                }
            ]
        }
    )

    assert tool.d9_lines(row) == expected


def test_rank_observation_refuses_unknown_side_to_move() -> None:
    with pytest.raises(ValueError, match="side-to-move"):
        tool.rank_observation(raw_row(stm="unknown"), top_k=3)


def test_flush_refuses_row_order_drift(tmp_path: Path) -> None:
    rows = [
        tool.rank_observation(raw_row(), top_k=3),
        tool.RankObservation(
            game_id=8,
            ply=5,
            indices=tool.rank_observation(raw_row(), top_k=3).indices,
            gaps_cp=tool.rank_observation(raw_row(), top_k=3).gaps_cp,
            count=3,
        ),
    ]
    source_path = tmp_path / "shard_000000.zarr"
    source = zarr.open_group(str(source_path), mode="w")
    source.create_dataset("game_id", data=np.asarray([7, 8], dtype=np.int64))
    source.create_dataset("ply_index", data=np.asarray([4, 5], dtype=np.int32))
    legal = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.uint8)
    policy = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float16)
    for index in rows[0].indices:
        legal[:, int(index)] = 1
    policy[:, int(rows[0].indices[0])] = 1.0
    source.create_dataset("legal_mask", data=legal)
    source.create_dataset("policy_target", data=policy)

    destination = tmp_path / "rank.zarr"
    receipt = tool._flush(
        observations=rows,
        order=np.asarray([0, 1]),
        source_path=source_path,
        destination=destination,
        top_k=3,
        source_summary_sha256="a" * 64,
        raw_config_sha256="b" * 64,
    )
    assert receipt["rows"] == 2
    stored = zarr.open_group(str(destination), mode="r")
    np.testing.assert_array_equal(
        np.asarray(stored[tool.GAP_FIELD][:, 1]),
        [7.0, 7.0],
    )

    with pytest.raises(ValueError, match="does not match derived order"):
        tool._flush(
            observations=rows,
            order=np.asarray([1, 0]),
            source_path=source_path,
            destination=tmp_path / "bad.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )

    first = rows[0]
    repeated = tool.RankObservation(
        game_id=first.game_id,
        ply=first.ply,
        indices=np.asarray(
            [first.indices[0], first.indices[0], first.indices[2]],
            dtype=np.uint16,
        ),
        gaps_cp=first.gaps_cp,
        count=3,
    )
    with pytest.raises(ValueError, match="repeat a compact move index"):
        tool._flush(
            observations=[repeated, rows[1]],
            order=np.asarray([0, 1]),
            source_path=source_path,
            destination=tmp_path / "repeated.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )

    decreasing = tool.RankObservation(
        game_id=first.game_id,
        ply=first.ply,
        indices=first.indices,
        gaps_cp=np.asarray([0.0, 17.0, 7.0], dtype=np.float32),
        count=3,
    )
    with pytest.raises(ValueError, match="gaps must be nondecreasing"):
        tool._flush(
            observations=[decreasing, rows[1]],
            order=np.asarray([0, 1]),
            source_path=source_path,
            destination=tmp_path / "decreasing.zarr",
            top_k=3,
            source_summary_sha256="a" * 64,
            raw_config_sha256="b" * 64,
        )


def _provenance_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    from tests.test_derive_corpus_targets import history_row, narrowed_phase, run_derive
    from tests.test_derive_parallel import write_split_corpus
    rows = [history_row(game_id=i) for i in range(6)]
    for row in rows:
        original = tool.d9_lines(row)
        original.sort(key=lambda line: -float(line[2]))
        for rank, line in enumerate(original, 1):
            line[0] = rank
        # Later d9 changes the winner; rank targets must still use phase0.
        row['phases'].append(narrowed_phase({9: {str(original[1][1]): 900.0}}))
    rows[2]['result'] = None
    rows[4]['phases'][0]['per_depth'] = []
    source = write_split_corpus(tmp_path, rows, [3, 3])
    derived = tmp_path / 'derived'
    run_derive(source, derived, 'uniform-d9', '--limit', '6', '--temp', '0.0005',
               '--rows-per-shard', '2', '--seed', '9', '--row-provenance',
               '--policy-observation', 'phase0', '--max-envelope-misses', '1')
    return source, derived, rows


def _rank_args(source: Path, derived: Path, out: Path) -> list[str]:
    return ['--raw', str(source), '--shards', str(derived), '--out', str(out),
            '--limit', '6', '--top-k', '3', '--rows-per-shard', '2', '--seed', '9',
            '--expected-rows', '4', '--expected-shards', '2',
            '--expected-source-summary-sha256', tool.file_sha256(derived / tool.DERIVE_SUMMARY)]


def test_rank_provenance_joins_repacked_outputs_after_both_drop_paths(tmp_path: Path) -> None:
    import json
    from scripts import corpus_row_provenance as refs
    source, derived, raw_rows = _provenance_fixture(tmp_path)
    paths = sorted(derived.glob('shard_*.zarr'))
    groups: list[Any] = [zarr.open_group(str(path), mode='a') for path in paths]
    references = [ref for path in paths for ref in refs.read(path / refs.FILENAME, rows=2)]
    arrays = {key: np.concatenate([group[key][:] for group in groups]) for key in groups[0].array_keys()}
    # Repack across output shards: both revisit earlier physical raw rows.
    order = np.array([3, 0, 2, 1])
    summary = json.loads((derived / tool.DERIVE_SUMMARY).read_text())
    for number, (path, group) in enumerate(zip(paths, groups)):
        selection = order[number * 2:number * 2 + 2]
        for key, values in arrays.items():
            group[key][:] = values[selection]
        (path / refs.FILENAME).unlink()
        stamp = refs.write(path / refs.FILENAME, [references[i] for i in selection],
                           arrays['x'][selection], arrays['game_id'][selection], arrays['ply_index'][selection])
        group.attrs['derive_row_provenance'] = stamp
        summary['shards'][number]['row_provenance'] = stamp
    (derived / tool.DERIVE_SUMMARY).write_text(json.dumps(summary))
    out = tmp_path / 'ranks'
    assert tool.main(_rank_args(source, derived, out)) == 0
    for path, source_group in zip(paths, groups):
        rank_group: Any = zarr.open_group(str(out / path.name), mode='r')
        for offset, game_id in enumerate(np.asarray(source_group['game_id'][:])):
            expected = tool.rank_observation(raw_rows[int(game_id)], top_k=3)
            np.testing.assert_array_equal(rank_group[tool.INDEX_FIELD][offset], expected.indices)
            np.testing.assert_array_equal(rank_group[tool.GAP_FIELD][offset], expected.gaps_cp)
    receipt = json.loads((out / tool.SUMMARY_NAME).read_text())
    assert receipt['row_provenance']['raw_shards_read_once'] == 2
    assert receipt['row_provenance']['policy_observation'] == 'phase0'
    assert receipt['row_provenance']['value_observation'] == 'latest-phase'
    assert not (out / '._rank_identity_cache').exists()


@pytest.mark.parametrize('corruption', ['history', 'source', 'duplicate'])
def test_rank_provenance_refuses_wrong_raw_identity(tmp_path: Path, corruption: str) -> None:
    import json
    from scripts import corpus_row_provenance as refs
    source, derived, _ = _provenance_fixture(tmp_path)
    paths = sorted(derived.glob('shard_*.zarr'))
    first: Any = zarr.open_group(str(paths[0]), mode='a')
    references = refs.read(paths[0] / refs.FILENAME, rows=2)
    if corruption == 'history':
        references[0]['input_key'] = '00' * 16
    elif corruption == 'source':
        references[0]['source_dir'] = str(tmp_path / 'other_source')
    else:
        # Duplicate a physical reference across outputs with matching replay data.
        second: Any = zarr.open_group(str(paths[1]), mode='r')
        other = refs.read(paths[1] / refs.FILENAME, rows=2)[0]
        references[0] = other
        for key in first.array_keys():
            first[key][0] = second[key][0]
    (paths[0] / refs.FILENAME).unlink()
    stamp = refs.write(paths[0] / refs.FILENAME, references,
                       np.asarray(first['x'][:]), np.asarray(first['game_id'][:]), np.asarray(first['ply_index'][:]))
    first.attrs['derive_row_provenance'] = stamp
    summary = json.loads((derived / tool.DERIVE_SUMMARY).read_text())
    summary['shards'][0]['row_provenance'] = stamp
    (derived / tool.DERIVE_SUMMARY).write_text(json.dumps(summary))
    out = tmp_path / 'bad_ranks'
    with pytest.raises(ValueError, match=r'identity mismatch|another raw source|duplicate'):
        tool.main(_rank_args(source, derived, out))
    assert not out.exists()
    assert not (out.with_name(out.name + '.writing') / tool.SUMMARY_NAME).exists()


@pytest.mark.parametrize('changed', ['payload', 'summary', 'manifest'])
def test_rank_provenance_refuses_input_mutation_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    source, derived, _ = _provenance_fixture(tmp_path)
    original = tool._flush
    mutated = False
    def mutate(**kwargs: Any) -> dict[str, Any]:
        nonlocal mutated
        receipt = original(**kwargs)
        if not mutated:
            if changed == 'payload':
                group: Any = zarr.open_group(str(kwargs['source_path']), mode='a')
                group['x'][0, 0, 0, 0] = float(group['x'][0, 0, 0, 0]) + 0.25
            else:
                path = derived / tool.DERIVE_SUMMARY if changed == 'summary' else source / 'manifest.json'
                path.write_bytes(path.read_bytes() + b'\n')
            mutated = True
        return receipt
    monkeypatch.setattr(tool, '_flush', mutate)
    out = tmp_path / 'changed_rank'
    with pytest.raises(ValueError, match=r'storage changed|summary or raw manifest changed'):
        tool.main(_rank_args(source, derived, out))
    assert not out.exists()
    assert (out.with_name(out.name + '.writing') / '._rank_identity_cache').is_dir()
