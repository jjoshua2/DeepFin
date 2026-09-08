from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import corpus_row_provenance as refs
from scripts import derive_corpus_targets as derive
from scripts import sf_d9_rank_sidecar as tool
from tests.test_derive_corpus_targets import history_row, run_derive
from tests.test_derive_parallel import write_split_corpus


def _fixture(tmp_path: Path, workers: int = 0) -> tuple[Path, Path, dict[str, Any]]:
    rows = [history_row(game_id=index) for index in range(6)]
    for row in rows:
        lines = tool.d9_lines(row)
        lines.sort(key=lambda line: -float(line[2]))
        for rank, line in enumerate(lines, 1):
            line[0] = rank
    # Complete rank slots but a missing legal move and duplicate move, as in G10.
    lines = tool.d9_lines(rows[1])
    lines[-1][1] = lines[0][1]
    rows[4]['result'] = None
    source = write_split_corpus(tmp_path, rows, [3, 3])
    derived = tmp_path / 'derived'
    flags = ['--workers', str(workers)] if workers else []
    summary = run_derive(
        source, derived, 'uniform-d9', '--limit', '6', '--temp', '0.0005',
        '--rows-per-shard', '2', '--seed', '9', '--row-provenance',
        '--policy-observation', 'phase0', '--max-policy-support-misses', '1', *flags,
    )
    assert summary['realized']['rows_dropped_policy_support'] == 1
    assert summary['realized']['rows_dropped_no_result'] == 1
    return source, derived, summary


def _args(source: Path, derived: Path, out: Path) -> list[str]:
    return [
        '--raw', str(source), '--shards', str(derived), '--out', str(out),
        '--limit', '6', '--top-k', '3', '--rows-per-shard', '2', '--seed', '9',
        '--expected-rows', '4', '--expected-shards', '2',
        '--expected-source-summary-sha256', tool.file_sha256(derived / tool.DERIVE_SUMMARY),
    ]


@pytest.mark.parametrize('workers', [0, 2])
def test_rank_after_real_serial_or_spawn_support_exclusion(
    tmp_path: Path, workers: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, derived, summary = _fixture(tmp_path, workers)
    original = tool.rank_observation
    visited: list[int] = []

    def observe(row: Any, *, top_k: int) -> tool.RankObservation:
        assert row['game_id'] != 1, 'excluded raw row must never enter rank extraction'
        visited.append(int(row['game_id']))
        return original(row, top_k=top_k)

    monkeypatch.setattr(tool, 'rank_observation', observe)
    required = tool.rank_cache_bytes(6, 2, 3)
    with pytest.raises(ValueError, match='rank/history cache needs'):
        tool.main([*_args(source, derived, tmp_path / 'insufficient-cache'),
            '--max-provenance-cache-bytes', str(required - 1),
        ])
    assert visited == [], 'insufficient cache must refuse before raw rank extraction'
    out = tmp_path / 'ranks'
    assert tool.main([*_args(source, derived, out),
        '--max-provenance-cache-bytes', str(required),
    ]) == 0
    assert sorted(visited) == [0, 2, 3, 5]
    receipt = json.loads((out / tool.SUMMARY_NAME).read_text())
    assert receipt['rows_dropped_policy_support'] == 1
    assert receipt['row_provenance']['rows_dropped_policy_support'] == 1
    assert receipt['row_provenance']['policy_support_exclusions_sha256'] == tool.file_sha256(
        derived / derive.POLICY_SUPPORT_MISSES_FILE,
    )
    for entry in summary['shards']:
        source_group: Any = zarr.open_group(str(derived / entry['path']), mode='r')
        rank_group: Any = zarr.open_group(str(out / entry['path']), mode='r')
        for i, game in enumerate(source_group['game_id'][:]):
            assert int(game) in {0, 2, 3, 5}
            assert source_group['legal_mask'][i, int(rank_group[tool.INDEX_FIELD][i, 0])] == 1
    assert not (out / '._rank_identity_cache').exists()


@pytest.mark.parametrize('corruption', [
    'counter', 'file', 'history', 'support', 'valid_row', 'source', 'identity_type',
])
def test_rank_refuses_forged_support_exclusion(tmp_path: Path, corruption: str) -> None:
    source, derived, summary = _fixture(tmp_path)
    entries = summary['realized']['policy_support_exclusions']
    if corruption == 'counter':
        summary['realized']['rows_dropped_policy_support'] = 0
    elif corruption == 'file':
        (derived / derive.POLICY_SUPPORT_MISSES_FILE).write_text('[]\n')
    else:
        if corruption == 'history':
            entries[0]['input_key'] = '00' * 16
        elif corruption == 'support':
            entries[0]['missing_moves'] = []
        elif corruption == 'valid_row':
            entries[0]['source_row'] = 0
        elif corruption == 'identity_type':
            # Python considers True == 1; provenance still requires an integer ID.
            entries[0]['game_id'] = True
        else:
            entries[0]['source_dir'] = str(tmp_path / 'another_source')
        (derived / derive.POLICY_SUPPORT_MISSES_FILE).write_text(
            ''.join(json.dumps(entry) + '\n' for entry in entries),
        )
    (derived / tool.DERIVE_SUMMARY).write_text(json.dumps(summary))
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='policy support exclusion'):
        tool.main(_args(source, derived, out))
    assert not out.exists()
    assert not (out.with_name(out.name + '.writing') / tool.SUMMARY_NAME).exists()


def test_rank_refuses_derived_reference_to_explicitly_excluded_row(tmp_path: Path) -> None:
    source, derived, summary = _fixture(tmp_path)
    path = derived / summary['shards'][0]['path']
    group: Any = zarr.open_group(str(path), mode='a')
    references = refs.read(path / refs.FILENAME, rows=2)
    excluded = summary['realized']['policy_support_exclusions'][0]
    references[0].update({key: excluded[key] for key in references[0]})
    group['game_id'][0] = excluded['game_id']
    group['ply_index'][0] = excluded['ply']
    (path / refs.FILENAME).unlink()
    stamp = refs.write(
        path / refs.FILENAME, references, np.asarray(group['x'][:]),
        np.asarray(group['game_id'][:]), np.asarray(group['ply_index'][:]),
    )
    group.attrs['derive_row_provenance'] = stamp
    summary['shards'][0]['row_provenance'] = stamp
    (derived / tool.DERIVE_SUMMARY).write_text(json.dumps(summary))
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='references an excluded policy support row'):
        tool.main(_args(source, derived, out))
    assert not out.exists()


def test_rank_exclusion_cannot_adopt_a_legacy_row(tmp_path: Path) -> None:
    row = history_row()
    row['schema'] = derive.ROW_SCHEMA_BARE_FEN
    with pytest.raises(ValueError, match='requires banked full-history input keys'):
        tool._verify_policy_support_exclusion(
            row, {}, raw_path=tmp_path / 'raw.zst', offset=0, raw_config='a' * 64,
        )


@pytest.mark.parametrize('corruption', ['truncated', 'appended', 'width', 'searchmoves'])
def test_rank_exclusion_does_not_hide_rank_width_or_protocol_defects(
    tmp_path: Path, corruption: str,
) -> None:
    row = history_row()
    lines = tool.d9_lines(row)
    if corruption == 'truncated':
        lines.pop()
    elif corruption == 'appended':
        lines.append([len(lines) + 1, lines[0][1], lines[0][2], 1])
    elif corruption == 'width':
        row['phases'][0]['width_streamed'] -= 1
    else:
        row['phases'][0]['searchmoves'] = [lines[0][1]]
    with pytest.raises(ValueError, match=r'malformed (ranks or scores|full-width metadata)'):
        tool._verify_policy_support_exclusion(
            row, {}, raw_path=tmp_path / 'raw.zst', offset=0, raw_config='a' * 64,
        )
