"""Explicit nonprefix selection reaches real derivation and rank consumers."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import corpus_row_provenance as refs
from scripts import derive_corpus_targets as derive
from scripts import sf_d9_rank_sidecar as rank
from tests.test_derive_corpus_targets import CONFIG_SHA, history_row, run_derive
from tests.test_derive_parallel import write_split_corpus


def fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    rows = [history_row(game_id=i, result=None if i == 3 else 1.0) for i in range(8)]
    for row in rows:
        lines = rank.d9_lines(row)
        lines.sort(key=lambda line: -float(line[2]))
        for i, line in enumerate(lines, 1):
            line[0] = i
    source = write_split_corpus(tmp_path, rows, [2, 2, 2, 2])
    selected = {
        'schema': 1, 'source_dir': str(source.resolve()), 'source_config_sha256': CONFIG_SHA,
        'source_manifest_sha256': rank.file_sha256(source / 'manifest.json'),
        # Intentionally reversed. The original corpus order governs selection.
        'shards': [{'source_shard': f'w00-{i:05d}.jsonl.zst', 'rows': 2,
                    'source_sha256': rank.file_sha256(source / f'w00-{i:05d}.jsonl.zst')}
                   for i in (3, 1)],
    }
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selected))
    return source, path, selected


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('limit', [3, 4])
def test_nonprefix_cli_derivation_and_rank(tmp_path: Path, workers: int, limit: int) -> None:
    source, selection, _ = fixture(tmp_path)
    out = tmp_path / 'derived'
    summary = run_derive(source, out, 'uniform-d9', '--source-shards', str(selection),
                         '--limit', str(limit), '--workers', str(workers), '--row-provenance',
                         '--policy-observation', 'phase0', '--rows-per-shard', '2', '--seed', '9')
    assert summary['realized']['rows_read'] == limit
    assert summary['realized']['rows_dropped_no_result'] == 1
    assert summary['realized']['rows_written'] == limit - 1
    assert [r['source_shard'] for r in summary['source_selection']['shards']] == [
        'w00-00001.jsonl.zst', 'w00-00003.jsonl.zst']
    seen = []
    for shard in sorted(out.glob('shard_*.zarr')):
        group = zarr.open_group(str(shard), mode='r')
        x = group['x']
        assert isinstance(x, zarr.Array)
        for ref in refs.read(shard / refs.FILENAME, rows=int(x.shape[0])):
            gid = ref['game_id']
            assert ref['source_dir'] == str(source.resolve())
            assert ref['source_config_sha256'] == CONFIG_SHA
            assert ref['source_namespace'] == refs._namespace(source, CONFIG_SHA)[0]
            assert ref['source_shard'] == f'w00-{gid // 2:05d}.jsonl.zst'
            assert ref['source_row'] == gid % 2
            seen.append(gid)
    assert sorted(seen) == [2, 6] + ([7] if limit == 4 else [])
    args = ['--raw', str(source), '--source-shards', str(selection), '--shards', str(out),
            '--out', str(tmp_path / 'ranks'), '--limit', str(limit), '--rows-per-shard', '2',
            '--seed', '9', '--expected-rows', str(limit - 1), '--expected-shards', str(limit // 2),
            '--expected-source-summary-sha256', rank.file_sha256(out / derive.SUMMARY_NAME)]
    assert rank.main(args) == 0
    ranked = json.loads((tmp_path / 'ranks' / rank.SUMMARY_NAME).read_text())
    assert ranked['raw_rows_read'] == limit
    assert ranked['source_selection'] == summary['source_selection']
    for shard in sorted(out.glob('shard_*.zarr')):
        group = zarr.open_group(str(shard), mode='r')
        sidecar = zarr.open_group(str(tmp_path / 'ranks' / shard.name), mode='r')
        expected_identity = rank._sha_arrays(
            np.asarray(group['game_id'][:], dtype=np.int64),
            np.asarray(group['ply_index'][:], dtype=np.int32))
        assert sidecar.attrs['source_row_identity_sha256'] == expected_identity
    # Source-bound derivation cannot silently be joined against the original prefix.
    without = list(args)
    i = without.index('--source-shards')
    del without[i:i + 2]
    without[without.index('--out') + 1] = str(tmp_path / 'wrong_ranks')
    with pytest.raises(ValueError, match='selections differ'):
        rank.main(without)


@pytest.mark.parametrize('corruption', ['duplicate', 'unknown', 'unclosed', 'source', 'config', 'missing_config', 'manifest', 'rows', 'hash', 'traversal'])
def test_bad_selection_is_rejected(tmp_path: Path, corruption: str) -> None:
    source, path, selection = fixture(tmp_path)
    if corruption == 'duplicate':
        selection['shards'].append(selection['shards'][0])
    elif corruption in ('unknown', 'unclosed', 'traversal'):
        name = '../w00-00003.jsonl.zst' if corruption == 'traversal' else 'w00-99999.jsonl.zst'
        selection['shards'][0]['source_shard'] = name
        if corruption == 'unclosed':
            shutil.copyfile(source / 'w00-00003.jsonl.zst', source / name)
    elif corruption == 'source':
        selection['source_dir'] = str(tmp_path / 'other')
    elif corruption == 'missing_config':
        del selection['source_config_sha256']
    elif corruption == 'config':
        selection['source_config_sha256'] = '0' * 64
    elif corruption == 'manifest':
        selection['source_manifest_sha256'] = '0' * 64
    elif corruption == 'rows':
        selection['shards'][0]['rows'] = 3
    else:
        selection['shards'][0]['source_sha256'] = '0' * 64
    path.write_text(json.dumps(selection))
    with pytest.raises(derive.CorpusIntegrityError):
        derive.select_corpus_record(source, derive.read_corpus_record(source), path)


def test_live_inventory_growth_is_harmless_but_selected_mutation_is_fatal(tmp_path: Path) -> None:
    source, path, _ = fixture(tmp_path)
    selected = derive.select_corpus_record(source, derive.read_corpus_record(source), path)
    extra = source / 'w00-00004.jsonl.zst'
    shutil.copyfile(source / 'w00-00000.jsonl.zst', extra)
    with (source / 'w00.progress.jsonl').open('a') as stream:
        stream.write(json.dumps({'codec': 'zstd', 'path': str(extra), 'rows': 2}) + '\n')
    derive.verify_source_selection(selected)
    refreshed = derive.select_corpus_record(source, derive.read_corpus_record(source), path)
    assert refreshed.source_selection == selected.source_selection
    assert refreshed.shards == selected.shards
    with selected.shards[0].open('ab') as stream:
        stream.write(b'mutation')
    with pytest.raises(derive.CorpusIntegrityError, match='changed during processing'):
        derive.verify_source_selection(selected)


def test_forged_support_exclusion_cannot_absorb_a_no_result_row(tmp_path: Path) -> None:
    row = history_row(result=None)
    lines = rank.d9_lines(row)
    lines[-1][1] = lines[0][1]  # A genuine support defect, but result filtering comes first.
    forged = {'reason': 'selected_phase0_policy_support', 'source_row': 0}
    with pytest.raises(ValueError, match='cannot replace a no-result drop'):
        rank._verify_policy_support_exclusion(
            row, forged, raw_path=tmp_path / 'w00-00000.jsonl.zst', offset=0, raw_config=CONFIG_SHA)
