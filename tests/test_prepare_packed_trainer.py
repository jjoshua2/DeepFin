import json
from pathlib import Path
import sys

import pytest
from scripts import prepare_packed_trainer as prep


def corpus(tmp_path):
    cohorts = []
    reuse = tmp_path / 'reuse'
    reuse.mkdir()
    for index in range(35):
        root = tmp_path / f'cohort{index:02d}'
        cohorts.append({'base': str(root)})
        for shard in range(32 if index == 0 else 9):
            path = root / f'shard_{shard:06d}.zarr' / 'x'
            path.mkdir(parents=True)
            (path / '.zarray').write_text(json.dumps({'shape': [8192, 175, 8, 8]}))
            if index == 0:
                (reuse / f'shard_{shard:06d}.zarr.zip').touch()
    return cohorts, reuse


def test_selection_preserves_source_partitions_and_spreads(tmp_path):
    cohorts, reuse = corpus(tmp_path)
    records = prep.select(cohorts, reuse)
    assert len(records) == len({r['source'] for r in records}) == 256
    assert len({Path(r['source']).parent for r in records}) == 35
    assert sum(r['reused'] is not None for r in records) == 32
    selected = [r for r in records if r['cohort'] == 34]
    assert len(selected) == 6
    assert selected[0]['source'].endswith('shard_000000.zarr')
    assert selected[-1]['source'].endswith('shard_000008.zarr')


def test_changed_selection_refused_before_output(tmp_path, monkeypatch):
    cohorts, reuse = corpus(tmp_path)
    source_plan = tmp_path / 'source.json'
    source_plan.write_text(json.dumps({'cohorts': cohorts}))
    out = tmp_path / 'out'
    external = tmp_path / 'external'
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'records': prep.select(cohorts, reuse),
                                    'output': str(out), 'external': str(external)}))
    changed = Path(cohorts[1]['base']) / 'shard_000000.zarr/x/.zarray'
    changed.write_text(json.dumps({'shape': [8191, 175, 8, 8]}))
    monkeypatch.setattr(prep.os, 'sched_setaffinity', lambda *_: None)
    monkeypatch.setattr(prep.os, 'nice', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--cohort-plan', str(source_plan),
        '--cohort-plan-sha256', prep.sha(source_plan), '--selection-plan', str(selection),
        '--selection-plan-sha256', prep.sha(selection), '--reuse', str(reuse),
        '--out', str(out), '--external', str(external), '--prepare'])
    with pytest.raises(ValueError, match='immutable plan'):
        prep.main()
    assert not out.exists()
    assert not external.exists()
