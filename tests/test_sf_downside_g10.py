"""Real selected/shuffled writer: phase0 identity, exclusions and copied bytes."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_policy_mix as policy
from scripts import corpus_row_provenance as provenance
from scripts import sf_policy_rewrite as tool
from tests.test_derive_corpus_targets import CONFIG_SHA, narrowed_phase, write_corpus
from tests.test_derive_parallel import run
from tests.test_sf_policy_rewrite import args, raw_row


def fixture(tmp_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    rows = [raw_row(game_id=i) for i in range(7)]
    for i, row in enumerate(rows):
        lines = row['phases'][0]['per_depth'][0]['lines']
        for j, line in enumerate(lines):
            line[2] = 500.125 - j * (35 + i)
        row['phases'].append(narrowed_phase({9: {str(lines[-1][1]): 900.0}}))
    rows[1]['result'] = None
    # Fully banked but invalid phase0 support: the real deriver records this drop.
    bad_lines = rows[3]['phases'][0]['per_depth'][0]['lines']
    bad_lines[-1][1] = bad_lines[0][1]
    raw = write_corpus(tmp_path, rows, row_schema=3, complete=False,
                       staircase=[{'depth': 9, 'width': 'all'}, {'depth': 10, 'width': '8'},
                                  {'depth': 12, 'width': '4'}])
    sf = tmp_path / 'sf'
    run(raw, sf, '--limit', '7', '--row-provenance', '--policy-observation', 'phase0',
        '--max-policy-support-misses', '1', temp=0.0005, rows_per_shard=3)
    summary_path = sf / tool.derive.SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    # Fixture launch helper has no adaptive gate option; make its selected metadata
    # describe G10. No payload target or provenance is fabricated.
    summary['corpus']['staircase_gate'] = {'policy': 'g10'}
    summary_path.write_text(json.dumps(summary))
    selection = {'schema': 1, 'source_dir': str(raw), 'source_config_sha256': CONFIG_SHA,
                 'source_manifest_sha256': tool.file_sha256(raw / 'manifest.json'),
                 'shards': [{'source_shard': p.name, 'rows': len(rows),
                             'source_sha256': tool.file_sha256(p)}
                            for p in sorted(raw.glob('*.jsonl.zst'))]}
    roster = tmp_path / 'selection.json'
    roster.write_text(json.dumps(selection))
    b100 = tmp_path / 'B100'
    shutil.copytree(sf, b100)
    mix = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
           'alpha': 1.0, 'bt4_temperature': 0.5, 'rows': 5, 'expected_shards': 2,
           'source_dir': str(sf), 'source_derive_summary_sha256': tool.file_sha256(summary_path),
           'mutated_arrays': ['policy_target']}
    for shard in b100.glob('*.zarr'):
        g: Any = zarr.open_group(str(shard), mode='a')
        legal = g['legal_mask'][:]
        g['policy_target'][:] = policy.mix_policy_targets(
            g['policy_target'][:], np.ones(legal.shape, dtype=np.float32), legal,
            alpha=1, scope='global', bt4_temperature=0.5)
        g.attrs.update(policy_target_mix_kind='global', policy_target_mix_alpha=1.0,
                       policy_target_mix_bt4_temperature=0.5)
    summary['policy_target_postprocess'] = mix
    (b100 / tool.derive.SUMMARY_NAME).write_text(json.dumps(summary))
    (b100 / 'bt4_policy_mix_summary.json').write_text(json.dumps(mix))
    invocation = args(raw, sf, tmp_path / 'out', '--tactical-recipe', 'allmove-downside300',
        '--tactical-bt4-source', str(b100), '--expected-bt4-summary-sha256',
        tool.file_sha256(b100 / tool.derive.SUMMARY_NAME), '--expected-bt4-mix-sha256',
        tool.file_sha256(b100 / 'bt4_policy_mix_summary.json'), '--selected-g10-roster', str(roster),
        '--expected-selected-g10-roster-sha256', tool.file_sha256(roster))
    return invocation, rows


def test_real_selected_partial_source_shuffled_join_and_all16_bytes(tmp_path: Path) -> None:
    invocation, rows = fixture(tmp_path)
    result = tool.rewrite(invocation)
    assert (result['rows'], result['shards']) == (5, 2)
    assert result['source_drop_counts_inherited'] is True
    assert result['raw_rows_decoded'] == 14  # one raw stream per derived shard, no whole-bank cache
    assert not (Path(invocation.raw) / 'summary.json').exists()
    assert (Path(invocation.out) / tool.derive.POLICY_SUPPORT_MISSES_FILE).read_bytes() == (
        Path(invocation.source) / tool.derive.POLICY_SUPPORT_MISSES_FILE).read_bytes()
    for entry in result['outputs']:
        source = Path(invocation.tactical_bt4_source) / entry['path']
        out = Path(invocation.out) / entry['path']
        before: Any = zarr.open_group(str(source), mode='r')
        after: Any = zarr.open_group(str(out), mode='r')
        for i, game in enumerate(before['game_id'][:]):
            obs = tool.observation(rows[int(game)], CONFIG_SHA, selected_phase0=True)
            expected, _ = tool.tactical_target(obs, before['policy_target'][i], downside=True)
            np.testing.assert_array_equal(after['policy_target'][i], expected)
        for column in tool.ARRAYS - {'policy_target'}:
            for p in (source / column).iterdir():
                assert p.read_bytes() == (out / column / p.name).read_bytes()
        assert (source / provenance.FILENAME).read_bytes() == (out / provenance.FILENAME).read_bytes()


def test_phase0_wins_over_later_d9_and_legacy_refuses(tmp_path: Path) -> None:
    _, rows = fixture(tmp_path)
    row = rows[0]
    obs = tool.observation(row, CONFIG_SHA, selected_phase0=True)
    assert obs.scores[-1] < obs.scores[0]
    assert row['phases'][1]['per_depth'][0]['lines'][0][2] == 900
    with pytest.raises(ValueError, match='single-phase'):
        tool.observation(row, CONFIG_SHA)
    for defect in ['missing', 'duplicate']:
        bad = copy.deepcopy(row)
        lines = bad['phases'][0]['per_depth'][0]['lines']
        if defect == 'missing':
            lines.pop()
        else:
            lines[-1][1] = lines[0][1]
        with pytest.raises(ValueError, match=r'roster|repeats|support'):
            tool.observation(bad, CONFIG_SHA, selected_phase0=True)


@pytest.mark.parametrize('defect', ['game', 'missing_provenance', 'unfinalized', 'nonpolicy'])
def test_real_writer_refuses_alignment_and_storage_defects(tmp_path: Path, defect: str) -> None:
    invocation, _ = fixture(tmp_path)
    sf = Path(invocation.source) / 'shard_000000.zarr'
    b100 = Path(invocation.tactical_bt4_source) / sf.name
    g: Any = zarr.open_group(str(sf), mode='a')
    if defect == 'game':
        g['game_id'][0] = 999
    elif defect == 'missing_provenance':
        (sf / provenance.FILENAME).unlink()
    elif defect == 'unfinalized':
        g.attrs['derive_run_finalized'] = False
    else:
        bg: Any = zarr.open_group(str(b100), mode='a')
        bg['search_wdl'][0] = [0, 1, 0]
    with pytest.raises((ValueError, FileNotFoundError), match=r'identity|provenance|finalized|source shard|nonpolicy'):
        tool.rewrite(invocation)
    assert not Path(invocation.out).exists()


def test_selected_mode_requires_downside_and_pilot_stays_unadmitted(tmp_path: Path) -> None:
    invocation, _ = fixture(tmp_path)
    invocation.tactical_recipe = 'legacy'
    with pytest.raises(ValueError, match='requires allmove'):
        tool.rewrite(invocation)
    invocation.tactical_recipe = 'allmove-downside300'
    invocation.pilot_shards, invocation.pilot_max_raw_rows = 1, 7
    result = tool.rewrite(invocation)
    assert (result['status'], result['rows']) == ('PILOT_COMPLETE_NOT_TRAINING', 3)
    assert not (Path(invocation.out) / tool.derive.SUMMARY_NAME).exists()


@pytest.mark.parametrize('defect', ['excluded', 'wrong_phase', 'missing_raw', 'cap'])
def test_selected_join_refuses_wrong_binding_before_publication(tmp_path: Path, defect: str) -> None:
    from scripts.sf_downside_g10 import SelectedG10

    invocation, _ = fixture(tmp_path)
    source = Path(invocation.source)
    summary = json.loads((source / tool.derive.SUMMARY_NAME).read_text())
    if defect == 'wrong_phase':
        summary['scheme']['policy_observation'] = 'latest-phase'
        with pytest.raises(ValueError, match='requires phase0'):
            SelectedG10(invocation, summary, Path(invocation.raw), source,
                        lambda row, config: tool.observation(row, config, selected_phase0=True))
        return
    joiner = SelectedG10(invocation, summary, Path(invocation.raw), source,
                        lambda row, config: tool.observation(row, config, selected_phase0=True))
    spec = summary['shards'][0]
    if defect == 'excluded':
        refs = provenance.read(source / spec['path'] / provenance.FILENAME, rows=spec['rows'])
        first = refs[0]
        joiner.exclusions[(first['source_shard'], first['source_row'])] = {}
        pattern = 'excluded'
    elif defect == 'missing_raw':
        for name in joiner.entries:
            (Path(invocation.raw) / name).unlink()
        pattern = 'No such file'
    else:
        pattern = 'raw-row cap'
    with pytest.raises((ValueError, FileNotFoundError), match=pattern):
        joiner.join(spec, lambda: None, 1 if defect == 'cap' else None)
    assert not Path(invocation.out).exists()
