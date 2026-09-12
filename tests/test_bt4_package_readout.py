"""Actual arena serialization and fixed-bank package admission; no model play."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from typing import Any

import chess
import pytest

from scripts import bt4_package_readout as tool
from scripts.arena_standard import SideSearch, arena_game_log_settings
from tests.test_bt4_calibration_contract import calibration as calibration, pin, save_contract


@pytest.fixture
def package(calibration: Any) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path, old, rows = calibration
    candidate, reference = path.parent/'T1.pt', path.parent/'T05.pt'
    candidate.write_bytes(b'candidate checkpoint; not a model')
    reference.write_bytes(b'different reference checkpoint; not a model')
    settings = arena_game_log_settings(mode='matched_sims', candidate=str(candidate), reference=str(reference),
        games=8, seed=42, openings_path=old['expected_settings']['openings'], openings_kind='book',
        opening_plies=16, sims_candidate=400, sims_reference=400, ms_per_move=None,
        max_plies=300, temperature=.1, gumbel_add_noise=True,
        search_candidate=SideSearch('training', 'config + CLI(policy_temp=0.5)', {'policy_temp': .5}, 1, 0),
        search_reference=SideSearch('training', 'config + CLI(policy_temp=1.0)', {'policy_temp': 1.}, 1, 0),
        volatility_candidate=None, uci_args='', syzygy_path=None, tb_max_pieces=0)
    c: dict[str, Any] = {'schema': 1, 'profile': 'explicit_checkpoint_prior_packages',
        'candidate': {'role': 'B100T1_epoch2', **pin(candidate)}, 'reference': {'role': 'B100_epoch2', **pin(reference)},
        'candidate_prior_temperature': .5, 'reference_prior_temperature': 1., 'sims': 400, 'pairs': 4, 'seed': 42,
        'settings': settings, 'execution': {'loop': 'rolling', 'compile': 'on', 'eval_max_batch': 4096,
            'max_concurrent_games': 128, 'max_seconds': 1500}, 'opening_panel': old['opening_panel'],
        'bank': old['bank'], 'results_path': str(path.parent/'results.jsonl')}
    rows[0]['settings'] = settings
    for row in rows[1:]:
        total = row['pair_id']*.5
        score = min(total, 1.) if row['half'] == 0 else max(0., total-1.)
        white_score = score if row['half'] == 0 else 1.-score
        row.update(score_candidate=score, result={0.: '0-1', .5: '1/2-1/2', 1.: '1-0'}[white_score])
    values = {'candidate': str(candidate), 'reference': str(reference), 'games': '8', 'mode': 'matched_sims',
        'sims': '400', 'seed': '42', 'openings': settings['openings'], 'opening-plies': '16', 'max-plies': '300',
        'temperature': '0.1', 'search-shape': 'training', 'compile': 'on', 'syzygy-max-pieces': '0',
        'games-out': c['bank']['path'], 'out': c['results_path'], 'eval-max-batch': '4096',
        'max-concurrent-games': '128', 'max-seconds': '1500', 'cand-gumbel': 'policy_temp=0.5',
        'ref-gumbel': 'policy_temp=1.0'}
    command = ['/qualified/python', '/qualified/scripts/arena_standard.py']
    for key, value in values.items():
        command.extend(['--'+key, value])
    process = path.parent/'process.json'
    process.write_text(json.dumps({'exit_code': 0, 'process_complete': True, 'command': command}))
    c['process'] = pin(process)
    save_package(path, c, rows)
    return path, c, rows


def save_package(path: Path, contract: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows[0]['fingerprint'] = tool.settings_fingerprint(rows[0]['settings'])
    bank = Path(contract['bank']['path'])
    bank.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    contract['bank'] = pin(bank)
    save_contract(path, contract)


@pytest.mark.parametrize('loop', ['rolling', 'chunked'])
def test_distinct_checkpoint_compensated_priors_actual_cli(
    package: Any, loop: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path, contract, rows = package
    if loop == 'chunked':
        contract['execution']['loop'] = loop
        for row in rows[1:]:
            row['loop'] = loop
        process = Path(contract['process']['path'])
        data = json.loads(process.read_text())
        data['command'].append('--no-rolling')
        process.write_text(json.dumps(data))
        contract['process'] = pin(process)
    save_package(path, contract, rows)
    monkeypatch.setattr('sys.argv', ['package-reader', '--contract', str(path)])
    tool.main()
    result = json.loads(capsys.readouterr().out)
    assert result['bank_complete']
    assert not result['launch_qualification_verified']
    assert result['candidate'] == contract['candidate']
    assert result['reference'] == contract['reference']
    assert result['pair_scores'] == [0., .25, .5, .75]
    assert result['result']['score'] == .375
    assert result['result']['pentanomial'] == {'WW': 0, 'WD_DW': 1, 'DD_WL': 1, 'LD_DL': 1, 'LL': 1}
    assert result['candidate_prior_temperature'] == .5
    assert result['reference_prior_temperature'] == 1.


@pytest.mark.parametrize('defect', ['wrong_side', 'candidate_prior', 'reference_prior', 'non_prior', 'volatility',
    'typed_setting', 'missing_pair', 'orphan', 'duplicate', 'score', 'pair_type', 'opening', 'compile', 'sequential'])
def test_rebound_bank_rejects_wrong_package_or_partial_success(package: Any, defect: str) -> None:
    path, c, rows = package
    settings = rows[0]['settings']
    if defect == 'wrong_side':
        settings['candidate'], settings['reference'] = settings['reference'], settings['candidate']
    elif defect.endswith('_prior') and defect != 'non_prior':
        settings['search_'+defect.removesuffix('_prior')]['gumbel']['policy_temp'] = .7
    elif defect == 'non_prior':
        settings['search_reference']['gumbel']['c_scale'] = 2.
    elif defect == 'volatility':
        settings['volatility_candidate'] = {'volatility_q_scale': .5}
    elif defect == 'typed_setting':
        settings['games'] = True
    elif defect == 'missing_pair':
        rows = rows[:-2]  # Process still exit0: insufficient for fixed-N completion.
    elif defect == 'orphan':
        rows = rows[:-1]
    elif defect == 'duplicate':
        rows[-1] = copy.deepcopy(rows[1])
    elif defect == 'score':
        rows[1]['score_candidate'] = 1.
    elif defect == 'pair_type':
        rows[1]['pair_id'] = False
    elif defect == 'opening':
        rows[1]['opening_fen'] = rows[3]['opening_fen']
    elif defect == 'compile':
        rows[1]['compile'] = 'off'
    else:
        rows[0]['info'] = {'sprt': {'verdict': 'H1'}}
    c['settings'] = settings
    save_package(path, c, rows)
    with pytest.raises(ValueError, match=r'setting|prior|bank|duplicate|score|identity|game|stopping'):
        tool.read_contract(path)


@pytest.mark.parametrize('defect', ['wrong_prior', 'equals_duplicate', 'unknown_override', 'abbreviation',
    'wrong_candidate', 'wrong_pool', 'wrong_loop', 'observed_mismatch', 'nonzero', 'incomplete'])
def test_actual_process_command_is_parsed_not_token_searched(package: Any, defect: str) -> None:
    path, c, _rows = package
    process = Path(c['process']['path'])
    data = json.loads(process.read_text())
    command = data['command']
    if defect == 'wrong_prior':
        command[command.index('--ref-gumbel')+1] = 'policy_temp=0.5'
    elif defect == 'equals_duplicate':
        command.append('--cand-gumbel=policy_temp=0.9')
    elif defect == 'unknown_override':
        command += ['--cand-gumbel-other', 'policy_temp=0.9']
    elif defect == 'abbreviation':
        command[command.index('--candidate')] = '--candid'
    elif defect == 'wrong_candidate':
        command[command.index('--candidate')+1] = c['reference']['path']
    elif defect == 'wrong_pool':
        command[command.index('--max-concurrent-games')+1] = '256'
    elif defect == 'wrong_loop':
        command.append('--no-rolling')
    elif defect == 'observed_mismatch':
        data['arena_cmdline'] = [*command, '--resume']
    elif defect == 'nonzero':
        data['exit_code'] = 1
    else:
        data['process_complete'] = False
    process.write_text(json.dumps(data))
    c['process'] = pin(process)
    save_contract(path, c)
    with pytest.raises(ValueError, match=r'command|duplicate|process'):
        tool.read_contract(path)


@pytest.mark.parametrize('defect', ['same_content', 'same_role', 'changed_bytes', 'bad_panel', 'nonfinite_prior'])
def test_content_and_panel_identity_refusals(package: Any, defect: str) -> None:
    path, c, _rows = package
    if defect == 'same_content':
        Path(c['reference']['path']).write_bytes(Path(c['candidate']['path']).read_bytes())
        c['reference'].update(pin(Path(c['reference']['path'])))
    elif defect == 'same_role':
        c['reference']['role'] = c['candidate']['role']
    elif defect == 'changed_bytes':
        Path(c['candidate']['path']).write_bytes(b'tampered')
    elif defect == 'bad_panel':
        panel = Path(c['opening_panel']['path'])
        entries = json.loads(panel.read_text())
        entries[0]['moves'] = []
        panel.write_text(json.dumps(entries))
        c['opening_panel'] = pin(panel)
    else:
        c['candidate_prior_temperature'] = float('nan')
    save_contract(path, c)
    with pytest.raises(ValueError, match=r'checkpoint|opening|prior'):
        tool.read_contract(path)


def test_null_history_refused_even_with_matching_valid_endpoint_and_bank(package: Any) -> None:
    path, contract, rows = package
    board = chess.Board()
    board.push_uci('0000')  # python-chess permits this API call, but it is not a legal move.
    rng = random.Random(601)
    for _ in range(15):
        board.push(rng.choice(list(board.legal_moves)))
    assert board.is_valid()
    assert not board.is_game_over()
    panel = Path(contract['opening_panel']['path'])
    entries = json.loads(panel.read_text())
    entries[0] = {'root_fen': board.root().fen(), 'moves': [m.uci() for m in board.move_stack], 'fen': board.fen()}
    panel.write_text(json.dumps(entries))
    contract['opening_panel'] = pin(panel)
    for row in rows[1:3]:
        row['opening_fen'] = row['start_fen'] = board.fen()
    save_package(path, contract, rows)
    with pytest.raises(ValueError, match='illegal opening move'):
        tool.read_contract(path)


@pytest.mark.parametrize('defect', ['none', 'prefix', 'supervisor', 'suffix', 'pid', 'budget', 'elapsed', 'bank'])
def test_explicit_preexec_recovery_retains_command_and_complete_bank_checks(package: Any, defect: str) -> None:
    path, contract, rows = package
    process = Path(contract['process']['path'])
    data = json.loads(process.read_text())
    hard = contract['execution']['max_seconds'] + 60
    wrapped = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{hard-30}s', *data['command']]
    data.update(arena_cmdline=list(wrapped), supervisor_command=list(wrapped), hard_seconds=hard,
                owner_pid=100, supervisor_pid=101, arena_pid=102, started_unix=1000., ended_unix=1100.)
    if defect == 'prefix':
        data['arena_cmdline'][0] = '/unqualified/timeout'
    elif defect == 'supervisor':
        data['supervisor_command'][3] = '1s'
    elif defect == 'suffix':
        data['arena_cmdline'].append('--resume')
    elif defect == 'pid':
        data['arena_pid'] = data['supervisor_pid']
    elif defect == 'budget':
        data['hard_seconds'] += 1
    elif defect == 'elapsed':
        data['ended_unix'] = data['started_unix'] + hard + 1
    elif defect == 'bank':
        rows = rows[:-1]
    process.write_text(json.dumps(data))
    contract['process'] = pin(process)
    save_package(path, contract, rows)
    with pytest.raises(ValueError, match='observed arena command'):
        tool.read_contract(path)
    if defect == 'none':
        result = tool.read_contract(path, allow_timeout_preexec=True)
        assert result['command_observation'] == 'exact_supervised_timeout_preexec_snapshot'
        assert result['bank_complete']
        assert result['result']['score'] == .375
    else:
        with pytest.raises(ValueError, match=r'preexec|complete fixed paired bank'):
            tool.read_contract(path, allow_timeout_preexec=True)


def test_explicit_null_observed_command_is_not_missing() -> None:
    process: dict[str, Any] = {'command': ['python', 'arena.py']}
    assert tool.command_observation(process, {}, allow_timeout_preexec=False) == 'not_recorded'
    process['arena_cmdline'] = None
    with pytest.raises(ValueError, match='observed arena command differs'):
        tool.command_observation(process, {}, allow_timeout_preexec=False)
