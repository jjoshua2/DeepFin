#!/usr/bin/env python3
"""Read one fixed paired match between explicit checkpoint/prior packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, NoReturn, cast

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_anti_engine.utils.game_log import read_game_log, settings_fingerprint
from scripts.bt4_recipe_readout import pinned, read_json, require, same, summary


def positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def checkpoint(ref: dict[str, Any]) -> tuple[Path, Any]:
    require(set(ref) == {'role', 'path', 'sha256'} and isinstance(ref['role'], str)
            and bool(ref['role'].strip()), 'checkpoint role/pin fields')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file(), 'checkpoint path')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    same(digest.hexdigest(), ref['sha256'], 'checkpoint content')
    return path, before


def stable(path: Path, before: Any) -> None:
    after = path.stat()
    require(all(getattr(before, k) == getattr(after, k)
                for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            'checkpoint changed during readout')


def panel_fens(ref: dict[str, Any], pairs: int) -> list[str]:
    panel = read_json(ref)
    require(isinstance(panel, list) and len(panel) == pairs, 'fixed opening panel count')
    fens = []
    for entry in panel:
        require(isinstance(entry, dict) and set(entry) == {'root_fen', 'moves', 'fen'}
                and isinstance(entry['moves'], list) and len(entry['moves']) == 16, 'opening fields/history')
        board = chess.Board(entry['root_fen'])
        require(board.is_valid(), 'opening root')
        for move in entry['moves']:
            parsed = chess.Move.from_uci(move)
            require(board.is_legal(parsed), 'illegal opening move')
            board.push(parsed)
        require(board.is_valid() and not board.is_game_over() and board.fen() == entry['fen'],
                'opening endpoint/history')
        fens.append(entry['fen'])
    require(len(set(fens)) == pairs, 'duplicate opening endpoints')
    return fens


def validate(contract: dict[str, Any]) -> None:
    require(set(contract) == {'schema', 'profile', 'candidate', 'reference', 'candidate_prior_temperature',
        'reference_prior_temperature', 'sims', 'pairs', 'seed', 'settings', 'execution',
        'opening_panel', 'bank', 'process', 'results_path'}, 'package contract fields')
    same(contract['schema'], 1, 'schema')
    same(contract['profile'], 'explicit_checkpoint_prior_packages', 'profile')
    for key in ('candidate_prior_temperature', 'reference_prior_temperature'):
        require(positive(contract[key]), 'positive finite per-side prior required')
    require(type(contract['sims']) is int and contract['sims'] > 0
            and type(contract['pairs']) is int and contract['pairs'] >= 2
            and type(contract['seed']) is int, 'integer simulations/pairs/seed')
    require(Path(contract['results_path']).is_absolute(), 'absolute results path required')
    same(sorted(contract['execution']), sorted(('loop', 'compile', 'eval_max_batch', 'max_concurrent_games', 'max_seconds')),
         'execution fields')
    execution = contract['execution']
    require(execution['loop'] in ('rolling', 'chunked') and execution['compile'] == 'on', 'execution mode')
    for key in ('eval_max_batch', 'max_concurrent_games'):
        require(type(execution[key]) is int and execution[key] > 0, 'positive execution size')
    require(positive(execution['max_seconds']), 'finite arena deadline')
    settings = contract['settings']
    require(isinstance(settings, dict) and {'ms_per_move', 'uci_args', 'syzygy', 'syzygy_max_pieces',
            'search_candidate', 'search_reference', 'volatility_candidate'} <= set(settings), 'full expected settings required')
    json.dumps(settings, allow_nan=False)
    required = {'mode': 'matched_sims', 'candidate': contract['candidate']['path'],
        'reference': contract['reference']['path'], 'games': 2*contract['pairs'], 'seed': contract['seed'],
        'sims_candidate': contract['sims'], 'sims_reference': contract['sims'], 'ms_per_move': None,
        'opening_plies': 16, 'openings_kind': 'book', 'max_plies': 300, 'temperature': .1,
        'gumbel_add_noise': True, 'volatility_candidate': None, 'uci_args': '',
        'syzygy': '', 'syzygy_max_pieces': 0}
    for key, value in required.items():
        same(settings.get(key), value, 'package setting ' + key)
    require(isinstance(settings.get('openings'), str) and bool(settings['openings']), 'opening book path')
    non_prior = []
    for side in ('candidate', 'reference'):
        search = settings['search_' + side]
        require(isinstance(search, dict) and search.get('shape') == 'training'
                and isinstance(search.get('gumbel'), dict), 'training search required')
        same(search['gumbel'].get('policy_temp'), contract[side + '_prior_temperature'], side + ' prior')
        non_prior.append({**{k:v for k,v in search.items() if k not in ('source', 'gumbel')},
                          'gumbel': {k:v for k,v in search['gumbel'].items() if k != 'policy_temp'}})
    same(non_prior[0], non_prior[1], 'non-prior search settings')


def command_check(command: Any, contract: dict[str, Any]) -> None:
    require(isinstance(command, list) and len(command) > 2 and all(isinstance(x, str) for x in command)
            and Path(command[1]).name == 'arena_standard.py', 'direct arena command required')

    class Parser(argparse.ArgumentParser):
        def error(self, message: str) -> NoReturn:
            raise ValueError('unsupported arena command: ' + message)

    settings, execution = contract['settings'], contract['execution']
    expected = {'--candidate': settings['candidate'], '--reference': settings['reference'],
        '--games': str(settings['games']), '--mode': 'matched_sims', '--sims': str(contract['sims']),
        '--seed': str(contract['seed']), '--openings': settings['openings'], '--opening-plies': '16',
        '--max-plies': '300', '--temperature': '0.1', '--search-shape': 'training', '--compile': 'on',
        '--syzygy-max-pieces': '0', '--games-out': contract['bank']['path'], '--out': contract['results_path'],
        '--eval-max-batch': str(execution['eval_max_batch']), '--max-concurrent-games': str(execution['max_concurrent_games'])}
    parser = Parser(allow_abbrev=False, add_help=False)
    for flag in [*expected, '--cand-gumbel', '--ref-gumbel', '--max-seconds']:
        parser.add_argument(flag, action='append', required=True)
    for flag in ('--device', '--label', '--report-every', '--compile-cache-dir', '--pgn-out'):
        parser.add_argument(flag, action='append')
    parser.add_argument('--no-rolling', action='count', default=0)
    args = vars(parser.parse_args(command[2:]))
    for key, values in args.items():
        if key != 'no_rolling' and values is not None:
            require(len(values) == 1, 'duplicate command option ' + key)
    for flag, value in expected.items():
        same(args[flag[2:].replace('-', '_')], [value], 'command ' + flag)
    for side, flag in [('candidate', 'cand_gumbel'), ('reference', 'ref_gumbel')]:
        value = args[flag][0]
        require(value.startswith('policy_temp=') and value.count('=') == 1, 'only prior override supported')
        prior = float(value.split('=')[1])
        require(positive(prior), 'finite command prior')
        same(prior, float(contract[side + '_prior_temperature']), 'command ' + side + ' prior')
    same(float(args['max_seconds'][0]), float(execution['max_seconds']), 'command deadline')
    same(args['no_rolling'], int(execution['loop'] == 'chunked'), 'command loop')
    require(args['device'] in (None, ['cuda']), 'command device')


def command_observation(process: dict[str, Any], contract: dict[str, Any], *, allow_timeout_preexec: bool) -> str:
    if 'arena_cmdline' not in process:
        return 'not_recorded'
    observed = process['arena_cmdline']
    if observed == process['command']:
        return 'actual_command'
    require(allow_timeout_preexec, 'observed arena command differs')
    seconds = process.get('hard_seconds')
    require(type(seconds) is int and seconds > 30
            and seconds == contract['execution']['max_seconds'] + 60, 'preexec hard budget')
    seconds = cast(int, seconds)
    expected = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', f'{seconds - 30}s',
                *process['command']]
    same(process.get('supervisor_command'), expected, 'preexec supervisor command')
    same(observed, expected, 'preexec observed command')
    pids = [process.get(k) for k in ('owner_pid', 'supervisor_pid', 'arena_pid')]
    require(all(type(pid) is int and pid > 0 for pid in pids) and len(set(pids)) == 3,
            'preexec supervisor/child identity')
    start, end = process.get('started_unix'), process.get('ended_unix')
    require(positive(start) and positive(end), 'preexec timestamps')
    require(0 < cast(float, end) - cast(float, start) <= seconds, 'preexec elapsed budget')
    return 'exact_supervised_timeout_preexec_snapshot'


def read_contract(path: Path, *, allow_timeout_preexec: bool = False) -> dict[str, Any]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    validate(contract)
    identities = [checkpoint(contract[side]) for side in ('candidate', 'reference')]
    require(identities[0][0].resolve() != identities[1][0].resolve()
            and contract['candidate']['sha256'] != contract['reference']['sha256']
            and contract['candidate']['role'] != contract['reference']['role'], 'distinct checkpoint packages required')
    fens = panel_fens(contract['opening_panel'], contract['pairs'])
    process = read_json(contract['process'])
    same(process.get('exit_code'), 0, 'process exit')
    same(process.get('process_complete'), True, 'process completion')
    command_check(process['command'], contract)
    observation = command_observation(process, contract, allow_timeout_preexec=allow_timeout_preexec)
    bank = pinned(contract['bank'])
    log = read_game_log(bank)
    require(not log.truncated_tail, 'torn game bank')
    same(log.header.get('driver'), 'arena_standard', 'bank driver')
    same(log.header.get('version'), 1, 'bank version')
    same(log.fingerprint, settings_fingerprint(log.settings), 'bank fingerprint')
    same(log.settings, contract['settings'], 'realized settings')
    require(log.info.get('sprt') is None, 'fixed pairs cannot use sequential stopping')
    require(len(log.games) == 2*contract['pairs'], 'complete fixed paired bank required')
    rows = {}
    for row in log.games:
        pair, half = row.get('pair_id'), row.get('half')
        require(type(pair) is int and 0 <= pair < contract['pairs'] and type(half) is int and half in (0, 1),
                'canonical pair identity')
        pair, half = cast(int, pair), cast(int, half)
        key = pair, half
        require(key not in rows, 'duplicate pair/half')
        for field, expected in {'opening_index': pair, 'a_is_white': half == 0, 'seed': contract['seed'],
                'opening_fen': fens[pair], 'start_fen': fens[pair], 'loop': contract['execution']['loop'],
                'compile': 'on', 'eval_hoist': str(contract['execution']['eval_max_batch'])}.items():
            same(row.get(field), expected, 'game ' + field)
        require(row.get('result') in ('1-0', '0-1', '1/2-1/2'), 'unfinished game')
        score = {'1-0': 1., '0-1': 0., '1/2-1/2': .5}[row['result']]
        if half == 1:
            score = 1.-score
        require(type(row.get('score_candidate')) in (int, float) and row['score_candidate'] == score, 'candidate score/result')
        rows[key] = score
    totals = [rows[p, 0]+rows[p, 1] for p in range(contract['pairs'])]
    for ref in ('opening_panel', 'process', 'bank'):
        pinned(contract[ref])
    for checkpoint_path, before in identities:
        stable(checkpoint_path, before)
    require(path.read_bytes() == raw, 'contract changed')
    return {'profile': contract['profile'], 'bank_complete': True,
        'contract_sha256': hashlib.sha256(raw).hexdigest(), 'candidate': contract['candidate'], 'reference': contract['reference'],
        'candidate_prior_temperature': contract['candidate_prior_temperature'],
        'reference_prior_temperature': contract['reference_prior_temperature'],
        'sims': contract['sims'], 'seed': contract['seed'], 'execution': contract['execution'],
        'bank': contract['bank'], 'opening_panel': contract['opening_panel'], 'process': contract['process'],
        'pair_scores': [s/2 for s in totals], 'result': summary(totals),
        'command_observation': observation,
        'checkpoint_content_verified_now': True, 'launch_qualification_verified': False,
        'limitations': ['Fixed-N nominal paired interval for these packages; not optimal temperature or training-seed uncertainty.',
            'Pinned process record and command checked; runtime provenance, checkpoint/book bytes and full history consumed at launch require external evidence.',
            'Legal panel history and endpoint order verified now; recipe labels and training lineage are declarations requiring separate qualification.']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--contract', type=Path, action='append', required=True)
    parser.add_argument('--allow-timeout-preexec-capture', action='store_true',
                        help='Admit only the exact recorded timeout fork/exec snapshot; retain all bank checks')
    args = parser.parse_args()
    require(len(args.contract) == 1, 'exactly one contract required')
    print(json.dumps(read_contract(args.contract[0], allow_timeout_preexec=args.allow_timeout_preexec_capture), indent=2))


if __name__ == '__main__':
    main()
