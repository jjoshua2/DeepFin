"""Hash-bound finite Gumbel generation; producer evidence is never root admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any
import uuid
from unittest.mock import patch

INITIAL_DONOR_SHA256 = 'a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0'
REQUIRED_SOURCE_FILES = (
    'scripts/core_finite_collect.py', 'chess_anti_engine/verified_outcome.py',
    'chess_anti_engine/core_generation_data.py', 'chess_anti_engine/selfplay/config.py',
    'chess_anti_engine/selfplay/finalize.py', 'chess_anti_engine/selfplay/manager.py',
    'chess_anti_engine/selfplay/network_turn.py', 'chess_anti_engine/selfplay/strict_tb.py',
    'chess_anti_engine/uci/model_loader.py', 'chess_anti_engine/stockfish/pool.py',
    'chess_anti_engine/stockfish/uci.py', 'chess_anti_engine/replay/shard.py',
    'chess_anti_engine/inference.py', 'chess_anti_engine/model/__init__.py',
    'chess_anti_engine/model/transformer.py', 'chess_anti_engine/selfplay/state.py',
    'chess_anti_engine/selfplay/opening.py', 'chess_anti_engine/utils/syzygy_config.py',
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked(ref: dict[str, Any]) -> Any:
    require(set(ref) == {'path', 'sha256'} and isinstance(ref['path'], str) and
            isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
            set(ref['sha256']) <= set('0123456789abcdef') and 'UNBOUND' not in ref['path'],
            'unbound identity reference')
    path = Path(ref['path'])
    require(path.is_absolute() and not path.is_symlink(), 'identity path must be absolute regular file')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 32 * 1024 * 1024, 'identity metadata bound exceeded')
        payload = stream.read(32 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) and
            hashlib.sha256(payload).hexdigest() == ref['sha256'], 'bound file changed: ' + ref['path'])

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON identity key: ' + key)
            value[key] = item
        return value

    return json.loads(payload, object_pairs_hook=unique_keys)


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inventory(path: Path, *, seal: bool = False) -> list[dict[str, Any]]:
    require(not path.is_symlink(), 'inventory root is symlink')
    entries = [path] if path.is_file() else sorted(path.rglob('*'))
    require(all(not p.is_symlink() for p in entries), 'inventory contains symlink')
    result = []
    for item in entries:
        if item.is_file():
            result.append({'path': '.' if path.is_file() else item.relative_to(path).as_posix(),
                           'bytes': item.stat().st_size, 'sha256': digest(item)})
            if seal:
                item.chmod(0o444)
                with item.open('rb') as stream:
                    os.fsync(stream.fileno())
    require(bool(result), 'empty file inventory')
    if seal and path.is_dir():
        for directory in sorted((p for p in entries if p.is_dir()), reverse=True):
            directory.chmod(0o555)
            sync_dir(directory)
        path.chmod(0o555)
        sync_dir(path)
    return result


def proc_birth(pid: int) -> int | None:
    try:
        return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def closed_predecessor(binding: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the separately closed learner owner, not a child status alone."""
    require(set(binding) == {'boundary', 'owner_terminal', 'owner_sha256', 'command'},
            'predecessor requires closed owner envelope')
    boundary = checked(binding['boundary'])
    terminal = checked(binding['owner_terminal'])
    command = binding['command']
    require(isinstance(command, list) and all(isinstance(x, str) for x in command) and
            len(binding['owner_sha256']) == 64 and terminal['status'] == 'PASS' and
            type(terminal['returncode']) is int and terminal['returncode'] == 0 and
            terminal['owned_processes_closed'] is True and terminal['custody'] == 'OWNED_PROCESSES_CLOSED' and
            terminal['supervisor_sha256'] == binding['owner_sha256'] and terminal['command'] == command,
            'predecessor learner owner not closed PASS or command changed')
    require(len(command) >= 10 and command[-10:-7] == ['-m', 'scripts.core_finite_learner', '--contract'] and
            command[-6:] == ['--contract-sha256', boundary['contract_sha256'], '--stage', boundary['stage']['path'],
                             '--stage-sha256', boundary['stage']['sha256']], 'predecessor command/boundary join differs')
    contract = checked({'path': command[-7], 'sha256': boundary['contract_sha256']})
    stage = checked(boundary['stage'])
    require(contract['run_uuid'] == boundary['run_uuid'] and contract['source_revision'] == boundary['source_revision'] and
            stage['generation'] == boundary['generation'] and stage['contract_sha256'] == boundary['contract_sha256'] and
            stage['generations'] == boundary['generation_manifests'] and stage['parent'] == boundary['parent'] and
            Path(binding['boundary']['path']) == Path(stage['output']) / 'boundary.json' and
            not Path(stage['output']).with_name(Path(stage['output']).name + '.failed.json').exists() and
            not (Path(stage['output']) / 'boundary.pending.json').exists() and
            Path(boundary['final_checkpoint']['path']) == Path(stage['output']) / 'learner.pt',
            'closed predecessor stage/contract/history join differs')
    return boundary


def validate_protocol(protocol: dict[str, Any], roster: list[dict[str, Any]]) -> None:
    require(protocol['schema'] == 'CORE_FINITE_GENERATION_V1', 'wrong generation protocol schema')
    require(str(uuid.UUID(protocol['run_uuid'])) == protocol['run_uuid'] and
            str(uuid.UUID(protocol['collection_uuid'])) == protocol['collection_uuid'], 'noncanonical run/collection UUID')
    for key in ('generation', 'attempts', 'concurrent_slots', 'seed'):
        require(type(protocol[key]) is int and protocol[key] >= (0 if key == 'seed' else 1), 'invalid integer: ' + key)
    require(protocol['concurrent_slots'] <= protocol['attempts'] == len(roster), 'roster/attempt/slot count differs')
    actor = protocol['actor']
    require(actor['kind'] in ('donor', 'learner') and type(actor['expected_checkpoint_step']) is int and
            actor['expected_checkpoint_step'] >= 0 and len(actor['sha256']) == 64, 'actor identity contract')
    require(actor['architecture']['_schema_version'] == 20 and
            actor['architecture']['input_extra_features'] == 'v2_threats', 'full A39 architecture contract required')
    if protocol['generation'] == 1:
        require(actor['kind'] == 'donor' and actor['sha256'] == INITIAL_DONOR_SHA256 and
                actor['predecessor'] is None, 'initial actor must be immutable A39, not a selected candidate')
    else:
        require(actor['kind'] == 'learner', 'later actors must follow the previous final learner')
        parent = closed_predecessor(actor['predecessor'])
        require(parent['status'] == 'COMPLETED_CORE_GENERATION_V1' and parent['run_uuid'] == protocol['run_uuid'] and
                parent['source_revision'] == protocol['source']['revision'] and parent['generation'] == protocol['generation'] - 1 and
                parent['final_step'] == actor['expected_checkpoint_step'] and
                parent['final_checkpoint'] == {'path': actor['path'], 'sha256': actor['sha256']},
                'actor is not the preplanned previous-generation final checkpoint')
    require([g['start_ordinal'] for g in roster] == list(range(len(roster))) and
            all(g['bank'] == 'train' and g['physical_game_id'] == protocol['collection_uuid'] + ':' + str(i)
                for i, g in enumerate(roster)), 'fixed TRAIN attempt identities/order differ')
    require(len({g['physical_game_id'] for g in roster}) == len(roster), 'duplicate physical attempt identity')
    require(len({canonical([g['root_fen'], g['opening_moves']]) for g in roster}) == len(roster), 'duplicate planned opening history')
    output = Path(protocol['output_root'])
    require(output.is_absolute() and not output.resolve().is_relative_to(Path(protocol['source']['checkout']).resolve()),
            'bulk generation output must be outside source checkout')
    require(protocol['runtime']['device'] == 'cuda:0' and len(protocol['runtime']['cpu_affinity']) == 2 and
            protocol['runtime']['threads'] == 2 and 0 < protocol['runtime']['inner_seconds'] < protocol['owner']['seconds'] <= 7200,
            'finite resource contract differs')
    sf = protocol['stockfish']
    require(type(sf['workers']) is int and 1 <= sf['workers'] <= 2 and
            type(sf['nodes']) is int and sf['nodes'] > 0 and sf['multipv'] > 0 and sf['hash_mb'] > 0,
            'finite SF diagnostic resource contract')
    require(protocol['serving']['device'] == 'cuda:0' and protocol['serving']['use_amp'] is True and
            protocol['serving']['amp_dtype'] == 'bfloat16', 'qualified eager BF16 serving contract required')


def source_guard(protocol: dict[str, Any]) -> dict[str, Any]:
    source = protocol['source']
    repo = Path(source['checkout']).resolve()
    require(Path(__file__).resolve() == repo / 'scripts/core_finite_collect.py', 'foreign collector entrypoint')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip() == source['revision'], 'source revision changed')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip(), 'source checkout dirty')
    require(set(REQUIRED_SOURCE_FILES) <= set(source['sha256']), 'incomplete source fingerprint')
    for relative, expected in source['sha256'].items():
        path = (repo / relative).resolve()
        require(path.is_relative_to(repo) and digest(path) == expected, 'source bytes changed: ' + relative)
    native = checked(source['native_qualification'])
    require(native['status'].startswith('ACCEPT') and native['source_revision'] == source['revision'] and
            Path(native['source_checkout']).resolve() == repo and len(native['native_images']) == 5,
            'native source/origin qualification absent for this checkout')
    terminal = checked(native['closed_terminal'])
    require(terminal['status'] == 'PASS' and terminal['returncode'] == 0 and
            terminal['owned_processes_closed'] is True, 'native qualification owner not closed PASS')
    for relative, expected in native['source_sha256'].items():
        require(digest(repo / relative) == expected, 'native source-input changed: ' + relative)
    require(len({x['module'] for x in native['native_images']}) == 5, 'duplicate native module identity')
    for entry in native['native_images']:
        require(Path(entry['path']).resolve().is_relative_to(repo) and digest(entry['path']) == entry['sha256'], 'native image changed')
    require(digest(protocol['owner']['path']) == protocol['owner']['sha256'], 'owner changed')
    require(Path(sys.executable).resolve() == Path(protocol['runtime']['python']).resolve(), 'interpreter changed')
    require(digest(protocol['stockfish']['path']) == protocol['stockfish']['sha256'], 'SF binary changed')
    for name in ('tablebase', 'stockfish'):
        qualification = checked(protocol[name]['qualification'])
        require(qualification['status'].startswith('ACCEPT') and
                qualification['source_revision'] == source['revision'], 'asset consumer qualification differs: ' + name)
    tb_qualification = checked(protocol['tablebase']['qualification'])
    require(tb_qualification['tablebase_path'] == protocol['tablebase']['path'] and
            tb_qualification['asset_inventory_sha256'] == protocol['tablebase']['asset_inventory_sha256'],
            'qualified strict TB asset identity differs')
    sf_qualification = checked(protocol['stockfish']['qualification'])
    require(sf_qualification['binary_sha256'] == protocol['stockfish']['sha256'] and
            sf_qualification['syzygy_path'] == protocol['tablebase']['path'], 'qualified SF binary/TB path differs')
    sys.path.insert(0, str(repo))
    return native


def project_import_origins(protocol: dict[str, Any]) -> dict[str, Any]:
    repo = Path(protocol['source']['checkout']).resolve()
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == 'chess_anti_engine' or name.startswith('chess_anti_engine.'):
            value = getattr(module, '__file__', None)
            require(value is not None and Path(value).resolve().is_relative_to(repo),
                    'foreign project import: ' + name)
            origins[name] = {'path': str(Path(value).resolve()), 'sha256': digest(value)}
    return origins


def build_configs(protocol: dict[str, Any], roster: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    from chess_anti_engine.selfplay.config import DiffFocusConfig, GameConfig, OpponentConfig, SearchConfig, TemperatureConfig
    from chess_anti_engine.selfplay.opening import OpeningConfig, resolve_slot_opening
    from chess_anti_engine.uci.model_loader import model_config_from_arch
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION
    from chess_anti_engine.utils.syzygy_config import validate_selfplay_syzygy
    import numpy as np

    values = protocol['configs']
    require(set(values) == {'game', 'search', 'temp', 'opening', 'opponent', 'diff_focus'}, 'unknown config group')
    cfg = {'game': GameConfig(**values['game']), 'search': SearchConfig(**values['search']),
           'temp': TemperatureConfig(**values['temp']), 'opening': OpeningConfig(**values['opening']),
           'opponent': OpponentConfig(**values['opponent']), 'diff_focus': DiffFocusConfig(**values['diff_focus'])}
    game, search, opening = cfg['game'], cfg['search'], cfg['opening']
    require(game.policy_row_retention == 'all_eligible' and game.selfplay_fraction == 1 and
            game.record_fast_ply_value is False and cfg['diff_focus'].enabled is False,
            'neutral eligible-policy retention not realized')
    require(search.mcts_type == 'gumbel' and search.simulations == search.fast_simulations == 64 and
            search.playout_cap_fraction == 1, 'retained full-simulation Gumbel64 route required')
    validate_selfplay_syzygy(asdict(game))
    require(game.syzygy_outcome_mode == 'rule50_match_v1' and game.syzygy_rescore_policy is True and
            game.syzygy_path == protocol['tablebase']['path'], 'strict TB/policy-only outcome route differs')
    require(game.policy_encoding == 'lc0_1858' and game.history_rep_fix is True and
            game.input_history_encoding == protocol['actor']['architecture']['input_history_encoding'] and
            game.input_extra_features == protocol['actor']['architecture']['input_extra_features'], 'actor/producer encoding differs')
    require(opening.opening_fen_list_path == protocol['starts']['path'] and opening.opening_fen_dole_per_iter == 1 and
            not opening.opening_book_path and not opening.opening_book_path_2 and
            opening.opening_fen_sf_refute_frac == 0, 'unplanned opening/refute route')
    lines = Path(protocol['starts']['path']).read_text().splitlines()
    require(lines == [g['root_fen'] + ' | ' + ' '.join(g['opening_moves']) for g in roster], 'starts are not exact ordered root+legal-history roster')
    queue = list(lines)
    rng = np.random.default_rng(protocol['seed'])
    for planned in roster:
        start = resolve_slot_opening(rng=rng, cfg=opening, is_selfplay=True, fen_dole_queue=queue)
        require(start.board.root().fen() == planned['root_fen'] and
                [m.uci() for m in start.board.move_stack] == planned['opening_moves'], 'opening consumer lost planned history')
    require(not queue, 'opening consumer left planned starts')
    model_config = model_config_from_arch(protocol['actor']['architecture'])
    require(canonical({'_schema_version': ARCH_SCHEMA_VERSION, **asdict(model_config)}) ==
            canonical(protocol['actor']['architecture']), 'architecture fields would be defaulted or dropped')
    return cfg, {k: asdict(v) for k, v in cfg.items()}


def verify_execution(protocol: dict[str, Any], protocol_sha256: str, admission: dict[str, Any]) -> None:
    require(admission['status'] == 'ROOT_ADMITS_CORE_GENERATION_V1' and
            admission['protocol_sha256'] == protocol_sha256 and admission['run_uuid'] == protocol['run_uuid'] and
            admission['generation'] == protocol['generation'] and admission['actor_sha256'] == protocol['actor']['sha256'] and
            admission['source_revision'] == protocol['source']['revision'], 'separate root stage admission absent or changed')
    args = Path('/proc', str(os.getppid()), 'cmdline').read_bytes().decode().split('\0')
    require(protocol['owner']['path'] in args and '--seconds' in args and '--resource' in args and
            args[args.index('--seconds') + 1] == str(protocol['owner']['seconds']) and
            args[args.index('--resource') + 1] == 'gpu', 'wrong direct finite owner')
    require(sorted(os.sched_getaffinity(0)) == protocol['runtime']['cpu_affinity'] and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2' and
            __debug__ and os.environ.get('PYTHONOPTIMIZE', '0') == '0', 'resource/guard context differs')


def state_sha256(state: dict[str, Any]) -> str:
    """Full digest of all normalized tensor keys/dtypes/shapes/C-order bytes."""
    import torch

    normalized: dict[str, Any] = {}
    for key, value in state.items():
        name = key.replace('_orig_mod.', '', 1)
        require(name not in normalized and isinstance(value, torch.Tensor), 'ambiguous/non-tensor actor state')
        normalized[name] = value
    result = hashlib.sha256()
    for name, value in sorted(normalized.items()):
        require(bool(torch.isfinite(value).all()), 'nonfinite actor state')
        result.update(canonical([name, str(value.dtype), list(value.shape)]))
        result.update(b'\n')
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def file_stamp(path: Path) -> dict[str, int]:
    value = path.lstat()
    return {k: getattr(value, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')}


def load_actor(protocol: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from chess_anti_engine.uci import model_loader

    actor = protocol['actor']
    path = Path(actor['path'])
    before = path.lstat()
    before_stamp = file_stamp(path)
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and not before.st_mode & 0o222 and
            digest(path) == actor['sha256'], 'actor must be immutable exact checkpoint file')
    observed: dict[str, Any] = {}
    original_load, original_build = torch.load, model_loader.build_model
    builds: list[int] = []

    def capture_load(*args: Any, **kwargs: Any) -> Any:
        require(not observed and Path(args[0]).resolve() == path.resolve() and
                kwargs.get('weights_only') is True, 'unexpected/duplicate actor load')
        payload = original_load(*args, **kwargs)
        require(isinstance(payload, dict) and canonical(payload.get('arch')) == canonical(actor['architecture']) and
                type(payload.get('step')) is int and payload['step'] == actor['expected_checkpoint_step'],
                'actor embedded architecture/historical checkpoint step differs')
        require(isinstance(payload.get('model'), dict) and len(payload['model']) == 494, 'incomplete494 actor checkpoint')
        observed.update(checkpoint_step=payload['step'], checkpoint_model_state_sha256=state_sha256(payload['model']))
        return payload

    def capture_build(config: Any) -> Any:
        model = original_build(config)
        builds.append(id(model))
        return model

    with patch.object(torch, 'load', capture_load), patch.object(model_loader, 'build_model', capture_build):
        model = model_loader.load_model_from_checkpoint(path, device='cuda:0', require_complete=True)
    require(builds == [id(model)] and len(model.state_dict()) == 494 and
            {str(p.device) for p in model.parameters()} == {'cuda:0'}, 'not one complete CUDA actor')
    observed['pre_state_sha256'] = state_sha256(model.state_dict())
    require(observed['pre_state_sha256'] == observed['checkpoint_model_state_sha256'], 'served weights differ from declared checkpoint model state')
    require(before_stamp == file_stamp(path) and digest(path) == actor['sha256'], 'actor changed while loading')
    observed.update(schema='CORE_ACTOR_SERVING_V1', actor_sha256=actor['sha256'], actor_kind=actor['kind'],
                    logical_generation=protocol['generation'], model_build_count=len(builds), state_tensors=494,
                    state_digest_method='sha256_sorted_compile_normalized_key_dtype_shape_C_order_uint8_V1',
                    checkpoint_stat=before_stamp)
    return model, observed


def validate_samples(audit: Any, samples: list[Any], model: Any, game: Any, planned: dict[str, Any],
                     *, proof: Any, verifier: Any, slots: int) -> list[dict[str, Any]]:
    import chess
    import numpy as np
    from chess_anti_engine.encoding import encode_cboard_for_model, model_encoding_kwargs, rep_fix
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves import index_to_move_strict, move_to_index_for_encoding
    from chess_anti_engine.verified_outcome import require_verified

    verified = require_verified(proof, audit=audit)
    require(0 <= audit.start.slot < slots and audit.start.start_ordinal == planned['start_ordinal'], 'unplanned recycled slot')
    require(model_encoding_kwargs(model) == {'input_history_encoding': game.input_history_encoding,
            'input_extra_features': game.input_extra_features} and rep_fix.current() is game.history_rep_fix is True,
            'actual actor/producer encoding or runtime repetition flag differs')
    board = chess.Board(audit.start.root_fen)
    for uci in audit.start.opening_moves:
        board.push_uci(uci)
    positions, played = {}, []
    for offset, action in enumerate(audit.played_move_indices):
        positions[offset] = board.copy(stack=True)
        move = index_to_move_strict(action, board)
        require(move in board.legal_moves, 'illegal full-history action')
        played.append(move.uci())
        board.push(move)
    require(board.fen() == chess.Board(audit.final_fen).fen(), 'final normalized legal board differs')
    records = {r.ply_index: r for r in audit.records}
    eligible = [r for r in audit.records if r.has_policy and not r.is_sf_refute_opp]
    require(len(records) == len(audit.records) and not any(r.is_sf_refute_opp for r in audit.records), 'duplicate/raw SF-refute record')
    require(len(samples) == (len(eligible) if verified['resolved'] else 0), 'neutral retention silently dropped eligible policy records')
    rows = []
    for i, sample in enumerate(samples):
        require(sample.ply_index in records and sample.game_id == audit.game_id and sample.has_policy and
                sample.is_network_turn is True and sample.is_selfplay is True, 'retained identity/NET/selfplay join')
        require(sample.input_history_encoding == game.input_history_encoding and sample.history_rep_fix is True,
                'sample encoding/repetition identity differs')
        record = records[sample.ply_index]
        position = positions[record.move_offset]
        require(record.has_policy and position.turn == record.pov_color and position.ply() == record.ply_index,
                'raw record POV/ply differs')
        expected = encode_cboard_for_model(model, CBoard.from_board(position))
        value = np.asarray(sample.x)
        require(value.shape == expected.shape == (175, 8, 8) and np.isfinite(value).all() and
                np.array_equal(value, expected), 'actual full-history175 encoding differs')
        legal = np.zeros(1858, dtype=np.uint8)
        for move in position.legal_moves:
            legal[move_to_index_for_encoding(move, position, policy_encoding=game.policy_encoding)] = 1
        require(sample.legal_mask is not None and np.array_equal(np.asarray(sample.legal_mask).astype(bool), legal.astype(bool)), 'actual legal mask differs')
        policy = np.asarray(sample.policy_target)
        require(policy.shape == (1858,) and np.isfinite(policy).all() and (policy >= 0).all() and
                abs(float(policy.sum()) - 1) < 1e-5 and not policy[legal == 0].any(), 'invalid legal Gumbel/TB target')
        verifier.verify_policy_sample(proof, sample, position, game.policy_encoding)
        rows.append({'physical_row_id': planned['physical_game_id'] + ':' + str(i),
                     'ply': sample.ply_index, 'move_offset': record.move_offset,
                     'history_sha256': hashlib.sha256(canonical([audit.start.root_fen,
                         list(audit.start.opening_moves) + played[:record.move_offset]])).hexdigest(),
                     'stored_input_float16_sha256': hashlib.sha256(np.ascontiguousarray(value, dtype='<f2').tobytes()).hexdigest()})
    require(len({r['ply'] for r in rows}) == len(rows), 'duplicate retained ply')
    return rows


def attach_mirror_ids(rows: list[dict[str, Any]], stored_x: Any, history_encoding: str) -> None:
    import numpy as np
    from chess_anti_engine.replay.augment import _mirror_x_batch

    require(stored_x.dtype == np.float16 and stored_x.shape == (len(rows), 175, 8, 8), 'stored175/F16 layout')
    mirrored = _mirror_x_batch(stored_x, np.ones(len(rows), dtype=bool), input_history_encoding=history_encoding)
    for i, row in enumerate(rows):
        require(hashlib.sha256(np.ascontiguousarray(stored_x[i], dtype='<f2').tobytes()).hexdigest() ==
                row['stored_input_float16_sha256'], 'raw/stored input identity differs')
        row['stored_input_float16_mirror_sha256'] = hashlib.sha256(np.ascontiguousarray(mirrored[i], dtype='<f2').tobytes()).hexdigest()


def exclusion_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate the fixed exclusions before any actor or engine is created."""
    rules = protocol['exclusions']
    prior = checked(rules['prior'])
    playing = checked(rules['playing_starts'])
    require(set(prior) >= {'history_sha256', 'stored_input_float16_sha256', 'coverage_scope'} and
            prior.get('donor_pretraining_exposure') == 'UNKNOWN', 'exposure coverage must be explicit')
    require(len(playing) == rules['playing_start_count'] and
            len({p['id'] for p in playing}) == len(playing), 'playing-start metadata count/identity differs')
    require(rules['prior_overlap_action'] in ('report', 'refuse') and rules['playing_overlap_action'] == 'refuse', 'unknown exclusion action')
    require(all(set(p) == {'id', 'input_sha256', 'mirror_sha256'} and
                all(isinstance(p[k], str) and len(p[k]) == 64 for k in ('input_sha256', 'mirror_sha256'))
                for p in playing), 'playing-start hash schema differs')
    return prior, playing


def exclusions_gate(completed: dict[int, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Report known exposure; apply only the explicitly predeclared collision rule."""
    rules = protocol['exclusions']
    prior, playing = exclusion_inputs(protocol)
    rows = [row for item in completed.values() for row in item['validation']['rows_detail']]
    known_histories, known_inputs = set(prior['history_sha256']), set(prior['stored_input_float16_sha256'])
    prior_matches = [row['physical_row_id'] for row in rows if row['history_sha256'] in known_histories or
                     {row['stored_input_float16_sha256'], row['stored_input_float16_mirror_sha256']} & known_inputs]
    retained = {row[key] for row in rows for key in ('stored_input_float16_sha256', 'stored_input_float16_mirror_sha256')}
    collisions = [{'id': p['id'], 'matching_hashes': sorted({p['input_sha256'], p['mirror_sha256']} & retained)}
                  for p in playing if {p['input_sha256'], p['mirror_sha256']} & retained]
    return {'schema': 'CORE_RETAINED_INPUT_EXCLUSIONS_V1', 'qualified': not collisions and
            (not prior_matches or rules['prior_overlap_action'] == 'report'),
            'known_prior_overlap_rows': prior_matches, 'playing_start_collisions': collisions,
            'prior_coverage_scope': prior['coverage_scope'], 'donor_pretraining_exposure': 'UNKNOWN',
            'no_roster_replacement': True}


def failure_sample_evidence(samples: list[Any]) -> dict[str, Any]:
    """Bounded failure-only hashes; never a substitute for qualified replay."""
    from dataclasses import fields
    import numpy as np

    used, rows = 0, []
    for sample in samples[:450]:
        row = {}
        for field in fields(sample):
            value = getattr(sample, field.name)
            if isinstance(value, np.ndarray):
                entry = {'shape': list(value.shape), 'dtype': value.dtype.str, 'bytes': value.nbytes}
                if not value.dtype.hasobject and value.nbytes <= 1024 * 1024 and used + value.nbytes <= 64 * 1024 * 1024:
                    entry['sha256'] = hashlib.sha256(value.tobytes(order='C')).hexdigest()
                    used += value.nbytes
                else:
                    entry['hash_status'] = 'UNHASHED_FIXED_CAP_OR_OBJECT'
                row[field.name] = entry
            else:
                row[field.name] = {'type': type(value).__name__, 'repr': repr(value)[:256]}
        rows.append(row)
    return {'status': 'UNQUALIFIED_SAMPLE_HASHES_ONLY', 'available_rows': len(samples),
            'rows': rows, 'hashed_bytes': used, 'maximum_rows': 450, 'maximum_bytes': 64 * 1024 * 1024,
            'learning_eligible': False}


def execute(protocol: dict[str, Any], protocol_sha256: str, roster: list[dict[str, Any]],
            cfg: dict[str, Any], profile: dict[str, Any], native: dict[str, Any]) -> int:
    import numpy as np
    import torch
    from chess_anti_engine.core_generation_data import capture_raw_targets, validate_stored_targets
    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.replay.shard import ShardMeta, samples_to_arrays, save_local_shard_arrays, validate_arrays
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.strict_tb import validate_stockfish_contract
    from chess_anti_engine.stockfish.pool import StockfishPool
    from chess_anti_engine.verified_outcome import VerifiedOutcomeVerifier

    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('finite generation deadline')))
    signal.alarm(protocol['runtime']['inner_seconds'])
    out = Path(protocol['output_root'])
    out.mkdir(exist_ok=False)
    write_json(out / 'assignment.json', roster)
    write_json(out / 'resolved-profile.json', profile)
    write_json(out / 'protocol-identity.json', {'protocol_sha256': protocol_sha256,
               'source_revision': protocol['source']['revision'], 'run_uuid': protocol['run_uuid'],
               'collection_uuid': protocol['collection_uuid'], 'generation': protocol['generation']})
    pool = verifier = model = evaluator = hook = None
    completed: dict[int, Any] = {}
    pending: dict[int, list[Any]] = {}
    processes: dict[int, tuple[Any, int]] = {}
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    serving: dict[str, Any] = {}
    forwards: dict[str, Any] = {'calls': 0, 'rows': 0}

    def close_pool() -> dict[str, Any]:
        nonlocal pool
        replacements = 0
        if pool is not None:
            replacements = pool.replacements
            pool.close()
            pool = None
        require(all(proc.poll() is not None and proc_birth(pid) != birth
                    for pid, (proc, birth) in processes.items()), 'observed SF birth survived close')
        return {'events': events, 'replacements': replacements, 'all_observed_births_closed': True,
                'observed_births': [{'pid': pid, 'birth': birth} for pid, (_proc, birth) in processes.items()],
                'outer_owner_complete_descendant_closure_still_required': True}

    try:
        require(not torch.cuda.is_initialized() and torch.__version__ == protocol['runtime']['torch_version'] and
                np.__version__ == protocol['runtime']['numpy_version'], 'CUDA preinitialized or numerical runtime changed')
        project_import_origins(protocol)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(protocol['seed'])
        origins = {}
        for entry in native['native_images']:
            module = importlib.import_module(entry['module'])
            require(Path(module.__file__).resolve() == Path(entry['path']).resolve() and
                    digest(module.__file__) == entry['sha256'], 'foreign native import')
            origins[entry['module']] = entry
        write_json(out / 'native-imports.json', origins)
        model, serving = load_actor(protocol)
        serving['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__,
            'device': torch.cuda.get_device_name(0), 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        verifier = VerifiedOutcomeVerifier(path=protocol['tablebase']['path'],
            asset_inventory_sha256=protocol['tablebase']['asset_inventory_sha256'],
            collection_uuid=protocol['collection_uuid'], run_uuid=protocol['run_uuid'], generation=protocol['generation'],
            source_revision=protocol['source']['revision'], concurrent_slots=protocol['concurrent_slots'],
            actor_sha256=protocol['actor']['sha256'], served_state_sha256=serving['pre_state_sha256'])
        write_json(out / 'actor-serving-start.json', serving)

        encode_tokens = getattr(model, '_encode_tokens', None)
        require(callable(encode_tokens), 'full actor lacks the qualified token-encoding entrypoint')

        def observed_encode(*args: Any, **kwargs: Any) -> Any:
            require(tuple(args[0].shape[1:]) == (175, 8, 8) and str(args[0].device) == 'cuda:0' and
                    not torch.is_grad_enabled() and torch.is_autocast_enabled('cuda') and
                    torch.get_autocast_dtype('cuda') == torch.bfloat16, 'actor forward device/encoding/AMP differs')
            forwards['calls'] += 1
            forwards['rows'] += int(args[0].shape[0])
            return encode_tokens(*args, **kwargs)

        # Both ordinary and compact legal-policy forwards enter this method;
        # a module-forward pre-hook alone misses direct forward_legal_policy.
        hook = patch.object(model, '_encode_tokens', observed_encode)
        hook.start()
        evaluator = DirectGPUEvaluator(model, **protocol['serving'])
        require(evaluator.model is model, 'evaluator serves another model')

        def on_samples(batch: Any) -> None:
            try:
                require(batch.games == 1 and batch.samples and len(batch.samples) == batch.positions,
                        'sample callback not one complete game')
                game_id = batch.samples[0].game_id
                require(game_id not in pending, 'duplicate pending game')
                pending[game_id] = list(batch.samples)
            except BaseException as error:
                write_json(out / ('failed-sample-callback-' + uuid.uuid4().hex + '.json'),
                    {'error': repr(error), 'games': batch.games, 'positions': batch.positions,
                     'evidence': failure_sample_evidence(list(batch.samples)),
                     'full_game_provenance_not_yet_delivered': True, 'learning_eligible': False})
                raise

        def on_game(audit: Any) -> None:
            ordinal = audit.start.start_ordinal
            planned = roster[ordinal] if type(ordinal) is int and 0 <= ordinal < len(roster) else None
            stage = out / ('.attempt-' + uuid.uuid4().hex)
            stage.mkdir()
            write_json(stage / 'attempt.json', {'planned': planned, 'actual': asdict(audit),
                       'actor_sha256': protocol['actor']['sha256'], 'served_state_sha256': serving['pre_state_sha256']})
            sync_dir(stage)
            sync_dir(out)
            samples = pending.get(audit.game_id, [])
            try:
                require(planned is not None and ordinal not in completed and len(samples) == audit.retained_samples,
                        'unplanned/repeated attempt or pending sample count differs')
                require(verifier is not None, 'verification owner unavailable')
                proof = verifier.verify(audit, planned, cfg['game'])
                verified = proof.metadata()
                common = {'run_uuid': protocol['run_uuid'], 'collection_uuid': protocol['collection_uuid'],
                          'source_revision': protocol['source']['revision'], 'generation': protocol['generation'],
                          'actor_sha256': protocol['actor']['sha256'], 'served_state_sha256': serving['pre_state_sha256'],
                          'physical_game_id': planned['physical_game_id'], 'game_id': audit.game_id,
                          'bank': 'train', 'rows': len(samples), 'scientific_class': verified['scientific_class']}
                write_json(stage / 'provenance.json', {'schema': 'CORE_GAME_PROVENANCE_V1', **common,
                           'planned': planned, 'actual': asdict(audit)})
                write_json(stage / 'outcome-proof.json', {'schema': 'CORE_PRODUCING_OUTCOME_PROOF_V1', **common,
                           'verification': verified, 'proof_sha256': proof.proof_sha256})
                rows = validate_samples(audit, samples, model, cfg['game'], planned, proof=proof,
                                        verifier=verifier, slots=protocol['concurrent_slots'])
                raw, raw_meta = capture_raw_targets(samples, audit, [r['physical_row_id'] for r in rows], proof=proof, game=cfg['game'])
                with (stage / 'raw-targets.npz').open('xb') as stream:
                    np.savez_compressed(stream, **raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                write_json(stage / 'raw-targets.json', {**raw_meta, **common})
                shard_files: list[dict[str, Any]] = []
                if samples:
                    arrays = samples_to_arrays(samples)
                    validate_stored_targets(raw, arrays, proof=proof, metadata=raw_meta,
                                            game_id=audit.game_id, plies=[r['ply'] for r in rows])
                    validate_arrays(arrays)
                    attach_mirror_ids(rows, arrays['x'], cfg['game'].input_history_encoding)
                    meta = ShardMeta(run_id=protocol['collection_uuid'], generated_at_unix=int(time.time()),
                        model_sha256=protocol['actor']['sha256'], games=1, positions=len(samples),
                        input_history_encoding=cfg['game'].input_history_encoding, history_rep_fix=True,
                        policy_encoding=cfg['game'].policy_encoding, sf_nodes=protocol['stockfish']['nodes'],
                        total_game_plies=audit.total_plies, selfplay_games=1, curriculum_games=0)
                    save_local_shard_arrays(stage / 'shard', arrs=arrays, meta=meta)
                    shard_files = inventory(stage / 'shard')
                validation = {'schema': 'VERIFIED_CORE_GAME_METADATA_V1', **common, 'full_history_verified': True,
                              'resolved': verified['resolved'], 'raw_targets_verified': bool(samples),
                              'usable_shard': bool(samples), 'rows_detail': rows,
                              'all_eligible_policy_rows_retained': True, 'verified_outcome_sha256': proof.proof_sha256,
                              'policy_row_retention': cfg['game'].policy_row_retention,
                              'shard_inventory_sha256': hashlib.sha256(canonical(shard_files)).hexdigest()}
                write_json(stage / 'validation.json', validation)
                full_inventory = inventory(stage, seal=True)
                final = out / ('game-' + str(ordinal))
                require(not final.exists(), 'fixed attempt already banked')
                stage.rename(final)
                sync_dir(out)
                evidence = {name: {'path': str(final / filename), 'sha256': digest(final / filename)} for name, filename in
                            (('outcome_proof', 'outcome-proof.json'), ('provenance', 'provenance.json'),
                             ('validation', 'validation.json'), ('raw_metadata', 'raw-targets.json'))}
                completed[ordinal] = {**common, 'validation': validation, 'evidence': evidence,
                    'path': str(final), 'inventory': full_inventory,
                    'shard': {'path': str(final / 'shard'), 'files': shard_files} if samples else None}
                write_json(out / ('completed-' + str(ordinal) + '.json'), completed[ordinal])
                pending.pop(audit.game_id, None)
            except BaseException as error:
                evidence = failure_sample_evidence(samples)
                actual_directory = stage if stage.exists() else out / ('game-' + str(ordinal))
                evidence.update(error=repr(error), ordinal=ordinal, planned=planned,
                                attempt_directory=str(actual_directory), pending_game_counts={str(k): len(v) for k, v in pending.items()})
                write_json(out / ('failed-callback-' + stage.name[1:] + '.json'), evidence)
                raise

        class WitnessedPool(StockfishPool):
            def _new_engine(self) -> Any:
                engine = super()._new_engine()
                try:
                    validate_stockfish_contract(engine, protocol['tablebase']['path'])
                    birth = proc_birth(engine.proc.pid)
                    require(birth is not None and engine.proc.poll() is None, 'SF died after explicit readiness')
                    with lock:
                        processes[engine.proc.pid] = (engine.proc, birth)
                        events.append({'pid': engine.proc.pid, 'birth': birth,
                                       'syzygy_path': engine.syzygy_path, 'syzygy_50_move_rule': engine.syzygy_50_move_rule,
                                       'syzygy_probe_limit': engine.syzygy_probe_limit,
                                       'ready_after_explicit_options': engine.syzygy_ready_after_requests})
                except BaseException:
                    engine.close()
                    raise
                return engine

        sf = protocol['stockfish']
        pool = WitnessedPool(path=sf['path'], nodes=sf['nodes'], multipv=sf['multipv'], hash_mb=sf['hash_mb'],
                            num_workers=sf['workers'], syzygy_path=protocol['tablebase']['path'],
                            syzygy_50_move_rule=True, syzygy_probe_limit=6, read_timeout_s=sf['read_timeout_seconds'])
        validate_stockfish_contract(pool, protocol['tablebase']['path'])
        queue = Path(protocol['starts']['path']).read_text().splitlines()
        _, stats = play_batch(model, device='cuda:0', rng=np.random.default_rng(protocol['seed']), stockfish=pool,
            evaluator=evaluator, games=protocol['concurrent_slots'], target_games=protocol['attempts'],
            on_game_complete=on_samples, on_game_provenance=on_game, fen_dole_queue=queue, **cfg)
        require(set(completed) == set(range(protocol['attempts'])) and stats.games == protocol['attempts'] and
                not queue and not pending and forwards['calls'] > 0, 'incomplete predetermined generation')
        hook.stop()
        hook = None
        serving['post_state_sha256'] = state_sha256(model.state_dict())
        require(serving['post_state_sha256'] == serving['pre_state_sha256'] and
                digest(protocol['actor']['path']) == protocol['actor']['sha256'] and
                file_stamp(Path(protocol['actor']['path'])) == serving['checkpoint_stat'], 'served actor mutated during generation')
        serving['forwards'] = {**forwards, 'scope': 'actual _encode_tokens calls, including compact legal-policy forwards'}
        write_json(out / 'actor-serving.json', serving)
        overlap = exclusions_gate(completed, protocol)
        write_json(out / 'input-exclusions.json', overlap)
        lifecycle = close_pool()
        write_json(out / 'stockfish-lifecycle.json', lifecycle)
        verifier.close()
        source_guard(protocol)
        write_json(out / 'project-imports.json', project_import_origins(protocol))
        readout = {'status': 'COLLECTED_CORE_GENERATION_V1', 'run_uuid': protocol['run_uuid'],
            'collection_uuid': protocol['collection_uuid'], 'generation': protocol['generation'],
            'source_revision': protocol['source']['revision'], 'protocol_sha256': protocol_sha256,
            'actor_sha256': protocol['actor']['sha256'], 'actor_serving': {'path': str(out / 'actor-serving.json'),
                'sha256': digest(out / 'actor-serving.json')}, 'attempts': protocol['attempts'],
            'all_planned_attempts_accounted': True, 'completed': completed,
            'class_counts': {kind: sum(x['scientific_class'] == kind for x in completed.values())
                for kind in ('NATURAL', 'STRICT_TB_ADJUDICATED', 'UNRESOLVED')},
            'resolved_nonempty_games': sum(x['rows'] > 0 and x['scientific_class'] != 'UNRESOLVED' for x in completed.values()),
            'playing_and_exposure_gate_qualified': overlap['qualified'],
            'replacement_or_outcome_selection': False, 'training_steps': 0,
            'root_acceptance_issued': False, 'learning_admitted': False,
            'elapsed_seconds': time.monotonic() - started, 'verification_handle_closed': True, 'pool_closed': True}
        write_json(out / 'readout.json', readout)
        inventory(out, seal=True)
        return 0 if overlap['qualified'] else 2
    except BaseException as error:
        cleanup_errors = []
        try:
            lifecycle = close_pool()
            write_json(out / 'stockfish-lifecycle-failed.json', lifecycle)
        except BaseException as closing:
            cleanup_errors.append(repr(closing))
        if verifier is not None:
            try:
                verifier.close()
            except BaseException as closing:
                cleanup_errors.append(repr(closing))
        write_json(out / 'failure.json', {'status': 'FAILED_CORE_GENERATION_V1', 'error': repr(error),
            'completed_ordinals': sorted(completed), 'uncompleted_ordinals_unknown_start_state':
                sorted(set(range(protocol['attempts'])) - set(completed)), 'cleanup_errors': cleanup_errors,
            'root_acceptance_issued': False, 'learning_admitted': False, 'replacement_permitted': False})
        inventory(out, seal=True)
        raise
    finally:
        signal.alarm(0)
        try:
            if pool is not None:
                pool.close()
        finally:
            if verifier is not None:
                verifier.close()
            if hook is not None:
                hook.stop()
            evaluator = model = None
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    protocol = checked({'path': str(args.protocol), 'sha256': args.protocol_sha256})
    roster = checked(protocol['roster'])
    require(digest(protocol['starts']['path']) == protocol['starts']['sha256'], 'starts changed')
    validate_protocol(protocol, roster)
    exclusion_inputs(protocol)
    if args.execute:
        require(args.admission is not None and args.admission_sha256 is not None, 'separate finite root admission required')
        verify_execution(protocol, args.protocol_sha256, checked({'path': str(args.admission), 'sha256': args.admission_sha256}))
    native = source_guard(protocol)
    cfg, profile = build_configs(protocol, roster)
    require(hashlib.sha256(canonical(profile)).hexdigest() == protocol['resolved_profile_sha256'], 'actual realized config profile changed')
    require(not Path(protocol['output_root']).exists(), 'output root already exists')
    if args.check:
        import torch
        require(not torch.cuda.is_initialized(), 'descriptor check initialized CUDA')
        report = {'status': 'CHECKED_CORE_GENERATION_DESCRIPTOR_V1', 'source_revision': protocol['source']['revision'],
                  'profile': profile, 'models_created': False, 'engines_created': False,
                  'tablebase_handles_opened': False, 'execution_admitted': False,
                  'project_imports': project_import_origins(protocol)}
        if args.report:
            write_json(args.report, report)
        return 0
    return execute(protocol, args.protocol_sha256, roster, cfg, profile, native)


if __name__ == '__main__':
    raise SystemExit(main())
