"""Transport a pinned small bank; compare outputs without evaluator/model imports."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import chess
from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg_bytes


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def read(ref: dict) -> bytes:
    p = Path(ref['path'])
    require(p.stat().st_size <= 2 * 1024**2, 'small transport input cap')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'transport hash')
    return data


def replay(case: dict) -> chess.Board:
    board = chess.Board(case['root_fen'])
    require(board.is_valid(), 'fixture root')
    for uci in case['moves']:
        move = chess.Move.from_uci(uci)
        require(move in board.legal_moves, 'fixture illegal move')
        board.push(move)
    return board


def export(plan: dict, out: Path) -> None:
    fixtures = json.loads(read(plan['fixtures']))
    histories = list(fixtures['histories'])
    require(len(histories) <= 16, 'explicit fixture bound')
    rows = json.loads(read(plan['value_rows']))
    raw = [json.loads(line) for line in read(plan['histories']).splitlines()]
    require(len(rows) == len(raw) == 128, 'fixed128 transport')
    import io
    with np.load(io.BytesIO(read(plan['matched_inputs'])), allow_pickle=False) as bank:
        saved = bank['ceres_byte_inputs'].copy()
    require(saved.dtype == np.uint8 and saved.shape == (128, 64, 137), 'saved byte input shape/type')
    values = list(fixtures['synthetic_values'])
    for i, (row, wrapped) in enumerate(zip(rows, raw, strict=True)):
        for key in ('worker_id', 'game_id', 'ply', 'input_key', 'physical_row'):
            require(row[key] == wrapped[key], '128 row alignment:' + key)
        data = wrapped['raw']
        case = {'id': f'bank_{i:03d}', 'root_fen': data['history_root_fen'], 'moves': data['history_uci']}
        require(replay(case).fen() == row['fen'], 'saved endpoint mismatch')
        histories.append(case)
        bits = {}
        for key, field in [('primary_bits', 'Ceres_primary'), ('secondary_bits', 'Ceres_secondary')]:
            original = np.asarray(row['native_outputs'][field], dtype=np.float64)
            half = original.astype('<f2')
            require(original.shape == (3,) and bool(np.isfinite(half).all())
                    and np.array_equal(half.astype(np.float64), original), 'raw logits not exact FP16 values')
            bits[key] = half.view('<u2').tolist()
        values.append({'id': case['id'], **bits})
    expected = {}
    for case in histories:
        encoded = encode_ceres_tpg_bytes(replay(case), q_negative_blunders=.03, q_positive_blunders=.03)
        expected[case['id']] = base64.b64encode(encoded.tobytes()).decode()
    # Stored Python byte convention is kept separately: a mismatch is an observation, not a repair.
    reference = {'python_bytes': expected, 'saved_bank_bytes': {f'bank_{i:03d}': base64.b64encode(x.tobytes()).decode() for i, x in enumerate(saved)}}
    out.mkdir()
    (out/'transport.json').write_text(json.dumps({'schema': 1, 'histories': histories, 'values': values})+'\n')
    (out/'reference.json').write_text(json.dumps(reference)+'\n')


def feature(offset: int) -> str:
    if offset < 104:
        return f'piece_history_slot{offset//13}_channel{offset%13}'
    if offset < 112:
        return f'repetition_slot{offset-104}'
    names = ['CanOO', 'CanOOO', 'OpponentCanOO', 'OpponentCanOOO', 'Move50Count', 'PlySinceLastMove', 'IsEnPassant', 'QPositiveBlunders', 'QNegativeBlunders']
    return names[offset-112] if offset < 121 else ('rank' + str(offset-121) if offset < 129 else 'file' + str(offset-129))


def compare(folder: Path) -> None:
    oracle = json.loads((folder/'oracle.json').read_text())
    reference = json.loads((folder/'reference.json').read_text())
    transport = json.loads((folder/'transport.json').read_text())
    require(len(oracle['encodings']) == len(transport['histories']) and len(oracle['outputs']) == len(transport['values']), 'oracle dimensions')
    mismatches = []
    for case, source in zip(oracle['encodings'], transport['histories'], strict=True):
        require(case['id'] == source['id'], 'oracle history order')
        actual = base64.b64decode(case['squares_base64'], validate=True)
        require(len(actual) == 64*137, 'oracle bytes')
        for convention in ('python_bytes', 'saved_bank_bytes'):
            if case['id'] not in reference[convention]:
                continue
            expected = base64.b64decode(reference[convention][case['id']], validate=True)
            for offset, (a, b) in enumerate(zip(actual, expected, strict=True)):
                if a != b:
                    mismatches.append({'id': case['id'], 'reference': convention, 'offset': offset,
                        'square_index': offset//137, 'feature': feature(offset%137), 'upstream': a, 'python': b})
    deltas = []
    for result, source in zip(oracle['outputs'], transport['values'], strict=True):
        require(result['id'] == source['id'], 'oracle value order')
        bits = np.asarray(result['bits'], dtype='<u2')
        require(bits.shape == (12,) and np.array_equal(bits.view('<f2').astype(float), np.asarray(result['values'])), 'getter bit transport')
        heads = []
        for key, temperature in [('primary_bits', .55), ('secondary_bits', 1.5)]:
            logits = np.asarray(source[key], dtype='<u2').view('<f2').astype(float) / temperature
            prob = np.exp(logits-logits.max())
            heads.append(prob/prob.sum())
        approximation = .6*heads[0]+.4*heads[1]
        deltas.append({'id': source['id'], 'upstream_getters': result,
            'mathematical_profile_wdl': approximation.tolist(),
            'actual_minus_mathematical_wdl': (np.asarray(result['values'][:3])-approximation).tolist()})
    report = {'status': 'COMPLETE_CPU_SEMANTICS_COMPARISON_NOT_NEURAL_PARITY', 'byte_mismatches': mismatches,
        'values': deltas, 'limits': 'Pinned upstream CPU getter arithmetic/encoder only; named parameterless profile, not effective deployment configuration, CUDA parity or teacher strength. Differences do not trigger production changes.'}
    (folder/'readout.json').write_text(json.dumps(report, indent=2)+'\n')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['export', 'compare'])
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'export':
        export(json.loads(a.plan.read_text()), a.out)
    else:
        compare(a.out)


if __name__ == '__main__':
    main()
