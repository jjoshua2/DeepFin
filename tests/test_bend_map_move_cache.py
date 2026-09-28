"""Cache admission, move ordering and independent-reference evidence contracts."""
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from native.bend_engine.u64_map_probe import move_cache as m

START = m.legal.fen_position(m.legal.START)
E4 = m.legal.fen_position('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')
E3 = m.legal.fen_position('rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1')


def references() -> list[m.Reference]:
    first = m.Reference(START, 0, 1, 0, {(12, 28, 0, 0): E4, (12, 20, 0, 0): E3})
    return [first, replace(first, halfmove=4, fullmove=3, history=4)]


def trace(refs: list[m.Reference], bits: int = 3) -> str:
    lines = []
    for number, (ref, route) in enumerate(zip(refs, m.accesses(refs, bits), strict=True)):
        lines.extend(f'reference {number} ' + ' '.join(map(str, move)) for move in ref.moves)
        lines += [f'reference-end {number}', f'request {number} {route} {len(ref.moves)}',
                  f'context {number} {ref.halfmove} {ref.fullmove} {ref.history}']
        for move, board in ref.moves.items():
            words = [*move, *(v for word in board[:8] for v in (word >> 32, word & 4294967295)), *board[8:]]
            lines.append(f'move {number} ' + ' '.join(map(str, words)))
        lines.append(f'end {number}')
    return '\n'.join([*lines, f'done {len(refs)}']) + '\n'


def test_hit_preserves_context_and_native_order() -> None:
    refs = references()
    assert m.verify(trace(refs), refs, 3) == {'filled': 1, 'hit': 1, 'bypassed': 0}


def test_empty_move_list_is_cached_not_treated_as_missing() -> None:
    refs = [m.Reference(START, 0, 1, 0, {})] * 2  # Parser contract, not a legal-board assertion.
    assert m.verify(trace(refs), refs, 3)['hit'] == 1


def test_full_cache_bypasses_new_boards_without_evicting_old_ones() -> None:
    a = references()[0]
    b = replace(a, position=E4)
    refs = [a, b, b, a]
    assert m.accesses(refs, 1) == ['filled', 'bypassed', 'bypassed', 'hit']
    assert m.verify(trace(refs, 1), refs, 1) == {'filled': 1, 'bypassed': 2, 'hit': 1}


def test_unavailable_ep_does_not_partition_cache() -> None:
    absent = (*E4[:-1], 64)
    assert m.canonical(absent) == m.canonical(E4)


@pytest.mark.parametrize('fen', [
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1',
])
def test_available_and_pinned_ep_remain_distinct(fen: str) -> None:
    position = m.legal.fen_position(fen)
    assert m.canonical(position) != m.canonical((*position[:-1], 64))


@pytest.mark.parametrize('bits', [0, 17, True, 1.0, -1])
def test_bad_reference_capacity(bits: int) -> None:
    with pytest.raises(ValueError, match='capacity'):
        m.accesses(references(), bits)


@pytest.mark.parametrize('fault', ['route', 'count', 'context', 'missing', 'extra', 'duplicate',
                                  'move', 'child', 'order', 'reference_order', 'newline', 'ordinal'])
def test_bad_results_cannot_pass(fault: str) -> None:
    refs = references()
    lines = trace(refs).splitlines()
    if fault == 'route':
        lines = [s.replace('request 1 hit', 'request 1 filled') for s in lines]
    elif fault == 'count':
        lines = [s.replace('request 0 filled 2', 'request 0 filled 1') for s in lines]
    elif fault == 'context':
        lines = [s.replace('context 1 4 3 4', 'context 1 0 1 0') for s in lines]
    elif fault == 'extra':
        lines.append('done 2')
    elif fault == 'missing':
        lines.pop(5)
    elif fault == 'reference_order':
        lines[0], lines[1] = lines[1], lines[0]
    elif fault == 'ordinal':
        lines = [s.replace('reference-end 0', 'reference-end 1') for s in lines]
    else:
        indices = [i for i, line in enumerate(lines) if line.startswith('move 0 ')]
        a, b = indices
        if fault == 'duplicate':
            lines[b] = lines[a]
        elif fault == 'order':
            lines[a], lines[b] = lines[b], lines[a]
        elif fault in ('move', 'child'):
            words = lines[a].split()
            words[3 if fault == 'move' else 6] = '2'
            lines[a] = ' '.join(words)
    changed = '\n'.join(lines) + ('' if fault == 'newline' else '\n')
    with pytest.raises(ValueError, match=r'cache|context|request|reference'):
        m.verify(changed, refs, 3)


def test_fixture_transport_is_bounded_and_covers_special_moves() -> None:
    cases = m.fixtures()
    assert len(cases) == len({case.name for case in cases})
    names = {case.name for case in cases}
    assert {'legal-checkmate', 'legal-stalemate', 'legal-promote_white', 'legal-castle_white',
            'interleaved', 'saturated', 'history-and-unused-ep'} <= names
    assert all(len(m.board_index.encode(case)) <= 100_000 for case in cases)


def test_oracle_diagnostics_cannot_be_accepted_as_move_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, 'end\n', 'warning\n'))
    with pytest.raises(ValueError, match='oracle failed'):
        m.Oracle(Path('oracle'), tmp_path).moves(START)
    assert len(list(tmp_path.glob('*.stderr'))) == 1


def test_oracle_reuses_only_identical_full_position_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0
    def execute(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal count
        count += 1
        return subprocess.CompletedProcess([], 0, 'end\n', '')
    monkeypatch.setattr(m.subprocess, 'run', execute)
    oracle = m.Oracle(Path('oracle'), tmp_path)
    assert oracle.moves(START) == oracle.moves(START) == {}
    oracle.moves(E4)
    assert count == 2
