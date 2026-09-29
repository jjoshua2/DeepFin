"""Native board adapter's protocol and independent identity-based ID oracle."""
from dataclasses import replace

import pytest

from native.bend_engine.u64_map_probe import board_index as b

START = (71776119061282560, 4755801206503243842, 2594073385365405732,
         9295429630892703873, 576460752303423496, 1152921504606846992,
         65535, 18446462598732840960, 1, 15, 64)


def snapshot() -> b.Snapshot:
    return b.Snapshot(START, 64, 0, 1, 0)


def test_start_position_known_hash_and_complete_words() -> None:
    assert b.fingerprint(START) == (206154439 << 32 | 2955100549)
    text = b.expected([snapshot()], 7)
    assert text.startswith('identity 0 16711680 65280 1107296256 66 ')
    assert text.endswith('context 0 64 0 1 0\nintern 0 assigned 0 size 1\nget 0 value 0 size 1\nend 1\n')
    b.verify(text, text)


def test_history_clock_and_raw_unused_ep_do_not_split_structural_id() -> None:
    first = snapshot()
    second = replace(first, raw_ep=20, halfmove=99, fullmove=42, history=4)
    text = b.expected([first, second], 7)
    assert 'context 1 20 99 42 4\n' in text
    assert 'intern 1 known 0 size 1\n' in text
    assert text.endswith('end 1\n')


@pytest.mark.parametrize('field', range(11))
def test_every_identity_field_controls_record_identity(field: int) -> None:
    fields: list[int] = list(START)
    fields[field] ^= 1
    changed = replace(snapshot(), identity=tuple(fields))
    text = b.expected([snapshot(), changed, snapshot()], 7)
    assert 'intern 1 assigned 1 size 2\n' in text
    assert 'intern 2 known 0 size 2\n' in text


def test_oracle_ids_do_not_depend_on_hash_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b, 'fingerprint', lambda _fields: 0)
    fields = (*START[:-1], 43)
    text = b.expected([snapshot(), replace(snapshot(), identity=fields)], 7)
    assert 'hash 0 0 0\n' in text
    assert 'hash 1 0 0\n' in text
    assert 'intern 1 assigned 1 size 2\n' in text


def test_full_index_preserves_known_and_does_not_return_rejected_id() -> None:
    changed = replace(snapshot(), identity=(*START[:-1], 43))
    text = b.expected([snapshot(), changed, snapshot()], 1)
    assert 'intern 1 full size 1\nget 1 missing size 1\n' in text
    assert 'intern 2 known 0 size 1\n' in text


@pytest.mark.parametrize('fault', ['identity', 'hash', 'clock', 'id', 'size', 'extra', 'newline'])
def test_corrupt_native_trace_rejected(fault: str) -> None:
    text = b.expected([snapshot()], 7)
    if fault == 'extra':
        changed = text + 'end 1\n'
    elif fault == 'newline':
        changed = text[:-1]
    else:
        old, new = {'identity': ('16711680 65280', '0 65280'),
                    'hash': ('206154439', '0'), 'clock': ('context 0 64 0 1 0', 'context 0 64 1 1 0'),
                    'id': ('assigned 0', 'assigned 1'), 'size': ('size 1', 'size 0')}[fault]
        assert old in text
        changed = text.replace(old, new, 1)
    with pytest.raises(ValueError, match='output differs'):
        b.verify(changed, text)


@pytest.mark.parametrize('bits', [0, 17, True, 7.0, -1])
def test_bad_capacity_transport(bits: int) -> None:
    with pytest.raises(ValueError, match='capacity'):
        b.encode(b.Case('bad', ('startpos',), bits))


@pytest.mark.parametrize('positions', [(), ('startpos',) * 129, ('',), ('startpos;startpos',),
                                        ('startpos\n',), ('startpos\x00',), ('stärtpos',)])
def test_invalid_or_unbounded_transport(positions: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match=r'count|transport'):
        b.encode(b.Case('bad', positions))


def test_transport_budget_is_not_only_position_count() -> None:
    with pytest.raises(ValueError, match='budget'):
        b.encode(b.Case('large', ('startpos ' + 'x' * 100000,)))


def test_corpus_chunks_cover_every_record_and_repeat_endpoints() -> None:
    records = [{'fen': f'fixture-{i}'} for i in range(129)]
    cases = b.corpus_cases({'corpora': [{'name': 'test', 'records': records}]})
    assert [len(case.positions) for case in cases] == [66, 66, 3]
    assert [p for case in cases for p in case.positions[:-2]] == ['fen ' + row['fen'] for row in records]
    assert all(case.positions[-2:] == (case.positions[0], case.positions[-3]) for case in cases)


def test_fixtures_include_both_sides_and_special_moves() -> None:
    cases = b.fixtures()
    assert cases == b.fixtures()
    assert len(cases) == len({c.name for c in cases}) == 11
    assert len([c for c in cases if c.name.startswith('ep-boundary-')]) == 8
    special = next(c for c in cases if c.name == 'native-special-moves')
    for move in ('e5d6', 'e4d3', 'e1g1', 'e1c1', 'e8g8', 'e8c8',
                 *(f'a7a8{p}' for p in 'qrbn'), *(f'a2a1{p}' for p in 'qrbn')):
        assert any(command.endswith('moves ' + move) for command in special.positions)
    assert all(len(b.encode(c)) <= 100000 for c in cases)
