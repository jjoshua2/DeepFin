"""The parser must not turn malformed/incomplete native output into success."""
import pytest

from native.bend_engine.collections_probe.run_probe import (
    SESSION_FOREIGN_DEFS,
    expected_checksum,
    expected_trace,
    expected_traversal,
    parse_benchmark,
    session_notice,
    verify_trace,
    verify_traversal,
)


def sample_text() -> str:
    rows = []
    for size in (16, 256, 4096):
        for sample in range(6):
            arms = ('list', 'queue') if sample % 2 == 0 else ('queue', 'list')
            rows.extend(f'{arm} {size} {sample} 10 {expected_checksum(size)} {size}' for arm in arms)
    return '\n'.join(rows) + '\n'


def test_reference_outputs() -> None:
    verify_trace(expected_trace())
    verify_traversal(expected_traversal())
    rows = parse_benchmark(sample_text())
    assert len(rows) == 36
    assert all(row['below_20ms'] for row in rows)
    assert sum(bool(row['warmup']) for row in rows) == 6


@pytest.mark.parametrize('mutation', ['empty', 'truncated', 'extra', 'value', 'length', 'wide'])
def test_bad_queue_trace(mutation: str) -> None:
    text = expected_trace()
    if mutation == 'empty':
        text = ''
    elif mutation == 'truncated':
        text = '\n'.join(text.splitlines()[:-1]) + '\n'
    elif mutation == 'extra':
        text += 'empty\n'
    elif mutation == 'value':
        text = text.replace('value ', 'bad ', 1)
    elif mutation == 'length':
        text = text.replace('length 0', 'length 1')
    else:
        words = text.splitlines()
        at = next(i for i, line in enumerate(words) if line.startswith('value '))
        row = words[at].split()
        words[at] = f'value 0 {row[2]}'
        text = '\n'.join(words) + '\n'
    with pytest.raises(ValueError, match='owning FIFO trace differs'):
        verify_trace(text)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'order', 'checksum', 'length', 'negative', 'decimal'])
def test_bad_benchmark(mutation: str) -> None:
    text = sample_text()
    rows = text.splitlines()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'extra':
        rows.append(rows[-1])
    elif mutation == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    else:
        words = rows[0].split()
        field, value = {'checksum': (4, '0'), 'length': (5, '0'), 'negative': (3, '-1'), 'decimal': (3, '1.5')}[mutation]
        words[field] = value
        rows[0] = ' '.join(words)
    with pytest.raises(ValueError, match='benchmark'):
        parse_benchmark('\n'.join(rows) + '\n')


@pytest.mark.parametrize('text', ['', 'select 0 0 33 1\n', 'backup 0 33 1 0 2 0\n'])
def test_bad_traversal(text: str) -> None:
    with pytest.raises(ValueError, match='search traversal differs'):
        verify_traversal(text)


def test_known_foreign_notice() -> None:
    assert len(SESSION_FOREIGN_DEFS) == 18
    text = 'All terms check, but 18 defs rely on unsafe or foreign code:\n' + '\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS) + '\n'
    assert session_notice(0, '', text) == text


@pytest.mark.parametrize('mutation', ['exit', 'stdout', 'missing', 'extra', 'count', 'name', 'clean'])
def test_unexpected_foreign_notice(mutation: str) -> None:
    text = 'All terms check, but 18 defs rely on unsafe or foreign code:\n' + '\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS) + '\n'
    code, output = 0, ''
    if mutation == 'exit':
        code = 1
    elif mutation == 'stdout':
        output = 'unexpected warning'
    elif mutation == 'missing':
        text = ''
    elif mutation == 'extra':
        text += '- new_unsafe\n'
    elif mutation == 'count':
        text = text.replace('18 defs', '19 defs')
    elif mutation == 'name':
        text = text.replace('- Job.load', '- invented')
    else:
        text = 'All terms check.\n'
    with pytest.raises(ValueError, match='unexpected session compiler diagnostic'):
        session_notice(code, output, text)
