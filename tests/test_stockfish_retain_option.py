from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts import gen_sf_rooted_corpus as corpus
from tests.test_gen_sf_rooted_corpus import worker_spec


def fake_engine(tmp_path: Path, advertisement: str) -> tuple[Path, Path]:
    log = tmp_path / 'commands.log'
    engine = tmp_path / 'engine'
    engine.write_text(
        f'#!{sys.executable}\n'
        'import sys\n'
        'from pathlib import Path\n'
        f'log = Path({str(log)!r})\n'
        'for line in sys.stdin:\n'
        '    command = line.strip()\n'
        '    with log.open("a") as stream: stream.write(command + "\\n")\n'
        '    if command == "uci":\n'
        f'        print({advertisement!r}, flush=True)\n'
        '        print("uciok", flush=True)\n'
        '    elif command == "isready": print("readyok", flush=True)\n'
        '    elif command == "quit": break\n',
    )
    engine.chmod(0o755)
    return engine, log


@pytest.mark.parametrize('enabled', [False, True])
def test_retention_wire_and_default(tmp_path: Path, enabled: bool) -> None:
    path, log = fake_engine(tmp_path, 'option name SyzygyRetainOnNewGame type check default false')
    sf = StockfishUCI(str(path), retain_syzygy_on_new_game=enabled, read_timeout_s=2)
    try:
        assert sf.retain_syzygy_option_sent is enabled
        sf.new_game()
    finally:
        sf.close()
    lines = log.read_text().splitlines()
    assert [line for line in lines if line.startswith('setoption')] == [
        'setoption name UCI_ShowWDL value true', 'setoption name Threads value 1',
        *(['setoption name SyzygyRetainOnNewGame value true'] if enabled else []),
    ]
    assert 'ucinewgame' in lines


@pytest.mark.parametrize('advertisement', [
    'id name stockfish',
    'info string option name SyzygyRetainOnNewGame type check default false',
    'option name SyzygyRetainOnNewGame type string default false',
    'option name SyzygyRetainOnNewGame2 type check default false',
])
def test_retention_rejects_unsupported_engine_and_closes(
    tmp_path: Path, advertisement: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, log = fake_engine(tmp_path, advertisement)
    children = []
    original = subprocess.Popen
    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', capture)
    with pytest.raises(ValueError, match='does not advertise'):
        StockfishUCI(str(path), retain_syzygy_on_new_game=True, read_timeout_s=2)
    assert children
    assert all(child.poll() is not None for child in children)
    assert not any(line.startswith('setoption') for line in log.read_text().splitlines())


def test_retention_is_not_enabled_by_truthy_string() -> None:
    with pytest.raises(ValueError, match='must be a bool'):
        StockfishUCI('/not-launched', retain_syzygy_on_new_game='false')  # pyright: ignore[reportArgumentType]


def test_worker_factory_forwards_retention_on_initial_and_replacement_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, log = fake_engine(tmp_path, 'option name SyzygyRetainOnNewGame type check default false')
    class ReplaceOnce(corpus.EngineLease):
        def __init__(self, factory):
            super().__init__(factory)
            self.respawn()
    monkeypatch.setattr(corpus, 'EngineLease', ReplaceOnce)
    out = tmp_path / 'corpus'
    out.mkdir()
    result = corpus.run_worker(worker_spec(
        out, game_ids=(), sf_binary=str(path), sf_retain_syzygy_on_new_game=True,
    ))
    assert result['failed'] is None
    assert result['realized']['sf_syzygy_retain_option_sent'] is True
    assert log.read_text().splitlines().count('setoption name SyzygyRetainOnNewGame value true') == 2


def test_retention_config_changes_identity_only_when_opted_in() -> None:
    parser = corpus.build_parser()
    args = parser.parse_args(['--out-dir', '/tmp/unused'])
    before = corpus.config_stamp(args, sf_binary='stockfish')
    assert 'sf_retain_syzygy_on_new_game' not in before
    args = parser.parse_args(['--out-dir', '/tmp/unused', '--sf-retain-syzygy-on-new-game'])
    after = corpus.config_stamp(args, sf_binary='stockfish')
    assert after.pop('sf_retain_syzygy_on_new_game') is True
    assert after == before


def test_retention_cannot_change_on_resume_and_legacy_namespace_stays_off() -> None:
    args = corpus.build_parser().parse_args(['--out-dir', '/tmp/unused'])
    off = corpus.config_stamp(args, sf_binary='stockfish')
    vars(args).pop('sf_retain_syzygy_on_new_game')
    assert corpus.config_stamp(args, sf_binary='stockfish') == off
    on = {**off, 'sf_retain_syzygy_on_new_game': True}
    corpus.refuse_resume_config_drift({'config_requested': off}, requested=off)
    corpus.refuse_resume_config_drift({'config_requested': on}, requested=on)
    for before, after in [(off, on), (on, off)]:
        with pytest.raises(ValueError, match='sf_retain_syzygy_on_new_game'):
            corpus.refuse_resume_config_drift({'config_requested': before}, requested=after)


def test_cli_retention_reaches_frozen_worker_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    original = corpus.WorkerSpec
    class Captured(Exception):
        pass
    def capture(**kwargs):
        seen.append(original(**kwargs))
        raise Captured
    monkeypatch.setattr(corpus, 'WorkerSpec', capture)
    monkeypatch.setattr(corpus, 'refuse_unopenable_syzygy', lambda path: ())
    monkeypatch.setattr(corpus.audit_targets, 'engine_identity', lambda path: 'test')
    args = corpus.build_parser().parse_args([
        '--out-dir', str(tmp_path/'corpus'), '--games', '1', '--workers', '1',
        '--stockfish', '/bin/true', '--sf-retain-syzygy-on-new-game',
    ])
    with pytest.raises(Captured):
        corpus.run(args)
    assert seen[0].sf_retain_syzygy_on_new_game is True
