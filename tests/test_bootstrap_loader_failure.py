"""BO2: bounded loader failures must terminate and prohibit training."""
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts import train_bootstrap as bootstrap


@pytest.mark.parametrize('fail', [False, True])
def test_loader_always_delivers_terminal_marker(monkeypatch, fail):
    error = OSError('bad shard')

    def load(path):
        if fail:
            raise error
        return [path], {}

    monkeypatch.setattr(bootstrap, 'load_npz', load)
    q = queue.Queue(maxsize=2)
    worker = threading.Thread(target=bootstrap._loader_thread, args=(['shard'], q))
    worker.start()
    try:
        first = q.get(timeout=1)
        assert first is error if fail else first == ['shard']
        assert q.get(timeout=1) is None
    finally:
        worker.join(timeout=1)
    assert not worker.is_alive()


@pytest.mark.parametrize('max_positions', [0, 1])
@pytest.mark.parametrize('fail', [False, True])
def test_cli_drains_loader_failure_before_training(tmp_path, monkeypatch, fail, max_positions):
    for name in 'abcdefghi':
        (tmp_path / f'{name}.npz').touch()
    error = OSError('synthetic unreadable shard')
    calls = []
    workers = []
    real_thread = threading.Thread

    def thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        workers.append(t)
        return t

    class BoundedQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            return super().get(block=block, timeout=2 if timeout is None else timeout)

    def load(path):
        if fail and path.endswith('a.npz'):
            raise error
        return [object()], {}

    def train(*_args, **_kwargs):
        calls.append('train')
        return SimpleNamespace(loss=0., wdl_loss=0., moves_left_loss=0.)

    monkeypatch.setattr(bootstrap.threading, 'Thread', thread)
    monkeypatch.setattr(bootstrap.queue, 'Queue', BoundedQueue)
    monkeypatch.setattr(bootstrap, 'load_yaml_file', lambda _: {'device': 'cpu'})
    monkeypatch.setattr(bootstrap, 'build_model', lambda _: object())
    monkeypatch.setattr(bootstrap, 'Trainer', lambda *a, **kw: SimpleNamespace(
        train_steps=train, save=lambda _: calls.append('save')))
    monkeypatch.setattr(bootstrap, 'ReplayBuffer',
                        lambda capacity, **kw: SimpleNamespace(capacity=capacity, add_many=lambda _: None))
    monkeypatch.setattr(bootstrap, 'load_npz', load)
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--config', 'unused',
        '--bootstrap-dir', str(tmp_path), '--out', str(tmp_path / 'unused.pt'),
        '--loaders', '2', '--batch-size', '1', '--max-positions', str(max_positions)])
    try:
        if fail:
            with pytest.raises(RuntimeError, match='shard loading failed') as caught:
                bootstrap.main()
            assert caught.value.__cause__ is error
            assert calls == []
        else:
            bootstrap.main()
            assert calls == ['train', 'save']
    finally:
        for worker in workers:
            worker.join(timeout=1)
    assert workers
    assert all(not worker.is_alive() for worker in workers)
