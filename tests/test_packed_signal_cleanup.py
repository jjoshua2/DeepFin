"""Real process-directed stops with a preexisting unblocked helper thread."""
import os
import signal
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize("stage_name", ["pair", "preparation"])
@pytest.mark.parametrize(("signal_name", "stop_point"),
    [(sig, point) for sig in ("SIGINT", "SIGTERM", "SIGALRM") for point in ("term", "kill")]
    + [("SIGALRM", "wait")])
def test_process_stop_during_cleanup_reaps_owned_child_with_unblocked_thread(
    tmp_path, monkeypatch, stage_name, signal_name, stop_point,
):
    from scripts import run_packed_trainer_pair as pair
    from scripts import run_packed_trainer_preparation as preparation
    stage = pair if stage_name == "pair" else preparation
    chosen = getattr(signal, signal_name)
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    original_handler = signal.getsignal(chosen)
    original_killpg = os.killpg
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c",
        "import signal,pathlib,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)", str(ready)],
        start_new_session=True)
    release = threading.Event()
    thread_ready = threading.Event()
    def unblocked_thread():
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {chosen})
        thread_ready.set()
        release.wait(12)
    thread = threading.Thread(target=unblocked_thread)
    thread.start()  # created before pair masks only its current thread
    delivered = []
    sent = []
    def interrupted(signum, _frame):
        delivered.append((signum, child.returncode))
        raise RuntimeError("process-directed cleanup stop")
    def killpg(group, sig):
        assert group == child.pid
        sent.append(sig)
        original_killpg(group, sig)
        if (stop_point == "term" and sig == signal.SIGTERM) or (stop_point == "kill" and sig == signal.SIGKILL):
            if stage_name == "pair":
                assert chosen in signal.pthread_sigmask(signal.SIG_BLOCK, set())
            os.kill(os.getpid(), chosen)  # actual process signal, never pthread_kill/main routing
            time.sleep(0.1)  # let the unblocked thread receive while main's OS mask is blocked
    actual_wait = child.wait
    wait_stop_sent = False
    def wait(*args, **kwargs):
        nonlocal wait_stop_sent
        if stop_point == "wait" and signal.SIGKILL in sent and not wait_stop_sent:
            wait_stop_sent = True
            os.kill(os.getpid(), chosen)
            time.sleep(0.1)
        return actual_wait(*args, **kwargs)
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("enter cleanup")
    try:
        assert thread_ready.wait(2)
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        signal.signal(chosen, interrupted)
        monkeypatch.setattr(stage.subprocess, "Popen", lambda *_args, **_kwargs: child)
        monkeypatch.setattr(stage.os, "killpg", killpg)
        monkeypatch.setattr(child, "wait", wait)
        with (tmp_path / "lease").open("a") as lease:
            kwargs = {"lease_fd": lease.fileno()} if stage_name == "pair" else {}
            with pytest.raises(RuntimeError, match="process-directed cleanup stop"):
                stage.run_stage(["already owned fixture child"], cwd=tmp_path, env=dict(os.environ),
                    log=tmp_path / "child.log", guard=guard, **kwargs)
        assert delivered == [(chosen, -signal.SIGKILL)]
        assert signal.getsignal(chosen) is interrupted
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask
        assert sent == [signal.SIGTERM, signal.SIGKILL]
        assert child.returncode == -signal.SIGKILL  # actual wait/reap already complete, not poll-induced reap
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    finally:
        # Own only this test's unreaped Popen leader/session; never enumerate or kill other processes.
        signal.signal(chosen, original_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        release.set()
        thread.join(timeout=2)
        if child.poll() is None:
            original_killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
        assert not thread.is_alive()



def test_ignored_stop_and_all_dispositions_remain_unchanged(monkeypatch):
    from scripts import run_packed_trainer_preparation as preparation
    prior = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    sent = []
    class OwnedSeam:
        pid = 123456789
        returncode = None
        def wait(self, timeout):
            assert timeout == 3
            self.returncode = 0
            return 0
    child = OwnedSeam()
    def killpg(group, sig):
        assert group == child.pid
        sent.append(sig)
        if sig == signal.SIGTERM:
            assert signal.getsignal(sig) == signal.SIG_IGN
            os.kill(os.getpid(), sig)  # real ignored process signal; group send is a seam
    monkeypatch.setattr(preparation.os, "killpg", killpg)
    try:
        preparation.stop_owned(child)
        assert child.returncode == 0
        assert sent == [signal.SIGTERM, signal.SIGKILL]
        assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == mask
    finally:
        signal.signal(signal.SIGTERM, prior)


def test_default_stop_terminates_supervisor_only_after_owned_reap(tmp_path):
    import json
    from pathlib import Path
    # Isolate the default-fatal disposition from pytest. The supervisor's own
    # wait writes its reap proof BEFORE original SIG_DFL is redelivered.
    code = r"""
import json, os, pathlib, signal, subprocess, sys, time
from scripts import run_packed_trainer_preparation as preparation
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_DFL)
ready = root/'ready'
child = subprocess.Popen([sys.executable, '-c',
    "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)", str(ready)], start_new_session=True)
start = pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
(root/'owned.json').write_text(json.dumps({'pid':child.pid,'start':start}))
limit = time.monotonic()+3
while not ready.exists() and time.monotonic()<limit:
    time.sleep(.01)
assert ready.exists()
killpg, wait = os.killpg, child.wait
def send(group, sig):
    assert group == child.pid
    killpg(group,sig)
    if sig == signal.SIGTERM:
        os.kill(os.getpid(), signal.SIGTERM)
def reap(*args, **kwargs):
    value=wait(*args, **kwargs)
    if child.returncode == -signal.SIGKILL:
        try:
            os.waitpid(child.pid, os.WNOHANG)
        except ChildProcessError:
            (root/'reaped.json').write_text(json.dumps({'returncode':child.returncode,'echld':True}))
    return value
preparation.os.killpg, child.wait = send, reap
preparation.stop_owned(child)
raise AssertionError('default SIGTERM failed to terminate after cleanup')
"""
    supervisor = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        _out, err = supervisor.communicate(timeout=10)
        assert supervisor.returncode == -signal.SIGTERM, err.decode()
        assert json.loads((tmp_path/"reaped.json").read_text()) == {"returncode": -signal.SIGKILL, "echld": True}
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=3)
        owned = tmp_path/"owned.json"
        if owned.exists():
            record = json.loads(owned.read_text())
            try:
                state = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                pass
            else:
                # Only the fixture-authenticated unreaped birth/session, never a
                # reused PID or an unrelated process, is eligible for cleanup.
                if state[19] == record["start"] and state[0] != "Z":
                    assert os.getpgid(record["pid"]) == record["pid"]
                    os.killpg(record["pid"], signal.SIGKILL)
