"""Exercise actual owned subprocesses with tiny pinned fake producer publications."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import ceres_materialize as tool


CHILD = """import argparse,hashlib,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--out');p.add_argument('--stop');a,_=p.parse_known_args()
m=json.loads(Path(a.manifest).read_text())
Path(m['pid_file']).write_text(str(os.getpid()))
Path(m['environment_file']).write_text(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS']}))
if m['mode']=='failure':raise SystemExit(7)
if m['mode']=='stop':
 signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(a.stop).touch()
 while True:time.sleep(.01)
out=Path(a.out);partial=out.with_name(out.name+'.writing');partial.mkdir();shard=partial/'shard_000000.zarr';shard.mkdir();(shard/'payload').write_bytes(b'fakevalue')
r={'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),'producer_sha256':m['producer_sha256'],'outputs':[{'path':shard.name,'output_storage_identity':hashlib.sha256(b'fakevalue').hexdigest()}]}
if m['mode']=='mismatch':r['manifest_sha256']='0'*64
(partial/m['summary_name']).write_text(json.dumps(r));(partial/'derive_targets_summary.json').write_text('{}');partial.rename(out)
"""


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cwd = tmp_path / "runtime"
    scripts = cwd / "scripts"
    scripts.mkdir(parents=True)
    base = tmp_path / "preparation"
    base.mkdir()
    monkeypatch.setattr(tool, "BASE", base)
    monkeypatch.setattr(tool, "LOCK", base / "preparation.lock")
    monkeypatch.setattr(tool, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(tool.owned, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(tool.owned, "CLEANUP_SECONDS", 0.3)
    real_disk = tool.shutil.disk_usage
    monkeypatch.setattr(
        tool.shutil, "disk_usage", lambda _: SimpleNamespace(free=tool.RESERVE + 2**30)
    )
    producer_file = scripts / "ceres_target_mix.py"
    producer_file.write_text(CHILD)
    epoch_file, arena_file = scripts / "bt4_one_epoch_screen.py", scripts / "arena.py"
    epoch_file.write_text("# fake admission\n")
    arena_file.write_text("# fake stage\n")
    producer_pins = {str(producer_file): tool.sha(producer_file)}
    manifest = tmp_path / "producer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": str(tmp_path / "source"),
                "entries": [],
                "mode": "success",
                "producer_sha256": producer_pins,
                "summary_name": "ceres_target_mix_summary.json",
                "pid_file": str(tmp_path / "producer.pid"),
                "environment_file": str(tmp_path / "environment.json"),
            }
        )
    )
    state, corpus = tmp_path / "state", tmp_path / "corpus"

    def read_manifest(filename, expected):
        tool.require(tool.sha(filename) == expected, "fake manifest pin")
        return json.loads(filename.read_text()), {}, []

    producer = SimpleNamespace(
        __file__=str(producer_file),
        SUMMARY="ceres_target_mix_summary.json",
        producer_pins=lambda: producer_pins,
        read_manifest=read_manifest,
        shared=SimpleNamespace(storage_identity=lambda p: tool.sha(p / "payload")),
    )
    admissions = []
    epoch = SimpleNamespace(
        __file__=str(epoch_file),
        arena=SimpleNamespace(__file__=str(arena_file)),
        CORPORA={"CeresB50": corpus},
        verify_ceres_recipe=lambda *a: admissions.append(a),
    )
    monkeypatch.setattr(tool, "modules", lambda _: (producer, epoch))
    monkeypatch.setattr(
        tool.subprocess,
        "check_output",
        lambda argv, **_kwargs: "a" * 40 + "\n" if argv[1] == "rev-parse" else "",
    )
    files = [
        producer_file,
        epoch_file,
        arena_file,
        Path(tool.owned.__file__),
        Path(tool.__file__),
        Path(sys.executable),
        manifest,
    ]
    p: dict[str, Any] = {
        "schema": 1,
        "profile": "CeresB50",
        "cwd": str(cwd),
        "commit": "a" * 40,
        "python": sys.executable,
        "state": str(state),
        "corpus": str(corpus),
        "producer_manifest": {"path": str(manifest), "sha256": tool.sha(manifest)},
        "producer_sha256": producer_pins,
        "pins": {
            str(f.resolve()) if f != Path(sys.executable) else str(f): tool.sha(f)
            for f in files
        },
        "supervisor_sha256": tool.sha(tool.__file__),
        "stop_paths": [
            str(base / "STOP"),
            str(state / "STOP"),
            str(state.parent / "STOP"),
        ],
    }

    def mode(name):
        m = json.loads(manifest.read_text())
        m["mode"] = name
        manifest.write_text(json.dumps(m))
        p["producer_manifest"]["sha256"] = p["pins"][str(manifest)] = tool.sha(manifest)

    return p, mode, admissions, real_disk


def test_success_has_actual_publication_and_terminal_admission_receipt(prepared):
    p, _, admissions, _ = prepared
    result = tool.execute(p, "f" * 64, time.time() + 120)
    assert result["status"] == "COMPLETE"
    assert result["returncode"] == 0
    assert result["profile"] == "CeresB50"
    assert result["corpus"] == p["corpus"]
    assert result["producer_manifest_sha256"] == p["producer_manifest"]["sha256"]
    assert result["producer_sha256"] == p["producer_sha256"]
    assert len(admissions) == 1
    assert result["derive_summary_sha256"] == tool.sha(
        Path(p["corpus"]) / "derive_targets_summary.json"
    )
    assert result["rewrite_summary_sha256"] == tool.sha(
        Path(p["corpus"]) / "ceres_target_mix_summary.json"
    )
    assert not tool.owned.process_group_present(result["pid"])
    env = json.loads((Path(p["state"]).parent / "environment.json").read_text())
    assert env == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    assert json.loads((Path(p["state"]) / "status.json").read_text()) == result
    with pytest.raises(ValueError, match="reuse"):
        tool.execute(p, "f" * 64, time.time() + 120)


def test_failed_child_never_qualifies_publication(prepared):
    p, mode, admissions, _ = prepared
    mode("failure")
    with pytest.raises(ValueError, match="exited7"):
        tool.execute(p, "f" * 64, time.time() + 120)
    r = json.loads((Path(p["state"]) / "status.json").read_text())
    assert r["status"] == "FAILED_OR_STOPPED"
    assert r["returncode"] == 7
    assert not admissions
    assert not Path(p["corpus"]).exists()
    assert not tool.owned.process_group_present(r["pid"])


def test_stop_kills_term_ignoring_producer_and_releases_preparation_lock(prepared):
    import fcntl

    p, mode, admissions, _ = prepared
    mode("stop")
    with pytest.raises(ValueError, match="STOP"):
        tool.execute(p, "f" * 64, time.time() + 120)
    r = json.loads((Path(p["state"]) / "status.json").read_text())
    assert r["status"] == "FAILED_OR_STOPPED"
    assert not admissions
    assert not tool.owned.process_group_present(r["pid"])
    with tool.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = int((Path(p["state"]).parent / "producer.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_zero_exit_with_wrong_manifest_is_not_complete(prepared):
    p, mode, _, _ = prepared
    mode("mismatch")
    with pytest.raises(ValueError, match="publication manifest"):
        tool.execute(p, "f" * 64, time.time() + 120)
    r = json.loads((Path(p["state"]) / "status.json").read_text())
    assert r["returncode"] == 0
    assert r["status"] == "FAILED_OR_STOPPED"
    assert Path(p["corpus"]).exists()  # Preserve failed publication for inspection.


def test_storage_identity_change_rejected(prepared):
    p, _, _, _ = prepared
    tool.execute(p, "f" * 64, time.time() + 120)
    (Path(p["corpus"]) / "shard_000000.zarr/payload").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output storage changed"):
        tool.verify_publication(p, lambda: None)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_profile",
        "missing_pin",
        "producer_pin",
        "overlap",
        "partial",
        "stop",
        "commit",
    ],
)
def test_preflight_rejects_before_child(prepared, defect):
    p, _, _, _ = prepared
    if defect == "unknown_profile":
        p["profile"] = "invented"
    elif defect == "missing_pin":
        p["pins"].pop(p["python"])
    elif defect == "producer_pin":
        p["producer_sha256"] = {"/unrelated.py": "0" * 64}
    elif defect == "overlap":
        p["state"] = p["corpus"]
    elif defect == "partial":
        tool.partial(p).mkdir()
    elif defect == "stop":
        Path(p["stop_paths"][0]).touch()
    else:
        p["commit"] = "b" * 40
    with pytest.raises(
        ValueError, match=r"profile|pin|producer|overlap|reuse|STOP|commit"
    ):
        tool.execute(p, "f" * 64, time.time() + 120)
    assert not (Path(p["state"]).parent / "producer.pid").exists()


def test_command_is_fixed_recipe_and_separate_timeout(prepared):
    p, _, _, _ = prepared
    command = tool.command(p, time.time() + 120)
    assert command[:3] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30s"]
    assert float(command[3][:-1]) <= 90
    for name in ["--bt4-weight", "--bt4-temperature", "--ceres-temperature"]:
        assert command[command.index(name) + 1] == "0.5"
    p["profile"] = "B100CeresV25"
    command = tool.command(p, time.time() + 120)
    assert command[5].endswith("/scripts/ceres_value_mix.py")
    assert "--bt4-weight" not in command
    assert "--execute" in command


def test_plan_mutation_during_finalization_cannot_publish_complete(
    prepared, monkeypatch
):
    p, _, _, _ = prepared
    plan_path = Path(p["state"]).parent / "plan.json"
    plan_path.write_text(json.dumps(p))
    plan_sha = tool.sha(plan_path)
    actual = tool.verify_publication

    def changed(plan, budget):
        result = actual(plan, budget)
        plan_path.write_text("{}")
        return result

    monkeypatch.setattr(tool, "verify_publication", changed)
    with pytest.raises(ValueError, match="frozen plan changed"):
        tool.execute(p, plan_sha, time.time() + 120, plan_path=plan_path)
    result = json.loads((Path(p["state"]) / "status.json").read_text())
    assert result["status"] == "FAILED_OR_STOPPED"


def test_value_profile_uses_value_admission_and_storage_proofs(prepared):
    p, _, admissions, _ = prepared
    producer, epoch = tool.modules(p)
    old = Path(producer.__file__)
    new = old.with_name("ceres_value_mix.py")
    old.rename(new)
    producer.__file__ = str(new)
    producer.SUMMARY = "ceres_value_mix_summary.json"
    producer.wdl = producer.shared
    epoch.CORPORA["B100CeresV25"] = Path(p["corpus"])
    epoch.verify_ceres_value_recipe = lambda *a: admissions.append(a)
    p["profile"] = "B100CeresV25"
    p["producer_sha256"].clear()
    p["producer_sha256"][str(new)] = tool.sha(new)
    p["pins"].pop(str(old))
    p["pins"][str(new)] = tool.sha(new)
    manifest = Path(p["producer_manifest"]["path"])
    data = json.loads(manifest.read_text())
    data["producer_sha256"] = p["producer_sha256"]
    data["summary_name"] = producer.SUMMARY
    manifest.write_text(json.dumps(data))
    p["producer_manifest"]["sha256"] = p["pins"][str(manifest)] = tool.sha(manifest)
    result = tool.execute(p, "f" * 64, time.time() + 120)
    assert result["status"] == "COMPLETE"
    assert result["profile"] == "B100CeresV25"
    assert len(admissions) == 1


def test_output_cap_prevents_complete_receipt(prepared, monkeypatch):
    p, _, _, _ = prepared
    monkeypatch.setattr(tool, "output_bytes", lambda _: tool.OUTPUT_CAP + 1)
    with pytest.raises(ValueError, match="32GiB"):
        tool.execute(p, "f" * 64, time.time() + 120)
    result = json.loads((Path(p["state"]) / "status.json").read_text())
    assert result["status"] == "FAILED_OR_STOPPED"
    assert Path(p["corpus"]).is_dir()


def test_preparation_lock_contention_does_not_start_attempt(prepared):
    import fcntl

    p, _, _, _ = prepared
    with tool.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            tool.execute(p, "f" * 64, time.time() + 120)
    assert not Path(p["state"]).exists()
    assert not Path(p["corpus"]).exists()
