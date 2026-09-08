"""Frozen common-input scope, real child resource propagation and owned cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from scripts import common_input_batch as batch


@pytest.fixture(autouse=True)
def metadata_cpu_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = set(os.sched_getaffinity(0))
    virtual = actual | set(range(4)) if len(actual) < 4 else actual
    monkeypatch.setattr(batch, "available_cpus", lambda: virtual)


def pin(path: Path, data: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return {"path": str(path), "sha256": batch.sha(path)}


def fixture(tmp_path: Path, concurrency: int = 2) -> dict[str, Any]:
    cpus = sorted(batch.available_cpus())
    # Production validates against the actual allowed cpuset; tests need four
    # available CPU ids for the two-lane admission case, without consuming them.
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / "checkout").mkdir()
    plan: dict[str, Any] = {
        "schema": 1,
        "state": str(state),
        "checkout": str(tmp_path / "checkout"),
        "commit": "a" * 40,
        "python": sys.executable,
        "pins": {},
        "runtime_qualification": pin(tmp_path / "runtime.json", {}),
        "preregistration": pin(tmp_path / "prereg.json", {}),
        "max_concurrent_sources": concurrency,
        "derive_options": {
            "scheme": "uniform-d9",
            "policy_observation": "phase0",
            "value_observation": "latest-phase",
            "value_scheme": "search",
            "temp": 0.0005,
            "floor": 0,
            "workers": 2,
            "row_provenance": True,
            "seed": 7,
            "rows_per_shard": 2,
        },
        "limits": {
            "wall_seconds_including_kill": 60,
            "numeric_threads": 2,
            "nice": 19,
            "ionice_class": 3,
            "CUDA_VISIBLE_DEVICES": "",
            "new_output_cache_bytes": 2**24,
            "minimum_free_bytes": 1,
            "adapter_index_cache_bytes": batch.raw_identity_cache_bytes(10, 2),
            "rank_index_cache_bytes": 8192,
        },
        "sources": [],
    }
    for i in range(2):
        name = f"source{i}"
        raw = tmp_path / name
        raw.mkdir()
        side = tmp_path / (name + "_side")
        side.mkdir()
        manifest = pin(
            raw / "manifest.json", {"identity": name, "config_sha256": "b" * 64}
        )
        entries = []
        metadata = []
        for k in (2, 4):
            path = raw / f"w00-{k:05d}.jsonl.zst"
            path.write_bytes(b"closed raw fixture")
            sidepath = side / f"w00-{k:05d}.bt4.zarr"
            sidepath.mkdir()
            attrs = pin(sidepath / ".zattrs", {"onnx_path": "fixture.onnx"})
            entries.append(
                {"source_shard": path.name, "rows": 5, "source_sha256": batch.sha(path)}
            )
            st = path.stat()
            metadata.append(
                {
                    "source_path": str(path),
                    "device": st.st_dev,
                    "inode": st.st_ino,
                    "bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "ctime_ns": st.st_ctime_ns,
                    "sidecar_path": str(sidepath),
                    "sidecar_attrs_snapshot": attrs,
                }
            )
        selection = pin(
            tmp_path / (name + "_selection.json"),
            {
                "schema": 1,
                "source_dir": str(raw),
                "source_config_sha256": "b" * 64,
                "source_manifest_sha256": manifest["sha256"],
                "shards": list(reversed(entries)),
            },
        )
        receipts = pin(
            tmp_path / (name + "_receipts.jsonl"),
            {
                "sidecar": "w00-00002.bt4.zarr",
                "onnx_sha256": "c" * 64,
                "policy_output": "policy",
                "providers": ["CPUExecutionProvider"],
                "remap_provenance": {},
            },
        )
        plan["sources"].append(
            {
                "source_id": name,
                "source_dir": str(raw),
                "sidecar_dir": str(side),
                "source_manifest": manifest,
                "selection": selection,
                "closed_bt4_receipts": receipts,
                "source_metadata": pin(tmp_path / (name + "_metadata.json"), metadata),
                "physical_rows": 10,
                "support_drop_ceiling": 1,
                "missing_result_fraction_ceiling": 0.2,
                "missing_result_count_ceiling": 2,
                "cpu_affinity": cpus[i * 2 : i * 2 + 2],
                "derived_output": str(state / name / "derived"),
                "adapted_output": str(state / name / "bt4"),
                "rank_output": str(state / name / "rank"),
            }
        )
    return plan


def proof(source: dict[str, Any]) -> dict[str, Any]:
    p = batch.read(source["selection"]["path"])
    p["shards"].sort(key=lambda x: x["source_shard"])
    return {
        **p,
        "path": source["selection"]["path"],
        "sha256": source["selection"]["sha256"],
        "order": "original corpus shard order; limit applies after selection",
    }


def test_manifest_nonprefix_scope_and_no_result_caps(tmp_path: Path) -> None:
    plan = fixture(tmp_path)
    batch.validate_manifest(plan)
    source = plan["sources"][0]
    summary: dict[str, Any] = {
        "source_selection": proof(source),
        "max_policy_support_misses": 1,
        "policy_support_misses_file": None,
        "realized": {
            "rows_dropped_policy_support": 0,
            "policy_support_exclusions": [],
            "rows_dropped_no_result": 2,
            "rows_read": 10,
            "rows_written": 8,
            "rows_dropped_envelope": 0,
        },
    }
    assert batch.actual_exclusions(source, summary) == set()
    rank = {
        "source_selection": proof(source),
        "raw_rows_read": 10,
        "rows_dropped_no_result": 2,
        "rows": 8,
        "row_provenance": {
            "join": "source-qualified-physical-row-and-full-history-keys-v1",
            "rows_dropped_policy_support": 0,
        },
    }
    covered = {
        e["source_shard"]: bytearray([1, 1, 1, 1, 0]) for e in batch.selected(source)
    }
    assert batch.verify_complement(source, summary, rank, covered) == 2
    covered[next(iter(covered))][0] = 0
    with pytest.raises(ValueError, match="complement"):
        batch.verify_complement(source, summary, rank, covered)
    summary["realized"].update(rows_dropped_no_result=3, rows_written=7)
    with pytest.raises(ValueError, match="missing-result cap"):
        batch.actual_exclusions(source, summary)


def test_zero_support_cap_preserves_default_summary_contract(tmp_path: Path) -> None:
    source = fixture(tmp_path)["sources"][0]
    source["support_drop_ceiling"] = 0
    summary = {
        "source_selection": proof(source),
        "realized": {
            "rows_read": 10,
            "rows_written": 8,
            "rows_dropped_no_result": 2,
            "rows_dropped_envelope": 0,
        },
    }
    assert batch.actual_exclusions(source, summary) == set()
    rank = {
        "source_selection": proof(source),
        "raw_rows_read": 10,
        "rows_dropped_no_result": 2,
        "rows": 8,
        "row_provenance": {
            "join": "source-qualified-physical-row-and-full-history-keys-v1"
        },
    }
    covered = {
        entry["source_shard"]: bytearray([1, 1, 1, 1, 0])
        for entry in batch.selected(source)
    }
    assert batch.verify_complement(source, summary, rank, covered) == 2


@pytest.mark.parametrize(
    "change", ["overlap", "rows", "semantics", "state_input", "source_mutation"]
)
def test_admission_rejects_invalid_frozen_contract(tmp_path: Path, change: str) -> None:
    plan = fixture(tmp_path)
    source = plan["sources"][0]
    if change == "overlap":
        plan["sources"][1]["cpu_affinity"] = source["cpu_affinity"]
    elif change == "rows":
        source["physical_rows"] = 11
    elif change == "semantics":
        plan["derive_options"]["value_observation"] = "phase0"
    elif change == "state_input":
        source["derived_output"] = source["source_dir"]
    else:
        path = Path(source["source_dir"]) / batch.selected(source)[0]["source_shard"]
        path.write_bytes(b"changed")
    with pytest.raises(
        ValueError,
        match=r"affinities overlap|selected row count|unsupported common|lane outputs|storage changed",
    ):
        batch.validate_manifest(plan)


@pytest.mark.parametrize("overlap", [False, True])
def test_lane_builds_real_selected_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlap: bool
) -> None:
    plan = fixture(tmp_path)
    plan["overlap_adapt_rank"] = overlap
    source = plan["sources"][0]
    calls = []
    monkeypatch.setattr(batch, "verify", lambda _p: None)

    def run_stage(name, argv, _plan, _source, _guard):
        calls.append((name, argv))
        if name == "derive":
            p = Path(source["derived_output"])
            p.mkdir()
            pin(
                p / "derive_targets_summary.json",
                {"realized": {"rows_written": 8}, "shards": [{}, {}, {}, {}]},
            )
        elif name == "qualify":
            pin(
                Path(source["derived_output"]).parent
                / "common_input_qualification.json",
                {"status": "complete"},
            )

    monkeypatch.setattr(batch, "stage", run_stage)
    monkeypatch.setattr(
        batch,
        "adapt_and_rank",
        lambda a, r, p, s, g: (
            run_stage("adapt", a, p, s, g),
            run_stage("rank", r, p, s, g),
        ),
    )
    batch.lane(plan, tmp_path / "manifest.json", "d" * 64, source, time.time() + 60)
    assert [n for n, _ in calls] == ["derive", "snapshot", "adapt", "rank", "qualify"]
    derive = dict(zip(calls[0][1][2::2], calls[0][1][3::2]))
    assert derive["--limit"] == "10"
    assert derive["--source-shards"] == source["selection"]["path"]
    assert derive["--workers"] == "2"
    assert derive["--seed"] == "7"
    assert derive["--rows-per-shard"] == "2"
    assert derive["--value-observation"] == "latest-phase"
    rank = calls[3][1]
    assert rank[rank.index("--expected-rows") + 1] == "8"
    assert rank[rank.index("--source-shards") + 1] == source["selection"]["path"]
    assert rank[rank.index("--max-provenance-cache-bytes") + 1] == "8192"
    adapt = calls[2][1]
    assert adapt[adapt.index("--max-index-bytes") + 1] == str(plan["limits"]["adapter_index_cache_bytes"])
    with pytest.raises(ValueError, match="unregistered tool"):
        batch.command(plan, "lc0_control_train.py")


def test_aggregate_usage_tolerates_rename_only_during_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "gone").write_bytes(b"x")
    real = Path.lstat

    def lstat(path):
        if path.name == "gone":
            raise FileNotFoundError(path)
        return real(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    assert batch.usage(tmp_path) == 0
    with pytest.raises(FileNotFoundError):
        batch.usage(tmp_path, stable=True)
    (tmp_path / "alias").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        batch.usage(tmp_path)


def test_guard_samples_usage_but_checks_stop_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fixture(tmp_path)
    counts = []
    monkeypatch.setattr(batch, "usage", lambda *_args, **_kwargs: counts.append(1) or 1)
    guard = batch.Guard(plan, time.time() + 60)
    guard.check()
    guard.check()
    assert len(counts) == 1
    (Path(plan["state"]) / "STOP").touch()
    with pytest.raises(ValueError, match="STOP"):
        guard.check()


@pytest.mark.skipif(
    len(os.sched_getaffinity(0)) < 4,
    reason="actual two disjoint CPU-pair propagation requires four available CPUs",
)
@pytest.mark.parametrize("failure", [False, True])
def test_actual_two_lane_children_and_late_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    plan = fixture(tmp_path)
    monkeypatch.setattr(batch, "verify", lambda _p: None)
    original = batch.lane_command
    child_script = tmp_path / "fixture_child.py"
    child_script.write_text("""import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
(p/'environment.json').write_text(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'threads':os.environ['OMP_NUM_THREADS'],'pythonpath':os.environ['PYTHONPATH']}))
if mode=='sleeper':
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 (p/'grandchild.pid').write_text(str(child.pid));time.sleep(60)
if mode=='fail':
 time.sleep(.2);raise SystemExit(7)
q=p/'common_input_qualification.json';q.write_text('{}')
(p/'lane_complete.json').write_text(json.dumps({'status':'complete','qualification_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
""")

    def lane_command(p, m, d, s, deadline):
        argv = original(p, m, d, s, deadline)
        cut = argv.index(p["python"])
        mode = (
            ("fail" if s["source_id"] == "source0" else "sleeper")
            if failure
            else "success"
        )
        return [
            *argv[:cut],
            sys.executable,
            str(child_script),
            str(Path(p["state"]) / s["source_id"]),
            mode,
        ]

    monkeypatch.setattr(batch, "lane_command", lane_command)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    try:
        if failure:
            with pytest.raises(ValueError, match="exit 7"):
                batch.execute(plan, tmp_path / "manifest.json", "f" * 64)
        else:
            batch.execute(plan, tmp_path / "manifest.json", "f" * 64)
        assert unrelated.poll() is None
        for source in plan["sources"]:
            p = Path(plan["state"]) / source["source_id"]
            env = batch.read(p / "environment.json")
            assert env["cpus"] == source["cpu_affinity"]
            assert env["nice"] == 19
            assert env["gpu"] == ""
            assert env["threads"] == "2"
            assert env["pythonpath"].endswith(plan["checkout"])
            if (p / "grandchild.pid").exists():
                pid = int((p / "grandchild.pid").read_text())
                status = Path(f"/proc/{pid}/stat")
                # SIGKILL delivery and orphan exit are asynchronous. Waiting
                # for the lane leader does not wait for this grandchild.
                deadline = time.monotonic() + 2
                while True:
                    try:
                        state = status.read_text().rsplit(")", 1)[1].split()[0]
                    except FileNotFoundError:
                        break
                    if state == "Z":
                        break
                    assert time.monotonic() < deadline, (
                        f"owned grandchild {pid} remained in state {state}"
                    )
                    time.sleep(0.01)
        assert (
            Path(plan["state"]) / ("failed.json" if failure else "completed.json")
        ).exists()
        with pytest.raises(ValueError, match="prior attempt"):
            batch.fresh_outputs(plan)
    finally:
        batch.stop_owned_group(unrelated, grace=0.1)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_config",
        "wrong_config",
        "extra_key",
        "extra_entry_key",
        "empty_shards",
        "boolean_rows",
    ],
)
def test_preflight_refuses_consumer_header_mismatch(
    tmp_path: Path, corruption: str
) -> None:
    plan = fixture(tmp_path)
    source = plan["sources"][0]
    path = Path(source["selection"]["path"])
    selection = batch.read(path)
    if corruption == "missing_config":
        del selection["source_config_sha256"]
    elif corruption == "wrong_config":
        selection["source_config_sha256"] = "0" * 64
    elif corruption == "extra_key":
        selection["unexpected"] = True
    elif corruption == "extra_entry_key":
        selection["shards"][0]["unexpected"] = True
    elif corruption == "empty_shards":
        selection["shards"] = []
    else:
        selection["shards"][0]["rows"] = True
    source["selection"] = pin(path, selection)
    with pytest.raises(ValueError, match=r"source-shards|row claim"):
        batch.validate_manifest(plan)
    assert not (Path(plan["state"]) / "started.json").exists()


@pytest.mark.parametrize("value", [1, "false", None])
def test_overlap_flag_requires_boolean(tmp_path: Path, value: Any) -> None:
    plan = fixture(tmp_path)
    plan["overlap_adapt_rank"] = value
    with pytest.raises(ValueError, match="overlap_adapt_rank must be boolean"):
        batch.validate_manifest(plan)


@pytest.mark.parametrize("mode", ["serial", "overlap", "failure", "stop"])
def test_real_post_snapshot_pair_barrier_and_lane_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Only producer payload work is synthetic; lane/execute/time/guards are real."""
    plan = fixture(tmp_path, concurrency=1)
    plan["sources"] = plan["sources"][:1]
    plan["sources"][0]["cpu_affinity"] = sorted(os.sched_getaffinity(0))[:2]
    plan["overlap_adapt_rank"] = mode != "serial"
    manifest = tmp_path / "plan.json"
    pin(manifest, plan)
    fake = tmp_path / "producer.py"
    fake.write_text("""import json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(sys.argv[1]);name=sys.argv[2];mode=sys.argv[3]
(p/(name+'.begin')).write_text(str(time.time()))
(p/(name+'.env')).write_text(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':os.environ['OMP_NUM_THREADS'],'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'pgid':os.getpgrp()}))
if name=='derive':
 (p/'derived').mkdir()
 (p/'derived'/'derive_targets_summary.json').write_text(json.dumps({'realized':{'rows_written':8},'shards':[{}]}))
elif name=='snapshot':
 assert (p/'derive.done').exists()
elif name in ('adapt','rank'):
 assert (p/'snapshot.done').exists()
 if mode!='serial':
  peer='rank' if name=='adapt' else 'adapt'
  deadline=time.time()+10
  while not (p/(peer+'.begin')).exists():
   assert time.time()<deadline,'second independent stage never started'
   time.sleep(.01)
 if name=='rank' and mode in ('failure','stop'):
  child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
  (p/'grandchild.pid').write_text(str(child.pid))
  if mode=='stop':(p.parent/'STOP').touch()
  time.sleep(60)
 if name=='adapt' and mode=='failure':
  deadline=time.time()+10
  while not (p/'grandchild.pid').exists():
   assert time.time()<deadline
   time.sleep(.01)
  raise SystemExit(7)
 time.sleep(.15 if name=='adapt' else .3)
elif name=='qualify':
 assert (p/'adapt.done').exists() and (p/'rank.done').exists()
 assert (p/'adapt.completed.json').exists() and (p/'rank.completed.json').exists()
 (p/'common_input_qualification.json').write_text('{}')
(p/(name+'.done')).write_text(str(time.time()))
""")
    lane_driver = tmp_path / "lane_driver.py"
    lane_driver.write_text(f"""import json,pathlib,signal,sys
sys.path.insert(0,{str(Path(batch.__file__).resolve().parents[1])!r})
from scripts import common_input_batch as b
plan=json.loads(pathlib.Path(sys.argv[1]).read_text());source=plan['sources'][0]
def stop(*_args):raise SystemExit('termination requested')
signal.signal(signal.SIGTERM,stop)
b.verify=lambda _p:None
p=str(pathlib.Path(plan['state'])/source['source_id'])
names={{'derive_corpus_targets.py':'derive','adapt_raw_bt4_sidecars.py':'adapt','sf_d9_rank_sidecar.py':'rank'}}
b.command=lambda _p,script,*args:[sys.executable,{str(fake)!r},p,names[script],{mode!r}]
b.internal_command=lambda _p,_m,_d,action,_s:[sys.executable,{str(fake)!r},p,action,{mode!r}]
b.lane(plan,pathlib.Path(sys.argv[1]),'f'*64,source,float(sys.argv[2]))
""")
    monkeypatch.setattr(batch, "verify", lambda _p: None)
    original = batch.lane_command

    def actual_lane(p, m, d, source, deadline):
        argv = original(p, m, d, source, deadline)
        return [
            *argv[: argv.index(p["python"])],
            sys.executable,
            str(lane_driver),
            str(manifest),
            str(deadline),
        ]

    monkeypatch.setattr(batch, "lane_command", actual_lane)
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    state = Path(plan["state"])
    parent = state / plan["sources"][0]["source_id"]
    try:
        if mode in ("failure", "stop"):
            with pytest.raises(ValueError, match=r"exit|STOP"):
                batch.execute(plan, manifest, batch.sha(manifest))
            assert (state / "failed.json").exists()
            assert not (parent / "qualify.begin").exists()
            pid = int((parent / "grandchild.pid").read_text())
            deadline = time.monotonic() + 2
            while True:
                try:
                    child_state = (
                        Path(f"/proc/{pid}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[0]
                    )
                except FileNotFoundError:
                    break
                if child_state == "Z":
                    break
                assert time.monotonic() < deadline, (
                    "owned grandchild survived lane failure"
                )
                time.sleep(0.01)
        else:
            batch.execute(plan, manifest, batch.sha(manifest))
            assert (state / "completed.json").exists()
            begin = {
                n: float((parent / (n + ".begin")).read_text()) for n in batch.STAGES
            }
            done = {
                n: float((parent / (n + ".done")).read_text()) for n in batch.STAGES
            }
            assert (
                done["derive"]
                <= begin["snapshot"]
                <= done["snapshot"]
                <= begin["adapt"]
            )
            if mode == "overlap":
                assert max(begin["adapt"], begin["rank"]) < min(
                    done["adapt"], done["rank"]
                )
            else:
                assert done["adapt"] < begin["rank"]
            assert max(done["adapt"], done["rank"]) < begin["qualify"]
            assert batch.read(parent / "lane_complete.json")["overlap_adapt_rank"] == (
                mode != "serial"
            )
            lane_pid = batch.read(parent / "lane.started.json")["pid"]
            for name in batch.STAGES:
                env = batch.read(parent / (name + ".env"))
                assert env == {
                    "cpus": plan["sources"][0]["cpu_affinity"],
                    "nice": 19,
                    "threads": "2",
                    "gpu": "",
                    "pgid": lane_pid,
                }
                assert (
                    batch.read(parent / (name + ".completed.json"))["resources"][
                        "exit_code"
                    ]
                    == 0
                )
        assert sentinel.poll() is None
    finally:
        batch.stop_owned_group(sentinel, grace=0.1)


@pytest.mark.parametrize(("field", "sizing"), [
    ("rank_index_cache_bytes", lambda: batch.rank_cache_bytes(10, 2, 3)),
    ("adapter_index_cache_bytes", lambda: batch.raw_identity_cache_bytes(10, 2)),
])
def test_cache_reservation_boundary_refuses_before_verify_or_spawn(tmp_path, monkeypatch, field, sizing):
    plan = fixture(tmp_path)
    plan["limits"][field] = sizing()
    batch.validate_manifest(plan)
    plan["limits"][field] -= 1

    def unexpected(*_args, **_kwargs):
        pytest.fail("insufficient cache must fail before verification or process launch")

    monkeypatch.setattr(batch, "verify", unexpected)
    monkeypatch.setattr(batch.subprocess, "Popen", unexpected)
    with pytest.raises(ValueError, match=field + " needs at least"):
        batch.execute(plan, tmp_path / "manifest.json", "a" * 64)
    assert not (Path(plan["state"]) / "started.json").exists()
    assert all(not Path(source["derived_output"]).exists() for source in plan["sources"])
