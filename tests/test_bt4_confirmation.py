"""Fresh confirmation contracts; disposable processes only, no teacher/GPU work."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import bt4_confirmation as runner


@pytest.fixture
def manifest(tmp_path, monkeypatch) -> dict[str, Any]:
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"runtime": {"executable": sys.executable}}))
    prereg = tmp_path / "preregistration.md"
    prereg.write_text("Prospective disposable fixture, no actual experiment.")
    qualified = tmp_path / "repo/scratchpad/bt4_joint20/sf_close_run02/completed_s100_readout.json"
    qualified.parent.mkdir(parents=True)
    search = {"shape": "training", "gumbel": {"policy_temp": 1.0, "c_scale": .1}, "tree_reuse": "cold"}
    qualified.write_text(json.dumps({"cells": {"C20T05:100": {"settings": {
        "search_candidate": search, "search_reference": search}}}}))
    monkeypatch.setattr(runner, "QUALIFIED_SEARCH_SHA", runner.sha(qualified))
    return {
        "schema": 1, "repository_root": str(tmp_path / "repo"), "state": str(tmp_path / "state"),
        "training_seed": 17, "arena_seed": 37, "sims": 100, "gpu_budget_seconds": 180,
        "cpu_stage_caps_seconds": {"schedule": 1800, "readout": 120},
        "stage_caps_seconds": {"reference_train": 60, "candidate_train": 60, "arena": 60},
        "reference": {"role": "S0", "corpus": str(tmp_path / "source"), "run": str(tmp_path / "ref-run"),
                      "derive_sha256": runner.SOURCE_SHA, "mix_sha256": None},
        "candidate": {"role": "G20T05", "corpus": str(tmp_path / "mixed"), "run": str(tmp_path / "cand-run"),
                      "derive_sha256": "a" * 64, "mix_sha256": "b" * 64},
        "runtime_manifest": {"path": str(runtime), "sha256": runner.sha(runtime)},
        "preregistration": {"path": str(prereg), "sha256": runner.sha(prereg)},
        "openings": str(tmp_path / "openings.fen"), "schedule_verifier": str(tmp_path / "verifier.py"),
        "launcher_sha256": runner.sha(runner.__file__),
    }


@pytest.fixture
def isolated(manifest, monkeypatch):
    obj = runner.Runner(manifest)
    obj.runtime.mkdir(parents=True)
    (obj.repo / "scratchpad").mkdir(exist_ok=True)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15))
    monkeypatch.setattr(obj, "check_pins", lambda: None)
    monkeypatch.setattr(obj, "check_runtime", lambda: {"fixture": True})
    return obj


@pytest.mark.parametrize(("candidate_role", "reference_role"), [("G20T05", "S0"), ("E0T05", "C20T05"), ("C20T05", "E0T05")])
def test_plan_both_trainers_fresh_seed_and_history_protocol(manifest, tmp_path, candidate_role, reference_role):
    manifest["candidate"]["role"] = candidate_role
    manifest["reference"]["role"] = reference_role
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    completed = subprocess.run([sys.executable, runner.__file__, "--manifest", str(path)],
                               capture_output=True, text=True, check=True, timeout=10)
    plan = json.loads(completed.stdout)
    assert plan["execute"] is False
    assert not Path(manifest["state"]).exists()
    for side in ("reference", "candidate"):
        command = plan["commands"][side + "_train"]
        assert command[command.index("--seed") + 1] == "17"
        assert command[command.index("--out-dir") + 1] == manifest[side]["run"]
        assert command[command.index("--shards") + 1] == manifest[side]["corpus"]
        assert "--resume" not in command
    arena = plan["commands"]["arena"]
    assert arena[arena.index("--openings-fen") + 1] == manifest["openings"]
    assert "--openings" not in arena
    assert "--opening-plies" not in arena
    assert arena[arena.index("--cand-gumbel") + 1] == arena[arena.index("--ref-gumbel") + 1] == "policy_temp=1.0"
    assert arena[arena.index("--seed") + 1] == "37"
    readout = plan["commands"]["readout"]
    assert readout[readout.index("--candidate-role") + 1] == candidate_role
    assert readout[readout.index("--reference-role") + 1] == reference_role
    schedule = plan["commands"]["schedule"]
    assert schedule[schedule.index("--seed") + 1] == "17"
    assert f"candidate={manifest['candidate']['run']}" in schedule
    assert f"reference={manifest['reference']['run']}" in schedule


@pytest.mark.parametrize(("field", "value", "match"), [
    ("training_seed", 0, "seed zero"), ("training_seed", True, "invalid training_seed"),
    ("gpu_budget_seconds", 29, "cannot cover"), ("gpu_budget_seconds", float("nan"), "invalid GPU"),
    ("sims", 200, "simulation"), ("adopt", True, "unsupported"),
    ("cpu_stage_caps_seconds", {"schedule": 1200, "readout": 120}, "register schedule"),
    ("stage_caps_seconds", {"reference_train": 20, "candidate_train": 60, "arena": 60}, "termination grace"),
])
def test_invalid_or_adoption_manifest_rejected(manifest, field, value, match):
    manifest[field] = value
    with pytest.raises(ValueError, match=match):
        runner.Runner(manifest)


def test_overlapping_outputs_and_same_recipe_rejected(manifest):
    manifest["candidate"]["run"] = manifest["reference"]["run"]
    with pytest.raises(ValueError, match="overlapping"):
        runner.Runner(manifest)
    manifest["candidate"]["run"] += "/nested"
    with pytest.raises(ValueError, match="overlapping"):
        runner.Runner(manifest)
    manifest["candidate"]["run"] = "/tmp/different-output"
    manifest["candidate"]["role"] = "S0"
    with pytest.raises(ValueError, match="distinct recipe"):
        runner.Runner(manifest)


@pytest.mark.parametrize("where", ["state", "reference", "candidate"])
def test_existing_or_partial_outputs_never_adopted(isolated, where, monkeypatch):
    path = isolated.state if where == "state" else Path(isolated.m[where]["run"])
    path.mkdir()
    (path / "checkpoint.pt").write_bytes(b"even an apparently completed checkpoint cannot be adopted")
    monkeypatch.setattr(isolated, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="existing output"):
        isolated.execute()
    assert (path / "checkpoint.pt").exists()


def completed_summary(manifest, side) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest[side]["run"])
    path.mkdir()
    (path / "checkpoint.pt").write_bytes(side.encode())
    summary: dict[str, Any] = {"seed": 17, "batch_size": 512, "steps_realized": 36936, "compute_loss_calls": 36936,
               "corpus": {"shard_dirs": [manifest[side]["corpus"]]},
               "checkpoints": [{"role": "last", "path": str(path / "checkpoint.pt"), "sha256": runner.sha(path / "checkpoint.pt")}],
               "train_window_steps": 88, "train_windows": 420,
               "train_window_metrics": [{"grad_nonfinite_skip_rate": 0, "transient_cuda_retry_batches": 0,
                                         "loss": 1.0, "grad_norm_mean": .1} for _ in range(420)],
               "sampling": {"mode": "game_epoch", "complete": True, "seed": 17, "batch_size": 512,
                            "rows_planned": runner.ROWS, "rows_realized": runner.ROWS,
                            "batches_planned": 36936, "batches_realized": 36936, "same_game_repeats_max": 0,
                            "decoded_rows_resident": 0, "shards": runner.SHARDS,
                            "plan_sha256": "new-seed-physical", "realized_sha256": "new-seed-physical"}}
    (path / "summary.json").write_text(json.dumps(summary))
    return path, summary


def test_completion_uses_fresh_seed_and_seed_specific_plan_not_old_step_count(manifest):
    path, summary = completed_summary(manifest, "reference")
    result = runner.training_complete(manifest, "reference")
    assert result["seed"] == 17
    summary["seed"] = 0
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="settings"):
        runner.training_complete(manifest, "reference")
    summary["seed"] = 17
    summary["sampling"]["same_game_repeats_max"] = 1
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete"):
        runner.training_complete(manifest, "reference")


def header(manifest) -> dict[str, Any]:
    settings: dict[str, Any] = {"mode": "matched_sims", "games": 1000, "seed": 37, "openings_kind": "fen", "opening_plies": None,
                "openings": manifest["openings"], "max_plies": 300, "temperature": .1, "gumbel_add_noise": True,
                "sims_candidate": 100, "sims_reference": 100,
                "candidate": manifest["candidate"]["run"] + "/checkpoint.pt", "reference": manifest["reference"]["run"] + "/checkpoint.pt",
                "search_candidate": copy.deepcopy(runner.qualified_search(manifest)),
                "search_reference": copy.deepcopy(runner.qualified_search(manifest))}
    return {"kind": "header", "driver": "arena_standard", "version": 1, "settings": settings,
            "fingerprint": hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]}


@pytest.mark.parametrize("mutation", ["reference_prior", "history", "checkpoint", "fingerprint", "both_search_scales", "tree_reuse"])
def test_header_rejects_wrong_prior_history_checkpoint_and_fingerprint(manifest, mutation):
    good = header(manifest)
    runner.validate_header(manifest, good)
    bad = copy.deepcopy(good)
    if mutation == "reference_prior":
        bad["settings"]["search_reference"]["gumbel"]["policy_temp"] = 1.5
    elif mutation == "history":
        bad["settings"]["openings_kind"] = "book"
    elif mutation == "checkpoint":
        bad["settings"]["candidate"] = bad["settings"]["reference"]
    elif mutation == "both_search_scales":
        for side in ("search_candidate", "search_reference"):
            bad["settings"][side]["gumbel"]["c_scale"] = .2
    elif mutation == "tree_reuse":
        for side in ("search_candidate", "search_reference"):
            bad["settings"][side]["tree_reuse"] = "warm"
    if mutation != "fingerprint":
        bad["fingerprint"] = hashlib.sha256(json.dumps(bad["settings"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    else:
        bad["fingerprint"] = "wrong"
    with pytest.raises(ValueError, match=r"fingerprint|off-protocol|qualified prior"):
        runner.validate_header(manifest, bad)


def test_schedule_rejects_prospective_or_cross_seed_evidence(manifest):
    state = Path(manifest["state"])
    state.mkdir()
    completions: dict[str, dict[str, Any]] = {side: {"summary_sha256": side, "physical_plan_sha256": side, "batches": 36936, "rows": runner.ROWS} for side in ("candidate", "reference")}
    report: dict[str, Any] = {"verifier_sha256": runner.VERIFIER_SHA, "seed": 17, "batch_size": 512, "runtime": {"numpy": "1.26.2"},
              "source_plan": {"plan_sha256": "canonical-fresh", "batches_planned": 36936}, "arms": {}}
    for side in completions:
        report["arms"][side] = {"training_completion_verified": True, "staging": "verified actual", "metadata_matches_source": True,
                                "corpus": manifest[side]["corpus"], "summary_sha256": side, "physical_plan_sha256": side,
                                "canonical_plan_sha256": "canonical-fresh"}
    path = state / "matched_schedule.json"
    path.write_text(json.dumps(report))
    runner.validate_schedule(manifest, completions)
    report["arms"]["candidate"]["training_completion_verified"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed matched"):
        runner.validate_schedule(manifest, completions)
    report["arms"]["candidate"]["training_completion_verified"] = True
    completions["candidate"]["batches"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different realized epoch counts"):
        runner.validate_schedule(manifest, completions)
    report["seed"] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="verifier identity"):
        runner.validate_schedule(manifest, completions)


@pytest.mark.parametrize("failure", [False, True])
def test_gpu_lease_rechecks_pins_and_closes_success_or_failed_accounting(isolated, monkeypatch, failure):
    isolated.state.mkdir()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "")
    events = []
    def pins():
        events.append("checked after lease")
        with (isolated.repo / "scratchpad/gpu0_experiment.lock").open() as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if failure:
            raise ValueError("opening bytes changed during wait")
    monkeypatch.setattr(isolated, "check_pins", pins)
    if failure:
        with pytest.raises(ValueError, match="opening bytes"), isolated.gpu("arena"):
            pytest.fail("changed identity must never launch")
    else:
        with isolated.gpu("arena") as lease:
            assert lease >= 0
    assert events == ["checked after lease"]
    charge = runner.read(isolated.state / "arena.gpu-charge.json")
    assert charge["complete"] is True
    assert charge["seconds"] >= 0
    assert charge["outcome"] == ("failed" if failure else "succeeded")
    with (isolated.repo / "scratchpad/gpu0_experiment.lock").open() as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_gpu_budget_and_disk_fail_before_lease(isolated, monkeypatch):
    isolated.state.mkdir()
    isolated.used = 121
    with pytest.raises(ValueError, match="remaining GPU"), isolated.gpu("arena"):
        pytest.fail("budget failure must not enter stage")
    assert not list(isolated.state.glob("*.gpu-charge.json"))
    isolated.used = 0
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=runner.RESERVE - 1))
    with pytest.raises(ValueError, match="reserve"), isolated.gpu("arena"):
        pytest.fail("disk failure must not enter stage")


def test_owned_child_failure_is_accounted_and_shared_lock_inherited(isolated, monkeypatch):
    isolated.state.mkdir()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "")
    code = ("import fcntl,sys; f=open(sys.argv[1]); "
            "\ntry: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)"
            "\nexcept BlockingIOError: print('inherited lease held',flush=True); sys.exit(7)"
            "\nraise RuntimeError('lease was released')")
    with pytest.raises(ValueError, match="exited 7"), isolated.gpu("reference_train") as fd:
        isolated.run("reference_train", [sys.executable, "-c", code, str(isolated.repo / "scratchpad/gpu0_experiment.lock")], fd)
    assert "inherited lease held" in (isolated.state / "reference_train.stdout").read_text()
    assert runner.read(isolated.state / "reference_train.gpu-charge.json")["outcome"] == "failed"
    assert isolated.child_reaped


def test_stop_terminates_only_owned_stage_and_preserves_partial_output(isolated):
    isolated.state.mkdir()
    ready = isolated.state / "partial-output"
    code = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('preserve'); time.sleep(30)"
    def stop_when_ready():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(.01)
        isolated.stop.set()
    watcher = threading.Thread(target=stop_when_ready)
    watcher.start()
    try:
        with pytest.raises(ValueError, match="stop requested"):
            isolated.run("schedule", [sys.executable, "-c", code, str(ready)])
    finally:
        watcher.join(timeout=12)
    assert ready.read_text() == "preserve"
    pid = runner.read(isolated.state / "schedule.process.json")["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_execute_releases_gpu_for_schedule_and_requires_proof_before_arena(isolated, monkeypatch):
    events = []
    held = False
    @contextmanager
    def lease(stage):
        nonlocal held
        held = True
        events.append(stage)
        try:
            yield 123
        finally:
            held = False
    def run(stage, _command, _gpu_fd=None, output=None):
        del output
        if stage == "schedule":
            assert not held
            events.append("schedule without GPU")
        else:
            assert held
    monkeypatch.setattr(isolated, "gpu", lease)
    monkeypatch.setattr(isolated, "run", run)
    monkeypatch.setattr(runner, "training_complete", lambda _m, side: {"fresh": side})
    def reject(_m, _completed):
        raise ValueError("no matched schedule proof")
    monkeypatch.setattr(runner, "validate_schedule", reject)
    with pytest.raises(ValueError, match="no matched schedule"):
        isolated.execute()
    assert events == ["reference_train", "candidate_train", "schedule without GPU"]
    assert runner.read(isolated.state / "failed.json")["partial_outputs_preserved"] is True


def test_short_remaining_gpu_time_rejects_launch_and_records_failed_charge(isolated, monkeypatch):
    isolated.state.mkdir()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "")
    isolated.m["stage_caps_seconds"]["candidate_train"] = .5
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("insufficient grace must not launch"))
    with pytest.raises(ValueError, match="stage time"), isolated.gpu("candidate_train") as fd:
        isolated.run("candidate_train", [sys.executable, "-c", "import time; time.sleep(30)"], fd)
    charge = runner.read(isolated.state / "candidate_train.gpu-charge.json")
    assert charge["complete"] is True
    assert charge["outcome"] == "failed"
    assert isolated.child_reaped


def test_stop_while_another_owner_holds_lease_never_launches(isolated, monkeypatch):
    isolated.state.mkdir()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: pytest.fail("other owner holds lease"))
    with (isolated.repo / "scratchpad/gpu0_experiment.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timer = threading.Timer(.05, isolated.stop.set)
        timer.start()
        try:
            with pytest.raises(ValueError, match="stop requested"), isolated.gpu("arena"):
                pytest.fail("lease must not be acquired")
        finally:
            timer.join(timeout=1)
    assert not list(isolated.state.glob("*.gpu-charge.json"))


def test_real_pin_check_rejects_changed_opening_before_runtime_probe(manifest, monkeypatch):
    Path(manifest["openings"]).write_bytes(b"immutable fixture history")
    monkeypatch.setattr(runner, "OPENINGS_SHA", runner.sha(manifest["openings"]))
    # Real hash validation reaches the opening before any expensive runtime work.
    Path(manifest["openings"]).write_bytes(b"changed history with the same filename")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: pytest.fail("must fail before runtime probe"))
    with pytest.raises(ValueError, match=r"identity changed.*openings"):
        runner.Runner(manifest).check_pins()


@pytest.mark.parametrize("temperature", [.5, None, 2.0])
def test_legacy_e0_binding_rejects_sharpened_or_unknown_temperature(tmp_path, monkeypatch, temperature):
    # Actual qualified E0 shape has no temperature field; the banked T.5 arm
    # retains the same kind/alpha/algorithm but explicitly changes temperature.
    raw = {"algorithm": "stored-top-set-only-v1", "alpha": 1.0, "kind": "top-max-ties", "schema": 1,
           "rows": runner.ROWS, "shards": runner.SHARDS}
    mix_path = tmp_path / "bt4_policy_mix_summary.json"
    derived = tmp_path / "derive_targets_summary.json"
    mix_path.write_text(json.dumps(raw))
    derived.write_text(json.dumps({"policy_target_postprocess": raw}))
    qualified = runner.sha(mix_path)
    monkeypatch.setitem(runner.QUALIFIED_MIX_SHA, "E0", qualified)
    runner.check_qualified_mix({"role": "E0", "mix_sha256": qualified}, tmp_path)
    changed = {**raw, "bt4_temperature": temperature}
    mix_path.write_text(json.dumps(changed))
    derived.write_text(json.dumps({"policy_target_postprocess": changed}))
    with pytest.raises(ValueError, match="qualified published recipe"):
        runner.check_qualified_mix({"role": "E0", "mix_sha256": runner.sha(mix_path)}, tmp_path)
    with pytest.raises(ValueError, match="identity changed"):
        runner.check_qualified_mix({"role": "E0", "mix_sha256": qualified}, tmp_path)


@pytest.mark.parametrize("field", ["algorithm", "source_derive_summary_sha256"])
def test_close_role_binds_algorithm_and_source_not_only_knobs(tmp_path, monkeypatch, field):
    mix = {"kind": "sf-cp-window", "alpha": 1.0, "bt4_temperature": .5, "sf_cp_window": 20.0, "sf_rank_cap": 3,
           "algorithm": "stored-top-ties-union-sf-d9-cp-window-v1", "source_derive_summary_sha256": runner.SOURCE_SHA}
    path = tmp_path / "bt4_policy_mix_summary.json"
    path.write_text(json.dumps(mix))
    (tmp_path / "derive_targets_summary.json").write_text(json.dumps({"policy_target_postprocess": mix}))
    qualified = runner.sha(path)
    monkeypatch.setitem(runner.QUALIFIED_MIX_SHA, "C20T05", qualified)
    runner.check_qualified_mix({"role": "C20T05", "mix_sha256": qualified}, tmp_path)
    mix[field] = "different"
    path.write_text(json.dumps(mix))
    with pytest.raises(ValueError, match="qualified published recipe"):
        runner.check_qualified_mix({"role": "C20T05", "mix_sha256": runner.sha(path)}, tmp_path)


def test_sharpened_exact_tie_role_cannot_be_relabeled_raw(tmp_path, monkeypatch):
    mix = {"kind": "top-max-ties", "algorithm": "stored-top-set-only-v1", "alpha": 1.0,
           "bt4_temperature": .5, "source_derive_summary_sha256": runner.SOURCE_SHA}
    path = tmp_path / "bt4_policy_mix_summary.json"
    path.write_text(json.dumps(mix))
    (tmp_path / "derive_targets_summary.json").write_text(json.dumps({"policy_target_postprocess": mix}))
    qualified = runner.sha(path)
    monkeypatch.setitem(runner.QUALIFIED_MIX_SHA, "E0T05", qualified)
    runner.check_qualified_mix({"role": "E0T05", "mix_sha256": qualified}, tmp_path)
    with pytest.raises(ValueError, match="qualified published recipe"):
        runner.check_qualified_mix({"role": "E0", "mix_sha256": qualified}, tmp_path)


def test_environment_overrides_inherited_live_configuration(isolated, monkeypatch):
    monkeypatch.setenv("CHESS_ANTI_ENGINE_LIVE_CONFIG", "/tmp/other-experiment.yaml")
    assert isolated.environment(False)["CHESS_ANTI_ENGINE_LIVE_CONFIG"] == str(isolated.runtime / "configs/pbt2_small.yaml")


def test_cpu_stage_timeout_preserves_partial_output(isolated, monkeypatch):
    isolated.state.mkdir()
    isolated.m["cpu_stage_caps_seconds"]["schedule"] = .3
    monkeypatch.setattr(runner, "KILL_GRACE_SECONDS", .1)
    output = isolated.state / "partial"
    code = "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text('partial'); time.sleep(60)"
    with pytest.raises(ValueError, match="exited"):
        isolated.run("schedule", [sys.executable, "-c", code, str(output)])
    assert output.read_text() == "partial"
    receipt = runner.read(isolated.state / "schedule.process.json")
    assert receipt["supervisor_command"][0] == "/usr/bin/timeout"
    assert receipt["cap_seconds_at_launch"] == .3
    assert isolated.child_reaped


def test_supervisor_retains_lease_and_kills_stage_after_coordinator_sigkill(isolated, tmp_path):
    isolated.state.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(isolated.m))
    coordinator_path = tmp_path / "coordinator.py"
    coordinator_path.write_text("""import fcntl,json,os,sys,time
from scripts import bt4_confirmation as runner
runner.KILL_GRACE_SECONDS = .2
r = runner.Runner(json.load(open(sys.argv[1])))
r.guard = lambda extra_disk=0: None
r.environment = lambda gpu: {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'CHESS_ANTI_ENGINE_LIVE_CONFIG': str(r.runtime / 'configs/pbt2_small.yaml')}
r.deadline = time.monotonic() + 1.2
with (r.repo / 'scratchpad/gpu0_experiment.lock').open('a') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
    r.run('reference_train', [sys.executable, '-c', code], lease.fileno())
""")
    process = subprocess.Popen([sys.executable, str(coordinator_path), str(manifest_path)],
                               env={**os.environ, "PYTHONPATH": str(Path(runner.__file__).resolve().parents[1]),
                                    "CUDA_VISIBLE_DEVICES": ""}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    supervisor_pid = None
    try:
        ready = isolated.state / "reference_train.process.json"
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), "coordinator did not start its owned supervisor"
        supervisor_pid = runner.read(ready)["pid"]
        with (isolated.repo / "scratchpad/gpu0_experiment.lock").open() as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process.kill()
            process.wait(timeout=3)
            # The surviving timeout and TERM-resistant child still retain the lease.
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + 6
            while True:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "orphan stage outlived its independent deadline"
                    time.sleep(.02)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        if supervisor_pid is not None:
            try:
                os.killpg(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(("field", "value"), [
    ("grad_nonfinite_skip_rate", .01), ("transient_cuda_retry_batches", 1),
    ("loss", float("nan")), ("grad_norm_mean", float("inf")),
])
def test_completion_rejects_failed_windows(manifest, field, value):
    path, summary = completed_summary(manifest, "candidate")
    summary["train_window_metrics"][-1][field] = value
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="training windows"):
        runner.training_complete(manifest, "candidate")


@pytest.mark.parametrize("mutation", ["cadence", "window_count", "checkpoint_path"])
def test_completion_rejects_window_cadence_count_and_checkpoint_path(manifest, mutation):
    path, summary = completed_summary(manifest, "candidate")
    if mutation == "cadence":
        summary["train_window_steps"] = 176
    elif mutation == "window_count":
        summary["train_window_metrics"].pop()
        summary["train_windows"] -= 1
    else:
        summary["checkpoints"][0]["path"] = str(path / "other.pt")
    (path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=r"training windows|checkpoint differs"):
        runner.training_complete(manifest, "candidate")


def test_runtime_probe_accepts_qualified_manifest_with_numpy(manifest, monkeypatch):
    expected = {"python": "3.10.12", "executable": "/qualified/python", "torch": "2.11.0+cu128",
                "cuda": "12.8", "numpy": "1.26.2", "native_extensions": {"test.module": "/qualified/module.so"},
                "native_extension_sha256": {"/qualified/module.so": "a" * 64}}
    Path(manifest["runtime_manifest"]["path"]).write_text(json.dumps({"runtime": expected}))
    actual = {k: v for k, v in expected.items() if k != "native_extension_sha256"}
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: json.dumps(actual))
    obj = runner.Runner(manifest)
    assert obj.check_runtime() == actual
    actual["numpy"] = "2.0.0"
    with pytest.raises(ValueError, match="NumPy"):
        obj.check_runtime()
