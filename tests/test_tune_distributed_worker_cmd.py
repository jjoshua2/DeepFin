import json
from pathlib import Path

import numpy as np

from chess_anti_engine.tune.process_cleanup import _list_matching_pids
from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.tune.harness import (
    _extract_saved_trial_config_keys,
    _patch_experiment_state_for_resume,
    _resolve_local_override_root as _resolve_harness_override_root,
)
from chess_anti_engine.tune.distributed_runtime import (
    _build_distributed_worker_cmd,
    _launch_inference_broker,
)
from chess_anti_engine.tune.replay_exchange import (
    _refresh_replay_shards_on_exploit,
    _trial_replay_shard_dir,
)


def test_build_distributed_worker_cmd_pins_trial_id() -> None:
    cmd = _build_distributed_worker_cmd(
        config={
            "distributed_server_url": "http://127.0.0.1:45453",
            "distributed_worker_username": "worker",
            "distributed_worker_password_file": "/tmp/pw",
            "stockfish_path": "/tmp/stockfish",
            "distributed_server_root": "/tmp/server",
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": True,
            "distributed_worker_sf_workers": 1,
            "distributed_worker_poll_seconds": 1.0,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_00"),
        trial_id="trial_00000",
        worker_index=0,
        worker_log=Path("/tmp/trial/worker_00/worker.log"),
    )

    assert "--trial-id" in cmd
    assert cmd[cmd.index("--trial-id") + 1] == "trial_00000"


def test_build_distributed_worker_cmd_adds_inference_slot() -> None:
    cmd = _build_distributed_worker_cmd(
        config={
            "distributed_server_url": "http://127.0.0.1:45453",
            "distributed_worker_username": "worker",
            "distributed_worker_password_file": "/tmp/pw",
            "stockfish_path": "/tmp/stockfish",
            "distributed_server_root": "/tmp/server",
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": False,
            "distributed_worker_sf_workers": 1,
            "distributed_worker_poll_seconds": 1.0,
            "distributed_inference_broker_enabled": True,
            "distributed_inference_max_batch_per_slot": 256,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_00"),
        trial_id="trial_00000",
        worker_index=0,
        worker_log=Path("/tmp/trial/worker_00/worker.log"),
    )

    assert "--inference-slot-name" in cmd
    assert "--inference-slot-max-batch" in cmd
    slot_name = cmd[cmd.index("--inference-slot-name") + 1]
    assert slot_name.endswith("-0")  # worker_index=0


def test_refresh_replay_shards_uses_override_root_for_donor(tmp_path: Path) -> None:
    recipient_trial = tmp_path / "train_trial_recipient"
    donor_trial = tmp_path / "train_trial_donor"
    recipient_trial.mkdir()
    donor_trial.mkdir()

    recipient_replay = recipient_trial / "replay_shards"
    recipient_replay.mkdir()

    override_root = tmp_path / "replay_override"
    donor_replay = override_root / donor_trial.name / "replay_shards"
    donor_replay.mkdir(parents=True)

    save_local_shard_arrays(
        donor_replay / f"shard_000000{LOCAL_SHARD_SUFFIX}",
        arrs={
            "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
            "policy_target": np.ones((1, 4672), dtype=np.float32),
            "wdl_target": np.array([1], dtype=np.int8),
            "priority": np.array([1.0], dtype=np.float32),
            "has_policy": np.array([1], dtype=np.uint8),
        },
    )

    summary = _refresh_replay_shards_on_exploit(
        config={"tune_replay_root_override": str(override_root)},
        replay_shard_dir=recipient_replay,
        recipient_trial_dir=recipient_trial,
        donor_trial_dir=donor_trial,
        keep_recent_fraction=1.0,
        keep_older_fraction=1.0,
        donor_shards=1,
        donor_skip_newest=0,
        shard_size=1,
        holdout_fraction=0.0,
    )

    assert summary["donor_available"] == 1
    assert summary["donor_copied"] == 1
    copied = sorted(recipient_replay.iterdir())
    assert len(copied) == 1
    assert copied[0].suffix == LOCAL_SHARD_SUFFIX


def test_replay_override_under_wsl_remaps_to_linux_run_sidecar(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_abc"
    trial_dir.mkdir()

    replay_dir = _trial_replay_shard_dir(
        config={
            "work_dir": "/home/josh/projects/chess/runs/pbt2_fresh_run9/tune",
            "tune_replay_root_override": "/mnt/c/chess_active/pbt2_fresh_run9_replay",
        },
        trial_dir=trial_dir,
    )

    assert replay_dir == Path("/home/josh/projects/chess/runs/pbt2_fresh_run9_replay") / trial_dir.name / "replay_shards"


def test_server_override_under_wsl_remaps_to_linux_run_sidecar(tmp_path: Path) -> None:
    server_root = _resolve_harness_override_root(
        raw_root="/mnt/c/chess_active/pbt2_fresh_run9_server",
        work_dir=Path("/home/josh/projects/chess/runs/pbt2_fresh_run9"),
        suffix="server",
    )

    assert server_root == Path("/home/josh/projects/chess/runs/pbt2_fresh_run9_server")


def test_build_distributed_worker_cmd_remaps_wsl_server_auth_paths(tmp_path: Path) -> None:
    tune_work_dir = tmp_path / "runs" / "pbt2_fresh_run9" / "tune"
    tune_work_dir.mkdir(parents=True)
    server_root = tune_work_dir.parent.with_name(f"{tune_work_dir.parent.name}_server")
    server_root.mkdir(parents=True, exist_ok=True)
    password_file = server_root / "tune_worker_current.password"
    password_file.write_text("secret\n", encoding="utf-8")

    cmd = _build_distributed_worker_cmd(
        config={
            "work_dir": str(tune_work_dir),
            "distributed_server_url": "http://127.0.0.1:45453",
            "distributed_worker_username": "tune_worker_old",
            "distributed_worker_password_file": "/mnt/c/chess_active/pbt2_fresh_run9_server/tune_worker_old.password",
            "stockfish_path": "/tmp/stockfish",
            "distributed_server_root": "/mnt/c/chess_active/pbt2_fresh_run9_server",
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": True,
            "distributed_worker_sf_workers": 1,
            "distributed_worker_poll_seconds": 1.0,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_00"),
        trial_id="trial_00000",
        worker_index=0,
        worker_log=Path("/tmp/trial/worker_00/worker.log"),
    )

    assert cmd[cmd.index("--username") + 1] == "tune_worker_current"
    assert cmd[cmd.index("--password-file") + 1] == str(password_file)


def test_patch_experiment_state_for_resume_adds_new_jsonable_keys(tmp_path: Path) -> None:
    state_file = tmp_path / "experiment_state-2026-03-29.json"
    state_file.write_text(
        json.dumps(
            {
                "trial_data": [
                    [json.dumps({"config": {"seed": 7, "lr": 1.0e-3}}), {"meta": "ignored"}],
                ]
            }
        ),
        encoding="utf-8",
    )

    added, skipped = _patch_experiment_state_for_resume(
        state_file=state_file,
        param_space={
            "seed": 7,
            "lr": 1.0e-3,
            "distributed_upload_compact_shard_size": 2000,
            "distributed_upload_compact_max_age_seconds": 90.0,
        },
    )

    assert added == {
        "distributed_upload_compact_shard_size",
        "distributed_upload_compact_max_age_seconds",
    }
    assert skipped == set()

    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    saved_trial = json.loads(saved_state["trial_data"][0][0])
    assert saved_trial["config"]["distributed_upload_compact_shard_size"] == 2000
    assert saved_trial["config"]["distributed_upload_compact_max_age_seconds"] == 90.0
    assert _extract_saved_trial_config_keys(experiment_state=saved_state) == {
        "seed",
        "lr",
        "distributed_upload_compact_shard_size",
        "distributed_upload_compact_max_age_seconds",
    }


def test_patch_experiment_state_for_resume_skips_non_jsonable_keys(tmp_path: Path) -> None:
    state_file = tmp_path / "experiment_state-2026-03-29.json"
    state_file.write_text(
        json.dumps(
            {
                "trial_data": [
                    [json.dumps({"config": {"seed": 7}}), {"meta": "ignored"}],
                ]
            }
        ),
        encoding="utf-8",
    )

    added, skipped = _patch_experiment_state_for_resume(
        state_file=state_file,
        param_space={
            "seed": 7,
            "new_search_space": object(),
        },
    )

    assert added == set()
    assert skipped == {"new_search_space"}

    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    saved_trial = json.loads(saved_state["trial_data"][0][0])
    assert saved_trial["config"] == {"seed": 7}


def test_list_matching_pids_filters_by_module_terms_and_exclusions() -> None:
    ps_output = "\n".join(
        [
            "101 /usr/bin/python3 -m chess_anti_engine.worker --trial-id t0 --work-dir /tmp/w0",
            "202 /usr/bin/python3 -m chess_anti_engine.worker --trial-id t0 --work-dir /tmp/w1",
            "303 /usr/bin/python3 -m chess_anti_engine.inference --publish-dir /tmp/p0 --slot-prefix s0",
        ]
    )

    pids = _list_matching_pids(
        module="chess_anti_engine.worker",
        required_terms=["--trial-id", "t0", "--work-dir", "/tmp/w0"],
        ps_output=ps_output,
        exclude_pids=[999],
    )
    assert pids == [101]

    excluded = _list_matching_pids(
        module="chess_anti_engine.worker",
        required_terms=["--trial-id", "t0"],
        ps_output=ps_output,
        exclude_pids=[101, 202],
    )
    assert excluded == []


def test_launch_inference_broker_does_not_inherit_worker_compile(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class DummyProc:
        def poll(self) -> int | None:
            return None

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return DummyProc()

    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime.terminate_matching_processes", lambda **kwargs: [])
    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime.subprocess.Popen", _fake_popen)

    publish_dir = tmp_path / "publish"
    trial_dir = tmp_path / "trial"
    publish_dir.mkdir()
    trial_dir.mkdir()

    _launch_inference_broker(
        config={
            "distributed_workers_per_trial": 2,
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": True,
            "distributed_server_root": str(tmp_path / "server"),
        },
        trial_id="trial_00000",
        publish_dir=publish_dir,
        trial_dir=trial_dir,
    )

    assert calls
    assert "--compile-inference" not in calls[0]


def test_launch_inference_broker_respects_dedicated_compile_flag(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class DummyProc:
        def poll(self) -> int | None:
            return None

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return DummyProc()

    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime.terminate_matching_processes", lambda **kwargs: [])
    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime.subprocess.Popen", _fake_popen)

    publish_dir = tmp_path / "publish"
    trial_dir = tmp_path / "trial"
    publish_dir.mkdir()
    trial_dir.mkdir()

    _launch_inference_broker(
        config={
            "distributed_workers_per_trial": 2,
            "distributed_worker_device": "cuda",
            "distributed_inference_use_compile": True,
            "distributed_server_root": str(tmp_path / "server"),
        },
        trial_id="trial_00000",
        publish_dir=publish_dir,
        trial_dir=trial_dir,
    )

    assert calls
    assert "--compile-inference" in calls[0]
