import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.tune.distributed_runtime import (
    _build_distributed_worker_cmd,
    _ensure_distributed_workers,
    _launch_inference_broker,
    _worker_launch_signature,
)
from chess_anti_engine.tune.harness import (
    _prepare_distributed_worker_auth,
    _patch_experiment_state_for_resume,
)
from chess_anti_engine.tune._utils import (
    resolve_local_override_root as _resolve_harness_override_root,
)
from chess_anti_engine.tune.process_cleanup import _list_matching_pids
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


def test_prepare_distributed_worker_auth_reads_password_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHESS_TEST_WORKER_PASSWORD", "secret-from-env")

    username, password_file = _prepare_distributed_worker_auth(
        server_root=tmp_path,
        config={
            "distributed_worker_username": "worker",
            "distributed_worker_password": None,
            "distributed_worker_password_env": "CHESS_TEST_WORKER_PASSWORD",
        },
    )

    assert username == "worker"
    assert password_file.read_text().strip() == "secret-from-env"


def test_build_distributed_worker_cmd_passes_stockfish_nice() -> None:
    cmd = _build_distributed_worker_cmd(
        config={
            "distributed_server_url": "http://127.0.0.1:45453",
            "distributed_worker_username": "worker",
            "distributed_worker_password_file": "/tmp/pw",
            "stockfish_path": "/tmp/stockfish",
            "distributed_server_root": "/tmp/server",
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": False,
            "distributed_worker_sf_workers": 3,
            "distributed_worker_poll_seconds": 1.0,
            "sf_nice": 19,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_00"),
        trial_id="trial_00000",
        worker_index=0,
        worker_log=Path("/tmp/trial/worker_00/worker.log"),
    )

    assert cmd[cmd.index("--sf-workers") + 1] == "3"
    assert cmd[cmd.index("--sf-nice") + 1] == "19"


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


def test_build_distributed_worker_cmd_adds_threaded_dispatcher_max_batch() -> None:
    cmd = _build_distributed_worker_cmd(
        config={
            "distributed_server_url": "http://127.0.0.1:45453",
            "distributed_worker_username": "worker",
            "distributed_worker_password_file": "/tmp/pw",
            "stockfish_path": "/tmp/stockfish",
            "distributed_server_root": "/tmp/server",
            "distributed_worker_device": "cuda",
            "distributed_worker_use_compile": True,
            "distributed_worker_threaded": True,
            "distributed_worker_selfplay_threads": 32,
            "distributed_worker_threaded_dispatcher": True,
            "distributed_worker_dispatcher_batch_wait_ms": 2.0,
            "distributed_worker_dispatcher_max_batch": 1024,
            "distributed_worker_dispatcher_target_batch": 170,
            "distributed_worker_sf_workers": 1,
            "distributed_worker_poll_seconds": 1.0,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_00"),
        trial_id="trial_00000",
        worker_index=0,
        worker_log=Path("/tmp/trial/worker_00/worker.log"),
    )

    assert "--threaded-dispatcher" in cmd
    assert cmd[cmd.index("--selfplay-threads") + 1] == "32"
    assert cmd[cmd.index("--dispatcher-batch-wait-ms") + 1] == "2.0"
    assert cmd[cmd.index("--dispatcher-max-batch") + 1] == "1024"
    assert cmd[cmd.index("--dispatcher-target-batch") + 1] == "170"


def test_build_distributed_worker_cmd_adds_multiple_inference_slots() -> None:
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
            "distributed_inference_slots_per_worker": 4,
            "seed": 123,
        },
        trial_root=Path("/tmp/trial/worker_01"),
        trial_id="trial_00000",
        worker_index=1,
        worker_log=Path("/tmp/trial/worker_01/worker.log"),
    )

    slot_names = cmd[cmd.index("--inference-slot-name") + 1].split(",")
    assert [name.rsplit("-", 1)[-1] for name in slot_names] == ["4", "5", "6", "7"]


class _FakeProc:
    pid = 1234
    returncode = None
    _cae_worker_launch_signature: tuple[object, ...] | None = None

    def __init__(self) -> None:
        self.stopped = False

    def poll(self):
        return None if not self.stopped else -15


def test_ensure_distributed_workers_restarts_on_launch_config_change(
    tmp_path: Path, monkeypatch,
) -> None:
    old_config = {
        "distributed_workers_per_trial": 1,
        "distributed_server_url": "http://127.0.0.1:45453",
        "distributed_server_root": str(tmp_path / "server"),
        "stockfish_path": "/tmp/stockfish",
        "distributed_worker_sf_workers": 1,
    }
    new_config = {**old_config, "distributed_worker_sf_workers": 2}
    old_proc = _FakeProc()
    old_proc._cae_worker_launch_signature = _worker_launch_signature(
        config=old_config, trial_id="trial_00000", worker_index=0,
    )
    launched: list[tuple[dict, int]] = []

    def _fake_stop(proc):
        proc.stopped = True

    def _fake_launch(*, config, trial_dir, trial_id, worker_index):
        _ = trial_dir
        proc = _FakeProc()
        proc._cae_worker_launch_signature = _worker_launch_signature(
            config=config, trial_id=trial_id, worker_index=worker_index,
        )
        launched.append((dict(config), int(worker_index)))
        return proc

    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime._stop_process", _fake_stop)
    monkeypatch.setattr("chess_anti_engine.tune.distributed_runtime._launch_distributed_worker", _fake_launch)

    out = cast(Any, _ensure_distributed_workers(
        config=new_config,
        trial_dir=tmp_path,
        trial_id="trial_00000",
        procs=cast(Any, [old_proc]),
    ))

    assert old_proc.stopped is True
    assert len(launched) == 1
    assert launched[0][0]["distributed_worker_sf_workers"] == 2
    assert out[0]._cae_worker_launch_signature == _worker_launch_signature(
        config=new_config, trial_id="trial_00000", worker_index=0,
    )


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


def test_server_override_under_wsl_remaps_to_linux_run_sidecar() -> None:
  # Harness now passes ``work_dir / "tune"`` to the canonical helper
  # (the helper expects the *tune* subdir and derives run_root = .parent).
    server_root = _resolve_harness_override_root(
        raw_root="/mnt/c/chess_active/pbt2_fresh_run9_server",
        tune_work_dir=Path("/home/josh/projects/chess/runs/pbt2_fresh_run9/tune"),
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

    added, skipped, saved_keys = _patch_experiment_state_for_resume(
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
    assert saved_keys == {
        "seed",
        "lr",
        "distributed_upload_compact_shard_size",
        "distributed_upload_compact_max_age_seconds",
    }

    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    saved_trial = json.loads(saved_state["trial_data"][0][0])
    assert saved_trial["config"]["distributed_upload_compact_shard_size"] == 2000
    assert saved_trial["config"]["distributed_upload_compact_max_age_seconds"] == 90.0


def test_patch_experiment_state_for_resume_overlays_selected_keys(tmp_path: Path) -> None:
    state_file = tmp_path / "experiment_state-2026-03-29.json"
    state_file.write_text(
        json.dumps(
            {
                "trial_data": [
                    [
                        json.dumps(
                            {
                                "config": {
                                    "seed": 7,
                                    "distributed_inference_broker_enabled": False,
                                }
                            }
                        ),
                        {"meta": "ignored"},
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )

    added, skipped, saved_keys = _patch_experiment_state_for_resume(
        state_file=state_file,
        param_space={
            "seed": 99,
            "distributed_inference_broker_enabled": True,
        },
        overlay_keys={"distributed_inference_broker_enabled"},
    )

    assert added == {"distributed_inference_broker_enabled"}
    assert skipped == set()
    assert saved_keys == {"seed", "distributed_inference_broker_enabled"}

    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    saved_trial = json.loads(saved_state["trial_data"][0][0])
    assert saved_trial["config"]["seed"] == 7
    assert saved_trial["config"]["distributed_inference_broker_enabled"] is True


def test_patch_experiment_state_for_resume_never_touches_construction_bound_keys(tmp_path: Path) -> None:
    # Model-encoding (policy_encoding) and optimizer-construction (optimizer)
    # keys are config-owned at model/optimizer build time, and the trainer is
    # built from config before checkpoint restore. So the resume overlay must
    # neither overwrite a present one nor inject an absent one — else the model/
    # optimizer is rebuilt away from the checkpoint the saved state fits. Safe
    # selfplay/infra keys still propagate.
    state_file = tmp_path / "experiment_state-2026-03-29.json"
    state_file.write_text(
        json.dumps(
            {
                "trial_data": [
                    [
                        json.dumps(
                            {"config": {"seed": 7, "policy_encoding": "az_4672", "optimizer": "nadamw"}}
                        ),
                        {"meta": "ignored"},
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )

    added, _skipped, saved_keys = _patch_experiment_state_for_resume(
        state_file=state_file,
        param_space={
            "seed": 7,
            "policy_encoding": "lc0_1858",  # construction-bound, PRESENT -> must NOT overwrite
            "optimizer": "aurora",          # construction-bound, PRESENT -> must NOT overwrite
            "embed_dim": 999,               # construction-bound, ABSENT  -> must NOT inject
            "history_rep_fix": True,        # safe selfplay flag, ABSENT  -> should propagate
            "num_samples": 4,               # safe infra key, ABSENT      -> should propagate
        },
        overlay_keys={"policy_encoding", "optimizer", "embed_dim", "history_rep_fix", "num_samples"},
    )

    cfg = json.loads(json.loads(state_file.read_text(encoding="utf-8"))["trial_data"][0][0])["config"]
    # construction-bound keys: present ones keep checkpoint value, absent ones not injected
    assert cfg["policy_encoding"] == "az_4672"
    assert cfg["optimizer"] == "nadamw"
    assert "embed_dim" not in cfg
    assert "embed_dim" not in added
    assert "embed_dim" not in saved_keys
    assert "policy_encoding" not in added
    assert "optimizer" not in added
    # safe keys propagate
    assert cfg["history_rep_fix"] is True
    assert cfg["num_samples"] == 4
    assert {"history_rep_fix", "num_samples"} <= added


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

    added, skipped, _saved = _patch_experiment_state_for_resume(
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

    def _fake_popen(cmd, **_kwargs):
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

    def _fake_popen(cmd, **_kwargs):
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
