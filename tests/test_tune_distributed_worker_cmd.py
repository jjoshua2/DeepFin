from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import save_npz_arrays
from chess_anti_engine.tune.trainable import _build_distributed_worker_cmd, _refresh_replay_shards_on_exploit


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

    save_npz_arrays(
        donor_replay / "shard_000000.npz",
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
    assert copied[0].suffix == ".npz"
