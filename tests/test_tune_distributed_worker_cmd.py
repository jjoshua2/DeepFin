from pathlib import Path

from chess_anti_engine.tune.trainable import _build_distributed_worker_cmd


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
