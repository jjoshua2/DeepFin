from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]


def _best_save_env(tmp_path: Path, *, fail_second_copy: bool) -> dict[str, str]:
    work = tmp_path / "work"
    trial = work / "tune" / "train_trial_demo_00000_0"
    trial.mkdir(parents=True)
    replay = tmp_path / "replay-root" / trial.name / "replay_shards"
    snapshot = tmp_path / "auto" / "regret_0.1000_iter10"
    replay.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (snapshot / "trainer.pt").write_bytes(b"checkpoint")
    (snapshot / "meta.json").write_text(json.dumps({
        "regret": 0.1,
        "iter": 10,
        "ema_winrate": 0.5,
        "opp_strength_ema": 1200.0,
    }))
    (replay / "shard.npz").write_bytes(b"replay")
    config_text = (REPO / "configs/pbt2_small.yaml").read_text()
    replay_root_line = re.compile(r"(?m)^(  tune_replay_root_override:\s*).*$")
    config_text, replacements = replay_root_line.subn(
        lambda match: f"{match.group(1)}{tmp_path / 'replay-root'}",
        config_text,
    )
    assert replacements == 1
    config = tmp_path / "config.yaml"
    config.write_text(config_text)
    env = os.environ | {
        "TRAIN_WORK_DIR": str(work),
        "TRAIN_AUTO_BEST_REGRET_DIR": str(tmp_path / "auto"),
        "TRAIN_BEST_POOLS_DIR": str(tmp_path / "pools"),
        "TRAIN_CONFIG": str(config),
    }
    if fail_second_copy:
        shim = tmp_path / "bin"
        shim.mkdir()
        counter = tmp_path / "cp-count"
        cp = shim / "cp"
        cp.write_text(
            '#!/bin/sh\nn=$(cat "$CP_COUNT" 2>/dev/null || echo 0); '
            'n=$((n+1)); echo $n > "$CP_COUNT"; '
            '[ $n -eq 2 ] && exit 23; exec /bin/cp "$@"\n'
        )
        cp.chmod(0o755)
        env["CP_COUNT"] = str(counter)
        env["PATH"] = str(shim) + ":" + env["PATH"]
    return env


def test_best_save_failed_copy_is_not_published_and_can_be_retried(tmp_path: Path) -> None:
    env = _best_save_env(tmp_path, fail_second_copy=True)
    command = ["bash", "scripts/train.sh", "best-save", "candidate"]
    failed = subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    final_pool = tmp_path / "pools" / "candidate"
    assert failed.returncode == 23
    assert not final_pool.exists()
    assert not list((tmp_path / "pools").glob(".best-save.*"))

    env["PATH"] = os.environ["PATH"]
    retried = subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    assert retried.returncode == 0, retried.stderr
    manifest = json.loads((final_pool / "manifest.json").read_text())
    assert manifest["label"] == "candidate"
    assert (final_pool / "seeds/slot_000/trainer.pt").read_bytes() == b"checkpoint"
    assert (final_pool / "seeds/slot_000/replay_shards/shard.npz").read_bytes() == b"replay"
