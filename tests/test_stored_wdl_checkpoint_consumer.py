"""Synthetic-only internal verified-model callback isolation contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
import pytest
import torch
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model


@pytest.mark.parametrize("bad_hash", [False, True])
def test_internal_consumer_only_after_strict_load(tmp_path: Path, bad_hash):
    cfg = ModelConfig(
        kind="transformer",
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        use_smolgen=False,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
        history_rep_fix=False,
    )
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    path = tmp_path / "synthetic.pt"
    torch.save({"arch": arch, "model": build_model(cfg).state_dict()}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    request = {
        "path": str(path),
        "sha256": "0" * 64 if bad_hash else sha,
        "architecture_json": json.dumps(arch),
    }
    script = """
import json,sys,resource
resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(60,60))
sys.path.insert(0,sys.argv[1])
from scripts.stored_wdl_checkpoint import _qualify
called=[]
def consume(model,receipt):
 assert not model.training
 assert all(not p.requires_grad for p in model.parameters())
 called.append(True)
 return {'synthetic_verified_tensors':receipt['state_tensors']}
try:
 report=_qualify(json.loads(sys.argv[2]),verified_model_callback=consume)
 print(json.dumps({'called':called,'diagnostic':report['diagnostic']}))
except ValueError:
 assert not called
 print(json.dumps({'called':called,'rejected':True}))
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    repo = str(Path(__file__).resolve().parents[1])
    run = subprocess.run(
        [sys.executable, "-I", "-c", script, repo, json.dumps(request)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    result = json.loads(run.stdout)
    assert result["called"] == ([] if bad_hash else [True])
