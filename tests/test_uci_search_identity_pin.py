"""Pin the UCI search's move choices on a fixed-seed CPU net.

The multi-GPU degrade machinery (device guards, flap limits, primary guard,
startup watchdogs) must not change what a healthy search computes. This test
runs a fixed-seed tiny net through ``SearchWorker`` -- classic single-device,
and a 1-device multi-GPU PUCV pool with and without compiled-bucket padding --
and pins a hash of every (bestmove, nodes, PV, score, root visit vector).

It runs in a subprocess with ONE torch thread: the pytest session caps torch
at 2 threads, and intra-op parallelism may change float reduction order.
If a deliberate search change moves these hashes, re-derive them on the
branch point and on the change and say so in the PR.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HARNESS = r'''
import hashlib, json, sys, threading
import chess, torch
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

torch.set_num_threads(1)
mode, pad = sys.argv[1], sys.argv[2] == "1"

def model():
    torch.manual_seed(1234)
    m = build_model(ModelConfig(input_extra_features="v1", embed_dim=32, num_layers=2,
                                num_heads=2, ffn_mult=2.0))
    m.eval()
    return m

def ev():
    kw = {"pad_to_compiled_buckets": True} if pad else {}
    return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        model(), device="cpu", max_batch=512, use_amp=False, n_slots=2, **kw))

FENS = [chess.STARTING_FEN,
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
        "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3",
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"]
w = SearchWorker(ev(), device="cpu",
                 gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=256, add_noise=False),
                 chunk_sims=128, n_walkers=1)
if mode == "pool1":
    w.install_multi_gpu_pucv([ev()], gather=24, as_factories=False)
out = []
for fen in FENS:
    w.reset_tree()
    r = w.run(chess.Board(fen), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=600)
    try:
        mv, vis = w._tree.get_children_visits(w._root_id)
        vis = [int(v) for v in vis]
        mv = [int(x) for x in mv]
    except Exception as e:
        # A mate-in-one root answers without building a tree; the message is
        # part of the pinned record.
        mv, vis = [], [str(e)]
    out.append(dict(fen=fen, best=r.bestmove_uci, nodes=r.nodes, pv=list(r.pv),
                    score=r.score_cp, mate=r.score_mate, visits=vis, moves=mv))
w.close()
print(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16])
'''

# Derived on origin/main 3a737f786 (the branch point of the degrade-gaps PR)
# and unchanged by it.
_PINNED = {
    ("classic", "0"): "875232a22f2d86ac",
    ("pool1", "0"): "a2d39e00bdc05854",
    ("pool1", "1"): "a2d39e00bdc05854",
}


@pytest.mark.parametrize(("mode", "pad"), sorted(_PINNED))
def test_search_identity_is_pinned(tmp_path, mode, pad) -> None:
    script = tmp_path / "identity.py"
    script.write_text(_HARNESS)
    repo = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(script), mode, pad], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path), timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = proc.stdout.strip().splitlines()[-1]
    assert got == _PINNED[(mode, pad)], (mode, pad, got)
