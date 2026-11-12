import json
import os
import subprocess
import sys
import pytest


def test_producer_to_bank_and_actual_loss(tmp_path):
    # One actual production selfplay path with synthetic logits and fake SF.
    # Opening is a complete standard-start trace, not a fabricated terminal FEN.
    source = r"""
import sys,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import chess,numpy as np,torch
from tests.test_game_provenance import _collect
from tests.selfplay_helpers import UniformPolicyValueModel
from chess_anti_engine.selfplay.config import GameConfig
import chess_anti_engine.selfplay.state as state
from chess_anti_engine.moves import move_to_index_for_encoding
from scripts.e2_finite_bank import FiniteBank
from chess_anti_engine.train.e2_batch_adapter import load_paired_batch,compute_pair_losses
b=chess.Board()
for u in ['f2f3','e7e5','g2g4']:b.push_uci(u)
state.resolve_slot_opening=lambda **kw:NS(board=b.copy(stack=True),source='synthetic_complete_history')
idx=move_to_index_for_encoding(chess.Move.from_uci('d8h4'),b,policy_encoding='lc0_1858')
class MateModel(UniformPolicyValueModel):
 def forward(self,x):
  out=super().forward(x);out['policy'][:,idx]=1000.;return out
bank=FiniteBank(Path(sys.argv[1]),run_id='natural',attempts=1,eligible_games=1)
_collect(model=MateModel().eval(),games=1,target_games=1,
 game=GameConfig(max_plies=8,selfplay_fraction=1.,input_history_encoding='lc0_root_legacy_meta',history_rep_fix=True),
 on_state_ready=bank.join.on_state_ready,on_game_complete=bank.join.on_game_complete,on_game_provenance=bank.join.on_game_provenance)
manifest=bank.finish();value=json.loads(manifest.read_bytes());r=value['attempts'][0];root=manifest.parent
pair=load_paired_batch(shard=root/r['shard'],candidate=root/r['candidate'],qualified=root/r['qualified'],expected_manifest_sha256=r['readback_sha256'],indices=(0,),expected_indices_sha256=hashlib.sha256(b'[0]').hexdigest(),categorical_bins=32,categorical_sigma=.04)
a={'policy':torch.zeros((1,1858)), 'wdl':torch.zeros((1,3)), 'categorical':torch.zeros((1,32))}
c={k:v.clone() for k,v in a.items()}
from chess_anti_engine.train import losses
original=losses.soft_cross_entropy;seen={}
def observe(pred,target):
 if pred is a['wdl']:seen['a']=target.clone()
 if pred is c['wdl']:seen['b']=target.clone()
 return original(pred,target)
losses.soft_cross_entropy=observe
loss=compute_pair_losses(pair,a,c,shared_kwargs={'w_policy':0.,'w_categorical':1.})
natural=torch.nn.functional.one_hot(pair.outcome20['wdl_t'],3).float()
torch.testing.assert_close(seen['b'],.8*seen['a']+.2*natural)
assert torch.isfinite(loss[0]['total']) and torch.isfinite(loss[1]['total'])
assert value['selected_start_ordinals']==[0]
assert json.loads((root/r['qualified']/'GAME-READBACK-COMMITTED.json').read_bytes())['all175_replayed']
"""
    subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)],
        check=True,
        timeout=90,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    result = json.loads(
        (tmp_path / "natural/BANK-READBACK-COMMITTED.json").read_bytes()
    )
    assert result["status"] == "FINITE_READBACK_BANK_NOT_EXPERIMENT_READY"


def test_incomplete_collection_cannot_commit(tmp_path):
    from scripts.e2_finite_bank import FiniteBank

    bank = FiniteBank(tmp_path, run_id="r", attempts=2, eligible_games=1)
    with pytest.raises(ValueError, match="incomplete"):
        bank.finish()
    assert not list(tmp_path.rglob("BANK-READBACK-COMMITTED.json"))


def test_all_excluded_attempts_never_commit(tmp_path):
    from scripts.e2_finite_bank import FiniteBank
    from tests.test_e2_game_join import fixture
    from types import SimpleNamespace as NS

    bank = FiniteBank(tmp_path, run_id="r", attempts=2, eligible_games=1)
    bank.join.on_state_ready(
        NS(
            continuous=False,
            natural_outcome_sink=None,
            game=NS(
                input_history_encoding="lc0_root_legacy_meta",
                history_rep_fix=True,
                syzygy_path=None,
            ),
        )
    )
    for ordinal in (1, 0):
        p, _ = fixture(ordinal=ordinal, gid=ordinal + 1, retained=False, terminal=False)
        bank.join.on_game_provenance(p)
    assert set(bank.records) == {0, 1}
    with pytest.raises(ValueError, match="insufficient eligible"):
        bank.finish()
    assert not list(tmp_path.rglob("BANK-READBACK-COMMITTED.json"))


def test_observation_tamper_prevents_final_manifest(tmp_path):
    from scripts.e2_finite_bank import FiniteBank

    bank = FiniteBank(tmp_path, run_id="r", attempts=1, eligible_games=1)
    # Isolate final publication guard with synthetic ledger metadata, no bank claim.
    bank.join.seen = {0}
    bank.records = {
        0: {
            "ordinal": 0,
            "game_id": 1,
            "exclusion": None,
            "observation_sha256": "0" * 64,
        }
    }
    (bank.join.directory / "0.observation.json").write_text("changed")
    with pytest.raises(ValueError, match="observation changed"):
        bank.finish()
    assert not list(tmp_path.rglob("BANK-READBACK-COMMITTED.json"))
