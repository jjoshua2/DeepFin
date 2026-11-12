import dataclasses
import chess
import pytest
from chess_anti_engine.natural_outcome import capture_natural_outcome, admit_rows


def capture(board, **kw):
    args = dict(resolved_result="0-1", adjudicated=False, tb_adjudicated=False,
                resigned=False, censored=False)
    args.update(kw)
    return capture_natural_outcome(board, **args)


def mate():
    b=chess.Board()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:b.push_uci(move)
    return b


def test_mate_rows_and_snapshot_before_rescore():
    b=mate();r=capture(b)
    assert r.termination=="CHECKMATE"
    assert admit_rows(r,expected_trace_sha256=r.trace_sha256,plies=[0,1],stm_white=[True,False],outcomes=[2,0],tb_rescored=False)==(2,0)
    b.pop()
    assert len(r.moves)==4
    with pytest.raises(ValueError,match="stored outcome"):
        admit_rows(r,expected_trace_sha256=r.trace_sha256,plies=[0],stm_white=[True],outcomes=[0],tb_rescored=False)


@pytest.mark.parametrize("flag",["adjudicated","tb_adjudicated","resigned","censored"])
def test_non_natural_rejected(flag):
    with pytest.raises(ValueError):capture(mate(),**{flag:True})


def test_claimable_not_automatic_draw():
    b=chess.Board()
    for move in ["g1f3","g8f6","f3g1","f6g8"]*2:b.push_uci(move)
    assert b.can_claim_threefold_repetition()
    with pytest.raises(ValueError,match="automatic natural"):
        capture(b,resolved_result="1/2-1/2")
    for move in ["g1f3","g8f6","f3g1","f6g8"]*2:b.push_uci(move)
    assert capture(b,resolved_result="1/2-1/2").termination=="FIVEFOLD_REPETITION"


def test_missing_history_and_clock_tamper():
    with pytest.raises(ValueError):capture(mate().copy(stack=False))
    b=mate();b.halfmove_clock=99
    with pytest.raises(ValueError,match="state mismatch"):capture(b)


@pytest.mark.parametrize("tb",[-2,-1,0,1,2,True,None])
def test_tb_never_substitutes_game_outcome(tb):
    r=capture(mate())
    with pytest.raises(ValueError):admit_rows(r,expected_trace_sha256=r.trace_sha256,plies=[0],stm_white=[True],outcomes=[2],tb_rescored=tb)


def test_identity_pov_and_duplicate_controls():
    r=capture(mate())
    common=dict(expected_trace_sha256=r.trace_sha256,plies=[0],stm_white=[True],outcomes=[2],tb_rescored=False)
    for change in [dict(expected_trace_sha256="0"*64),dict(stm_white=[False]),dict(plies=[True]),dict(plies=[0,0],stm_white=[True,True],outcomes=[2,2])]:
        with pytest.raises(ValueError):admit_rows(r,**(common|change))
    with pytest.raises(ValueError):admit_rows(dataclasses.replace(r,termination="STALEMATE"),**common)


@pytest.mark.parametrize("outcome",[0,1,2])
def test_production_loss_actual_global_mixture(monkeypatch,outcome):
    import torch
    from chess_anti_engine.train import losses
    logits=torch.tensor([[.2,.1,-.3]])
    outputs={"policy":torch.zeros((1,3)),"wdl":logits}
    search=torch.tensor([[.2,.3,.5]])
    batch={"x":torch.zeros((1,1,1,1)),"policy_t":torch.tensor([[1.,0.,0.]]),
           "wdl_t":torch.tensor([outcome]),"has_policy":torch.ones(1),
           "is_network_turn":torch.ones(1),"search_wdl":search,
           "has_search_wdl":torch.ones(1),"has_sf_wdl":torch.zeros(1)}
    original=losses.soft_cross_entropy;captured=[]
    def observe(pred,target):
        if pred is logits:captured.append(target.detach().clone())
        return original(pred,target)
    monkeypatch.setattr(losses,"soft_cross_entropy",observe)
    def run(frac):
        captured.clear()
        losses.compute_loss(outputs,batch,sf_wdl_frac=0.,search_wdl_frac=frac,
            use_adjusted_wdl_target=False,wdl_terminal_outcome_plies=0,
            sf_search_dampen_sf_low=0.,sf_search_dampen_sf_high=0.,
            w_policy=0.,w_sf_move=0.,w_sf_eval=0.,w_categorical=0.,
            w_volatility=0.,w_moves_left=0.)
        assert len(captured)==1
        return captured[0]
    a=run(1.);b=run(.8)
    onehot=torch.nn.functional.one_hot(torch.tensor([outcome]),3).float()
    torch.testing.assert_close(a,search)
    torch.testing.assert_close(b-a,.2*(onehot-search))
    # Production fallback is why an external admission gate must require labels.
    batch["has_search_wdl"]=torch.zeros(1)
    torch.testing.assert_close(run(.8),onehot)


@pytest.mark.parametrize("row,mask",[([float("nan"),0.,1.],True),([.2,.3,.5],False),([0.,0.,0.],True),([-.1,.1,1.],True)])
def test_missing_invalid_search_admission(row,mask):
    from chess_anti_engine.natural_outcome import require_search_labels
    with pytest.raises(ValueError):require_search_labels([row],[mask])
    require_search_labels([[.2,.3,.5]],[True])


@pytest.mark.parametrize("moves",[["e2e4"],tuple(["e2e4"]*1601),("x"*10000,)])
def test_forged_receipt_rejected_before_replay(monkeypatch,moves):
    r=dataclasses.replace(capture(mate()),moves=moves)
    def forbidden(*args,**kwargs):raise AssertionError("replay started")
    monkeypatch.setattr(chess,"Board",forbidden)
    with pytest.raises(ValueError,match="bounded immutable"):
        admit_rows(r,expected_trace_sha256=r.trace_sha256,plies=[0],stm_white=[True],outcomes=[2],tb_rescored=False)
