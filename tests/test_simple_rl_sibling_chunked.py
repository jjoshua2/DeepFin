"""The sibling term's chunked forward with a backward per chunk (``--sibling-chunk-pairs``).

The term is ``w * mean over the step's pairs``; ``SiblingTerm.penalty`` forwards
at most ``chunk_pairs`` pairs at a time, backpropagates each chunk inside the
Trainer's penalty hook, and returns a 0-valued leaf. These tests hold it to the
single-forward term it replaced. That reference lives ONLY here (the production
code has one path): one Trainer update, through the real ``train_steps``, on a
tiny model in fp32, must give every parameter the same ``.grad`` at the
optimizer step, the same clipped gradient norm, and the same weights after the
update, at chunk sizes of 1 pair, 2 pairs, n - 1, n and more than n, over a plan
whose last batch is short (the Trainer scales that batch's loss by rows /
batch_size, and the term must carry the same factor). Also covered: the
Trainer's ``zero_grad`` runs before the hook (the term's gradients survive to
the step), only one chunk's graph is alive at a time, the counters equal the
reference's, bf16 autocast on CPU, and the guards.
"""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import sibling as sib
from chess_anti_engine.simple_rl.learn import ScheduledSampler, TrainSettings, attach_anchor, build_trainer, plan_batches
from tests.simple_rl_helpers import tiny_model_config
from tests.test_simple_rl_accum_steps import _arrays

N_ROWS, BATCH, ROWS_WITH_PAIRS, W = 37, 16, 4, 3.0   # batches of 16, 16 and a short 5; 8 pairs per full step


class FakePairs:
    """The SidecarPairs interface the term reads: every row carries exactly 2 pairs,
    with fixed random child planes, dq and min_visits."""

    def __init__(self, n_rows: int, planes: int, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.row_count = np.full(n_rows, 2, dtype=np.int64)
        self.row_start = np.arange(n_rows, dtype=np.int64) * 2
        n = 2 * n_rows
        self._xa = (rng.random((n, planes, 8, 8)) < 0.1).astype(np.float32)
        self._xb = (rng.random((n, planes, 8, 8)) < 0.1).astype(np.float32)
        self.dq = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        self.min_visits = rng.integers(2, 40, n).astype(np.float32)
        self.probe = np.zeros(0, np.int64)

    def child_inputs(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._xa[ids], self._xb[ids]


class ReferenceTerm(sib.SiblingTerm):
    """The term as it was before chunking: ONE grad-enabled forward over every
    pair of the step, returning ``w * mean`` for the Trainer to scale and
    backpropagate with the root loss."""

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        del outputs, batch
        ids = self._batch_pairs()
        self.steps += 1
        if not ids.size:
            self.per_step.append([0, 0.0])
            return None
        weighted, resid = self._pair_loss(*self._tensors(ids))
        loss = weighted.mean()
        self.pairs_seen += int(ids.size)
        self.active += int((weighted.detach() > 0).sum())
        self.loss_sum += float(weighted.detach().sum())
        self.abs_resid_sum += float(resid.detach().abs().sum())
        self.per_step.append([int(ids.size), float(loss.detach())])
        return self.w * loss


def _fake_pairs(n_rows: int, planes: int) -> Any:
    """Typed as Any: the term reads the SidecarPairs interface, which FakePairs provides."""
    return FakePairs(n_rows, planes)


def _plan(n: int = N_ROWS, b: int = BATCH) -> list[tuple[np.ndarray, np.ndarray]]:
    batches, _ = plan_batches(n, batch_size=b, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(5))
    return batches


ORDER = {"loss": "order", "order_margin": 0.05, "order_min_dq": 0.1}   # |dq| ~ U(-0.5, 0.5): 80% of pairs eligible


def _run(term_cls: type, *, chunk: int, steps: int = 3, cpu_bf16: bool = False, w_policy: float = 1.0,
         w_wdl: float = 1.0, form: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train ``steps`` updates through the real Trainer with the term on its hook;
    return per-step optimizer-step grads, clipped norms, final weights, counters."""
    rows = _arrays(N_ROWS)
    settings = TrainSettings(device="cpu", mirror_prob=0.0, batch_size=BATCH, w_policy=w_policy, w_wdl=w_wdl)
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    if cpu_bf16:   # the Trainer's own AMP context is CUDA-only; stand in with CPU bf16 autocast around the same span
        trainer._amp_context = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
    sampler = ScheduledSampler(rows, _plan(), rng=np.random.default_rng(0))
    pairs = _fake_pairs(N_ROWS, int(rows["x"].shape[1]))
    term = term_cls(trainer.model, sampler, pairs, w=W, rows=ROWS_WITH_PAIRS, chunk_pairs=chunk, batch_size=BATCH,
                    device="cpu", **(form or {}))
    attach_anchor(trainer, term)
    grads: list[dict[str, torch.Tensor]] = []
    norms: list[float] = []
    trainer.opt.register_step_pre_hook(lambda _o, _a, _k: grads.append(
        {n: p.grad.detach().clone() for n, p in trainer.model.named_parameters() if p.grad is not None}))
    real_zclip = trainer._zclip_step

    def zclip(*a: Any, **k: Any) -> Any:
        out = real_zclip(*a, **k)
        norms.append(float(out[0]))
        return out

    trainer._zclip_step = zclip
    trainer.train_steps(sampler, batch_size=BATCH, steps=steps)
    return {"grads": grads, "norms": norms, "summary": term.summary(), "state": term.state(),
            "weights": {n: p.detach().clone() for n, p in trainer.model.named_parameters()}}


def _assert_equal_runs(got: dict[str, Any], ref: dict[str, Any], *, rtol: float = 2e-4, atol: float = 1e-7) -> None:
    assert len(got["grads"]) == len(ref["grads"]) == 3
    for ga, gb in zip(got["grads"], ref["grads"], strict=True):
        assert set(ga) == set(gb)
        live = [n for n, g in gb.items() if float(g.abs().max()) > 1e-7]
        assert live
        for n in live:
            np.testing.assert_allclose(ga[n].float().numpy(), gb[n].float().numpy(), rtol=rtol, atol=atol, err_msg=n)
    np.testing.assert_allclose(got["norms"], ref["norms"], rtol=rtol)
    for n, p in ref["weights"].items():
        np.testing.assert_allclose(got["weights"][n].float().numpy(), p.float().numpy(), rtol=rtol, atol=1e-6,
                                   err_msg=n)
    for k in ("steps", "pairs", "active_pairs"):
        assert got["summary"][k] == ref["summary"][k], k
    assert got["summary"]["mean_weighted_loss"] == got["summary"]["mean_weighted_huber"]
    for k in ("mean_weighted_loss", "mean_abs_resid_train"):
        assert got["summary"][k] == pytest.approx(ref["summary"][k], rel=1e-5)
    assert [r[0] for r in got["summary"]["per_step"]] == [r[0] for r in ref["summary"]["per_step"]]
    np.testing.assert_allclose([r[1] for r in got["summary"]["per_step"]],
                               [r[1] for r in ref["summary"]["per_step"]], rtol=1e-5)


def test_the_plan_has_a_short_last_batch_and_full_steps_of_eight_pairs() -> None:
    assert [len(o) for o, _ in _plan()] == [16, 16, 5]
    ref = _run(ReferenceTerm, chunk=1)
    assert [r[0] for r in ref["summary"]["per_step"]] == [8, 8, 8]


@pytest.mark.parametrize("chunk", [1, 2, 7, 8, 13])
def test_chunked_update_equals_the_single_forward_term(chunk: int) -> None:
    """1 pair, 2 pairs, n - 1, n and > n per chunk (n = 8 pairs per step), over a
    short last batch: same grads at the step, same clipped norm, same weights."""
    _assert_equal_runs(_run(sib.SiblingTerm, chunk=chunk), _run(ReferenceTerm, chunk=chunk))


def test_the_equality_is_not_vacuous_without_the_term() -> None:
    """The comparison above sees the term: a run with w 0 differs from the reference."""

    class Off(ReferenceTerm):
        def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
            super().penalty(outputs, batch)
            return None

    with pytest.raises(AssertionError):
        _assert_equal_runs(_run(Off, chunk=8), _run(ReferenceTerm, chunk=8))


def test_the_term_alone_trains_through_the_trainer() -> None:
    """Every other weight 0: the Trainer's backward is well-defined (the 0-valued
    leaf) and the update equals the reference's."""
    _assert_equal_runs(_run(sib.SiblingTerm, chunk=3, w_policy=0.0, w_wdl=0.0),
                       _run(ReferenceTerm, chunk=3, w_policy=0.0, w_wdl=0.0))


def test_bf16_autocast_on_cpu_matches_the_reference_within_bf16_tolerance() -> None:
    probe = torch.zeros(2, 2)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = probe @ probe
    except RuntimeError:
        pytest.skip("CPU autocast bf16 unsupported here")
    got, ref = _run(sib.SiblingTerm, chunk=3, cpu_bf16=True), _run(ReferenceTerm, chunk=3, cpu_bf16=True)
    fp32 = _run(ReferenceTerm, chunk=3)
    with pytest.raises(AssertionError):   # autocast really ran: bf16 is not the fp32 update
        _assert_equal_runs(ref, fp32)
    for ga, gb in zip(got["grads"], ref["grads"], strict=True):
        num = sum(float((ga[n].float() - gb[n].float()).norm() ** 2) for n in gb) ** 0.5
        den = sum(float(gb[n].float().norm() ** 2) for n in gb) ** 0.5
        assert num / den < 2e-2, num / den
    np.testing.assert_allclose(got["norms"], ref["norms"], rtol=2e-2)


def test_the_trainer_zeroes_gradients_before_the_hook_so_the_terms_gradients_reach_the_step() -> None:
    """Order proof: grads are None when the hook runs (zero_grad came first), and
    after the hook the term's gradient is in .grad and is still there at the step."""
    seen: dict[str, Any] = {}

    class Spy(sib.SiblingTerm):
        def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
            seen.setdefault("before", []).append(all(p.grad is None for p in self.model.parameters()))
            out = super().penalty(outputs, batch)
            seen.setdefault("after", []).append(
                {n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None})
            return out

    term_only = _run(Spy, chunk=3, w_policy=0.0, w_wdl=0.0)
    assert seen["before"] == [True, True, True]
    for after, at_step in zip(seen["after"], term_only["grads"], strict=True):
        assert after
        for n, g in after.items():
            torch.testing.assert_close(at_step[n], g)   # nothing between the hook and the step wiped or rescaled it


def test_only_one_chunks_graph_is_alive_at_a_time() -> None:
    """CPU stand-in for the memory claim (peak = root batch + ONE chunk): when a
    chunk's forward starts, every output tensor of the previous chunk forward is
    already freed."""
    rows = _arrays(N_ROWS)
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu", mirror_prob=0.0, batch_size=BATCH),
                            log_dir=None)
    sampler = ScheduledSampler(rows, _plan(), rng=np.random.default_rng(0))
    term = sib.SiblingTerm(trainer.model, sampler, _fake_pairs(N_ROWS, int(rows["x"].shape[1])), w=W,
                           rows=ROWS_WITH_PAIRS, chunk_pairs=2, batch_size=BATCH, device="cpu")
    attach_anchor(trainer, term)
    real_loss = sib.sibling_loss
    alive: list[weakref.ref[torch.Tensor]] = []
    calls = {"n": 0, "stale": 0}

    def watched(model: Any, xa: Any, xb: Any, dq: Any, mv: Any) -> Any:
        gc.collect()
        calls["stale"] += sum(r() is not None for r in alive)
        alive.clear()
        calls["n"] += 1
        weighted, resid = real_loss(model, xa, xb, dq, mv)
        alive.extend([weakref.ref(weighted), weakref.ref(resid)])
        return weighted, resid

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sib, "sibling_loss", watched)
        trainer.train_steps(sampler, batch_size=BATCH, steps=2)
    assert calls["n"] == 8   # 2 steps x 8 pairs / 2 per chunk
    assert calls["stale"] == 0


def test_a_hook_entered_with_uncleared_gradients_is_refused() -> None:
    rows = _arrays(N_ROWS)
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu", mirror_prob=0.0, batch_size=BATCH),
                            log_dir=None)
    sampler = ScheduledSampler(rows, _plan(), rng=np.random.default_rng(0))
    term = sib.SiblingTerm(trainer.model, sampler, _fake_pairs(N_ROWS, int(rows["x"].shape[1])), w=W,
                           rows=ROWS_WITH_PAIRS, chunk_pairs=2, batch_size=BATCH, device="cpu")
    for p in trainer.model.parameters():
        p.grad = torch.zeros_like(p)
    sampler.sample_batch_arrays(BATCH)
    with pytest.raises(RuntimeError, match="not cleared"):
        term.penalty({}, {"x": torch.zeros(16, 1, 8, 8)})


def test_a_sampler_whose_loss_scale_the_term_cannot_reproduce_is_refused() -> None:
    term = sib.SiblingTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0, served=[np.arange(4)]),
                           _fake_pairs(4, 1), w=1.0, rows=4, chunk_pairs=2, batch_size=4, device="cpu")
    term.sampler.cursor = 1
    with pytest.raises(RuntimeError, match="stored_row_quota"):
        term.penalty({}, {"x": torch.zeros(4, 1, 8, 8)})


@pytest.mark.parametrize("bad", [{"chunk_pairs": 0}, {"batch_size": 0}])
def test_the_term_refuses_a_non_positive_chunk_or_batch(bad: dict[str, int]) -> None:
    kw: dict[str, Any] = {"w": 1.0, "rows": 1, "chunk_pairs": 2, "batch_size": 4, "device": "cpu", **bad}
    with pytest.raises(ValueError, match=f"{next(iter(bad))} must be >= 1"):
        sib.SiblingTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0), _fake_pairs(1, 1), **kw)


def test_chunk_pairs_is_recorded_per_segment_and_a_snapshot_resumes_at_another_chunk_size() -> None:
    """chunk_pairs changes memory, not the update: restore at another size is
    allowed and appends a segment [the step it starts at, the new size]."""
    state = _run(sib.SiblingTerm, chunk=3)["state"]
    assert state["chunk_pairs"] == 3
    assert state["chunk_segments"] == [[0, 3]]
    other = sib.SiblingTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0), _fake_pairs(1, 1), w=1.0, rows=1,
                            chunk_pairs=4, batch_size=4, device="cpu")
    other.restore(state)
    assert other.chunk_segments == [[0, 3], [state["steps"], 4]]
    assert other.summary()["chunk_segments"] == [[0, 3], [state["steps"], 4]]
    same = sib.SiblingTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0), _fake_pairs(1, 1), w=1.0, rows=1,
                           chunk_pairs=3, batch_size=4, device="cpu")
    same.restore(state)
    assert same.chunk_segments == [[0, 3]]


@pytest.mark.parametrize("chunk", [1, 3, 8, 13])
def test_the_order_form_chunked_equals_its_single_forward_reference(chunk: int) -> None:
    got, ref = _run(sib.SiblingTerm, chunk=chunk, form=ORDER), _run(ReferenceTerm, chunk=chunk, form=ORDER)
    _assert_equal_runs(got, ref)
    assert got["summary"]["loss"] == "order"
    assert got["summary"]["active_pairs"] > 0
    with pytest.raises(AssertionError):   # the order form is not the diff form
        _assert_equal_runs(got, _run(ReferenceTerm, chunk=chunk))


class _TableNet(torch.nn.Module):
    """Child i's WDL logits are row i of a parameter table (inputs are one-hot rows):
    each child's value is a free parameter, so the loss's gradient is exact."""

    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(logits))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"wdl": x @ self.table}


def _order_grad(v_a: float, v_b: float, dq: float, *, margin: float = 0.0556, min_dq: float = 0.0203
                ) -> tuple[float, torch.Tensor, _TableNet]:
    """One pair (A, B) whose children have values v_a, v_b (v = p_L - p_W: the
    child's WDL is p_W = max(-v, 0), p_L = max(v, 0), the rest drawn)."""
    def logits(v: float) -> list[float]:
        p_w, p_l = max(-v, 0.0), max(v, 0.0)
        return [float(np.log(max(p_w, 1e-13))), float(np.log(1 - p_w - p_l)), float(np.log(max(p_l, 1e-13)))]

    net = _TableNet([logits(v_a), logits(v_b)])
    eye = torch.eye(2)
    weighted, _ = sib.sibling_order_loss(net, eye[:1], eye[1:], torch.tensor([dq]), torch.tensor([64.0]),
                                         margin=margin, min_dq=min_dq)
    weighted.sum().backward()
    assert net.table.grad is not None
    return float(weighted.sum()), net.table.grad.clone(), net


def test_the_order_form_has_exactly_zero_gradient_on_a_pair_ordered_beyond_the_margin() -> None:
    loss, grad, _ = _order_grad(0.30, 0.10, dq=0.15)    # search: A > B; net: A - B = 0.20 > margin
    assert loss == 0.0
    assert torch.count_nonzero(grad) == 0
    loss, grad, _ = _order_grad(0.10, 0.30, dq=0.01)    # mis-ordered but |dq| < min_dq: not a clear pair
    assert loss == 0.0
    assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize(("v_a", "v_b", "dq"), [(0.10, 0.30, 0.15), (0.30, 0.10, -0.15), (0.20, 0.18, 0.15)])
def test_the_order_form_pushes_a_mis_ordered_or_too_close_pair_the_searchs_way(v_a: float, v_b: float,
                                                                              dq: float) -> None:
    loss, grad, net = _order_grad(v_a, v_b, dq=dq)
    assert loss > 0
    assert torch.count_nonzero(grad) > 0
    with torch.no_grad():
        before = sib.child_values(net.table)
        net.table.sub_(grad)                             # one gradient-descent step
        after = sib.child_values(net.table)
    s = float(np.sign(dq))
    assert s * float(after[0] - after[1]) > s * float(before[0] - before[1])


def test_the_new_sibling_settings_leave_the_resume_key_unchanged_at_their_defaults(tmp_path: Any) -> None:
    """An unsealed phase begun before these settings existed trained exactly their
    defaults, so its snapshot must still resume: at the defaults the key equals the
    key of the settings without those fields; any other value changes it."""
    import dataclasses

    from chess_anti_engine.simple_rl import learn

    (tmp_path / "init.pt").write_bytes(b"i")
    (tmp_path / "probe.npz").write_bytes(b"p")
    (tmp_path / "gen_0001" / "collect").mkdir(parents=True)
    (tmp_path / "gen_0001" / "collect" / "manifest.json").write_text("{}")

    def key(settings: TrainSettings) -> str:
        return learn.snapshot_resume_key(settings=settings, generation=1, init_checkpoint=tmp_path / "init.pt",
                                         prev_trainer_checkpoint=None, run_dir=tmp_path, gens=[1],
                                         probe_path=tmp_path / "probe.npz",
                                         batches=[(np.arange(4), np.zeros(0, np.int64))])

    base = TrainSettings(device="cpu")
    assert set(learn.KEY_NEUTRAL_DEFAULTS) == {"sibling_loss", "sibling_order_margin", "sibling_order_min_dq",
                                               "sibling_unit"}
    assert learn.KEY_EXCLUDED == ("sibling_chunk_pairs",)
    for k, v in learn.KEY_NEUTRAL_DEFAULTS.items():
        assert getattr(base, k) == v
    real_asdict = learn.asdict

    def old_asdict(obj: Any) -> dict[str, Any]:
        d = real_asdict(obj)
        for k in (*learn.KEY_NEUTRAL_DEFAULTS, *learn.KEY_EXCLUDED):
            del d[k]
        return d

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(learn, "asdict", old_asdict)
        before = key(base)
    assert key(base) == before
    assert key(dataclasses.replace(base, sibling_chunk_pairs=3)) == before   # never keyed: memory, not the update
    assert key(dataclasses.replace(base, sibling_loss="order", sibling_order_margin=0.0556,
                                   sibling_order_min_dq=0.0203)) != before


def test_the_order_margin_is_capped_at_the_searchs_own_gap() -> None:
    """m_i = min(m, |dq|): a pair the search separates by 0.03 is satisfied at
    s * dv = 0.03 (an uncapped m = 0.0556 would still push it apart); at 0.02 it is
    not; a pair with |dq| = 0.2 uses m itself. (Losses carry the version-2 gap
    weight min(|dq|, 0.62): x 0.03 and x 0.2 here.)"""
    loss, grad, _ = _order_grad(0.0301, 0.00, dq=0.03)   # 1e-4 above the boundary: float32 softmax round-off
    assert loss == 0.0
    assert torch.count_nonzero(grad) == 0
    loss, grad, _ = _order_grad(0.02, 0.00, dq=0.03)
    assert loss > 0
    assert torch.count_nonzero(grad) > 0
    assert loss == pytest.approx((0.03 - 0.02) * 0.03, abs=1e-6)
    loss, _, _ = _order_grad(0.03, 0.00, dq=0.2)       # |dq| > m: the margin is m = 0.0556
    assert loss == pytest.approx((0.0556 - 0.03) * 0.2, abs=1e-5)
    loss, grad, _ = _order_grad(0.06, 0.00, dq=0.2)
    assert loss == 0.0
    assert torch.count_nonzero(grad) == 0
