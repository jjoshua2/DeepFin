"""The sibling term's ORDER form, version 2: the gap weight, and the root-list unit (``--sibling-unit roots``).

Covered:
- the gap weight min(|dq|, 0.62): a mis-ordered large gap costs more than a small one, in proportion, bounded;
- root lists: every pair among a root's children with |dq| >= t, each root's mean, roots combined with weight
  n_eff = n / (1 + (n - 1) * 0.13) normalised over the step's roots;
- the chunked list term equals a single-forward reference written independently here, through the real Trainer,
  at chunk budgets of one root, a few roots and all roots, over a short last batch;
- the list materialiser on a tiny real collect: every root's list is exactly its eligible children, byte-equal to
  an independent rebuild, and every pair-sidecar pair lies inside its row's list;
- the unit reaches the term inside train_generation, trains the value head, and is recorded with the child
  positions forwarded per step (the equal-budget check);
- refusals: a sidecar of the other unit, roots with the diff form, a version-1 order snapshot; the resume key
  carries the order form's version.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import sibling as sib
from chess_anti_engine.simple_rl.learn import ScheduledSampler, TrainSettings, attach_anchor, build_trainer, plan_batches
from tests.simple_rl_helpers import tiny_model_config
from tests.test_simple_rl_accum_steps import _arrays
from tests.test_simple_rl_sibling import _fresh_run, _independent_children, _settings
from tests.test_simple_rl_sibling import run as run  # the module-scoped collect fixture

N_ROWS, BATCH, ROOTS, W, T, M = 37, 16, 5, 3.0, 0.05, 0.08   # batches of 16, 16, 5


def test_the_gap_weight_scales_a_mis_ordered_pair_by_its_gap_bounded_at_0_62() -> None:
    dq = torch.tensor([0.1, 0.5, 1.5, -0.3])
    dv = torch.zeros(4)                       # the net calls every pair a tie: rank = min(m, |dq|) = m
    wv = torch.tensor([1.0, 1.0, 1.0, 0.5])
    got = sib.order_pair_loss(dv, dq, wv, margin=M, min_dq=T)
    want = torch.tensor([0.1 * M, 0.5 * M, 0.62 * M, 0.5 * 0.3 * M])
    torch.testing.assert_close(got, want)
    assert sib.ORDER_GAP_CAP == 0.62
    assert sib.ORDER_FORM_VERSION == 2


def test_root_pairs_and_weights() -> None:
    q = np.array([0.30, 0.31, 0.10, -0.20], np.float32)
    ia, ib = sib.root_pairs(q, 0.05)
    assert sorted(zip(ia.tolist(), ib.tolist(), strict=True)) == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert sib.root_weight(0) == 0.0
    assert sib.root_weight(1) == 1.0
    assert sib.root_weight(24) == pytest.approx(24 / (1 + 23 * 0.13))
    assert sib.ROOT_PAIR_ICC == 0.13


class FakeLists:
    """The SidecarLists interface: root r has 2 + (r mod 5) children with fixed random
    planes, q and visits; every row carries a list."""

    def __init__(self, n_rows: int, planes: int, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        self.row_count = (2 + np.arange(n_rows) % 5).astype(np.int64)
        self.row_start = np.r_[0, np.cumsum(self.row_count)[:-1]].astype(np.int64)
        n = int(self.row_count.sum())
        self._x = (rng.random((n, planes, 8, 8)) < 0.1).astype(np.float32)
        self.q = rng.uniform(-0.6, 0.6, n).astype(np.float32)
        self.visits = rng.integers(2, 40, n).astype(np.float32)
        self.probe_roots: list[tuple[int, int]] = []

    def child_inputs(self, ids: np.ndarray) -> np.ndarray:
        return self._x[ids]


class ReferenceListTerm(sib.SiblingListTerm):
    """ONE grad-enabled forward over every child of the step's roots; the loss written
    out pair by pair: per root, the mean over pairs (|dq| >= t) of
    (min v / 64) * min(|dq|, 0.62) * relu(min(m, |dq|) - sign(dq) * dv); roots
    weighted n / (1 + (n - 1) 0.13), normalised; returned for the Trainer to scale."""

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        del outputs, batch
        rows = self._batch_roots()
        self.steps += 1
        spans = [(int(self.lists.row_start[r]), int(self.lists.row_count[r])) for r in rows]
        ids = np.concatenate([np.arange(st, st + c) for st, c in spans]).astype(np.int64)
        v = sib.child_values(self.model(torch.from_numpy(self.lists.child_inputs(ids)).float())["wdl"])
        num: Any = 0.0
        den, n_all, o = 0.0, 0, 0
        for st, c in spans:
            terms = []
            for i in range(c):
                for j in range(i + 1, c):
                    dq = float(self.lists.q[st + i] - self.lists.q[st + j])
                    if abs(dq) < T:
                        continue
                    wv = min(float(self.lists.visits[st + i]), float(self.lists.visits[st + j])) / 64.0
                    dv = v[o + i] - v[o + j]
                    terms.append(wv * min(abs(dq), 0.62) * torch.relu(min(M, abs(dq)) - float(np.sign(dq)) * dv))
            o += c
            if terms:
                n = len(terms)
                a = n / (1 + (n - 1) * 0.13)
                num = num + a * torch.stack(terms).mean()
                den += a
                n_all += n
        if not n_all:
            self.per_step.append([0, 0.0])
            return None
        loss = num / den
        self.pairs_seen += n_all
        self.loss_sum += float(loss.detach())
        self.per_step.append([n_all, float(loss.detach())])
        return self.w * loss


def _plan() -> list[tuple[np.ndarray, np.ndarray]]:
    batches, _ = plan_batches(N_ROWS, batch_size=BATCH, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(5))
    return batches


def _run(term_cls: type, *, chunk: int, w_policy: float = 1.0, w_wdl: float = 1.0) -> dict[str, Any]:
    rows = _arrays(N_ROWS)
    settings = TrainSettings(device="cpu", mirror_prob=0.0, batch_size=BATCH, w_policy=w_policy, w_wdl=w_wdl)
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    sampler = ScheduledSampler(rows, _plan(), rng=np.random.default_rng(0))
    lists: Any = FakeLists(N_ROWS, int(rows["x"].shape[1]))
    term = term_cls(trainer.model, sampler, lists, w=W, rows=ROOTS, chunk_pairs=chunk, batch_size=BATCH,
                    device="cpu", loss="order", order_margin=M, order_min_dq=T)
    attach_anchor(trainer, term)
    grads: list[dict[str, torch.Tensor]] = []
    trainer.opt.register_step_pre_hook(lambda _o, _a, _k: grads.append(
        {n: p.grad.detach().clone() for n, p in trainer.model.named_parameters() if p.grad is not None}))
    trainer.train_steps(sampler, batch_size=BATCH, steps=3)
    return {"grads": grads, "summary": term.summary(), "state": term.state(),
            "weights": {n: p.detach().clone() for n, p in trainer.model.named_parameters()}}


def _assert_equal(got: dict[str, Any], ref: dict[str, Any]) -> None:
    assert len(got["grads"]) == len(ref["grads"]) == 3
    for ga, gb in zip(got["grads"], ref["grads"], strict=True):
        live = [n for n, g in gb.items() if float(g.abs().max()) > 1e-7]
        assert live
        for n in live:
            np.testing.assert_allclose(ga[n].numpy(), gb[n].numpy(), rtol=2e-4, atol=1e-7, err_msg=n)
    for n, p in ref["weights"].items():
        np.testing.assert_allclose(got["weights"][n].numpy(), p.numpy(), rtol=2e-4, atol=1e-6, err_msg=n)
    assert [r[0] for r in got["summary"]["per_step"]] == [r[0] for r in ref["summary"]["per_step"]]
    np.testing.assert_allclose([r[1] for r in got["summary"]["per_step"]],
                               [r[1] for r in ref["summary"]["per_step"]], rtol=1e-5)


@pytest.mark.parametrize("chunk", [1, 4, 100])   # budget 2 / 8 / 200 child positions: one root, a few, all
def test_the_chunked_list_term_equals_the_single_forward_reference(chunk: int) -> None:
    got, ref = _run(sib.SiblingListTerm, chunk=chunk), _run(ReferenceListTerm, chunk=chunk)
    _assert_equal(got, ref)
    assert got["summary"]["unit"] == "roots"
    assert got["summary"]["roots"] > 0
    assert got["summary"]["active_pairs"] > 0


def test_the_list_term_alone_trains_and_the_comparison_is_not_vacuous() -> None:
    _assert_equal(_run(sib.SiblingListTerm, chunk=4, w_policy=0.0, w_wdl=0.0),
                  _run(ReferenceListTerm, chunk=4, w_policy=0.0, w_wdl=0.0))

    class Unweighted(ReferenceListTerm):   # a global mean over pairs instead of the n_eff-weighted root means
        def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
            out = super().penalty(outputs, batch)
            return None if out is None else out * 1.37

    with pytest.raises(AssertionError):
        _assert_equal(_run(Unweighted, chunk=4), _run(ReferenceListTerm, chunk=4))


def test_the_list_term_records_its_child_positions_per_step() -> None:
    s = _run(sib.SiblingListTerm, chunk=4)["summary"]
    lists = FakeLists(N_ROWS, int(_arrays(N_ROWS)["x"].shape[1]))   # the same draws as _run
    served = ScheduledSampler(_arrays(N_ROWS), _plan(), rng=np.random.default_rng(0))
    want = 0
    for _ in range(3):
        served.sample_batch_arrays(BATCH)
        for r in np.asarray(served.served[-1])[:ROOTS]:   # a root with no pair >= t is not forwarded
            st, c = int(lists.row_start[r]), int(lists.row_count[r])
            if sib.root_pairs(lists.q[st:st + c], T)[0].size:
                want += c
    assert s["child_positions"] == want
    assert s["child_positions_per_step"] == pytest.approx(want / 3)


def test_a_version_1_order_snapshot_and_another_unit_are_refused() -> None:
    state = _run(sib.SiblingListTerm, chunk=4)["state"]
    assert state["order_form_version"] == 2
    assert state["unit"] == "roots"
    from types import SimpleNamespace

    fake: Any = FakeLists(2, 1)
    other: Any = sib.SiblingListTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0), fake, w=1.0,
                                     rows=1, chunk_pairs=4, batch_size=4, device="cpu", loss="order",
                                     order_margin=M, order_min_dq=T)
    with pytest.raises(RuntimeError, match="version"):
        other.restore({**state, "order_form_version": 1})
    with pytest.raises(RuntimeError, match="unit"):
        other.restore({**state, "unit": "pairs"})
    with pytest.raises(ValueError, match="order form only"):
        sib.SiblingListTerm(torch.nn.Linear(1, 1), SimpleNamespace(cursor=0), fake, w=1.0,
                            rows=1, chunk_pairs=4, batch_size=4, device="cpu", loss="diff")


def test_the_resume_key_carries_the_order_form_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.simple_rl import learn

    (tmp_path / "init.pt").write_bytes(b"i")
    (tmp_path / "probe.npz").write_bytes(b"p")
    (tmp_path / "gen_0001" / "collect").mkdir(parents=True)
    (tmp_path / "gen_0001" / "collect" / "manifest.json").write_text("{}")
    side = tmp_path / "side" / "gen_0001"
    side.mkdir(parents=True)
    (side / sib.SIDECAR_MANIFEST).write_text("{}")
    s = TrainSettings(device="cpu", w_sibling=1.0, sibling_dir=str(tmp_path / "side"), sibling_rows=8,
                      sibling_loss="order", sibling_order_margin=M, sibling_order_min_dq=T)

    def key() -> str:
        return learn.snapshot_resume_key(settings=s, generation=1, init_checkpoint=tmp_path / "init.pt",
                                         prev_trainer_checkpoint=None, run_dir=tmp_path, gens=[1],
                                         probe_path=tmp_path / "probe.npz",
                                         batches=[(np.arange(4), np.zeros(0, np.int64))])

    now = key()
    monkeypatch.setattr(sib, "ORDER_FORM_VERSION", 1)
    assert key() != now


# ---------------------------------------------------------------------------
# On the tiny real collect
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lists_side(run: dict[str, Path]) -> Path:
    side = run["tmp"] / "lists_side"
    sib.materialise(run["collect"], side / "gen_0001", min_visits=2, holdout_frac=0.25, log=lambda _m: None,
                    unit="roots")
    return side


def test_every_root_list_is_its_eligible_children_byte_equal_to_a_rebuild(run: dict[str, Path],
                                                                           lists_side: Path) -> None:
    man = json.loads((lists_side / "gen_0001" / sib.SIDECAR_MANIFEST).read_text())
    pair_man = json.loads((run["side"] / "gen_0001" / sib.SIDECAR_MANIFEST).read_text())
    assert (man["kind"], man["version"]) == (sib.LIST_KIND, sib.LIST_SIDECAR_VERSION)
    assert man["complete"]
    assert man["totals"]["children"] > 20
    assert man["totals"]["rows_with_pairs"] == pair_man["totals"]["rows_with_pairs"]
    checked = 0
    for name in man["shards"]:
        d = lists_side / "gen_0001" / name.replace(".zarr", "")
        z = {k: np.load(d / f"{k}.npy") for k in sib.LIST_FIELDS}
        pz = {k: np.load(run["side"] / "gen_0001" / name.replace(".zarr", "") / f"{k}.npy")
              for k in ("row", "action_a", "action_b", "q_a", "q_b")}
        assert (z["visits"] >= 2).all()
        for r in np.unique(z["row"]):
            sel = np.flatnonzero(z["row"] == r)
            assert sel.size >= 2
            acts = z["action"][sel]
            assert (np.diff(acts) > 0).all()            # each eligible child once, in id order
            for a, b, qa, qb in zip(pz["action_a"][pz["row"] == r], pz["action_b"][pz["row"] == r],
                                    pz["q_a"][pz["row"] == r], pz["q_b"][pz["row"] == r], strict=True):
                assert a in acts
                assert b in acts                          # every sampled pair lies inside its row's list
                assert z["q"][sel][acts == a][0] == qa
                assert z["q"][sel][acts == b][0] == qb
        for j in range(len(z["row"])):
            (cx,) = _independent_children(run["collect"], int(z["game_id"][j]), int(z["ply_index"][j]),
                                          (int(z["action"][j]),))
            assert cx.tobytes() == z["x"][j].tobytes()
            checked += 1
    assert checked == man["totals"]["children"]


def test_a_sidecar_of_the_other_unit_is_refused(run: dict[str, Path], lists_side: Path) -> None:
    rows = np.zeros((0, 2), np.int64)
    with pytest.raises(ValueError, match="simple_rl_sibling_lists"):
        sib.load_sidecars(run["run"], [1], rows, run["side"], holdout_frac=0.25, newest=1, unit="roots")
    with pytest.raises(ValueError, match="simple_rl_sibling_pairs"):
        sib.load_sidecars(run["run"], [1], rows, lists_side, holdout_frac=0.25, newest=1, unit="pairs")
    with pytest.raises(ValueError, match="order form only"):
        _settings(run, w_sibling=1.0, sibling_dir=str(lists_side), sibling_rows=8, sibling_unit="roots").validate()


def test_the_root_unit_trains_the_value_head_inside_train_generation(run: dict[str, Path],
                                                                     lists_side: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.simple_rl.learn import train_generation

    built: list[str] = []
    real_init = sib.SiblingListTerm.__init__

    def spy(self: Any, *a: Any, **k: Any) -> None:
        built.append(type(self).__name__)
        real_init(self, *a, **k)

    monkeypatch.setattr(sib.SiblingListTerm, "__init__", spy)
    d = _fresh_run(run, "roots_unit")
    init = torch.load(run["ckpt"], map_location="cpu", weights_only=False)["model"]
    s = train_generation(run_dir=d, generation=1, init_checkpoint=run["ckpt"], prev_trainer_checkpoint=None,
                         settings=_settings(run, w_sibling=1.0, sibling_rows=8, sibling_dir=str(lists_side),
                                            sibling_unit="roots", sibling_loss="order", sibling_order_margin=0.05,
                                            sibling_order_min_dq=0.001, w_policy=0.0, w_wdl=0.0))
    assert built == ["SiblingListTerm"]
    sb = s["sibling"]
    assert sb["unit"] == "roots"
    assert sb["roots"] > 0
    assert sb["pairs"] > 0
    assert sb["active_pairs"] > 0
    assert sb["child_positions_per_step"] > 0
    assert sb["order_form_version"] == 2
    after = torch.load(d / "gen_0001" / "train" / "model.pt", map_location="cpu", weights_only=False)["model"]
    moved = max(float((after[k].float() - init[k].float()).abs().max()) for k in init if k.startswith("value_wdl."))
    assert moved > 0
    man = json.loads((d / "gen_0001" / "train" / "manifest.json").read_text())["config"]["settings"]
    assert man["sibling_unit"] == "roots"
    assert sb["probe_before"]["sign_agree"] is not None or sb["probe_before"]["roots"] == 0


def test_a_root_with_one_eligible_child_writes_no_list(run: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """With two of every three candidates excluded (a stub terminal rule), some roots
    keep exactly one eligible child: they carry no pair, so no list is written."""
    calls = {"n": 0}

    def every_third(_board: Any) -> str | None:
        calls["n"] += 1
        return None if calls["n"] % 3 == 0 else "game_over"

    monkeypatch.setattr(sib, "search_terminal", every_third)
    out = run["tmp"] / "lists_one_child"
    man = sib.materialise(run["collect"], out / "gen_0001", min_visits=2, holdout_frac=0.25, log=lambda _m: None,
                          unit="roots")
    assert sum(r["counts"]["rows_with_one_child"] for r in man["shards"].values()) > 0   # non-vacuous
    for name in man["shards"]:
        rows = np.load(out / "gen_0001" / name.replace(".zarr", "") / "row.npy")
        _, n = np.unique(rows, return_counts=True)
        assert (n >= 2).all()
