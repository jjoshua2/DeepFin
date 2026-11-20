"""simple_rl training phase: window sampling, loss switches, SL mix, tripwires."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays
from chess_anti_engine.simple_rl.collect import concat_arrays
from chess_anti_engine.simple_rl.evaluate import check_tripwires
from chess_anti_engine.simple_rl.learn import (
    TRAIN_KEYS,
    ReferenceAnchor,
    TrainSettings,
    WindowSampler,
    attach_anchor,
    build_trainer,
    is_holdout_game,
    load_window,
)
from tests.simple_rl_helpers import tiny_model, tiny_model_config


def _rows(n: int, *, marker: float, seed: int, game_base: int = 0) -> tuple[list[ReplaySample], list[int]]:
    """Rows whose x[0,0,0] carries ``marker`` (generation / source tag)."""
    rng = np.random.default_rng(seed)
    samples, games = [], []
    for i in range(n):
        legal = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
        support = rng.choice(COMPACT_POLICY_SIZE, size=12, replace=False)
        legal[support] = 1
        target = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float32)
        target[support] = rng.dirichlet(np.ones(12))
        x = (rng.standard_normal((175, 8, 8)) * 0.1).astype(np.float32)
        x[0, 0, 0] = marker
        samples.append(ReplaySample(x=x, policy_target=target, wdl_target=int(rng.integers(0, 3)),
                                    legal_mask=legal, is_network_turn=True, is_selfplay=True,
                                    search_wdl=rng.dirichlet(np.ones(3)).astype(np.float32)))
        games.append(game_base + i // 4)
    return samples, games


def _write_generation(run_dir: Path, g: int, n: int) -> None:
    shard_dir = run_dir / f"gen_{g:04d}" / "collect" / "shards"
    shard_dir.mkdir(parents=True)
    samples, games = _rows(n, marker=float(g), seed=g, game_base=1000 * g)
    save_local_shard_arrays(shard_dir / "shard_00000.zarr", arrs=samples_to_arrays(samples))
    np.savez(shard_dir / "shard_00000.meta.npz", game_id=np.asarray(games, dtype=np.int64),
             ply_index=np.arange(n, dtype=np.int32), opening_id=np.asarray(["o"] * n),
             tb_policy_override=np.zeros(n, dtype=bool), x_hash=np.arange(n, dtype=np.uint64))


def test_window_holds_exactly_the_last_k_generations(tmp_path: Path) -> None:
    for g in range(1, 6):
        _write_generation(tmp_path, g, 40 + g)
    settings = TrainSettings(window=3, holdout_frac=0.25, sample_with_replacement=True)
    train, holdout, gens, ids = load_window(tmp_path, 5, settings)
    assert sorted(set(ids[:, 0].tolist())) == [3, 4, 5]
    assert gens == [3, 4, 5]
    markers = train["x"][:, 0, 0, 0].astype(int)
    assert set(markers) == {3, 4, 5}
    assert set(holdout["x"][:, 0, 0, 0].astype(int)) == {5}
    def held(g: int) -> int:
        return sum(1 for i in range(40 + g) if is_holdout_game(1000 * g + i // 4, 0.25))

    assert held(5) > 0, "fixture should hold out at least one game"
    assert int(holdout["x"].shape[0]) == held(5)
    assert int(train["x"].shape[0]) == sum(40 + g - held(g) for g in (3, 4, 5))


def test_sampler_is_uniform_over_window_rows() -> None:
    samples = [samples_to_arrays(_rows(n, marker=float(g), seed=g)[0]) for g, n in ((1, 100), (2, 300))]
    rows = concat_arrays(samples)
    sampler = WindowSampler(rows, rng=np.random.default_rng(0))
    counts = {1: 0, 2: 0}
    for _ in range(200):
        batch = sampler.sample_batch_arrays(64)
        for m in batch["x"][:, 0, 0, 0].astype(int):
            counts[m] += 1
    share = counts[1] / (counts[1] + counts[2])
    assert abs(share - 0.25) < 0.02, share  # 100 of 400 rows
    assert sampler.drawn_rows == 200 * 64


def test_sl_mix_fraction_is_realised_in_the_batch_content() -> None:
    own = samples_to_arrays(_rows(50, marker=1.0, seed=1)[0])
    sl = samples_to_arrays(_rows(50, marker=9.0, seed=2)[0])
    sampler = WindowSampler(own, rng=np.random.default_rng(0), sl_rows=sl, sl_frac=0.25)
    for _ in range(5):
        batch = sampler.sample_batch_arrays(64)
        assert int((batch["x"][:, 0, 0, 0] == 9.0).sum()) == 16
    assert sampler.realised_sl_rows == 5 * 16
    with pytest.raises(ValueError, match="needs an SL corpus"):
        WindowSampler(own, rng=np.random.default_rng(0), sl_frac=0.25)


def _sampler(seed: int = 0) -> WindowSampler:
    rows = samples_to_arrays(_rows(64, marker=0.0, seed=seed)[0])
    rows = {k: v for k, v in rows.items() if k in TRAIN_KEYS or k.startswith("_")}
    return WindowSampler(rows, rng=np.random.default_rng(seed))


def _one_step(settings: TrainSettings, *, anchor: ReferenceAnchor | None = None) -> dict[str, torch.Tensor | None]:
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    if anchor is not None:
        attach_anchor(trainer, anchor)
    captured: dict[str, torch.Tensor | None] = {}

    def grab(_opt: Any, _args: Any, _kwargs: Any) -> None:
        for name, p in trainer.model.named_parameters():
            captured[name] = None if p.grad is None else p.grad.detach().clone()

    handle = trainer.opt.register_step_pre_hook(grab)
    try:
        trainer.train_steps(_sampler(), batch_size=16, steps=1)
    finally:
        handle.remove()
    assert captured, "optimizer step never ran"
    return captured


def _max_grad(grads: dict[str, torch.Tensor | None], prefix: str) -> float:
    vals = [float(g.abs().max()) for n, g in grads.items() if n.startswith(prefix) and g is not None]
    return max(vals) if vals else 0.0


def test_w_wdl_zero_leaves_value_head_gradients_at_zero() -> None:
    on = _one_step(TrainSettings(w_wdl=1.0, mirror_prob=0.0))
    off = _one_step(TrainSettings(w_wdl=0.0, mirror_prob=0.0))
    assert _max_grad(on, "value_wdl.") > 0.0
    assert _max_grad(off, "value_wdl.") == 0.0
    assert _max_grad(off, "policy_own.") > 0.0
    # every auxiliary head stays untrained
    for head in ("value_sf_eval.", "value_categorical.", "volatility.", "moves_left.", "policy_soft.", "policy_future."):
        assert _max_grad(on, head) == 0.0, head


def test_value_search_weight_reaches_the_loss() -> None:
    trainer = build_trainer(tiny_model_config(), TrainSettings(value_outcome_weight=1.0, value_search_weight=3.0),
                            log_dir=None)
    assert trainer._loss_kwargs["search_wdl_frac"] == pytest.approx(0.75)
    a = _one_step(TrainSettings(value_search_weight=0.0, mirror_prob=0.0))
    b = _one_step(TrainSettings(value_search_weight=1.0, value_outcome_weight=0.0, mirror_prob=0.0))
    names = [n for n in a if n.startswith("value_wdl.")]
    assert any(not torch.equal(cast(torch.Tensor, a[n]), cast(torch.Tensor, b[n])) for n in names
               if a[n] is not None and b[n] is not None)


def test_kl_anchor_gradients_are_nonzero_and_reach_the_consumer() -> None:
    # Policy anchor: with the policy loss off, policy_own learns ONLY through the anchor.
    value_only = TrainSettings(w_policy=0.0, w_wdl=1.0, mirror_prob=0.0)
    assert _max_grad(_one_step(value_only), "policy_own.") == 0.0
    pol = ReferenceAnchor(tiny_model(seed=7), policy_coef=1.0, wdl_coef=0.0)
    assert _max_grad(_one_step(value_only, anchor=pol), "policy_own.") > 0.0
    assert pol.diagnostics()["rows"] == 16
    assert pol.diagnostics()["mean_policy_kl"] > 0
    # WDL anchor: with w_wdl = 0, value_wdl learns ONLY through the anchor.
    policy_only = TrainSettings(w_policy=1.0, w_wdl=0.0, mirror_prob=0.0)
    assert _max_grad(_one_step(policy_only), "value_wdl.") == 0.0
    wdl = ReferenceAnchor(tiny_model(seed=7), policy_coef=0.0, wdl_coef=1.0)
    assert _max_grad(_one_step(policy_only, anchor=wdl), "value_wdl.") > 0.0
    assert wdl.diagnostics()["mean_wdl_kl"] > 0


def test_tripwire_refuses_to_run_unarmed_and_flags_a_breach() -> None:
    rows = [{"generation": 1, "policy_entropy": 1.0, "target_entropy": 0.6, "draw_rate": 0.5,
             "decisive_rate": 0.5, "repeat_rate_within": 0.18, "repeat_rate_vs_previous": 0.0,
             "kl_target_prior": 0.3}]
    with pytest.raises(ValueError, match="cannot fail"):
        check_tripwires(rows, {"max_repeat_rate": None})
    assert check_tripwires(rows, {"max_repeat_rate": 0.2}) == []
    breaches = check_tripwires(rows, {"max_repeat_rate": 0.1, "min_target_entropy": 0.5})
    assert len(breaches) == 1
    assert "repeat_rate_within" in breaches[0]
    assert check_tripwires([{**rows[0], "policy_entropy": None}], {"min_policy_entropy": 0.5})


def test_temperature_clock_counts_plies_from_the_start_position() -> None:
    from types import SimpleNamespace

    from chess_anti_engine.selfplay.config import TemperatureConfig
    from chess_anti_engine.selfplay.network_turn import _temperature_clock

    state = SimpleNamespace(cboards=[SimpleNamespace(ply=20)], starting_ply_arr=np.asarray([16]),
                            temp=TemperatureConfig())
    assert _temperature_clock(cast(Any, state), 0) == 11  # legacy: fullmove number
    state.temp = TemperatureConfig(schedule_plies_from_start=True)
    assert _temperature_clock(cast(Any, state), 0) == 4


def test_sl_corpus_without_what_the_loss_needs_is_refused(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import load_sl_corpus

    good, _ = _rows(10, marker=9.0, seed=4)
    save_local_shard_arrays(tmp_path / "good" / "a.zarr", arrs=samples_to_arrays(good))
    out = load_sl_corpus(tmp_path / "good", max_rows=100, seed=0, need_search=True)
    assert int(out["x"].shape[0]) == 10
    no_search = [ReplaySample(x=s.x, policy_target=s.policy_target, wdl_target=s.wdl_target,
                              legal_mask=s.legal_mask) for s in good]
    save_local_shard_arrays(tmp_path / "nosearch" / "a.zarr", arrs=samples_to_arrays(no_search))
    assert int(load_sl_corpus(tmp_path / "nosearch", max_rows=100, seed=0, need_search=False)["x"].shape[0]) == 10
    with pytest.raises(ValueError, match="search_wdl"):
        load_sl_corpus(tmp_path / "nosearch", max_rows=100, seed=0, need_search=True)
    no_legal = [ReplaySample(x=s.x, policy_target=s.policy_target, wdl_target=s.wdl_target) for s in good]
    save_local_shard_arrays(tmp_path / "nolegal" / "a.zarr", arrs=samples_to_arrays(no_legal))
    with pytest.raises(ValueError, match="legal"):
        load_sl_corpus(tmp_path / "nolegal", max_rows=100, seed=0, need_search=False)


def test_search_value_wdl_keeps_q_and_a_valid_distribution() -> None:
    from chess_anti_engine.simple_rl.learn import search_value_wdl

    q = np.array([0.5, -0.9, 0.0, 1.0])
    d = np.array([0.3, 0.3, 0.6, 0.2])
    wdl = search_value_wdl(q, d)
    np.testing.assert_allclose(wdl[:, 0] - wdl[:, 2], q, atol=1e-12)
    np.testing.assert_allclose(wdl.sum(axis=1), 1.0, atol=1e-12)
    assert (wdl >= 0).all()
    np.testing.assert_allclose(wdl[:, 1], [0.3, 0.1, 0.6, 0.0])  # draw clipped to 1 - |q|


def _rows_with_root(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = samples_to_arrays(_rows(n, marker=0.0, seed=seed)[0])
    support = rows["legal_mask"] > 0
    rows["root_survivor_q"] = rng.uniform(-0.8, 0.8, n).astype(np.float32)
    rows["has_root_survivor_q"] = np.ones(n, np.uint8)
    rows["root_survivor_q"][0] = np.nan  # survivor outside the searched support
    rows["has_root_survivor_q"][1] = 0  # e.g. a solved root
    rows["root_net_wdl"] = rng.dirichlet(np.ones(3), n).astype(np.float32)
    rows["has_root_net_wdl"] = np.ones(n, np.uint8)
    rows["root_log_prior"] = np.where(support, rng.normal(size=support.shape), 0).astype(np.float32)
    rows["has_root_log_prior"] = np.ones(n, np.uint8)
    rows["root_child_visits"] = np.where(support, 1.0, -1.0).astype(np.float32)
    return rows


def test_value_target_is_the_survivor_q_by_default_with_the_chosen_draw_source() -> None:
    from chess_anti_engine.simple_rl.learn import prepare_targets, search_value_wdl

    rows = _rows_with_root(20, seed=5)
    stored = rows["search_wdl"].copy()
    settings = TrainSettings(value_search_weight=1.0)
    assert settings.value_search_source == "survivor_q"
    out, info = prepare_targets(rows, settings)
    q, d = rows["root_survivor_q"], rows["root_net_wdl"][:, 1]
    np.testing.assert_allclose(out["search_wdl"][2:], search_value_wdl(q[2:], d[2:]), atol=1e-6)
    np.testing.assert_array_equal(out["search_wdl"][:2], stored[:2])  # fallback rows keep the stored field
    assert info["value_from_survivor_q"] == 18
    assert info["value_fallback_stored_search_wdl"] == 2
    assert "root_survivor_q" not in out  # source fields never reach the batch
    zero, _ = prepare_targets(rows, TrainSettings(value_search_weight=1.0, value_draw_source="zero"))
    np.testing.assert_allclose(zero["search_wdl"][2:, 1], 0.0)
    legacy, info2 = prepare_targets(rows, TrainSettings(value_search_weight=1.0,
                                                        value_search_source="stored_search_wdl"))
    np.testing.assert_array_equal(legacy["search_wdl"], stored)
    assert "value_from_survivor_q" not in info2
    off, _ = prepare_targets(rows, TrainSettings(value_search_weight=0.0))
    np.testing.assert_array_equal(off["search_wdl"], stored)


def test_prior_mix_from_the_actor_prior_and_from_a_reference_net() -> None:
    from chess_anti_engine.simple_rl.learn import prepare_targets

    rows = _rows_with_root(12, seed=6)
    target = rows["policy_target"].astype(np.float64)
    legal = rows["legal_mask"] > 0
    out, info = prepare_targets(rows, TrainSettings(policy_prior_mix=0.75, prior_mix_source="actor"))
    lp = np.where(legal, rows["root_log_prior"].astype(np.float64), -np.inf)
    prior = np.exp(lp - lp.max(axis=1, keepdims=True))
    prior /= prior.sum(axis=1, keepdims=True)
    t = target / target.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out["policy_target"], 0.75 * t + 0.25 * prior, atol=1e-6)
    assert info["prior_mixed_rows"] == 12
    uniform = np.where(legal, 1.0, 0.0)
    calls: list[int] = []

    def reference(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        calls.append(len(x))
        return np.where(mask, 1.0, 0.0)

    ref, _ = prepare_targets(rows, TrainSettings(policy_prior_mix=0.5), reference)
    np.testing.assert_allclose(ref["policy_target"], 0.5 * t + 0.5 * uniform / uniform.sum(axis=1, keepdims=True),
                               atol=1e-6)
    assert calls == [12]
    with pytest.raises(ValueError, match="reference"):
        prepare_targets(rows, TrainSettings(policy_prior_mix=0.5))
    same, _ = prepare_targets(rows, TrainSettings())
    np.testing.assert_array_equal(same["policy_target"], rows["policy_target"])


def test_default_schedule_window_follows_the_presentation_count(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import phase_generations

    assert phase_generations(5, TrainSettings()) == [5]
    assert phase_generations(5, TrainSettings(presentations=2)) == [4, 5]
    assert phase_generations(1, TrainSettings(presentations=2)) == [1]
    for g in range(1, 4):
        _write_generation(tmp_path, g, 30)
    _, _, gens, ids = load_window(tmp_path, 3, TrainSettings(presentations=2, holdout_frac=0.0))
    assert gens == [2, 3]
    assert ids.shape[0] == 60
    with pytest.raises(ValueError, match="cap is 2"):
        TrainSettings(presentations=3).validate()
    with pytest.raises(ValueError, match="sample_with_replacement"):
        TrainSettings(window=4).validate()
    with pytest.raises(ValueError, match="sample_with_replacement"):
        TrainSettings(updates=10).validate()
    TrainSettings(window=4, updates=10, sample_with_replacement=True).validate()


def test_plan_batches_presents_every_row_exactly_once_with_a_short_last_batch() -> None:
    from chess_anti_engine.simple_rl.learn import plan_batches

    batches, sl_end = plan_batches(203, batch_size=64, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(1))
    sizes = [b[0].size for b in batches]
    assert sizes == [64, 64, 64, 11]  # remainder served as a short final batch, not dropped
    served = np.concatenate([b[0] for b in batches])
    np.testing.assert_array_equal(np.sort(served), np.arange(203))
    assert sl_end == 0
    # SL share: 16 per full batch, proportional in the short one; consecutive, never reused.
    batches, sl_end = plan_batches(100, batch_size=64, sl_frac=0.25, sl_available=1000,
                                   rng=np.random.default_rng(2), sl_start=40)
    assert [b[0].size for b in batches] == [48, 48, 4]
    assert [b[1].size for b in batches] == [16, 16, 1]
    sl = np.concatenate([b[1] for b in batches])
    np.testing.assert_array_equal(sl, np.arange(40, 73))
    assert sl_end == 73
    with pytest.raises(ValueError, match="exhausted"):
        plan_batches(100, batch_size=64, sl_frac=0.25, sl_available=50, rng=np.random.default_rng(2), sl_start=40)


def _scheduled(n: int, batches: list[np.ndarray], seed: int = 0) -> Any:
    from chess_anti_engine.simple_rl.learn import ScheduledSampler

    rows = samples_to_arrays(_rows(n, marker=0.0, seed=seed)[0])
    rows = {k: v for k, v in rows.items() if k in TRAIN_KEYS or k.startswith("_")}
    return ScheduledSampler(rows, [(b, np.zeros(0, np.int64)) for b in batches], rng=np.random.default_rng(seed))


def test_a_nonfinite_gradient_fails_the_phase_instead_of_consuming_a_presentation() -> None:
    """Review lead L1: the Trainer skips a non-finite step but still advances;
    a scheduled row would then count as presented with no update."""
    trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu", mirror_prob=0.0), log_dir=None)
    with torch.no_grad():
        next(iter(trainer.model.parameters())).fill_(float("nan"))
    sampler = _scheduled(32, [np.arange(16), np.arange(16, 32)])
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_steps(sampler, batch_size=16, steps=2)


def test_a_short_final_batch_keeps_the_per_row_coefficient_of_a_full_batch() -> None:
    """Review lead L2: every row's gradient coefficient is 1/batch_size, so the
    4-row tail contributes 4/16 of the gradient its row-mean would."""
    def grads(sampler: Any) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        trainer = build_trainer(tiny_model_config(), TrainSettings(device="cpu", mirror_prob=0.0), log_dir=None)
        got: dict[str, torch.Tensor] = {}

        def grab(_o: Any, _a: Any, _k: Any) -> None:
            for name, p in trainer.model.named_parameters():
                if p.grad is not None:
                    got[name] = p.grad.detach().clone()

        trainer.opt.register_step_pre_hook(grab)
        trainer.train_steps(sampler, batch_size=16, steps=1)
        return got

    class _PlainFour:
        """The same 4 rows with no exact-presentation marker: a plain row mean."""

        def __init__(self, inner: Any) -> None:
            self.inner, self.rng = inner, inner.rng

        def __len__(self) -> int:
            return 4

        def sample_batch_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
            return self.inner.sample_batch_arrays(batch_size)

    tail = grads(_scheduled(20, [np.arange(16, 20)]))
    plain = grads(_PlainFour(_scheduled(20, [np.arange(16, 20)])))
    names = [n for n, g in plain.items() if float(g.abs().max()) > 1e-6]
    assert names
    for name in names:
        np.testing.assert_allclose(tail[name].numpy(), plain[name].numpy() * (4 / 16), rtol=1e-4, atol=1e-7)


def test_continue_refuses_or_reapplies_a_changed_optimizer_hparam(tmp_path: Path) -> None:
    """Review item 2: exact resume restores the checkpoint's lr / floor, so a
    requested --lr was silently ignored. Now: refused, or re-applied with
    --reapply-optimizer-hparams and read back off the param groups after a step."""
    from chess_anti_engine.simple_rl.learn import continue_optimizer, realised_optimizer

    first = build_trainer(tiny_model_config(), TrainSettings(device="cpu", lr=1e-3, mirror_prob=0.0), log_dir=None)
    first.train_steps(_scheduled(16, [np.arange(16)]), batch_size=16, steps=1)
    ckpt = tmp_path / "gen1.pt"
    first.save(ckpt)

    same = build_trainer(tiny_model_config(), TrainSettings(device="cpu", lr=1e-3), log_dir=None)
    assert continue_optimizer(same, ckpt, TrainSettings(device="cpu", lr=1e-3)) == {"differed": {}, "reapplied": False}

    lower = TrainSettings(device="cpu", lr=5e-5, lr_floor=1e-6, mirror_prob=0.0)
    with pytest.raises(ValueError, match="reapply-optimizer-hparams"):
        continue_optimizer(build_trainer(tiny_model_config(), lower, log_dir=None), ckpt, lower)

    reapply = TrainSettings(device="cpu", lr=5e-5, lr_floor=1e-6, mirror_prob=0.0, reapply_optimizer_hparams=True)
    trainer = build_trainer(tiny_model_config(), reapply, log_dir=None)
    check = continue_optimizer(trainer, ckpt, reapply)
    assert check["reapplied"]
    assert set(check["differed"]) == {"peak_lr", "lr_floor"}
    step_before = int(trainer.step)
    trainer.train_steps(_scheduled(16, [np.arange(16)]), batch_size=16, steps=1)
    assert int(trainer.step) == step_before + 1  # the moments and step count did continue
    got = realised_optimizer(trainer)
    assert got["peak_lr"] == pytest.approx(5e-5)
    assert got["lr_floor"] == pytest.approx(1e-6)
    assert got["param_group_lr"] == pytest.approx([5e-5] * len(got["param_group_lr"]), rel=1e-3)


def test_plan_batches_order_is_a_seeded_non_identity_permutation() -> None:
    from chess_anti_engine.simple_rl.common import seed_for
    from chess_anti_engine.simple_rl.learn import plan_batches

    def order(generation: int) -> np.ndarray:
        rng = np.random.default_rng(seed_for(7, "train", generation))
        batches, _ = plan_batches(500, batch_size=64, sl_frac=0.0, sl_available=0, rng=rng)
        return np.concatenate([b[0] for b in batches])

    a, b = order(3), order(4)
    np.testing.assert_array_equal(np.sort(a), np.arange(500))
    assert not np.array_equal(a, np.arange(500))
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, order(3))  # same seed, same order


def test_heldout_policy_metrics_skip_value_only_rows() -> None:
    """Review item 6: a fast playout-cap row has no policy target; its zeroed
    target must not enter the held-out policy CE / entropy."""
    from chess_anti_engine.simple_rl.learn import heldout_metrics

    rows = {k: v for k, v in samples_to_arrays(_rows(12, marker=0.0, seed=3)[0]).items() if not k.startswith("_")}
    model = tiny_model(seed=1).eval()
    base = heldout_metrics(model, rows, "cpu")
    poisoned = {k: v.copy() for k, v in rows.items()}
    poisoned["has_policy"] = np.ones(12, dtype=np.uint8)
    poisoned["has_policy"][6:] = 0
    poisoned["policy_target"][6:] = 0
    poisoned["policy_target"][6:, 0] = 1.0  # what a value-only row's target slot might hold
    got = heldout_metrics(model, poisoned, "cpu")
    head = heldout_metrics(model, {k: v[:6] for k, v in rows.items() if not k.startswith("_")}, "cpu")
    assert got["policy_rows"] == 6
    assert got["rows"] == 12
    assert got["policy_ce"] == pytest.approx(head["policy_ce"])
    assert got["policy_entropy"] == pytest.approx(head["policy_entropy"])
    assert got["value_nll"] == pytest.approx(base["value_nll"])
