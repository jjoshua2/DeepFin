"""simple_rl ``--value-rows-per-game r``: the WDL term on r rows per game.

The rows are chosen by a hash of (game_id, ply) -- the same in any row order,
on a resume and in every arm -- and every stored row keeps its policy term.
The WDL term is normalised per CHOSEN row over the phase (each chosen row's
weight is 1 / its share of the network-turn rows), so ``w_wdl`` keeps its
meaning, and r = every row is bit-identical to the unmasked loss.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl.learn import (
    ScheduledSampler,
    TrainSettings,
    build_trainer,
    refuse_masked_heads_under_accumulation,
    train_generation,
    value_row_choice,
    value_row_weights,
)
from tests.simple_rl_helpers import tiny_model_config
from tests.test_simple_rl_accum_steps import _arrays, _run


def test_the_choice_is_r_rows_per_game_and_independent_of_row_order() -> None:
    rng = np.random.default_rng(3)
    games = np.repeat(np.arange(40, dtype=np.int64) + 7000, rng.integers(1, 30, size=40))
    plies = np.concatenate([np.arange(int((games == g).sum())) for g in np.unique(games)]).astype(np.int64)
    for r in (1, 2, 5):
        chosen = value_row_choice(games, plies, r)
        for g in np.unique(games):
            assert int(chosen[games == g].sum()) == min(r, int((games == g).sum()))
        perm = rng.permutation(games.size)
        assert np.array_equal(value_row_choice(games[perm], plies[perm], r), chosen[perm])
    assert value_row_choice(games, plies, 10_000).all()


def _sampler(rows: dict[str, np.ndarray], idx: np.ndarray) -> ScheduledSampler:
    return ScheduledSampler(rows, [(idx, np.zeros(0, dtype=np.int64))], rng=np.random.default_rng(0))


def _settings(**kw: Any) -> TrainSettings:
    base: dict[str, Any] = {"device": "cpu", "batch_size": 32, "holdout_frac": 0.0, "seed": 7, "lr": 1e-3,
                            "mirror_prob": 0.0}
    return TrainSettings(**{**base, **kw})


def _with_weights(rows: dict[str, np.ndarray], chosen: np.ndarray) -> dict[str, np.ndarray]:
    out = dict(rows)
    f = float(chosen.mean())
    out["value_row_weight"] = np.where(chosen, 1.0 / f, 0.0).astype(np.float32)
    return out


def _head_grads(settings: TrainSettings, rows: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """d(loss)/d(head logits) per row, captured on the Trainer's own forward."""
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    got: dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Any, out: Any) -> None:
        for name in ("wdl", "policy_own", "policy"):
            t = out.get(name) if isinstance(out, dict) else None
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(lambda g, n=name: got.__setitem__(n, g.detach().clone()))

    handle = trainer.model.register_forward_hook(hook)
    n = int(rows["x"].shape[0])
    trainer.train_steps(_sampler(rows, np.arange(n)), batch_size=n, steps=1)
    handle.remove()
    return got


def test_wdl_gradient_is_zero_on_unchosen_rows_and_policy_stays_on_every_row() -> None:
    rows = _arrays(32, seed=1)
    chosen = np.zeros(32, dtype=bool)
    chosen[[1, 6, 9, 17, 30]] = True
    g = _head_grads(_settings(batch_size=32), _with_weights(rows, chosen))
    wdl = g["wdl"].abs().sum(dim=1)
    assert torch.all(wdl[torch.from_numpy(~chosen)] == 0)
    assert torch.all(wdl[torch.from_numpy(chosen)] > 0)
    pol = g.get("policy_own", g.get("policy"))
    assert pol is not None
    assert torch.all(pol.abs().sum(dim=1) > 0)  # policy on every stored row, chosen or not


def _param_grads(settings: TrainSettings, rows: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    got: dict[str, torch.Tensor] = {}

    def grab(_o: Any, _a: Any, _k: Any) -> None:
        got.update({n: p.grad.detach().clone() for n, p in trainer.model.named_parameters() if p.grad is not None})

    trainer.opt.register_step_pre_hook(grab)
    n = int(rows["x"].shape[0])
    trainer.train_steps(_sampler(rows, np.arange(n)), batch_size=int(settings.batch_size), steps=1)
    return got


def test_the_value_term_is_the_mean_over_the_chosen_rows_not_over_the_batch() -> None:
    """A batch that is the whole phase: f = k / n, so the masked value term must
    equal an unmasked step on the k chosen rows alone (value only)."""
    rows = _arrays(32, seed=2)
    chosen = np.zeros(32, dtype=bool)
    chosen[[0, 5, 11, 12, 20, 27, 31]] = True
    masked = _param_grads(_settings(batch_size=32, w_policy=0.0), _with_weights(rows, chosen))
    alone = _param_grads(_settings(batch_size=int(chosen.sum()), w_policy=0.0),
                         {k: (v[chosen] if not str(k).startswith("_") else v) for k, v in rows.items()})
    live = [n for n, t in alone.items() if float(t.abs().max()) > 1e-7]
    assert live
    for n in live:
        np.testing.assert_allclose(masked[n].numpy(), alone[n].numpy(), rtol=2e-4, atol=1e-7, err_msg=n)


def test_every_row_chosen_is_bit_identical_to_the_unmasked_loss() -> None:
    rows = _arrays(32, seed=4)
    plain = _param_grads(_settings(), rows)
    all_rows = _param_grads(_settings(), _with_weights(rows, np.ones(32, dtype=bool)))
    assert set(plain) == set(all_rows)
    for n in plain:
        assert torch.equal(plain[n], all_rows[n]), n


def test_train_generation_records_the_choice_and_trains_value_on_r_rows_per_game(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.train import trainer as trainer_mod

    seen: list[torch.Tensor] = []
    real = trainer_mod.compute_loss

    def spy(*args: Any, **kwargs: Any) -> Any:
        batch = kwargs["batch"] if "batch" in kwargs else next(a for a in args if isinstance(a, dict) and "wdl_t" in a)
        seen.append(batch["value_row_weight"].detach().cpu().clone())  # KeyError = the weight was dropped
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "compute_loss", spy)
    run = _run(tmp_path, 100)  # 25 games of 4 rows
    settings = _settings(batch_size=10, value_rows_per_game=1, mirror_prob=0.5)
    summary = train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                               prev_trainer_checkpoint=None, settings=settings)
    weights = torch.cat(seen[:10])  # the phase's 10 training batches (heldout/probe passes add no loss calls)
    assert int((weights > 0).sum()) == 25  # what the LOSS received: one value row per game
    assert torch.allclose(weights[weights > 0], torch.full((25,), 4.0))  # 1 / f, f = 25 / 100
    assert summary["value_rows"] == {"value_rows_per_game": 1, "value_rows": 25, "network_turn_rows": 100,
                                     "games": 25, "value_row_share": 0.25}
    manifest = json.loads((run / "gen_0001" / "train" / "manifest.json").read_text())
    assert manifest["config"]["settings"]["value_rows_per_game"] == 1


def test_accumulation_and_settings_refuse_what_the_mask_cannot_honour(tmp_path: Path) -> None:
    rows = _arrays(8)
    rows["value_row_weight"] = np.where(np.arange(8) % 4 == 0, 4.0, 0.0).astype(np.float32)
    with pytest.raises(ValueError, match="value_rows_per_game"):
        refuse_masked_heads_under_accumulation(rows, _settings(accum_steps=2, value_rows_per_game=1))
    with pytest.raises(ValueError, match="sl_mix_frac"):
        _settings(value_rows_per_game=2, sl_mix_frac=0.1, sl_mix_dir="x").validate()
    with pytest.raises(ValueError, match="w_wdl 0"):
        _settings(value_rows_per_game=2, w_wdl=0.0).validate()
    with pytest.raises(ValueError, match=">= 0"):
        _settings(value_rows_per_game=-1).validate()
    with pytest.raises(ValueError, match="accum_steps > 1"):  # at validation, before any collect runs
        _settings(value_rows_per_game=1, accum_steps=2).validate()
    run = _run(tmp_path, 40)
    with pytest.raises(ValueError, match="value_rows_per_game"):
        train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                         prev_trainer_checkpoint=None, settings=_settings(batch_size=4, accum_steps=2,
                                                                          value_rows_per_game=1))


def test_accumulation_is_refused_by_the_driver_before_anything_is_pinned_or_collected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "vra"
    with pytest.raises((SystemExit, ValueError), match="accum_steps > 1"):
        simple_rl_loop.main(_argv(run, openings, model_config, "--value-rows-per-game", "2", "--accum-steps", "2"))
    assert not (run / "run_config.json").exists()
    assert not (run / "gen_0001").exists()


def test_the_value_share_counts_network_turn_rows_only(tmp_path: Path) -> None:
    """f is the chosen share of NETWORK-TURN rows (the WDL term's own mask), not
    of all phase rows: rows off the network turn get weight 0 and are not in f's
    denominator."""
    run = _run(tmp_path, 100)  # 25 games of 4 rows, ply = row index
    row_ids = np.stack([np.ones(100, dtype=np.int64), np.arange(100, dtype=np.int64)], axis=1)
    net = np.ones(100, dtype=bool)
    net[::3] = False  # 34 rows off the network turn, 66 on it
    weight, info = value_row_weights(run, row_ids, net, 2)
    chosen = value_row_choice(np.arange(100) // 4 + 1000, np.arange(100), 2)
    k = int((chosen & net).sum())
    assert info["network_turn_rows"] == 66
    assert info["value_rows"] == k
    assert info["value_row_share"] == pytest.approx(k / 66)
    np.testing.assert_allclose(weight[chosen & net], 66 / k, rtol=1e-6)
    assert not weight[~net].any()
    assert not weight[~chosen].any()


def test_driver_flag_is_pinned_and_an_old_run_resumes_at_every_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "vr"
    argv = _argv(run, openings, model_config, "--value-rows-per-game", "2")
    assert simple_rl_loop.main(argv) == 0
    assert json.loads((run / "run_config.json").read_text())["settings"]["value_rows_per_game"] == 2
    summary = json.loads((run / "gen_0001" / "train" / "manifest.json").read_text())["summary"]
    assert summary["value_rows"]["value_rows_per_game"] == 2  # read off the train phase that ran
    more = list(argv)
    more[more.index("--generations") + 1] = "2"
    more[more.index("--value-rows-per-game") + 1] = "3"
    with pytest.raises(SystemExit, match="value_rows_per_game"):
        simple_rl_loop.main(more)
    old = tmp_path / "old"
    old_argv = _argv(old, openings, model_config)
    assert simple_rl_loop.main(old_argv) == 0
    cfg_path = old / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    del cfg["settings"]["value_rows_per_game"]  # a run pinned before the flag
    cfg_path.write_text(json.dumps(cfg))
    assert simple_rl_loop.main(old_argv) == 0  # the default reads as every row
