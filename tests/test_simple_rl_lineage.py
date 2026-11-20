"""Unit tests for simple_rl stage lineage (chess_anti_engine/simple_rl/lineage.py),
the sha-cache seeding at manifest write, and generation 1's optional start from
the init checkpoint's optimizer state."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import lineage
from chess_anti_engine.simple_rl.common import MANIFEST, input_ref, sha256_tree, write_manifest
from chess_anti_engine.simple_rl.evaluate import LEDGER
from tests.simple_rl_helpers import tiny_model_config


def _seal(run_dir: Path, g: int, stage: str, inputs: dict[str, Path] | None = None) -> Path:
    d = lineage.stage_dir(run_dir, g, stage)
    d.mkdir(parents=True, exist_ok=True)
    (d / "payload.bin").write_bytes(f"{g}{stage}".encode())
    refs = {k: {"path": str(p), "sha256": sha256_tree(p)} for k, p in (inputs or {}).items()}
    (d / MANIFEST).write_text(json.dumps({"stage": stage, "inputs": refs, "outputs": {}}))
    return d


def test_two_redos_in_the_same_utc_second_get_distinct_superseded_dirs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    monkeypatch.setattr(lineage.time, "gmtime", lambda *_a: lineage.time.struct_time((2026, 11, 20, 1, 2, 3, 4, 324, 0)))
    first, second = [], []
    for out in (first, second):
        _seal(run, 0, "init")
        _seal(run, 1, "collect")
        _seal(run, 1, "train")
        res = lineage.unseal_after(run, 1, "collect", include_self=False)
        out.append(res["moved_to"])
    assert first[0] != second[0]
    assert Path(first[0]).name == "20261120T010203Z"
    assert Path(second[0]).name == "20261120T010203Z-1"
    for moved in (first[0], second[0]):
        assert (Path(moved) / "gen_0001" / "train" / MANIFEST).is_file()  # both redos' stages kept


def test_sha_cache_rehashes_a_file_rewritten_in_place_with_the_same_size(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    f = run / "model.pt"
    f.write_bytes(b"aaaa")
    cache = lineage._ShaCache(run)
    first = cache.sha(f)
    ino = f.stat().st_ino
    with f.open("r+b") as fh:  # same inode, same size, new bytes
        fh.write(b"bbbb")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert f.stat().st_ino == ino
    assert f.stat().st_size == 4
    assert cache.sha(f) == sha256_tree(f) != first


def test_unseal_rewinds_the_arena_ledger_even_when_no_evaluation_reruns(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _seal(run, 0, "init")
    for g in (1, 2):
        for s in ("collect", "train", "eval"):
            _seal(run, g, s)
    (run / LEDGER).write_text("".join(json.dumps({"generation": g, "match": "vs_prev"}) + "\n" for g in (1, 2)))
    lineage.unseal_after(run, 1, "train", include_self=True)  # a --no-eval run would never rewrite gen 1/2 rows
    rows = [json.loads(x) for x in (run / LEDGER).read_text().splitlines() if x.strip()]
    assert rows == []
    _seal(run, 1, "train")
    _seal(run, 1, "eval")
    (run / LEDGER).write_text(json.dumps({"generation": 1}) + "\n" + json.dumps({"generation": 7}) + "\n")
    lineage.unseal_after(run, 1, "train", include_self=False)
    rows = [json.loads(x) for x in (run / LEDGER).read_text().splitlines() if x.strip()]
    assert rows == [{"generation": 7}]  # only the unsealed evaluation's rows go


def test_verify_stage_refuses_a_missing_upstream_input(tmp_path: Path) -> None:
    run = tmp_path / "run"
    init = _seal(run, 0, "init")
    _seal(run, 1, "collect", inputs={"checkpoint": init / "payload.bin"})
    lineage.verify_before(run, 1, "collect", inclusive=True)  # intact: accepted
    (init / "payload.bin").unlink()
    with pytest.raises(SystemExit, match="no longer match their upstream"):
        lineage.verify_before(run, 1, "collect", inclusive=True)  # what a standalone complete-stage skip runs


def test_manifest_write_seeds_the_sha_cache_and_a_later_change_is_rehashed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stage = run / "gen_0001" / "train"
    stage.mkdir(parents=True)
    model = stage / "model.pt"
    model.write_bytes(b"weights-v1")
    outside = tmp_path / "book.jsonl"
    outside.write_text("line\n")
    write_manifest(stage, stage="train", config={}, inputs={"book": input_ref(outside)},
                   outputs={"model": input_ref(model)}, counts={}, seeds={})
    cache = json.loads((run / lineage.SHA_CACHE).read_text())
    key = str(model.resolve())
    assert cache[key]["sha256"] == sha256_tree(model)
    st = model.stat()
    assert cache[key]["stamp"] == [st.st_size, st.st_mtime_ns, st.st_ino]

    calls: list[Path] = []
    real = lineage.sha256_tree

    def counting(p: Path) -> str:
        calls.append(Path(p))
        return real(p)

    lineage.sha256_tree = counting  # type: ignore[assignment]
    try:
        warm = lineage._ShaCache(run)
        assert warm.sha(model) == cache[key]["sha256"]
        assert calls == []  # served from the seeded entry: no re-read
        model.write_bytes(b"weights-v2!")  # size and mtime move
        assert warm.sha(model) == real(model)
        assert calls == [model]
    finally:
        lineage.sha256_tree = real  # type: ignore[assignment]


def test_seeding_skips_a_file_that_changed_after_it_was_hashed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stage = run / "gen_0001" / "train"
    stage.mkdir(parents=True)
    model = stage / "model.pt"
    model.write_bytes(b"one")
    ref = input_ref(model)
    model.write_bytes(b"three")  # rewritten between hashing and the manifest
    write_manifest(stage, stage="train", config={}, inputs={}, outputs={"model": ref}, counts={}, seeds={})
    cache_path = run / lineage.SHA_CACHE
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    assert str(model.resolve()) not in cache


# --------------------------------------------------------------------------
# --init-optimizer-state checkpoint (generation 1)
# --------------------------------------------------------------------------

def _trained_checkpoint(path: Path, settings) -> tuple[Path, dict[str, torch.Tensor]]:
    """A trainer checkpoint with non-empty optimizer moments (two real steps)."""
    from chess_anti_engine.simple_rl.learn import build_trainer

    torch.manual_seed(3)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    for p in trainer.model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    trainer.opt.step()
    trainer.opt.step()
    trainer.save(path)
    first = next(iter(trainer.opt.state.values()))
    return path, {k: v.clone() for k, v in first.items() if torch.is_tensor(v)}


def test_init_optimizer_state_checkpoint_restores_moments_and_keeps_the_requested_lr(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, init_optimizer_from_checkpoint

    saved = TrainSettings(device="cpu", lr=7e-4)
    ckpt, moments = _trained_checkpoint(tmp_path / "init.pt", saved)
    want = TrainSettings(device="cpu", lr=3e-6, init_optimizer_state="checkpoint")
    trainer = build_trainer(tiny_model_config(), want, log_dir=None)
    info = init_optimizer_from_checkpoint(trainer, ckpt, want)
    restored = next(iter(trainer.opt.state.values()))
    for k, v in moments.items():
        assert torch.equal(restored[k], v)  # the consumer holds the checkpoint's moments
    assert [float(g["lr"]) for g in trainer.opt.param_groups] == [3e-6, 3e-6]  # read off the groups
    assert info["checkpoint_param_group_lr"] == [7e-4, 7e-4]
    assert info["state_step_min"] == info["state_step_max"] == 2
    assert trainer.step == 0


def test_init_optimizer_state_checkpoint_refuses_missing_state_and_foreign_layouts(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, init_optimizer_from_checkpoint
    from tests.simple_rl_helpers import save_tiny_checkpoint

    want = TrainSettings(device="cpu", lr=3e-6, init_optimizer_state="checkpoint")
    empty = save_tiny_checkpoint(tmp_path / "fresh.pt")  # trainer checkpoint with an empty optimizer state
    with pytest.raises(ValueError, match="has no optimizer state"):
        init_optimizer_from_checkpoint(build_trainer(tiny_model_config(), want, log_dir=None), empty, want)
    payload = torch.load(empty, map_location="cpu", weights_only=False)
    del payload["opt"]
    torch.save(payload, tmp_path / "noopt.pt")
    with pytest.raises(ValueError, match="has no optimizer state"):
        init_optimizer_from_checkpoint(build_trainer(tiny_model_config(), want, log_dir=None), tmp_path / "noopt.pt", want)

    aurora, _ = _trained_checkpoint(tmp_path / "aurora.pt", TrainSettings(device="cpu", lr=3e-6, optimizer="aurora"))
    with pytest.raises(ValueError, match="written by aurora, not the requested --optimizer nadamw"):
        init_optimizer_from_checkpoint(build_trainer(tiny_model_config(), want, log_dir=None), aurora, want)
    # Same optimizer, different matrix scope: the parameter layout check.
    scoped = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora", init_optimizer_state="checkpoint",
                           matrix_optimizer_scope="mlp_out")
    with pytest.raises(ValueError, match="optimizer parameter layout"):
        init_optimizer_from_checkpoint(build_trainer(tiny_model_config(), scoped, log_dir=None), aurora, scoped)


def test_init_optimizer_state_checkpoint_refuses_differing_hparams_unless_reapplied(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, init_optimizer_from_checkpoint

    ckpt, _ = _trained_checkpoint(tmp_path / "init.pt", TrainSettings(device="cpu", lr=3e-6))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    for g in payload["opt"]["param_groups"]:
        g["weight_decay"] = 0.5
    torch.save(payload, ckpt)
    want = TrainSettings(device="cpu", lr=3e-6, init_optimizer_state="checkpoint")
    with pytest.raises(ValueError, match="hyperparameters differ"):
        init_optimizer_from_checkpoint(build_trainer(tiny_model_config(), want, log_dir=None), ckpt, want)
    again = TrainSettings(device="cpu", lr=3e-6, init_optimizer_state="checkpoint", reapply_optimizer_hparams=True)
    trainer = build_trainer(tiny_model_config(), again, log_dir=None)
    info = init_optimizer_from_checkpoint(trainer, ckpt, again)
    assert info["reapplied"]
    assert [float(g["weight_decay"]) for g in trainer.opt.param_groups] == [1e-4, 0.0]
    assert np.isfinite(info["state_step_max"])


def test_init_optimizer_state_reaches_generation_one_through_the_cli(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag is pinned, wired through train_settings, consumed by
    train_generation at generation 1, and recorded off the live optimizer."""
    from chess_anti_engine.simple_rl.learn import TrainSettings
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _argv, _setup

    openings, model_config = _setup(tmp_path, monkeypatch)
    argv = _argv(tmp_path / "run_random", openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    with pytest.raises(SystemExit, match="random init has none"):  # refused before pinning or collecting
        simple_rl_loop.main([*argv, "--init-optimizer-state", "checkpoint"])
    assert not (tmp_path / "run_random" / "run_config.json").exists()
    assert not (tmp_path / "run_random" / "gen_0001").exists()

    ckpt, _ = _trained_checkpoint(tmp_path / "init.pt", TrainSettings(device="cpu", lr=1e-3))
    argv = _argv(tmp_path / "run", openings, model_config)
    argv[argv.index("--generations") + 1] = "2"  # gen 2 must NOT restore the init moments (review F4)
    argv[argv.index("--optimizer-state") + 1] = "fresh"
    i = argv.index("--random-init")
    argv[i:i + 1] = ["--init-checkpoint", str(ckpt)]
    del argv[argv.index("--model-config"):argv.index("--model-config") + 2]
    assert simple_rl_loop.main([*argv, "--init-optimizer-state", "checkpoint"]) == 0
    summary = json.loads((tmp_path / "run" / "gen_0001" / "train" / "metrics.json").read_text())["summary"]
    init = summary["optimizer_from_init_checkpoint"]
    assert init is not None
    assert init["state_step_min"] == 2  # moments came from the checkpoint
    assert summary["realised_lr_param_groups"] == [1e-3, 1e-3]
    pinned = json.loads((tmp_path / "run" / "run_config.json").read_text())["settings"]
    assert pinned["init_optimizer_state"] == "checkpoint"
    gen2 = json.loads((tmp_path / "run" / "gen_0002" / "train" / "metrics.json").read_text())["summary"]
    assert gen2["optimizer_from_init_checkpoint"] is None
    assert gen2["optimizer_continued"] is False  # --optimizer-state fresh: a fresh optimizer at gen 2


def _aurora_knobs_differ(tmp_path: Path) -> tuple[Path, dict]:
    """An Aurora checkpoint whose polar knobs are NOT this run's configured ones."""
    from chess_anti_engine.simple_rl.learn import TrainSettings

    ckpt, _ = _trained_checkpoint(tmp_path / "aurora.pt", TrainSettings(device="cpu", lr=3e-6, optimizer="aurora"))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    knobs = {"aurora_polar_steps": 8, "aurora_polar_method": "polar_express", "aurora_pp_beta": 0.25}
    for g in payload["opt"]["param_groups"]:
        assert "aurora_polar_steps" in g
        g.update(knobs)
    torch.save(payload, ckpt)
    return ckpt, knobs


def test_init_optimizer_refuses_any_group_key_difference_and_names_both_values(tmp_path: Path) -> None:
    """Review F1: load_state_dict replaces EVERY group key; the old diff saw
    only weight_decay/betas/eps/momentum_decay, so Aurora polar knobs changed
    silently. Every differing key is now listed with both values."""
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, init_optimizer_from_checkpoint

    ckpt, knobs = _aurora_knobs_differ(tmp_path)
    want = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora", init_optimizer_state="checkpoint")
    trainer = build_trainer(tiny_model_config(), want, log_dir=None)
    configured = {k: trainer.opt.param_groups[0][k] for k in knobs}
    assert configured != knobs  # the fixture really differs from the configured defaults
    with pytest.raises(ValueError, match="param-group hyperparameters differ") as exc:
        init_optimizer_from_checkpoint(trainer, ckpt, want)
    msg = str(exc.value)
    for k, v in knobs.items():
        if configured[k] != v:
            assert f"{k}: configured {configured[k]!r} vs checkpoint {v!r}" in msg


def test_init_optimizer_reapply_uses_configured_and_adopt_keeps_the_checkpoints(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl.learn import (
        TrainSettings, build_trainer, init_optimizer_from_checkpoint, realised_optimizer,
    )

    ckpt, knobs = _aurora_knobs_differ(tmp_path)
    reapply = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora", init_optimizer_state="checkpoint",
                            reapply_optimizer_hparams=True)
    t1 = build_trainer(tiny_model_config(), reapply, log_dir=None)
    configured = {k: t1.opt.param_groups[0][k] for k in knobs}
    info = init_optimizer_from_checkpoint(t1, ckpt, reapply)
    assert info["group_resolution"] == "configured"
    assert info["reapplied"]
    assert {k: t1.opt.param_groups[0][k] for k in knobs} == configured  # read off the consumer
    assert {d["key"] for d in info["group_differed"]} >= {k for k in knobs if configured[k] != knobs[k]}

    adopt = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora", init_optimizer_state="checkpoint",
                          adopt_checkpoint_optimizer_hparams=True)
    t2 = build_trainer(tiny_model_config(), adopt, log_dir=None)
    info = init_optimizer_from_checkpoint(t2, ckpt, adopt)
    assert info["group_resolution"] == "checkpoint"
    assert not info["reapplied"]
    for g in t2.opt.param_groups:
        assert {k: g[k] for k in knobs} == knobs
    realised = realised_optimizer(t2)["groups"]
    assert all(r[k] == v for r in realised for k, v in knobs.items())  # recorded in metrics/manifest
    assert info["group_hparams_realised"] == realised


def test_init_optimizer_initial_lr_is_the_schedule_peak_not_the_warmup_start(tmp_path: Path) -> None:
    """Review N1."""
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, init_optimizer_from_checkpoint

    ckpt, _ = _trained_checkpoint(tmp_path / "init.pt", TrainSettings(device="cpu", lr=3e-6))
    want = TrainSettings(device="cpu", lr=1e-3, warmup_steps=10, warmup_start=1e-5, init_optimizer_state="checkpoint")
    trainer = build_trainer(tiny_model_config(), want, log_dir=None)
    init_optimizer_from_checkpoint(trainer, ckpt, want)
    peaks = [float(v) for v in trainer._scheduler.base_lrs]
    assert [float(g["initial_lr"]) for g in trainer.opt.param_groups if "initial_lr" in g] in ([], peaks)
    assert all(float(g["lr"]) < p for g, p in zip(trainer.opt.param_groups, peaks, strict=True))


@pytest.mark.parametrize(("kw", "match"), [
    ({"matrix_optimizer_scope": "bogus", "optimizer": "aurora"}, "matrix_optimizer_scope must be one of"),
    ({"matrix_optimizer_scope": "mlp_out"}, "would be ignored"),  # nadamw does not read it
    ({"reapply_optimizer_hparams": True, "adopt_checkpoint_optimizer_hparams": True}, "mutually exclusive"),
])
def test_train_settings_refuse_ignored_optimizer_knobs(kw: dict, match: str) -> None:
    """Review F2."""
    from chess_anti_engine.simple_rl.learn import TrainSettings

    with pytest.raises(ValueError, match=match):
        TrainSettings(device="cpu", **kw).validate()
    TrainSettings(device="cpu", optimizer="aurora", matrix_optimizer_scope="mlp_out").validate()


def test_the_matrix_scope_flag_is_a_closed_choice() -> None:
    from scripts import simple_rl_loop

    with pytest.raises(SystemExit):
        simple_rl_loop.parser().parse_args(["train", "--run-dir", "x", "--openings-dir", "x", "--generation", "1",
                                            "--matrix-optimizer-scope", "bogus"])


def test_init_optimizer_source_is_refused_before_pinning(tmp_path: Path) -> None:
    """Review F3: no optimizer state / another optimizer's state is refused by
    validate_before_pinning, from the header, before run_config.json exists."""
    from chess_anti_engine.simple_rl.learn import TrainSettings
    from scripts import simple_rl_loop as loop
    from tests.simple_rl_helpers import save_tiny_checkpoint

    empty = save_tiny_checkpoint(tmp_path / "fresh.pt")
    nadam, _ = _trained_checkpoint(tmp_path / "nadam.pt", TrainSettings(device="cpu"))
    base = ["run", "--run-dir", str(tmp_path / "run"), "--openings-dir", str(tmp_path), "--generations", "1",
            "--init-optimizer-state", "checkpoint"]
    cases = [([*base, "--init-checkpoint", str(empty)], "has no optimizer state"),
             ([*base, "--init-checkpoint", str(nadam), "--optimizer", "aurora"], "written by nadamw"),
             ([*base, "--random-init"], "random init has none")]
    for argv, match in cases:
        a = loop.parser().parse_args(argv)
        with pytest.raises(SystemExit, match=match):
            loop.validate_before_pinning(a, tmp_path / "run", {"collect", "train"})
    assert not (tmp_path / "run").exists()
    ok = loop.parser().parse_args([*base, "--init-checkpoint", str(nadam)])
    loop.validate_before_pinning(ok, tmp_path / "run", {"collect", "train"})


def test_a_pinned_run_without_the_new_optimizer_keys_resumes_at_their_defaults(tmp_path: Path) -> None:
    """Review F5: the keys added since pinning default on resume; a non-default
    value is a config change."""
    from scripts import simple_rl_loop as loop

    run = tmp_path / "run"
    run.mkdir()
    argv = ["run", "--run-dir", str(run), "--openings-dir", str(tmp_path), "--generations", "1", "--random-init"]
    a = loop.parser().parse_args(argv)
    loop.check_run_config(a, run)
    stored = json.loads((run / "run_config.json").read_text())
    for k in ("init_optimizer_state", "matrix_optimizer_scope", "adopt_checkpoint_optimizer_hparams"):
        stored["settings"].pop(k)
    (run / "run_config.json").write_text(json.dumps(stored))
    loop.check_run_config(a, run)  # old pin, defaults now: accepted
    for extra in (["--init-optimizer-state", "checkpoint"], ["--matrix-optimizer-scope", "mlp_out"],
                  ["--adopt-checkpoint-optimizer-hparams"]):
        with pytest.raises(SystemExit, match="run settings differ"):
            loop.check_run_config(loop.parser().parse_args([*argv, *extra]), run)


def test_continue_follows_the_same_group_rule(tmp_path: Path) -> None:
    """A run that adopted the checkpoint's knobs keeps them under continue;
    without a choice the continue restore is refused like the init one."""
    from chess_anti_engine.simple_rl.learn import TrainSettings, build_trainer, continue_optimizer

    ckpt, knobs = _aurora_knobs_differ(tmp_path)
    plain = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora")
    with pytest.raises(ValueError, match="param-group hyperparameters differ"):
        continue_optimizer(build_trainer(tiny_model_config(), plain, log_dir=None), ckpt, plain)
    adopt = TrainSettings(device="cpu", lr=3e-6, optimizer="aurora", adopt_checkpoint_optimizer_hparams=True)
    trainer = build_trainer(tiny_model_config(), adopt, log_dir=None)
    check = continue_optimizer(trainer, ckpt, adopt)
    assert check["group_resolution"] == "checkpoint"
    assert all(g[k] == v for g in trainer.opt.param_groups for k, v in knobs.items())
