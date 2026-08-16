"""The three lc0-control DRIVERS, at their ``main()`` entry points.

⚑⚑ WHY THIS FILE EXISTS. The library modules under it were well tested and the
WIRING that decides whether any of them takes effect was not: a reviewer
applied TWO mutants at once — deleting the realized-guard call from
``lc0_control_train`` outright, and breaking ``_LossCapture.worst`` to
``readouts[0]`` so a leak starting late is never seen — and all 36 tests
still passed. This codebase's signature defect is a value that is accepted and
then silently ignored, and an entry point with no test is where that lives.

Everything here runs on CPU at a toy width. The architecture is deliberately
NOT production's: these tests are about control flow, and
``tests/test_lc0_control_config.py`` plus ``tests/test_lc0_control_arch.py``
own the "is it the right net" question.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from chess_anti_engine.eval import lc0_control_arch
from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.train.value_blend_guard import (
    ValueBlendMisconfigured,
    value_blend_readout,
)
from scripts import lc0_control_eval, lc0_control_heldout, lc0_control_train

REPO = Path(__file__).resolve().parent.parent
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"
PLANES = 175


def _tiny_config(tmp_path: Path, **overrides: Any) -> Path:
    """The control config at toy width, with the value blend left alone.

    Built by LOADING the shipped config and shrinking it, not by writing a new
    one: every trainer knob under test — the optimizer, the loss weights, the
    blend — stays whatever production's copy says, so a rename in the real file
    breaks this test instead of drifting away from it.
    """
    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw["model"].update(embed_dim=32, num_layers=2, num_heads=2, ffn_mult=1.5)
    raw["model"].pop("ffn_mult_by_layer", None)
    raw["train"]["batch_size"] = 4
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    path = tmp_path / "tiny_control.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _lc0_like_sample(
    seed: int, *, with_sf_wdl: bool, with_search_wdl: bool = True,
) -> ReplaySample:
    """One converted-lc0-looking row: a search WDL, and no SF label.

    ⚑ ``with_search_wdl=False`` is review F1's corpus: `compute_loss` falls the
    SEARCH component back to the raw one-hot exactly as it falls the SF
    component back, and `--shards` is a CLI argument like any other.
    """
    rng = np.random.default_rng(seed)
    policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
    policy /= policy.sum()
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
    legal[rng.choice(COMPACT_POLICY_SIZE, size=20, replace=False)] = 1
    search = rng.random(3).astype(np.float32)
    search /= search.sum()
    sample = ReplaySample(
        x=rng.random((PLANES, 8, 8)).astype(np.float32),
        policy_target=policy,
        wdl_target=int(seed % 3),
        legal_mask=legal,
        search_wdl=search if with_search_wdl else None,
        moves_left=0.1,
        has_policy=True,
        is_selfplay=True,
        is_network_turn=True,
        game_id=seed,
        ply_index=0,
    )
    if with_sf_wdl:
        sf = rng.random(3).astype(np.float32)
        sample.sf_wdl = sf / sf.sum()
    return sample


def _write_shards(
    shard_dir: Path, seeds: list[int], *, with_sf_wdl: bool = False,
    with_search_wdl: bool = True,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        _lc0_like_sample(s, with_sf_wdl=with_sf_wdl, with_search_wdl=with_search_wdl)
        for s in seeds
    ]
    save_local_shard_arrays(
        shard_dir / "shard_000000.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            positions=len(samples),
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=True,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
        ),
    )
    return shard_dir


# ── lc0_control_train: the realized guard, and whether it is REACHED ──────────


def test_the_control_run_completes_and_writes_a_summary(tmp_path: Path) -> None:
    """Baseline. Without this the failure tests below cannot be told from a
    driver that always exits non-zero."""
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    out = tmp_path / "run"
    rc = lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
    ])
    assert rc == 0
    assert (out / "checkpoint.pt").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    realized = summary["realized"]
    assert realized["leaked_to_outcome"] == 0.0
    assert realized["outcome_borne_frac (game_frac + leak)"] == pytest.approx(0.30)
    assert realized["search_wdl_frac (realized)"] == pytest.approx(0.70)
    assert summary["valid_control"] is False, "--allow-arch-drift must be recorded"


def test_the_realized_guard_fails_the_run_and_writes_no_checkpoint(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE HEADLINE CHECK, PROVEN REACHABLE.

    It previously could not fail from any input: launch guard 1 refuses every
    ``sf_wdl_frac > 0`` config, nothing downstream can raise the frac, and
    ``--allow-leak`` — the only way past guard 1 — skipped this assert too. The
    flag now opens only the LAUNCH gates, so this is the state that fires it:
    production's blend, pointed at rows that carry no SF label.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.5, "search_wdl_frac": 0.2})
    out = tmp_path / "leak_run"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift", "--allow-leak",
        ])
    assert "RAW GAME OUTCOME" in str(excinfo.value)
    assert not (out / "checkpoint.pt").exists(), (
        "a run that failed the realized guard must leave no checkpoint behind"
    )


def test_the_trainer_publishes_the_effective_label_mass_and_warns_on_a_leak(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑⚑ REVIEW F3 — THE PR'S ONLY PRODUCTION-PATH BEHAVIOUR CHANGE, TESTED.

    ``TrainMetrics.sf_wdl_effective_frac`` / ``search_wdl_effective_frac`` and
    ``Trainer._warn_if_value_blend_leaks_to_outcome`` shipped with ZERO tests:
    deleting both the ``_RATIO_METRIC_FIELDS`` entries and the call left 81
    tests green, and a column publishing a constant 0.0 would make the
    TB-derived leak read as a FULL leak on production forever.

    Both directions, because they kill different mutants:

    * SF-labelled rows -> the column reads 1.0 and the trainer stays SILENT.
      Deleting the ratio-map entry leaves the field at its 0.0 default, which
      reads as "nothing was labelled" and fires the warning spuriously.
    * unlabelled rows -> the trainer WARNS. Deleting the call in
      ``train_steps`` silences it while every other column stays right.

    Driven through the real trainer, not by calling the method: the finding is
    about wiring, and a direct call cannot see a missing call site.
    """
    labelled = _write_shards(tmp_path / "sf_rows", list(range(16)), with_sf_wdl=True)
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.5, "search_wdl_frac": 0.2})
    out = tmp_path / "labelled_run"
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        rc = lc0_control_train.main([
            "--config", str(config), "--shards", str(labelled),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift", "--allow-leak",
        ])
    assert rc == 0
    metrics = json.loads((out / "summary.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["sf_wdl_effective_frac"] == pytest.approx(1.0), (
        "the column must reach TrainMetrics; at its 0.0 default the TB-derived "
        "leak reads as a full leak on production"
    )
    assert metrics["search_wdl_effective_frac"] == pytest.approx(1.0)
    assert not [r for r in caplog.records if "RAW GAME OUTCOME" in r.getMessage()], (
        "every row carries both labels — nothing leaked"
    )

    caplog.clear()
    unlabelled = _write_shards(tmp_path / "lc0_rows", list(range(16)))
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"), \
            pytest.raises(SystemExit):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(unlabelled),
            "--out-dir", str(tmp_path / "leak_warn"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
            "--allow-arch-drift", "--allow-leak",
        ])
    warned = [r.getMessage() for r in caplog.records if "RAW GAME OUTCOME" in r.getMessage()]
    assert warned, "the production-path warning did not fire on a real leak"
    assert "0.5000 of the WDL target" in warned[0]
    assert "sf_wdl_frac leaked 0.5000" in warned[0]


def test_launch_refuses_a_production_blend_without_allow_leak(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.5, "search_wdl_frac": 0.2})
    with pytest.raises(SystemExit, match="REFUSING TO LAUNCH"):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "x"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
        ])


def test_an_all_outcome_value_target_is_refused(tmp_path: Path) -> None:
    """⚑ Review F6 / Codex #2. ``sf=0`` and ``search=0`` leaks NOTHING and
    trains 100% of the value target on the raw one-hot outcome — the old gate
    passed it, printing ``leaked_to_outcome 0.00``."""
    lc0_rows = _write_shards(tmp_path / "lc0", list(range(8)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.0, "search_wdl_frac": 0.0})
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(lc0_rows),
            "--out-dir", str(tmp_path / "y"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
        ])
    assert "collapses onto the raw game outcome" in str(excinfo.value)


def test_the_realized_guard_also_catches_the_all_outcome_target(tmp_path: Path) -> None:
    """⚑ The ``outcome_borne_frac`` bar, REACHED THROUGH THE DRIVER.

    Launch guard 2 refuses ``sf=0 / search=0`` first, so the realized bar is a
    second line of defence and needs the launch gates open to be exercised at
    all. That is exactly what ``--allow-leak`` is for now — and this run leaks
    NOTHING (``leaked_to_outcome`` is 0.00, because ``sf_wdl_frac`` is 0) while
    training 100% of the value target on the raw outcome. The old guard passed
    it.
    """
    shards = _write_shards(tmp_path / "rows", list(range(16)))
    config = _tiny_config(tmp_path, train={"sf_wdl_frac": 0.0, "search_wdl_frac": 0.0})
    out = tmp_path / "all_outcome"
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift", "--allow-leak",
        ])
    message = str(excinfo.value)
    assert "1.0000 of its mass on the RAW GAME OUTCOME" in message
    assert "leaked=0.0000" in message, (
        "the leak is genuinely zero here — that is why the leak bar could not "
        "see this state"
    )
    assert not (out / "checkpoint.pt").exists()


def test_a_corpus_with_no_search_label_is_refused_at_launch(tmp_path: Path) -> None:
    """⚑⚑ REVIEW F1 AT THE DRIVER — the LARGER term, on the launch path.

    The shipped control is `sf_wdl_frac: 0.0` / `search_wdl_frac: 0.70`, so on
    rows that carry no `has_search_wdl` **the whole value target is the raw
    game outcome** — and every guard passed, because the corpus preflight
    measured only the SF label. Today's converted corpus reads
    `has_search_wdl 204800/204800`, so nothing produced so far is wrong; a
    mixed `--shards` list or a v5 conversion lands here.
    """
    rows = _write_shards(tmp_path / "no_search", list(range(8)), with_search_wdl=False)
    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)), "--shards", str(rows),
            "--out-dir", str(tmp_path / "ns"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
        ])
    message = str(excinfo.value)
    assert "NO search value label" in message
    assert "search_wdl label coverage" not in message, "that line is a print, not the refusal"


def test_a_partially_labelled_corpus_is_refused(tmp_path: Path) -> None:
    """⚑ Codex #2 at its own level: ``any()`` called a mixed corpus labelled.

    One SF-labelled row anywhere in ``--shards`` used to set ``have_sf=True``,
    which made the converter's gate return ``[]`` and waved the whole corpus
    through. Coverage is 2/10 here, and neither regime's reasoning applies.
    """
    lc0_rows = _write_shards(tmp_path / "lc0", list(range(8)))
    sf_rows = _write_shards(tmp_path / "sf", [100, 101], with_sf_wdl=True)
    with pytest.raises(SystemExit, match="PARTIALLY sf_wdl-labelled"):
        lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)),
            "--shards", str(lc0_rows), str(sf_rows),
            "--out-dir", str(tmp_path / "z"), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
        ])


def test_the_realized_guard_is_judged_on_the_worst_step_not_the_first() -> None:
    """⚑ The reviewer's second mutant: ``worst`` -> ``readouts[0]``.

    A leak that begins partway through a run — a live edit, a controller push —
    is invisible to the first step and diluted to nothing by a mean over 800.
    """
    capture = lc0_control_train._LossCapture()
    capture.readouts = [
        value_blend_readout(sf_wdl_frac=0.0, search_wdl_frac=0.7,
                            sf_effective_rows=0.0, search_effective_rows=512.0,
                            batch_rows=512.0),
        value_blend_readout(sf_wdl_frac=0.5, search_wdl_frac=0.2,
                            sf_effective_rows=0.0, search_effective_rows=512.0,
                            batch_rows=512.0),
    ]
    assert capture.worst.leaked_to_outcome == pytest.approx(0.5)


def test_the_capture_records_the_kwargs_compute_loss_was_called_with() -> None:
    """A configured value is not an applied value — this is the applied one."""
    capture = lc0_control_train._LossCapture()
    capture.observe(
        {"sf_wdl_frac": 0.5, "search_wdl_frac": 0.2},
        {
            "sf_wdl_effective_rows": torch.tensor(0.0),
            "search_wdl_effective_rows": torch.tensor(8.0),
            "batch_rows": torch.tensor(8.0),
        },
    )
    assert capture.calls == 1
    assert capture.worst.sf_wdl_frac == pytest.approx(0.5)
    assert capture.worst.leaked_to_outcome == pytest.approx(0.5)


def test_the_architecture_guard_refuses_a_drifted_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard 0 must be the reason the run stops, before any GPU time is spent.

    ⚑ THE DRIFTED CONTROL IS NOW CONSTRUCTED, NOT ASSUMED. Until 2026-08-16
    this passed the SHIPPED config in and expected a refusal, because the
    shipped config was drifted — it predated the bt4heads promotion and could
    not carry keys that were not yet in `main`'s schema. That premise expired
    when PR #439 merged and the config was re-pointed, and a test that reads
    "the guard refuses" while actually asserting "the config is broken" stops
    testing the guard the moment the config is fixed.

    So: the shipped config PASSES (guard 0 is not a constant refusal), a copy
    with one `model:` key moved is REFUSED and names that key, and
    `allow_drift` downgrades the refusal to a banner.
    """
    monkeypatch.delenv("CHESS_LIVE_PRODUCTION_CONFIG", raising=False)
    assert not lc0_control_train.preflight_architecture(
        CONTROL, allow_drift=False,
    ).startswith("DRIFTED"), "the shipped control config must pass guard 0"

    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw["model"]["num_layers"] = int(raw["model"]["num_layers"]) + 1
    drifted = tmp_path / "drifted_control.yaml"
    drifted.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        lc0_control_train.preflight_architecture(drifted, allow_drift=False)
    assert "architecture is NOT production's" in str(excinfo.value)
    assert "num_layers" in str(excinfo.value)
    assert lc0_control_train.preflight_architecture(
        drifted, allow_drift=True,
    ).startswith("DRIFTED")


def test_the_driver_is_what_reads_the_live_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REVIEW F6, at the level where the env var is now honoured.

    ``assert_control_matches_live_architecture`` no longer reads the
    environment — the DRIVER resolves it and passes the path in. That moved a
    behaviour, so it needs a test at the new location, or "the check prefers
    the live file" becomes a claim about deleted code.

    Both directions: a live file the control matches PASSES and names the live
    file, and one it does not REFUSES.
    """
    config = _tiny_config(tmp_path)
    control_model = yaml.safe_load(config.read_text(encoding="utf-8"))["model"]

    live = tmp_path / "live_pbt2_small.yaml"
    live.write_text(yaml.safe_dump({"model": dict(control_model)}), encoding="utf-8")
    monkeypatch.setenv("CHESS_LIVE_PRODUCTION_CONFIG", str(live))
    provenance = lc0_control_train.preflight_architecture(config, allow_drift=False)
    assert str(live) in provenance, "the guard did not read the file the env named"

    moved = dict(control_model)
    moved["num_layers"] = int(moved["num_layers"]) + 1
    live.write_text(yaml.safe_dump({"model": moved}), encoding="utf-8")
    with pytest.raises(SystemExit, match="num_layers"):
        lc0_control_train.preflight_architecture(config, allow_drift=False)


def test_the_built_model_is_checked_against_the_pinned_parameter_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ REVIEW F4 AT THE DRIVER — guard 0b, which had no caller at all.

    The config-level check compares ``model:`` keys; this one compares the net
    those keys BUILT against the pinned count. Nothing passed ``model=`` before
    2026-08-16, so ``LIVE_ARCH_PIN['trainable_params_unique_storage']`` — the
    arch fix's own headline number — gated nothing on any path. (The count is
    also re-measured for real, against the live architecture, by
    ``test_lc0_control_arch.py::test_this_tree_builds_the_pinned_live_architecture``;
    this one is about the DRIVER passing ``model=`` at all.)

    The live file here matches the control exactly, so the section check
    passes and the ONLY thing that can refuse the run is the parameter count.
    """
    config = _tiny_config(tmp_path)
    control_model = yaml.safe_load(config.read_text(encoding="utf-8"))["model"]
    live = tmp_path / "live_pbt2_small.yaml"
    live.write_text(yaml.safe_dump({"model": dict(control_model)}), encoding="utf-8")
    monkeypatch.setenv("CHESS_LIVE_PRODUCTION_CONFIG", str(live))
  # The pin's `model:` section IS the live file's here, so the pin's count
  # applies — and it is deliberately not the count this tree builds.
    monkeypatch.setitem(
        lc0_control_arch.LIVE_ARCH_PIN, "model", dict(control_model),
    )
    monkeypatch.setitem(
        lc0_control_arch.LIVE_ARCH_PIN, "trainable_params_unique_storage", 12_345,
    )
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    with pytest.raises(SystemExit, match="trainable params"):
        lc0_control_train.main([
            "--config", str(config), "--shards", str(shards),
            "--out-dir", str(tmp_path / "pin_run"), "--steps", "1",
            "--batch-size", "4", "--device", "cpu", "--no-compile",
        ])


# ── lc0_control_heldout ───────────────────────────────────────────────────────


def _freeze(tmp_path: Path, shards: Path, *, sample: int) -> tuple[int, Path]:
    out = tmp_path / "frozen.json"
    rc = lc0_control_heldout.main([
        "freeze", "--shards", str(shards), "--out", str(out),
        "--sample", str(sample), "--seed", "0",
    ])
    return rc, out


def test_freeze_refuses_a_pool_smaller_than_the_requested_sample(
    tmp_path: Path,
) -> None:
    """⚑ Codex #1. It used to write the short set, print a sha256 and exit 0 —
    and every downstream threshold is derived at n=100,000."""
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=100)
    assert rc == 1
    assert not out.exists(), "an underpowered frozen set must not exist on disk"


def test_freeze_writes_the_artifact_when_the_pool_is_large_enough(
    tmp_path: Path,
) -> None:
    shards = _write_shards(tmp_path / "held", list(range(10)))
    rc, out = _freeze(tmp_path, shards, sample=10)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["row_ids"]) == len(payload["input_ids"]) == 10


def test_purity_exits_one_on_an_empty_train_directory(tmp_path: Path) -> None:
    """⚑ Codex #6, at the driver: it printed PURE and exited 0."""
    shards = _write_shards(tmp_path / "held", list(range(10)))
    _rc, frozen = _freeze(tmp_path, shards, sample=10)
    empty = tmp_path / "no_train"
    empty.mkdir()
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(empty),
    ]) == 1


def test_purity_exits_one_on_exposure_the_record_id_cannot_see(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ Review F5: same inputs, different lc0 targets — 0 by id, N by input."""
    held = tmp_path / "held"
    train = tmp_path / "train"
    for target_seed, where in ((1, held), (2, train)):
        where.mkdir(parents=True, exist_ok=True)
        samples = []
        for seed in range(6):
            sample = _lc0_like_sample(seed, with_sf_wdl=False)
            rng = np.random.default_rng(9_000 + target_seed)
            policy = rng.random(COMPACT_POLICY_SIZE).astype(np.float32)
            sample.policy_target = policy / policy.sum()
            samples.append(sample)
        save_local_shard_arrays(
            where / "shard_000000.zarr", arrs=samples_to_arrays(samples),
            meta=ShardMeta(positions=len(samples)),
        )
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
    ]) == 1
    out = capsys.readouterr().out
    assert "intersecting ids     0" in out, "the record-id count must still read 0"
    assert "EXPOSED inputs       6" in out


def test_the_purity_receipt_ties_the_trained_corpus_to_the_check(
    tmp_path: Path,
) -> None:
    """⚑⚑ REVIEW F5 — NOTHING TIED THE TRAINED ROWS TO THE PURITY CHECK.

    ``summary.json`` recorded no shard list and no frozen sha256, and
    ``purity --train-shards`` was a free-floating argument. The receipt makes
    the link machine-checkable, so all three directions are tested here: the
    covered corpus LAUNCHES and banks the sha, an UNCOVERED directory is
    refused, and a receipt recording ``pure: false`` is refused.
    """
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 108)))
    other = _write_shards(tmp_path / "other", list(range(200, 206)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    receipt = tmp_path / "purity.json"
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
        "--receipt", str(receipt),
    ]) == 0
    banked = json.loads(receipt.read_text(encoding="utf-8"))
    assert banked["pure"] is True
    assert banked["train_shards"] == [str(train.resolve())]

    def _run(shards: list[Path], out: str) -> int:
        return lc0_control_train.main([
            "--config", str(_tiny_config(tmp_path)),
            "--shards", *[str(s) for s in shards],
            "--out-dir", str(tmp_path / out), "--steps", "1", "--batch-size", "4",
            "--device", "cpu", "--no-compile", "--allow-arch-drift",
            "--purity-receipt", str(receipt),
        ])

    assert _run([train], "covered") == 0
    summary = json.loads(
        (tmp_path / "covered" / "summary.json").read_text(encoding="utf-8"),
    )
    assert summary["corpus"]["shard_dirs"] == [str(train.resolve())]
    assert summary["corpus"]["label_coverage"]["search_wdl"] == {
        "labelled_rows": 8, "rows": 8,
    }
    assert summary["purity_receipt"]["frozen_sha256"] == banked["frozen_sha256"]

    with pytest.raises(SystemExit, match="NOT covered by the purity receipt"):
        _run([train, other], "uncovered")

    banked["pure"] = False
    receipt.write_text(json.dumps(banked), encoding="utf-8")
    with pytest.raises(SystemExit, match="records pure=False"):
        _run([train], "impure")


def test_a_run_without_a_purity_receipt_is_not_a_valid_control(
    tmp_path: Path,
) -> None:
    """The receipt is optional so plumbing runs stay possible — but a run that
    skipped it must SAY so in the artifact, by name, rather than inherit a
    `valid_control: true` it did not earn."""
    shards = _write_shards(tmp_path / "rows", list(range(8)))
    out = tmp_path / "no_receipt"
    assert lc0_control_train.main([
        "--config", str(_tiny_config(tmp_path)), "--shards", str(shards),
        "--out-dir", str(out), "--steps", "1", "--batch-size", "4",
        "--device", "cpu", "--no-compile", "--allow-arch-drift",
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid_control"] is False
    assert any("purity" in p for p in summary["validity_problems"])
    assert summary["purity_receipt"] is None


def test_purity_exits_zero_on_a_genuinely_disjoint_split(tmp_path: Path) -> None:
    held = _write_shards(tmp_path / "held", list(range(6)))
    train = _write_shards(tmp_path / "train", list(range(100, 106)))
    _rc, frozen = _freeze(tmp_path, held, sample=6)
    assert lc0_control_heldout.main([
        "purity", "--frozen", str(frozen), "--train-shards", str(train),
    ]) == 0


def test_chance_prints_the_jensen_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    shards = _write_shards(tmp_path / "held", list(range(6)))
    _rc, frozen = _freeze(tmp_path, shards, sample=6)
    assert lc0_control_heldout.main([
        "chance", "--frozen", str(frozen), "--shards", str(shards),
    ]) == 0
    out = capsys.readouterr().out
  # 20 legal moves in every synthetic row, so the two statistics coincide here
  # and the test is about the driver printing both, not about Jensen.
    assert "E[1/n_legal]         0.050000" in out
    assert "1/E[n_legal]         0.050000" in out


# ── lc0_control_eval ──────────────────────────────────────────────────────────


def _scores(path: Path, hits: np.ndarray, *, checkpoint: str | None = None) -> Path:
    np.savez_compressed(
        path,
        row_ids=np.array([f"id{i:04d}" for i in range(hits.size)], dtype="U32"),
        hit=hits.astype(np.uint8),
        meta=np.array([json.dumps({"checkpoint": checkpoint, "rows": int(hits.size)})],
                      dtype=object),
        allow_pickle=True,
    )
    return path


def test_compare_refuses_a_slope_it_cannot_resolve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ The Δ_train gap: 5,000 paired rows gave a 1.0721 pp halfwidth against a
    bar derived at n=100,000, and the rig printed it without comment."""
    rng = np.random.default_rng(0)
    a = (rng.random(5_000) < 0.30).astype(np.uint8)
    b = (rng.random(5_000) < 0.33).astype(np.uint8)
    rc = lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", a)),
        "--b", str(_scores(tmp_path / "b.npz", b)),
    ])
    assert rc == 1
    assert "exceeds the pre-committed bar" in capsys.readouterr().err


def test_compare_reports_when_the_halfwidth_clears_the_bar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The same refusal must not fire at the n the prereg was written for."""
    rng = np.random.default_rng(1)
    a = (rng.random(100_000) < 0.30).astype(np.uint8)
    b = a.copy()
    flip = rng.choice(100_000, size=300, replace=False)
    b[flip] = 1 - b[flip]
    rc = lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", a)),
        "--b", str(_scores(tmp_path / "b.npz", b)),
    ])
    assert rc == 0
    assert "halfwidth" in capsys.readouterr().out


def test_compare_mcnemar_arithmetic_is_pinned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The estimator the 0.392 pp bar is derived from had no regression test.

    Hand-built cells: b = 10 (A only), c = 30 (B only), 1,000 concordant hits
    and 8,960 concordant misses, n = 10,000.
    """
    n, b_cells, c_cells, both = 10_000, 10, 30, 1_000
    hit_a = np.zeros(n, dtype=np.uint8)
    hit_b = np.zeros(n, dtype=np.uint8)
    hit_a[:both] = hit_b[:both] = 1
    hit_a[both:both + b_cells] = 1
    hit_b[both + b_cells:both + b_cells + c_cells] = 1
    lc0_control_eval.main([
        "compare", "--a", str(_scores(tmp_path / "a.npz", hit_a)),
        "--b", str(_scores(tmp_path / "b.npz", hit_b)),
        "--max-halfwidth-pp", "10.0",
    ])
    out = capsys.readouterr().out
    delta = (c_cells - b_cells) / n
    var = (b_cells + c_cells - (c_cells - b_cells) ** 2 / n) / n**2
    half = 1.959963985 * float(np.sqrt(var))
    assert f"delta (B - A)        {delta * 100:+.4f} pp" in out
    assert f"(halfwidth {half * 100:.4f} pp)" in out
  # Cross-checked against an independent implementation:
  # `scipy.stats.binomtest(10, 40, 0.5).pvalue` == 0.0022214337732293643.
    assert lc0_control_eval._exact_mcnemar_p(10, 30) == pytest.approx(
        0.0022214337732293643, rel=1e-9,
    )


def test_compare_refuses_unpaired_score_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    a = _scores(tmp_path / "a.npz", np.ones(10, dtype=np.uint8))
    b = tmp_path / "b.npz"
    np.savez_compressed(
        b, row_ids=np.array([f"other{i}" for i in range(10)], dtype="U32"),
        hit=np.ones(10, dtype=np.uint8),
        meta=np.array([json.dumps({"checkpoint": None})], dtype=object),
        allow_pickle=True,
    )
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    assert "cannot be paired" in capsys.readouterr().err


def test_the_paired_halfwidth_reproduces_the_preregs_resolution_table() -> None:
    """⚑ REVIEW F2a — the shipped defaults cited a file that did not exist.

    ``docs/lc0_positive_control_prereg.md`` is committed now, and its resolution
    table is DERIVED here rather than remembered, so a drift between the doc
    and the code is a test failure instead of a discrepancy nobody re-checks.
    """
    assert lc0_control_eval.paired_halfwidth_pp(
        n=100_000, discordance=0.10,
    ) == pytest.approx(0.196, abs=5e-4)
    assert lc0_control_eval.paired_halfwidth_pp(
        n=100_000, discordance=0.20,
    ) == pytest.approx(0.277, abs=5e-4)
    assert lc0_control_eval.MATERIAL_BAR_PP == 0.392
  # ...and it is the number the CLI actually applies, not a parallel constant.
    derived = 2 * lc0_control_eval.paired_halfwidth_pp(n=100_000, discordance=0.10)
    assert pytest.approx(derived, abs=5e-4) == lc0_control_eval.MATERIAL_BAR_PP


def test_the_shuffled_control_floor_is_the_collision_rate_not_one_over_n_legal() -> None:
    """⚑⚑ REVIEW F2b — GUARD 1 WAS COMPARED AGAINST THE WRONG QUANTITY.

    The control scores each prediction against ANOTHER row's target, so its
    expectation is ``SUM_m p_pred(m)*p_tgt(m)``. Verified against a brute-force
    average over MANY permutations of the same arrays, which is the definition
    the derivation claims and cannot be satisfied by restating the formula.
    """
    rng = np.random.default_rng(0)
  # A deliberately non-uniform target marginal, like real lc0 argmaxes.
    target = rng.choice(20, size=4_000, p=np.array([0.3, 0.2, 0.1] + [0.4 / 17] * 17))
    predicted = rng.choice(20, size=4_000, p=np.array([0.25, 0.25, 0.1] + [0.4 / 17] * 17))

    expected = lc0_control_eval.shuffled_control_expectation(predicted, target)
    empirical = float(np.mean([
        (predicted == target[np.random.default_rng(s).permutation(target.size)]).mean()
        for s in range(200)
    ]))
    assert expected == pytest.approx(empirical, abs=2e-3)

  # ...and it is nowhere near the uniform-mover floor the prereg named. With 20
  # equally-legal moves that floor is 0.05; the collision rate is well above it
  # here and ~19x BELOW it on real lc0 rows. Either way it is a different
  # quantity, and a gate cannot be judged against the wrong one.
    assert expected != pytest.approx(1 / 20, abs=1e-3)


def test_the_negative_control_gate_can_pass_and_can_fail() -> None:
    """⚑ BOTH DIRECTIONS REACHABLE. A gate that cannot fail is not a gate, and
    one that cannot pass is a constant."""
    at_floor, problem = lc0_control_eval.negative_control_problem(
        observed=0.0033, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert problem is None, "a control sitting at its own floor must PASS"
    assert abs(at_floor) < 5.0

    z, problem = lc0_control_eval.negative_control_problem(
        observed=0.35, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert problem is not None
    assert z > 100.0
    assert "manufacturing agreement" in problem
  # ⚑ Landing BELOW the floor is not a failure: there is no mechanism behind
  # it. Only the "still agrees" direction has one.
    _z, low = lc0_control_eval.negative_control_problem(
        observed=0.0, expected=0.003283, rows=100_000, max_z=5.0,
    )
    assert low is None
  # A control that scored nothing checked nothing.
    _z, empty = lc0_control_eval.negative_control_problem(
        observed=0.0, expected=0.0, rows=0, max_z=5.0,
    )
    assert empty is not None


def test_score_exits_one_when_the_negative_control_does_not_collapse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate, THROUGH ``cmd_score``. The arithmetic being right is not the
    same claim as the driver acting on it — that gap is what F2b was about.

    ``_score_rows`` is stubbed because a leaking rig cannot be built out of a
    real model on demand; everything downstream of it is the real path.
    """
    rows = 60
    shards = _write_shards(tmp_path / "held", list(range(rows)))
    _rc, frozen = _freeze(tmp_path, shards, sample=rows)

    def _rigged(*_args: Any, **_kwargs: Any) -> tuple[Any, int, Any, Any]:
        moves = np.arange(rows, dtype=np.int64) % 3
        return np.ones(rows, dtype=np.uint8), 0, moves, moves

    monkeypatch.setattr(lc0_control_eval, "_score_rows", _rigged)
    rc = lc0_control_eval.main([
        "score", "--config", str(_tiny_config(tmp_path)), "--frozen", str(frozen),
        "--shards", str(shards), "--out", str(tmp_path / "s.npz"),
        "--device", "cpu", "--shuffle-targets",
    ])
    assert rc == 1
    assert "manufacturing agreement" in capsys.readouterr().err
  # The artifact is still written: the floor and the z belong WITH the score,
  # or nobody can re-check the verdict later.
    meta = json.loads(str(np.load(tmp_path / "s.npz", allow_pickle=True)["meta"][0]))
    assert meta["shuffled_collision_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert meta["negative_control_z"] > 5.0


def test_compare_refuses_a_zero_discordance_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ REVIEW F8. ``var = (b + c - (c-b)^2/n)/n^2`` is 0 at ``b == c == 0``,
    so the halfwidth bar — the only thing enforcing the prereg's n — passed at
    ANY n, including a file compared against itself."""
    hits = (np.arange(50) % 3 == 0).astype(np.uint8)
    a = _scores(tmp_path / "a.npz", hits)
    b = _scores(tmp_path / "b.npz", hits.copy())
    assert lc0_control_eval.main(["compare", "--a", str(a), "--b", str(b)]) == 1
    err = capsys.readouterr().err
    assert "discordant on ZERO" in err
    assert "below the prereg's resolution point" in err


def test_the_shuffled_target_control_permutes_targets_not_hits() -> None:
    """⚑ Prereg guard 1. Permuting the HIT VECTOR preserves the rate exactly and
    gives a negative control that cannot fail; permuting the TARGETS does not.
    """
    predicted = np.arange(1_000, dtype=np.int64) % 20
    target = predicted.copy()
    permuted = target[np.random.default_rng(0).permutation(target.size)]
    assert (predicted == target).mean() == 1.0
    assert (predicted == permuted).mean() < 0.2


# ── the eval scorer's checkpoint load ─────────────────────────────────────────


def test_scoring_refuses_a_checkpoint_that_does_not_fit_the_config(
    tmp_path: Path,
) -> None:
    """⚑ Codex #3. ``Trainer.load`` tolerates a partial load down to 50% of the
    keys, so a config/arch drift scores a HYBRID of trained and fresh tensors
    under the checkpoint's name. Top-1 is this arm's only yardstick."""
    from chess_anti_engine.model import build_model, model_config_from_flat_config
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    (tmp_path / "wide_dir").mkdir()
    (tmp_path / "narrow_dir").mkdir()
    wide = _tiny_config(tmp_path / "wide_dir")
    raw = yaml.safe_load(wide.read_text(encoding="utf-8"))
    raw["model"]["embed_dim"] = 64
    wide_path = tmp_path / "wide.yaml"
    wide_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    flat = flatten_run_config_defaults(load_yaml_file(str(wide_path)))
    model = build_model(model_config_from_flat_config(flat))
    ckpt = tmp_path / "wide.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    with pytest.raises(RuntimeError, match="require_complete"):
        lc0_control_eval._load_trainer(
            _tiny_config(tmp_path / "narrow_dir"), ckpt, "cpu", seed=0,
        )


def test_the_random_init_floor_is_reproducible_from_its_seed(
    tmp_path: Path,
) -> None:
    """⚑ Review F4 / Codex #4. Unseeded, four draws of the identical command
    spanned 2.18 pp — 5.6x the 0.392 pp material bar — straddling chance."""
    config = _tiny_config(tmp_path)

    def fingerprint(seed: int) -> float:
  # ⚑ Over EVERY parameter. The first tensor alone is a LayerNorm-style
  # constant init and is identical under any seed, so a probe that reads only
  # it passes whether or not the seed is applied — a check that cannot fail.
        trainer = lc0_control_eval._load_trainer(config, None, "cpu", seed=seed)
        return float(
            sum(float(p.detach().double().sum()) for p in trainer.model.parameters()),
        )

    assert fingerprint(7) == fingerprint(7)
    assert fingerprint(7) != fingerprint(8)


def test_the_guard_message_names_the_outcome_share_not_only_the_leak() -> None:
    """Both bars, in one place, because they fail on disjoint configurations."""
    from chess_anti_engine.train.value_blend_guard import (
        assert_no_silent_outcome_fallback,
    )

    all_outcome = value_blend_readout(
        sf_wdl_frac=0.0, search_wdl_frac=0.0,
        sf_effective_rows=8.0, search_effective_rows=8.0, batch_rows=8.0,
    )
    assert all_outcome.leaked_to_outcome == 0.0
    with pytest.raises(ValueBlendMisconfigured, match="RAW GAME OUTCOME"):
        assert_no_silent_outcome_fallback(all_outcome)
