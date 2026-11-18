"""Real Tiny global-union value-share plumbing; no real-data qualification."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.fixed_population import manifest_identity
from scripts import core_finite_learner as core
from scripts import core_selfplay_union_stored_row_once as entry
from scripts import core_selfplay_union_stored_row_once_proof as proof
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import frozen
from tests.test_core_selfplay_union_manifest import compiled, legacy, plan
from tests.test_core_selfplay_union_stored_row_once import closed_parent, contract, stage


def selected(tmp_path: Path, beta: float, *, split: bool = False,
             search_target: list[float] | None = None, search_available: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    c, _ = contract(tmp_path, split=split)
    if search_target is not None:
        root = tmp_path / "search-population"
        root.mkdir()
        branches = [{"kind": "legacy", "manifest": legacy(root, collection=i, rows=rows,
            architecture=c["architecture"], search_target=search_target, search_available=search_available)}
            for i, rows in ((1, 3), (2, 2))]
        pref = plan(root, branches, expected_sources=2, expected_rows=5, architecture=c["architecture"])
        manifest, mref = compiled(root, pref)
        c["population"] = {**manifest_identity(manifest, mref), "plan": pref}
    c["retained_search_share"] = beta
    _, method = core.read_ref(c["method"])
    method["training"].update(search_wdl_frac=beta, sf_wdl_frac=0.0, outcome_wdl_frac=1 - beta)
    c["method"] = frozen(tmp_path / "share-method.json", method)
    return c, frozen(tmp_path / "share-contract.json", c)


def variant(tmp_path: Path, c: dict[str, Any], name: str, *, beta: float | None,
            split: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    c = copy.deepcopy(c)
    c["updates_per_segment"] = [1, 2] if split else [3]
    c["sampling"]["quota_ledger"] = str(tmp_path / (name + "-quota.jsonl"))
    _, method = core.read_ref(c["method"])
    method["training"]["closed_segments_updates"] = c["updates_per_segment"]
    if beta is None:
        del c["retained_search_share"]
        for key in ("search_wdl_frac", "sf_wdl_frac", "outcome_wdl_frac"):
            del method["training"][key]
    else:
        c["retained_search_share"] = beta
        method["training"].update(search_wdl_frac=beta, sf_wdl_frac=0.0, outcome_wdl_frac=1 - beta)
    c["method"] = frozen(tmp_path / (name + ".method.json"), method)
    return c, frozen(tmp_path / (name + ".contract.json"), c)


@pytest.mark.parametrize("beta", [0.0, 0.1, 1.0])
def test_actual_trainer_fraction_saved_identity_and_passive_delivery(tmp_path: Path, beta: float) -> None:
    # Deliberately non-unit raw mass; existing loss performs normalization.
    c, cref = selected(tmp_path, beta, search_target=[1.0, 2.0, 5.0])
    full = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "full"))
    delivery = entry.validate_closed_model_delivery(cref, closed_parent(tmp_path, cref, full, proof=True))
    record, certificate = full["retained_search"], full["own_live_certificate"]
    assert record["retained_search_share"] == beta
    assert record["realized_search_mass"] == beta
    assert record["search_fallback_outcome_mass"] == 0
    assert record["search_availability_frac"] == 1
    if beta == 0:
        assert record["population_validation"] is None
    else:
        assert record["population_validation"]["rows"] == 5
    assert certificate["initial"]["retained_search_loss_kwargs"]["search_wdl_frac"] == beta
    assert certificate["final_retained_search_loss_kwargs"] == entry.retained_search_loss_kwargs(c)
    assert certificate["actual_calls"] == {"forward": 3, "optimizer_pre": 3, "optimizer_post": 3, "save": 1}
    assert certificate["extra_forwards"] == 0
    assert delivery["retained_search"] == record
    payload = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert payload[entry.CHECKPOINT_KEY]["identity"]["retained_search_share"] == beta
    assert full["quota_state"]["last_batch_rows"] == 1


def test_absent_and_explicit_zero_keep_actual_model_optimizer_and_five_rng_identical(tmp_path: Path) -> None:
    c, cref = selected(tmp_path, 0.0)  # No search fields: explicit zero is supported.
    explicit = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "zero"))
    old, oldref = variant(tmp_path, c, "legacy", beta=None)
    legacy_boundary = proof.execute(oldref, stage(tmp_path, oldref, old, 1, None, "legacy"))
    a = torch.load(explicit["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(legacy_boundary["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "zero vs absent " + key)
    assert explicit["rng"] == legacy_boundary["rng"]
    assert explicit["quota_state"] == legacy_boundary["quota_state"]
    assert "retained_search" not in legacy_boundary
    assert "retained_search" not in legacy_boundary["own_live_certificate"]
    assert "retained_search_share" not in b[entry.CHECKPOINT_KEY]["identity"]
    assert explicit["retained_search"]["search_availability_frac"] == 0
    assert explicit["retained_search"]["realized_search_mass"] == 0


def test_positive_fraction_changes_real_optimizer_without_order_or_rng_change(tmp_path: Path) -> None:
    c, cref = selected(tmp_path, 0.1, search_target=[1.0, 2.0, 5.0])
    positive = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "positive"))
    zero, zref = variant(tmp_path, c, "zero", beta=0.0)
    baseline = proof.execute(zref, stage(tmp_path, zref, zero, 1, None, "baseline"))
    a = torch.load(positive["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert any(not torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert proof.fingerprint(a["opt"], torch) != proof.fingerprint(b["opt"], torch)
    assert positive["rng"] == baseline["rng"]
    assert positive["quota_state"] == baseline["quota_state"]
    assert positive["quota_ledger_checkpoint"]["sha256"] == baseline["quota_ledger_checkpoint"]["sha256"]


def test_positive_split_full_state_resume_and_real_tail_match_full_pass(tmp_path: Path) -> None:
    c, cref = selected(tmp_path, 0.1, split=True, search_target=[1.0, 2.0, 5.0])
    first = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "first"))
    last = proof.execute(cref, stage(tmp_path, cref, c, 2, closed_parent(tmp_path, cref, first, proof=True), "last"))
    direct, dref = variant(tmp_path, c, "direct", beta=0.1)
    full = proof.execute(dref, stage(tmp_path, dref, direct, 1, None, "direct"))
    a = torch.load(last["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step"):
        require_equal(a[key], b[key], "retained search split " + key)
    assert last["rng"] == full["rng"]
    assert last["quota_state"] == full["quota_state"]
    assert last["retained_search"] == full["retained_search"]
    assert last["own_live_certificate"]["actual_calls"]["forward"] == 2


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, float("nan"), float("inf"), "0.1"])
def test_invalid_share_refuses_contract_before_model_or_draw(tmp_path: Path, bad: Any) -> None:
    c, _ = selected(tmp_path, 0.1)
    c["retained_search_share"] = bad
    with pytest.raises(ValueError, match=r"retained search share must be finite in \[0,1\]"):
        entry._validate_contract_metadata(c)
    with pytest.raises(ValueError, match=r"retained search share must be finite in \[0,1\]"):
        core.make_trainer(c, tmp_path / "invalid-tb", retained_search_share=bad)
    assert not Path(c["sampling"]["quota_ledger"]).exists()


@pytest.mark.parametrize(("key", "value"), [("search_wdl_frac", 0.2), ("sf_wdl_frac", 0.1), ("outcome_wdl_frac", 1.0)])
def test_method_fraction_alias_mismatch_refuses(tmp_path: Path, key: str, value: float) -> None:
    c, _ = selected(tmp_path, 0.1)
    _, method = core.read_ref(c["method"])
    method["training"][key] = value
    c["method"] = frozen(tmp_path / "wrong-method.json", method)
    with pytest.raises(ValueError, match="retained-search shares differ"):
        entry._validate_contract_metadata(c)


@pytest.mark.parametrize(("target", "available"), [(None, True), ([0.2, 0.3, 0.5], False)])
def test_missing_search_coverage_refuses_actual_entry_before_model_or_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: list[float] | None, available: bool,
) -> None:
    c, cref = selected(tmp_path, 0.1, search_target=target, search_available=available)
    calls: list[bool] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls.append(True)
        pytest.fail("model construction reached without complete search targets")

    monkeypatch.setattr(core, "make_trainer", forbidden)
    with pytest.raises(ValueError, match="retained search target"):
        entry.execute(cref, stage(tmp_path, cref, c, 1, None, "refused"))
    assert not calls
    assert Path(c["sampling"]["quota_ledger"]).stat().st_size == 0
    assert (tmp_path / "refused/draws.jsonl").stat().st_size == 0
    assert not (tmp_path / "refused/learner.pt").exists()


@pytest.mark.parametrize("target", [[0.0, 0.0, 0.0], [-0.1, 0.6, 0.5], [float("nan"), 0.5, 0.5],
                                     [float("inf"), 0.5, 0.5], [float("-inf"), 0.5, 0.5]])
def test_raw_target_validation_refuses_zero_negative_and_nonfinite(target: list[float]) -> None:
    class Population:
        membership = ((8, "a" * 64, 1),)
        _buffer = SimpleNamespace(_chunks=[{"x": np.zeros((1, 1)),
            "search_wdl": np.asarray([target], dtype=np.float16), "has_search_wdl": np.ones(1, dtype=np.uint8)}])

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError, match="finite nonnegative positive-mass vectors"):
        entry.validate_retained_search_population(Population(), np)


@pytest.mark.parametrize("fault", ["boundary_share", "certificate_live", "saved_tag"])
def test_intervention_identity_tamper_refuses_real_closed_delivery(tmp_path: Path, fault: str) -> None:
    c, cref = selected(tmp_path, 0.1, search_target=[1.0, 2.0, 5.0])
    if fault == "saved_tag":
        class Tamper(proof.OwnLiveProof):
            def closed_checkpoint(self, context: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
                checkpoint = context["checkpoint"]
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                payload[entry.CHECKPOINT_KEY]["identity"]["retained_search_share"] = 0
                checkpoint.chmod(0o644)
                torch.save(payload, checkpoint)
                checkpoint.chmod(0o444)
                boundary["checkpoint"]["sha256"] = core.file_sha(checkpoint)
                return super().closed_checkpoint(context, boundary)

        with pytest.raises(ValueError, match="final union saved family tag"):
            entry.execute(cref, stage(tmp_path, cref, c, 1, None, "tampered"), observer=Tamper())
        return
    full = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "full"))
    binding = closed_parent(tmp_path, cref, full, proof=True)
    altered = copy.deepcopy(full)
    if fault == "boundary_share":
        altered["retained_search"]["retained_search_share"] = 0
    else:
        altered["own_live_certificate"]["final_retained_search_loss_kwargs"]["search_wdl_frac"] = 0
    path = Path(binding["boundary"]["path"])
    path.parent.chmod(0o755)
    try:
        path.chmod(0o644)
        path.unlink()
        binding["boundary"] = frozen(path, altered)
    finally:
        path.chmod(0o444)
        path.parent.chmod(0o555)
    with pytest.raises(ValueError, match="retained search"):
        entry.validate_closed_model_delivery(cref, binding)


def test_general_loss_missing_search_keeps_outcome_fallback_and_reports_zero_coverage() -> None:
    from tests.test_value_blend_nan_labels import _base_batch, _f, _run

    batch = _base_batch()
    baseline = _run(batch, sf_wdl_frac=0.0, search_wdl_frac=0.0)
    missing = _run(batch, sf_wdl_frac=0.0, search_wdl_frac=0.1)
    assert _f(missing, "wdl_ce") == _f(baseline, "wdl_ce")
    assert _f(missing, "search_wdl_effective_rows") == 0
    c = {"retained_search_share": 0.1, "population": {"rows": 4, "sources": 1}}
    # General loss fallback remains supported, but this selected finite route
    # cannot certify that fallback as the requested 10% search intervention.
    with pytest.raises(ValueError, match="full usable coverage and zero fallback"):
        entry.retained_search_record(c, {"search_wdl_effective_frac": 0.0}, None)


def test_omitting_option_cannot_ignore_positive_selected_method(tmp_path: Path) -> None:
    c, _ = selected(tmp_path, 0.1)
    del c["retained_search_share"]
    # Keep the exact positive method reference. Deleting the intervention key
    # must not route a .1 method into the old default-zero constructor.
    with pytest.raises(ValueError, match="retained-search shares differ"):
        entry._validate_contract_metadata(c)
    assert not Path(c["sampling"]["quota_ledger"]).exists()
