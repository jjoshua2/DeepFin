from tests.test_e2_bank_gate import bank as bank
import hashlib

import pytest
import torch

from scripts.e2_bank_gate import qualify_game
from chess_anti_engine.train.e2_batch_adapter import (
    load_paired_batch,
    compute_pair_losses,
)


@pytest.fixture
def qualified(bank, tmp_path):
    shard, candidate, digest = bank
    out = qualify_game(shard, candidate, digest, tmp_path)
    manifest_hash = hashlib.sha256(
        (out / "GAME-READBACK-COMMITTED.json").read_bytes()
    ).hexdigest()
    return {
        "shard": shard,
        "candidate": candidate,
        "qualified": out,
        "expected_manifest_sha256": manifest_hash,
        "indices": (1, 0, 1),
        "expected_indices_sha256": hashlib.sha256(b"[1,0,1]").hexdigest(),
        "categorical_bins": 32,
        "categorical_sigma": 0.04,
    }


def test_production_collate_and_loss_blends_only_once(qualified, monkeypatch):
    from chess_anti_engine.train import losses

    pair = load_paired_batch(**qualified)
    assert torch.equal(pair.search_only["x"], pair.outcome20["x"])
    assert torch.equal(
        pair.search_only["search_wdl"], torch.tensor([[0.25, 0.5, 0.25]] * 3)
    )
    a = {
        "policy": torch.zeros((3, 1858)),
        "wdl": torch.zeros((3, 3)),
        "categorical": torch.zeros((3, 32)),
    }
    b = {k: v.clone() for k, v in a.items()}
    original = losses.soft_cross_entropy
    seen = {}

    def observe(pred, target):
        if pred is a["wdl"]:
            seen["a"] = target.clone()
        if pred is b["wdl"]:
            seen["b"] = target.clone()
        if pred is b["categorical"]:
            seen["cat_b"] = target.clone()
        return original(pred, target)

    monkeypatch.setattr(losses, "soft_cross_entropy", observe)
    compute_pair_losses(
        pair, a, b, shared_kwargs={"w_policy": 0.0, "w_categorical": 1.0}
    )
    natural = torch.nn.functional.one_hot(pair.outcome20["wdl_t"], 3).float()
    torch.testing.assert_close(seen["b"], 0.8 * seen["a"] + 0.2 * natural)
    torch.testing.assert_close(seen["cat_b"], pair.outcome20["categorical_t"])
    # Feeding preblended target into .8 again would differ.
    assert not torch.allclose(seen["b"], 0.8 * seen["b"] + 0.2 * natural)


@pytest.mark.parametrize(
    "change",
    [
        {"categorical_bins": 16},
        {"categorical_sigma": 0.05},
        {"indices": (0, 1, 1)},
        {"expected_manifest_sha256": "0" * 64},
    ],
)
def test_contract_mismatch_rejected(qualified, change):
    with pytest.raises(ValueError, match=r".+"):
        load_paired_batch(**(qualified | change))


def test_target_archive_tamper_rejected(qualified):
    p = qualified["qualified"] / "arm-targets.npz"
    p.write_bytes(p.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="target identity"):
        load_paired_batch(**qualified)


@pytest.mark.parametrize("key", ["has_search_wdl", "categorical_t", "x"])
def test_both_arm_tensor_tamper_rejected(qualified, key):
    pair = load_paired_batch(**qualified)
    pair.search_only[key].flatten()[0] = 9
    pair.outcome20[key].flatten()[0] = 9
    outputs = {
        "policy": torch.zeros((3, 1858)),
        "wdl": torch.zeros((3, 3)),
        "categorical": torch.zeros((3, 32)),
    }
    with pytest.raises(ValueError, match="tensor identity"):
        compute_pair_losses(pair, outputs, outputs, shared_kwargs={})


@pytest.mark.parametrize("field", ["has_search_wdl", "policy_encoding"])
def test_physical_masks_and_encoding_rejected_even_with_rebound_fixture(
    qualified, field
):
    import json
    import shutil
    from scripts.e2_bank_gate import inventory

    shard = qualified["shard"]
    if field == "has_search_wdl":
        shutil.rmtree(shard / "has_search_wdl")
    else:
        attrs = json.loads((shard / ".zattrs").read_text())
        attrs["policy_encoding"] = "full4672"
        (shard / ".zattrs").write_text(json.dumps(attrs))
    manifest = qualified["qualified"] / "GAME-READBACK-COMMITTED.json"
    m = json.loads(manifest.read_text())
    m["source_inventory"] = inventory(shard)
    manifest.write_text(json.dumps(m))
    qualified["expected_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    with pytest.raises((ValueError, RuntimeError), match=r".+"):
        load_paired_batch(**qualified)
