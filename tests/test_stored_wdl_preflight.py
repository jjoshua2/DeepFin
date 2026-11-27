import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.stored_wdl_preflight import FEATURES, HISTORY, Receipt, prepare, snapshot


def fixture(tmp_path: Path) -> dict[str, Any]:
    paths = tuple(tmp_path / name for name in ("data", "donor", "candidate"))
    for p in paths:
        p.write_bytes(p.name.encode())
    return {"paths": paths,
                "receipts": tuple(Receipt(hashlib.sha256(p.read_bytes()).hexdigest(), "external-receipt/" + p.name) for p in paths),
                "history": HISTORY, "features": FEATURES, "architectures": ("exact-arch-v1", "exact-arch-v1"),
                "expected_architecture": "exact-arch-v1",
                "training_games": (frozenset({"corpus/train1"}), frozenset({"corpus/train2"})),
                "memberships_complete": (True, True),
                "holdout_games": frozenset({"corpus/test1", "corpus/test2"})}


def test_snapshot_plan_is_not_loader_and_survives_path_replacement(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    plan = prepare(**kwargs)
    kwargs["paths"][1].write_bytes(b"replacement")
    assert plan.donor == b"donor"
    assert not plan.inference_ready
    assert not plan.provenance_authenticated
    with pytest.raises(ValueError, match="identity"):
        prepare(**kwargs)


@pytest.mark.parametrize("change", [
    {"history": "fen_only"}, {"features": "unknown"},
    {"architectures": ("exact-arch-v1", "different")}, {"expected_architecture": ""},
    {"memberships_complete": (True, False)}, {"memberships_complete": (1, 1)},
    {"holdout_games": frozenset({"corpus/train1", "corpus/test2"})},
    {"holdout_games": frozenset({"only-one"})},
    {"training_games": (set(), set())},
])
def test_bad_declarations_rejected_before_io(tmp_path: Path, change: dict[str, Any]) -> None:
    kwargs = fixture(tmp_path)
    kwargs.update(change)
    for p in kwargs["paths"]:
        p.unlink()
    with pytest.raises(ValueError, match=r"encoding|architecture|membership|overlap|source games"):
        prepare(**kwargs)


def test_missing_receipt_and_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="authority"):
        snapshot(p, Receipt("a" * 64, ""))
    with pytest.raises(ValueError, match="nonempty"):
        snapshot(p, Receipt(hashlib.sha256(b"").hexdigest(), "external"))


def test_retains_immutable_declarations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.stored_wdl_preflight as module
    kwargs = fixture(tmp_path)
    for field in ("paths", "receipts", "architectures", "training_games", "memberships_complete"):
        kwargs[field] = list(kwargs[field])
    expected_receipts = tuple(kwargs["receipts"])
    original = module.snapshot

    def changing_caller(path: Path, receipt: Receipt) -> bytes:
        result = original(path, receipt)
        for field in ("paths", "receipts", "architectures", "training_games", "memberships_complete"):
            kwargs[field].clear()
        return result

    monkeypatch.setattr(module, "snapshot", changing_caller)
    plan = prepare(**kwargs)
    assert plan.receipts_declared == expected_receipts
    assert plan.history_declared == HISTORY
    assert plan.features_declared == FEATURES
    assert plan.architectures_declared == ("exact-arch-v1", "exact-arch-v1")
    assert plan.expected_architecture_declared == "exact-arch-v1"
    assert plan.holdout_games_declared == frozenset({"corpus/test1", "corpus/test2"})
    assert plan.memberships_complete_declared == (True, True)
    assert plan.training_games == (frozenset({"corpus/train1"}), frozenset({"corpus/train2"}))
