"""Legacy and lazy API integration; UNRUN under the small component route.

These pytest cases use the normal project conftest. They are distinct from the
standalone native unittest module and require their own admitted environment.
"""
from __future__ import annotations

import importlib
import pickle
from types import SimpleNamespace

import pytest


def test_legacy_root_contract_exports_are_identical_objects() -> None:
    from scripts import bt4_generation_evaluator as evaluator
    from scripts import bt4_model_arithmetic as arithmetic
    from scripts import bt4_raw_corpus_sidecar as sidecar
    from scripts import bt4_root_contracts as contracts
    from scripts import gen_sf_rooted_corpus as corpus

    assert evaluator.BT4RootOutput is contracts.BT4RootOutput
    assert evaluator._search_policy_logits is contracts._search_policy_logits
    assert evaluator._search_wdl_logits is contracts._search_wdl_logits
    assert corpus.input_tensor_key is contracts.input_tensor_key
    assert sidecar.validate_wdl_values is contracts.validate_wdl_values
    assert arithmetic.validate_wdl_values is contracts.validate_wdl_values
    # Existing pickle GLOBAL references resolve the old module's identical alias.
    assert pickle.loads(b"cscripts.bt4_generation_evaluator\nBT4RootOutput\n.") is contracts.BT4RootOutput
    assert pickle.loads(pickle.dumps(contracts.BT4RootOutput)) is contracts.BT4RootOutput


@pytest.mark.parametrize("package_name", ["mcts", "selfplay", "eval"])
def test_lazy_package_export_dispatch_with_synthetic_targets(
    package_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module(f"chess_anti_engine.{package_name}")
    names = tuple(package.__all__)
    missing = object()
    original = {name: package.__dict__.get(name, missing) for name in names}
    values = {name: object() for name in names}
    calls: list[tuple[str, str]] = []

    def target(module: str, parent: str) -> SimpleNamespace:
        calls.append((module, parent))
        return SimpleNamespace(**values)

    monkeypatch.setattr(package, "import_module", target)
    try:
        for name in names:
            package.__dict__.pop(name, None)
        for name in names:
            assert getattr(package, name) is values[name]
            assert package.__dict__[name] is values[name]
        assert set(names).issubset(dir(package))
        assert all(parent == package.__name__ for _, parent in calls)
        count = len(calls)
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(package, "unknown_export_for_contract_test")
        assert len(calls) == count
    finally:
        for name, value in original.items():
            if value is missing:
                package.__dict__.pop(name, None)
            else:
                package.__dict__[name] = value


@pytest.mark.parametrize("package_name", ["mcts", "selfplay", "eval"])
def test_real_lazy_exports_keep_original_target_identity(package_name: str) -> None:
    package = importlib.import_module(f"chess_anti_engine.{package_name}")
    for name in package.__all__:
        target = package._EXPORTS[name] if package_name != "eval" else ".puzzles"
        actual = importlib.import_module(target, package.__name__)
        assert getattr(package, name) is getattr(actual, name)
