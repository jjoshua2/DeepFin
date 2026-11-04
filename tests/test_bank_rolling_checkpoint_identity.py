"""One-pass copied-script banking; all checkpoints are inert text markers."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.script_loading import load_script_module

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _checkout(tmp: Path) -> tuple[Path, Path]:
    root = tmp / "fake checkout [bank]"
    (root / "scripts").mkdir(parents=True)
    for name in ("bank_rolling_checkpoints.sh", "bank_rolling_checkpoints.py"):
        if (SCRIPTS / name).exists():
            shutil.copy2(SCRIPTS / name, root / "scripts" / name)
    return root, root / "scripts/bank_rolling_checkpoints.sh"


def _checkpoint(tune: Path, trial: str, index: int, marker: str, clock: int = 100) -> Path:
    p = tune / trial / f"checkpoint_{index:06d}"
    p.mkdir(parents=True)
    (p / "trainer.pt").write_text(marker)
    os.utime(p, ns=(clock, clock))
    return p


def _run(script: Path, tune: Path, keep: int = 24, bank: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "ONCE": "1", "EVERY": "5", "KEEP": str(keep),
           "SLEEP": "600", "TUNE_DIR": str(tune)}
    if bank is not None:
        env["BANK_DIR"] = str(bank)
    return subprocess.run(["bash", str(script)], env=env, capture_output=True,
                          text=True, check=False, timeout=10)


def _bank(root: Path) -> Path:
    return root / "data/salvage/rolling"


def _markers(bank: Path) -> list[str]:
    return sorted(p.read_text() for p in bank.rglob("trainer.pt"))


def _snapshot(bank: Path) -> dict[str, bytes]:
    return {str(p.relative_to(bank)): p.read_bytes() for p in bank.rglob("*")
            if p.is_file() and not p.is_symlink() and p.name != ".bank.lock"}


def test_distinct_trials_same_index_are_both_banked(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "experiment A/tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    _checkpoint(tune, "train_trial_B", 5, "B")
    result = _run(script, tune)
    assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["A", "B"]


def test_distinct_experiments_same_trial_basename_are_both_banked(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    for experiment, marker in (("one", "A"), ("two", "B")):
        tune = tmp_path / experiment / "tune"
        _checkpoint(tune, "train_trial_same", 5, marker)
        result = _run(script, tune)
        assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["A", "B"]


def test_unchanged_rerun_and_symlink_alias_do_not_duplicate(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "original/tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    alias = tmp_path / "alias tune"
    alias.symlink_to(tune, target_is_directory=True)
    for source in (tune, tune, alias):
        result = _run(script, source)
        assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["A"]


def test_legacy_pinned_slots_consume_global_budget_without_suppressing_donor(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    legacy = _bank(root) / "checkpoint_000005"
    legacy.mkdir(parents=True)
    (legacy / "trainer.pt").write_text("legacy unknown")
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "old", 10)
    _checkpoint(tune, "train_trial_B", 5, "new", 20)
    result = _run(script, tune, keep=2)
    assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["legacy unknown", "new"]
    assert len([p for p in _bank(root).iterdir() if p.name != ".bank.lock"]) == 2


@pytest.mark.parametrize("count", [2, 3])
def test_legacy_at_or_over_capacity_refuses_without_mutation(tmp_path: Path, count: int) -> None:
    root, script = _checkout(tmp_path)
    for i in range(count):
        p = _bank(root) / f"checkpoint_{i:06d}"
        p.mkdir(parents=True)
        (p / "trainer.pt").write_text(f"legacy{i}")
    before = _snapshot(_bank(root))
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 20, "new")
    result = _run(script, tune, keep=2)
    assert result.returncode != 0
    assert "legacy" in result.stderr
    assert "KEEP=2" in result.stderr
    assert _snapshot(_bank(root)) == before


def test_rank_before_copy_skips_evicted_and_late_discovered_old_sources(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 500, "old high index", 10)
    _checkpoint(tune, "train_trial_B", 5, "new low index", 20)
    result = _run(script, tune, keep=1)
    assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["new low index"]
    assert "checkpoint_000500" not in result.stdout
    _checkpoint(tune, "train_trial_C", 1000, "late old", 5)
    result = _run(script, tune, keep=1)
    assert result.returncode == 0, result.stderr
    assert "banked" not in result.stdout
    assert _markers(_bank(root)) == ["new low index"]


def test_source_pruned_bank_remains_candidate_and_equal_clocks_are_deterministic(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "retained", 100)
    assert _run(script, tune, keep=1).returncode == 0
    shutil.rmtree(source)
    _checkpoint(tune, "train_trial_B", 10, "old", 50)
    assert _run(script, tune, keep=1).returncode == 0
    assert _markers(_bank(root)) == ["retained"]
    _checkpoint(tune, "train_trial_C", 15, "equal", 100)
    assert _run(script, tune, keep=1).returncode == 0
    before = _snapshot(_bank(root))
    result = _run(script, tune, keep=1)
    assert result.returncode == 0, result.stderr
    assert _snapshot(_bank(root)) == before
    assert "banked" not in result.stdout


@pytest.mark.parametrize("change", ["rewrite", "replace", "sidecar", "provenance"])
def test_changed_source_or_provenance_refuses_without_relabel_or_eviction(tmp_path: Path, change: str) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "original")
    assert _run(script, tune).returncode == 0
    if change == "provenance":
        metadata = next(_bank(root).glob("bank_*/.bank-provenance.json"))
        value = json.loads(metadata.read_text())
        value["source_path"] = str(tmp_path / "other/checkpoint_000005")
        metadata.write_text(json.dumps(value))
    elif change == "replace":
        (source / "trainer.pt").unlink()
        (source / "trainer.pt").write_text("replacement")
    elif change == "rewrite":
        (source / "trainer.pt").write_text("changed")
    else:
        (source / "extra.json").write_text("{}")
    before = _snapshot(_bank(root))
    result = _run(script, tune)
    assert result.returncode != 0
    assert _snapshot(_bank(root)) == before
    assert _markers(_bank(root)) == ["original"]


@pytest.mark.parametrize("fault", [OSError, KeyboardInterrupt])
def test_copy_failure_preserves_bank_and_cleans_only_own_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: type[BaseException]) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_failure")
    bank = tmp_path / "bank"
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "old", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    before = _snapshot(bank)
    _checkpoint(tune, "train_trial_B", 10, "new", 20)

    def fail_copy(_source: Path, dest: Path, **_kwargs: Any) -> None:
        (dest / "partial").write_text("partial")
        raise fault("injected copy interruption")

    monkeypatch.setattr(module.shutil, "copytree", fail_copy)
    with pytest.raises(fault, match="injected"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _snapshot(bank) == before
    assert not list(bank.glob(".bank-stage-*"))


def test_old_writer_mixed_bank_is_unidentified_pinned_not_managed(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "new-format")
    assert _run(script, tune, keep=1).returncode == 0
    legacy = _bank(root) / "checkpoint_000010"
    legacy.mkdir()
    (legacy / "trainer.pt").write_text("old-writer")
    before = _snapshot(_bank(root))
    result = _run(script, tune, keep=1)
    assert result.returncode != 0
    assert "legacy" in result.stderr
    assert "over budget" in result.stderr
    assert _snapshot(_bank(root)) == before


def test_bank_dir_override_is_real_and_unknown_symlink_is_not_followed_for_trim(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "new")
    selected = tmp_path / "explicit bank"
    selected.mkdir()
    protected = tmp_path / "protected fixture"
    protected.mkdir()
    (protected / "keep").write_text("untouched")
    (selected / "checkpoint_000001").symlink_to(protected, target_is_directory=True)
    result = _run(script, tune, keep=2, bank=selected)
    assert result.returncode == 0, result.stderr
    assert (protected / "keep").read_text() == "untouched"
    assert (selected / "checkpoint_000001").is_symlink()
    assert not _bank(root).exists()
    assert _markers(selected) == ["new"]


def test_every_filter_still_skips_nonmultiples(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "eligible")
    _checkpoint(tune, "train_trial_B", 6, "excluded")
    result = _run(script, tune)
    assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["eligible"]


def test_source_change_during_copy_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_changing_source")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    old = _checkpoint(tune, "train_trial_A", 5, "old", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    before = _snapshot(bank)
    new = _checkpoint(tune, "train_trial_B", 10, "new", 20)
    original = module.shutil.copytree

    def changing(source: Path, dest: Path, **kwargs: Any) -> Any:
        result = original(source, dest, **kwargs)
        (source / "trainer.pt").write_text("changed during copy")
        return result

    monkeypatch.setattr(module.shutil, "copytree", changing)
    with pytest.raises(module.TransientSourceError, match="source changed during"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _snapshot(bank) == before
    assert not list(bank.glob(".bank-stage-*"))
    assert (old / "trainer.pt").read_text() == "old"
    assert (new / "trainer.pt").read_text() == "changed during copy"


def test_provenance_write_failure_is_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_metadata_failure")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "old", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    before = _snapshot(bank)
    _checkpoint(tune, "train_trial_B", 10, "new", 20)
    write = Path.write_text

    def fail(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == ".bank-provenance.json":
            raise OSError("injected provenance failure")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="injected"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _snapshot(bank) == before
    assert not list(bank.glob(".bank-stage-*"))


def test_partial_publication_failure_reports_unconfirmed_cap_and_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_partial_publication")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "oldA", 10)
    _checkpoint(tune, "train_trial_B", 5, "oldB", 11)
    module.bank_once(tune, bank, every=5, keep=2)
    _checkpoint(tune, "train_trial_C", 5, "newC", 20)
    _checkpoint(tune, "train_trial_D", 5, "newD", 30)
    rename = Path.rename
    calls = 0

    def fail_second(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second publication failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="injected"):
        module.bank_once(tune, bank, every=5, keep=2)
    assert _markers(bank) == ["newD", "oldA", "oldB"]
    assert "global KEEP not confirmed" in capsys.readouterr().err
    assert not list(bank.glob(".bank-stage-*"))
    monkeypatch.setattr(Path, "rename", rename)
    module.bank_once(tune, bank, every=5, keep=2)
    assert _markers(bank) == ["newC", "newD"]


def test_trim_failure_is_not_reported_as_realized_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_trim_failure")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "old", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    old_bank = next(bank.glob("bank_*"))
    _checkpoint(tune, "train_trial_B", 5, "new", 20)
    remove = module.shutil.rmtree

    def fail(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == old_bank:
            raise PermissionError("injected trim failure")
        remove(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", fail)
    with pytest.raises(PermissionError, match="injected"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _markers(bank) == ["new", "old"]
    assert "global KEEP not confirmed" in capsys.readouterr().err


def test_managed_looking_bank_symlink_is_pinned_not_followed(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "protected")
    assert _run(script, tune).returncode == 0
    path = next(_bank(root).glob("bank_*"))
    protected = tmp_path / "protected fixture"
    path.rename(protected)
    path.symlink_to(protected, target_is_directory=True)
    before = _snapshot(protected)
    result = _run(script, tune)
    assert result.returncode != 0
    assert _snapshot(protected) == before
    assert path.is_symlink()


def test_two_cooperative_writers_serialize_one_pass(tmp_path: Path) -> None:
    import sys
    import time

    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "A")
    _checkpoint(tune, "train_trial_B", 5, "B")
    helper = SCRIPTS / "bank_rolling_checkpoints.py"
    ready, release, attempted, acquired = [tmp_path / name for name in ("ready", "release", "attempted", "acquired")]
    setup = f"import runpy,time; from pathlib import Path; m=runpy.run_path({str(helper)!r}); "
    first = setup + f'''
copy=m['shutil'].copytree
def blocking(src,dst,**kw):
 Path({str(ready)!r}).touch()
 deadline=time.monotonic()+15
 while not Path({str(release)!r}).exists():
  if time.monotonic()>deadline: raise TimeoutError('release deadline')
  time.sleep(.01)
 return copy(src,dst,**kw)
m['shutil'].copytree=blocking
m['bank_once'](Path({str(tune)!r}),Path({str(bank)!r}),every=5,keep=2)
'''
    second = setup + f'''
lock=m['fcntl'].flock
def observed(fd,operation):
 Path({str(attempted)!r}).touch()
 lock(fd,operation)
 Path({str(acquired)!r}).touch()
m['fcntl'].flock=observed
m['bank_once'](Path({str(tune)!r}),Path({str(bank)!r}),every=5,keep=2)
'''

    def wait_file(path: Path) -> None:
        deadline = time.monotonic() + 10
        while not path.exists():
            if time.monotonic() > deadline:
                raise AssertionError(f"missing subprocess boundary: {path.name}")
            time.sleep(.01)

    children: list[subprocess.Popen[str]] = []
    try:
        children.append(subprocess.Popen([sys.executable, "-c", first], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        wait_file(ready)
        children.append(subprocess.Popen([sys.executable, "-c", second], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        wait_file(attempted)
        assert not acquired.exists()  # First process is inside copy with lock held.
        release.touch()
        for child in children:
            out, err = child.communicate(timeout=15)
            assert child.returncode == 0, (out, err)
        assert acquired.exists()
        assert _markers(bank) == ["A", "B"]
        assert not list(bank.glob(".bank-stage-*"))
    finally:
        release.touch(exist_ok=True)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("keep", [0, -1])
def test_invalid_retention_cannot_delete_or_create_a_bank(tmp_path: Path, keep: int) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    result = _run(script, tune, keep=keep)
    assert result.returncode != 0
    assert "positive integers" in result.stderr
    assert not _bank(root).exists()


def test_unrecognized_files_and_abandoned_stages_are_pinned_slots(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    bank = _bank(root)
    bank.mkdir(parents=True)
    (bank / "operator-note").write_text("do not delete")
    stage = bank / ".bank-stage-abandoned"
    stage.mkdir()
    (stage / "partial").write_text("not automatically owned")
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    before = _snapshot(bank)
    result = _run(script, tune, keep=2)
    assert result.returncode != 0
    assert "2 unidentified legacy entries" in result.stderr
    assert _snapshot(bank) == before


def test_managed_metadata_without_weights_is_not_treated_as_completed_bank(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    assert _run(script, tune).returncode == 0
    weights = next(_bank(root).glob("bank_*/trainer.pt"))
    weights.unlink()
    before = _snapshot(_bank(root))
    result = _run(script, tune)
    assert result.returncode != 0
    assert "untrusted existing bank entry" in result.stderr
    assert _snapshot(_bank(root)) == before


@pytest.mark.parametrize("location", ["tune", "trial", "checkpoint", "nested", "bank_alias", "trial_alias", "checkpoint_alias"])
def test_destination_inside_source_is_rejected_before_any_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_containment")
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    if location == "trial_alias":
        external_trial = tmp_path / "external-trial"
        source.parent.rename(external_trial)
        (tune / "train_trial_A").symlink_to(external_trial, target_is_directory=True)
        source = external_trial / source.name
    elif location == "checkpoint_alias":
        external_source = tmp_path / "external" / source.name
        external_source.parent.mkdir()
        source.rename(external_source)
        source.symlink_to(external_source, target_is_directory=True)
        source = external_source
    destination = {"tune": tune, "trial": source.parent, "checkpoint": source}.get(location, source / "new-bank")
    if location == "bank_alias":
        alias = tmp_path / "bank-alias"
        alias.symlink_to(source, target_is_directory=True)
        destination = alias / "new-bank"
    before = module._signature(tune)
    source_before = module._signature(source)

    def no_copy(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("copy must not be reached")

    monkeypatch.setattr(module.shutil, "copytree", no_copy)
    with pytest.raises(ValueError, match="outside source tune/trial/checkpoint roots"):
        module.bank_once(tune, destination, every=5, keep=24)
    assert module._signature(tune) == before
    assert module._signature(source) == source_before
    assert not (destination / ".bank.lock").exists()


def test_sibling_bank_of_source_tune_is_supported(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "experiment/tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    bank = tune.parent / "bank"
    result = _run(script, tune, bank=bank)
    assert result.returncode == 0, result.stderr
    assert _markers(bank) == ["A"]
    assert not _bank(root).exists()


def test_symlink_source_weights_refused_without_following_target(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    target = tmp_path / "external-weights"
    target.write_text("external A")
    (source / "trainer.pt").unlink()
    (source / "trainer.pt").symlink_to(target)
    before = _snapshot(source)
    result = _run(script, tune)
    assert result.returncode != 0
    assert "regular non-symlink trainer.pt" in result.stderr
    assert not list(_bank(root).glob("bank_*"))
    assert _snapshot(source) == before
    assert target.read_text() == "external A"


def test_existing_linked_weights_are_pinned_not_deleted_or_migrated(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    assert _run(script, tune).returncode == 0
    weights = next(_bank(root).glob("bank_*/trainer.pt"))
    target = tmp_path / "external-weights"
    target.write_text("operator A")
    weights.unlink()
    weights.symlink_to(target)
    before = _snapshot(_bank(root))
    result = _run(script, tune)
    assert result.returncode != 0
    assert "untrusted existing bank entry" in result.stderr
    assert weights.is_symlink()
    assert _snapshot(_bank(root)) == before
    assert target.read_text() == "operator A"


def test_optional_sidecar_link_is_preserved_not_an_independent_snapshot(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    target = tmp_path / "external-sidecar"
    target.write_text("old external data")
    (source / "optional-sidecar").symlink_to(target)
    assert _run(script, tune).returncode == 0
    sidecar = next(_bank(root).glob("bank_*/optional-sidecar"))
    assert sidecar.is_symlink()
    target.write_text("changed external data")
    result = _run(script, tune)
    assert result.returncode == 0, result.stderr
    assert "banked" not in result.stdout  # Target content is not in the signature.
    assert sidecar.read_text() == "changed external data"


def test_dangling_reserved_provenance_link_cannot_redirect_metadata_write(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    target = tmp_path / "outside-bank.json"
    (source / ".bank-provenance.json").symlink_to(target)
    result = _run(script, tune)
    assert result.returncode != 0
    assert "reserved bank metadata" in result.stderr
    assert not target.exists()
    assert not list(_bank(root).glob("bank_*"))
    assert (source / ".bank-provenance.json").is_symlink()


def test_symlink_bank_provenance_is_pinned_without_following_or_migrating(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "A")
    assert _run(script, tune).returncode == 0
    provenance = next(_bank(root).glob("bank_*/.bank-provenance.json"))
    target = tmp_path / "operator-metadata.json"
    target.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(target)
    before = target.read_bytes()
    result = _run(script, tune)
    assert result.returncode != 0
    assert "untrusted existing bank entry" in result.stderr
    assert provenance.is_symlink()
    assert target.read_bytes() == before
    assert _markers(_bank(root)) == ["A"]


def test_fifo_provenance_is_pinned_without_opening_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_fifo_metadata")
    bank = tmp_path / "bank_candidate"
    bank.mkdir()
    (bank / "trainer.pt").write_text("A")
    metadata = bank / ".bank-provenance.json"
    os.mkfifo(metadata)

    def no_read(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unknown FIFO provenance must never be opened")

    monkeypatch.setattr(Path, "read_text", no_read)
    assert module._managed(bank) is None
    assert metadata.exists()


def test_fifo_reserved_lock_is_refused_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_fifo_lock")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "A")
    bank.mkdir()
    lock = bank / ".bank.lock"
    os.mkfifo(lock)

    def no_open(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("known non-regular reserved lock must never be opened")

    monkeypatch.setattr(module.os, "open", no_open)
    with pytest.raises(ValueError, match="must be a regular file"):
        module.bank_once(tune, bank, every=5, keep=24)
    assert list(bank.iterdir()) == [lock]


def test_partial_trim_preserves_selected_new_but_pins_incomplete_old_on_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_partial_trim")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "obsolete A", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    obsolete = next(bank.glob("bank_*"))
    _checkpoint(tune, "train_trial_B", 5, "selected B", 20)
    real_rmtree = module.shutil.rmtree

    def partial_trim(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == obsolete:
            (obsolete / "trainer.pt").unlink()
            raise OSError("obsolete entry partially removed before trim fault")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", partial_trim)
    with pytest.raises(OSError, match="partially removed"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert not (obsolete / "trainer.pt").exists()  # Old bank is NOT rolled back.
    assert (obsolete / ".bank-provenance.json").is_file()
    assert _markers(bank) == ["selected B"]
    assert len(list(bank.glob("bank_*"))) == 2  # Final KEEP is unconfirmed.
    monkeypatch.setattr(module.shutil, "rmtree", real_rmtree)
    before = _snapshot(bank)
    with pytest.raises(ValueError, match="1 unidentified legacy entries consume KEEP=1"):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _snapshot(bank) == before  # Operator disposition, not implicit cleanup.
    assert obsolete.exists()
    assert _markers(bank) == ["selected B"]


def test_once_incomplete_source_reports_transient_failure_without_internal_retry(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = tune / "train_trial_A/checkpoint_000005"
    source.mkdir(parents=True)
    result = _run(script, tune)
    assert result.returncode != 0
    assert result.returncode == 75
    assert "TEMPORARY SOURCE" in result.stderr
    assert not list(_bank(root).glob("bank_*"))
    (source / "trainer.pt").write_text("completed A")
    result = _run(script, tune)  # Explicit restart, not an internal retry claim.
    assert result.returncode == 0, result.stderr
    assert _markers(_bank(root)) == ["completed A"]


@pytest.mark.parametrize("where", ["resolve", "signature", "copy"])
def test_new_source_disappearance_is_transient_without_bank_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_source_disappearance")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    _checkpoint(tune, "train_trial_A", 5, "retained A", 10)
    module.bank_once(tune, bank, every=5, keep=1)
    before = _snapshot(bank)
    source = _checkpoint(tune, "train_trial_B", 5, "new B", 20)
    if where == "resolve":
        real_resolve = Path.resolve
        calls = 0

        def pruning_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
            nonlocal calls
            if path == source:
                calls += 1
                if calls == 2:  # Discovery after the containment check.
                    shutil.rmtree(source)
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", pruning_resolve)
    elif where == "signature":
        real_signature = module._signature

        def pruning_signature(path: Path) -> str:
            if path == source:
                shutil.rmtree(source)
            return real_signature(path)

        monkeypatch.setattr(module, "_signature", pruning_signature)
    else:
        def pruning_copy(*_args: Any, **_kwargs: Any) -> None:
            shutil.rmtree(source)
            raise FileNotFoundError("source pruned during copy")

        monkeypatch.setattr(module.shutil, "copytree", pruning_copy)
    with pytest.raises(module.TransientSourceError):
        module.bank_once(tune, bank, every=5, keep=1)
    assert _snapshot(bank) == before
    assert not list(bank.glob(".bank-stage-*"))


def test_containment_resolve_tolerates_pruned_trial_without_writing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script_module("bank_rolling_checkpoints.py", "bank_identity_containment_prune")
    tune, bank = tmp_path / "tune", tmp_path / "bank"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    trial = source.parent
    real_resolve = Path.resolve

    def pruning_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == trial and trial.exists():
            shutil.rmtree(trial)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", pruning_resolve)
    module.bank_once(tune, bank, every=5, keep=1)
    assert not trial.exists()
    assert list(bank.iterdir()) == [bank / ".bank.lock"]


def test_missing_payload_in_still_present_banked_source_is_fatal(tmp_path: Path) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    source = _checkpoint(tune, "train_trial_A", 5, "A")
    assert _run(script, tune).returncode == 0
    (source / "trainer.pt").unlink()
    before = _snapshot(_bank(root))
    result = _run(script, tune)
    assert result.returncode == 1
    assert "already-banked source lost trainer.pt" in result.stderr
    assert _snapshot(_bank(root)) == before


def _inject_helper(script: Path, code: str) -> None:
    helper = script.with_suffix(".py")
    text = helper.read_text()
    helper.write_text(text.replace('if __name__ == "__main__":', code + '\nif __name__ == "__main__":'))


def _sleep_gate(tmp_path: Path) -> tuple[Path, Path, Path]:
    tools = tmp_path / "fixture-bin"
    tools.mkdir()
    entered, release = tmp_path / "poll-sleep-entered", tmp_path / "poll-sleep-release"
    sleeper = tools / "sleep"
    sleeper.write_text(f'''#!/usr/bin/python3
from pathlib import Path
import time
Path({str(entered)!r}).touch()
deadline=time.monotonic()+10
while not Path({str(release)!r}).exists():
    if time.monotonic()>deadline: raise SystemExit(91)
    time.sleep(.01)
time.sleep(.02)
''')
    sleeper.chmod(0o755)
    return tools, entered, release


@pytest.mark.parametrize("race", ["incomplete", "changed_during_copy"])
def test_daemon_retries_only_precommit_source_race_at_next_poll(tmp_path: Path, race: str) -> None:
    import signal
    import time

    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    _checkpoint(tune, "train_trial_A", 5, "retained A", 10)
    assert _run(script, tune, keep=1).returncode == 0
    before = _snapshot(_bank(root))
    new = _checkpoint(tune, "train_trial_B", 5, "initial B", 20)
    if race == "incomplete":
        (new / "trainer.pt").unlink()
    else:
        flag = tmp_path / "mutated-once"
        _inject_helper(script, f'''
_real_copy = shutil.copytree
def _racing_copy(src, dst, **kwargs):
    result = _real_copy(src, dst, **kwargs)
    if not Path({str(flag)!r}).exists():
        (Path(src) / "trainer.pt").write_text("completed B")
        Path({str(flag)!r}).touch()
    return result
shutil.copytree = _racing_copy
''')
    tools, entered, release = _sleep_gate(tmp_path)
    env = {"PATH": f"{tools}:/usr/bin:/bin", "TUNE_DIR": str(tune), "KEEP": "1", "EVERY": "5", "SLEEP": "600"}
    output = tmp_path / "daemon-output"
    errors = tmp_path / "daemon-errors"
    with output.open("w") as out, errors.open("w") as err:
        child = subprocess.Popen(["bash", str(script)], env=env, stdout=out, stderr=err, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not entered.exists():
                assert child.poll() is None, errors.read_text()
                assert time.monotonic() < deadline, "missing controlled poll boundary"
                time.sleep(.01)
            assert "TEMPORARY SOURCE" in errors.read_text()
            assert _snapshot(_bank(root)) == before
            assert not list(_bank(root).glob(".bank-stage-*"))
            if race == "incomplete":
                (new / "trainer.pt").write_text("completed B")
                os.utime(new, ns=(20, 20))
            release.touch()
            deadline = time.monotonic() + 10
            while _markers(_bank(root)) != ["completed B"]:
                assert child.poll() is None, errors.read_text()
                assert time.monotonic() < deadline, "source was not banked on a later poll"
                time.sleep(.01)
            assert child.poll() is None  # Continuous backup survived the race.
        finally:
            release.touch(exist_ok=True)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
    assert child.returncode is not None


@pytest.mark.parametrize("failure", ["unchanged_copy", "cleanup", "partial_trim", "banked_rewrite", "unsafe_destination"])
def test_daemon_does_not_retry_unsafe_or_uncertain_bank_failures(tmp_path: Path, failure: str) -> None:
    root, script = _checkout(tmp_path)
    tune = tmp_path / "tune"
    old = _checkpoint(tune, "train_trial_A", 5, "A", 10)
    assert _run(script, tune, keep=1).returncode == 0
    new = _checkpoint(tune, "train_trial_B", 5, "B", 20)
    tools, entered, _release = _sleep_gate(tmp_path)
    env = {"PATH": f"{tools}:/usr/bin:/bin", "TUNE_DIR": str(tune), "KEEP": "1", "EVERY": "5", "SLEEP": "600"}
    if failure == "unchanged_copy":
        _inject_helper(script, '''
def _failed_copy(*args, **kwargs):
    raise OSError("destination copy failure; source unchanged")
shutil.copytree = _failed_copy
''')
    elif failure == "cleanup":
        _inject_helper(script, '''
_real_copy = shutil.copytree
def _changing_copy(src, dst, **kwargs):
    result = _real_copy(src, dst, **kwargs)
    (Path(src) / "trainer.pt").write_text("source changed")
    return result
def _failed_cleanup(*args, **kwargs):
    raise PermissionError("stage cleanup uncertain")
shutil.copytree = _changing_copy
shutil.rmtree = _failed_cleanup
''')
    elif failure == "partial_trim":
        _inject_helper(script, '''
def _partial_trim(path, **kwargs):
    (Path(path) / "trainer.pt").unlink()
    raise OSError("obsolete entry partially trimmed")
shutil.rmtree = _partial_trim
''')
    elif failure == "banked_rewrite":
        (old / "trainer.pt").write_text("already banked source changed")
    else:
        env["BANK_DIR"] = str(new / "unsafe-bank")
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 1, result.stderr
    assert not entered.exists()  # No later poll after unsafe/uncertain outcome.
    assert "retrying at the next poll" not in result.stderr
    if failure == "partial_trim":
        assert _markers(_bank(root)) == ["B"]
        assert "global KEEP not confirmed" in result.stderr
