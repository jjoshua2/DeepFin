"""Config values and their provenance digest must describe one captured file."""

import builtins
import hashlib
from pathlib import Path

import pytest

from chess_anti_engine.eval import production_shape
from chess_anti_engine.utils.config_yaml import load_yaml_file


@pytest.mark.parametrize("route", ["direct", "live", "arena_fallback"])
@pytest.mark.parametrize("mutation", ["stable", "replace", "delete"])
def test_config_values_and_digest_share_snapshot(tmp_path, monkeypatch, route, mutation):
    from scripts import arena_standard

    original = b"\xef\xbb\xbfselfplay:\r\n  gumbel_vloss_weight: 0.25\r\n"
    replacement = b"selfplay:\n  gumbel_vloss_weight: 0.75\n"
    target = tmp_path / "config.yaml"
    target.write_bytes(original)
    monkeypatch.setenv(production_shape.LIVE_CONFIG_ENV, str(target))
    if route == "arena_fallback":
        monkeypatch.setenv(production_shape.LIVE_CONFIG_ENV, str(tmp_path / "missing.yaml"))
        monkeypatch.setattr(arena_standard, "IN_TREE_CONFIG", target)

    read_bytes = Path.read_bytes
    captures = []

    def capture_then_mutate(path):
        captured = read_bytes(path)
        if path == target:
            captures.append(captured)
            if mutation == "replace":
                staged = tmp_path / "replacement.yaml"
                staged.write_bytes(replacement)
                staged.replace(target)
            elif mutation == "delete":
                target.unlink()
        return captured

    monkeypatch.setattr(Path, "read_bytes", capture_then_mutate)
    if route == "direct":
        cfg, reason = production_shape.load_config_file(
            target, provenance="synthetic fixture", authoritative=True,
        )
        assert reason == ""
    elif route == "live":
        cfg = production_shape.load_live_config()
    else:
        cfg = arena_standard.production_config()

    assert cfg is not None
    assert cfg.path == target
    assert cfg.authoritative is (route != "arena_fallback")
    if route == "arena_fallback":
        assert "in-tree fallback" in cfg.provenance
        assert "does not exist" in cfg.provenance
    assert captures == [original]
    assert cfg.sha256 == hashlib.sha256(original).hexdigest()
    assert cfg.flat["gumbel_vloss_weight"] == 0.25
    assert f"sha256={cfg.sha256[:12]}" in cfg.header()
    if mutation == "delete":
        assert not target.exists()
    else:
        assert read_bytes(target) == (replacement if mutation == "replace" else original)


@pytest.mark.parametrize(
    "raw",
    [
        b"", b"# comment\r\n", b"null\n", b"{}\n",
        b"[1, 2]\n", b"false\n", b"7\n", b"plain scalar\n",
        b"bad: [unterminated\n", b"value: \xff\n",
        "value: caf\u00e9\n".encode(),
        "value: 1\n".encode("utf-16"),
        "value: 1\n".encode("utf-32"),
        b"\xef\xbb\xbfselfplay:\r\n  gumbel_vloss_weight: 0.25\r\n",
        b"value: |\r\n  first\r\n  second\r\n",
    ],
)
def test_snapshot_yaml_matches_existing_text_loader(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_bytes(raw)

    def result(snapshot):
        try:
            if snapshot:
                return load_yaml_file(path, raw_bytes=raw)
            return load_yaml_file(path)
        except Exception as error:
            return type(error), str(error)

    assert result(True) == result(False)


@pytest.mark.parametrize("raw", [b"", b"value: 1\n"])
def test_supplied_snapshot_does_not_recheck_source(tmp_path, monkeypatch, raw):
    path = tmp_path / "already-removed.yaml"

    def unexpected_access(*_args, **_kwargs):
        pytest.fail("captured snapshots must not recheck or reopen their source path")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", unexpected_access)
        patch.setattr(Path, "read_text", unexpected_access)
        assert load_yaml_file(path, raw_bytes=raw) == ({} if not raw else {"value": 1})
    with pytest.raises(FileNotFoundError) as error:
        load_yaml_file(path)
    assert str(error.value) == str(path)


@pytest.mark.parametrize("snapshot", [False, True])
def test_yaml_dependency_diagnostic_is_preserved(tmp_path, monkeypatch, snapshot):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    real_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("synthetic unavailable dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    with pytest.raises(RuntimeError) as error:
        load_yaml_file(path, raw_bytes=b"{}\n" if snapshot else None)
    assert str(error.value) == "PyYAML is required to load config files. Install pyyaml."


@pytest.mark.parametrize(
    ("raw", "diagnostic"),
    [
        (b"[1, 2]\n", "ValueError: YAML root must be a mapping/dict, got list"),
        (b"bad: [unterminated\n", "ParserError:"),
        (b"value: \xff\n", "UnicodeDecodeError:"),
        (b"selfplay:\n  a_key_no_schema_defines: 1\n", "a_key_no_schema_defines"),
    ],
)
def test_config_parse_failures_remain_named(tmp_path, raw, diagnostic):
    path = tmp_path / "config.yaml"
    path.write_bytes(raw)
    cfg, reason = production_shape.load_config_file(
        path, provenance="synthetic fixture", authoritative=True,
    )
    assert cfg is None
    assert reason.startswith(f"{path} does not flatten: ")
    assert diagnostic in reason


def test_config_read_failure_remains_distinct(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")

    def unavailable(_path):
        raise OSError("synthetic read failure")

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    cfg, reason = production_shape.load_config_file(
        path, provenance="synthetic fixture", authoritative=True,
    )
    assert cfg is None
    assert reason == f"{path} cannot be read: OSError: synthetic read failure"
