"""The install roles in pyproject.toml, and the engine's import closure.

``match`` (= the core install) must be exactly what the UCI engine needs: the
test below derives the banned set from pyproject itself (every package some
other extra declares that core does not), then imports the engine in a fresh
interpreter and asserts none of them were loaded. A training-only module-level
import reaching ``uci/`` therefore fails here, not on an operator's install.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Python 3.10 has no tomllib; pytest itself depends on tomli there.
tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = "chess-anti-engine"
ROLE_EXTRAS = ("match", "client", "server", "dev")
# Names other code and docs install by: worker self-update (`[worker]`), README
# (`train,tune,server`), error messages (`tune`, `server`, `train`), CI (`dev`,
# `cpu`), the CUDA dev profile (`cu130`), ONNX export (`onnx`).
LEGACY_EXTRAS = ("worker", "train", "tune", "server", "onnx", "dev", "cpu", "cu130")
# Before the role split these were core dependencies; every pre-existing extra
# must keep installing them so `.[train]` etc. never shrinks.
FORMER_CORE = frozenset({"pyyaml", "zarr"})


def _project() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]


def _name(req: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _self_extras(req: str) -> list[str]:
    m = re.match(rf"\s*{re.escape(SELF)}\[([^\]]+)\]", req)
    return [e.strip() for e in m.group(1).split(",")] if m else []


def _effective(extra: str | None, project: dict, seen: frozenset[str] = frozenset()) -> set[str]:
    """Distribution names a `pip install .[extra]` installs directly (core + extra)."""
    names = {_name(r) for r in project["dependencies"]}
    if extra is None:
        return names
    for req in project["optional-dependencies"][extra]:
        nested = _self_extras(req)
        if nested:
            for sub in nested:
                if sub not in seen:
                    names |= _effective(sub, project, seen | {extra})
        else:
            names.add(_name(req))
    return names


def test_role_and_legacy_extras_exist() -> None:
    extras = _project()["optional-dependencies"]
    for name in (*ROLE_EXTRAS, *LEGACY_EXTRAS):
        assert name in extras, f"extra {name!r} missing from pyproject.toml"


def test_match_is_the_engine_set_and_inside_dev() -> None:
    project = _project()
    match = _effective("match", project)
    assert match == {"python-chess", "numpy", "torch"}
    assert match <= _effective("dev", project)


def test_match_contains_no_training_only_package() -> None:
    project = _project()
    match = _effective("match", project)
    training_only = set()
    for extra in ("client", "server", "train", "tune", "onnx"):
        training_only |= _effective(extra, project) - _effective(None, project)
    assert training_only, "the role extras declare nothing beyond core?"
    assert not (match & training_only), sorted(match & training_only)


def test_dev_is_a_superset_of_every_extra() -> None:
    project = _project()
    dev = _effective("dev", project)
    for extra in project["optional-dependencies"]:
        assert _effective(extra, project) <= dev, f"dev lacks part of [{extra}]"


def test_worker_alias_resolves_to_client() -> None:
    project = _project()
    assert _effective("worker", project) == _effective("client", project)


@pytest.mark.parametrize("extra", ["client", "worker", "train", "tune", "server", "onnx", "dev"])
def test_former_core_dependencies_survive_in_every_role(extra: str) -> None:
    assert _effective(extra, _project()) >= FORMER_CORE


def test_console_script_targets_import_light_cli() -> None:
    assert _project()["scripts"]["deepfin"] == "chess_anti_engine.uci.cli:main"


def _import_names(dist: str) -> set[str]:
    """Top-level import names for a distribution, from installed metadata."""
    owners = importlib.metadata.packages_distributions()
    found = {mod for mod, dists in owners.items()
             if any(d.lower().replace("_", "-") == dist for d in dists)}
    return found or {dist.replace("-", "_")}


def test_engine_import_closure_excludes_non_match_packages() -> None:
    project = _project()
    core = _effective(None, project)
    banned_dists = set()
    for extra in project["optional-dependencies"]:
        banned_dists |= _effective(extra, project) - core
    banned = sorted({mod for dist in banned_dists for mod in _import_names(dist)})
    assert {"zarr", "yaml", "ray", "requests"} <= set(banned)
    code = textwrap.dedent("""
        import json, sys
        import chess_anti_engine.uci.__main__
        import chess_anti_engine.uci.selfcheck
        print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))
        print(json.dumps("chess_anti_engine.worker" in sys.modules))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False,
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    loaded = set(json.loads(lines[-2]))
    assert not (loaded & set(banned)), sorted(loaded & set(banned))
    assert json.loads(lines[-1]) is False, "engine import pulled in the selfplay worker"


def test_engine_compile_cache_does_not_import_the_worker(tmp_path: Path) -> None:
    """The default `--compile` path must not pull in `worker` (zarr/replay).

    It used to `from chess_anti_engine.worker import _configure_shared_compile_cache`,
    so a `match` install could not start the engine with its default flags.
    """
    code = textwrap.dedent(f"""
        import sys
        import chess_anti_engine.compile_cache as cc

        def probe(*, cache_dir):
            print("WORKER_IMPORTED", "chess_anti_engine.worker" in sys.modules, flush=True)
            print("CACHE_DIR", cache_dir, flush=True)
            raise SystemExit(0)

        cc.configure_shared_compile_cache = probe
        from chess_anti_engine.uci.__main__ import main
        sys.argv = ["deepfin", "--checkpoint", "unused", "--compile",
                    "--compile-cache-dir", {str(tmp_path)!r}]
        main()
        raise SystemExit("compile-cache configuration was never reached")
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False,
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WORKER_IMPORTED False" in proc.stdout
    assert f"CACHE_DIR {tmp_path}" in proc.stdout


def test_worker_keeps_its_compile_cache_name() -> None:
    import chess_anti_engine.compile_cache as cc
    import chess_anti_engine.worker as worker

    assert worker._configure_shared_compile_cache is cc.configure_shared_compile_cache
