"""Research entry points never put their torch.compile cache inside a checkout.

The arena used to resolve the production worker cache, which lives in the live
tree (``runs/pbt2_small/server/worker_cache``), so research compiles wrote
there. The default is now outside every checkout, overridable by
``--compile-cache-dir`` and ``$CAE_RESEARCH_COMPILE_CACHE_DIR``. The
distributed worker's own shared cache is not affected.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from scripts import arena_standard as arena

REPO_ROOT = Path(__file__).resolve().parent.parent


def _inside_repo(path: Path) -> bool:
    return path.resolve().is_relative_to(REPO_ROOT)


@pytest.fixture
def no_production_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse() -> Path:
        raise AssertionError("the research compile cache must not be read from the production yaml")
    monkeypatch.setattr(arena, "production_config_path", _refuse)
    monkeypatch.delenv(arena.RESEARCH_COMPILE_CACHE_ENV, raising=False)


@pytest.mark.usefixtures("no_production_yaml")
def test_default_is_outside_the_repo_and_ignores_the_production_yaml() -> None:
    path = arena.default_compile_cache_dir()
    assert path.is_absolute()
    assert not _inside_repo(path), path
    assert path == Path(arena.RESEARCH_COMPILE_CACHE_DEFAULT).expanduser().resolve()


@pytest.mark.usefixtures("no_production_yaml")
def test_env_var_overrides_the_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(arena.RESEARCH_COMPILE_CACHE_ENV, str(tmp_path / "cc"))
    assert arena.default_compile_cache_dir() == (tmp_path / "cc").resolve()


@pytest.mark.parametrize("flag", [False, True])
def test_arena_main_configures_a_cache_outside_the_repo(monkeypatch: pytest.MonkeyPatch,
                                                        tmp_path: Path, flag: bool) -> None:
    """End to end through main(), compiled run: the cache it configures is the
    flag's when given, else the research default -- never inside the repo."""
    monkeypatch.delenv(arena.RESEARCH_COMPILE_CACHE_ENV, raising=False)
    seen: list[Path] = []
    monkeypatch.setattr(arena, "_configure_shared_compile_cache", seen.append)
    monkeypatch.setattr(arena, "run_arena", lambda **_k: {"elo": 0.0, "elo_ci95": [0.0, 0.0], "pairs": 0})
    argv = ["arena_standard.py", "--candidate", "a.pt", "--reference", "b.pt", "--games", "2",
            "--mode", "matched_sims", "--search-shape", "play", "--device", "cuda", "--compile", "on"]
    if flag:
        argv += ["--compile-cache-dir", str(tmp_path / "flag")]
    monkeypatch.setattr("sys.argv", argv)
    arena.main()
    assert len(seen) == 1
    assert not _inside_repo(seen[0]), seen[0]
    assert seen[0] == ((tmp_path / "flag").resolve() if flag else arena.default_compile_cache_dir())


# Every research script that points inductor at a cache, and the only forms it
# may use: the arena's research default, its own flag, or a home-cache literal.
_ALLOWED = re.compile(
    r"_configure_shared_compile_cache\(\s*(default_compile_cache_dir\(\)|cache_dir\)|"
    r"\s*cache_dir=Path\(\"~/\.cache/deepfin/worker_cache\"\)\.expanduser\(\))"
)


def test_no_research_script_resolves_the_worker_cache_in_the_repo() -> None:
    callers: dict[str, Any] = {}
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        text = script.read_text()
        for m in re.finditer(r"_configure_shared_compile_cache\(", text):
            if text[max(0, m.start() - 4):m.start()] == "def ":
                continue
            callers.setdefault(script.name, []).append(bool(_ALLOWED.match(text, m.start())))
    assert {"arena_standard.py", "match_vs_handicapped_sf.py"} <= set(callers)
    bad = {name: ok for name, ok in callers.items() if not all(ok)}
    assert not bad, f"unrecognised compile-cache source in {sorted(bad)}"
