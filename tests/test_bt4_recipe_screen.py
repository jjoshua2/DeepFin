from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import bt4_recipe_screen as tool
from scripts import bt4_direct_screen as owned
from scripts import bt4_recipe_readout as reader
from tests.test_bt4_recipe_readout import make as make, panel as panel, put


@pytest.mark.parametrize("explicit", [False, True])
def test_owned_stage_forwards_runtime_and_environment(tmp_path, monkeypatch, explicit):
    original = tmp_path / "legacy"
    overlay = tmp_path / "overlay"
    original.mkdir()
    overlay.mkdir()
    monkeypatch.setattr(owned, "RUNTIME", original)
    monkeypatch.setattr(
        owned, "environment", lambda gpu: {**os.environ, "MARKER": "legacy"}
    )
    monkeypatch.setattr(owned, "disk_guard", lambda out: None)
    out = tmp_path / "result"
    cmd = [
        sys.executable,
        "-c",
        'import os; print(os.getcwd()); print(os.environ["MARKER"])',
    ]
    kwargs = (
        {"cwd": overlay, "env": {**os.environ, "MARKER": "overlay"}} if explicit else {}
    )
    receipt = owned.run_owned_stage(
        cmd, out, 35, None, "probe", {}, manifest={}, **kwargs
    )
    assert receipt["process_complete"] is True
    assert receipt["cwd"] == str(overlay if explicit else original)
    assert (out / "probe.log").read_text().splitlines() == [
        str(overlay if explicit else original),
        "overlay" if explicit else "legacy",
    ]
    assert receipt["gpu_seconds"] == 0
    with pytest.raises(ValueError, match="together"):
        owned.run_owned_stage(
            cmd, tmp_path / "bad", 35, None, "probe", {}, manifest={}, cwd=overlay
        )


@pytest.fixture
def package(make, tmp_path, monkeypatch):
    low = make(values=[0.0] * 128)
    high = make(name="high_template", low=False)
    manifests = {"low": low, "high": high}
    base_launch = json.loads(Path(low["launch"]["path"]).read_text())
    m: dict[str, Any] = {
        "schema": 1,
        "profile": "B100_H20",
        "output": str(tmp_path / "screen"),
    }
    for key in ("candidate", "reference", "book", "runtime", "preregistration"):
        m[key] = dict(base_launch["identities"][key])
        m[key].pop("git_sha", None)
    m["candidate"]["role"], m["reference"]["role"] = "B100", "H20"
    m["candidate_training"] = put(
        tmp_path / "train.json", {"training_charge_seconds": 100.0}
    )
    m["reference_training"] = put(tmp_path / "reftrain.json", {})
    m["live_config"] = put(tmp_path / "config.json", {})
    for key, file in [
        ("launcher_sha256", tool.__file__),
        ("reader_sha256", reader.__file__),
        ("supervisor_sha256", owned.__file__),
    ]:
        m[key] = owned.sha(file)
    runtime = tmp_path / "runtime"
    (runtime / "scripts").mkdir(parents=True)
    rt = {"executable": sys.executable, "native_extension_sha256": {}}
    templates = {}
    for stage, manifest in manifests.items():
        templates[stage] = {k: manifest[k]["path"] for k in ("bank", "result")}
        manifest["expected_execution"]["max_seconds"] = tool.MAX_SECONDS
        if stage == "low":
            manifest["expected_execution"]["max_concurrent_games"] = 256
            manifest["expected_execution"]["arena_pool_size"] = 256
    (runtime / "scripts/arena_standard.py").write_text(
        """import json,os,sys,shutil
from pathlib import Path
arg=lambda key:sys.argv[sys.argv.index(key)+1]
low=arg('--sims')=='100'
source=json.loads(os.environ['RECIPE_TEST_TEMPLATES'])['low' if low else 'high']
shutil.copyfile(source['bank'],arg('--games-out'))
record=json.loads(Path(source['result']).read_text())
record.update(argv=sys.argv,game_log=arg('--games-out'),git_sha=os.environ['RECIPE_TEST_HEAD'],max_seconds=5340.0,
 max_concurrent_games=256 if low else128,arena_pool_size=256 if low else128)
Path(arg('--out')).write_text(json.dumps(record)+'\\n')
""".replace("else128", "else 128")
    )
    monkeypatch.setenv("RECIPE_TEST_TEMPLATES", json.dumps(templates))
    monkeypatch.setenv("RECIPE_TEST_HEAD", tool.OVERLAY_HEAD)
    proof = {
        "status": "PASS_RECIPE_SCREEN_PREPARATION",
        "inputs": {
            k: m[k]
            for k in (
                "candidate",
                "reference",
                "candidate_training",
                "reference_training",
                "book",
                "runtime",
                "live_config",
                "preregistration",
            )
        },
        "opening_panel": low["opening_panel"],
        "observed": {
            "prefix_matches": True,
            "cuda_initialized": False,
            "runtime": {"executable": sys.executable},
            "settings": {k: v["expected_settings"] for k, v in manifests.items()},
            "execution": {k: v["expected_execution"] for k, v in manifests.items()},
        },
    }
    proof.update(
        {k: m[k] for k in ("launcher_sha256", "reader_sha256", "supervisor_sha256")}
    )
    m["preparation"] = put(tmp_path / "prepared.json", proof)
    monkeypatch.setattr(tool, "inputs", lambda m: (runtime, rt))
    monkeypatch.setattr(tool, "storage", lambda *args: {})
    monkeypatch.setattr(
        owned, "qualified_search", lambda: low["expected_settings"]["search_candidate"]
    )
    monkeypatch.setattr(owned, "disk_guard", lambda out: None)

    @contextlib.contextmanager
    def lease(_out):
        with (tmp_path / "fake-lease").open("a") as f:
            yield f.fileno()

    monkeypatch.setattr(tool, "gpu_lease", lease)
    return m, manifests


def test_actual_synthetic_child_negative_low_still_runs_high(package):
    m, _ = package
    tool.execute(m)
    out = Path(m["output"])
    low = json.loads((out / "low/readout.stdout.json").read_text())
    assert low["sprt"]["verdict"] == "H0"
    assert (out / "high/complete.json").is_file()
    high = json.loads((out / "high/readout.stdout.json").read_text())
    assert high["fixed_core_cross_budget"]["bootstrap"]["samples"] == 10000
    for stage, width in [("low", 256), ("high", 128)]:
        process = json.loads((out / stage / "process.json").read_text())
        assert process["command"][
            process["command"].index("--max-concurrent-games") + 1
        ] == str(width)
        assert "--no-rolling" not in process["command"]
        assert process["hard_seconds"] == 5400
        assert process["supervisor_command"][3] == "5370s"
    assert json.loads((out / "complete.json").read_text())["complete"] is True


def test_invalid_low_preserves_failure_and_never_dispatches_high(package):
    m, templates = package
    path = Path(templates["low"]["bank"]["path"])
    path.write_text(path.read_text() + "{broken")
    with pytest.raises(reader.InvalidCell, match="invalid low"):
        tool.execute(m)
    out = Path(m["output"])
    assert (out / "low/readout.stdout.json").exists()
    assert (out / "failed.json").exists()
    assert not (out / "high").exists()
    assert not (out / "complete.json").exists()


def test_changed_preparation_pin_refused_before_stage(package):
    m, _ = package
    Path(m["preparation"]["path"]).write_text("{}")
    with pytest.raises(reader.InvalidCell, match="changed pin"):
        tool.execute(m)
    assert not Path(m["output"]).exists()


def test_stop_at_stage_boundary_keeps_high_absent(package, monkeypatch):
    m, _ = package
    real_certify = tool.certify

    def stop_after_low(cell, path, out):
        result = real_certify(cell, path, out)
        (out.parent / "STOP").touch()
        return result

    monkeypatch.setattr(tool, "certify", stop_after_low)
    with pytest.raises(reader.InvalidCell, match="STOP"):
        tool.execute(m)
    assert not (Path(m["output"]) / "high").exists()


def test_cpu_preparation_settings_match_actual_cli_defaults(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    import chess
    import numpy as np
    import torch

    from scripts import arena_standard as arena
    from chess_anti_engine.uci import model_loader

    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _n: None)
    loads = []

    def load(path, *, device):
        loads.append((path, device))
        return SimpleNamespace(use_dynamic_relations=False)

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", load)

    def openings(_path, *, n_pairs, max_plies, rng):
        assert max_plies == 16
        assert rng.bit_generator.state == np.random.default_rng(42).bit_generator.state
        return [chess.Board()] * n_pairs

    monkeypatch.setattr(arena, "load_paired_openings", openings)
    monkeypatch.setattr(arena, "_configure_shared_compile_cache", lambda path: None)
    side = arena.apply_search_overrides(
        arena.resolve_search_shape("training"), spec="policy_temp=1.0"
    )
    monkeypatch.setattr(owned, "qualified_search", side.as_record)
    rt = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "native_extensions": {},
        "native_extension_sha256": {},
    }
    m = {
        "candidate": {"path": str(tmp_path / "b.pt")},
        "reference": {"path": str(tmp_path / "h.pt")},
        "book": {"path": str(tmp_path / "book.zip")},
    }
    Path(m["book"]["path"]).touch()

    def run_probe(cmd, **kwargs):
        assert cmd[:4] == [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "270s",
        ]
        assert kwargs["timeout"] == 305
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        i = cmd.index("-c")
        monkeypatch.setattr(sys, "argv", ["-c", *cmd[i + 2 :]])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cmd[i + 1], {})
        return output.getvalue()

    monkeypatch.setattr(tool.subprocess, "check_output", run_probe)
    proof = tool.cpu_probe(m, tmp_path, rt)
    assert loads == [(m["candidate"]["path"], "cpu"), (m["reference"]["path"], "cpu")]
    assert proof["dynamic_relations"] == [False, False]
    original_builder = arena.arena_game_log_settings
    original_run = arena.run_arena

    class HeaderCaptured(Exception):
        pass

    captured = {}

    def header(**kwargs):
        captured["header"] = original_builder(**kwargs)
        raise HeaderCaptured

    def run(**kwargs):
        captured["arguments"] = kwargs
        original_run(**kwargs)

    monkeypatch.setattr(arena, "arena_game_log_settings", header)
    monkeypatch.setattr(arena, "run_arena", run)
    for stage in ("low", "high"):
        cmd = tool.command(m, rt, stage, tmp_path / stage)
        monkeypatch.setattr(sys, "argv", cmd[1:])
        with pytest.raises(HeaderCaptured):
            arena.main()
        assert captured["header"] == proof["settings"][stage]
        assert captured["arguments"]["tb_max_pieces"] == 6
        assert captured["arguments"]["rolling"] is True
        assert captured["arguments"]["max_seconds"] == 5340.0


def test_stop_during_high_readout_refuses_package_completion(package, monkeypatch):
    m, _ = package
    real_certify = tool.certify

    def stop_after_high(cell, path, out):
        result = real_certify(cell, path, out)
        if out.name == "high":
            (out.parent / "STOP").touch()
        return result

    monkeypatch.setattr(tool, "certify", stop_after_high)
    with pytest.raises(reader.InvalidCell, match="STOP"):
        tool.execute(m)
    out = Path(m["output"])
    assert (out / "high/readout.stdout.json").exists()
    assert not (out / "high/complete.json").exists()
    assert not (out / "complete.json").exists()
    assert (out / "failed.json").exists()
