#!/usr/bin/env python3
"""Two fixed arena stages for qualified original-corpus recipe checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts import bt4_direct_screen as owned
from scripts import bt4_recipe_readout as reader

HERE = Path(__file__).resolve().parent.parent
OVERLAY_HEAD = "1b8e5036290789ec3a037339af0150e9f6fa76d8"
OVERLAY = {
    "scripts/arena_standard.py": "fc6f42f5cc566ef1cd1948112d1ae9cde151851750438ba621b4b51327458095",
    "chess_anti_engine/eval/sprt.py": "8e8937c660783981636eed2e0b4b51bac62e8c5ad2055878ddaf994779d7fec1",
}
LOOKAHEAD_OVERLAY_HEAD = "82298a5d4010e7097f473712e3720de229522427"
LOOKAHEAD_OVERLAY = {
    "scripts/arena_standard.py": "c7df9061d119f0f53de90b529a38a641b4fabce88f3eab9afb37a80f11cd6a96",
    "chess_anti_engine/eval/sprt.py": "5f778573d4a702e625b3b6d3220536076dd41fd866c749bda9c9fb8aff749056",
}


def arena_overlay(m: dict[str, Any]) -> tuple[str, dict[str, str]]:
    if reader.lookahead_pairs(m) is not None:
        return LOOKAHEAD_OVERLAY_HEAD, LOOKAHEAD_OVERLAY
    return OVERLAY_HEAD, OVERLAY


H20_SHA = "0a711fcf10ff87fc8360d3fd4b3035b170a7c15172616317c4a9687ae99d7017"
HARD_SECONDS = 5400
MAX_SECONDS = 5340.0


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": owned.sha(path)}


def write(path: Path, value: Any) -> dict[str, str]:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return pin(path)


def environment(runtime: Path, *, gpu: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CHESS_LIVE_PRODUCTION_CONFIG", None)
    env.update(
        PYTHONPATH=str(runtime),
        CUDA_VISIBLE_DEVICES="0" if gpu else "",
        CHESS_ANTI_ENGINE_LIVE_CONFIG=str(runtime / "configs/pbt2_small.yaml"),
        PYTHONUNBUFFERED="1",
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLOSC_NTHREADS",
    ):
        env[key] = "2"
    return env


def inputs(m: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate immutable upstream qualification; no model loading or inference."""
    overlay_head, overlay = arena_overlay(m)
    reader.require(m["schema"] == 1 and m["profile"] in ("B100_H20", reader.MATCHED_PROFILE), "unknown profile")
    for name, filename in [
        ("launcher_sha256", __file__),
        ("reader_sha256", reader.__file__),
        ("supervisor_sha256", owned.__file__),
    ]:
        owned.pin(filename, m[name])
    reader.pinned(m["preregistration"])
    reader.require(m["book"]["sha256"] == owned.BOOK_SHA, "unregistered book")
    owned.pin(m["book"]["path"], m["book"]["sha256"])
    roles = (reader.matched_training_pair(m) if m["profile"] == reader.MATCHED_PROFILE
             else ("B100", "H20"))
    for key, role in zip(("candidate", "reference"), roles):
        reader.require(m[key]["role"] == role, "candidate/reference direction differs")
        owned.pin(m[key]["path"], m[key]["sha256"])
        # Existing complete training, checkpoint, summary and canonical schedule contract.
        owned.verify_candidate_training(
            {"candidate": m[key], "candidate_training": m[key + "_training"]}
        )
    if m["profile"] == "B100_H20":
        reader.require(m["reference"]["sha256"] == H20_SHA, "reference is not qualified H20")
    frozen = reader.read_json(m["runtime"])
    reader.require(
        frozen["status"] == "CPU_QUALIFIED_INACTIVE_RUNTIME_IDENTITY",
        "runtime not qualified",
    )
    heads = frozen["identities"]["heads"]
    reader.require(len(heads) == 1, "ambiguous runtime root")
    runtime = Path(next(iter(heads)))
    reader.require(
        runtime.is_absolute() and heads[str(runtime)] == overlay_head,
        "wrong CUDA overlay",
    )
    reader.same(
        subprocess.check_output(
            ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
        ).strip(),
        overlay_head,
        "overlay head",
    )
    reader.require(
        not subprocess.check_output(
            [
                "git",
                "-C",
                str(runtime),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip(),
        "dirty overlay",
    )
    qualification = reader.read_json(frozen["qualification"])
    reader.require(
        qualification["status"]
        == "PASS_INACTIVE_ORDERED_ARENA_RUNTIME_CPU_QUALIFICATION"
        and qualification["runtime_commit"] == overlay_head,
        "wrong runtime qualification",
    )
    reader.same(
        qualification["difference"]["exact_merged_overlay_hashes"],
        overlay,
        "qualified overlay",
    )
    for name, digest in overlay.items():
        owned.pin(runtime / name, digest)
    original = reader.read_json(frozen["original_runtime_manifest"])
    rt = frozen["runtime"]
    for key in ("python", "executable", "torch", "cuda", "numpy"):
        reader.same(rt[key], original["runtime"][key], "historical runtime " + key)
    reader.require(
        rt["torch"] == "2.11.0+cu128"
        and rt["numpy"] == "1.26.2"
        and rt["cuda"] == "12.8"
        and rt["python"].startswith("3.10.12"),
        "wrong old stack",
    )
    reader.require(len(rt["native_extensions"]) == 4, "missing native extensions")
    for path, digest in rt["native_extension_sha256"].items():
        owned.pin(path, digest)
    reader.require(
        set(rt["native_extensions"].values()) == set(rt["native_extension_sha256"]),
        "native identity coverage differs",
    )
    reader.pinned(m["live_config"])
    reader.require(
        Path(m["live_config"]["path"]) == runtime / "configs/pbt2_small.yaml",
        "wrong live config",
    )
    return runtime, rt


def storage(
    m: dict[str, Any], runtime: Path, rt: dict[str, Any]
) -> dict[str, list[int]]:
    paths = {
        Path(__file__).resolve(),
        Path(reader.__file__).resolve(),
        Path(owned.__file__).resolve(),
    }
    for key in (
        "candidate",
        "reference",
        "book",
        "runtime",
        "live_config",
        "preregistration",
        "candidate_training",
        "reference_training",
    ):
        paths.add(Path(m[key]["path"]))
    for role in ("candidate", "reference"):
        receipt = reader.read_json(m[role + "_training"])
        paths.add(Path(receipt["run"]) / "summary.json")
        paths.add(Path(receipt["schedule"]["path"]))
    paths.update(runtime / name for name in arena_overlay(m)[1])
    paths.update(Path(name) for name in rt["native_extension_sha256"])
    result = {}
    for path in paths:
        st = path.stat()
        result[str(path)] = [
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        ]
    return result


def cpu_probe(m: dict[str, Any], runtime: Path, rt: dict[str, Any]) -> dict[str, Any]:
    """Use the actual frozen PGN sampler; full history is retained, never FEN-only."""
    code = """import contextlib,importlib,json,sys
with contextlib.redirect_stdout(sys.stderr):
 import numpy as np, torch
 from pathlib import Path
 from scripts import arena_standard as a
 from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
 torch.set_num_threads(2)
 torch.set_num_interop_threads(2)
 relations=[]
 for checkpoint in sys.argv[2:4]:
  model=load_model_from_checkpoint(checkpoint,device='cpu')
  relations.append(bool(getattr(model,'use_dynamic_relations',False)))
  del model
 if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
 side=a.apply_search_overrides(a.resolve_search_shape('training'),spec='policy_temp=1.0')
 panels=[]
 for count in (500,128):
  boards=a.load_paired_openings(Path(sys.argv[1]),n_pairs=count,max_plies=16,rng=np.random.default_rng(42))
  panels.append([dict(root_fen=b.root().fen(),moves=[x.uci() for x in b.move_stack],fen=b.fen()) for b in boards])
 if panels[0][:128]!=panels[1]: raise ValueError('book history prefix differs')
 settings={};execution={}
 for name,games,sims,pool in [('low',1000,100,256),('high',256,400,128)]:
  settings[name]=a.arena_game_log_settings(mode='matched_sims',candidate=sys.argv[2],reference=sys.argv[3],games=games,seed=42,
   openings_path=sys.argv[1],openings_kind='book',opening_plies=16,sims_candidate=sims,sims_reference=sims,ms_per_move=None,
   max_plies=300,temperature=.1,gumbel_add_noise=True,search_candidate=side,search_reference=side,
   volatility_candidate=None,uci_args='',syzygy_path=None,tb_max_pieces=6)
  leaf=a.arena_uncapped_leaf_rows(max_concurrent_games=pool,sides=(side,side),relations=tuple(relations))
  if leaf>4096: raise ValueError('evaluator cap binds')
  execution[name]=dict(loop='rolling',compile='on',eval_hoist='4096',eval_max_batch=4096,eval_leaf_cap_uncapped=leaf,
   max_concurrent_games=pool,arena_pool_size=pool,max_seconds=5340.0,hard_seconds=5400)
 actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
  native_extensions={name:importlib.import_module(name).__file__ for name in json.loads(sys.argv[4])})
print(json.dumps(dict(panel=panels[0],prefix_matches=True,settings=settings,execution=execution,runtime=actual,dynamic_relations=relations,cuda_initialized=torch.cuda.is_initialized())))
"""
    cmd = [
        rt["executable"],
        "-c",
        code,
        m["book"]["path"],
        m["candidate"]["path"],
        m["reference"]["path"],
        json.dumps(list(rt["native_extensions"])),
    ]
    result = json.loads(
        subprocess.check_output(
            owned.timeout_command(cmd, 300),
            cwd=runtime,
            env=environment(runtime, gpu=False),
            text=True,
            timeout=305,
        )
    )
    allowance = reader.lookahead_pairs(m)
    if allowance is not None:
        result["settings"]["low"]["sprt_lookahead_pairs"] = allowance
    reader.same(
        result["runtime"],
        {k: v for k, v in rt.items() if k != "native_extension_sha256"},
        "actual runtime",
    )
    reader.same(
        result["settings"]["low"]["search_candidate"],
        owned.qualified_search(),
        "qualified C100 search",
    )
    return result


def prepare(m: dict[str, Any], out: Path) -> None:
    reader.require(
        not out.exists() and out.is_absolute(),
        "preparation output must be new absolute path",
    )
    runtime, rt = inputs(m)
    before = storage(m, runtime, rt)
    result = cpu_probe(m, runtime, rt)
    reader.same(storage(m, runtime, rt), before, "preparation source stability")
    panel_pin = write(out.with_suffix(".panel.json"), result.pop("panel"))
    reader.opening_panel(panel_pin)
    write(
        out,
        {
            "schema": 1,
            **({"sprt_lookahead_pairs": m["sprt_lookahead_pairs"]} if "sprt_lookahead_pairs" in m else {}),
            **({"profile": m["profile"]} if m["profile"] == reader.MATCHED_PROFILE else {}),
            "status": "PASS_RECIPE_SCREEN_PREPARATION",
            "opening_panel": panel_pin,
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
            "launcher_sha256": m["launcher_sha256"],
            "reader_sha256": m["reader_sha256"],
            "supervisor_sha256": m["supervisor_sha256"],
            "observed": result,
            "history_proof": "Frozen PGN sampler, fresh NumPy default_rng(42) at500 and128; exact root/moves/endpoint prefix. Arenas use that same book CLI.",
        },
    )


def command(m: dict[str, Any], rt: dict[str, Any], stage: str, out: Path) -> list[str]:
    low = stage == "low"
    allowance = reader.lookahead_pairs(m)
    cmd = [
        rt["executable"],
        "scripts/arena_standard.py",
        "--candidate",
        m["candidate"]["path"],
        "--reference",
        m["reference"]["path"],
        "--games",
        "1000" if low else "256",
        "--mode",
        "matched_sims",
        "--sims",
        "100" if low else "400",
        "--seed",
        "42",
        "--openings",
        m["book"]["path"],
        "--opening-plies",
        "16",
        "--max-plies",
        "300",
        "--temperature",
        "0.1",
        "--search-shape",
        "training",
        "--cand-gumbel",
        "policy_temp=1.0",
        "--ref-gumbel",
        "policy_temp=1.0",
        "--compile",
        "on",
        "--device",
        "cuda",
        "--max-concurrent-games",
        "256" if low else "128",
        "--eval-max-batch",
        "4096",
        "--max-seconds",
        str(MAX_SECONDS),
        "--games-out",
        str(out / "arena.games.jsonl"),
        "--out",
        str(out / "arena.results.jsonl"),
    ]
    if low:
        cmd += [
            "--sprt",
            "elo0=0,elo1=15,alpha=.05,beta=.10,first_pairs=128,step_pairs=64",
        ]
    if low and allowance is not None:
        cmd += ["--sprt-lookahead-pairs", str(allowance)]
    return cmd


def stopped(out: Path) -> None:
    reader.require(not (out / "STOP").exists(), "STOP requested")
    owned.disk_guard(out)


@contextlib.contextmanager
def gpu_lease(out: Path):
    with (owned.ROOT / "scratchpad/gpu0_experiment.lock").open("a") as lease:
        while True:
            stopped(out)
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            reader.require(
                not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    text=True,
                    timeout=10,
                ).strip(),
                "competing GPU process",
            )
            yield lease.fileno()
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def certify(cell: dict[str, Any], path: Path, out: Path) -> dict[str, Any]:
    identity = write(path, cell)
    cmd = [
        sys.executable,
        str(Path(reader.__file__).resolve()),
        "--manifest",
        identity["path"],
        "--expected-manifest-sha256",
        identity["sha256"],
    ]
    env = environment(HERE, gpu=False)
    result = subprocess.run(
        owned.timeout_command(cmd, 120),
        cwd=HERE,
        env=env,
        text=True,
        capture_output=True,
        timeout=125,
        check=False,
    )
    # Preserve invalid reader output and stderr before interpreting the return code.
    (out / "readout.stdout.json").write_text(result.stdout)
    (out / "readout.stderr.log").write_text(result.stderr)
    reader.require(
        result.returncode == 0,
        f"operationally invalid {out.name} readout: {result.returncode}",
    )
    return json.loads(result.stdout)


def execute(m: dict[str, Any]) -> None:
    runtime, rt = inputs(m)
    proof = reader.read_json(m["preparation"])
    reader.same(reader.lookahead_pairs(proof), reader.lookahead_pairs(m), "prepared lookahead")
    if m["profile"] == reader.MATCHED_PROFILE:
        reader.same(proof.get("profile"), m["profile"], "prepared profile")
    reader.require(
        proof["status"] == "PASS_RECIPE_SCREEN_PREPARATION", "preparation not complete"
    )
    reader.require(
        set(proof["inputs"])
        == {
            "candidate",
            "reference",
            "candidate_training",
            "reference_training",
            "book",
            "runtime",
            "live_config",
            "preregistration",
        },
        "incomplete preparation inputs",
    )
    for key, value in proof["inputs"].items():
        reader.same(m[key], value, "prepared " + key)
    for key in ("launcher_sha256", "reader_sha256", "supervisor_sha256"):
        reader.same(proof[key], m[key], "prepared " + key)
    reader.opening_panel(proof["opening_panel"])
    reader.require(
        proof["observed"]["prefix_matches"] is True
        and proof["observed"]["cuda_initialized"] is False,
        "history or CPU preparation incomplete",
    )
    reader.same(
        proof["observed"]["runtime"],
        {k: v for k, v in rt.items() if k != "native_extension_sha256"},
        "prepared runtime",
    )
    for settings in proof["observed"]["settings"].values():
        reader.same(
            settings["search_candidate"], owned.qualified_search(), "prepared search"
        )
        reader.same(
            settings["search_reference"],
            owned.qualified_search(),
            "prepared reference search",
        )
    out = Path(m["output"])
    reader.require(
        out.is_absolute() and out.parent.is_dir() and not out.exists(),
        "output must be new",
    )
    stopped(out)
    out.mkdir()
    write(out / "manifest.json", m)
    low_manifest = None
    training_charge = reader.read_json(m["candidate_training"])[
        "training_charge_seconds"
    ]
    reader.require(
        type(training_charge) in (int, float) and 0 < training_charge <= 16200,
        "training charge exceeds cap",
    )
    charges = 0.0
    try:
        for stage in ("low", "high"):
            stopped(out)
            stage_out = out / stage
            cmd = command(m, rt, stage, stage_out)
            settings, execution = (
                proof["observed"]["settings"][stage],
                proof["observed"]["execution"][stage],
            )
            reader.check_command(
                cmd,
                settings,
                execution,
                stage_out / "arena.games.jsonl",
                stage_out / "arena.results.jsonl",
                stage == "low",
            )
            identities = {
                k: {x: m[k][x] for x in ("path", "sha256")}
                for k in (
                    "candidate",
                    "reference",
                    "book",
                    "runtime",
                    "preregistration",
                )
            }
            identities["runtime"]["git_sha"] = arena_overlay(m)[0]
            if m["profile"] == reader.MATCHED_PROFILE:
                identities.update({k: m[k] for k in ("candidate_training", "reference_training")})
            launch = {
                "settings": settings,
                "execution": execution,
                "opening_panel": proof["opening_panel"],
                "candidate_role": m["candidate"]["role"],
                "reference_role": m["reference"]["role"],
                "command": cmd,
                "identities": identities,
                "preparation": m["preparation"],
            }
            if m["profile"] == reader.MATCHED_PROFILE:
                launch.update(profile=m["profile"], training={
                    k: m[k] for k in ("candidate", "reference", "candidate_training", "reference_training")
                })
            launch_pin = write(out / f"{stage}.launch.json", launch)
            with gpu_lease(out) as fd:
                # Recheck after any lease wait, before reading weights in the arena.
                inputs(m)
                before = storage(m, runtime, rt)
                process = owned.run_owned_stage(
                    cmd,
                    stage_out,
                    HARD_SECONDS,
                    fd,
                    "arena",
                    {},
                    manifest=m,
                    stop_paths=(out / "STOP",),
                    cwd=runtime,
                    env=environment(runtime, gpu=True),
                )
            reader.same(storage(m, runtime, rt), before, "arena input stability")
            charges += process["gpu_seconds"]
            reader.require(charges <= 2 * HARD_SECONDS, "arena package cap exceeded")
            cell = {
                "schema": 1,
                "mode": "low_sprt" if stage == "low" else "high_fixed128",
                "bank": pin(stage_out / "arena.games.jsonl"),
                "result": pin(stage_out / "arena.results.jsonl"),
                "process": pin(stage_out / "process.json"),
                "launch": launch_pin,
                "opening_panel": proof["opening_panel"],
                "expected_settings": settings,
                "expected_execution": execution,
            }
            if m["profile"] == reader.MATCHED_PROFILE:
                cell["profile"] = m["profile"]
            if stage == "high":
                cell["low_manifest"] = low_manifest
            cell_path = out / f"{stage}.reader_manifest.json"
            stopped(out)
            report = certify(cell, cell_path, stage_out)
            stopped(out)
            reader.require(
                report["status"] == "VALID_CELL", "reader did not certify cell"
            )
            write(
                stage_out / "complete.json",
                {
                    "complete": True,
                    "readout": pin(stage_out / "readout.stdout.json"),
                    "reader_manifest": pin(cell_path),
                    "gpu_seconds": process["gpu_seconds"],
                },
            )
            if stage == "low":
                low_manifest = pin(cell_path)
        stopped(out)
        write(
            out / "complete.json",
            {
                "complete": True,
                "profile": m["profile"],
                **({"candidate_role": m["candidate"]["role"], "reference_role": m["reference"]["role"]}
                   if m["profile"] == reader.MATCHED_PROFILE else {}),
                "gpu_seconds": charges,
                "training_gpu_seconds": training_charge,
                "package_gpu_seconds": training_charge + charges,
                "low": pin(out / "low/complete.json"),
                "high": pin(out / "high/complete.json"),
                "promotion": "NONE; same-seed development screen",
            },
        )
    except BaseException as error:
        write(
            out / "failed.json",
            {
                "complete": False,
                "error": repr(error),
                "gpu_seconds_completed_stages": charges,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--out", type=Path, help="new CPU preparation receipt; prepare only"
    )
    args = parser.parse_args()
    m = reader.read_json(
        {"path": str(args.manifest), "sha256": args.expected_manifest_sha256}
    )
    if args.prepare:
        reader.require(args.out is not None, "--prepare requires --out")
        prepare(m, args.out)
    else:
        reader.require(args.out is None, "--out is preparation only")
        execute(m)


if __name__ == "__main__":
    main()
