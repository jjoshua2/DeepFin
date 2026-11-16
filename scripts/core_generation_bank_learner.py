"""Finite new-family learning from caller-trusted, postclosure accepted banks.

Logical collection generation, learner segment and optimizer step are distinct.
No collector, promotion writer, target profile, or interrupted-stage recovery.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import copy
import inspect
import os
from pathlib import Path
import random
import re
import subprocess
from types import SimpleNamespace
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_generation_bank_acceptance as admission

CONTRACT = "core_generation_bank_learning_v1"
STAGE = "core_generation_bank_stage_v1"
BOUNDARY = "COMPLETED_CORE_GENERATION_BANK_SEGMENT_V1"
CHECKPOINT_KEY = "core_generation_bank_learner_v1"
SOURCE_FILES = (*core.SOURCE_FILES, "scripts/core_generation_bank_acceptance.py", "scripts/core_generation_bank_learner.py")
ANCHOR_CONTRACT = "core_generation_bank_policy_anchor_v1"
ANCHOR_STAGE = "core_generation_bank_policy_anchor_stage_v1"
ANCHOR_BOUNDARY = "COMPLETED_CORE_GENERATION_BANK_POLICY_ANCHOR_SEGMENT_V1"
ANCHOR_CHECKPOINT_KEY = "core_generation_bank_policy_anchor_v1"
ANCHOR_SOURCE_FILES = (*SOURCE_FILES, "scripts/core_generation_bank_policy_anchor.py", "chess_anti_engine/train/frozen_policy_anchor.py", "chess_anti_engine/policy_anchor_spec.py")


def plain(value: Any) -> Any:
    """Copy validated readonly metadata to existing dict/list helper contracts."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def settings(c: dict[str, Any]) -> dict[str, Any]:
    """Share strict training-setting validation, never alias a data manifest."""
    return {key: c[key] for key in ("run_uuid", "source_revision", "source_files", "architecture", "batch_size",
        "games_per_generation", "max_games_per_generation", "replay_generations", "seed", "runtime", "lr", "mirror_prob", "limits")} | {
            "schema": "core_finite_learning_v1", "donor": c["initial_model"], "updates_per_generation": c["updates_per_segment"]}


def validate_contract(c: dict[str, Any], *, _anchored: bool = False) -> dict[str, Any]:
    core.require(set(c) == ({"schema", "run_uuid", "source_revision", "source_files", "architecture", "initial_model",
        "initial_actor_step", "first_generation", "batch_size", "games_per_generation", "max_games_per_generation",
        "updates_per_segment", "replay_generations", "seed", "runtime", "lr", "mirror_prob", "limits", "bank_limits"} | ({"policy_anchor"} if _anchored else set()))
        and c["schema"] == (ANCHOR_CONTRACT if _anchored else CONTRACT), "invalid bank learning contract")
    core.integer(c["first_generation"], "first logical generation")
    core.integer(c["initial_actor_step"], "initial actor step", 0)
    core.require(set(ANCHOR_SOURCE_FILES if _anchored else SOURCE_FILES) <= set(c["source_files"]), "missing bank learner source pins")
    if _anchored:
        from chess_anti_engine.policy_anchor_spec import validate_spec

        core.require(validate_spec(c["policy_anchor"], c["architecture"]) == c["policy_anchor"], "noncanonical policy anchor")
    core.require(set(c["bank_limits"]) == {"max_file_bytes", "max_total_bytes", "max_games"}, "invalid bank metadata limits")
    for name, value in c["bank_limits"].items():
        core.integer(value, name)
    core.require(c["bank_limits"]["max_file_bytes"] <= (128 << 20), "bank metadata cap exceeds128MiB")
    projected = settings(c)
    core.validate_contract(projected)
    return projected


def source_guard(c: dict[str, Any]) -> None:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    core.require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core.REPO, env=env, text=True).strip()
                 == c["source_revision"], "bank learner source revision mismatch")
    core.require(not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=core.REPO, env=env), "dirty bank learner source")
    for name, expected in c["source_files"].items():
        core.require(core.file_sha(core.REPO / name) == expected, "bank learner source changed")
    for module, relative in ((core, "scripts/core_finite_learner.py"), (admission, "scripts/core_generation_bank_acceptance.py")):
        core.require(Path(inspect.getfile(module)).resolve() == core.REPO / relative, "foreign bank learner helper")
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_generation_bank_learner.py", "foreign bank learner entry")


def metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str], *, _anchored: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    core.require(set(stage) == {"schema", "contract_sha256", "segment", "banks", "parent", "output"}
                 and stage["schema"] == (ANCHOR_STAGE if _anchored else STAGE) and stage["contract_sha256"] == contract_ref["sha256"], "invalid bank learner stage")
    segment = core.integer(stage["segment"], "segment")
    core.require(segment <= len(c["updates_per_segment"]) and type(stage["banks"]) is list
                 and len(stage["banks"]) == segment, "bank learner segment horizon mismatch")
    generations, identities = [], []
    physical_ids: set[str] = set()
    for index, binding in enumerate(stage["banks"]):
        core.require(type(binding) is dict and set(binding) == {"root_acceptance", "independent_review"}, "invalid accepted-bank binding")
        bank = admission.validate_accepted_generation(cast(dict[str, str], binding["root_acceptance"]),
            cast(dict[str, str], binding["independent_review"]), **c["bank_limits"])
        identity = plain(bank.identity)
        core.require(identity["generation"] == c["first_generation"] + index, "nonconsecutive logical bank generation")
        core.require(core.canonical(identity["architecture"]) == core.canonical(c["architecture"]), "bank/learner architecture mismatch")
        games = plain(bank.games)
        core.require(c["games_per_generation"] <= len(games) <= c["max_games_per_generation"], "bank game count mismatch")
        for game in games:
            pid = game["physical_game_id"]
            core.require(pid not in physical_ids, "duplicate physical game across banks")
            physical_ids.add(pid)
            core.verify_inventory(game["shard"])
        generations.append({"generation": identity["generation"], "games": games})
        identities.append(identity)
    core.require(identities[0]["actor_sha256"] == c["initial_model"]["sha256"]
                 and identities[0]["actor_checkpoint_step"] == c["initial_actor_step"], "initial model/actor provenance mismatch")
    parent = None
    if segment == 1:
        core.require(stage["parent"] is None, "first segment must bootstrap model only")
    else:
        parent = closed_parent(stage["parent"], c, contract_ref, stage, _anchored=_anchored)
        core.require(identities[-1]["actor_sha256"] == parent["checkpoint"]["sha256"]
                     and identities[-1]["actor_checkpoint_step"] == parent["step"], "bank actor is not adjacent learner checkpoint")
        core.require(parent["bank_identities"] == identities[:-1], "parent historical bank identities changed")
    return generations, identities, parent


def closed_parent(binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str], stage: dict[str, Any], *, _anchored: bool = False) -> dict[str, Any]:
    core.require(type(binding) is dict and set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"}, "invalid bank parent binding")
    path, parent = core.read_ref(binding["boundary"])
    _, prior = core.read_ref(parent["stage"])
    output = core.external(prior["output"])
    core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
                 and not (output / "boundary.pending.json").exists(), "failed or incomplete bank parent")
    core.require(set(prior) == {"schema", "contract_sha256", "segment", "banks", "parent", "output"}
                 and prior["schema"] == (ANCHOR_STAGE if _anchored else STAGE) and prior["contract_sha256"] == contract_ref["sha256"]
                 and prior["segment"] == parent["segment"] == stage["segment"] - 1
                 and prior["banks"] == stage["banks"][:-1] == parent["accepted_banks"]
                 and prior["parent"] == parent["parent"], "bank parent original stage mismatch")
    suffix = ["-m", "scripts.core_generation_bank_policy_anchor" if _anchored else "scripts.core_generation_bank_learner", "--contract", contract_ref["path"],
              "--contract-sha256", contract_ref["sha256"], "--stage", parent["stage"]["path"], "--stage-sha256", parent["stage"]["sha256"]]
    command = binding["command"]
    core.require(type(command) is list and len(command) > len(suffix) and all(type(arg) is str and arg for arg in command)
                 and command[-len(suffix):] == suffix, "bank parent command mismatch")
    core.require(type(binding["owner_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]), "invalid bank parent owner hash")
    _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
    core.require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
                 and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
                 and terminal["supervisor_sha256"] == binding["owner_sha256"] and terminal["command"] == command,
                 "bank parent owner not closed PASS")
    core.require(parent["status"] == (ANCHOR_BOUNDARY if _anchored else BOUNDARY) and parent["contract_sha256"] == contract_ref["sha256"]
                 and parent["source_revision"] == c["source_revision"] and parent["run_uuid"] == c["run_uuid"]
                 and parent["logical_generation"] == c["first_generation"] + stage["segment"] - 2
                 and parent["step"] == sum(c["updates_per_segment"][:stage["segment"] - 1])
                 and parent["initial_model"] == c["initial_model"] and parent["initial_actor_step"] == c["initial_actor_step"],
                 "bank parent family/contract/source/state mismatch")
    if _anchored:
        core.require(parent["policy_anchor"] == c["policy_anchor"], "parent frozen reference identity differs")
    checkpoint = core.external(parent["checkpoint"]["path"])
    core.require(checkpoint == output / "learner.pt" and not (checkpoint.stat().st_mode & 0o222)
                 and core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint custody differs")
    return parent


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str], segment: int, *, _anchored: bool = False) -> dict[str, Any]:
    return {"contract_sha256": contract_ref["sha256"], "run_uuid": c["run_uuid"], "source_revision": c["source_revision"],
            "segment": segment, "logical_generation": c["first_generation"] + segment - 1} | ({"policy_anchor": c["policy_anchor"]} if _anchored else {})


def fresh_model(trainer: Any, c: dict[str, Any], torch: Any) -> None:
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION
    from chess_anti_engine.tune.fresh_rl_donor import load_fresh_donor
    from chess_anti_engine.tune.startup_intent import SCHEMA, contract_digest
    from scripts.bootstrap_checkpoint_resume import require_equal

    scheduler = copy.deepcopy(trainer._scheduler.state_dict())
    zclip = copy.deepcopy(trainer.zclip_state_dict())
    values = {"lr": c["lr"]["peak"], "optimizer": "nadamw", "lr_schedule": "cosine", "batch_size": c["batch_size"], "accum_steps": 1}
    config = {**values, "rl_startup_intent": "fresh_bootstrap", "bootstrap_handoff_mode": "fresh_rl_complete_v1",
              "bootstrap_checkpoint": c["initial_model"]["path"], "bootstrap_checkpoint_sha256": c["initial_model"]["sha256"],
              "bootstrap_model_contract": c["architecture"], "rl_startup_contract": {"schema": SCHEMA, "run_uuid": c["run_uuid"],
              "donor_sha256": c["initial_model"]["sha256"], "model_contract_sha256": contract_digest(c["architecture"]), "settings": values}}
    load_fresh_donor(config=config, trainer=trainer, checkpoint=None, restore=SimpleNamespace(seed_warmstart_used=False),
                     torch_module=torch, schema_version=ARCH_SCHEMA_VERSION)
    core.require(trainer.step == 0 and not trainer.opt.state and trainer._swa_model is None, "bank bootstrap state not fresh")
    require_equal(trainer._scheduler.state_dict(), scheduler, "fresh scheduler")
    require_equal(trainer.zclip_state_dict(), zclip, "fresh zclip")


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    return _execute(contract_ref, stage_ref, _anchored=False)


def _execute(contract_ref: dict[str, str], stage_ref: dict[str, str], *, _anchored: bool) -> dict[str, Any]:
    _, c = core.read_ref(contract_ref)
    _, stage = core.read_ref(stage_ref)
    projected = validate_contract(c, _anchored=_anchored)
    source_guard(c)
    generations, identities, parent = metadata(c, stage, contract_ref, _anchored=_anchored)
    output = core.external(stage["output"])
    core.require(not core.failure_marker(output).exists(), "failed bank segment output cannot be reused")
    output.mkdir(parents=False, exist_ok=False)
    trainer = None
    drawn = applied = 0
    try:
        import numpy as np
        import torch
        from scripts.bootstrap_checkpoint_resume import resume_bootstrap
        from scripts.offline_epoch_resume import _restore_rng, _rng_state

        for helper, name in ((resume_bootstrap, "scripts/bootstrap_checkpoint_resume.py"), (_rng_state, "scripts/offline_epoch_resume.py")):
            core.require(Path(inspect.getfile(helper)).resolve() == core.REPO / name, "foreign bank state helper")
        core.require((torch.__version__, np.__version__) == (c["runtime"]["torch_version"], c["runtime"]["numpy_version"]), "bank runtime versions differ")
        torch.set_num_threads(c["runtime"]["torch_threads"])
        backend = core.backend_profile(torch)
        core.require((torch.cuda.device_count() if torch.cuda.is_available() else 0) == c["runtime"]["cuda_devices"], "bank CUDA visibility differs")
        if parent is not None:
            core.require(parent["backend"] == backend, "bank continuation backend changed")
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        rng = np.random.default_rng(c["seed"])
        with (output / "draws.jsonl").open("x") as journal:
            def on_draw(rows: Any) -> None:
                nonlocal drawn
                journal.write(core.canonical({"draw_batch": drawn // c["batch_size"], "rows": [asdict(row) for row in rows]}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                drawn += len(rows)

            # Complete actual payload admission/decode before any model construction.
            replay = core.load_replay(projected, generations, rng, on_draw, first_generation=c["first_generation"])
            core.import_custody()
            trainer = core.make_trainer(projected, output / "tb")
            if parent is None:
                fresh_model(trainer, c, torch)
            else:
                checkpoint = core.external(parent["checkpoint"]["path"])
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                core.require(state.get(ANCHOR_CHECKPOINT_KEY if _anchored else CHECKPOINT_KEY) == checkpoint_identity(c, contract_ref, stage["segment"] - 1, _anchored=_anchored), "bank checkpoint family tag mismatch")
                core.require((CHECKPOINT_KEY if _anchored else ANCHOR_CHECKPOINT_KEY) not in state, "ambiguous bank checkpoint family")
                if _anchored:
                    from chess_anti_engine.policy_anchor_spec import CHECKPOINT_KEY as REFERENCE_KEY

                    core.require(state.get(REFERENCE_KEY) == c["policy_anchor"], "saved frozen reference identity differs")
                del state
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed during tag read")
                resume_bootstrap(trainer, checkpoint, parent["checkpoint"]["sha256"], parent["step"], c["seed"])
                # resume_bootstrap seeds Torch; restore ALL five streams afterwards.
                if not _anchored:
                    _restore_rng(parent["rng"], rng, cast(Any, np.random))
                    core.require(core.canonical(_rng_state(rng, cast(Any, np.random))) == core.canonical(parent["rng"]), "bank RNG restoration differs")
            if _anchored:
                trainer.configure_frozen_policy_anchor(c["policy_anchor"])
                if parent is not None:
                    _restore_rng(parent["rng"], rng, cast(Any, np.random))
                    core.require(core.canonical(_rng_state(rng, cast(Any, np.random))) == core.canonical(parent["rng"]), "bank RNG restoration differs")
            core.import_custody()
            start = trainer.step
            count = c["updates_per_segment"][stage["segment"] - 1]

            def counted(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
                nonlocal applied
                applied += 1

            hook = trainer.opt.register_step_post_hook(counted)
            try:
                metrics = trainer.train_steps(replay, batch_size=c["batch_size"], steps=count)
            finally:
                hook.remove()
            core.require(trainer.step == start + count and applied == count == metrics.train_steps_done, "bank applied update count mismatch")
            core.require(metrics.grad_nonfinite_skip_rate == 0 and metrics.transient_cuda_retry_batches == 0, "bank skips/retries refused")
            core.require(drawn == count * c["batch_size"] == metrics.train_samples_seen, "bank draw/applied view mismatch")
        if c["runtime"]["device"].startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint = output / "learner.pt"
        trainer.save(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tag_key = ANCHOR_CHECKPOINT_KEY if _anchored else CHECKPOINT_KEY
        core.require(tag_key not in state, "bank checkpoint already tagged")
        state[tag_key] = checkpoint_identity(c, contract_ref, stage["segment"], _anchored=_anchored)
        pending = checkpoint.with_name("learner.pt.family.pending")
        with pending.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, checkpoint)
        del state
        trainer.writer.close()
        source_guard(c)
        core.import_custody()
        final_generations, final_identities, final_parent = metadata(c, stage, contract_ref, _anchored=_anchored)
        core.require(final_generations == generations and final_identities == identities and final_parent == parent,
                     "accepted bank/parent metadata changed during segment")
        boundary = {"status": ANCHOR_BOUNDARY if _anchored else BOUNDARY, "run_uuid": c["run_uuid"], "source_revision": c["source_revision"],
            "contract_sha256": contract_ref["sha256"], "stage": stage_ref, "parent": stage["parent"],
            "segment": stage["segment"], "logical_generation": identities[-1]["generation"], "accepted_banks": stage["banks"],
            "bank_identities": identities, "initial_model_policy": "model_only_fresh_optimizer",
            "state_handoff": "model_only_bootstrap" if parent is None else "adjacent_full_state_continuation", "initial_model": c["initial_model"], "initial_actor_step": c["initial_actor_step"],
            "step_start": start, "step": trainer.step, "applied_updates": applied, "applied_views": applied * c["batch_size"],
            "checkpoint": {"path": str(checkpoint), "sha256": core.file_sha(checkpoint)},
            "drawn_rows": drawn, "draws_sha256": core.file_sha(output / "draws.jsonl"), "membership": replay.membership,
            "metrics": asdict(metrics), "rng": _rng_state(rng, cast(Any, np.random)), "backend": backend,
            "recovery_scope": "same-family closed segment boundary only; interrupted output is terminal/unqualified",
            "champion_changed": False, "collector_lineage_qualified": False}
        if _anchored:
            boundary.update(policy_anchor=trainer.frozen_policy_anchor_state(), policy_anchor_diagnostics=trainer.frozen_policy_anchor_diagnostics())
        core.publish_boundary(output, boundary)
        return boundary
    except BaseException as error:
        failure = {"status": "FAILED_CORE_GENERATION_BANK_POLICY_ANCHOR_NO_RECOVERY_V1" if _anchored else "FAILED_CORE_GENERATION_BANK_SEGMENT_NO_RECOVERY_V1", "error": repr(error),
                   "contract": contract_ref, "stage": stage_ref, "drawn_rows": drawn, "completed_optimizer_calls": applied}
        if _anchored and trainer is not None:
            failure["policy_anchor_diagnostics"] = trainer.frozen_policy_anchor_diagnostics()
        try:
            marker = core.failure_marker(output)
            core.write_json(marker, failure)
            marker.chmod(0o444)
            core.fsync_directory(marker.parent)
        finally:
            if trainer is not None:
                trainer.writer.close()
            core.seal(output)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256}, {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
