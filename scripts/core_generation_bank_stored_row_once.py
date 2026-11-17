"""Finite one-pass stored-row family over fixed accepted banks.

Additive entry point; historical bank, anchor and exact-epoch families are not
relabeled. Only full batches are supported. A failed output is never resumed.
Continuation requires the complete immutable closed journal chain and full-state
checkpoint with the same permutation/cursor. No launcher or training authority.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import os
from pathlib import Path
import random
import re
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_generation_bank_acceptance as admission
from scripts import core_generation_bank_learner as bank
from chess_anti_engine.replay.stored_row_once import (
    SAMPLING, QuotaLedger, StoredRowOnceReplay, verify_journals,
)

CONTRACT = "core_generation_bank_stored_row_once_v1"
ANCHOR_CONTRACT = "core_generation_bank_policy_anchor_stored_row_once_v1"
STAGE = "core_generation_bank_stored_row_once_stage_v1"
BOUNDARY = "COMPLETED_CORE_GENERATION_BANK_STORED_ROW_ONCE_SEGMENT_V1"
CHECKPOINT_KEY = "core_generation_bank_stored_row_once_v1"
SOURCE_FILES = (*bank.ANCHOR_SOURCE_FILES, "chess_anti_engine/replay/stored_row_once.py",
                "scripts/core_generation_bank_stored_row_once.py")


def validate_contract(c: dict[str, Any], *, _anchored: bool) -> dict[str, Any]:
    core.require(c.get("schema") == (ANCHOR_CONTRACT if _anchored else CONTRACT), "stored-row family schema")
    legacy = dict(c)
    sampling = legacy.pop("sampling", None)
    legacy["schema"] = bank.ANCHOR_CONTRACT if _anchored else bank.CONTRACT
    projected = bank.validate_contract(legacy, _anchored=_anchored)
    core.require(set(SOURCE_FILES) <= set(c["source_files"]), "missing stored-row source pins")
    core.require(type(sampling) is dict and set(sampling) == {
        "schema", "order_seed", "expected_rows", "presentations_per_row", "mirror_shares_quota",
        "partial_batch", "lineage_uuid", "quota_ledger"}, "stored-row sampling contract")
    core.require(sampling["schema"] == SAMPLING and sampling["presentations_per_row"] == 1
                 and type(sampling["presentations_per_row"]) is int
                 and sampling["mirror_shares_quota"] is True
                 and sampling["partial_batch"] == "REFUSE_BEFORE_MODEL_CONSTRUCTION"
                 and sampling["lineage_uuid"] == c["run_uuid"], "stored-row quota/lineage contract")
    ledger = core.external(sampling["quota_ledger"])
    core.require(str(ledger) == sampling["quota_ledger"], "noncanonical quota ledger")
    core.integer(sampling["order_seed"], "stored-row order seed", 0)
    n = core.integer(sampling["expected_rows"], "stored-row expected population")
    core.require(n <= c["limits"]["max_rows"] and n % c["batch_size"] == 0,
                 "last partial batch unsupported: refuse before first forward")
    core.require(sum(c["updates_per_segment"]) * c["batch_size"] == n,
                 "stored-row dose must exhaust exactly once, no drop/pad/repeat")
    return projected


def source_guard(c: dict[str, Any]) -> None:
    bank.source_guard(c)
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_generation_bank_stored_row_once.py",
                 "foreign stored-row entry")
    core.require(Path(inspect.getfile(StoredRowOnceReplay)).resolve()
                 == core.REPO / "chess_anti_engine/replay/stored_row_once.py", "foreign stored-row sampler")


def metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str], *,
             _anchored: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    core.require(set(stage) == {"schema", "contract_sha256", "segment", "banks", "parent", "output"}
                 and stage["schema"] == STAGE and stage["contract_sha256"] == contract_ref["sha256"],
                 "invalid stored-row stage")
    segment = core.integer(stage["segment"], "stored-row segment")
    core.require(segment <= len(c["updates_per_segment"]) and type(stage["banks"]) is list
                 and 1 <= len(stage["banks"]) <= c["replay_generations"], "fixed stored-row bank horizon")
    core.require(not core.external(c["sampling"]["quota_ledger"]).is_relative_to(core.external(stage["output"])),
                 "stored-row quota ledger inside sealed segment output")
    generations: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    physical_ids: set[str] = set()
    for index, binding in enumerate(stage["banks"]):
        core.require(type(binding) is dict and set(binding) == {"root_acceptance", "independent_review"},
                     "invalid accepted stored-row bank binding")
        accepted = admission.validate_accepted_generation(binding["root_acceptance"], binding["independent_review"],
                                                         **c["bank_limits"])
        if "exclusion_index" in accepted.evidence:
            core.require("scripts/core_exclusion_index.py" in c["source_files"], "missing exclusion learner source")
        identity = bank.plain(accepted.identity)
        games = bank.plain(accepted.games)
        core.require(identity["generation"] == c["first_generation"] + index
                     and core.canonical(identity["architecture"]) == core.canonical(c["architecture"]),
                     "stored-row bank generation/architecture mismatch")
        core.require(c["games_per_generation"] <= len(games) <= c["max_games_per_generation"], "stored-row bank game bounds")
        for game in games:
            core.require(game["physical_game_id"] not in physical_ids, "duplicate physical game across banks")
            physical_ids.add(game["physical_game_id"])
            core.verify_inventory(game["shard"])
        generations.append({"generation": identity["generation"], "games": games})
        identities.append(identity)
    core.require(identities[0]["actor_sha256"] == c["initial_model"]["sha256"]
                 and identities[0]["actor_checkpoint_step"] == c["initial_actor_step"], "stored-row initial actor identity")
    core.require(sum(game["rows"] for gen in generations for game in gen["games"])
                 == c["sampling"]["expected_rows"], "accepted stored-row population/dose mismatch")
    parent = None
    if segment == 1:
        core.require(stage["parent"] is None, "first stored-row segment must bootstrap")
    else:
        parent = closed_parent(stage["parent"], c, contract_ref, stage, _anchored=_anchored)
        core.require(parent["bank_identities"] == identities, "fixed stored-row banks changed across segments")
    return generations, identities, parent


def closed_parent(binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str],
                  stage: dict[str, Any], *, _anchored: bool) -> dict[str, Any]:
    core.require(type(binding) is dict and set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"},
                 "stored-row closed parent binding")
    path, parent = core.read_ref(binding["boundary"])
    _, prior = core.read_ref(parent["stage"])
    output = core.external(prior["output"])
    core.require(path == output / "boundary.json" and not core.failure_marker(output).exists()
                 and not (output / "boundary.pending.json").exists(), "failed/incomplete stored-row parent")
    core.require(set(prior) == {"schema", "contract_sha256", "segment", "banks", "parent", "output"}
                 and prior["schema"] == STAGE and prior["contract_sha256"] == contract_ref["sha256"]
                 and prior["segment"] == parent["segment"] == stage["segment"] - 1
                 and prior["banks"] == stage["banks"] == parent["accepted_banks"]
                 and prior["parent"] == parent["parent"], "stored-row parent stage/roster mismatch")
    suffix = ["-m", "scripts.core_generation_bank_stored_row_once", "--contract", contract_ref["path"],
              "--contract-sha256", contract_ref["sha256"], "--stage", parent["stage"]["path"],
              "--stage-sha256", parent["stage"]["sha256"]]
    command = binding["command"]
    core.require(type(command) is list and len(command) > len(suffix)
                 and all(type(arg) is str and arg for arg in command) and command[-len(suffix):] == suffix,
                 "stored-row parent command mismatch")
    core.require(type(binding["owner_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]),
                 "stored-row parent owner hash")
    _, terminal = core.read_ref(binding["owner_terminal"], readonly=False)
    core.require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
                 and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
                 and terminal["supervisor_sha256"] == binding["owner_sha256"] and terminal["command"] == command,
                 "stored-row parent not closed PASS")
    core.require(parent["status"] == BOUNDARY and parent["contract_sha256"] == contract_ref["sha256"]
                 and parent["source_revision"] == c["source_revision"] and parent["run_uuid"] == c["run_uuid"]
                 and parent["sampling"] == c["sampling"]
                 and parent["step"] == sum(c["updates_per_segment"][:stage["segment"] - 1])
                 and parent["initial_model"] == c["initial_model"] and parent["initial_actor_step"] == c["initial_actor_step"],
                 "stored-row parent family/state mismatch")
    if _anchored:
        core.require(parent["policy_anchor"] == c["policy_anchor"], "stored-row frozen reference changed")
    checkpoint = core.external(parent["checkpoint"]["path"])
    core.require(checkpoint == output / "learner.pt" and not checkpoint.stat().st_mode & 0o222
                 and core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "stored-row parent checkpoint custody")
    return parent


def checkpoint_identity(c: dict[str, Any], contract_ref: dict[str, str], segment: int, *,
                        _anchored: bool = False) -> dict[str, Any]:
    return {"contract_sha256": contract_ref["sha256"], "run_uuid": c["run_uuid"],
            "source_revision": c["source_revision"], "segment": segment,
            "sampling": c["sampling"]} | ({"policy_anchor": c["policy_anchor"]} if _anchored else {})


def validate_closed_segment(contract_ref: dict[str, str], binding: dict[str, Any], *,
                            require_complete: bool = True) -> dict[str, Any]:
    """Validate this new family explicitly; never relabel a legacy delivery.

    This returns bounded closed-family metadata, not a Root acceptance, scientific
    eligibility, historical exposure audit, serving qualification or launch grant.
    """
    _, c = core.read_ref(contract_ref)
    anchored = c.get("schema") == ANCHOR_CONTRACT
    validate_contract(c, _anchored=anchored)
    source_guard(c)
    _, candidate = core.read_ref(binding["boundary"])
    _, prior_stage = core.read_ref(candidate["stage"])
    generations, _, _ = metadata(c, prior_stage, contract_ref, _anchored=anchored)
    next_stage = dict(prior_stage, segment=prior_stage["segment"] + 1)
    closed = closed_parent(binding, c, contract_ref, next_stage, _anchored=anchored)
    membership = tuple((gen["generation"], core.digest(game["shard"]["files"]), game["rows"])
                       for gen in generations for game in gen["games"])
    class Population:
        rng = None
        def __init__(self) -> None:
            self.membership = membership
        def __len__(self) -> int:
            return sum(row[2] for row in self.membership)
    def forbidden(_rows: Any) -> None:
        raise ValueError("metadata consumer cannot draw training rows")
    replay = StoredRowOnceReplay(Population(), seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                                 expected_rows=c["sampling"]["expected_rows"], on_draw=forbidden)
    replay.restore(closed["quota_state"], closed["draw_journals"],
                   max_file_bytes=c["bank_limits"]["max_file_bytes"], max_total_bytes=c["bank_limits"]["max_total_bytes"])
    core.require(closed["quota_state"]["cursor"] == closed["step"] * c["batch_size"]
                 and closed["membership"] == [list(row) for row in membership]
                 and closed["applied_views"] == closed["drawn_rows"] == closed["metrics"]["train_samples_seen"]
                 and closed["applied_updates"] == closed["metrics"]["train_steps_done"]
                 and closed["metrics"]["transient_cuda_retry_batches"] == 0
                 and closed["metrics"]["grad_nonfinite_skip_rate"] == 0, "closed stored-row counters differ")
    complete = replay.remaining == 0 and closed["segment"] == len(c["updates_per_segment"])
    core.require(closed["pass_complete"] is complete and (complete or not require_complete), "stored-row pass incomplete")
    core.require(closed["quota_ledger_checkpoint"]["path"] == c["sampling"]["quota_ledger"], "closed quota ledger identity")
    return {"contract": contract_ref, "boundary": binding["boundary"], "checkpoint": closed["checkpoint"],
            "run_uuid": c["run_uuid"], "source_revision": c["source_revision"], "pass_complete": complete,
            "population_rows": len(replay), "cumulative_presentations": closed["quota_state"]["cursor"],
            "optimizer_updates": closed["step"], "row_identity": ["generation", "source_sha256", "row_index"],
            "presentations_per_row": 1, "mirror_shares_quota": True,
            "scope": "closed stored-row lineage only; no authority or historical/global exposure claim"}


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    _, c = core.read_ref(contract_ref)
    core.require(c.get("schema") in (CONTRACT, ANCHOR_CONTRACT), "not stored-row contract family")
    return _execute(contract_ref, stage_ref, _anchored=c["schema"] == ANCHOR_CONTRACT)


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
    draw_batches = 0
    prior_journals: list[dict[str, str]] = []
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
        prior_journals = [] if parent is None else parent["draw_journals"]
        with QuotaLedger(core.external(c["sampling"]["quota_ledger"]), first=parent is None,
                         expected_journals=prior_journals, max_file_bytes=c["bank_limits"]["max_file_bytes"],
                         max_total_bytes=c["bank_limits"]["max_total_bytes"]) as ledger, (output / "draws.jsonl").open("x") as journal:
            if parent is not None:
                core.require(ledger.checkpoint() == parent["quota_ledger_checkpoint"], "quota ledger/checkpoint mismatch")
            def on_draw(rows: Any) -> None:
                nonlocal drawn, draw_batches
                encoded = core.canonical({"draw_batch": sum(c["updates_per_segment"][:stage["segment"] - 1]) + draw_batches,
                                          "rows": [asdict(row) for row in rows]}) + "\n"
                core.require(journal.tell() + len(encoded.encode()) <= c["bank_limits"]["max_file_bytes"], "local draw journal byte cap")
                ledger.reserve(encoded.encode())
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())
                drawn += len(rows)
                draw_batches += 1

            # Complete actual payload admission/decode before any model construction.
            loaded = core.load_replay(projected, generations, rng, None, first_generation=c["first_generation"])
            replay = StoredRowOnceReplay(loaded, seed=c["sampling"]["order_seed"], batch_size=c["batch_size"],
                                         expected_rows=c["sampling"]["expected_rows"], on_draw=on_draw)
            if parent is not None:
                prior_journals = parent["draw_journals"]
                replay.restore(parent["quota_state"], prior_journals,
                               max_file_bytes=c["bank_limits"]["max_file_bytes"],
                               max_total_bytes=c["bank_limits"]["max_total_bytes"])
                core.require(parent["quota_state"]["cursor"] == parent["step"] * c["batch_size"],
                             "parent checkpoint/journal cumulative dose differs")
            core.require(replay.remaining >= c["updates_per_segment"][stage["segment"] - 1] * c["batch_size"],
                         "stored-row segment overdraw before model construction")
            core.import_custody()
            trainer = core.make_trainer(projected, output / "tb")
            if parent is None:
                bank.fresh_model(trainer, c, torch)
            else:
                checkpoint = core.external(parent["checkpoint"]["path"])
                core.require(core.file_sha(checkpoint) == parent["checkpoint"]["sha256"], "bank parent checkpoint changed")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                core.require(state.get(CHECKPOINT_KEY) == {"identity": checkpoint_identity(c, contract_ref, stage["segment"] - 1, _anchored=_anchored),
                                                             "quota_state": parent["quota_state"], "draw_journals": prior_journals,
                                                             "quota_ledger_checkpoint": parent["quota_ledger_checkpoint"]}, "bank checkpoint family tag mismatch")
                core.require(bank.CHECKPOINT_KEY not in state and bank.ANCHOR_CHECKPOINT_KEY not in state,
                             "legacy checkpoint cannot masquerade as stored-row family")
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
            quota_ledger_checkpoint = ledger.checkpoint()
        journal_path = output / "draws.jsonl"
        journal_path.chmod(0o444)
        draw_journals = [*prior_journals, {"path": str(journal_path), "sha256": core.file_sha(journal_path)}]
        quota_state = replay.state()
        observed = verify_journals(replay, draw_journals, max_file_bytes=c["bank_limits"]["max_file_bytes"],
                                   max_total_bytes=c["bank_limits"]["max_total_bytes"])
        core.require(observed == {key: quota_state[key] for key in ("cursor", "batches", "row_prefix_sha256")},
                     "current checkpoint/journal cursor differs")
        core.require(quota_state["cursor"] == trainer.step * c["batch_size"], "cumulative optimizer/quota dose mismatch")
        if stage["segment"] == len(c["updates_per_segment"]):
            core.require(replay.remaining == 0, "completed pass must exhaust every stored row once")
        if c["runtime"]["device"].startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint = output / "learner.pt"
        trainer.save(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tag_key = CHECKPOINT_KEY
        core.require(tag_key not in state, "bank checkpoint already tagged")
        state[tag_key] = {"identity": checkpoint_identity(c, contract_ref, stage["segment"], _anchored=_anchored),
                          "quota_state": quota_state, "draw_journals": draw_journals,
                          "quota_ledger_checkpoint": quota_ledger_checkpoint}
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
        boundary = {"status": BOUNDARY, "run_uuid": c["run_uuid"], "source_revision": c["source_revision"],
            "contract_sha256": contract_ref["sha256"], "stage": stage_ref, "parent": stage["parent"],
            "segment": stage["segment"], "logical_generation": identities[-1]["generation"], "sampling": c["sampling"],
            "quota_state": quota_state, "draw_journals": draw_journals,
            "quota_ledger_checkpoint": quota_ledger_checkpoint, "pass_complete": replay.remaining == 0, "accepted_banks": stage["banks"],
            "bank_identities": identities, "initial_model_policy": "model_only_fresh_optimizer",
            "state_handoff": "model_only_bootstrap" if parent is None else "adjacent_full_state_continuation", "initial_model": c["initial_model"], "initial_actor_step": c["initial_actor_step"],
            "step_start": start, "step": trainer.step, "applied_updates": applied, "applied_views": metrics.train_samples_seen,
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
        failure = {"status": "FAILED_STORED_ROW_ONCE_SEGMENT_NO_RECOVERY_V1", "error": repr(error),
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
