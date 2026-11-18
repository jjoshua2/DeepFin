"""Passive proof for an actual capped-cycle update, without extra forwards."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_closed_once_actor as actor_bridge
from scripts import core_capped_on_policy_cycle as entry
from scripts import core_selfplay_union_policy_anchor_stored_row_once_proof as anchor_proof
from scripts import core_selfplay_union_stored_row_once_proof as plain_proof
from scripts.core_selfplay_union_stored_row_once_proof import aliases, file_hash, fingerprint, stamp

SCHEMA = "core_capped_cycle_own_live_saved_payload_proof_v1"
STATUS = "CAPPED_CYCLE_OWN_LIVE_SAVED_PAYLOAD_PENDING_ROOT_CLOSURE_V1"


def snapshot(trainer: Any, c: dict[str, Any]) -> dict[str, Any]:
    return (plain_proof.state_snapshot(trainer) if c["policy_anchor"] is None
            else anchor_proof.state_snapshot(trainer))


class OwnLiveProof(anchor_proof.OwnLiveProof):
    """Reuse passive hooks/save brackets; new cycle setup and tag are explicit."""

    def _snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        trainer, c = context["trainer"], context["contract"]
        if c["policy_anchor"] is None:
            core.require(trainer._frozen_policy_anchor_identity is None and trainer._frozen_policy_anchor is None,
                         "unanchored cycle configured a teacher")
            reference = None
        else:
            reference = anchor_proof.reference_snapshot(context)
        expected = {key: c["retained_search_share"] if key == "search_wdl_frac" else core.CORE_KWARGS[key]
                    for key in ("w_policy", "w_wdl", "sf_wdl_frac", "search_wdl_frac",
                                "use_adjusted_wdl_target", "wdl_terminal_outcome_plies", "wdl_terminal_outcome_sf_frac")}
        core.require({key: trainer._loss_kwargs[key] for key in expected} == expected, "actual capped cycle target shares")
        actual_recipe = entry.optimizer_recipe(trainer)
        core.require(core.canonical(actual_recipe) == core.canonical(context["optimizer_recipe"]),
                     "actual cycle optimizer/scheduler recipe differs")
        return {"state": fingerprint(snapshot(trainer, c), context["torch"]),
                "model": anchor_proof.model_facts(trainer, context["torch"], c["architecture"]),
                "rng": self._rng(context), "optimizer_names": trainer._optimizer_param_names(),
                "optimizer_recipe": actual_recipe, "reference": reference}

    def before_train(self, context: dict[str, Any]) -> None:
        core.require(self.context is None, "cycle observer cannot be reused")
        self.context = context
        trainer, c, parent = context["trainer"], context["contract"], context["parent"]
        core.require(context["population"].identity == {k:v for k,v in c["population"].items() if k != "plan"}
            and trainer.step == parent["student_step"] and trainer.grad_clip_mode == "none",
            "actual cycle setup/cumulative step/noClip")
        core.require(Path(inspect.getfile(type(trainer))).resolve() == core.REPO / "scripts/core_finite_learner.py",
                     "foreign cycle Trainer")
        self.initial = self._observe(context, lambda: self._snapshot(context))
        if parent["fresh"]:
            core.require(trainer.step == 0 and not trainer.opt.state
                and trainer.fresh_rl_donor_provenance["sha256"] == c["initial_model"]["sha256"],
                "cycle initial model-only optimizer was not fresh")
            self.initial.update(fresh_optimizer_slots=len(trainer.opt.state), initial_student_step=trainer.step)
            self.initial["initial_donor_provenance"] = dict(trainer.fresh_rl_donor_provenance)
        else:
            core.require(self.initial["state"] == parent["boundary"]["own_live_certificate"]["own_live_state_fingerprint"],
                         "cycle initial full model/optimizer/scheduler state differs from parent")
        self.initial.update(actual_student_device=str(trainer.device), actual_Trainer_use_amp=trainer.use_amp,
            backend=context["backend"], source_origins={"entry":str(Path(entry.__file__).resolve()),
            "proof":str(Path(__file__).resolve()), "trainer":str(Path(inspect.getfile(type(trainer))).resolve()),
            "model":str(Path(inspect.getfile(type(trainer.model))).resolve()),
            "population":str(Path(inspect.getfile(type(context["population"]))).resolve()),
            "torch":str(Path(context["torch"].__file__).resolve()), "numpy":str(Path(context["np"].__file__).resolve())},
            actual_torch_version=context["torch"].__version__, actual_numpy_version=context["np"].__version__)
        core.require(str(trainer.device) == c["runtime"]["device"] and trainer.use_amp is c["runtime"]["use_amp"],
                     "cycle actual device/AMP")
        def forward(_model: Any, _args: Any) -> None:
            self.counts["forward"] += 1
        def pre(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            self.counts["optimizer_pre"] += 1
        def post(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            self.counts["optimizer_post"] += 1
        self.hooks.extend((trainer.model.register_forward_pre_hook(forward),
            trainer.opt.register_step_pre_hook(pre), trainer.opt.register_step_post_hook(post)))
        if trainer._frozen_policy_anchor is not None:
            def reference_forward(_model: Any, _args: Any) -> None:
                self.reference_calls += 1
            self.hooks.append(trainer._frozen_policy_anchor.model.register_forward_pre_hook(reference_forward))

    def closed_checkpoint(self, context: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
        core.require(context is self.context and self.save_after, "observer final callback ordering")
        trainer, torch, c = context["trainer"], context["torch"], context["contract"]
        self.failed()  # Remove passive hooks even if final proof subsequently refuses.
        before_rng = core.canonical(self._rng(context))
        checkpoint = context["checkpoint"]
        actual_sha, before = self._observe(context, lambda: file_hash(checkpoint))
        core.require(boundary["checkpoint"] == {"path": str(checkpoint), "sha256": actual_sha}, "final actual checkpoint digest")

        def decode() -> Any:
            with os.fdopen(os.open(checkpoint, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                core.require(stamp(os.fstat(stream.fileno())) == before, "final decode fd changed")
                value = torch.load(stream, map_location="cpu", weights_only=True)
                core.require(stamp(os.fstat(stream.fileno())) == before, "final checkpoint changed during decode")
                return value

        payload = self._observe(context, decode)
        expected_tag = entry.checkpoint_tag(c, context["contract_ref"], boundary)
        core.require(type(payload) is dict and payload.pop(entry.CHECKPOINT_KEY) == expected_tag, "final union saved family tag")
        payload = cast(dict[str, Any], payload)
        core.require(set(payload) == set(snapshot(trainer, c)) and fingerprint(payload, torch) == self.save_before["state"]
                     and aliases(payload["model"]) == self.save_before["model"]["aliases"], "saved payload differs from own-live state")
        core.require(payload["step"] == boundary["step"] == trainer.step and core.canonical(payload["arch"]) == core.canonical(c["architecture"])
                     and payload["peak_lr"] == c["lr"]["peak"] and payload["grad_clip_mode"] == "none", "saved union recipe/counters")
        names, opt = payload["opt_param_names"], payload["opt"]
        ownership: dict[Any, str] = {}
        offset = 0
        core.require(names == self.save_before["optimizer_names"] and names is not None, "saved optimizer names")
        names = cast(list[str], names)
        for group in opt["param_groups"]:
            core.require(group["lr"] == c["lr"]["peak"], "saved flat union LR")
            roster = names[offset:offset + len(group["params"])]
            offset += len(roster)
            for index, name in zip(group["params"], roster, strict=True):
                core.require(index not in ownership and name not in ownership.values() and name in payload["model"], "optimizer slot ownership")
                ownership[index] = name
        core.require(offset == len(names) and opt["state"] and set(opt["state"]) <= set(ownership), "active optimizer ownership")
        for slot in opt["state"].values():
            step = slot["step"].item() if isinstance(slot["step"], torch.Tensor) else slot["step"]
            core.require(type(step) in (int, float) and step == trainer.step, "saved active slot cumulative step")
        provenance = payload["fresh_rl_donor_provenance"]
        core.require(provenance["mode"] == "fresh_rl_complete_v1" and provenance["sha256"] == c["initial_model"]["sha256"]
                     and provenance["architecture"] == c["architecture"], "saved actual model-only donor provenance")
        model_hash = fingerprint(payload["model"], torch)
        del payload
        core.require(file_hash(checkpoint) == (actual_sha, before), "final checkpoint changed after proof")
        core.require(before_rng == core.canonical(self._rng(context)) == core.canonical(boundary["rng"])
                     == core.canonical(self.save_after["rng"]), "final five RNG streams changed")
        core.require(self._observe(context, lambda: self._snapshot(context)) == self.save_after, "live state changed during final decode")
        return {"schema": SCHEMA, "status": STATUS, "contract": context["contract_ref"], "stage": context["stage_ref"],
                "checkpoint": boundary["checkpoint"], "population_identity": boundary["population_identity"],
                "source_origin_binding": boundary["source_origin_binding"], "sampling": c["sampling"], "step": trainer.step,
                "step_start": boundary["step_start"], "applied_updates": boundary["applied_updates"], "actual_views": boundary["applied_views"],
                "cumulative_views": boundary["usage_state"]["presentations"], "quota_prefix_sha256": boundary["quota_state"]["row_prefix_sha256"],
                "journal_chain": boundary["draw_journals"], "ledger_checkpoint": boundary["quota_ledger_checkpoint"],
                "initial": {key: value for key, value in self.initial.items() if key != "reference"}, "saved_model": self.save_after["model"], "saved_model_state_sha256": model_hash,
                "own_live_state_fingerprint": self.save_before["state"], "save_before_equals_after": True,
                "five_rng_sha256": core.digest(boundary["rng"]), "RNG_scope": "Five streams live in boundary, not learner.pt.",
                "actual_calls": dict(self.counts), "active_optimizer_slots": len(opt["state"]), "optimizer_named_parameters": len(ownership),
                "additional_final_CPU_checkpoint_decodes": 1, "extra_forwards": 0,
                "policy_anchor": {"identity": c["policy_anchor"], "initial_reference": self.initial["reference"],
                    "final_reference_sha256": core.digest(self.save_after["reference"]) if self.save_after["reference"] is not None else None,
                    "actual_reference_forwards": self.reference_calls,
                    "diagnostics": trainer.frozen_policy_anchor_diagnostics(), "teacher_unchanged": True,
                    "no_optimizer_ownership": True, "no_reference_gradients": True, "all_reference_modules_eval": True},
                "usage_state": boundary["usage_state"], "parent_initial_state_matches": None if context["parent"]["fresh"] else True,
                "state_handoff": boundary["state_handoff"],
                "retained_search": boundary["retained_search"],
                "scope": "Observed full-state capped-cycle continuation and saved payload; not playing strength or execution authority."}


def execute(contract_ref: dict[str,str], stage_ref: dict[str,str]) -> dict[str,Any]:
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_capped_on_policy_cycle_proof.py", "foreign cycle proof")
    return entry.execute(contract_ref, stage_ref, observer=OwnLiveProof(),
                         resolve_delivery=actor_bridge.closed_delivery, validate_payload=actor_bridge.validate_payload)


def validate_closed_model_delivery(contract_ref: dict[str, str], binding: dict[str, Any]) -> dict[str, Any]:
    """Bind the genuine metadata resolver; owned consumer reads actual journals."""
    return entry.validate_closed_model_delivery(contract_ref, binding, resolve_delivery=actor_bridge.closed_delivery)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "contract-sha256", "stage", "stage-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256},
                     {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
