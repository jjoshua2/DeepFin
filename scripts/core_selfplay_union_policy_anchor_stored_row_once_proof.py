"""Passive own-live to saved-payload witness for the fixed-union learner.

Runs inside the learner's already admitted owner. No launcher, extra forward,
model qualification, Root acceptance or scientific authority is provided.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import inspect
import os
from pathlib import Path
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_selfplay_union_policy_anchor_stored_row_once as entry
from scripts.core_selfplay_union_stored_row_once_proof import aliases, file_hash, fingerprint, stamp

SCHEMA = "core_selfplay_union_policy_anchor_own_live_saved_payload_proof_v1"
STATUS = "UNION_POLICY_ANCHOR_OWN_LIVE_SAVED_PAYLOAD_OBSERVED_PENDING_ROOT_CLOSURE_V1"




def state_snapshot(trainer: Any) -> dict[str, Any]:
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION

    core.require(trainer._swa_model is None and not hasattr(trainer.model, "_orig_mod"),
                 "union proof requires eager nonswa model")
    result = {"model": trainer.model.state_dict(), "opt": trainer.opt.state_dict(),
              "scheduler": trainer._scheduler.state_dict(), "step": trainer.step, "peak_lr": float(trainer._peak_lr),
              "fresh_rl_donor_provenance": trainer.fresh_rl_donor_provenance,
              "zclip": trainer.zclip_state_dict(), "grad_clip_mode": trainer.grad_clip_mode,
              "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(trainer._model_config)},
              "frozen_reference_policy_anchor_v1": trainer.frozen_policy_anchor_state()}
    names = trainer._optimizer_param_names()
    if names is not None:
        result["opt_param_names"] = names
    return result


def actual_model_facts(model: Any, torch: Any, architecture: dict[str, Any]) -> dict[str, Any]:
    from chess_anti_engine.encoding import model_input_plane_count

    state = model.state_dict()
    planes = model_input_plane_count(model)
    core.require(model.input_extra_features == architecture["input_extra_features"]
        and model.input_history_encoding == architecture["input_history_encoding"]
        and model.history_rep_fix is architecture["history_rep_fix"]
        and model.policy_encoding == architecture["policy_encoding"], "actual runtime model encoding differs")
    if architecture["kind"] == "transformer":
        actual_configuration = asdict(model.cfg)
        expected_configuration = {key: planes if key == "in_planes" else architecture[key] for key in actual_configuration}
        core.require(core.canonical(actual_configuration) == core.canonical(expected_configuration), "actual Transformer configuration differs")
    else:
        core.require(architecture["kind"] == "tiny" and model.trunk[0].in_channels == planes, "actual Tiny input configuration differs")
        actual_configuration = {"kind": "tiny", "first_convolution_input_planes": model.trunk[0].in_channels}
    shapes = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()}
    storages: dict[Any, int] = {}
    for value in model.parameters():
        key = (str(value.device), value.untyped_storage().data_ptr())
        storages[key] = value.untyped_storage().nbytes() // value.element_size()
    return {"state_entries": len(state), "input_planes": planes, "actual_model_configuration": actual_configuration,
            "unique_parameter_numel": sum(storages.values()), "naive_state_numel": sum(v.numel() for v in state.values()),
            "key_shapes": shapes, "key_shape_dtype_sha256": core.digest(shapes), "aliases": aliases(state),
            "model_state_sha256": fingerprint(state, torch), "finite": True,
            "parameter_devices": sorted({str(v.device) for v in model.parameters()}),
            "parameter_dtypes": sorted({str(v.dtype) for v in model.parameters()}), "architecture": architecture}


def model_facts(trainer: Any, torch: Any, architecture: dict[str, Any]) -> dict[str, Any]:
    core.require(core.canonical(state_snapshot(trainer)["arch"]) == core.canonical(architecture),
                 "observed normalized architecture differs")
    return actual_model_facts(trainer.model, torch, architecture)


def reference_snapshot(context: dict[str, Any]) -> dict[str, Any] | None:
    trainer, c, torch = context["trainer"], context["contract"], context["torch"]
    core.require(trainer.frozen_policy_anchor_state() == c["policy_anchor"], "observed configured anchor differs")
    anchor = trainer._frozen_policy_anchor
    if c["policy_anchor"]["coefficient"] == 0:
        core.require(anchor is None, "zero union anchor loaded teacher")
        return None
    core.require(anchor is not None and anchor.spec == c["policy_anchor"], "missing selected positive teacher")
    anchor = cast(Any, anchor)
    anchor.custody()
    teacher = anchor.model
    core.require(all(not module.training for module in teacher.modules())
        and all(not value.requires_grad and value.grad is None for value in teacher.parameters()),
        "teacher mode/gradient differs")
    optimized = {id(value) for group in trainer.opt.param_groups for value in group["params"]}
    core.require(not any(id(value) in optimized for value in teacher.parameters()), "teacher optimizer ownership")
    student_storage = {(str(value.device), value.untyped_storage().data_ptr())
        for value in (*trainer.model.parameters(), *trainer.model.buffers()) if value.numel()}
    core.require(not any((str(value.device), value.untyped_storage().data_ptr()) in student_storage
        for value in (*teacher.parameters(), *teacher.buffers()) if value.numel()), "teacher aliases actual student")
    facts = actual_model_facts(teacher, torch, c["architecture"])
    core.require(facts["parameter_devices"] == sorted({str(value.device) for value in trainer.model.parameters()})
        and facts["parameter_dtypes"] == sorted({str(value.dtype) for value in trainer.model.parameters()}),
        "actual reference/student device or parameter dtype differs")
    return {"checkpoint": c["policy_anchor"]["reference"]["checkpoint"],
        "checkpoint_stamp": list(anchor.stamp), "model": facts}


def rows_before(context: dict[str, Any]) -> int:
    from chess_anti_engine.replay.stored_row_once import rows_after_batches

    c = context["contract"]
    prior = sum(c["updates_per_segment"][:context["stage"]["segment"] - 1])
    return rows_after_batches(len(context["replay"]), c["batch_size"], prior)


class OwnLiveProof:
    """Explicit passive callback object; hooks count calls and never replace math."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []
        self.context: dict[str, Any] | None = None
        self.initial: dict[str, Any] = {}
        self.save_before: dict[str, Any] = {}
        self.save_after: dict[str, Any] = {}
        self.reference_calls = 0
        self.counts = {"forward": 0, "optimizer_pre": 0, "optimizer_post": 0, "save": 0}

    def _rng(self, context: dict[str, Any]) -> dict[str, Any]:
        from scripts.offline_epoch_resume import _rng_state

        return _rng_state(context["rng"], context["np"].random)

    def _observe(self, context: dict[str, Any], action: Callable[[], Any]) -> Any:
        before = core.canonical(self._rng(context))
        value = action()
        core.require(before == core.canonical(self._rng(context)), "passive union observation changed one of five RNG streams")
        return value

    def _snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        trainer = context["trainer"]
        return {"state": fingerprint(state_snapshot(trainer), context["torch"]),
                "model": model_facts(trainer, context["torch"], context["contract"]["architecture"]),
                "rng": self._rng(context), "optimizer_names": trainer._optimizer_param_names(),
                "reference": reference_snapshot(context)}

    def before_train(self, context: dict[str, Any]) -> None:
        core.require(self.context is None, "observer cannot be reused across segments")
        self.context = context
        trainer, c = context["trainer"], context["contract"]
        core.require(context["population"].identity == {k: v for k, v in c["population"].items() if k != "plan"}
                     and trainer.step == sum(c["updates_per_segment"][:context["stage"]["segment"] - 1])
                     and trainer.grad_clip_mode == "none", "actual union setup/step/noClip differs")
        core.require(Path(inspect.getfile(type(trainer))).resolve() == core.REPO / "scripts/core_finite_learner.py",
                     "foreign actual Trainer constructor")
        self.initial = self._observe(context, lambda: self._snapshot(context))
        self.initial.update(actual_student_device=str(trainer.device), actual_Trainer_use_amp=trainer.use_amp,
                            backend=context["backend"], source_origins={"entry": str(Path(entry.__file__).resolve()),
                            "proof": str(Path(__file__).resolve()), "trainer": str(Path(inspect.getfile(type(trainer))).resolve()),
                            "model": str(Path(inspect.getfile(type(trainer.model))).resolve()),
                            "population": str(Path(inspect.getfile(type(context["population"]))).resolve()),
                            "torch": str(Path(context["torch"].__file__).resolve()), "numpy": str(Path(context["np"].__file__).resolve())},
                            actual_torch_version=context["torch"].__version__, actual_numpy_version=context["np"].__version__)
        core.require(str(trainer.device) == c["runtime"]["device"] and trainer.use_amp is c["runtime"]["use_amp"],
                     "observed actual device/AMP differs")

        def forward(_model: Any, _args: Any) -> None:
            self.counts["forward"] += 1

        def pre(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            self.counts["optimizer_pre"] += 1

        def post(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            self.counts["optimizer_post"] += 1

        self.hooks.extend((trainer.model.register_forward_pre_hook(forward),
                           trainer.opt.register_step_pre_hook(pre), trainer.opt.register_step_post_hook(post)))
        anchor = trainer._frozen_policy_anchor
        if anchor is not None:
            def reference_forward(_model: Any, _args: Any) -> None:
                self.reference_calls += 1
            self.hooks.append(anchor.model.register_forward_pre_hook(reference_forward))

    def before_save(self, context: dict[str, Any]) -> None:
        core.require(context is self.context and not self.save_before, "one observed save per segment")
        count = context["contract"]["updates_per_segment"][context["stage"]["segment"] - 1]
        core.require(self.counts == {"forward": count, "optimizer_pre": count, "optimizer_post": count, "save": 0},
                     "actual union forward/optimizer calls differ")
        self.save_before = self._observe(context, lambda: self._snapshot(context))
        core.require(self.save_before["reference"] == self.initial["reference"], "frozen teacher changed during training")
        diagnostics = context["trainer"].frozen_policy_anchor_diagnostics()
        core.require((diagnostics is None and self.reference_calls == 0) or
            (diagnostics is not None and diagnostics["reference_forwards"] == self.reference_calls
             and 0 <= self.reference_calls <= count
             and diagnostics["total_rows"] == len(context["replay"]) - context["replay"].remaining - rows_before(context)),
            "actual teacher calls/dose differ")
        self.counts["save"] += 1

    def after_save(self, context: dict[str, Any]) -> None:
        core.require(context is self.context and self.save_before and not self.save_after, "observed save ordering")
        self.save_after = self._observe(context, lambda: self._snapshot(context))
        core.require(self.save_before == self.save_after, "Trainer.save changed own-live model/optimizer/RNG")

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
        expected_tag = {"identity": entry.checkpoint_identity(c, context["contract_ref"], context["stage"]["segment"]),
                        "quota_state": boundary["quota_state"], "draw_journals": boundary["draw_journals"],
                        "quota_ledger_checkpoint": boundary["quota_ledger_checkpoint"],
                        "population_identity": boundary["population_identity"], "source_origin_binding": boundary["source_origin_binding"]}
        core.require(type(payload) is dict and payload.pop(entry.CHECKPOINT_KEY) == expected_tag, "final union saved family tag")
        payload = cast(dict[str, Any], payload)
        core.require(set(payload) == set(state_snapshot(trainer)) and fingerprint(payload, torch) == self.save_before["state"]
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
                "cumulative_views": boundary["quota_state"]["cursor"], "quota_prefix_sha256": boundary["quota_state"]["row_prefix_sha256"],
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
                "scope": "Observed own-live and saved state in same learner process; pending independent closed Root delivery, not playing strength or authority."}

    def failed(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_selfplay_union_policy_anchor_stored_row_once_proof.py", "foreign union observer wrapper")
    return entry.execute(contract_ref, stage_ref, observer=OwnLiveProof())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args(argv)
    boundary = execute({"path": args.contract, "sha256": args.contract_sha256}, {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": boundary["status"], "checkpoint": boundary["checkpoint"], "proof_schema": SCHEMA}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
