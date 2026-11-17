"""Passive bounded-order own-live proof; reuses accepted physics/fingerprints.

No extra forward, launcher, Root acceptance or model adoption is supplied.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_selfplay_union_bounded_shuffle as entry
from scripts import core_selfplay_union_stored_row_once_proof as baseline

SCHEMA = "core_selfplay_union_bounded_own_live_saved_payload_proof_v1"
STATUS = "BOUNDED_UNION_OWN_LIVE_SAVED_PAYLOAD_OBSERVED_PENDING_ROOT_CLOSURE_V1"
# Byte-exact accepted tensor/state/architecture/five-RNG physics utilities.
fingerprint = baseline.fingerprint
aliases = baseline.aliases
state_snapshot = baseline.state_snapshot
stamp = baseline.stamp
file_hash = baseline.file_hash


class OwnLiveProof(baseline.OwnLiveProof):
    def _snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        result = super()._snapshot(context)
        result["bounded_state_sha256"] = core.digest(context["replay"].checkpoint())
        return result

    def before_train(self, context: dict[str, Any]) -> None:
        super().before_train(context)
        self.initial["source_origins"].update(entry=str(Path(entry.__file__).resolve()), proof=str(Path(__file__).resolve()))

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
                "cumulative_views": boundary["quota_state"]["emitted"], "quota_prefix_sha256": boundary["quota_state"]["reservation_prefix"],
                "journal_chain": boundary["draw_journals"], "ledger_checkpoint": boundary["quota_ledger_checkpoint"],
                "initial": self.initial, "saved_model": self.save_after["model"], "saved_model_state_sha256": model_hash,
                "own_live_state_fingerprint": self.save_before["state"], "save_before_equals_after": True,
                "five_rng_sha256": core.digest(boundary["rng"]), "bounded_state_sha256": core.digest(boundary["quota_state"]),
                "RNG_scope": "Five streams live in boundary; bounded shuffle/mirror frontier also bound in checkpoint tag.",
                "actual_calls": dict(self.counts), "active_optimizer_slots": len(opt["state"]), "optimizer_named_parameters": len(ownership),
                "additional_final_CPU_checkpoint_decodes": 1, "extra_forwards": 0,
                "scope": "Observed own-live and saved state in same learner process; pending independent closed Root delivery, not playing strength or authority."}



def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    core.require(Path(__file__).resolve() == core.REPO / "scripts/core_selfplay_union_bounded_shuffle_proof.py",
                 "foreign bounded observer wrapper")
    return entry.execute(contract_ref, stage_ref, observer=OwnLiveProof())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "contract-sha256", "stage", "stage-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    boundary = execute({"path": args.contract, "sha256": args.contract_sha256}, {"path": args.stage, "sha256": args.stage_sha256})
    print(core.canonical({"status": boundary["status"], "checkpoint": boundary["checkpoint"], "proof_schema": SCHEMA}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
