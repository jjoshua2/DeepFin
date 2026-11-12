"""Internal CPU-only paired Trainer consumer. Caller must use the strict loader.

This does not authenticate a caller-supplied model/report or launch a process.
Real bank use requires a reviewed isolated runner binding those objects, complete
resource/custody/TB controls and load-time admission. Synthetic mode is explicit.
"""

from __future__ import annotations
import copy
import hashlib
import inspect
import json
import math
import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.train.e2_batch_adapter import tensor_hash
from chess_anti_engine.train.e2_objective import canonical, freeze_objective


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_digest(model):
    return digest([(k, tensor_hash(v)) for k, v in sorted(model.state_dict().items())])


def optimizer_state_digest(optimizer):
    def freeze(value):
        if isinstance(value, torch.Tensor):
            return {"tensor": tensor_hash(value)}
        if isinstance(value, dict):
            return [
                (str(k), freeze(v))
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            ]
        if isinstance(value, (list, tuple)):
            return [freeze(v) for v in value]
        return value

    return digest(freeze(optimizer.state_dict()))


def trunk_digest(model):
    return digest(
        [
            (k, tensor_hash(v))
            for k, v in sorted(model.state_dict().items())
            if k.startswith(("embed.", "blocks."))
        ]
    )


def trace_digest(pairs):
    return digest([list(p.tensor_bindings) for p in pairs])


def optimizer_contract(overrides):
    defaults = {
        name: p.default
        for name, p in inspect.signature(Trainer).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
        and p.default is not inspect.Parameter.empty
    }
    values = defaults | overrides
    required = {
        "device": "cpu",
        "use_amp": False,
        "use_compile": False,
        "feature_dropout_p": 0.0,
        "mirror_prob": 0.0,
        "resid_channel_dropout": 0.0,
        "resid_channel_balance_weight": 0.0,
        "prefetch_batches": False,
        "accum_steps": 1,
        "swa_start": 0,
        "rebuild_sf_targets": False,
        "rebuild_categorical_target": False,
        "aurora_cuda_graphs": False,
        "log_dir": None,
    }
    if any(values.get(k) != v for k, v in required.items()):
        raise ValueError("deterministic CPU trainer controls required")
    if (
        values.get("lr_schedule")
        not in ("sqrt_release", "wsd_sqrt", "wsd_sqrt_release")
        or type(values.get("lr_release_cycle_steps")) is not int
        or values["lr_release_cycle_steps"] != 120
    ):
        raise ValueError("explicit global 120-step release schedule required")
    if (
        values.get("optimizer") != "aurora"
        or values.get("matrix_optimizer_scope") != "mlp_out"
    ):
        raise ValueError("explicit production optimizer routing required")
    if type(values.get("lr")) not in (float, int) or not 0 < values["lr"] <= 0.003:
        raise ValueError("bounded positive LR required")
    # No mutable/object-valued optional configuration in this narrow contract.
    if any(
        v is not None and type(v) not in (bool, int, float, str)
        for v in values.values()
    ):
        raise ValueError("non-scalar trainer configuration unsupported")
    return values


class _TraceTrainer(Trainer):
    def __init__(self, model, *, fixed_loss, **kwargs):
        self.fixed_loss = dict(fixed_loss)
        self.pending: dict[str, torch.Tensor] | None = None
        self.runtime_ready = False
        super().__init__(model, **kwargs)
        self.runtime_ready = True

    @property
    def _loss_kwargs(self):
        if not self.runtime_ready:
            return super()._loss_kwargs
        return dict(self.fixed_loss)

    def _iter_training_batches(self, buf, *, batch_size, mirror_prob, count, **kwargs):
        del buf, kwargs
        if count != 1 or mirror_prob != 0 or self.pending is None:
            raise ValueError("unexpected batch consumption")
        batch = self.pending
        self.pending = None
        if batch["x"].shape[0] != batch_size:
            raise ValueError("batch size mismatch")
        yield {k: v.clone() for k, v in batch.items()}


def run_verified_pair(
    model,
    contract,
    expected_objective_sha256,
    pairs,
    *,
    trainer_overrides,
    seed,
    synthetic=False,
) -> dict[str, Any]:
    """Private strict-loader seam; synthetic short runs never claim64x256."""
    payload = json.loads(contract.payload_json)
    pins = payload["pins"]
    verified = freeze_objective(
        payload["effective_shared_kwargs"],
        donor_sha256=pins["donor"],
        architecture_sha256=pins["architecture"],
        optimizer_plan_sha256=pins["optimizer_plan"],
        batch_trace_sha256=pins["batch_trace"],
    )
    if verified != contract or contract.sha256 != expected_objective_sha256:
        raise ValueError("objective identity mismatch")
    plan = optimizer_contract(trainer_overrides)
    if (
        digest(plan) != pins["optimizer_plan"]
        or trace_digest(pairs) != pins["batch_trace"]
    ):
        raise ValueError("optimizer/trace identity mismatch")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded RNG seed required")
    steps = len(pairs)
    size = int(pairs[0].search_only["x"].shape[0]) if pairs else 0
    if synthetic:
        if not 1 <= steps <= 64 or not 1 <= size <= 4:
            raise ValueError("tiny synthetic bounds exceeded")
    elif (steps, size) != (64, 256):
        raise ValueError("exact64x256 required")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("CPU model required")
    shared = payload["effective_shared_kwargs"]
    for key in (
        "sf_sparse_params",
        "sf_policy_floor",
        "sf_shape",
        "exact_objective_mask_weights",
    ):
        if shared[key] is not None:
            raise ValueError("unsupported objective object field")
    initial = state_digest(model)
    initial_trunk = trunk_digest(model)
    result = []
    original_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        for arm in range(2):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            child = copy.deepcopy(model)
            for p in child.parameters():
                p.requires_grad_(True)
            fixed_loss = {
                k: v
                for k, v in shared.items()
                if k
                not in (
                    "report_exact_masked_sums",
                    "exact_corpus_rows",
                    "exact_objective_mask_weights",
                )
            }
            fixed_loss["search_wdl_frac"] = (1.0, 0.8)[arm]
            trainer = _TraceTrainer(child, fixed_loss=fixed_loss, **plan)
            if (
                trainer.step != 0
                or len(trainer.opt.state) != 0
                or state_digest(child) != initial
            ):
                raise ValueError("fresh optimizer/model requirement failed")
            rows = []
            applied_lrs = []

            def record_step(opt, _args, _kwargs, observed=applied_lrs):
                observed.append([float(g["lr"]) for g in opt.param_groups])

            step_hook = trainer.opt.register_step_pre_hook(record_step)
            try:
                for index, pair in enumerate(pairs):
                    current = tuple(
                        (
                            k,
                            tensor_hash(pair.search_only[k]),
                            tensor_hash(pair.outcome20[k]),
                        )
                        for k in sorted(pair.search_only)
                    )
                    if current != pair.tensor_bindings:
                        raise ValueError("load-time tensor identity mismatch")
                    if set(pair.search_only) != set(pair.outcome20) or any(
                        not torch.equal(pair.search_only[k], pair.outcome20[k])
                        for k in pair.search_only
                        if k != "categorical_t"
                    ):
                        raise ValueError("common paired tensor mismatch")
                    batch = (pair.search_only, pair.outcome20)[arm]
                    if batch["x"].shape[0] != size:
                        raise ValueError("shared trace row count mismatch")
                    trainer.pending = batch
                    before = trainer.step
                    before_hook_count = len(applied_lrs)
                    lrs = [float(g["lr"]) for g in trainer.opt.param_groups]
                    metrics = trainer.train_steps(
                        cast(Any, SimpleNamespace(exact_without_replacement=False)),
                        batch_size=size,
                        steps=1,
                    )
                    if len(applied_lrs) != before_hook_count + 1:
                        raise ValueError("actual optimizer call count mismatch")
                    if (
                        trainer.pending is not None
                        or trainer.step != before + 1
                        or metrics.train_steps_done != 1
                        or metrics.train_samples_seen != size
                    ):
                        raise ValueError("update/consumption count mismatch")
                    if (
                        not math.isfinite(metrics.loss)
                        or metrics.grad_nonfinite_skip_rate != 0
                        or metrics.transient_cuda_retry_batches != 0
                    ):
                        raise ValueError("nonfinite/skipped/retried update")
                    if any(
                        not torch.isfinite(v).all() for v in child.state_dict().values()
                    ):
                        raise ValueError("nonfinite model state")
                    rows.append(
                        {
                            "cursor": index + 1,
                            "step": trainer.step,
                            "group_lr_before": lrs,
                            "group_lr_applied": applied_lrs[-1],
                            "group_lr_after": [
                                float(g["lr"]) for g in trainer.opt.param_groups
                            ],
                            "loss": float(metrics.loss),
                            "model_sha256": state_digest(child),
                            "torch_rng_sha256": tensor_hash(torch.get_rng_state()),
                            "optimizer_state_sha256": optimizer_state_digest(
                                trainer.opt
                            ),
                            "trunk_sha256": trunk_digest(child),
                        }
                    )
                result.append(
                    {
                        "arm": arm,
                        "initial_model_sha256": initial,
                        "steps": rows,
                        "final_model_sha256": state_digest(child),
                        "initial_trunk_sha256": initial_trunk,
                        "final_trunk_sha256": trunk_digest(child),
                        "optimizer_state_entries": len(trainer.opt.state),
                    }
                )
            finally:
                step_hook.remove()
                trainer.writer.close()
        if [r["group_lr_applied"] for r in result[0]["steps"]] != [
            r["group_lr_applied"] for r in result[1]["steps"]
        ]:
            raise ValueError("arm LR trace differs")
        if state_digest(model) != initial:
            raise ValueError("input model mutated")
        return {
            "status": "SYNTHETIC_RUNTIME_ONLY"
            if synthetic
            else "FINITE_RUNTIME_COMPLETED",
            "objective_sha256": contract.sha256,
            "optimizer_plan_sha256": digest(plan),
            "batch_trace_sha256": trace_digest(pairs),
            "steps_per_arm": steps,
            "batch_size": size,
            "arms": result,
            "checkpoint_loader_binding": "caller obligation; not authenticated by this consumer",
        }
    finally:
        torch.set_rng_state(original_rng[0])
        np.random.set_state(original_rng[1])
        random.setstate(original_rng[2])
