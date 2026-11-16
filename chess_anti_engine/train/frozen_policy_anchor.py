"""Explicit finite-family forward KL(reference legal || student legal).

Not an evaluation loss or configuration/reload surface. Reference checkpoints are
caller-selected immutable inputs; historical training provenance may be unknown.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import random
import stat
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from chess_anti_engine.policy_anchor_spec import canonical


def _rng() -> tuple[Any, ...]:
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _same_rng(first: tuple[Any, ...], second: tuple[Any, ...]) -> bool:
    a, b = first[1], second[1]
    cuda_a, cuda_b = first[3], second[3]
    return (first[0] == second[0] and a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
            and torch.equal(first[2], second[2]) and ((cuda_a is None and cuda_b is None) or
            (cuda_a is not None and cuda_b is not None and len(cuda_a) == len(cuda_b)
             and all(torch.equal(x, y) for x, y in zip(cuda_a, cuda_b, strict=True)))))


@contextmanager
def preserve_rng():
    before = _rng()
    try:
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        if before[3] is not None:
            torch.cuda.set_rng_state_all(before[3])


def _stamp(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"))


def _checkpoint(ref: dict[str, str]) -> tuple[int, ...]:
    path = Path(ref["path"])
    if path.resolve() != path or path.is_symlink() or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("reference checkpoint must be canonical external")
    before = _stamp(path)
    if not stat.S_ISREG(before[-1]) or before[-1] & 0o222:
        raise ValueError("reference checkpoint must be readonly regular")
    h = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if tuple(getattr(opened, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")) != before:
            raise ValueError("reference checkpoint changed before open")
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
        if tuple(getattr(after, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")) != before:
            raise ValueError("reference checkpoint changed during read")
    if _stamp(path) != before or h.hexdigest() != ref["sha256"]:
        raise ValueError("reference checkpoint custody differs")
    return before


def primary_logits(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    value = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if value is None:
        raise ValueError("missing primary policy logits for frozen anchor")
    return value


def eligible_legal(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("frozen anchor requires two-dimensional policy logits")
    rows = logits.shape[0]
    masks = []
    for key in ("is_network_turn", "has_policy"):
        value = batch.get(key)
        if value is None or value.shape != (rows,) or not bool(torch.isfinite(value).all()) or not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("invalid/missing frozen anchor eligibility mask: " + key)
        masks.append(value.bool())
    eligible = masks[0] & masks[1]
    if not bool(eligible.any()):
        return eligible, torch.empty((0, logits.shape[1]), dtype=torch.bool, device=logits.device)
    legal, flag = batch.get("legal_mask"), batch.get("has_legal_mask")
    if legal is None or flag is None or legal.shape != logits.shape or flag.shape != (rows,):
        raise ValueError("missing/width-mismatched explicit frozen anchor legal mask")
    if not bool(torch.isfinite(legal[eligible]).all()) or not bool(((legal[eligible] == 0) | (legal[eligible] == 1)).all()) or not bool((flag[eligible] == 1).all()):
        raise ValueError("invalid/unflagged active frozen anchor legal mask")
    selected = legal[eligible].bool()
    if not bool(selected.any(dim=-1).all()):
        raise ValueError("empty active frozen anchor legal mask")
    return eligible, selected


def forward_kl(student: torch.Tensor, reference: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """One F32 KL per eligible row; exactly zero illegal-logit gradients."""
    if student.shape != reference.shape or student.shape != legal.shape or student.ndim != 2 or not bool(legal.any(dim=-1).all()):
        raise ValueError("invalid frozen anchor KL shapes/support")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("nonfinite active frozen anchor logits")
    log_p = torch.log_softmax(student.float().masked_fill(~legal, -torch.inf), dim=-1)
    log_q = torch.log_softmax(reference.detach().float().masked_fill(~legal, -torch.inf), dim=-1)
    # Set both illegal log terms to zero before multiplication: no0*(-inf).
    p_safe = torch.where(legal, log_p, 0.0)
    q_safe = torch.where(legal, log_q, 0.0)
    result = (log_q.exp() * (q_safe - p_safe)).sum(dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite frozen anchor KL")
    return result


class FrozenPolicyAnchor:
    """Separate frozen model; never registered beneath the student or optimizer."""

    def __init__(self, spec: dict[str, Any], student: torch.nn.Module):
        from chess_anti_engine.uci import model_loader

        self.spec = copy.deepcopy(spec)
        reference = spec["reference"]
        self.stamp = _checkpoint(reference["checkpoint"])
        original_load, original_build = torch.load, model_loader.build_model
        student_storage = {(str(t.device), t.untyped_storage().data_ptr()) for t in (*student.parameters(), *student.buffers()) if t.numel()}
        loads: list[bool] = []

        def reject_alias(model: torch.nn.Module) -> None:
            if any((str(t.device), t.untyped_storage().data_ptr()) in student_storage for t in (*model.parameters(), *model.buffers()) if t.numel()):
                raise ValueError("frozen reference aliases student")

        def checked_build(config: Any) -> torch.nn.Module:
            model = original_build(config)
            reject_alias(model)  # Before loading or freezing a possible alias.
            return model

        def checked_load(*args: Any, **kwargs: Any) -> Any:
            loads.append(True)
            if len(loads) != 1 or Path(args[0]) != Path(reference["checkpoint"]["path"]) or kwargs.get("weights_only") is not True:
                raise ValueError("unexpected frozen reference decode")
            payload = original_load(*args, **kwargs)
            if not isinstance(payload, dict) or canonical(payload.get("arch")) != canonical(reference["architecture"]) or type(payload.get("step")) is not int or payload["step"] != reference["checkpoint_step"]:
                raise ValueError("frozen reference embedded architecture/step differs")
            return payload

        with preserve_rng(), patch.object(torch, "load", checked_load), patch.object(model_loader, "build_model", checked_build):
            model = model_loader.load_model_from_checkpoint(Path(reference["checkpoint"]["path"]),
                device=str(next(student.parameters()).device), require_complete=True)
        if len(loads) != 1:
            raise ValueError("frozen reference was not decoded exactly once")
        if _checkpoint(reference["checkpoint"]) != self.stamp:
            raise ValueError("reference changed during setup")
        reject_alias(model)
        # Alias refusal precedes freezing: never change student requires_grad.
        model.requires_grad_(False)
        self.model = model.eval()
        self.versions = self._versions()
        self.rows = self.total_rows = self.forwards = 0
        self.kl_sum = self.forward_seconds = 0.0
        self.verify()

    def _versions(self) -> tuple[Any, ...]:
        return tuple((name, id(value), value._version, str(value.device), value.untyped_storage().data_ptr())
            for name, value in (*self.model.named_parameters(remove_duplicate=False), *self.model.named_buffers(remove_duplicate=False)))

    def verify(self) -> None:
        if self._versions() != self.versions or any(module.training for module in self.model.modules()) or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError("frozen reference state/mode/gradient changed")

    def custody(self) -> None:
        self.verify()
        if _checkpoint(self.spec["reference"]["checkpoint"]) != self.stamp:
            raise ValueError("frozen reference file changed")

    def penalty(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        student = primary_logits(outputs)
        eligible, legal = eligible_legal(student, batch)
        self.total_rows += int(student.shape[0])
        if not bool(eligible.any()):
            return None
        self.verify()
        x = batch["x"][eligible]
        relations = batch.get("relations")
        if not bool(torch.isfinite(x).all()) or (relations is not None and not bool(torch.isfinite(relations[eligible]).all())):
            raise ValueError("nonfinite active frozen reference inputs")
        before = _rng()
        started = time.perf_counter()
        self.forwards += 1
        with torch.no_grad():
            try:
                reference = self.model(x, relations=relations[eligible]) if relations is not None else self.model(x)
            finally:
                self.forward_seconds += time.perf_counter() - started
        if not _same_rng(before, _rng()):
            raise ValueError("frozen reference forward changed RNG")
        self.verify()
        terms = forward_kl(student[eligible], primary_logits(reference), legal)
        count = int(terms.numel())
        self.rows += count
        self.kl_sum += float(terms.detach().sum())
        return self.spec["coefficient"] * terms.mean()

    def diagnostics(self) -> dict[str, Any]:
        return {"eligible_rows": self.rows, "total_rows": self.total_rows, "reference_forwards": self.forwards,
                "kl_row_sum": self.kl_sum, "mean_kl": self.kl_sum / self.rows if self.rows else None,
                "weighted_row_sum": self.spec["coefficient"] * self.kl_sum, "forward_host_seconds": self.forward_seconds,
                "timing_scope": "host interval/enqueue time; CPU work elapsed on CPU, unsynchronized on CUDA, not GPU forward latency"}
