"""Internal CPU policy consumer, not a checkpoint loader or launch interface.

A reviewed isolated verified-model seam must bind model identity before use.
Masks are legal moves from strictly decoded stored current-position metadata;
not stored search policies, full history authentication or draw-adjudication.
"""

import hashlib
import numpy as np
import torch
from chess_anti_engine.moves.encode import legal_move_mask_for_encoding
from scripts.stored_wdl_transition import decode
from scripts.stored_wdl_compare import rows_digest


def policy_probe(
    model, x, row_ids, group_ids, *, expected_rows_sha256, batch_size=16
) -> dict:
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("bounded integer batch required")
    ids, groups = tuple(row_ids), tuple(group_ids)
    if rows_digest(x, ids, groups) != expected_rows_sha256:
        raise ValueError("ordered input identity mismatch")
    x = x.copy()
    masks = np.stack([legal_move_mask_for_encoding(decode(row)) for row in x])
    if masks.shape != (len(x), 1858) or not masks.any(1).all():
        raise ValueError("nonempty standard-chess legal masks required")
    if any(p.device.type != "cpu" for p in model.parameters()) or any(
        m.training for m in model.modules()
    ):
        raise ValueError("CPU eval model required")
    logits = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            output = model(torch.from_numpy(x[start : start + batch_size].copy()))[
                "policy_own"
            ]
            if (
                output.shape != (min(batch_size, len(x) - start), 1858)
                or output.dtype != torch.float32
                or output.device.type != "cpu"
                or not bool(torch.isfinite(output).all())
            ):
                raise ValueError("finite compact1858 deployed policy logits required")
            logits.append(output.detach().numpy().copy())
    raw = np.concatenate(logits)
    masked = np.where(masks, raw.astype(np.float64), -np.inf)
    exponential = np.exp(masked - masked.max(1, keepdims=True))
    probabilities = exponential / exponential.sum(1, keepdims=True)
    return {
        "logits": raw,
        "legal_mask": masks,
        "probabilities": probabilities,
        "rows_sha256": expected_rows_sha256,
        "legal_mask_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
        "head": "policy_own",
        "encoding": "compact1858",
        "temperature": 1.0,
        "mask_source": "strict stored current-position standard-chess legal moves",
        "scope": "raw network prior, not search-temperature prior or move quality; repetition truth and full history unverified",
        "controls_measured_here": [],
        "real_model_identity_bound_here": False,
    }
