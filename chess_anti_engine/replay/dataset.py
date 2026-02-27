from __future__ import annotations

import numpy as np
import torch

from .buffer import ReplaySample


def collate(samples: list[ReplaySample], *, device: str) -> dict[str, torch.Tensor]:
    x = np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False)
    policy_t = np.stack([s.policy_target for s in samples], axis=0).astype(np.float32, copy=False)
    wdl_t = np.array([s.wdl_target for s in samples], dtype=np.int64)

    has_policy = np.array([1.0 if getattr(s, "has_policy", True) else 0.0 for s in samples], dtype=np.float32)

    # Optional aux targets
    sf_wdl = np.zeros((len(samples), 3), dtype=np.float32)
    has_sf_wdl = np.zeros((len(samples),), dtype=np.float32)

    sf_move_index = np.zeros((len(samples),), dtype=np.int64)
    has_sf_move = np.zeros((len(samples),), dtype=np.float32)

    sf_policy_t = np.zeros_like(policy_t, dtype=np.float32)
    has_sf_policy = np.zeros((len(samples),), dtype=np.float32)

    moves_left = np.zeros((len(samples),), dtype=np.float32)
    has_moves_left = np.zeros((len(samples),), dtype=np.float32)
    is_network_turn = np.zeros((len(samples),), dtype=np.bool_)

    # categorical value (default 32 bins for now)
    categorical_t = np.zeros((len(samples), 32), dtype=np.float32)
    has_categorical = np.zeros((len(samples),), dtype=np.float32)

    # soft policy + future policy
    policy_soft_t = np.zeros_like(policy_t, dtype=np.float32)
    has_policy_soft = np.zeros((len(samples),), dtype=np.float32)
    future_policy_t = np.zeros_like(policy_t, dtype=np.float32)
    has_future = np.zeros((len(samples),), dtype=np.float32)

    # volatility
    volatility_t = np.zeros((len(samples), 3), dtype=np.float32)
    has_volatility = np.zeros((len(samples),), dtype=np.float32)

    sf_volatility_t = np.zeros((len(samples), 3), dtype=np.float32)
    has_sf_volatility = np.zeros((len(samples),), dtype=np.float32)

    from chess_anti_engine.moves import POLICY_SIZE as _PS
    legal_mask_t = np.zeros((len(samples), _PS), dtype=np.float32)
    has_legal_mask = np.zeros((len(samples),), dtype=np.float32)

    for i, s in enumerate(samples):
        if s.sf_wdl is not None:
            sf_wdl[i] = s.sf_wdl.astype(np.float32, copy=False)
            has_sf_wdl[i] = 1.0
        if s.sf_move_index is not None:
            sf_move_index[i] = int(s.sf_move_index)
            has_sf_move[i] = 1.0
        if getattr(s, "sf_policy_target", None) is not None:
            sf_policy_t[i] = s.sf_policy_target.astype(np.float32, copy=False)
            has_sf_policy[i] = 1.0
        if s.moves_left is not None:
            moves_left[i] = float(s.moves_left)
            has_moves_left[i] = 1.0
        if s.is_network_turn is not None:
            is_network_turn[i] = bool(s.is_network_turn)
        if s.categorical_target is not None:
            categorical_t[i] = s.categorical_target.astype(np.float32, copy=False)
            has_categorical[i] = 1.0
        if s.policy_soft_target is not None:
            policy_soft_t[i] = s.policy_soft_target.astype(np.float32, copy=False)
            has_policy_soft[i] = 1.0
        if s.future_policy_target is not None:
            future_policy_t[i] = s.future_policy_target.astype(np.float32, copy=False)
            has_future[i] = 1.0
        if s.volatility_target is not None:
            volatility_t[i] = s.volatility_target.astype(np.float32, copy=False)
            has_volatility[i] = 1.0
        if getattr(s, "sf_volatility_target", None) is not None:
            sf_volatility_t[i] = s.sf_volatility_target.astype(np.float32, copy=False)
            has_sf_volatility[i] = 1.0
        if getattr(s, "legal_mask", None) is not None:
            legal_mask_t[i] = s.legal_mask.astype(np.float32, copy=False)
            has_legal_mask[i] = 1.0

    return {
        "x": torch.from_numpy(x).to(device),
        "policy_t": torch.from_numpy(policy_t).to(device),
        "wdl_t": torch.from_numpy(wdl_t).to(device),
        "has_policy": torch.from_numpy(has_policy).to(device),
        "sf_wdl": torch.from_numpy(sf_wdl).to(device),
        "has_sf_wdl": torch.from_numpy(has_sf_wdl).to(device),
        "sf_move_index": torch.from_numpy(sf_move_index).to(device),
        "has_sf_move": torch.from_numpy(has_sf_move).to(device),
        "sf_policy_t": torch.from_numpy(sf_policy_t).to(device),
        "has_sf_policy": torch.from_numpy(has_sf_policy).to(device),
        "moves_left": torch.from_numpy(moves_left).to(device),
        "has_moves_left": torch.from_numpy(has_moves_left).to(device),
        "is_network_turn": torch.from_numpy(is_network_turn).to(device),
        "categorical_t": torch.from_numpy(categorical_t).to(device),
        "has_categorical": torch.from_numpy(has_categorical).to(device),
        "policy_soft_t": torch.from_numpy(policy_soft_t).to(device),
        "has_policy_soft": torch.from_numpy(has_policy_soft).to(device),
        "future_policy_t": torch.from_numpy(future_policy_t).to(device),
        "has_future": torch.from_numpy(has_future).to(device),
        "volatility_t": torch.from_numpy(volatility_t).to(device),
        "has_volatility": torch.from_numpy(has_volatility).to(device),
        "sf_volatility_t": torch.from_numpy(sf_volatility_t).to(device),
        "has_sf_volatility": torch.from_numpy(has_sf_volatility).to(device),
        "legal_mask": torch.from_numpy(legal_mask_t).to(device),
        "has_legal_mask": torch.from_numpy(has_legal_mask).to(device),
    }
