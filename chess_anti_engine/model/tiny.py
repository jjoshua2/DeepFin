from __future__ import annotations

import torch
import torch.nn as nn

from chess_anti_engine.moves import POLICY_SIZE


class TinyNet(nn.Module):
    """Small model: (C,8,8) -> policy logits (POLICY_SIZE) and WDL logits (3)."""

    def __init__(self, in_planes: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64, 256),
            nn.Mish(),
            nn.Linear(256, POLICY_SIZE),
        )
        self.wdl = nn.Sequential(
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return {
            "policy": self.policy(h),
            "wdl": self.wdl(h),
        }
