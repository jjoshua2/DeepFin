"""Policy/value overlay API; implementation shares the existing storage verifier."""
from .target_overlay import (
    BaseSeals as BaseSeals,
    begin_target_shard as begin_target_shard,
    finish_target_shard as finish_target_shard,
    qualify_target_roots as qualify_target_roots,
)
