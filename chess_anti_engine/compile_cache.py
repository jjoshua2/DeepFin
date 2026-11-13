"""Shared TorchInductor/Triton cache configuration.

Lives in its own stdlib-only module so the UCI engine can reuse the selfplay
worker's cache layout without importing ``chess_anti_engine.worker`` (whose
module-level imports pull in the replay/zarr stack the ``match`` install does
not ship). ``worker._configure_shared_compile_cache`` is this function.
"""
from __future__ import annotations

import os
from pathlib import Path


def configure_shared_compile_cache(*, cache_dir: Path) -> None:
    """Point TorchInductor/Triton caches at a shared worker cache root.

    Uses forced env var assignment (not setdefault) so it overrides the
    system default /tmp location.  All workers sharing the same cache_dir
    reuse autotuned kernels and FX graph compilations.
    """
    compile_cache_root = cache_dir / "compile_cache"
    inductor_dir = compile_cache_root / "torchinductor"
    triton_dir = compile_cache_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
  # Enable FX graph cache so compiled graphs persist across restarts.
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
