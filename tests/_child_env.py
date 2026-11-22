"""The environment for a python child a test starts with an `env` built from
scratch (no ``os.environ``).

Such a child does not inherit what `tests/conftest.py` sets for the suite, so
without these two variables it sees the GPU outright. On WSL2 even
``CUDA_VISIBLE_DEVICES=""`` alone is not enough: ``torch.cuda.is_available()``
then still runs ``cuInit``, which opens and holds /dev/dxg. See the comment
above ``PYTORCH_NVML_BASED_CUDA_CHECK`` in `tests/conftest.py`.
`tests/test_child_env_guard.py` fails on a new scratch env for a python child
that does not go through `gpu_free_env`.
"""
from __future__ import annotations

GPU_FREE = {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}


def gpu_free_env(env: dict[str, str]) -> dict[str, str]:
    """``env`` plus the two variables that keep a child off the real GPU."""
    return {**env, **GPU_FREE}
