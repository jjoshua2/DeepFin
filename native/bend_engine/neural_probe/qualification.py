"""Readiness, artifact retention and exact-checkpoint reuse for explicit probes.

These checks are not neural/CUDA qualification. Only the executing probe can
produce that result; a compiled package or a memory snapshot cannot establish it.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import torch

from .backend import CHECKPOINT_FORMAT, execution_spec, package_manifest
from .checkpoint import EagerReference, LoadedCheckpoint, target

if TYPE_CHECKING:
    from collections.abc import Generator


def protect_inputs(report: Path, inputs: list[Path]) -> None:
    """Reject path, hardlink and symlink aliases before writing ANY report."""
    for source in inputs:
        if (report.resolve() == source.resolve()
                or (report.exists() and source.exists() and report.samefile(source))):
            raise ValueError('report destination must differ from protected input: ' + str(source))


def write_report(path: Path, report: dict[str, object]) -> None:
    """Bank stage progress atomically; never leave a half-written JSON readout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.bend-report-', delete=False) as stream:
            name = stream.name
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


@contextmanager
def workspace(destination: Path | None) -> Generator[Path, None, None]:
    """An explicit work directory is new and retained; default work is disposable."""
    if destination is None:
        with tempfile.TemporaryDirectory(prefix='bend-checkpoint-') as temp:
            yield Path(temp)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        yield destination.resolve()


@contextmanager
def compilation_cache(work: Path) -> Generator[None, None, None]:
    """Keep Inductor work private and restore the caller's environment on failure."""
    key = 'TORCHINDUCTOR_CACHE_DIR'
    previous = os.environ.get(key)
    os.environ[key] = str(work / 'inductor')
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def tools(bun: str | None, cc: str | None, cxx: str | None) -> dict[str, str]:
    result = {}
    for name, command in (('bun', bun), ('cc', cc), ('cxx', cxx), ('cmake', 'cmake')):
        executable = shutil.which(command) if command else None
        if executable is None:
            raise ValueError('required native tool not found: ' + name)
        result[name] = executable
    return result


def device_snapshot(device: str, index: int) -> dict[str, object]:
    """Queries may initialize a CUDA context, but never move/execute the model."""
    resolved, dtype = target(device, index)
    result: dict[str, object] = {
        'device': str(resolved), 'dtype': str(dtype), 'torch_version': str(torch.__version__),
        'cuda_build': torch.version.cuda,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'memory_is_reserved': False,
    }
    if device == 'cuda':
        props = torch.cuda.get_device_properties(resolved)
        free, total = torch.cuda.mem_get_info(resolved)
        result.update(name=props.name, compute_capability=[props.major, props.minor],
                      free_bytes_at_preflight=free, total_bytes=total)
    return result


def verified_package(package: Path, loaded: LoadedCheckpoint, *, batch: int,
                     device: str, device_index: int) -> dict[str, object]:
    """Do not execute a cached native package merely because its filename matches."""
    manifest, encoding = package_manifest(package)
    expected_dtype = 'float32' if device == 'cpu' else 'bfloat16'
    if (manifest['format'] != CHECKPOINT_FORMAT or encoding != loaded.encoding
            or manifest['batch'] != batch
            or execution_spec(manifest) != (device, expected_dtype, device_index)):
        raise ValueError('reused package does not match requested encoding/batch/target')
    for key in ('checkpoint_sha256', 'weights_key', 'resolved_model_config'):
        # JSON canonicalization also handles tuples serialized as lists in sidecars.
        if key not in manifest or json.dumps(manifest[key], sort_keys=True) != json.dumps(loaded.identity[key], sort_keys=True):
            raise ValueError('reused package does not match checkpoint identity: ' + key)
    return manifest


def reuse_reference(loaded: LoadedCheckpoint, *, device: str, device_index: int) -> EagerReference:
    resolved, dtype = target(device, device_index)
    model = loaded.model.to(device=resolved, dtype=dtype).eval()
    return EagerReference(model, resolved, dtype)
