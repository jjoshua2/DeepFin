"""Strict checkpoint-owned export for the opt-in Bend evaluator, never a trainer.

Only self-describing trainer checkpoints are accepted. No tolerant load, YAML
fallback, random-weight fallback, or live package constant rebinding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.model_loader import model_config_from_arch
from .adapter import Encoding

CHECKPOINT_FORMAT = 'deepfin-checkpoint-tuple-policy-wdl-v3'
BATCHES = (1, 2, 4, 8, 16)


def fingerprint(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def device_contract(device: str, dtype: str) -> torch.device:
    if device != 'cpu' and re.fullmatch(r'cuda:(?:0|[1-9][0-9]*)', device) is None:
        raise ValueError('device must be cpu or an explicit cuda:N')
    if dtype not in ('float32', 'bfloat16') or (device == 'cpu' and dtype != 'float32'):
        raise ValueError('supported execution: CPU float32 or CUDA float32/bfloat16')
    if device.startswith('cuda:') and int(device.split(':')[1]) > 127:
        raise ValueError('CUDA device index exceeds native representation')
    return torch.device(device)


def require_device(device: torch.device) -> None:
    if device.type == 'cuda' and (not torch.cuda.is_available()
            or device.index is None or device.index >= torch.cuda.device_count()):
        raise RuntimeError(f'{device} requires an available CUDA device; no CPU fallback')


def execution_manifest(device: str, dtype: str) -> dict[str, Any]:
    target = device_contract(device, dtype)
    require_device(target)
    return {'device': device, 'dtype': dtype, 'torch_cuda': torch.version.cuda,
            'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'cuda_capability': list(torch.cuda.get_device_capability(target))
                               if target.type == 'cuda' else None}


def validate_checkpoint_manifest(data: dict[str, Any]) -> None:
    """Validate before spawning native code. Fingerprints are identity, not trust."""
    device, dtype = data.get('device'), data.get('dtype')
    if not isinstance(device, str) or not isinstance(dtype, str):
        raise ValueError('checkpoint manifest needs explicit device/dtype')
    target = device_contract(device, dtype)
    if data.get('row_independent') is not True or data.get('output_order') != ['policy', 'wdl']:
        raise ValueError('checkpoint manifest has unsupported row/output contract')
    if data.get('weights') != 'checkpoint-supplied':
        raise ValueError('checkpoint manifest must identify checkpoint-supplied weights')
    cfg = data.get('model_config')
    if not isinstance(cfg, dict) or data.get('model_config_sha256') != digest_json(cfg):
        raise ValueError('checkpoint model configuration fingerprint mismatch')
    for field in ('input_history_encoding', 'input_extra_features', 'history_rep_fix'):
        if cfg.get(field) != data.get(field):
            raise ValueError('checkpoint model configuration disagrees with encoding')
    if (cfg.get('use_dynamic_relations') is not False
            or cfg.get('policy_dynamic_relations') is not False):
        raise ValueError('checkpoint manifest requires unsupported relation inputs')
    sha = data.get('checkpoint_sha256')
    if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
        raise ValueError('checkpoint manifest is missing its source fingerprint')
    if (data.get('torch_cuda') != torch.version.cuda
            or type(data.get('cxx11_abi')) is not bool
            or data['cxx11_abi'] != bool(torch._C._GLIBCXX_USE_CXX11_ABI)):
        raise ValueError('checkpoint package/runtime CUDA or C++ ABI mismatch')
    require_device(target)
    capability = list(torch.cuda.get_device_capability(target)) if target.type == 'cuda' else None
    if data.get('cuda_capability') != capability:
        raise ValueError('checkpoint CUDA capability mismatch; re-export for this GPU')


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    config: ModelConfig
    encoding: Encoding
    sha256: str


def checkpoint_config(arch: dict[str, Any]) -> tuple[ModelConfig, Encoding]:
    if type(arch.get('_schema_version')) is not int:
        raise ValueError('checkpoint arch needs an integer schema version')
    required = {'kind', 'input_history_encoding', 'input_extra_features',
                'history_rep_fix', 'policy_encoding'}
    if not required <= arch.keys():
        raise ValueError(f'checkpoint arch missing required identity: {sorted(required - arch.keys())}')
    cfg = model_config_from_arch(arch)
    if cfg.kind not in ('tiny', 'transformer') or cfg.policy_encoding != 'lc0_1858':
        raise ValueError('unsupported checkpoint model kind/policy encoding')
    if cfg.use_dynamic_relations or cfg.policy_dynamic_relations:
        raise ValueError('checkpoint requires relation inputs not carried by this adapter')
    encoding = Encoding(cfg.input_history_encoding, cfg.input_extra_features, cfg.history_rep_fix)
    # Only this documented inference-only switch differs from checkpoint config.
    return replace(cfg, use_gradient_checkpointing=False), encoding


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    """Read a single stable file; never inspect unrelated params.json or defaults."""
    before = fingerprint(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if fingerprint(path) != before:
        raise ValueError('checkpoint changed while being read; use an immutable copy')
    if not isinstance(payload, dict) or not isinstance(payload.get('arch'), dict):
        raise ValueError('checkpoint needs embedded arch; bare weights are not accepted')
    cfg, encoding = checkpoint_config(payload['arch'])
    state = payload.get('model')
    if not isinstance(state, dict) or not state or any(not isinstance(k, str)
            or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('checkpoint needs a nonempty tensor model state_dict')
    for name, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite checkpoint tensor: {name}')
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    model.eval().cpu().float()
    if hasattr(model, '_inference_only'):
        setattr(model, '_inference_only', True)
    return LoadedCheckpoint(model, cfg, encoding, before)


class TupleOutputs(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model(x)
        policy = result['policy'] if 'policy' in result else result['policy_own']
        # The native wire is always F32, including packages doing BF16 arithmetic.
        return policy.float(), result['wdl'].float()


def export_checkpoint(path: Path, directory: Path, *, device: str, dtype: str,
                      batch: int) -> tuple[Path, LoadedCheckpoint]:
    from native.bend_engine.aoti_probe.build_test_package import _resolve_package_cxx
    import torch._inductor.config as config

    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported static checkpoint batch')
    execution = execution_manifest(device, dtype)  # fail before loading/allocating model
    loaded = load_checkpoint(path)
    directory.mkdir(parents=True, exist_ok=False)  # cannot overwrite a live package
    package = directory / 'model.pt2'
    target = torch.device(device)
    scalar = torch.float32 if dtype == 'float32' else torch.bfloat16
    model = TupleOutputs(loaded.model.to(device=target, dtype=scalar)).eval()
    example = torch.zeros((batch, loaded.encoding.channels, 8, 8), device=target, dtype=scalar)
    # This deliberately freezes one checkpoint, unlike production rebindable packages.
    with torch.no_grad(), config.patch({'compile_threads': 1, 'cpp.cxx': (_resolve_package_cxx(),)}):
        graph = torch.export.export(model, (example,))
        torch._inductor.aoti_compile_and_package(graph, package_path=str(package))
    model_config = asdict(loaded.config)
    manifest = {'format': CHECKPOINT_FORMAT, 'torch_version': str(torch.__version__),
                'sha256': fingerprint(package), 'checkpoint_sha256': loaded.sha256,
                'model_config': model_config, 'model_config_sha256': digest_json(model_config),
                'weights': 'checkpoint-supplied', 'output_order': ['policy', 'wdl'],
                'row_independent': True, 'batch': batch, 'channels': loaded.encoding.channels,
                'policy_width': 1858, **asdict(loaded.encoding), **execution}
    # A failed build has no manifest and cannot be mistaken for a valid package.
    with package.with_suffix('.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return package, loaded
