"""Strict, opt-in checkpoint export for the Bend evaluator; never infer trained status.

Requires the self-describing arch/model layout written by current Trainer.save.
No params.json search, architecture override, tolerant migration or fallback weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from chess_anti_engine.inference import _policy_output
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, build_model
from chess_anti_engine.uci.model_loader import model_config_from_arch
from .adapter import Encoding
from .backend import BATCHES, CHECKPOINT_FORMAT


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    encoding: Encoding
    identity: dict[str, object]


def load_checkpoint(path: Path, *, weights_key: str = 'model') -> LoadedCheckpoint:
    if weights_key not in ('model', 'swa_model'):
        raise ValueError('weights_key must be model or swa_model')
    path = path / 'trainer.pt' if path.is_dir() else path
    # One descriptor: atomic replacement of the original path cannot mix versions.
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
        source.seek(0)
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        source.seek(0)
        if hashlib.file_digest(source, 'sha256').hexdigest() != digest:
            raise ValueError('checkpoint changed while loading; use an immutable copy')
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('arch'), dict):
        raise ValueError('checkpoint requires embedded arch; re-save with current Trainer')
    arch = checkpoint['arch']
    version = arch.get('_schema_version')
    if type(version) is not int or not 1 <= version <= ARCH_SCHEMA_VERSION:
        raise ValueError('unsupported checkpoint architecture schema')
    required = ('kind', 'policy_encoding', 'input_history_encoding', 'input_extra_features', 'history_rep_fix')
    if any(key not in arch for key in required):
        raise ValueError('checkpoint must explicitly declare model and encoding metadata')
    encoding = Encoding(arch['input_history_encoding'], arch['input_extra_features'], arch['history_rep_fix'])
    cfg = model_config_from_arch(arch)
    if cfg.kind not in ('tiny', 'transformer'):
        raise ValueError('unsupported checkpoint model kind')
    state = checkpoint.get(weights_key)
    if (not isinstance(state, dict) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())):
        raise ValueError('checkpoint requires the selected nonempty tensor state dictionary')
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError('checkpoint contains nonfinite weights')
    model = build_model(cfg)
    # Unlike a warm-start loader, this gate must never initialize missing layers.
    model.load_state_dict(state, strict=True)
    for name, actual in model.state_dict().items():
        if not torch.isfinite(actual).all().item() or not torch.equal(actual, state[name].to(actual.dtype)):
            raise ValueError('checkpoint alias/dtype conflict after exact weight load: ' + name)
    if hasattr(model, '_inference_only'):
        setattr(model, '_inference_only', True)
    model.eval().cpu().float()
    identity: dict[str, object] = {
        'checkpoint_sha256': digest, 'checkpoint_name': path.name, 'weights_key': weights_key,
        'weights': 'checkpoint-provided; training provenance not inferred',
        'arch': arch, 'resolved_model_config': asdict(cfg),
        'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    step = checkpoint.get('step')
    if type(step) is int and step >= 0:
        identity['checkpoint_step'] = step
    return LoadedCheckpoint(model, encoding, identity)


def target(device: str, index: int) -> tuple[torch.device, torch.dtype]:
    if type(index) is not int or not 0 <= index <= 127:
        raise ValueError('invalid device index')
    if device == 'cpu':
        if index != 0:
            raise ValueError('CPU device index must be zero')
        return torch.device('cpu'), torch.float32
    if device != 'cuda':
        raise ValueError('device must be explicitly cpu or cuda')
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError('requested CUDA device is unavailable; no CPU fallback')
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('requested CUDA device lacks BF16 support')
    return torch.device('cuda', index), torch.bfloat16


class OutputTuple(torch.nn.Module):
    """Use the production policy/policy_own alias, never an auxiliary policy head."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return _policy_output(outputs).float(), outputs['wdl'].float()


class EagerReference(torch.nn.Module):
    """Independent eager singleton forwards behind the verifier's CPU tensor API."""
    def __init__(self, model: torch.nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.model, self.device, self.dtype = model, device, dtype

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(x.to(device=self.device, dtype=self.dtype))
        return {'policy': _policy_output(outputs).float().cpu(), 'wdl': outputs['wdl'].float().cpu()}


def export_checkpoint(loaded: LoadedCheckpoint, package: Path, *, batch: int = 4,
                      device: str = 'cpu', device_index: int = 0) -> EagerReference:
    if type(batch) is not int or batch not in BATCHES:
        raise ValueError('unsupported fixed checkpoint batch')
    resolved, dtype = target(device, device_index)
    if package.exists() or package.with_suffix('.json').exists():
        raise FileExistsError('refusing to overwrite evaluator package or manifest')
    import torch._inductor.config as config
    from native.bend_engine.aoti_probe.build_test_package import _resolve_package_cxx
    model = loaded.model.to(device=resolved, dtype=dtype).eval()
    example = torch.zeros((batch, loaded.encoding.channels, 8, 8), device=resolved, dtype=dtype)
    wrapper = OutputTuple(model).eval()
    with torch.no_grad():
        policy, wdl = wrapper(example)
        if policy.shape != (batch, 1858) or wdl.shape != (batch, 3):
            raise ValueError('checkpoint requires compact policy 1858 and WDL3 logits')
        if not torch.isfinite(policy).all().item() or not torch.isfinite(wdl).all().item():
            raise ValueError('checkpoint forward produced nonfinite logits')
    package.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), config.patch({'compile_threads': 1, 'cpp.cxx': (_resolve_package_cxx(),)}):
        graph = torch.export.export(wrapper, (example,))
        torch._inductor.aoti_compile_and_package(graph, package_path=str(package))
    manifest = {
        'format': CHECKPOINT_FORMAT, 'torch_version': str(torch.__version__),
        'sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
        'channels': loaded.encoding.channels, 'batch': batch, 'policy_width': 1858,
        'device': device, 'device_index': device_index,
        'dtype': 'float32' if device == 'cpu' else 'bfloat16',
        # Only project eval-mode models are accepted. The runner tests actual rows;
        # this declaration alone is not a numeric qualification or proof.
        'row_independent': True, 'qualification': 'not established by export alone',
        **asdict(loaded.encoding), **loaded.identity,
    }
    package.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return EagerReference(model, resolved, dtype)
