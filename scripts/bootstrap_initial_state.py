"""Record actual initialized model tensors without consuming any random numbers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def record_initial_state(model: Any, output: Path, *, seed: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        payload = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, 'little'))
        digest.update(payload)
    result = {'schema': 1, 'seed': seed, 'tensor_sha256': digest.hexdigest(),
              'scope': 'Actual build_model state before Trainer or checkpoint restoration'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result
