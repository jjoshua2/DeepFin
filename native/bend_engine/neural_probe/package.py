"""Export a seeded, UNTRAINED project TinyNet as a fixed CPU AOTI test package."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import json

from .adapter import Encoding
from .backend import FORMAT

SEED = 20260918
DEFAULT_ENCODING = Encoding('lc0_root_legacy_meta', 'v2_threats', True)


def smoke_model(encoding: Encoding = DEFAULT_ENCODING):
    import torch
    from chess_anti_engine.model import ModelConfig, build_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = build_model(ModelConfig(kind='tiny', **asdict(encoding)))
    return model.eval()


def export_smoke(package: Path, encoding: Encoding = DEFAULT_ENCODING):
    import torch
    import torch._inductor.config as config
    # Match the existing AOTI probe's package compiler selection, isolated cache.
    from native.bend_engine.aoti_probe.build_test_package import _resolve_package_cxx
    package = package.resolve()
    if package.exists() or package.with_suffix('.json').exists():
        raise FileExistsError('refusing to overwrite an existing evaluator package/manifest')
    package.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = smoke_model(encoding)

    class Outputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            result = self.model(x)
            return result['policy'], result['wdl']

    with torch.no_grad(), config.patch({'compile_threads': 1, 'cpp.cxx': (_resolve_package_cxx(),)}):
        graph = torch.export.export(Outputs().eval(), (torch.zeros((1, encoding.channels, 8, 8)),))
        torch._inductor.aoti_compile_and_package(graph, package_path=str(package))
    manifest = {'format': FORMAT, 'torch_version': str(torch.__version__),
                'sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
                'channels': encoding.channels, 'batch': 1, 'policy_width': 1858,
                'model': 'project TinyNet', 'weights': 'seeded-untrained', 'seed': SEED,
                **asdict(encoding)}
    package.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # CLI owns a disposable compilation directory; never consumes a live cache.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='bend-neural-aoti-') as cache:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache
        export_smoke(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
