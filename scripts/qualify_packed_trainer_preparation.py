"""CPU worker used only beneath the hard-bounded preparation supervisor."""
import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'directory', 'packed', 'result'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.runtime))
    path = a.runtime / 'scripts/qualify_packed_zarr_epoch.py'
    spec = importlib.util.spec_from_file_location('packed_qualifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from numcodecs.blosc import set_nthreads
    set_nthreads(2)
    result = module.qualify(a.directory, a.packed, {
        'batch_size': 512, 'seed': 121, 'input_planes': 175,
        'input_history_encoding': 'lc0_root_legacy_meta', 'history_rep_fix': True,
        'mirror_augmentation': True, 'plan_workers': 2, 'load_workers': 2,
        'max_working_set_bytes': 12 * 1024**3})
    result['dependency_distribution_versions'] = {name: importlib.metadata.version(name)
        for name in ('numpy', 'torch', 'zarr', 'numcodecs', 'psutil', 'python-chess', 'PyYAML')}
    result['python_version'] = sys.version
    with a.result.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
