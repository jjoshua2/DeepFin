"""Full paired tensor verification worker; bank each completed arm immediately."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def write_atomic(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def authenticate_bank(preparation, roots):
    for label, suffix, binding in [('directory', '.zarr', 'source'), ('packed', '.zarr.zip', 'zip')]:
        root = Path(roots[label])
        expected = [f'shard_{i:06d}{suffix}' for i in range(len(preparation['records']))]
        if sorted(path.name for path in root.iterdir()) != expected:
            raise ValueError('staged qualification roster drift')
        for name, record in zip(expected, preparation['records'], strict=True):
            if (root / name).resolve(strict=True) != Path(record[binding]):
                raise ValueError('staged qualification binding drift')
    for record in preparation['records']:
        source = Path(record['source'])
        members = sorted(path for path in source.rglob('*') if path.is_file())
        if [str(path.relative_to(source)) for path in members] != sorted(record['members']):
            raise ValueError('source member roster drift')
        for path in members:
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['members'][str(path.relative_to(source))]:
                raise ValueError('source bytes drift')
        if hashlib.sha256(Path(record['zip']).read_bytes()).hexdigest() != record['zip_sha256']:
            raise ValueError('reused archive drift')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    runtime, out = Path(plan['runtime']), Path(plan['out'])
    sys.path.insert(0, str(runtime))
    path = runtime / 'scripts/qualify_packed_zarr_epoch.py'
    spec = importlib.util.spec_from_file_location('packed_qualification', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from numcodecs.blosc import set_nthreads
    set_nthreads(2)
    preparation = json.loads(Path(plan['preparation_receipt']).read_text())
    authenticate_bank(preparation, plan['roots'])
    options = {'batch_size': 512, 'seed': 121, 'input_planes': 175,
        'input_history_encoding': 'lc0_root_legacy_meta', 'history_rep_fix': True,
        'mirror_augmentation': True, 'plan_workers': 2, 'load_workers': 2,
        'max_working_set_bytes': 12 * 1024**3}
    runs = []
    for name, packed in [('directory', False), ('packed', True)]:
        print(f'starting full {name} tensor stream', flush=True)
        result = module.measure(Path(plan['roots'][name]), packed=packed, options=options)
        write_atomic(out / f'{name}.json', result)
        authenticate_bank(preparation, plan['roots'])
        runs.append(result)
        print(f'completed {name}: {result["rows"]} rows', flush=True)
    if any(run['rows'] != plan['rows'] for run in runs) or runs[0]['sequence_sha256'] != runs[1]['sequence_sha256']:
        raise ValueError('complete row/order/tensor parity failed')
    result = {'status': 'PASS_MATCHED_PACKED_ZARR_SAMPLER', 'runs': runs,
        'dependency_distribution_versions': {name: importlib.metadata.version(name)
            for name in ('numpy', 'torch', 'zarr', 'numcodecs', 'psutil', 'python-chess', 'PyYAML')},
        'python_version': sys.version}
    write_atomic(out / 'qualification.json', result)


if __name__ == '__main__':
    main()
