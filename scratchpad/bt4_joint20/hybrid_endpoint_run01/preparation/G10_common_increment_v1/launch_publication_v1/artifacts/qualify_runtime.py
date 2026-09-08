"""CPU imports/CLI/native compatibility only; never reads corpus rows or launches stages."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STATE = Path(__file__).resolve().parent
ROOT = Path('/tmp/deepfin-g10-increment-tools')
PREP = STATE / 'runtime_snapshot_preparation.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    prep = json.loads(PREP.read_text())
    assert Path.cwd() == ROOT
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == prep['commit']
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], text=True).strip()
    import numpy
    import torch
    from numcodecs import blosc
    thread_before = {'torch': torch.get_num_threads(), 'blosc': blosc.get_nthreads()}
    from scripts import derive_corpus_targets as derive
    from scripts import sf_d9_rank_sidecar as rank
    from scripts import adapt_raw_bt4_sidecars as adapter
    for name in ('chess_anti_engine.encoding._features_ext', 'chess_anti_engine.encoding._lc0_ext',
                 'chess_anti_engine.mcts._mcts_tree', 'chess_anti_engine.nnue._nnue_ext'):
        importlib.import_module(name)
    assert Path(derive.__file__).resolve() == ROOT / 'scripts/derive_corpus_targets.py'
    assert Path(rank.__file__).resolve() == ROOT / 'scripts/sf_d9_rank_sidecar.py'
    assert Path(adapter.__file__).resolve() == ROOT / 'scripts/adapt_raw_bt4_sidecars.py'
    assert any('--source-shards' in a.option_strings for a in derive.build_parser()._actions)
    assert any('--source-shards' in a.option_strings for a in rank.build_parser()._actions)
    try:
        rank._verify_policy_support_exclusion({'result': None}, {}, raw_path=Path('not-read'), offset=0, raw_config='not-read')
    except ValueError as error:
        assert str(error) == 'policy support exclusion cannot replace a no-result drop'
    else:
        raise AssertionError('result-present guard absent')
    cli = {}
    for script in ('derive_corpus_targets.py', 'sf_d9_rank_sidecar.py', 'adapt_raw_bt4_sidecars.py'):
        command = [sys.executable, str(ROOT / 'scripts' / script), '--help']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        if script != 'adapt_raw_bt4_sidecars.py':
            assert '--source-shards' in result.stdout
        log = STATE / (script + '.help.txt')
        with log.open('x') as stream:
            stream.write(result.stdout)
            stream.write(result.stderr)
        cli[script] = {'argv': command, 'exit_code': result.returncode, 'output': str(log), 'sha256': sha(log)}
    imports = {}
    pins = dict(prep['initial_pins'])
    for name, module in sorted(sys.modules.copy().items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).absolute()
        if path.is_file() and path.is_relative_to(ROOT):
            imports[name] = str(path)
            pins[str(path)] = sha(path)
    versions = {}
    for distribution, module_name in [('torch', 'torch'), ('numpy', 'numpy'), ('zarr', 'zarr'),
                                     ('numcodecs', 'numcodecs'), ('python-chess', 'chess'), ('zstandard', 'zstandard')]:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        versions[distribution] = {'version': importlib.metadata.version(distribution), 'module_path': str(path)}
        pins[str(path)] = sha(path)
    # Pin only libraries actually loaded by this qualified process, not an environment walk.
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split()) >= 6 and line.split()[-1].startswith('/') and '.so' in line.split()[-1]})
    for path in libraries:
        pins[path] = sha(path)
    for path in (Path(sys.executable).resolve(), PREP, Path(__file__), STATE / 'python_bootstrap/sitecustomize.py'):
        pins[str(path)] = sha(path)
    for path, digest in pins.items():
        if path in prep['initial_pins']:
            assert digest == prep['initial_pins'][path]
    threads = {'torch': torch.get_num_threads(), 'blosc': blosc.get_nthreads()}
    assert threads == thread_before == {'torch': 2, 'blosc': 2}
    assert not torch.cuda.is_initialized() and torch.version.cuda is None
    result = {'status': 'qualified', 'checkout': str(ROOT), 'commit': prep['commit'], 'python': prep['python'],
              'pins': pins, 'features': {'closed_shard_selection': True, 'support_exclusion_requires_result': True},
              'qualification': {'method': 'Unchanged v3 CPU environment and23 compatible native source files;4 unchanged native links. Actual imports,3 direct help subprocesses and the result-present guard checked. No data reads or stages.',
                                'prior_receipt': prep['prior_receipt'], 'elapsed_seconds': time.monotonic() - started,
                                'python': sys.version, 'executable': sys.executable, 'resolved_executable': str(Path(sys.executable).resolve()),
                                'packages': versions, 'repository_imports': imports, 'loaded_libraries': libraries,
                                'cli_checks': cli, 'threads_before_repo_imports': thread_before, 'threads_after': threads,
                                'torch_cuda_build': torch.version.cuda, 'cuda_initialized': torch.cuda.is_initialized(),
                                'numpy_version': numpy.__version__, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                                'snapshot_preparation_sha256': sha(PREP),
                                'code_validation': {'path': '/home/josh/projects/chess/scratchpad/corpus_shard_selection_v1/author_validation.json',
                                                    'sha256': sha('/home/josh/projects/chess/scratchpad/corpus_shard_selection_v1/author_validation.json')}}}
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], text=True).strip()
    with (STATE / 'runtime_qualification.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'pins': len(pins), 'repository_modules': len(imports),
                      'libraries': len(libraries), 'elapsed_seconds': result['qualification']['elapsed_seconds'],
                      'receipt_sha256': sha(STATE / 'runtime_qualification.json')}))


if __name__ == '__main__':
    main()
