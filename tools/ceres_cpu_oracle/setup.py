"""Task-local SDK/source/restore setup. Called only by the bounded owned operator."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    root = Path(plan['work_root'])
    for name in ('cli_home', 'nuget', 'nuget_http', 'tmp'):
        (root/name).mkdir()
    archive = root/'sdk.tar.gz'
    digest = hashlib.sha512()
    count = 0
    with urllib.request.urlopen(plan['sdk']['url'], timeout=30) as response, archive.open('xb') as out:
        while block := response.read(1024**2):
            count += len(block)
            if count > 241 * 1024**2:
                raise ValueError('SDK download byte limit')
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != plan['sdk']['sha512']:
        raise ValueError('SDK SHA512 mismatch')
    sdk = root/'sdk'
    sdk.mkdir()
    with tarfile.open(archive) as tar:
        total = 0
        for member in tar:
            total += member.size
            if total > 2 * 1024**3:
                raise ValueError('SDK extracted size limit')
            tar.extract(member, sdk, filter='data')
    source = root/'source'
    subprocess.run(['git', 'init', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', 'https://github.com/dje-dev/Ceres.git', plan['upstream_commit']], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != plan['upstream_commit']:
        raise ValueError('source commit')
    for name, digest_expected in plan['upstream_source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest_expected:
            raise ValueError('upstream source mismatch: '+name)
    harness = root/'harness'
    harness.mkdir()
    for name in ('Oracle.csproj', 'Program.cs'):
        shutil.copyfile(Path(plan['author_root'])/'tools/ceres_cpu_oracle'/name, harness/name)
    (root/'global.json').write_text(json.dumps({'sdk': {'version': '10.0.401', 'rollForward': 'disable'}}))
    subprocess.run([str(sdk/'dotnet'), 'restore', str(harness/'Oracle.csproj'),
        '--disable-build-servers', '--disable-parallel', '-p:RestorePackagesWithLockFile=true',
        '-m:1', '/nodeReuse:false', '/p:UseSharedCompilation=false'], cwd=root, check=True)
    # Package lock/asset evidence retained without reading package payloads again.
    metadata = []
    for name in ('packages.lock.json', 'project.assets.json'):
        metadata.extend({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in root.rglob(name))
    (root/'restore_evidence.json').write_text(json.dumps({'upstream_commit': commit, 'sdk_sha512': plan['sdk']['sha512'], 'metadata': metadata}, indent=2)+'\n')


if __name__ == '__main__':
    main()
