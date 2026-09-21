from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess

ROOT = Path.cwd()
BASE = 'fe658a2680ec7941742c045feff0ccb806a898ad'
PIN = 'aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae'
COMPILER = Path('/tmp/deepfin-slider-compiler')
REPORT = Path('/tmp/slider-validation')
REPORT.mkdir(exist_ok=True)
def run(*args, **kw):
    return subprocess.check_output(args, **kw).decode().strip()
if not COMPILER.exists():
    subprocess.run(['git','init',str(COMPILER)], check=True)
    subprocess.run(['git','-C',str(COMPILER),'fetch','--depth','1','https://github.com/jjoshua2/bend.git',PIN], check=True)
    subprocess.run(['git','-C',str(COMPILER),'checkout','--detach','FETCH_HEAD'], check=True)
assert run('git','-C',str(COMPILER),'rev-parse','HEAD') == PIN
raw = Path('.github/slider-part0.b64').read_text() + Path('.github/slider-part1.b64').read_text()
raw = raw.replace('TPshtKpZFg', 'TPshtKZFg')
patch = gzip.decompress(base64.b64decode(raw))
assert hashlib.sha256(patch).hexdigest() == '7d768e358d4fe5cc2ce8d9a5dda6da14f30a87e518c99cbc3bc9f20a79acefec'
(REPORT/'candidate.patch').write_bytes(patch)
subprocess.run(['git','apply','--check',str(REPORT/'candidate.patch')], check=True)
subprocess.run(['git','apply',str(REPORT/'candidate.patch')], check=True)
files = [line.removeprefix('+++ b/') for line in patch.decode().splitlines() if line.startswith('+++ b/')]
vendor = Path('native/bend_engine/standalone/proofs/u64')
vendor.mkdir(parents=True, exist_ok=True)
for name in ['LAWS.bend','PROOF.bend','Words.bend','LICENSE']:
    source = 'LICENSE' if name == 'LICENSE' else 'demos/proof_u64/' + name
    content = subprocess.check_output(['git','-C',str(COMPILER),'show',PIN+':'+source])
    (vendor/name).write_bytes(content)
    files.append(str(vendor/name))
files.append('.github/workflows/bend-slider-laws.yml')
assert len(files) == len(set(files)) == 20
index = REPORT/'candidate-index'
index.unlink(missing_ok=True)
env = dict(os.environ, GIT_INDEX_FILE=str(index))
subprocess.run(['git','read-tree',BASE], env=env, check=True)
subprocess.run(['git','add','--',*files], env=env, check=True)
old = run('git','write-tree',env=env)
assert old == '00bb12ce6f6baf4f2a0b3ddc485d96085bf6f064', old
# Repair only proof evaluation and its documentation. Original eight conditions
# are retained and an additional universal theorem proves reflection agreement.
repair = gzip.decompress(base64.b64decode(Path('.github/slider-repair.b64').read_text()))
assert hashlib.sha256(repair).hexdigest() == 'b39fe7f8c70a4b5e66389da87abfaa0cf13f688dd3e223facc3211231de81efd'
(REPORT/'repair.patch').write_bytes(repair)
subprocess.run(['git','apply','--check',str(REPORT/'repair.patch')], check=True)
subprocess.run(['git','apply',str(REPORT/'repair.patch')], check=True)
files.append('native/bend_engine/standalone/proofs/Fast.bend')
assert len(files) == len(set(files)) == 21
subprocess.run(['git','diff','--check'], check=True)
original = run('git','ls-tree','-r','--name-only',BASE,'native/bend_engine').splitlines()
for name in original:
    if name.endswith('.bend'):
        assert Path(name).read_bytes() == subprocess.check_output(['git','show',BASE+':'+name]), name
subprocess.run(['git','add','--',*files], env=env, check=True)
tree = run('git','write-tree',env=env)
assert tree == 'a7640c87ee9d6955f873f458a4cdcd25e8981f6d', tree
identity = {'candidate_tree':tree, 'prior_candidate_tree':old, 'base_commit':BASE, 'compiler_commit':PIN,
            'original_engine_bend_files_unchanged':True, 'files':sorted(files)}
(REPORT/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
