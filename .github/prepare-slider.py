from pathlib import Path
import base64, gzip, hashlib, json, os, runpy, subprocess

ns = runpy.run_path('.github/prepare-slider-base.py')
report, env, files = ns['REPORT'], ns['env'], ns['files']
old = ns['tree']
assert old == 'a7640c87ee9d6955f873f458a4cdcd25e8981f6d'
patch = gzip.decompress(base64.b64decode(Path('.github/slider-boxed.b64').read_text()))
assert hashlib.sha256(patch).hexdigest() == 'b9b4d06284575cfc7e4a8b49a4a66205de7bde504fc687a92fbbb1f553e90fb2'
(report/'boxed.patch').write_bytes(patch)
subprocess.run(['git','apply','--check',str(report/'boxed.patch')],check=True)
subprocess.run(['git','apply',str(report/'boxed.patch')],check=True)
subprocess.run(['git','add','--',*files],env=env,check=True)
new = subprocess.check_output(['git','write-tree'],env=env,text=True).strip()
assert new == '1405694cafd21da1ab5526ebc7a57edfdf4a8d8e', new
changed = subprocess.check_output(['git','diff','--name-only',old,new],text=True).splitlines()
assert changed == [
 'docs/experiments/2026-09-20-bend-slider-laws.md',
 'native/bend_engine/standalone/LAWS.bend',
 'native/bend_engine/standalone/PROOF.bend',
 'native/bend_engine/standalone/proofs/Core.bend',
 'native/bend_engine/standalone/proofs/README.md',
 'native/bend_engine/standalone/proofs/Spec.bend',
 'native/bend_engine/standalone/verify_laws.js'], changed
# Engine/adapter/oracle/build inputs and compiler/proof dependencies are IDENTICAL
# to the fresh five-mode native-tested candidate; only certificate packaging,
# its new forged-certificate control and documentation differ.
identity = ns['identity']
identity.update(candidate_tree=new, native_tested_tree=old, proof_only_changes=changed,
 native_evidence_run=35556227949, native_evidence_job=106200288255,
 native_evidence_artifact=10620212310)
(report/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
subprocess.run(['git','diff','--check'],check=True)
print(json.dumps(identity,indent=2))
