// Additive opt-in gate. Prior laws/checkers remain unchanged and are re-executed.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1 || (args.length===3 && args[1]==='--report'),
  'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler);
const root=path.resolve(import.meta.dirname,'../..'), engine=path.dirname(root);
const suite=path.join(root,'proofs/exact_index'), cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-exact-index-laws-'));
const required=['low_projection_exact','lookup_index_exact','lookup_sequence_ordinal',
  'lookup_collision_sound','lookup_collision_complete','lookup_state_recovery'];
const negatives=[], sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(s=>s.split('#')[0]).join('\n');
function manifest(s) {
  assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
  assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
  for(const text of ['import ./LAWS.bend as Laws','import ../layout/PROOF.bend as LayoutProof',
    'import ../successor/PROOF.bend as SuccessorProof']) assert.ok(code(path.join(s,'PROOF.bend')).includes(text));
  assert.ok(code(path.join(s,'consumer.bend')).includes('import ./PROOF.bend as Proof'));
}
function policy(file,base,seen=new Set()) {
  file=path.resolve(file);
  const rel=path.relative(base,file);
  assert.ok(rel.startsWith('standalone/') || rel==='bitboard_probe/Sliders.bend','unexpected proof boundary');
  assert.ok(fs.lstatSync(file).isFile() && !fs.lstatSync(file).isSymbolicLink(),'nonregular proof input');
  assert.equal(fs.realpathSync(file),file,'symlinked proof path');
  if(seen.has(file))return seen;
  seen.add(file);const s=code(file);
  assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
  for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)) {
    if(m[1]==='Base')continue;
    assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign proof import');
    policy(path.resolve(path.dirname(file),m[1]),base,seen);
  }
  return seen;
}
function invoke(file,timeout=600000) {
  const r=spawnSync(process.execPath,[cli,file],{encoding:'utf8',timeout,maxBuffer:16<<20,
    env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,`checker error: ${r.error}`);assert.equal(r.signal,null);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r) {assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function replace(f,before,after) {
  const s=fs.readFileSync(f,'utf8');assert.equal(s.split(before).length,2,'nonunique mutation anchor');
  fs.writeFileSync(f,s.replace(before,after));
}
function copy(name) {
  const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
  for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
  fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});
  fs.mkdirSync(path.join(e,'bitboard_probe'));
  fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
  return {e,r,s:path.join(r,'proofs/exact_index')};
}
function semantic(name,edit,entry) {
  const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry),60000);
  assert.equal(r.status,1,`${name}: wrong status\n${r.text}`);
  assert.match(r.text,/expected[\s\S]*observed/,`${name}: not a type/proof rejection\n${r.text}`);
  negatives.push({name,rejected:true,kind:'checker',entry});console.error('PASS rejection: '+name);
}
function rejectedPolicy(name,edit,check,reason) {
  const d=copy(name);edit(d);assert.throws(()=>check(d),reason);
  negatives.push({name,rejected:true,kind:'manifest/import policy'});
}
try {
  manifest(suite);
  const graph=policy(path.join(suite,'consumer.bend'),engine);
  for(const f of ['verify.js','verify_native.js','probe.bend'])graph.add(path.join(suite,f));
  graph.add(path.join(root,'verify_compiler.js'));graph.add(path.join(root,'toolchain.json'));
  const snapshot=()=>Object.fromEntries([...graph].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=snapshot();
  const parentRun=spawnSync(process.execPath,[path.join(root,'proofs/layout/verify.js'),compiler],
    {encoding:'utf8',timeout:900000,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(parentRun.error,undefined,`${parentRun.error}`);assert.equal(parentRun.signal,null);
  assert.equal(parentRun.status,0,parentRun.stdout+parentRun.stderr);
  const parent=JSON.parse(parentRun.stdout);
  assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,48);
  assert.equal(parent.negative_controls.length,13);assert.equal(parent.inherited_negative_controls,42);
  assert.ok(parent.negative_controls.every(n=>n.rejected));console.error('PASS: inherited 48 laws and 55 controls');
  clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: all 54 laws and importing consumer');

  semantic('projection-allows-33-bits',d=>replace(path.join(d.s,'Projection.bend'),
    'Nat.is_le(k,32n)','Nat.is_le(k,33n)'),'Projection.bend');
  semantic('inclusive-projection-bound',d=>replace(path.join(d.s,'Projection.bend'),
    'Nat.is_lt(Bits.value(x),Bits.power(k))','Nat.is_le(Bits.value(x),Bits.power(k))'),'Projection.bend');
  semantic('inclusive-ordinal-bound',d=>replace(path.join(d.s,'Correspondence.bend'),
    'def ordinal(+key: U32, +i: Nat, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},\n  bounded: {Nat.is_lt(i,U32.to_nat(Spec.size(key)))',
    'def ordinal(+key: U32, +i: Nat, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},\n  bounded: {Nat.is_le(i,U32.to_nat(Spec.size(key)))'),'Correspondence.bend');
  semantic('recover-irrelevant-bits',d=>replace(path.join(d.s,'Correspondence.bend'),
    '== U64.and(occ,Spec.mask(key)) : U64}:','== occ : U64}:'),'Correspondence.bend');
  for(const [name,after] of [['runtime-index-zero','0'],['runtime-index-right-shift',
    'U32.shrn(U64.low(U64.pext(occupied, mask)),1n)'],['runtime-index-truncated-two-bits',
    'U32.and(U64.low(U64.pext(occupied, mask)),3)'],['runtime-index-high-half',
    'U64.high(U64.pext(occupied, mask))'],['runtime-pext-operands-reversed',
    'U64.low(U64.pext(mask, occupied))']]) {
    semantic(name,d=>replace(path.join(d.e,'bitboard_probe/Sliders.bend'),
      'U64.low(U64.pext(occupied, mask))',after),'Boundary.bend');
  }
  rejectedPolicy('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),
    'law lookup_state_recovery:','def removed:'),d=>manifest(d.s),/./);
  rejectedPolicy('missing-proof-definition',d=>replace(path.join(d.s,'PROOF.bend'),
    'def Laws.lookup_state_recovery(key,occ,valid):','def removed(key: U32):'),d=>manifest(d.s),/./);
  rejectedPolicy('missing-layout-proof-import',d=>replace(path.join(d.s,'PROOF.bend'),
    'import ../layout/PROOF.bend as LayoutProof\n',''),d=>manifest(d.s),/./);
  rejectedPolicy('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Projection.bend'),'\n?TODO\n'),
    d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  rejectedPolicy('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Projection.bend'),'\nimport "./oracle.c"\n'),
    d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign proof import/);
  const unsafe=copy('unsafe-projection');
  replace(path.join(unsafe.s,'Projection.bend'),'def take_bounded(','@unsafe\ndef take_bounded(');
  assert.throws(()=>policy(path.join(unsafe.s,'Projection.bend'),unsafe.e),/unsafe dependency/);
  const r=invoke(path.join(unsafe.s,'Projection.bend'),60000);
  assert.match(r.text,/unsafe or foreign/);assert.throws(()=>clean(r));
  negatives.push({name:'unsafe-projection',rejected:true,kind:'policy and exact checker-output guard',raw_cli_status:r.status});
  assert.deepEqual(snapshot(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const result={proof_gate:'PASS',compiler_revision:PIN.revision,...identity,new_universal_laws:required,
    new_law_count:6,inherited_law_count:48,aggregate_law_count:54,inherited_negative_controls:55,
    negative_controls:negatives,consumer:'PASS',source_sha256s:before,
    scope:'Exact U32/full-PEXT correspondence, ordinal, relevant-occupancy recovery and both collision directions; NOT prefix offsets, affine table storage or attack geometry'};
  assert.equal(negatives.length,15);
  const text=JSON.stringify(result,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
