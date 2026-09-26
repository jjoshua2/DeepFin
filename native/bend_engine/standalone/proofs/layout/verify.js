// Bounded, opt-in aggregate gate. No routine CI/model/perft workload is added.
// Usage: bun native/bend_engine/standalone/proofs/layout/verify.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';

const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]);
const identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '../..');
const engine = path.dirname(root);
const suite = path.join(root, 'proofs/layout');
const cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-layout-laws-'));
const required = ['mask_population','tight_population_bound','shift_in_range','size_positive',
  'tight_size_bound','size_matches_capacity','full_index_bound','lookup_index_bound'];
const sha = b => createHash('sha256').update(b).digest('hex');
const code = f => fs.readFileSync(f, 'utf8').split('\n').map(s => s.split('#')[0]).join('\n');
const negatives = [];
function invoke(file, timeout = 600000) {
  const r = spawnSync(process.execPath, [cli, file], {encoding:'utf8', timeout,
    maxBuffer: 16 << 20, env:{...process.env, BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error, undefined, `checker execution error: ${r.error}`);
  assert.equal(r.signal, null, `checker signal: ${r.signal}`);
  return {status:r.status, text:(r.stdout+r.stderr).trim()};
}
function clean(r) {
  assert.equal(r.status,0,r.text);
  assert.equal(r.text,'All terms check.',r.text);
}
function policy(file, base, seen = new Set()) {
  file = path.resolve(file);
  const rel = path.relative(base,file);
  assert.ok(rel.startsWith('standalone/') || rel === 'bitboard_probe/Sliders.bend',
    'proof import outside approved implementation boundary');
  assert.ok(fs.lstatSync(file).isFile() && !fs.lstatSync(file).isSymbolicLink(), 'nonregular proof file');
  assert.equal(fs.realpathSync(file),file,'symlinked proof path');
  if (seen.has(file)) return seen;
  seen.add(file);
  const s = code(file);
  assert.doesNotMatch(s, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const m of s.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (m[1] === 'Base') continue;
    assert.match(m[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign or unexpected import');
    policy(path.resolve(path.dirname(file),m[1]),base,seen);
  }
  return seen;
}
function manifest(s) {
  assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
  assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
  assert.match(code(path.join(s,'PROOF.bend')), /import \.\/LAWS\.bend as Laws/);
  assert.match(code(path.join(s,'consumer.bend')), /import \.\/PROOF\.bend as Proof/);
  assert.match(code(path.join(s,'consumer.bend')), /import \.\.\/successor\/PROOF\.bend as OrdinalProof/);
}
function observations(s, r) {
  // Size is an inline runtime expression, so retain its exact scalar link.
  // This guard is not a theorem about the Array builder or its offsets.
  const norm = x => x.replace(/\s+/g,'');
  assert.ok(norm(code(path.join(r,'Tables.bend'))).includes(
    '+mask=slider(sq,bishop,U64.zero(),True{})+size=U32.shln(1,U32.to_nat(U64.popcount(mask)))'));
  const spec = norm(code(path.join(s,'Spec.bend')));
  assert.ok(spec.includes('Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.zero(),True{})'));
  assert.ok(spec.includes('defsize(key:U32)->U32:U32.shln(1,U32.to_nat(U64.popcount(mask(key))))'));
}
function snapshot(files) {
  return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
}
function replace(f,before,after) {
  const s=fs.readFileSync(f,'utf8');
  assert.equal(s.split(before).length,2,`nonunique mutation anchor: ${before}`);
  fs.writeFileSync(f,s.replace(before,after));
}
function copy(name) {
  const e=path.join(temp,name), r=path.join(e,'standalone');
  fs.mkdirSync(r,{recursive:true});
  for (const f of ['Tables.bend','Subsets.bend']) fs.copyFileSync(path.join(root,f),path.join(r,f));
  fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});
  fs.mkdirSync(path.join(e,'bitboard_probe'));
  fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
  return {e,r,s:path.join(r,'proofs/layout')};
}
function semantic(name,edit,entry) {
  const d=copy(name);edit(d);
  // Target the affected lemma. False first-key Cases statements fail early;
  // no expensive valid proof is replaced, assumed, or removed in the real tree.
  const r=invoke(path.join(d.s,entry),60000);
  assert.equal(r.status,1,`${name}: mutation accepted\n${r.text}`);
  assert.match(r.text,/expected[\s\S]*observed/,`${name}: wrong rejection\n${r.text}`);
  negatives.push({name,rejected:true,kind:'checker rejects affected lemma',entry});
  console.error('PASS rejection: '+name);
}
function rejectedPolicy(name,edit,check,reason) {
  const d=copy(name);edit(d);
  assert.throws(()=>check(d),reason);
  negatives.push({name,rejected:true,kind:'manifest/import policy'});
}
try {
  manifest(suite);observations(suite,root);
  const graph=policy(path.join(suite,'consumer.bend'),engine);
  const tracked=new Set([...graph,path.join(suite,'verify.js'),path.join(root,'verify_compiler.js'),path.join(root,'toolchain.json')]);
  const before=snapshot(tracked);
  const inherited=spawnSync(process.execPath,[path.join(root,'proofs/successor/verify.js'),compiler],
    {encoding:'utf8',timeout:180000,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(inherited.error,undefined,`${inherited.error}`);
  assert.equal(inherited.signal,null);
  assert.equal(inherited.status,0,inherited.stdout+inherited.stderr);
  const parent=JSON.parse(inherited.stdout);
  assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,40);
  assert.equal(parent.negative_controls.length,16);assert.equal(parent.prior_negative_controls,26);
  assert.ok(parent.negative_controls.every(n=>n.rejected));
  clean(invoke(path.join(suite,'consumer.bend')));
  console.error('PASS: all 48 laws, importing consumer and inherited controls');

  semantic('rook-limit-eleven',d=>replace(path.join(d.s,'Spec.bend'),',9,12)',',9,11)'),'Facts.bend');
  semantic('bishop-limit-eight',d=>replace(path.join(d.s,'Spec.bend'),',9,12)',',8,12)'),'Facts.bend');
  semantic('runtime-ray-includes-edges',d=>replace(path.join(d.r,'Tables.bend'),
    'Bool.and(mask, U32.is_eq(later, 64))','False{}'),'Cases.bend');
  semantic('runtime-mask-always-empty',d=>{
    const f=path.join(d.r,'Tables.bend'),s=fs.readFileSync(f,'utf8');
    const a=s.indexOf('  +d = Bool.pick(U32, bishop, 4, 0)'),b=s.indexOf('\ndef fill(',a);
    assert.ok(a>0&&b>a);fs.writeFileSync(f,s.slice(0,a)+'  U64.zero()\n'+s.slice(b));
  },'Cases.bend');
  semantic('inline-size-shift-32',d=>replace(path.join(d.s,'Spec.bend'),
    'U32.shln(1,U32.to_nat(U64.popcount(mask(key))))','U32.shln(1,32n)'),'Consequences.bend');
  semantic('lookup-index-64-exceeds-bishop-b2',d=>replace(path.join(d.e,'bitboard_probe/Sliders.bend'),
    'U64.low(U64.pext(occupied, mask))','64'),'Consequences.bend');
  semantic('geometric-count-includes-endpoint',d=>replace(path.join(d.s,'Geometry.bend'),
    'Nat.sub(distance,1n)','distance'),'Cases.bend');
  rejectedPolicy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law lookup_index_bound:','def removed:'),
    d=>manifest(d.s),/./);
  rejectedPolicy('missing-proof-definition',d=>replace(path.join(d.s,'PROOF.bend'),
    'def Laws.lookup_index_bound(key,occ,valid):','def removed(key: U32):'),d=>manifest(d.s),/./);
  rejectedPolicy('proof-hole',d=>fs.appendFileSync(path.join(d.s,'PROOF.bend'),'\n?TODO\n'),
    d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  rejectedPolicy('foreign-proof-import',d=>fs.appendFileSync(path.join(d.s,'Spec.bend'),'\nimport "./oracle.c"\n'),
    d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign or unexpected import/);
  rejectedPolicy('missing-prior-ordinal-import',d=>replace(path.join(d.s,'consumer.bend'),
    'import ../successor/PROOF.bend as OrdinalProof\n',''),d=>manifest(d.s),/./);
  const unsafe=copy('unsafe-low-projection');
  replace(path.join(unsafe.s,'Low.bend'),'def low_le(','@unsafe\ndef low_le(');
  assert.throws(()=>policy(path.join(unsafe.s,'Consequences.bend'),unsafe.e),/unsafe dependency/);
  const warned=invoke(path.join(unsafe.s,'Consequences.bend'),60000);
  assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
  negatives.push({name:'unsafe-low-projection',rejected:true,kind:'policy and exact checker-output guard',raw_cli_status:warned.status});

  assert.deepEqual(snapshot(tracked),before,'proof sources changed during verification');
  assert.deepEqual(verifyCompiler(compiler),identity);
  const result={proof_gate:'PASS',compiler_revision:PIN.revision,...identity,
    new_universal_laws:required,new_law_count:required.length,finite_key_domain:[0,127],
    aggregate_law_count:48,inherited_law_count:40,inherited_negative_controls:42,
    negative_controls:negatives,consumer:'PASS',source_sha256s:before,
    proof_method:'Complete finite-domain checker reduction of actual mask counts; structural arbitrary-occupancy bound',
    scope:'Mask populations, exact positive U32 sizes and actual PEXT lookup index bounds; NOT prefix offsets, affine initialization, full ray geometry or end-to-end native correctness'};
  const text=JSON.stringify(result,null,2)+'\n';
  if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
