// Focused public promotion; never reports inherited gates as newly executed.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const argv=process.argv.slice(2), only=argv.includes('--controls-only');
const args=argv.filter(x=>x!=='--controls-only');
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),
  'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler);
const suite=import.meta.dirname, root=path.resolve(suite,'../..'), engine=path.dirname(root);
const cli=path.join(compiler,'bend2/main.ts'), tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-lookup-'));
const names=['lookup_from_headers','selected_state_is_masked','initialized_lookup'], controls=[];
const sha=b=>createHash('sha256').update(b).digest('hex');
const text=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
function invoke(f){
  const r=spawnSync(process.execPath,[cli,f],{encoding:'utf8',timeout:900000,maxBuffer:64<<20,
    env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text.slice(-2500));assert.equal(r.text,'All terms check.',r.text.slice(-2500));}
function manifest(s){
  assert.deepEqual([...text(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);
  assert.deepEqual([...text(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);
  assert.match(text(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
  assert.match(text(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function policy(f,boundary,seen=new Set()){
  f=path.resolve(f);const rel=path.relative(boundary,f);
  assert.ok(!rel.split(path.sep).includes('..')&&
    (rel.startsWith('standalone/')||['legal_probe/Chess.bend','bitboard_probe/Sliders.bend'].includes(rel)), 'escaped source boundary');
  assert.ok(fs.lstatSync(f).isFile(),'nonregular input');assert.equal(fs.realpathSync(f),f,'symlinked input');
  if(seen.has(f))return seen;seen.add(f);
  const s=text(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
  for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){
    if(m[1]==='Base')continue;
    assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');
    policy(path.resolve(path.dirname(f),m[1]),boundary,seen);
  }return seen;
}
function replace(f,old,next){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(old).length,2,`unique mutation: ${old}`);fs.writeFileSync(f,s.replace(old,next));}
function copy(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/lookup')};}
function reject(name,entry,edit,location){
  const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry));
  assert.equal(r.status,1,`${name}: not a semantic rejection`);
  assert.match(r.text,/expected[\s\S]*observed/);
  assert.doesNotMatch(r.text,/a defined name|no such file|a decreasing self-call|more than once|RangeError|Maximum call stack|Segmentation fault/);
  assert.ok(location.test(r.text),`${name}: wrong location; ${r.text.slice(-1500)}`);
  controls.push({name,rejected:true,kind:'semantic source refinement',entry,
    diagnostic_sha256:sha(r.text),diagnostic_bytes:Buffer.byteLength(r.text),excerpt:r.text.slice(0,160)+'\n...\n'+r.text.slice(-700)});
  console.error(`PASS rejection: ${name}`);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
  manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
  for(const f of ['focused.js','verify.js','verify_native.py','probe.bend'])graph.add(path.join(suite,f));
  const hashes=()=>Object.fromEntries([...graph].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes();
  const archive=path.resolve(root,'../../../docs/experiments/evidence/bend-stored-data/supplementary');
  for(const f of ['Route.bend','Pipeline.bend','Canonical.bend','MaskedLookup.bend','probe.bend'])
    assert.equal(sha(fs.readFileSync(path.join(suite,f))),sha(fs.readFileSync(path.join(archive,f+'.txt'))),`promoted archive differs: ${f}`);
  if(!only){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS public lookup consumer');}
  const chess=d=>path.join(d.e,'legal_probe/Chess.bend');
  reject('actual-offset-plus-one','Route.bend',d=>replace(chess(d),'Sliders.lookup(table, U64.low(offset), index)',
    'Sliders.lookup(table, U32.inc(U64.low(offset)), index)'),/Location: model\b/);
  reject('actual-mask-from-next-slot','Route.bend',d=>replace(chess(d),'slide_mask(sq, occ, Array.get(U64, table, sq))',
    'slide_mask(sq, occ, Array.get(U64, table, U32.inc(sq)))'),/Location: model\b/);
  reject('actual-offset-from-next-slot','Route.bend',d=>replace(chess(d),'Array.get(U64, table, U32.add(128, sq))',
    'Array.get(U64, table, U32.add(129, sq))'),/Location: model\b/);
  reject('actual-data-address-subtraction','Route.bend',d=>replace(path.join(d.e,'bitboard_probe/Sliders.bend'),
    'Array.get(U64, table, U32.add(offset, index))','Array.get(U64, table, U32.sub(offset, index))'),/Location: model\b/);
  reject('actual-occupancy-ignored','Route.bend',d=>replace(chess(d),'Sliders.pext_index(occ, mask)',
    'Sliders.pext_index(U64.zero(), mask)'),/Location: model\b/);
  reject('unmasked-state-claim','Canonical.bend',d=>replace(path.join(d.s,'Canonical.bend'),
    '== U64.and(occ,Layout.mask(key)) : U64}:','== occ : U64}:'),/Location: state\b/);
  reject('public-zero-result','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'Result.expected(Nat.add(before,k),occ)) : Array<U64> & U64}', 'U64.zero()) : Array<U64> & U64}'),/Location: LAWS\.initialized_lookup\b/);
  reject('public-oversized-budget','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'Nat.add(Data.count(before,after),k),129n)', 'Nat.add(Data.count(before,after),k),130n)'),/Location: LAWS\.initialized_lookup\b/);
  guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law initialized_lookup:','def omitted:'),d=>manifest(d.s),/./);
  guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.initialized_lookup(','def omitted('),d=>manifest(d.s),/./);
  guard('missing-law-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
  guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
  guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Route.bend'),'\n?hole\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Route.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign import/);
  guard('unsafe-dependency',d=>replace(path.join(d.s,'Route.bend'),'def actual(','@unsafe\ndef actual('),d=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
  guard('symlinked-input',d=>{const f=path.join(d.s,'Route.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Route.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
  assert.throws(()=>clean({status:0,text:'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name:'zero-status-warning',rejected:true,kind:'exact-output wrapper unit test, no new compiler invocation'});
  assert.equal(controls.length,17);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:only?'NOT_RUN':'PASS',consumer:only?'NOT_RUN':'PASS',controls_gate:'PASS',
    compiler_revision:PIN.revision,...identity,new_laws:names,new_law_count:3,negative_controls:controls,
    inherited_gate_run:false,archive_bend_sources_identical:true,source_sha256s:before,
    scope:'Public source-to-source actual Chess.slide composition; independent relevant-mask and blocker-ray geometry remain unproved.'};
  const out=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
