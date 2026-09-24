// Independent ray/lookup refinement. Controls-only receipts do not claim consumer execution.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),only=args.includes('--controls-only'),a=args.filter(x=>x!=='--controls-only');
assert.ok(a.length===1||(a.length===3&&a[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler=path.resolve(a[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname,engine=path.resolve(suite,'../../..');
const cli=path.join(compiler,'bend2/main.ts'),temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-ray-laws-'));
const names=['seven_steps_reach_edge','ray_matches_independent_path','slider_matches_independent_rays','relevant_mask_preserves_attacks','initialized_lookup_matches_independent_rays'];
const sha=b=>createHash('sha256').update(b).digest('hex'),controls=[];
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
function invoke(file,timeout=1200000){const p=spawnSync(process.execPath,[cli,file],{encoding:'utf8',timeout,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(p.error,undefined,String(p.error));assert.equal(p.signal,null,'crash/interruption is not semantic rejection');return {status:p.status,text:(p.stdout+p.stderr).trim()};}
function clean(p){assert.equal(p.status,0,p.text.slice(-2000));assert.equal(p.text,'All terms check.',p.text.slice(-2000));}
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),names);
 const proof=code(path.join(s,'PROOF.bend'));assert.deepEqual([...proof.matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),names);
 assert.match(proof,/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,boundary,seen=new Set()){
 f=path.resolve(f);const rel=path.relative(boundary,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'),'escaped source scope');
 assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlinked proof input');
 if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');graph(path.resolve(path.dirname(f),m[1]),boundary,seen);}return seen;}
function replace(f,x,y){const t=fs.readFileSync(f,'utf8');assert.equal(t.split(x).length,2,'nonunique mutation site');fs.writeFileSync(f,t.replace(x,y));}
function copy(name){const e=path.join(temp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/ray'),table:path.join(e,'standalone/Tables.bend')};}
function semantic(name,entry,edit,location,kind='new implementation/specification refinement'){
 const d=copy(name);edit(d);const p=invoke(path.join(d.s,entry));assert.equal(p.status,1,`${name}: expected ordinary rejection`);
 assert.match(p.text,/expected[\s\S]*observed/,`${name}: no semantic diagnostic`);
 assert.doesNotMatch(p.text,/no such file|a defined name|more than once|a decreasing self-call|RangeError|Maximum call stack|Segmentation fault/,'harness/type-discipline error is not semantic rejection');
 assert.ok(location.test(p.text),`${name}: wrong location ${p.text.slice(-900)}`);
 controls.push({name,rejected:true,kind,entry,diagnostic_sha256:sha(p.text),diagnostic_bytes:Buffer.byteLength(p.text),excerpt:p.text.slice(0,120)+'\n...\n'+p.text.slice(-700)});
 console.error('PASS rejection: '+name);fs.rmSync(d.e,{recursive:true,force:true});}
function guard(name,edit,check,pattern){const d=copy(name);edit(d);assert.throws(()=>check(d),pattern);controls.push({name,rejected:true,kind:'manifest/import policy'});fs.rmSync(d.e,{recursive:true,force:true});}
try{
 manifest(suite);const inputs=graph(path.join(suite,'consumer.bend'),engine);
 for(const f of ['focused.js','verify.js','verify_native.js','probe.bend'])inputs.add(path.join(suite,f));
 const hashes=()=>Object.fromEntries([...inputs].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));const before=hashes();
 if(!only){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: five public laws and importing consumer');}
 semantic('actual-ignore-blocker','Traversal.bend',d=>replace(d.table,
  'stop = Bool.or(U32.is_eq(next, 64), U64.test_bit(occ, U32.to_nat(next)))','stop = U32.is_eq(next, 64)'),/Location: actual\b/);
 semantic('actual-omit-blocker','Traversal.bend',d=>replace(d.table,
  'acc = U64.or(acc, Bool.pick(U64, add, U64.bit(U32.to_nat(next)), U64.zero()))',
  'acc = U64.or(acc, Bool.pick(U64, Bool.and(add,Bool.not(U64.test_bit(occ,U32.to_nat(next)))), U64.bit(U32.to_nat(next)), U64.zero()))'),/Location: actual\b/);
 semantic('actual-drop-accumulator','Traversal.bend',d=>{
  const t=fs.readFileSync(d.table,'utf8');assert.ok(t.includes('case 0n: acc'));fs.writeFileSync(d.table,t.replace('case 0n: acc','case 0n: U64.zero()'));},/Location: stopped\b/);
 semantic('actual-ignore-file-boundary','Trace.bend',d=>replace(d.table,
  'Bool.and(U32.is_lt(x, 8), U32.is_lt(y, 8))','U32.is_lt(y, 8)'),/Location: square_\d+\b/,'actual coordinate implementation');
 semantic('truncate-independent-route','Trace.bend',d=>replace(path.join(d.s,'Trace.bend'),
  'R.path(7n,s,d)','R.path(6n,s,d)'),/Location: square_\d+\b/,'finite input-only route certificate');
 semantic('six-steps-do-not-reach-every-edge','Termination.bend',d=>replace(path.join(d.s,'Termination.bend'),
  'R.path(7n,s,d)','R.path(6n,s,d)'),/Location: square_\d+\b/,'independent termination certificate');
 semantic('allow-absent-mask-bit','Masking.bend',d=>{
  const f=path.join(d.s,'Masking.bend'),t=fs.readFileSync(f,'utf8');
  const x='e: {U64.and(mask,U64.bit(x)) == U64.bit(x) : U64}) ->';assert.ok(t.includes(x));
  fs.writeFileSync(f,t.replace(x,'e: {U64.and(mask,U64.bit(x)) == U64.zero() : U64}) ->'));},/Location: bit_preserved\b/,'structural mask-observation proof');
 semantic('ignore-first-interior-observation','Frame.bend',d=>replace(path.join(d.s,'Path.bend'),
  'Bool.and(eq(U64.test_bit(a,x),U64.test_bit(b,x)),agree(y <> rest,a,b))','agree(y <> rest,a,b)'),/Location: interior_only\b/,'independent path specification');
 semantic('actual-mask-retains-terminal-square','Masks.bend',d=>replace(d.table,
  'Bool.not(Bool.and(mask, U32.is_eq(later, 64)))','True{}'),/Location: (finite|actual)\b/,'promoted actual-mask refinement');
 semantic('initialized-lookup-returns-original-buffer','Lookup.bend',d=>{
  const f=path.join(d.s,'Lookup.bend'),t=fs.readFileSync(f,'utf8');
  const x='(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),S.rays(Nat.add(before,k),occ)) : Array<U64> & U64}:';
  assert.equal(t.split(x).length,2);fs.writeFileSync(f,t.replace(x,'(Array.new(U64,d,seed),S.rays(Nat.add(before,k),occ)) : Array<U64> & U64}:'));},/Location: actual\b/,'complete actual-array result');
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law seven_steps_reach_edge:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.seven_steps_reach_edge(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Traversal.bend'),'\n?hole\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Traversal.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e),/foreign import/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Traversal.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Traversal.bend.original',f);},d=>graph(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 guard('unsafe-proof',d=>replace(path.join(d.s,'Traversal.bend'),'def actual(','@unsafe\ndef actual('),d=>graph(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
 assert.throws(()=>clean({status:0,text:'All terms check.\nWARNING: unsafe or foreign dependency'}));
 controls.push({name:'zero-exit-with-warning',rejected:true,kind:'synthetic exact-output wrapper unit; not compiler execution'});
 assert.equal(controls.length,19);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={focused_gate:only?'NOT_RUN':'PASS',consumer:only?'NOT_RUN':'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:5,new_laws:names,inherited_gate_run:false,negative_controls:controls,source_sha256s:before,scope:'Independent complete coordinate rays, arbitrary occupancy blocker inclusion and terminal-mask irrelevance, composed with actual initialized array lookup. No native allocation/lifetime/compiler proof.'};
 const text=JSON.stringify(report,null,2)+'\n';if(a[2])fs.writeFileSync(a[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
