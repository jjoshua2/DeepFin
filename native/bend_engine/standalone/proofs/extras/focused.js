// Focused final-phase source contracts. verify.js also retains the complete parent gate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=import.meta.dirname,cli=path.join(compiler,'bend2/main.ts');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-extras-laws-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const names=['bounded_extras_read','table_pipeline_extras_read'],controls=[];
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(x=>x.split('#')[0]).join('\n');
function invoke(f){const r=spawnSync(process.execPath,[cli,f],{encoding:'utf8',timeout:180000,maxBuffer:64<<20,
  env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};}
function clean(r){assert.equal(r.status,0,r.text.slice(0,2000));assert.equal(r.text,'All terms check.',r.text.slice(0,2000));}
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),names);
  const p=code(path.join(s,'PROOF.bend'));assert.deepEqual([...p.matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),names);
  assert.match(p,/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function policy(f,boundary,seen=new Set()){
  f=path.resolve(f);const rel=path.relative(boundary,f);assert.ok(!rel.split(path.sep).includes('..')&&rel.startsWith('standalone/'),'escaped scope');
  assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlinked proof input');
  if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
  for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;
    assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');policy(path.resolve(path.dirname(f),m[1]),boundary,seen);}return seen;}
function replace(f,old,value){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(old).length,2,'nonunique mutation');fs.writeFileSync(f,s.replace(old,value));}
function copy(name){const e=path.join(tmp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
  for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
  fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});return {e,r,s:path.join(r,'proofs/extras')};}
function reject(name,entry,edit,location,kind='new refinement'){
  const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,`${name}: not a rejection`);
  assert.match(r.text,/expected[\s\S]*observed/,`${name}: missing semantic diagnostic`);
  assert.doesNotMatch(r.text,/a defined name|no such file|a decreasing self-call|more than once|RangeError|Maximum call stack|Segmentation fault/);
  assert.ok(location.test(r.text),`${name}: wrong failure location; diagnostic_sha256=${sha(r.text)}; tail=${r.text.slice(-500)}`);
  controls.push({name,rejected:true,kind,entry,diagnostic_sha256:sha(r.text),diagnostic_bytes:Buffer.byteLength(r.text),
    excerpt:r.text.slice(0,180)+'\n...\n'+r.text.slice(-650)});console.error('PASS rejection: '+name);}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
  manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
  for(const f of ['focused.js','verify.js','verify_native.js','probe.bend','unique_probe.bend'])graph.add(path.join(suite,f));
  const hashes=()=>Object.fromEntries([...graph].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes();clean(invoke(path.join(suite,'consumer.bend')));
  reject('include-first-extras-slot','Primitive.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.is_lt(q,256)','U32.is_le(q,256)'),/Location: routes/);
  reject('include-last-extras-slot','Primitive.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.is_le(512,q)','U32.is_le(511,q)'),/Location: routes/);
  reject('remove-query-bound','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'for budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool}\n  for qb: {U32.is_lt(q,131072) == True{} : Bool}',
    'for budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool}\n  for qb: {U32.is_le(q,131072) == True{} : Bool}'),/Location: LAWS\.bounded_extras_read/);
  reject('allow-overlong-square-range','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),'Nat.add(n,k),64n','Nat.add(n,k),65n'),/Location: LAWS\.bounded_extras_read/);
  reject('return-original-array','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    '(Tables.extras(n,U32.from_nat(k),a),S.value(Array.get(U64,a,q)))','(a,S.value(Array.get(U64,a,q)))'),/Location: LAWS\.bounded_extras_read/);
  reject('wrong-actual-first-write','Actual.bend',d=>{const f=path.join(d.r,'Tables.bend');const s=fs.readFileSync(f,'utf8');const i=s.indexOf('def extras(');
    fs.writeFileSync(f,s.slice(0,i)+s.slice(i).replace('U32.add(256, sq)','U32.add(0, sq)'));},/Location: \.\.\/storage\/Build\.extras\b/,'actual implementation dependency');
  reject('write-in-zero-count-case','Actual.bend',d=>{const f=path.join(d.r,'Tables.bend');const s=fs.readFileSync(f,'utf8');const i=s.indexOf('def extras(');
    fs.writeFileSync(f,s.slice(0,i)+s.slice(i).replace('case 0n: a','case 0n: Array.set(U64,a,0,U64.zero())'));},/Location: \.\.\/storage\/Build\.extras\b/,'actual implementation dependency');
  guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law table_pipeline_extras_read:','def omitted:'),d=>manifest(d.s),/./);
  guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.table_pipeline_extras_read(','def omitted('),d=>manifest(d.s),/./);
  guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
  guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Actual.bend'),'\n?hole\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Actual.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign import/);
  guard('symlinked-proof',d=>{const f=path.join(d.s,'Steps.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Steps.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
  const d=copy('unsafe-frame');replace(path.join(d.s,'Actual.bend'),'def pair(','@unsafe\ndef pair(');
  assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
  const unsafe=invoke(path.join(d.s,'PROOF.bend'));assert.equal(unsafe.status,0);assert.match(unsafe.text,/unsafe or foreign/);assert.throws(()=>clean(unsafe));
  controls.push({name:'unsafe-frame',rejected:true,kind:'policy and exact output',raw_cli_status:unsafe.status});
  assert.equal(controls.length,14);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const r={focused_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:names,new_law_count:2,inherited_gate_run:false,
    negative_controls:controls,consumer:'PASS',source_sha256s:before,
    scope:'Bounded actual extras preserves full read pair outside [256,512); prior actual table loop supplies complete shape. Stored-header values and final geometric lookup remain open.'};
  const text=JSON.stringify(r,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
