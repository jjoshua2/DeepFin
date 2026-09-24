// New header contracts only; verify.js retains the complete inherited gate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),controlsOnly=args.includes('--controls-only');
const filtered=args.filter(x=>x!=='--controls-only');
assert.ok(filtered.length===1||(filtered.length===3&&filtered[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler=path.resolve(filtered[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=import.meta.dirname,cli=path.join(compiler,'bend2/main.ts');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-header-laws-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const names=['own_block_header','stored_header_after_tables','stored_header_after_extras'],controls=[];
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(x=>x.split('#')[0]).join('\n');
function invoke(f){const r=spawnSync(process.execPath,[cli,f],{encoding:'utf8',timeout:1000000,maxBuffer:64<<20,
  env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};}
function clean(r){assert.equal(r.status,0,r.text.slice(0,2000));assert.equal(r.text,'All terms check.',r.text.slice(0,2000));}
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),names);
 const p=code(path.join(s,'PROOF.bend'));assert.deepEqual([...p.matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),names);
 assert.match(p,/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function policy(f,boundary,seen=new Set()){
 f=path.resolve(f);const rel=path.relative(boundary,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='bitboard_probe/Sliders.bend'),'escaped scope');
 assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlinked proof input');
 if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;
  assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');policy(path.resolve(path.dirname(f),m[1]),boundary,seen);}return seen;}
function replace(f,old,value){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(old).length,2,'nonunique mutation');fs.writeFileSync(f,s.replace(old,value));}
function copy(name){const e=path.join(tmp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});
 fs.mkdirSync(path.join(e,'bitboard_probe'));fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));return {e,r,s:path.join(r,'proofs/headers')};}
function reject(name,entry,edit,location,kind='new refinement'){
 const d=copy(name);edit(d);const started=performance.now(),r=invoke(path.join(d.s,entry));assert.equal(r.status,1,`${name}: not a rejection`);
 assert.match(r.text,/expected[\s\S]*observed/,`${name}: missing semantic diagnostic`);
 assert.doesNotMatch(r.text,/a defined name|no such file|a decreasing self-call|more than once|RangeError|Maximum call stack|Segmentation fault/);
 assert.ok(location.test(r.text),`${name}: wrong failure location; hash=${sha(r.text)}; tail=${r.text.slice(-700)}`);
 controls.push({name,rejected:true,kind,entry,diagnostic_sha256:sha(r.text),diagnostic_bytes:Buffer.byteLength(r.text),
  excerpt:r.text.slice(0,180)+'\n...\n'+r.text.slice(-650)});console.error(`PASS rejection: ${name} (${((performance.now()-started)/1000).toFixed(2)}s)`);}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 for(const f of ['focused.js','verify.js','verify_native.js','probe.bend'])graph.add(path.join(suite,f));
 const hashes=()=>Object.fromEntries([...graph].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
 const before=hashes();
 if(!controlsOnly){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: all three header laws and importing consumer');}
 reject('equal-keys-are-not-separated','Addresses.bend',d=>replace(path.join(d.s,'Addresses.bend'),
  'e: {Nat.is_lt(i,j) == True{} : Bool}', 'e: {Nat.is_le(i,j) == True{} : Bool}'),/Location: nat_ne\b/);
 reject('prefix-address-shifted-left','Addresses.bend',d=>replace(path.join(d.s,'Facts.bend'),
  'U32.add(128,key(k))','U32.add(127,key(k))'),/Location: facts\b/);
 reject('allow-key-128','Addresses.bend',d=>replace(path.join(d.s,'Addresses.bend'),
  'def facts(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool})','def facts(+k: Nat,valid: {Nat.is_lt(k,129n) == True{} : Bool})'),/Location: facts\b/);
 reject('header-prefix-value-off-by-one','Header.bend',d=>replace(path.join(d.s,'Header.bend'),
  '{S.value(Array.get(U64,headers(O.pack(c),k,at),F.offset(k))) == U64.from_u32(at) : U64}',
  '{S.value(Array.get(U64,headers(O.pack(c),k,at),F.offset(k))) == U64.from_u32(U32.inc(at)) : U64}'),/Location: prefix_value\b/,'new value contract');
 reject('header-mask-is-zero','Header.bend',d=>replace(path.join(d.s,'Header.bend'),
  '{S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))) == Layout.mask(F.key(k)) : U64}',
  '{S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))) == U64.zero() : U64}'),/Location: mask_value\b/,'new value contract');
 reject('actual-prefix-write-is-zero','OneBlock.bend',d=>replace(path.join(d.r,'Tables.bend'),
  'U64.from_u32(at))','U64.zero())'),/Location: .*Build\.tables\b/,'actual implementation dependency');
 reject('wrong-public-prefix-value','PROOF.bend',d=>replace(path.join(d.s,'Spec.bend'),
  'U64.from_u32(Prefix.prefix(k))','U64.from_u32(U32.inc(Prefix.prefix(k)))'),/Location: LAWS\.own_block_header\b/,'public contract');
 reject('return-original-array','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
  '(Tables.tables(Spec.count(before,after),Spec.key(k),P.prefix(k),a),Spec.expected(Nat.add(before,k),which))',
  '(a,Spec.expected(Nat.add(before,k),which))'),/Location: LAWS\.stored_header_after_tables\b/,'public contract');
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law stored_header_after_extras:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.stored_header_after_extras(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Stored.bend'),'\n?hole\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Stored.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign import/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Stored.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Stored.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 guard('unsafe-dependency',d=>replace(path.join(d.s,'Stored.bend'),'def actual(','@unsafe\ndef actual('),d=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
 assert.throws(()=>clean({status:0,text:'All terms check.\nWARNING: unsafe or foreign dependency'}));
 controls.push({name:'zero-exit-with-warning',rejected:true,kind:'exact-output wrapper unit control; not a new compiler execution'});
 assert.equal(controls.length,17);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const r={focused_gate:controlsOnly?'NOT_RUN':'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:names,new_law_count:3,
  promoted_prior_supplementary_laws:['own_block_header'],inherited_gate_run:false,negative_controls:controls,consumer:controlsOnly?'NOT_RUN':'PASS',source_sha256s:before,
  scope:'Actual stored mask/prefix header values through any bounded sequence of real blocks and final extras; independent mask geometry and final slider contents/lookup remain separate.'};
 const text=JSON.stringify(r,null,2)+'\n';if(filtered[2])fs.writeFileSync(filtered[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
