// New stored-data contracts only; verify.js retains the complete inherited gate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync,spawn} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),controlsOnly=args.includes('--controls-only');
const filtered=args.filter(x=>x!=='--controls-only');
assert.ok(filtered.length===1||(filtered.length===3&&filtered[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler=path.resolve(filtered[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=import.meta.dirname,cli=path.join(compiler,'bend2/main.ts');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-data-laws-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const names=['later_tables_preserve_data','stored_data_after_tables','stored_data_after_extras'],controls=[];
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
 fs.mkdirSync(path.join(e,'bitboard_probe'));fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));return {e,r,s:path.join(r,'proofs/data')};}
const jobs=[];
function invokeAsync(f){return new Promise((resolve,reject)=>{
 const child=spawn(process.execPath,[cli,f],{env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 let out='',err='',size=0,timeout=false;const timer=setTimeout(()=>{timeout=true;child.kill('SIGKILL');},1000000);
 function append(which,b){size+=b.length;if(size>64*1024*1024){child.kill('SIGKILL');return;}if(which===0)out+=b.toString();else err+=b.toString();}
 child.stdout.on('data',b=>append(0,b));child.stderr.on('data',b=>append(1,b));
 child.on('error',e=>{clearTimeout(timer);reject(e);});
 child.on('close',(status,signal)=>{clearTimeout(timer);if(timeout||signal||size>64*1024*1024)reject(new Error(`checker terminated: timeout=${timeout}, signal=${signal}, bytes=${size}`));
  else resolve({status,text:(out+err).trim()});});
 });}
function reject(name,entry,edit,location,kind='new refinement'){
 jobs.push(async()=>{const d=copy(name);edit(d);const started=performance.now(),r=await invokeAsync(path.join(d.s,entry));
 assert.equal(r.status,1,`${name}: not a rejection`);
 assert.match(r.text,/expected[\s\S]*observed/,`${name}: missing semantic diagnostic`);
 assert.doesNotMatch(r.text,/a defined name|no such file|a decreasing self-call|more than once|RangeError|Maximum call stack|Segmentation fault/);
 assert.ok(location.test(r.text),`${name}: wrong failure location; hash=${sha(r.text)}; tail=${r.text.slice(-700)}`);
 console.error(`PASS rejection: ${name} (${((performance.now()-started)/1000).toFixed(2)}s)`);
 return {name,rejected:true,kind,entry,diagnostic_sha256:sha(r.text),diagnostic_bytes:Buffer.byteLength(r.text),
  excerpt:r.text.slice(0,180)+'\n...\n'+r.text.slice(-650)};});
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 for(const f of ['focused.js','verify.js','verify_native.js'])graph.add(path.join(suite,f));
 const hashes=()=>Object.fromEntries([...graph].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
 const before=hashes();
 if(!controlsOnly){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: all three stored-data laws and importing consumer');}
 reject('metadata-query-can-overlap','Preserve.bend',d=>{
   const f=path.join(d.s,'Preserve.bend'),s=fs.readFileSync(f,'utf8');
   fs.writeFileSync(f,s.replace('+lower: {U32.is_le(512,q) == True{} : Bool}', '+lower: {U32.is_le(128,q) == True{} : Bool}'));
 },/Location: headers\b/);
 reject('admit-block-start-as-earlier','Preserve.bend',d=>replace(path.join(d.s,'Preserve.bend'),
   'before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool}) ->\n  {S.value(Array.get(U64,P.block',
   'before: {U32.is_le(q,P.prefix(k)) == True{} : Bool}) ->\n  {S.value(Array.get(U64,P.block'),/Location: block\b/);
 reject('wrong-successor-bound','Index.bend',d=>replace(path.join(d.s,'Index.bend'),
   'e: {Nat.is_le(1n+n,b) == True{} : Bool}', 'e: {Nat.is_le(n,b) == True{} : Bool}'),/Location: predecessor\b/);
 reject('computed-entry-is-zero','OneBlock.bend',d=>replace(path.join(d.s,'OneBlock.bend'),
   '(P.block(O.pack(c),k),S.expected(k,i))','(P.block(O.pack(c),k),U64.zero())'),/Location: model\b/,'new value contract');
 reject('shifted-data-query','OneBlock.bend',d=>replace(path.join(d.s,'Spec.bend'),
   'U32.add(P.prefix(k),U32.from_nat(i))', 'U32.inc(U32.add(P.prefix(k),U32.from_nat(i)))'),/Location: model\b/,'actual fill observation');
 reject('return-original-array','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
   '(Spec.table(a,before,after,k),Spec.expected(Nat.add(before,k),i))',
   '(a,Spec.expected(Nat.add(before,k),i))'),/Location: LAWS\.stored_data_after_tables\b/,'public contract');
 reject('include-exclusive-interior-end','PROOF.bend',d=>{
   const f=path.join(d.s,'LAWS.bend'),s=fs.readFileSync(f,'utf8');
   fs.writeFileSync(f,s.replace('for ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k)))))',
     'for ib: {Nat.is_le(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k)))))'));
 },/Location: LAWS\.stored_data_after_tables\b/,'public contract');
 reject('actual-later-metadata-corrupts-data','Preserve.bend',d=>replace(path.join(d.r,'Tables.bend'),
   '      tables(p, U32.inc(key), U32.add(at, size),',
   '      a = Array.set(U64,a,512,U64.zero())\n      tables(p, U32.inc(key), U32.add(at, size),'),/Location: .*Build\.tables\b/,'actual implementation dependency');
 // At most two disposable checker processes; deterministic result order.
 for(let i=0;i<jobs.length;i+=2)controls.push(...await Promise.all(jobs.slice(i,i+2).map(f=>f())));
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law stored_data_after_extras:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.stored_data_after_extras(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF\.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Stored.bend'),'\n?hole\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Stored.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign import/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Stored.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Stored.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 guard('unsafe-dependency',d=>replace(path.join(d.s,'Stored.bend'),'def actual(','@unsafe\ndef actual('),d=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
 assert.throws(()=>clean({status:0,text:'All terms check.\nWARNING: unsafe or foreign dependency'}));
 controls.push({name:'zero-exit-with-warning',rejected:true,kind:'exact-output wrapper unit control; not a new compiler execution'});
 assert.equal(controls.length,17);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const r={focused_gate:controlsOnly?'NOT_RUN':'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:names,new_law_count:3,
  inherited_gate_run:false,negative_controls:controls,consumer:controlsOnly?'NOT_RUN':'PASS',source_sha256s:before,
  scope:'Computed entries survive later metadata, complete table blocks and extras; values refer to actual Tables.slider, not independent source ray geometry or the final Chess lookup.'};
 const text=JSON.stringify(r,null,2)+'\n';if(filtered[2])fs.writeFileSync(filtered[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
