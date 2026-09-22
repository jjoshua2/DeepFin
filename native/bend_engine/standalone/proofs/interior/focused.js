// Opt-in focused development gate. verify.js also checks every inherited law/control.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const raw=process.argv.slice(2),controlsOnly=raw.includes('--controls-only');
const args=raw.filter(x=>x!=='--controls-only');
assert.ok(raw.filter(x=>x==='--controls-only').length<=1,'duplicate controls-only flag');
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE] [--controls-only]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=import.meta.dirname,cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-interior-proofs-'));
const required=['allocation_mask_exact','allocation_mask_injective','prefix_interior_certificate',
  'ordered_normalized_addresses_distinct','interior_read_uses_unmasked_address'];
const sha=x=>createHash('sha256').update(x).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
const negatives=[];
function invoke(file,checker=cli,timeout=900000){
 const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`checker failed: ${r.error}`);assert.equal(r.signal,null);
 return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
 file=path.resolve(file);const rel=path.relative(boundary,file);
 assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='bitboard_probe/Sliders.bend'),'escaped proof boundary');
 assert.ok(fs.lstatSync(file).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(file),file,'symlinked proof input');
 if(seen.has(file))return seen;seen.add(file);const text=code(file);assert.doesNotMatch(text,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of text.matchAll(/^\s*import\s+(\S+)/gm)){
  if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');
  policy(path.resolve(path.dirname(file),m[1]),boundary,seen);
 }return seen;
}
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/prefix\/PROOF\.bend as PrefixProof/);
 assert.match(code(path.join(s,'../prefix/Certified.bend')),/import \.\.\/layout\/PROOF\.bend as LayoutProof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(file,a,b){const text=fs.readFileSync(file,'utf8');assert.equal(text.split(a).length,2,`nonunique anchor ${a}`);fs.writeFileSync(file,text.replace(a,b));}
function copy(name,base=false){
 const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});fs.mkdirSync(path.join(e,'bitboard_probe'));
 fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
 const d={e,r,s:path.join(r,'proofs/interior'),checker:cli};
 if(base){const c=path.join(e,'compiler');fs.mkdirSync(c);for(const f of ['base.bend','bend.ts','comp.ts','main.ts'])fs.copyFileSync(path.join(compiler,'bend2',f),path.join(c,f));d.base=path.join(c,'base.bend');d.checker=path.join(c,'main.ts');}
 return d;
}
function semantic(name,entry,edit,base=false){
 const d=copy(name,base);edit(d);const r=invoke(path.join(d.s,entry),d.checker,120000);
 assert.equal(r.status,1,`${name}: expected rejection\n${r.text}`);
 assert.match(r.text,/expected[\s\S]*observed/,`${name}: not ordinary proof rejection\n${r.text}`);
 assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault|no such file/,'crashes/missing files are not semantic rejection');
 negatives.push({name,rejected:true,kind:'affected source refinement',entry});console.error('PASS rejection: '+name);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
 const before=hashes(tracked);
 if(!controlsOnly){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: five public interior-address laws and importing consumer');}
 else console.error('CONTROL DEVELOPMENT ONLY: successful source/consumer check not run');
 semantic('normalization-is-zero','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.and(x,U32.sub(131072,1))','0'));
 semantic('mask-loses-bit-16','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.and(x,U32.sub(131072,1))','U32.and(x,65535)'));
 semantic('normalization-increments-input','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.and(x,U32.sub(131072,1))','U32.and(U32.inc(x),131071)'));
 semantic('inclusive-allocation-bound','Mask.bend',d=>replace(path.join(d.s,'Mask.bend'),'U32.is_lt(x,131072)','U32.is_le(x,131072)'));
 semantic('allocation-bound-one-too-wide','Mask.bend',d=>replace(path.join(d.s,'Mask.bend'),'U32.is_lt(x,131072)','U32.is_lt(x,131073)'));
 semantic('inclusive-block-end','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.is_lt(x,finish(k))','U32.is_le(x,finish(k))'));
 semantic('missing-block-lower-bound','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.is_le(Prefix.prefix(k),x)','U32.is_le(512,Prefix.prefix(k))'));
 semantic('actual-get-forces-zero','Read.bend',d=>replace(d.base,'Array.get.go(T, a, n, U32.and(i, U32.sub(n, 1)))','Array.get.go(T, a, n, 0)'),true);
 semantic('actual-get-masks-wrong-width','Read.bend',d=>replace(d.base,'Array.get.go(T, a, n, U32.and(i, U32.sub(n, 1)))','Array.get.go(T, a, n, U32.and(i, U32.sub(n, 2)))'),true);
 semantic('wrong-direct-read-address','Read.bend',d=>replace(path.join(d.s,'Read.bend'),
  '{Array.get(U64,O.pack(c),x) == Array.get.go(U64,O.pack(c),131072,x) : Array<U64> & U64}:',
  '{Array.get(U64,O.pack(c),x) == Array.get.go(U64,O.pack(c),131072,U32.inc(x)) : Array<U64> & U64}:'));
 semantic('missing-actual-size-certificate','Read.bend',d=>replace(path.join(d.s,'Read.bend'),
  'size: {O.count(Array.size(U64,O.pack(c))) == 131072 : U32}',
  'size: {131072 == 131072 : U32}'));
 semantic('falsely-excludes-first-block-addresses','Interval.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.is_le(512,x)','U32.is_le(1024,x)'));
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law prefix_interior_certificate:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.prefix_interior_certificate(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('missing-prefix-certificate-producer',d=>replace(path.join(d.s,'Certified.bend'),'import ../prefix/PROOF.bend as PrefixProof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Mask.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Mask.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Mask.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Mask.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 const d=copy('unsafe-mask');replace(path.join(d.s,'Mask.bend'),'def allocation(','@unsafe\ndef allocation(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe/);
 const warned=invoke(path.join(d.s,'Mask.bend'),cli,120000);assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 negatives.push({name:'unsafe-mask',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(negatives.length,21);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={focused_gate:controlsOnly?'NOT RUN':'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:required,new_law_count:5,inherited_gate_run:false,
  negative_controls:negatives,consumer:controlsOnly?'NOT RUN':'PASS',source_sha256s:before,
  scope:'Absolute interior-address bounds and allocation-mask injectivity; actual complete read-pair normalization under explicit source size. Not relative-offset arithmetic, arbitrary-array path separation or final geometry.'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
