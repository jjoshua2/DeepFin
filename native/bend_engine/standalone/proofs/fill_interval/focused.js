// Numeric-budget and actual fill-loop contracts. verify.js retains the full parent.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),controlsOnly=args.includes('--controls-only');
const cleanArgs=args.filter(x=>x!=='--controls-only');
assert.ok(cleanArgs.length===1||(cleanArgs.length===3&&cleanArgs[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
assert.equal(args.filter(x=>x==='--controls-only').length,controlsOnly?1:0);
const compiler=path.resolve(cleanArgs[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const root=path.resolve(suite,'../..'),engine=path.dirname(root),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fill-proofs-'));
const required=['bounded_clear_before','bounded_clear_after','bounded_fill_preserves_before','bounded_fill_preserves_after','block_fill_preserves_before','block_fill_preserves_after'];
const sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
const negatives=[];
function invoke(file,checker=cli,timeout=900000){
 const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
 file=path.resolve(file);const rel=path.relative(boundary,file);
 assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='bitboard_probe/Sliders.bend'),'escaped proof boundary');
 assert.ok(fs.lstatSync(file).isFile(),'nonregular proof');assert.equal(fs.realpathSync(file),file,'symlinked proof');
 if(seen.has(file))return seen;seen.add(file);const s=code(file);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){
  if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign or unexpected proof import');
  policy(path.resolve(path.dirname(file),m[1]),boundary,seen);
 }return seen;
}
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/prefix\/PROOF\.bend as PrefixProof/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,a,b){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(a).length,2,`nonunique mutation ${a}`);fs.writeFileSync(f,s.replace(a,b));}
function copy(name,withCompiler=false){
 const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});fs.mkdirSync(path.join(e,'bitboard_probe'));
 fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
 const d={e,r,s:path.join(r,'proofs/fill_interval'),checker:cli};
 if(withCompiler){const c=path.join(e,'compiler');fs.mkdirSync(c);for(const f of ['bend.ts','comp.ts','base.bend','main.ts'])fs.copyFileSync(path.join(compiler,'bend2',f),path.join(c,f));d.checker=path.join(c,'main.ts');d.base=path.join(c,'base.bend');}
 return d;
}
function semantic(name,entry,edit,withCompiler=false,expectedLocation=null){
 const d=copy(name,withCompiler);edit(d);const r=invoke(path.join(d.s,entry),d.checker,60000);
 assert.equal(r.status,1,`${name}: must reject\n${r.text}`);assert.match(r.text,/expected[\s\S]*observed/,`${name}: not ordinary semantic rejection\n${r.text}`);
 assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault|cannot infer|an annotated term|parse error|no such file|unfilled law|unbound variable|consumed twice/i,'invalid program or crash is not semantic rejection');
 const location=r.text.match(/^Location:\s*(.*)$/m)?.[1];assert.ok(location,'missing rejection location');
 if(expectedLocation)assert.match(location,expectedLocation,'wrong rejection layer');
 const diagnostic=r.text.replaceAll(d.e,'<disposable-source>');
 negatives.push({name,rejected:true,kind:'affected source refinement',entry,failure_location:location,diagnostic_sha256:sha(diagnostic)});console.error('PASS rejection '+name);
}
function guard(name,edit,check){const d=copy(name);edit(d);assert.throws(()=>check(d));negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),path.join(root,'verify_compiler.js'),path.join(root,'toolchain.json')]);
 const before=hashes(tracked);
 if(!controlsOnly){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS six fill interval laws and importing consumer');}
 semantic('increment-observation-stuck','Arithmetic.bend',d=>replace(path.join(d.s,'Arithmetic.bend'),
  '{U32.to_nat(U32.inc(a)) == 1n+U32.to_nat(a) : Nat}',
  '{U32.to_nat(a) == 1n+U32.to_nat(a) : Nat}'),false,/inc_value/);
 semantic('positive-budget-omits-first-write','Naturals.bend',d=>{
  const f=path.join(d.s,'Naturals.bend'),text=fs.readFileSync(f,'utf8');
  const first=text.indexOf('def current('),last=text.indexOf('def tail_sum(');
  const part=text.slice(first,last);assert.equal(part.split('Nat.add(1n+n,U32.to_nat(at))').length,2);
  fs.writeFileSync(f,text.slice(0,first)+part.replace('Nat.add(1n+n,U32.to_nat(at))','Nat.add(n,U32.to_nat(at))')+text.slice(last));
 },false,/current/);
 semantic('nat-sum-one-ahead','Sum.bend',d=>replace(path.join(d.s,'Sum.bend'),
  '{U32.to_nat(U32.add(a,b)) == Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}',
  '{U32.to_nat(U32.add(a,b)) == 1n+Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}'),false,/u32/);
 semantic('block-budget-one-extra','Block.bend',d=>replace(path.join(d.s,'Block.bend'),
  '{Nat.add(U32.to_nat(size),U32.to_nat(start)) == U32.to_nat(end) : Nat}',
  '{Nat.add(1n+U32.to_nat(size),U32.to_nat(start)) == U32.to_nat(end) : Nat}'),false,/count/);
 semantic('inclusive-before-query','Clear.bend',d=>replace(path.join(d.s,'Clear.bend'),
  '+qb: {U32.is_lt(q,at) == True{} : Bool}', '+qb: {U32.is_le(q,at) == True{} : Bool}'),false,/before/);
 semantic('inclusive-after-allocation-bound','Clear.bend',d=>replace(path.join(d.s,'Clear.bend'),
  '+qb: {U32.is_lt(q,131072) == True{} : Bool}', '+qb: {U32.is_le(q,131072) == True{} : Bool}'),false,/after/);
 semantic('clear-omits-first-address','Clear.bend',d=>replace(path.join(d.r,'proofs/separation/Spec.bend'),
  'Bool.and(disjoint(t,at,j),clear(t,p,U32.inc(at),j))','clear(t,p,U32.inc(at),j)'),false,/before|after/);
 semantic('clear-repeats-write-address','Clear.bend',d=>replace(path.join(d.r,'proofs/separation/Spec.bend'),
  'clear(t,p,U32.inc(at),j)','clear(t,p,at,j)'),false,/before|after/);
 // These actual implementation mutations are rejected by the imported original
 // fill refinement; do not claim that their first failing location is new code.
 semantic('actual-fill-writes-next-address','Actual.bend',d=>replace(path.join(d.r,'Tables.bend'),
  'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))',
  'a = Array.set(U64, a, U32.inc(at), slider(sq, bishop, subset, False{}))'),false,/model|frame|fill/);
 semantic('actual-zero-count-fill-writes','Actual.bend',d=>{
  const f=path.join(d.r,'Tables.bend'),text=fs.readFileSync(f,'utf8'),pos=text.indexOf('def fill(');
  assert.ok(pos>=0&&text.slice(pos).includes('case 0n: a'));
  fs.writeFileSync(f,text.slice(0,pos)+text.slice(pos).replace('case 0n: a','case 0n: Array.set(U64,a,0,U64.zero())'));
 },false,/model|frame|fill/);
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law block_fill_preserves_after:','def omitted:'),d=>manifest(d.s));
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.block_fill_preserves_after(','def omitted('),d=>manifest(d.s));
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
 guard('missing-prefix-certificate-producer',d=>replace(path.join(d.s,'Certified.bend'),'import ../prefix/PROOF.bend as PrefixProof\n',''),d=>manifest(d.s));
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Clear.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e));
 guard('foreign-witness',d=>fs.appendFileSync(path.join(d.s,'Clear.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e));
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Clear.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Clear.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e));
 const d=copy('unsafe-frame');replace(path.join(d.s,'Actual.bend'),'def frame_before(','@unsafe\ndef frame_before(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e));const warned=invoke(path.join(d.s,'Actual.bend'),cli,60000);
 assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 negatives.push({name:'unsafe-frame',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(negatives.length,19);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={...(controlsOnly?{control_gate:'PASS'}:{focused_gate:'PASS'}),compiler_revision:PIN.revision,...identity,
  new_laws:required,new_law_count:6,inherited_gate_run:false,consumer:controlsOnly?'NOT RUN':'PASS',
  negative_controls:negatives,source_sha256s:before,
  scope:'Numeric Nat budgets derive actual incrementing fill-clear conditions; actual full blocks preserve outside queries. Not stored metadata or final inside-block contents/geometry.'};
 const text=JSON.stringify(report,null,2)+'\n';if(cleanArgs[2])fs.writeFileSync(cleanArgs[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
