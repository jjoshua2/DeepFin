// Four new frame contracts only. Use verify.js for the mandatory inherited aggregate.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler);
const root=path.resolve(import.meta.dirname,'../..'),engine=path.dirname(root);
const suite=path.join(root,'proofs/separation'),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-separation-'));
const required=['write_preserves_other_location','write_preserves_separation',
  'fill_preserves_unwritten_location','fill_preserves_separation'];
const negatives=[],sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
function invoke(file,checker=cli){
  const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout:60000,maxBuffer:16<<20,
    env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,`checker error: ${r.error}`);assert.equal(r.signal,null);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
  file=path.resolve(file);const rel=path.relative(boundary,file);
  assert.ok(!rel.split(path.sep).includes('..')&&rel.startsWith('standalone/'),'escaped proof boundary');
  assert.ok(fs.lstatSync(file).isFile(),'nonregular proof input');
  assert.equal(fs.realpathSync(file),file,'symlinked proof input');
  if(seen.has(file))return seen;seen.add(file);
  const s=code(file);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
  for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){
    if(m[1]==='Base')continue;
    assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign or unexpected import');
    policy(path.resolve(path.dirname(file),m[1]),boundary,seen);
  }return seen;
}
function manifest(s){
  assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
  const proof=code(path.join(s,'PROOF.bend'));
  assert.deepEqual([...proof.matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
  assert.match(proof,/import \.\/LAWS\.bend as Laws/);
  assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,b,a){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(b).length,2,`nonunique mutation ${b}`);fs.writeFileSync(f,s.replace(b,a));}
function copy(name){const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
  for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
  fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});return {e,r,s:path.join(r,'proofs/separation')};}
function reject(name,entry,edit){
  const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry));
  assert.equal(r.status,1,`${name}: must reject\n${r.text}`);
  assert.match(r.text,/expected[\s\S]*observed/,`${name}: not a semantic rejection\n${r.text}`);
  assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault/,'crash is not rejection');
  negatives.push({name,rejected:true,kind:'source contract/refinement',entry});
  console.error('PASS rejection: '+name);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);
  negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
  manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
  const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js'].map(f=>path.join(suite,f)),
    path.join(root,'proofs/storage/probe.bend'),path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
  const before=hashes(tracked);
  clean(invoke(path.join(suite,'consumer.bend')));
  reject('leaf-alias-declared-separated','Model.bend',d=>replace(path.join(d.s,'Spec.bend'),'case O.Leaf{}: False{}','case O.Leaf{}: True{}'));
  reject('left-overlap-declared-separated','Model.bend',d=>replace(path.join(d.s,'Spec.bend'),'case True{} True{}: l','case True{} True{}: True{}'));
  reject('right-overlap-declared-separated','Model.bend',d=>replace(path.join(d.s,'Spec.bend'),'case False{} False{}: r','case False{} False{}: True{}'));
  reject('omit-index-normalization','Actual.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'apart(t,n,U32.and(i,U32.sub(n,1)),U32.and(j,U32.sub(n,1)))','apart(t,n,i,j)'));
  reject('numeric-inequality-is-not-separation','Actual.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'apart(t,n,U32.and(i,U32.sub(n,1)),U32.and(j,U32.sub(n,1)))','Bool.not(U32.is_eq(i,j))'));
  reject('clear-omits-first-write','Fill.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'Bool.and(disjoint(t,at,j),clear(t,p,U32.inc(at),j))','clear(t,p,U32.inc(at),j)'));
  reject('clear-repeats-address','Fill.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'clear(t,p,U32.inc(at),j)','clear(t,p,at,j)'));
  reject('actual-fill-writes-next-address','Fill.bend',d=>replace(path.join(d.r,'Tables.bend'),
    'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))',
    'a = Array.set(U64, a, U32.inc(at), slider(sq, bishop, subset, False{}))'));
  reject('actual-fill-zero-case-writes','Fill.bend',d=>{
    const f=path.join(d.r,'Tables.bend'),s=fs.readFileSync(f,'utf8'),pos=s.indexOf('def fill(');
    assert.ok(pos>=0&&s.slice(pos).includes('case 0n: a'));
    fs.writeFileSync(f,s.slice(0,pos)+s.slice(pos).replace('case 0n: a','case 0n: Array.set(U64,a,0,U64.zero())'));
  });
  guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law fill_preserves_separation:','def omitted:'),d=>manifest(d.s),/./);
  guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.fill_preserves_separation(','def omitted('),d=>manifest(d.s),/./);
  guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
  guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
  guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Model.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Model.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign or unexpected/);
  guard('symlinked-proof',d=>{const f=path.join(d.s,'Model.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Model.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
  const d=copy('unsafe-frame');replace(path.join(d.s,'Actual.bend'),'def frame(','@unsafe\ndef frame(');
  assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
  const warned=invoke(path.join(d.s,'PROOF.bend'));assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
  negatives.push({name:'unsafe-frame',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
  assert.equal(negatives.length,17);
  assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:required,new_law_count:4,
    inherited_gate_run:false,negative_controls:negatives,consumer:'PASS',source_sha256s:before,
    scope:'Normalized-path frame laws for actual Array.set/get and Tables.fill; not numeric prefix bounds or final geometric lookup refinement'};
  const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
