// Opt-in actual-affine-storage gate. The previous 56-law gate is unmodified.
// Usage: bun .../storage/verify.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler);
const root=path.resolve(import.meta.dirname,'../..'),engine=path.dirname(root);
const suite=path.join(root,'proofs/storage'),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-storage-laws-'));
const required=['read_preserves_storage','write_read_same_location','initialized_read',
  'write_preserves_shape','fill_preserves_shape','tables_preserve_shape','extras_preserve_shape','fill_one_read'];
const negatives=[],sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
function invoke(file,checker=cli,timeout=60000){
  const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout,maxBuffer:16<<20,
    env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,`execution error: ${r.error}`);assert.equal(r.signal,null,`checker signal: ${r.signal}`);
  return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
  file=path.resolve(file);const rel=path.relative(boundary,file);
  assert.ok(!rel.split(path.sep).includes('..'),'escaped proof boundary');
  assert.ok(rel.startsWith('standalone/'),'unapproved proof dependency');
  assert.ok(fs.lstatSync(file).isFile(),'nonregular proof input');
  assert.equal(fs.realpathSync(file),file,'symlinked proof path');
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
  fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});return {e,r,s:path.join(r,'proofs/storage')};}
function reject(name,entry,edit,base=false){
  const d=copy(name);let checker=cli;
  if(base){const c=path.join(d.e,'compiler');fs.mkdirSync(c);
    for(const f of ['bend.ts','comp.ts','base.bend','main.ts'])fs.copyFileSync(path.join(compiler,'bend2',f),path.join(c,f));
    d.base=path.join(c,'base.bend');checker=path.join(c,'main.ts');}
  edit(d);const r=invoke(path.join(d.s,entry),checker);
  assert.equal(r.status,1,`${name}: must reject\n${r.text}`);
  assert.match(r.text,/expected[\s\S]*observed/,`${name}: not an ordinary proof rejection\n${r.text}`);
  assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault/,'crash is not proof rejection');
  negatives.push({name,rejected:true,kind:'affected source refinement',entry,disposable_base_mutation:base});
  console.error('PASS rejection: '+name);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);
  negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
  manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
  const tracked=new Set([...graph,...['verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),
    path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
  const before=hashes(tracked);
  // Independent suites in one fail-closed aggregate, avoiding a redundant second
  // normalization of the expensive finite chess-domain proof in the consumer.
  const prior=spawnSync(process.execPath,[path.join(root,'proofs/address/verify.js'),compiler],
    {encoding:'utf8',timeout:1500000,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(prior.error,undefined,`${prior.error}`);assert.equal(prior.signal,null);
  assert.equal(prior.status,0,prior.stdout+prior.stderr);
  const parent=JSON.parse(prior.stdout);
  assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,56);
  assert.equal(parent.negative_controls.length,17);assert.equal(parent.inherited_negative_controls,55);
  assert.ok(parent.negative_controls.every(n=>n.rejected));
  console.error('PASS: unchanged 56-law parent, all 72 inherited controls');
  clean(invoke(path.join(suite,'consumer.bend')));
  console.error('PASS: eight actual-array laws and importing consumer');
  reject('set-is-noop','Roundtrip.bend',d=>replace(d.base,'Array.set.fin(T, Array.swap(T, a, i, v))','a'),true);
  reject('swap-leaf-retains-old-value','Roundtrip.bend',d=>replace(d.base,'(ALeaf{v}, x)','(ALeaf{x}, v)'),true);
  reject('read-duplicates-returned-leaf','Read.bend',d=>replace(d.base,'(ALeaf{x}, x)','(ANode{ALeaf{x}, ALeaf{x}}, x)'),true);
  reject('swap-discards-sibling','Write.bend',d=>replace(d.base,'(ANode{nxs, ys}, old)','(nxs, old)'),true);
  reject('read-selects-wrong-child','Refinement.bend',d=>replace(d.base,
    'Array.swap.lo(T, ys, Array.get.go(T, xs, h, i))','Array.swap.hi(T, xs, Array.get.go(T, ys, h, i))'),true);
  reject('fill-writes-wrong-value','Fill.bend',d=>replace(path.join(d.r,'Tables.bend'),
    'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))','a = Array.set(U64, a, at, U64.zero())'));
  reject('fill-writes-next-address','Fill.bend',d=>replace(path.join(d.r,'Tables.bend'),
    'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))','a = Array.set(U64, a, U32.inc(at), slider(sq, bishop, subset, False{}))'));
  for(const [fn,start] of [['fill','def fill('],['tables','def tables('],['extras','def extras(']]){
    reject(fn+'-base-case-destroys-storage','Build.bend',d=>{
      const f=path.join(d.r,'Tables.bend'),s=fs.readFileSync(f,'utf8'),pos=s.indexOf(start);
      assert.ok(pos>=0);const tail=s.slice(pos);assert.ok(tail.includes('case 0n: a'));
      fs.writeFileSync(f,s.slice(0,pos)+tail.replace('case 0n: a','case 0n: ALeaf{U64.zero()}'));});
  }
  guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law fill_one_read:','def omitted:'),d=>manifest(d.s),/./);
  guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.fill_one_read(','def omitted('),d=>manifest(d.s),/./);
  guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
  guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
  guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Read.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
  guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Read.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign or unexpected/);
  guard('symlinked-proof',d=>{const f=path.join(d.s,'Read.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Read.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
  const d=copy('unsafe-representation');replace(path.join(d.s,'Representation.bend'),'def reify(','@unsafe\ndef reify(');
  assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
  const warned=invoke(path.join(d.s,'PROOF.bend'));assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
  negatives.push({name:'unsafe-representation',rejected:true,kind:'policy and exact checker-output guard',raw_cli_status:warned.status});
  assert.equal(negatives.length,18);
  assert.deepEqual(hashes(tracked),before,'proof inputs changed');assert.deepEqual(verifyCompiler(compiler),identity);
  const report={proof_gate:'PASS',compiler_revision:PIN.revision,...identity,new_universal_laws:required,new_law_count:8,
    inherited_law_count:56,aggregate_law_count:64,inherited_negative_controls:72,negative_controls:negatives,
    consumer:'PASS',source_sha256s:before,
    method:'Structural reification of arbitrary affine arrays; actual Base APIs and Tables recursion, no new axioms',
    scope:'Read storage identity, same-location write/read, initialized value, exact tree-shape preservation and one actual fill write; NOT prefix bounds, nonaliasing, complete table contents, ray geometry or native lifetime'};
  const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
