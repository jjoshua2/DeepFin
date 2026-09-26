// Exact-source producer/proof gate. Negative controls never accept parser/affine failures.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-safety-proofs-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),text=p=>fs.readFileSync(p,'utf8');
const code=p=>text(p).split('\n').map(s=>s.split('#')[0]).join('\n');
const names=['guarded_castle_requires_full_check','generated_castle_requires_full_check','generated_castle_fast_step_equals_checked_step','guarded_castling_blocker_filter_equals_full_filter'];
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,e,seen=new Set()){f=path.resolve(f);const rel=path.relative(e,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'));assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlink proof');if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or hole');for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign dependency');graph(path.resolve(path.dirname(f),m[1]),e,seen);}return seen;}
let invocation=0;
function invoke(f){const log=path.join(tmp,'invocation-'+(invocation++)+'.log');const fd=fs.openSync(log,'w');const start=performance.now();let r;
  try{r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{stdio:['ignore',fd,fd],timeout:90000,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});}finally{fs.closeSync(fd);}
  const raw=text(log);assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,raw,seconds:(performance.now()-start)/1000};}
function safe(r){assert.equal(r.status,0,r.raw.slice(-3000));assert.equal(r.raw.trim(),'All terms check.');}
function mutation(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/castle_safety'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(p,a,b){const s=text(p);assert.equal(s.split(a).length,2,'nonunique mutation site: '+p);fs.writeFileSync(p,s.replace(a,b));}
const controls=[];
function reject(name,entry,edit,location){const d=mutation(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': expected failure');assert.match(r.raw,/expected[\s\S]*observed/,name+': ordinary refinement failure required');assert.doesNotMatch(r.raw,/consumed more than once|no such file|a defined name|a pattern \(a binder|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault/i);assert.match(r.raw,location,name+': wrong source location');controls.push({name,kind:'source semantic/refinement',rejected:true,entry,location:r.raw.match(/Location: ([^\n]+)/)?.[1],diagnostic_sha256:sha(r.raw),diagnostic_excerpt:r.raw.slice(-2000)});console.error('PASS control '+name);}
function policy(name,edit,test){const d=mutation(name);edit(d);assert.throws(()=>test(d));controls.push({name,kind:'manifest/import policy',rejected:true});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);graph(path.join(suite,'probe.bend'),engine,closure);
  for(const f of ['focused.js','verify_native.js'])closure.add(path.join(suite,f));for(const f of ['toolchain.json','verify_compiler.js'])closure.add(path.resolve(suite,'../..',f));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=invoke(path.join(suite,'consumer.bend'));safe(consumer);
  reject('actual-exempts-castling-source','Required.bend',d=>{
    replace(d.chess,'Bool.or(U32.is_eq(flag, 1), U64.test_bit(sensitive, U32.to_nat(src)))','Bool.or(U32.is_eq(flag, 1), Bool.and(Bool.not(U32.is_eq(flag, 2)), U64.test_bit(sensitive, U32.to_nat(src))))');
    replace(d.chess,'Ply{src, dst, promotion, flag} = m','Ply{src, dst, promotion, +flag} = m');
  },/Location: (?:Required\.)?guard\b/);
  reject('actual-blocker-mask-omits-kings','Filter.bend',d=>replace(d.chess,'sensitive = U64.and(own, U64.or(rays, get_kings(b)))','sensitive = U64.and(own, rays)'),/Location: (?:Filter\.)?blockers\b/);
  reject('actual-fast-step-bypasses-check','Filter.bend',d=>replace(d.chess,'case True{}: filter_step(b, m, r)','case True{}:\n      (table, acc) = r\n      (table, Con{m, acc})'),/Location: (?:Filter\.)?step\b/);
  reject('bit-cover-drops-owned-king','Cover.bend',d=>replace(path.join(d.s,'Cover.bend'),'Bool.and(own,Bool.or(ray,king)) == True{}','Bool.and(own,ray) == True{}'),/Location: (?:Cover\.)?head\b/);
  reject('false-empty-source-boundary','Boundaries.bend',d=>replace(path.join(d.s,'Boundaries.bend'),
    '{Chess.filter_requires(S.sensitive(Position.empty(),U64.zero()),Chess.Ply{4,6,0,2}) == False{} : Bool}',
    '{Chess.filter_requires(S.sensitive(Position.empty(),U64.zero()),Chess.Ply{4,6,0,2}) == True{} : Bool}'),/Location: empty_source\b/);
  reject('false-owned-king-boundary','Boundaries.bend',d=>replace(path.join(d.s,'Boundaries.bend'),
    '{Chess.filter_requires(S.sensitive(Position.put(5,True{},4,Position.empty()),U64.zero()),Chess.Ply{4,6,0,2}) == True{} : Bool}',
    '{Chess.filter_requires(S.sensitive(Position.put(5,True{},4,Position.empty()),U64.zero()),Chess.Ply{4,6,0,2}) == False{} : Bool}'),/Location: king_source\b/);
  policy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law guarded_castle_requires_full_check:','def missing:'),d=>manifest(d.s));
  policy('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.guarded_castle_requires_full_check(','def missing('),d=>manifest(d.s));
  policy('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  policy('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  policy('hole',d=>fs.appendFileSync(path.join(d.s,'Cover.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('foreign',d=>fs.appendFileSync(path.join(d.s,'Cover.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('unsafe',d=>fs.appendFileSync(path.join(d.s,'Cover.bend'),'\n@unsafe\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('symlink',d=>{const p=path.join(d.s,'Cover.bend');fs.renameSync(p,p+'.original');fs.symlinkSync('Cover.bend.original',p);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>safe({status:0,raw:'All terms check.\nWARNING: unsafe dependency'}));controls.push({name:'warning-on-zero-exit',kind:'synthetic output-wrapper unit; not compiler execution',rejected:true});
  assert.equal(controls.length,15);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',consumer:'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:4,new_laws:names,consumer_seconds:consumer.seconds,consumer_raw:consumer.raw,negative_controls:controls,source_sha256s:before,inherited_gate_run:false,scope:'Generated castle source is always sensitive, independently of ray values. Actual fast-step complete pair equals full checked step; actual filter_blockers equals full filter on guarded-castle-only lists. No semantic attack-correctness or unrestricted full-generator legality theorem.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}

// All checks and synchronous cleanup have completed; do not retain runtime handles.
process.exit(0);
