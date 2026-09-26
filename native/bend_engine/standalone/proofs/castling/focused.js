// Opt-in public castling source laws and rejection controls.
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
const engine=path.resolve(suite,'../../..'),temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-castling-laws-'));
const names=['castling_make_move_exact_update','castling_make_move_preserves_representation'];
const sha=x=>createHash('sha256').update(x).digest('hex'),text=p=>fs.readFileSync(p,'utf8');
const code=p=>text(p).split('\n').map(l=>l.split('#')[0]).join('\n'),controls=[];
function graph(f,boundary,seen=new Set()){
  f=path.resolve(f);const rel=path.relative(boundary,f);
  assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'),'escaped scope');
  assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlinked proof input');
  if(seen.has(f))return seen;seen.add(f);const source=code(f);assert.doesNotMatch(source,/@unsafe|\?/,'unsafe dependency or proof hole');
  for(const m of source.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');graph(path.resolve(path.dirname(f),m[1]),boundary,seen);}return seen;
}
function manifest(s){
  assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),names);
  const p=code(path.join(s,'PROOF.bend'));assert.deepEqual([...p.matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),names);
  assert.match(p,/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function invoke(f){const start=performance.now();const r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{encoding:'utf8',timeout:180000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,output:(r.stdout+r.stderr).trim(),elapsed_seconds:(performance.now()-start)/1000};}
function clean(r){assert.equal(r.status,0,r.output.slice(-2500));assert.equal(r.output,'All terms check.');}
function copy(name){const e=path.join(temp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/castling'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(f,from,to){const s=text(f);assert.equal(s.split(from).length,2,'nonunique mutation');fs.writeFileSync(f,s.replace(from,to));}
function reject(name,entry,edit,location){
  const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': must reject');
  assert.match(r.output,/expected[\s\S]*observed/,name+': not semantic');assert.doesNotMatch(r.output,/no such file|RangeError|Maximum call stack|Segmentation fault|more than once|a decreasing self-call|a defined name|a pattern \(a binder/);
  assert.match(r.output,location,name+': wrong location');controls.push({name,rejected:true,kind:'source semantic/refinement',entry,diagnostic_sha256:sha(r.output),diagnostic_bytes:Buffer.byteLength(r.output),excerpt:r.output.slice(-1600)});
}
function guard(name,edit,check){const d=copy(name);edit(d);assert.throws(()=>check(d));controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);
  for(const f of ['focused.js','verify_native.js','probe.bend'])closure.add(path.join(suite,f));
  closure.add(path.resolve(suite,'../../toolchain.json'));closure.add(path.resolve(suite,'../../verify_compiler.js'));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=invoke(path.join(suite,'consumer.bend'));clean(consumer);
  reject('actual-omits-rook','Actual.bend',d=>replace(d.chess,'+rook_to = select_u64(castle, U64.bit(U32.to_nat(U32.div(U32.add(src, dst), 2))), U64.zero())','+rook_to = select_u64(castle, U64.zero(), U64.zero())'),/Location: unfold\b/);
  reject('actual-chooses-wrong-rook','Actual.bend',d=>replace(d.chess,'rook_from = Bool.pick(U32, U32.is_eq(U32.and(dst, 7), 6), U32.inc(dst), U32.sub(dst, 2))','rook_from = Bool.pick(U32, U32.is_eq(U32.and(dst, 7), 6), dst, U32.sub(dst, 2))'),/Location: unfold\b/);
  reject('model-wrong-rook-midpoint','Actual.bend',d=>replace(path.join(d.s,'Actual.bend'),'U64.bit(U32.to_nat(U32.div(U32.add(src,dst),2)))','U64.bit(U32.to_nat(U32.div(U32.add(src,dst),3)))'),/Location: unfold\b/);
  reject('public-result-is-original-board','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'== C.expected(b,src,dst) : Chess.Board}','== b : Chess.Board}'),/Location: Laws\.castling_make_move_exact_update\b/);
  reject('omit-consistency-premise','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'good: {Board.valid(b) == True{} : Bool}','good: {True{} == True{} : Bool}'),/Location: Laws\.castling_make_move_preserves_representation\b/);
  reject('omit-rook-target-freshness','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'fresh: {Board.fresh_mask(b,R.rook(route)) == True{} : Bool}','fresh: {True{} == True{} : Bool}'),/Location: Laws\.castling_make_move_preserves_representation\b/);
  reject('wrong-public-flag','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'Chess.Ply{src,dst,0,2}','Chess.Ply{src,dst,0,3}'),/Location: Laws\.castling_make_move_exact_update\b/);
  reject('false-preservation-result','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'{Board.valid(Chess.make_move(b,Chess.Ply{R.src(route),R.dst(route),0,2})) == True{} : Bool}','{Board.valid(Chess.make_move(b,Chess.Ply{R.src(route),R.dst(route),0,2})) == False{} : Bool}'),/Location: Laws\.castling_make_move_preserves_representation\b/);
  guard('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law castling_make_move_exact_update:','def missing:'),d=>manifest(d.s));
  guard('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.castling_make_move_exact_update(','def missing('),d=>manifest(d.s));
  guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  guard('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Actual.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Actual.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  guard('unsafe-proof',d=>replace(path.join(d.s,'Actual.bend'),'def actual(','@unsafe\ndef actual('),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  guard('symlink-proof',d=>{const f=path.join(d.s,'Actual.bend');fs.renameSync(f,f+'.orig');fs.symlinkSync('Actual.bend.orig',f);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>clean({status:0,output:'All terms check.\nWARNING: unsafe or foreign dependency'}));controls.push({name:'warning-on-zero-exit',rejected:true,kind:'synthetic output-wrapper unit; not compiler execution'});
  assert.equal(controls.length,17);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',controls_gate:'PASS',consumer:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:2,new_laws:names,inherited_gate_run:false,consumer_seconds:consumer.elapsed_seconds,negative_controls:controls,source_sha256s:before,scope:'Actual flag2/promotion0 complete Board update; conditional partition preservation for four coordinate routes with initially empty rook destination. Not legal castling, rights/path/king-safety or reachability.'};
  const enc=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],enc);console.log(enc.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
