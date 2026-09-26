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
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-filter-proofs-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),text=p=>fs.readFileSync(p,'utf8');
const code=p=>text(p).split('\n').map(s=>s.split('#')[0]).join('\n');
const names=['scan_excludes_castling','final_filter_preserves_predicate','both_castling_sides_provenance','filtered_castling_provenance','legal_castle_guard_and_route','legal_castle_preserves_representation'];
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,e,seen=new Set()){f=path.resolve(f);const rel=path.relative(e,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'));assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlink proof');if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or hole');for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign dependency');graph(path.resolve(path.dirname(f),m[1]),e,seen);}return seen;}
function invoke(f){const start=performance.now(),r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{encoding:'utf8',timeout:180000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,raw:r.stdout+r.stderr,seconds:(performance.now()-start)/1000};}
function safe(r){assert.equal(r.status,0,r.raw.slice(-3000));assert.equal(r.raw.trim(),'All terms check.');}
function mutation(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/castle_filter'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(p,a,b){const s=text(p);assert.equal(s.split(a).length,2,'nonunique mutation site: '+p);fs.writeFileSync(p,s.replace(a,b));}
const controls=[];
function reject(name,entry,edit,location){const d=mutation(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': expected failure');assert.match(r.raw,/expected[\s\S]*observed/,name+': ordinary refinement failure required');assert.doesNotMatch(r.raw,/consumed more than once|no such file|a defined name|a pattern \(a binder|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault/i);assert.match(r.raw,location,name+': wrong source location');controls.push({name,kind:'source semantic/refinement',rejected:true,entry,location:r.raw.match(/Location: ([^\n]+)/)?.[1],diagnostic_sha256:sha(r.raw),diagnostic_excerpt:r.raw.slice(-2000)});console.error('PASS control '+name);}
function policy(name,edit,test){const d=mutation(name);edit(d);assert.throws(()=>test(d));controls.push({name,kind:'manifest/import policy',rejected:true});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);graph(path.join(suite,'probe.bend'),engine,closure);
  for(const f of ['focused.js','verify_native.js'])closure.add(path.join(suite,f));for(const f of ['toolchain.json','verify_compiler.js'])closure.add(path.resolve(suite,'../..',f));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=invoke(path.join(suite,'consumer.bend'));safe(consumer);
  reject('actual-scan-manufactures-flag-two','Scan.bend',d=>replace(d.chess,'Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, ep_sq)))','2'),/Location: (?:Scan\.)?destinations\b/);
  reject('actual-fast-filter-invents-move','Filter.bend',d=>replace(d.chess,'(table, Con{m, acc})','(table, Con{Ply{0,1,0,2}, acc})'),/Location: (?:Filter\.)?fast_step_then\b/);
  reject('actual-checked-filter-invents-move','Filter.bend',d=>replace(d.chess,'(table, retain_move(check, m, acc))','(table, retain_move(check, Ply{0,1,0,2}, acc))'),/Location: (?:Filter\.)?after_then\b/);
  reject('actual-legal-moves-seeds-false-castle','Legal.bend',d=>replace(d.chess,
    'scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Nil{})))))',
    'scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Con{Ply{0,1,0,2},Nil{}})))))'),/Location: (?:Legal\.)?then\b/);
  function lawEdit(d,name,from,to){const p=path.join(d.s,'LAWS.bend'),s=text(p),start=s.indexOf('law '+name+':');assert.ok(start>=0);let end=s.indexOf('\nlaw ',start+4);if(end<0)end=s.length;const section=s.slice(start,end);assert.equal(section.split(from).length,2);fs.writeFileSync(p,s.slice(0,start)+section.replace(from,to)+s.slice(end));}
  reject('public-omits-castling-flag','consumer.bend',d=>lawEdit(d,'legal_castle_guard_and_route','for is_castle: {S.flag(m) == 2 : U32}','for is_castle: Unit'),/Location: LAWS\.legal_castle_guard_and_route\b/);
  reject('public-omits-result-membership','consumer.bend',d=>lawEdit(d,'legal_castle_guard_and_route','for here: E.member(m,E.moves(Chess.legal_moves(table,b)))','for here: Unit'),/Location: LAWS\.legal_castle_guard_and_route\b/);
  reject('public-omits-input-consistency','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'for good: {Board.valid(b) == True{} : Bool}','for good: {True{} == True{} : Bool}'),/Location: LAWS\.legal_castle_preserves_representation\b/);
  reject('public-omits-input-predicate','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),'for before: S.all(P,E.moves(r))','for before: Unit'),/Location: LAWS\.final_filter_preserves_predicate\b/);
  policy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law scan_excludes_castling:','def missing:'),d=>manifest(d.s));
  policy('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.scan_excludes_castling(','def missing('),d=>manifest(d.s));
  policy('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  policy('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  policy('hole',d=>fs.appendFileSync(path.join(d.s,'Lists.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('foreign',d=>fs.appendFileSync(path.join(d.s,'Lists.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('unsafe',d=>fs.appendFileSync(path.join(d.s,'Lists.bend'),'\n@unsafe\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('symlink',d=>{const p=path.join(d.s,'Lists.bend');fs.renameSync(p,p+'.original');fs.symlinkSync('Lists.bend.original',p);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>safe({status:0,raw:'All terms check.\nWARNING: unsafe dependency'}));controls.push({name:'warning-on-zero-exit',kind:'synthetic output-wrapper unit; not compiler execution',rejected:true});
  assert.equal(controls.length,17);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',consumer:'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:6,new_laws:names,consumer_seconds:consumer.seconds,consumer_raw:consumer.raw,negative_controls:controls,source_sha256s:before,inherited_gate_run:false,scope:'Actual noncastle scan exclusion, paired castling provenance and final-filter predicate preservation compose into guard/route and representation guarantees for every flag-two member of actual legal_moves. No assumed absence from scan. Not independent semantic attack correctness, legality/completeness, table identity or lifetime.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
