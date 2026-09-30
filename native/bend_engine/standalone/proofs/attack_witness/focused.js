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
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'attack-witness-proofs-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),text=p=>fs.readFileSync(p,'utf8');
const code=p=>text(p).split('\n').map(s=>s.split('#')[0]).join('\n');
const names=['mask_reduction_has_square_witness','actual_attacked_uses_witness_reduction','single_king_in_check_uses_witness_reduction'];
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,e,seen=new Set()){f=path.resolve(f);const rel=path.relative(e,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'));assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlink proof');if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or hole');for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign dependency');graph(path.resolve(path.dirname(f),m[1]),e,seen);}return seen;}
function invoke(f){const start=performance.now(),r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{encoding:'utf8',timeout:180000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,raw:r.stdout+r.stderr,seconds:(performance.now()-start)/1000};}
function safe(r){assert.equal(r.status,0,r.raw.slice(-3000));assert.equal(r.raw.trim(),'All terms check.');}
function mutation(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/attack_witness'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(p,a,b){const s=text(p);assert.equal(s.split(a).length,2,'nonunique mutation site: '+p);fs.writeFileSync(p,s.replace(a,b));}
const controls=[];
function reject(name,entry,edit,location){const d=mutation(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': expected failure');assert.match(r.raw,/expected[\s\S]*observed/,name+': ordinary refinement failure required');assert.doesNotMatch(r.raw,/consumed more than once|no such file|a defined name|a pattern \(a binder|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault/i);assert.match(r.raw,location,name+': wrong source location');controls.push({name,kind:'source semantic/refinement',rejected:true,entry,location:r.raw.match(/Location: ([^\n]+)/)?.[1],diagnostic_sha256:sha(r.raw),diagnostic_excerpt:r.raw.slice(-2000)});console.error('PASS control '+name);}
function policy(name,edit,test){const d=mutation(name);edit(d);assert.throws(()=>test(d));controls.push({name,kind:'manifest/import policy',rejected:true});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);graph(path.join(suite,'probe.bend'),engine,closure);
  for(const f of ['focused.js','verify_native.py'])closure.add(path.join(suite,f));for(const f of ['toolchain.json','verify_compiler.js'])closure.add(path.resolve(suite,'../..',f));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=invoke(path.join(suite,'consumer.bend'));safe(consumer);

  reject('actual-wrong-attacker-color','Flow.bend',d=>replace(d.chess,
    '(table, Bool.not(U64.is_zero(U64.and(hits, color(b, by)))))',
    '(table, Bool.not(U64.is_zero(U64.and(hits, color(b, U32.xor(by, 1))))))'),/Location: (?:Flow\.)?bishop\b/);
  reject('actual-drops-knight','Flow.bend',d=>replace(d.chess,
    'U64.and(na, get_knights(b))','U64.zero()'),/Location: (?:Flow\.)?knight\b/);
  reject('actual-drops-diagonal-queens','Flow.bend',d=>replace(d.chess,
    'U64.or(get_bishops(b), get_queens(b))','get_bishops(b)'),/Location: (?:Flow\.)?bishop\b/);
  reject('actual-drops-straight-queens','Flow.bend',d=>replace(d.chess,
    'U64.or(get_rooks(b), get_queens(b))','get_rooks(b)'),/Location: (?:Flow\.)?rook\b/);
  reject('actual-wrong-pawn-direction','Flow.bend',d=>replace(d.chess,
    'attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))',
    'attacked_pawn(b, sq, by, attack(0, table, sq, by, occupied(b)))'),/Location: (?:Flow\.)?attacked\b/);
  reject('actual-checks-own-attacks','King.bend',d=>replace(d.chess,
    'attacked(table, b, U64.ctz(U64.and(get_kings(b), color(b, side))), U32.xor(side, 1))',
    'attacked(table, b, U64.ctz(U64.and(get_kings(b), color(b, side))), side)'),/Location: (?:King\.)?checked\b/);
  reject('model-drops-attacker-ownership','Bits.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'Bool.and(ba,Bool.or(b,q))),own)','Bool.and(ba,Bool.or(b,q))),True{})'),/Location: (?:Bits\.)?pointwise\b/);
  reject('actual-intersects-attack-classes','Flow.bend',d=>replace(d.chess,
    'hits = U64.or(hits, U64.and(ba, U64.or(get_bishops(b), get_queens(b))))',
    'hits = U64.and(hits, U64.and(ba, U64.or(get_bishops(b), get_queens(b))))'),/Location: (?:Flow\.)?bishop\b/);
  reject('public-drops-single-king-premise','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'for one: {U64.and(Chess.get_kings(b),Chess.color(b,side)) == U64.bit(sq) : U64}',
    'for one: {True{} == True{} : Bool}'),/Location: LAWS\.single_king_in_check_uses_witness_reduction\b/);
  reject('public-drops-king-square-bound','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'for bound: {Nat.is_lt(sq,64n) == True{} : Bool}',
    'for bound: {True{} == True{} : Bool}'),/Location: LAWS\.single_king_in_check_uses_witness_reduction\b/);
  policy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law mask_reduction_has_square_witness:','def missing:'),d=>manifest(d.s));
  policy('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.mask_reduction_has_square_witness(','def missing('),d=>manifest(d.s));
  policy('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  policy('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  policy('hole',d=>fs.appendFileSync(path.join(d.s,'Bits.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('foreign',d=>fs.appendFileSync(path.join(d.s,'Bits.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('unsafe',d=>fs.appendFileSync(path.join(d.s,'Bits.bend'),'\n@unsafe\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('symlink',d=>{const p=path.join(d.s,'Bits.bend');fs.renameSync(p,p+'.original');fs.symlinkSync('Mask.bend.original',p);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>safe({status:0,raw:'All terms check.\nWARNING: unsafe dependency'}));controls.push({name:'warning-on-zero-exit',kind:'synthetic output-wrapper unit; not compiler execution',rejected:true});
  assert.equal(controls.length,19);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',consumer:'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:3,new_laws:names,consumer_seconds:consumer.seconds,consumer_raw:consumer.raw,negative_controls:controls,source_sha256s:before,inherited_gate_run:false,scope:'Exact attack aggregation as a per-square owned-piece witness relation over actual retrieved masks, and actual in_check under an explicit singleton owned-king premise. Complete table/result pairs preserved between implementations. Independent geometry of table masks, pinned-attacker legality and general chess soundness remain separate.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
