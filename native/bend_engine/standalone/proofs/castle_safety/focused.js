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
const names=['owned_king_source_requires_check','blocker_stage_uses_sensitive_mask','generated_castle_requires_check','generated_castle_takes_checked_step','positive_check_rejects_current_move','complete_generator_equals_forced_castle_checks'];
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,e,seen=new Set()){f=path.resolve(f);const rel=path.relative(e,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'));assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlink proof');if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or hole');for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign dependency');graph(path.resolve(path.dirname(f),m[1]),e,seen);}return seen;}
function invoke(f){const start=performance.now(),r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{encoding:'utf8',timeout:180000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,raw:r.stdout+r.stderr,seconds:(performance.now()-start)/1000};}
function safe(r){assert.equal(r.status,0,r.raw.slice(-3000));assert.equal(r.raw.trim(),'All terms check.');}
function mutation(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/castle_safety'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(p,a,b){const s=text(p);assert.equal(s.split(a).length,2,'nonunique mutation site: '+p);fs.writeFileSync(p,s.replace(a,b));}
const controls=[];
function reject(name,entry,edit,location){const d=mutation(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': expected failure');assert.match(r.raw,/expected[\s\S]*observed/,name+': ordinary refinement failure required');assert.doesNotMatch(r.raw,/consumed more than once|no such file|a defined name|a pattern \(a binder|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault/i);assert.match(r.raw,location,name+': wrong source location');controls.push({name,kind:'source semantic/refinement',rejected:true,entry,location:r.raw.match(/Location: ([^\n]+)/)?.[1],diagnostic_sha256:sha(r.raw),diagnostic_excerpt:r.raw.slice(-2000)});console.error('PASS control '+name);}
function policy(name,edit,test){const d=mutation(name);edit(d);assert.throws(()=>test(d));controls.push({name,kind:'manifest/import policy',rejected:true});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);graph(path.join(suite,'probe.bend'),engine,closure);graph(path.join(suite,'generator_probe.bend'),engine,closure);
  for(const f of ['focused.js','verify_native.js','verify_forced.js'])closure.add(path.join(suite,f));for(const f of ['toolchain.json','verify_compiler.js'])closure.add(path.resolve(suite,'../..',f));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=invoke(path.join(suite,'consumer.bend'));safe(consumer);

  reject('actual-sensitive-mask-omits-kings','Step.bend',d=>replace(d.chess,
    'sensitive = U64.and(own, U64.or(rays, get_kings(b)))','sensitive = U64.and(own, rays)'),/Location: (?:Step\.)?blockers\b/);
  reject('actual-required-tests-destination','Mask.bend',d=>replace(d.chess,
    'Bool.or(U32.is_eq(flag, 1), U64.test_bit(sensitive, U32.to_nat(src)))',
    'Bool.or(U32.is_eq(flag, 1), U64.test_bit(sensitive, U32.to_nat(dst)))'),/Location: (?:Mask\.)?owned\b/);
  reject('actual-required-branch-skips-check','Step.bend',d=>replace(d.chess,
    'case True{}: filter_step(b, m, r)',
    'case True{}:\n      (table, acc) = r\n      (table, Con{m, acc})'),/Location: (?:Step\.)?checked\b/);
  reject('actual-check-uses-opponent','Step.bend',d=>replace(d.chess,
    'filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))',
    'filter_after(m, acc, in_check(table, make_move(b, m), U32.xor(get_turn(b), 1)))'),/Location: (?:Step\.)?checked\b/);
  reject('actual-check-uses-parent-board','Step.bend',d=>replace(d.chess,
    'filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))',
    'filter_after(m, acc, in_check(table, b, get_turn(b)))'),/Location: (?:Step\.)?checked\b/);
  reject('actual-positive-check-retains-move','Step.bend',d=>replace(d.chess,
    '(table, retain_move(check, m, acc))','(table, retain_move(False{}, m, acc))'),/Location: (?:Step\.)?rejected\b/);
  reject('audit-model-disables-checking','Pipeline.bend',d=>replace(path.join(d.s,'Forced.bend'),
    'Bool.or(U32.is_eq(ChainSpec.flag(m),2),Chess.filter_requires(S.sensitive(b,rays),m))','False{}'),/Location: (?:Pipeline\.)?choose\b/);
  function publicPart(d,startLaw,endLaw,from,to){
    const p=path.join(d.s,'LAWS.bend'),s=text(p),a=s.indexOf('law '+startLaw+':'),b=endLaw?s.indexOf('law '+endLaw+':'):s.length;
    assert.ok(a>=0&&b>a);const part=s.slice(a,b);assert.equal(part.split(from).length,2);fs.writeFileSync(p,s.slice(0,a)+part.replace(from,to)+s.slice(b));
  }
  reject('public-omits-owned-king-premise','consumer.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'for present: {S.owned_king(b,src) == True{} : Bool}','for present: {True{} == True{} : Bool}'),/Location: LAWS\.owned_king_source_requires_check\b/);
  reject('public-omits-generator-membership','consumer.bend',d=>publicPart(d,'generated_castle_requires_check','generated_castle_takes_checked_step',
    'for here: E.member(m,E.moves(Chess.legal_moves(table,b)))','for here: Unit'),/Location: LAWS\.generated_castle_requires_check\b/);
  reject('public-omits-castling-tag','consumer.bend',d=>publicPart(d,'generated_castle_requires_check','generated_castle_takes_checked_step',
    'for tag: {ChainSpec.flag(m) == 2 : U32}','for tag: Unit'),/Location: LAWS\.generated_castle_requires_check\b/);
  policy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law owned_king_source_requires_check:','def missing:'),d=>manifest(d.s));
  policy('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.owned_king_source_requires_check(','def missing('),d=>manifest(d.s));
  policy('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  policy('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  policy('hole',d=>fs.appendFileSync(path.join(d.s,'Mask.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('foreign',d=>fs.appendFileSync(path.join(d.s,'Mask.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('unsafe',d=>fs.appendFileSync(path.join(d.s,'Mask.bend'),'\n@unsafe\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('symlink',d=>{const p=path.join(d.s,'Mask.bend');fs.renameSync(p,p+'.original');fs.symlinkSync('Mask.bend.original',p);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>safe({status:0,raw:'All terms check.\nWARNING: unsafe dependency'}));controls.push({name:'warning-on-zero-exit',kind:'synthetic output-wrapper unit; not compiler execution',rejected:true});
  assert.equal(controls.length,19);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',consumer:'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:6,new_laws:names,consumer_seconds:consumer.seconds,consumer_raw:consumer.raw,negative_controls:controls,source_sha256s:before,inherited_gate_run:false,scope:'Actual sensitive mask includes owned kings, generated flag2 members require full checking for any ray mask, actual fast step equals the real make_move/in_check/filter_after path, and positive check rejection preserves the original table/accumulator. Includes complete actual-generator equality to a variant forcing flag2 checks explicitly. Not independent attack semantics or full-generator completeness.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
