// Exact-source producer/proof gate. Negative controls never accept parser/affine failures.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report')||(args.length===4&&args[1]==='--controls-only'&&args[2]==='--report'),'usage: focused.js COMPILER [--report FILE | --controls-only --report FILE]');
const controlsOnly=args[1]==='--controls-only',reportPath=controlsOnly?args[3]:args[2];
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'initialized-attacks-proofs-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),text=p=>fs.readFileSync(p,'utf8');
const code=p=>text(p).split('\n').map(s=>s.split('#')[0]).join('\n');
const names=['initialized_piece_attack_matches_geometry','initialized_attacked_matches_geometry','initialized_single_king_check_matches_geometry'];
function manifest(s){assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(x=>x[1]),names);assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(x=>x[1]),names);assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);}
function graph(f,e,seen=new Set()){f=path.resolve(f);const rel=path.relative(e,f);assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='legal_probe/Chess.bend'||rel==='bitboard_probe/Sliders.bend'));assert.ok(fs.lstatSync(f).isFile(),'nonregular proof input');assert.equal(fs.realpathSync(f),f,'symlink proof');if(seen.has(f))return seen;seen.add(f);const s=code(f);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or hole');for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign dependency');graph(path.resolve(path.dirname(f),m[1]),e,seen);}return seen;}
function invoke(f){const start=performance.now(),r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),f],{encoding:'utf8',timeout:1800000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return {status:r.status,raw:r.stdout+r.stderr,seconds:(performance.now()-start)/1000};}
function safe(r){assert.equal(r.status,0,r.raw.slice(-3000));assert.equal(r.raw.trim(),'All terms check.');}
function mutation(name){const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});return {e,s:path.join(e,'standalone/proofs/initialized_attacks'),chess:path.join(e,'legal_probe/Chess.bend')};}
function replace(p,a,b){const s=text(p);assert.equal(s.split(a).length,2,'nonunique mutation site: '+p);fs.writeFileSync(p,s.replace(a,b));}
const controls=[];
function reject(name,entry,edit,location){const d=mutation(name);edit(d);const r=invoke(path.join(d.s,entry));assert.equal(r.status,1,name+': expected failure');assert.match(r.raw,/expected[\s\S]*observed/,name+': ordinary refinement failure required');assert.doesNotMatch(r.raw,/consumed more than once|no such file|a defined name|a pattern \(a binder|a decreasing self-call|Maximum call stack|RangeError|Segmentation fault/i);assert.match(r.raw,location,name+': wrong source location');controls.push({name,kind:'source semantic/refinement',rejected:true,entry,location:r.raw.match(/Location: ([^\n]+)/)?.[1],diagnostic_sha256:sha(r.raw),diagnostic_excerpt:r.raw.slice(-2000)});console.error('PASS control '+name);}
function policy(name,edit,test){const d=mutation(name);edit(d);assert.throws(()=>test(d));controls.push({name,kind:'manifest/import policy',rejected:true});}
try{
  manifest(suite);const closure=graph(path.join(suite,'consumer.bend'),engine);graph(path.join(suite,'probe.bend'),engine,closure);
  for(const f of ['focused.js','verify_native.py'])closure.add(path.join(suite,f));closure.add(path.resolve(suite,'../attack_witness/verify_native.py'));for(const f of ['toolchain.json','verify_compiler.js'])closure.add(path.resolve(suite,'../..',f));
  const hashes=()=>Object.fromEntries([...closure].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before=hashes(),consumer=controlsOnly?{status:null,raw:'',seconds:0}:invoke(path.join(suite,'consumer.bend'));if(!controlsOnly)safe(consumer);
  const positive_modules=[];for(const entry of ['Routing.bend','Wire.bend','Facts.bend']){const checked=invoke(path.join(suite,entry));safe(checked);positive_modules.push({entry,checked:true,seconds:checked.seconds});}

  // Internal certificate premises here are discharged by Masks/Composition in the public theorem.
  // These controls are refinements, not replacement axioms or claims about unchecked public statements.
  reject('actual-bishop-uses-rook-key','Routing.bend',d=>replace(d.chess,
    'case 2:\n      slide(table, U32.add(64, sq), occ)','case 2:\n      slide(table, sq, occ)'),/Location: (?:Routing\.)?bishop\b/);
  reject('actual-queen-omits-diagonal','Routing.bend',d=>replace(d.chess,
    'or_result(rook, slide(table, U32.add(64, sq), occ))','(table, rook)'),/Location: (?:Routing\.)?queen_tail_actual\b/);
  reject('actual-wrong-attacker-color','../attack_witness/Flow.bend',d=>replace(d.chess,
    'U64.and(hits, color(b, by))','U64.and(hits, color(b, U32.xor(by, 1)))'),/Location: (?:Flow\.)?bishop\b/);
  reject('actual-omitted-knight','../attack_witness/Flow.bend',d=>replace(d.chess,
    'U64.and(na, get_knights(b))','U64.zero()'),/Location: (?:Flow\.)?knight\b/);
  reject('actual-unreversed-pawn','../attack_witness/Flow.bend',d=>replace(d.chess,
    'attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))',
    'attacked_pawn(b, sq, by, attack(0, table, sq, by, occupied(b)))'),/Location: (?:Flow\.)?attacked\b/);
  reject('actual-checks-own-side','../attack_witness/King.bend',d=>replace(d.chess,
    'attacked(table, b, U64.ctz(U64.and(get_kings(b), color(b, side))), U32.xor(side, 1))',
    'attacked(table, b, U64.ctz(U64.and(get_kings(b), color(b, side))), side)'),/Location: (?:King\.)?checked\b/);
  reject('model-omits-reverse-pawn','Wire.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'mask(Piece.Pawn{},sq,Bool.not(by),occ)','mask(Piece.Pawn{},sq,by,occ)'),/Location: (?:Wire\.)?queries\b/);
  reject('model-omits-bishop-mask','Wire.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'mask(Piece.Bishop{},sq,False{},occ)}','U64.zero()}'),/Location: (?:Wire\.)?queries\b/);
  reject('certificate-discards-initialized-table','Wire.bend',d=>replace(path.join(d.s,'Wire.bend'),
    '(B.run(d,seed,n,extra),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))) : Array<U64> & U64}',
    '(Array.new(U64,d,seed),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))) : Array<U64> & U64}'),/Location: (?:Wire\.)?queries\b/);
  reject('scalar-domain-omits-square-bound','Facts.bend',d=>replace(path.join(d.s,'Facts.bend'),
    'def checked(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool})',
    'def checked(+q: Nat,bound: {True{} == True{} : Bool})'),/Location: (?:Facts\.)?checked\b/);
  policy('missing-law',d=>replace(path.join(d.s,'LAWS.bend'),'law initialized_piece_attack_matches_geometry:','def missing:'),d=>manifest(d.s));
  policy('missing-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.initialized_piece_attack_matches_geometry(','def missing('),d=>manifest(d.s));
  policy('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
  policy('missing-consumer-import',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
  policy('hole',d=>fs.appendFileSync(path.join(d.s,'Spec.bend'),'\n?missing\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('foreign',d=>fs.appendFileSync(path.join(d.s,'Spec.bend'),'\nimport "./oracle.c"\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('unsafe',d=>fs.appendFileSync(path.join(d.s,'Spec.bend'),'\n@unsafe\n'),d=>graph(path.join(d.s,'consumer.bend'),d.e));
  policy('symlink',d=>{const p=path.join(d.s,'Spec.bend');fs.renameSync(p,p+'.original');fs.symlinkSync('Spec.bend.original',p);},d=>graph(path.join(d.s,'consumer.bend'),d.e));
  assert.throws(()=>safe({status:0,raw:'All terms check.\nWARNING: unsafe dependency'}));controls.push({name:'warning-on-zero-exit',kind:'synthetic output-wrapper unit; not compiler execution',rejected:true});
  assert.equal(controls.length,19);assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:controlsOnly?'NOT_RUN':'PASS',consumer:controlsOnly?'NOT_RUN':'PASS',controls_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:3,new_laws:names,consumer_seconds:consumer.seconds,consumer_raw:consumer.raw,positive_modules,negative_controls:controls,source_sha256s:before,inherited_gate_run:false,scope:'All six initialized piece queries, complete initialized attacked and singleton in_check equal independent target-centred geometry. Input read certificates are derived internally. Source controls include three new dispatch/composition boundaries and retained attack-witness implementation bridges. No general forward-reverse ray theorem, castling singleton propagation or native lifetime claim.'};
  const json=JSON.stringify(report,null,2)+'\n';if(reportPath)fs.writeFileSync(reportPath,json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
