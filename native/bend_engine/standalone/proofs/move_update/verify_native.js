// Actual ordinary make_move and scalar-update helpers versus per-square sets.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-move-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Position.bend','standalone/Text.bend',
 'standalone/proofs/move_update/probe.bend','standalone/proofs/move_update/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(cmd,argv,timeout=90000){
 const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-3500));assert.equal(r.stderr,'','unexpected compiler/native diagnostic');return r.stdout;}
const empty=()=>Array.from({length:64},()=>({kinds:new Set(),colors:new Set()}));
const clone=a=>a.map(x=>({kinds:new Set(x.kinds),colors:new Set(x.colors)}));
function put(a,sq,kind,color){a[sq]={kinds:new Set([kind]),colors:new Set([color])};}
function consistent(a){return a.every(x=>(!x.kinds.size&&!x.colors.size)||(x.kinds.size===1&&x.colors.size===1));}
function encode(a,meta){
 const planes=Array(8).fill(0n);
 a.forEach((x,i)=>{const bit=1n<<BigInt(i);for(const k of x.kinds)planes[k]|=bit;for(const c of x.colors)planes[c===1?6:7]|=bit;});
 return [...planes.flatMap(x=>[Number(x>>32n),Number(x&0xffffffffn)]),...meta];
}
function decode(values){
 const a=empty();
 for(let p=0;p<8;p++){
  const word=(BigInt(values[2*p])<<32n)|BigInt(values[2*p+1]);
  for(let sq=0;sq<64;sq++)if(word&(1n<<BigInt(sq))){if(p<6)a[sq].kinds.add(p);else a[sq].colors.add(p===6?1:0);}
 }return a;
}
let state=0x59724613;
function random(){state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;}
function randomBoard(){const a=empty();for(let i=0;i<64;i++)if(random()%3)put(a,i,random()%6,random()%2);return a;}
const corner=new Map([[0,2],[7,1],[56,8],[63,4]]);
function ordinary(a,meta,src,dst){
 // The low-level decoder's priority is relevant on explicit empty/inconsistent diagnostics.
 const kind=[0,1,2,3,4].find(k=>a[src].kinds.has(k))??5;
 const b=clone(a);b[src]={kinds:new Set(),colors:new Set()};b[dst]={kinds:new Set(),colors:new Set()};
 const white=meta[0]===1;put(b,dst,kind,white?1:0);
 const lost=(corner.get(src)||0)|(corner.get(dst)||0)|(kind===5?(white?3:12):0);
 const ep=kind===0&&((src^dst)===16)?Math.floor((src+dst)/2):64;
 return {board:b,meta:[(meta[0]^1)>>>0,(meta[1]&~lost)>>>0,ep]};
}
const fixtures=[],counts={quiet:0,capture:0,all_source_destination_pairs:0,empty_source:0,raw_metadata:0,inconsistent_input:0,clear_mask:0,scalar_update:0};
let validMoves=0,invalidBefore=0;
function addMove(category,a,meta,src,dst){
 const expected=ordinary(a,meta,src,dst),good=consistent(a);
 if(good){assert.ok(consistent(expected.board));validMoves++;}else invalidBefore++;
 fixtures.push({category,input:[0,src,dst,0,...encode(a,meta)],expected:encode(expected.board,expected.meta)});counts[category]++;
}
for(let sq=0;sq<64;sq++)for(let kind=0;kind<6;kind++)for(let color=0;color<2;color++){
 const dst=(sq+17)%64,a=randomBoard();put(a,sq,kind,color);a[dst]={kinds:new Set(),colors:new Set()};
 const meta=[color,random()%16,random()%65];addMove('quiet',a,meta,sq,dst);
 const b=clone(a);put(b,dst,(kind+1)%6,1-color);addMove('capture',b,meta,sq,dst);
}
for(let src=0;src<64;src++)for(let dst=0;dst<64;dst++){
 const a=empty(),kind=(src+dst)%6,color=(src^dst)%2;put(a,src,kind,color);
 if(src!==dst && (src+dst)%2)put(a,dst,(kind+2)%6,1-color);
 addMove('all_source_destination_pairs',a,[color,15,64],src,dst);
}
for(let sq=0;sq<64;sq++){
 addMove('empty_source',empty(),[1,15,64],sq,(sq+1)%64);
 const a=randomBoard();put(a,sq,sq%6,sq%2);addMove('raw_metadata',a,[random(),random(),random()],sq,(sq+31)%64);
}
for(let j=0;j<128;j++){
 const a=randomBoard(),sq=j%64;a[(sq+8)%64]={kinds:new Set([0,4]),colors:new Set([0,1])};
 addMove('inconsistent_input',a,[j%2,15,64],sq,(sq+17)%64);
}
for(let j=0;j<256;j++){
 const a=randomBoard(),meta=[random(),random(),random()];
 if(j%2)a[(j+1)%64]={kinds:new Set([1,3]),colors:new Set([0,1])};
 const hi=j===0?0:j===1?0xffffffff:random(),lo=j===0?0:j===1?0xffffffff:random();
 const mask=(BigInt(hi)<<32n)|BigInt(lo),b=clone(a);
 for(let i=0;i<64;i++)if(mask&(1n<<BigInt(i)))b[i]={kinds:new Set(),colors:new Set()};
 if(consistent(a))assert.ok(consistent(b));
 fixtures.push({category:'clear_mask',input:[1,hi,lo,0,...encode(a,meta)],expected:encode(b,meta)});counts.clear_mask++;
}
for(let j=0;j<512;j++){
 const values=Array.from({length:19},()=>random()),putFlag=j%2;
 if(j<8){values[0]=j%2?0xffffffff:0;values[1]=j%2?0:0xffffffff;}
 const sets=Array.from({length:3},(_,p)=>{const w=(BigInt(values[p*2])<<32n)|BigInt(values[p*2+1]);return new Set(Array.from({length:64},(_,i)=>i).filter(i=>w&(1n<<BigInt(i))));});
 const result=new Set([...sets[0]].filter(i=>!sets[1].has(i)));if(putFlag)for(const i of sets[2])result.add(i);
 let word=0n;for(const i of result)word|=1n<<BigInt(i);
 const expected=[Number(word>>32n),Number(word&0xffffffffn),...values.slice(2)];
 fixtures.push({category:'scalar_update',input:[2,putFlag,0,0,...values],expected});counts.scalar_update++;
}
assert.equal(fixtures.length,6656);
const malformed=[];
const base=fixtures[0].input;
for(const [i,val] of [[0,'3'],[1,'64'],[2,'64'],[3,'1'],[4,'4294967296'],[4,'-1'],[4,'x']]){const a=base.map(String);a[i]=val;malformed.push(a);}
malformed.push(base.slice(1).map(String));malformed.push(Array.from({length:65},()=>base).flat().map(String));
function compare(raw,rows,label){
 const lines=raw.trimEnd().split('\n');assert.equal(lines.length,rows.length,label+': row count');
 for(let i=0;i<rows.length;i++){
  const got=lines[i].split(' ').map(Number),want=rows[i].expected;assert.equal(got.length,19,label+': complete Board fields');
  for(let f=0;f<19;f++)assert.equal(got[f],want[f],`${label} row ${i} (${rows[i].category}), field ${f}`);
 }
}
try{
 const c=path.join(temp,'probe.c'),cli=path.join(compiler,'bend2/main.ts');
 run(process.execPath,[cli,path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe],120000);
  const hash=createHash('sha256');let rows=0;
  for(let i=0;i<fixtures.length;i+=64){const batch=fixtures.slice(i,i+64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);compare(raw,batch,`${mode} batch ${i}`);hash.update(raw);rows+=batch.length;}
  for(const bad of malformed){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid move-update/);}
  modes.push({mode,rows,complete_board_fields:rows*19,invalid_rejections:malformed.length,output_sha256:hash.digest('hex')});console.error(`PASS ${mode}: ${rows} complete Board rows`);
 }
 const mutations=[];
 for(const [name,edit] of [
  ['actual-ordinary-move-noop',code=>{const start=code.indexOf('def make_move('),end=code.indexOf('\ndef put_move(',start);assert.ok(start>=0&&end>start);return code.slice(0,start)+'def make_move(+b: Board,m: Ply) -> Board:\n  b\n'+code.slice(end);}],
  ['actual-kernel-keeps-removed-bits',code=>{const needle='U64.or(U64.and_not(bb, remove), select_u64(put, target, U64.zero()))';assert.equal(code.split(needle).length,2);return code.replace(needle,'U64.or(bb, select_u64(put, target, U64.zero()))');}]
 ]){
  const copy=path.join(temp,name);fs.cpSync(engine,copy,{recursive:true});const chess=path.join(copy,'legal_probe/Chess.bend');fs.writeFileSync(chess,edit(fs.readFileSync(chess,'utf8')));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');run(process.execPath,[cli,path.join(copy,'standalone/proofs/move_update/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',mc,'-pthread','-lm','-o',exe]);
  const batch=fixtures.slice(0,64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
  let rejection;try{compare(raw,batch,name);}catch(e){rejection=e;}
  assert.ok(rejection instanceof assert.AssertionError,name+': must fail as a value mismatch');
  mutations.push({name,rejected:true,compiled_and_executed:true,message:rejection.message,actual:rejection.actual,expected:rejection.expected});
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,rows_per_mode:fixtures.length,complete_board_fields_per_mode:fixtures.length*19,
  cases_by_operation:counts,representation_consistent_move_inputs:validMoves,inconsistent_move_diagnostics:invalidBefore,
  malformed_requests_per_mode:malformed.length,fixture_sha256:sha(JSON.stringify(fixtures)),modes,mutations,source_sha256s:before,
  cc:run(cc,['--version']).split('\n')[0],
  scope:'Actual flag0/promotion0 make_move plus scalar-kernel/probe deletion lift. Includes explicitly nonlegal raw calls; not a legal-move generator or castling/promotion/EP theorem. No proof model executes in the native candidate.'};
 const result=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],result);console.log(result.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
