// Independent square-set reference for actual flag-zero promotion updates.
// Does not import the Bend proof model; all original runtime sources stay unchanged.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-promotion-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Position.bend','standalone/Text.bend',
 'standalone/proofs/promotion/probe.bend','standalone/proofs/promotion/choices_probe.bend','standalone/proofs/promotion/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(cmd,argv,timeout=120000){
 const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;
}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-3500));assert.equal(r.stderr,'','unexpected compiler/native diagnostic');return r.stdout;}
const empty=()=>Array.from({length:64},()=>({kinds:new Set(),colors:new Set()}));
const clone=a=>a.map(x=>({kinds:new Set(x.kinds),colors:new Set(x.colors)}));
function put(a,sq,kind,color){a[sq]={kinds:new Set([kind]),colors:new Set([color])};}
function consistent(a){return a.every(x=>(!x.kinds.size&&!x.colors.size)||(x.kinds.size===1&&x.colors.size===1));}
function encode(a,meta){
 const planes=Array(8).fill(0n);
 a.forEach((x,i)=>{const bit=1n<<BigInt(i);for(const k of x.kinds)planes[k]|=bit;for(const c of x.colors)planes[c===1?6:7]|=bit;});
 return [...planes.flatMap(x=>[Number(x>>32n),Number(x&0xffffffffn)]),...meta];
}
let state=0x6b214397;
function random(){state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;}
function randomBoard(){const a=empty();for(let i=0;i<64;i++)if(random()%3)put(a,i,random()%6,random()%2);return a;}
const corner=new Map([[0,2],[7,1],[56,8],[63,4]]);
function promoted(a,meta,src,dst,prom){
 // Original source kind determines current metadata behavior, not the promoted kind.
 // This models the existing low-level API, not independent legal-move metadata.
 const originalKind=[0,1,2,3,4].find(k=>a[src].kinds.has(k))??5;
 const out=clone(a);out[src]={kinds:new Set(),colors:new Set()};out[dst]={kinds:new Set(),colors:new Set()};
 const white=meta[0]===1;put(out,dst,prom,white?1:0);
 const lost=(corner.get(src)||0)|(corner.get(dst)||0)|(originalKind===5?(white?3:12):0);
 const ep=originalKind===0&&((src^dst)===16)?Math.floor((src+dst)/2):64;
 return {board:out,meta:[(meta[0]^1)>>>0,(meta[1]&~lost)>>>0,ep]};
}
const fixtures=[],counts={all_pairs_four_choices:0,pawn_final_rank_geometry:0,raw_metadata:0,inconsistent_input:0};
let consistentInputs=0;
function add(category,a,meta,src,dst,prom){
 const out=promoted(a,meta,src,dst,prom);
 if(consistent(a)){assert.ok(consistent(out.board));consistentInputs++;}
 fixtures.push({category,input:[0,src,dst,prom,...encode(a,meta)],expected:encode(out.board,out.meta)});counts[category]++;
}
for(let src=0;src<64;src++)for(let dst=0;dst<64;dst++)for(let prom=1;prom<=4;prom++){
 const a=empty(),kind=(src+dst)%6,side=(src^dst)%2;put(a,src,kind,side);
 if(src!==dst&&(src+dst)%2)put(a,dst,(kind+2)%6,1-side);
 add('all_pairs_four_choices',a,[side,15,64],src,dst,prom);
}
for(const side of [0,1])for(let file=0;file<8;file++)for(const delta of [-1,0,1]){
 const targetFile=file+delta;if(targetFile<0||targetFile>=8)continue;
 const src=(side?48:8)+file,dst=(side?56:0)+targetFile;
 for(let prom=1;prom<=4;prom++){
  const a=randomBoard();put(a,src,0,side);
  a[dst]={kinds:new Set(),colors:new Set()};if(delta!==0)put(a,dst,3,1-side);
  add('pawn_final_rank_geometry',a,[side,15,64],src,dst,prom);
 }
}
for(let i=0;i<256;i++){
 const a=randomBoard(),src=i%64,dst=(src+17)%64;
 if(i%4===0)a[src]={kinds:new Set(),colors:new Set()};
 add('raw_metadata',a,[random(),random(),random()],src,dst,1+i%4);
}
for(let i=0;i<128;i++){
 const a=randomBoard(),src=i%64,dst=(src+17)%64;
 a[(src+8)%64]={kinds:new Set([0,4]),colors:new Set([0,1])};
 add('inconsistent_input',a,[i%2,15,64],src,dst,1+i%4);
}
assert.equal(fixtures.length,16944);assert.equal(counts.pawn_final_rank_geometry,176);
const distinct=new Set(fixtures.map(x=>JSON.stringify(x.input)));assert.equal(distinct.size,fixtures.length);
const malformed=[],base=fixtures[0].input;
for(const [i,val] of [[0,'1'],[1,'64'],[2,'64'],[3,'0'],[3,'5'],[4,'4294967296'],[4,'-1'],[4,'x']]){
 const a=base.map(String);a[i]=val;malformed.push(a);
}
malformed.push(base.slice(1).map(String));malformed.push(Array.from({length:65},()=>base).flat().map(String));
function compare(raw,rows,label){
 const lines=raw.trimEnd().split('\n');assert.equal(lines.length,rows.length,label+': row count');
 for(let i=0;i<rows.length;i++){
  const got=lines[i].split(' ').map(Number);assert.equal(got.length,19,label+': full Board fields');
  for(let j=0;j<19;j++)assert.equal(got[j],rows[i].expected[j],`${label} row ${i} (${rows[i].category}) field ${j}`);
 }
}
const choices=[];
for(let src=0;src<64;src++)for(const flag of [0,1,2,4294967295]){
 const dst=(src+8)%64;
 choices.push({input:[src,dst,flag],expected:[...Array.from({length:4},(_,i)=>`${src} ${dst} ${i+1} 0`),'9 10 0 3'].join('\n')+'\n'});
}
const badChoices=[['64','0','0'],['0','64','0'],['x','0','0'],['0','0','4294967296'],['0','0'],['0','0','0','0']];
try{
 const cli=path.join(compiler,'bend2/main.ts'),c=path.join(temp,'probe.c'),gc=path.join(temp,'choices.c');
 run(process.execPath,[cli,path.join(suite,'probe.bend'),'-o',c]);
 run(process.execPath,[cli,path.join(suite,'choices_probe.bend'),'-o',gc]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode),gexe=path.join(temp,mode+'-choices');
  for(const [source,binary] of [[c,exe],[gc,gexe]])run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,source,'-pthread','-lm','-o',binary]);
  const h=createHash('sha256');let rows=0;
  for(let i=0;i<fixtures.length;i+=64){
   const batch=fixtures.slice(i,i+64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
   compare(raw,batch,`${mode} batch ${i}`);h.update(raw);rows+=batch.length;
  }
  for(const bad of malformed){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid promotion/);}
  const ghash=createHash('sha256');
  for(const x of choices){const raw=run(gexe,['--threads','1',...x.input]);assert.equal(raw,x.expected,`${mode} promotion choices ${x.input}`);ghash.update(raw);}
  for(const bad of badChoices){const r=invoke(gexe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid promotion-choice/);}
  modes.push({mode,board_rows:rows,board_fields:rows*19,invalid_board_requests:malformed.length,
   choice_requests:choices.length,choice_entries:choices.length*5,invalid_choice_requests:badChoices.length,
   output_sha256:h.digest('hex'),choices_sha256:ghash.digest('hex')});
  console.error(`PASS ${mode}: ${rows} full Board rows, ${choices.length} exact choice lists`);
 }
 const mutations=[];
 for(const [name,which,from,to] of [
  ['actual-ignores-promotion','probe.bend','+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)','+put = kind'],
  ['actual-forces-queen','probe.bend','+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)','+put = Bool.pick(U32, U32.is_zero(promotion), kind, 4)'],
  ['actual-duplicates-knight-choice','choices_probe.bend','Con{Ply{src, dst, 1, 0}, Con{Ply{src, dst, 2, 0},','Con{Ply{src, dst, 1, 0}, Con{Ply{src, dst, 1, 0},']
 ]){
  const copy=path.join(temp,name);fs.cpSync(engine,copy,{recursive:true});const chess=path.join(copy,'legal_probe/Chess.bend');
  const text=fs.readFileSync(chess,'utf8');assert.equal(text.split(from).length,2);fs.writeFileSync(chess,text.replace(from,to));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');
  run(process.execPath,[cli,path.join(copy,'standalone/proofs/promotion',which),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',mc,'-pthread','-lm','-o',exe]);
  let rejection;
  if(which==='probe.bend'){
   const batch=fixtures.slice(0,64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
   try{compare(raw,batch,name);}catch(e){rejection=e;}
  }else{
   const raw=run(exe,['--threads','1',...choices[0].input]);try{assert.equal(raw,choices[0].expected,name);}catch(e){rejection=e;}
  }
  assert.ok(rejection instanceof assert.AssertionError,`${name}: expected wrong-value rejection`);
  mutations.push({name,rejected:true,compiled_and_executed:true,message:rejection.message,actual:rejection.actual,expected:rejection.expected});
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,board_rows_per_mode:fixtures.length,
  distinct_board_inputs:distinct.size,board_fields_per_mode:fixtures.length*19,consistent_inputs:consistentInputs,
  cases_by_operation:counts,choice_requests_per_mode:choices.length,fixture_sha256:sha(JSON.stringify(fixtures)),
  modes,mutations,source_sha256s:before,cc:run(cc,['--version']).split('\n')[0],
  scope:'Actual flag-zero promotion updates and actual put_move four-choice lists with sentinel tail. All source/destination/choice combinations are raw updates, not legal chess moves. Complete Board comparison; no proof model, search, tables, or model executes in the candidate.'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
