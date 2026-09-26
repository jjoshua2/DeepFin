// Opt-in actual flag-one execution. The reference is outside the candidate.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-en-passant-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Position.bend','standalone/Text.bend',
 'standalone/proofs/en_passant/probe.bend','standalone/proofs/en_passant/verify_native.js'];
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
const corner=new Map([[0,2],[7,1],[56,8],[63,4]]);
function reference(a,meta,src,dst){
 // Square sets model the raw update. Metadata mirrors current API behavior,
 // not independently justified chess metadata or legal en-passant rights.
 const originalKind=[0,1,2,3,4].find(k=>a[src].kinds.has(k))??5;
 const out=clone(a);
 // Adjacent rank, same file: independent bounded-coordinate capture location.
 const rank=Math.floor(dst/8),captured=(rank%2===0?rank+1:rank-1)*8+dst%8;
 for(const sq of [src,dst,captured])out[sq]={kinds:new Set(),colors:new Set()};
 const white=meta[0]===1;put(out,dst,originalKind,white?1:0);
 const lost=(corner.get(src)||0)|(corner.get(dst)||0)|(originalKind===5?(white?3:12):0);
 const ep=originalKind===0&&((src^dst)===16)?Math.floor((src+dst)/2):64;
 return {board:out,meta:[(meta[0]^1)>>>0,(meta[1]&~lost)>>>0,ep]};
}
const fixtures=[],counts={all_raw_pairs:0,pawn_coordinate_geometries:0,raw_metadata:0,inconsistent_diagnostic:0};
let consistentInputs=0;
function add(category,a,meta,src,dst){
 const out=reference(a,meta,src,dst);
 if(consistent(a)){assert.ok(consistent(out.board));consistentInputs++;}
 fixtures.push({category,input:[0,src,dst,0,...encode(a,meta)],expected:encode(out.board,out.meta)});counts[category]++;
}
for(let src=0;src<64;src++)for(let dst=0;dst<64;dst++){
 const a=empty(),side=(src^dst)&1;put(a,dst^8,0,1-side);put(a,src,(src+dst)%6,side);
 add('all_raw_pairs',a,[side,15,64],src,dst);
}
for(const side of [0,1])for(let file=0;file<8;file++)for(const df of [-1,1]){
 if(file+df<0||file+df>=8)continue;
 const rank=side?4:3,src=rank*8+file,dst=(rank+(side?1:-1))*8+file+df;
 const a=empty();put(a,src,0,side);put(a,dst^8,0,1-side);put(a,0,3,1-side);put(a,63,5,side);
 add('pawn_coordinate_geometries',a,[side,15,dst],src,dst);
}
for(let i=0;i<64;i++){
 const a=empty(),dst=(i+9)%64;put(a,i,0,i%2);put(a,dst^8,2,1-i%2);
 add('raw_metadata',a,[(i*1029933+13)>>>0,(i*93853)>>>0,i],i,dst);
}
for(let i=0;i<128;i++){
 const a=empty(),src=i%64,dst=(src+17)%64;put(a,src,0,i%2);
 a[(src+5)%64]={kinds:new Set([0,3]),colors:new Set([0,1])};
 add('inconsistent_diagnostic',a,[i%2,15,64],src,dst);
}
assert.equal(fixtures.length,4316);assert.equal(consistentInputs,4188);
const distinct=new Set(fixtures.map(x=>JSON.stringify(x.input)));
const malformed=[],base=fixtures[0].input;
for(const [i,val] of [[0,'1'],[1,'64'],[2,'64'],[3,'1'],[4,'-1'],[4,'4294967296'],[4,'x']]){
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
try{
 const cli=path.join(compiler,'bend2/main.ts'),c=path.join(temp,'probe.c');
 run(process.execPath,[cli,path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const h=createHash('sha256');let rows=0;
  for(let i=0;i<fixtures.length;i+=64){
   const batch=fixtures.slice(i,i+64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
   compare(raw,batch,`${mode} batch ${i}`);h.update(raw);rows+=batch.length;
  }
  for(const bad of malformed){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid en-passant/);}
  modes.push({mode,board_rows:rows,board_fields:rows*19,invalid_rejections:malformed.length,output_sha256:h.digest('hex')});
  console.error(`PASS ${mode}: ${rows} complete Boards and ${malformed.length} malformed requests`);
 }
 const mutations=[];
 for(const [name,to] of [['actual-omits-remote-capture','dst'],['actual-captures-wrong-rank','U32.xor(dst,16)']]){
  const copy=path.join(temp,name);fs.cpSync(engine,copy,{recursive:true});const chess=path.join(copy,'legal_probe/Chess.bend');
  const text=fs.readFileSync(chess,'utf8'),from='cap = Bool.pick(U32, U32.is_eq(flag, 1), U32.xor(dst, 8), dst)';
  assert.equal(text.split(from).length,2);fs.writeFileSync(chess,text.replace(from,`cap = Bool.pick(U32, U32.is_eq(flag, 1), ${to}, dst)`));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');
  run(process.execPath,[cli,path.join(copy,'standalone/proofs/en_passant/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',mc,'-pthread','-lm','-o',exe]);
  const batch=fixtures.slice(0,64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
  let rejection;try{compare(raw,batch,name);}catch(e){rejection=e;}
  assert.ok(rejection instanceof assert.AssertionError,`${name}: expected wrong-value rejection`);
  mutations.push({name,rejected:true,compiled_and_executed:true,message:rejection.message,actual:rejection.actual,expected:rejection.expected});
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,board_rows_per_mode:fixtures.length,
  distinct_input_pairs:distinct.size,board_fields_per_mode:fixtures.length*19,consistent_input_cases:consistentInputs,
  cases_by_operation:counts,fixture_sha256:sha(JSON.stringify(fixtures)),modes,mutations,source_sha256s:before,
  cc:run(cc,['--version']).split('\n')[0],
  scope:'Actual flag1/promotion0 Board updates; all raw source/destination pairs and selected metadata/inconsistent diagnostics. Not legal EP, board reachability, or a metadata-rule theorem. Native squares are 0..63; modes repeat fixtures.'};
 const encoded=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],encoded);console.log(encoded.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
