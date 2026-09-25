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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-special-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Position.bend','standalone/Text.bend',
 'standalone/proofs/special_update/probe.bend','standalone/proofs/special_update/verify_native.js'];
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
function reference(a,meta,src,dst,promotion,flag){
 const kind=[0,1,2,3,4].find(k=>a[src].kinds.has(k))??5;
 const b=clone(a);const remove=new Set([src,dst]);if(flag===1)remove.add(dst^8);
 for(const sq of remove)b[sq]={kinds:new Set(),colors:new Set()};
 const white=meta[0]===1;put(b,dst,promotion||kind,white?1:0);
 const lost=(corner.get(src)||0)|(corner.get(dst)||0)|(kind===5?(white?3:12):0);
 const ep=kind===0&&((src^dst)===16)?Math.floor((src+dst)/2):64;
 return {board:b,meta:[(meta[0]^1)>>>0,(meta[1]&~lost)>>>0,ep]};
}
const fixtures=[],counts={};let validMoves=0,invalidBefore=0;
function add(category,a,meta,src,dst,promotion,flag){
 const expected=reference(a,meta,src,dst,promotion,flag);
 if(consistent(a)){assert.ok(consistent(expected.board));validMoves++;}else invalidBefore++;
 fixtures.push({category,input:[flag===1?1:0,src,dst,promotion,...encode(a,meta)],expected:encode(expected.board,expected.meta)});
 counts[category]=(counts[category]||0)+1;
}
// Geometrically promotion-shaped fixtures, not validated legal games.
for(let color=0;color<2;color++)for(let file=0;file<8;file++)for(let delta=-1;delta<=1;delta++){
 const df=file+delta;if(df<0||df>=8)continue;
 const src=(color?6:1)*8+file,dst=(color?7:0)*8+df;
 for(let p=1;p<=4;p++){
  const a=randomBoard();put(a,src,0,color);
  a[dst]={kinds:new Set(),colors:new Set()};if(delta)put(a,dst,3,1-color);
  add('promotion_shaped',a,[color,15,64],src,dst,p,0);
 }
}
for(let src=0;src<64;src++)for(let p=1;p<=4;p++)for(let color=0;color<2;color++){
 const dst=(src+17)%64,a=randomBoard();put(a,src,src%6,color);
 if(color)put(a,dst,(p+1)%6,1-color);else a[dst]={kinds:new Set(),colors:new Set()};
 add('promotion_raw_square_type_color',a,[color,random()%16,random()%65],src,dst,p,0);
}
for(let color=0;color<2;color++)for(let file=0;file<8;file++)for(const delta of [-1,1]){
 const df=file+delta;if(df<0||df>=8)continue;
 const src=(color?4:3)*8+file,dst=(color?5:2)*8+df,a=randomBoard();
 put(a,src,0,color);put(a,dst^8,0,1-color);a[dst]={kinds:new Set(),colors:new Set()};
 add('en_passant_shaped',a,[color,15,dst],src,dst,0,1);
}
for(let src=0;src<64;src++)for(let j=0;j<8;j++){
 const dst=(src+7*j)%64,a=randomBoard();put(a,src,j%6,j%2);
 add('en_passant_raw_positions',a,[j%2,random()%16,random()%65],src,dst,0,1);
}
for(let i=0;i<128;i++){
 const src=i%64,dst=(src+19)%64;
 add('promotion_raw_metadata',randomBoard(),[random(),random(),random()],src,dst,i%4+1,0);
 add('en_passant_raw_metadata',randomBoard(),[random(),random(),random()],src,dst,0,1);
}
for(let i=0;i<64;i++)for(let flag=0;flag<2;flag++){
 const a=randomBoard(),src=i,dst=(i+17)%64;
 a[(src+31)%64]={kinds:new Set([0,4]),colors:new Set([0,1])};
 add('inconsistent_input',a,[i%2,15,64],src,dst,flag?0:i%4+1,flag);
}
const malformed=[],base=fixtures[0].input;
for(const [i,value] of [[0,'2'],[1,'64'],[2,'64'],[3,'0'],[3,'5'],[4,'4294967296'],[4,'-1'],[4,'x']]){
 const row=base.map(String);row[i]=value;malformed.push(row);
}
const badEP=base.map(String);badEP[0]='1';malformed.push(badEP);
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
  for(const bad of malformed){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid special-update/);}
  modes.push({mode,rows,complete_board_fields:rows*19,invalid_rejections:malformed.length,output_sha256:hash.digest('hex')});console.error(`PASS ${mode}: ${rows} complete Board rows`);
 }
 const mutations=[];
 for(const [name,category,from,to] of [
  ['actual-promotion-ignored','promotion_shaped','+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)','+put = kind'],
  ['actual-en-passant-victim-at-destination','en_passant_shaped','cap = Bool.pick(U32, U32.is_eq(flag, 1), U32.xor(dst, 8), dst)','cap = dst']
 ]){
  const copy=path.join(temp,name);fs.cpSync(engine,copy,{recursive:true});const chess=path.join(copy,'legal_probe/Chess.bend');
  const source=fs.readFileSync(chess,'utf8');assert.equal(source.split(from).length,2);fs.writeFileSync(chess,source.replace(from,to));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');run(process.execPath,[cli,path.join(copy,'standalone/proofs/special_update/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',mc,'-pthread','-lm','-o',exe]);
  const batch=fixtures.filter(x=>x.category===category).slice(0,64),raw=run(exe,['--threads','1',...batch.flatMap(x=>x.input)]);
  let rejection;try{compare(raw,batch,name);}catch(e){rejection=e;}
  assert.ok(rejection instanceof assert.AssertionError,name+': must fail as a value mismatch');
  mutations.push({name,rejected:true,compiled_and_executed:true,category,message:rejection.message,actual:rejection.actual,expected:rejection.expected});
 }

 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,rows_per_mode:fixtures.length,complete_board_fields_per_mode:fixtures.length*19,
  cases_by_operation:counts,representation_consistent_move_inputs:validMoves,inconsistent_move_diagnostics:invalidBefore,
  malformed_requests_per_mode:malformed.length,fixture_sha256:sha(JSON.stringify(fixtures)),modes,mutations,source_sha256s:before,
  cc:run(cc,['--version']).split('\n')[0],
  scope:'Actual typed promotions (flag0) and en-passant updates (flag1/promotion0), including nonlegal raw calls; complete field agreement and partition, not legal-move generation, castling or independent metadata correctness. No proof model executes in the native candidate.'};
 const result=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],result);console.log(result.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
