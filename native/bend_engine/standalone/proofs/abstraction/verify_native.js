// Actual complete-board observations; the structural snapshot is source-only.
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
const engine=path.resolve(suite,'../../..'),cc=process.env.CC||'clang',temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-whole-board-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const inputs=['legal_probe/Chess.bend','standalone/Position.bend','standalone/Text.bend','bitboard_probe/Sliders.bend',
 'standalone/proofs/abstraction/probe.bend','standalone/proofs/abstraction/verify_native.js'];
const hashes=()=>Object.fromEntries(inputs.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))])),before=hashes();
function invoke(cmd,argv){const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout:120000,maxBuffer:32<<20,
 env:{...process.env,BEND_NO_TELEMETRY:'1',TERM:'dumb'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stdout+r.stderr).slice(-2000));assert.equal(r.stderr,'');return r.stdout;}
let rng=0x49b30adc;
function random(){rng^=rng<<13;rng^=rng>>>17;rng^=rng<<5;return rng>>>0;}
function board(){return Array.from({length:64},()=>random()%5===0?null:[random()%6,random()%2]);}
function encode(cells,meta){const planes=Array(8).fill(0n);cells.forEach((c,i)=>{if(c){const bit=1n<<BigInt(i);planes[c[0]]|=bit;planes[c[1]===1?6:7]|=bit;}});
 return [...planes.flatMap(x=>[Number(x>>32n),Number(x&0xffffffffn)]),...meta];}
function start(){const b=Array(64).fill(null),back=[3,1,2,4,5,2,1,3];for(let i=0;i<8;i++){b[i]=[back[i],1];b[i+8]=[0,1];b[i+48]=[0,0];b[i+56]=[back[i],0];}return b;}
const fixtures=[],counts={fresh_insert:0,random_board:0,metadata:0,empty:0,initial:0};
function add(category,op,kind,color,sq,cells,meta){let after=cells.map(x=>x&&x.slice()),outmeta=meta;
 if(op===0){assert.equal(cells[sq],null);after[sq]=[kind,color];}
 if(op===1)outmeta=[kind,color,sq];
 if(op===2){after=Array(64).fill(null);outmeta=[1,0,64];}
 if(op===3){after=start();outmeta=[1,15,64];}
 const raw=encode(after,outmeta),observations=after.flatMap(c=>c||[6,0]);
 fixtures.push({category,input:[op,kind,color,sq,...encode(cells,meta)],expected:[...raw,...observations]});counts[category]++;}
for(let sq=0;sq<64;sq++)for(let kind=0;kind<6;kind++)for(let color=0;color<2;color++){
 const b=board();b[sq]=null;add('fresh_insert',0,kind,color,sq,b,[random(),random(),random()]);}
for(let i=0;i<192;i++)add('random_board',4,0,0,0,board(),[random(),random(),random()]);
for(let i=0;i<64;i++)add('metadata',1,random(),random(),random(),board(),[random(),random(),random()]);
add('empty',2,0,0,0,board(),[random(),random(),random()]);add('initial',3,0,0,0,board(),[random(),random(),random()]);
assert.equal(fixtures.length,1026);
function compare(raw,rows,offset){const lines=raw.trimEnd().split('\n');assert.equal(lines.length,rows.length,'row count');
 lines.forEach((line,i)=>{const got=line.split(' ').map(Number);assert.equal(got.length,147,`row ${offset+i}: all64 squares and all19 raw fields required`);
 got.forEach((v,j)=>assert.equal(v,rows[i].expected[j],`row ${offset+i}, field ${j}: ${rows[i].category}`));
 const cells=Array.from({length:64},(_,q)=>{const k=got[19+2*q],c=got[20+2*q];return k===6?null:[k,c];});
 assert.deepEqual(encode(cells,got.slice(16,19)),got.slice(0,19),'all observed abstract squares reconstruct complete Board');});}
function execute(exe){let output='';for(let i=0;i<fixtures.length;i+=32){const f=fixtures.slice(i,i+32),raw=run(exe,['--threads','1',...f.flatMap(x=>x.input)]);compare(raw,f,i);output+=raw;}return output;}
const base=[0,0,0,0,...encode(Array(64).fill(null),[1,0,64])];
const alter=(i,x)=>base.map((v,j)=>i===j?x:v);
const invalid=[alter(0,5),alter(1,6),alter(2,2),alter(3,64),alter(4,'4294967296'),alter(4,'-1'),alter(4,'x'),base.slice(0,-1),Array.from({length:65},()=>base).flat()];
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const out=execute(exe);for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid snapshot/);}
  modes.push({mode,boards:fixtures.length,square_observations:fixtures.length*64,output_fields:fixtures.length*147,invalid_rejections:invalid.length,output_sha256:sha(out)});
  console.error(`PASS ${mode}: ${fixtures.length} complete boards, ${fixtures.length*64} squares`);
 }
 const mutations=[];
 for(const [name,file,from,to] of [
  ['actual-wrong-pawn-tag','legal_probe/Chess.bend','U64.test_bit(get_pawns(b), U32.to_nat(sq)), 0,','U64.test_bit(get_pawns(b), U32.to_nat(sq)), 1,'],
  ['skipped-observation-square','standalone/proofs/abstraction/probe.bend','squares(p,U32.inc(sq),b)','squares(p,U32.add(sq,2),b)']]){
  const m=path.join(temp,name);fs.cpSync(engine,m,{recursive:true});const f=path.join(m,file),text=fs.readFileSync(f,'utf8');assert.equal(text.split(from).length,2);fs.writeFileSync(f,text.replace(from,to));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');
  run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(m,'standalone/proofs/abstraction/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1',mc,'-pthread','-lm','-o',exe]);
  const chunk=fixtures.slice(0,32),raw=run(exe,['--threads','1',...chunk.flatMap(x=>x.input)]);let diagnostic='';
  try{compare(raw,chunk,0);}catch(e){assert.ok(e instanceof assert.AssertionError);diagnostic=e.message;}
  assert.match(diagnostic,/row \d+, field \d+/);mutations.push({name,rejected:true,compiled_and_executed:true,diagnostic});
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const result={native_gate:'PASS',compiler_revision:PIN.revision,...identity,boards_per_mode:1026,squares_per_mode:65664,
  output_fields_per_mode:150822,cases:counts,modes,mutations,source_sha256s:before,fixture_sha256:sha(JSON.stringify(fixtures)),
  cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
  scope:'Actual Board operations and all64 guarded decoder observations, externally reconstructed against independent square arrays. Structural Spec.observe/restore are source-proved but not lowered/executed here; their direct probe hits the pinned compiler arity limit. No unrestricted snapshot bijection, parser freshness, move legality or perft claim.'};
 const text=JSON.stringify(result,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
