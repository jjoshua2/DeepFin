// Actual occupied/piece/color reads versus an independent 64-square reference.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-decoder-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['legal_probe/Chess.bend','standalone/Position.bend','standalone/Text.bend',
  'bitboard_probe/Sliders.bend','standalone/proofs/decoder/probe.bend','standalone/proofs/decoder/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(command,argv,timeout=90000){
  const r=spawnSync(command,argv.map(String),{encoding:'utf8',timeout,maxBuffer:32<<20,
    env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;
}
function run(command,argv){const r=invoke(command,argv);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-3000));
  assert.equal(r.stderr,'','unexpected native/tool diagnostic');return r.stdout;}
let rng=0x729cf150;
function random(){rng^=rng<<13;rng^=rng>>>17;rng^=rng<<5;return rng>>>0;}
function randomBoard(){return Array.from({length:64},()=>random()%4===0?null:[random()%6,random()%2]);}
function encode(cells,metadata){
  const bits=Array(8).fill(0n);
  cells.forEach((cell,sq)=>{if(cell){const bit=1n<<BigInt(sq);bits[cell[0]]|=bit;bits[cell[1]===1?6:7]|=bit;}});
  return [...bits.flatMap(x=>[Number(x>>32n),Number(x&0xffffffffn)]),...metadata];
}
function initial(){const board=Array(64).fill(null),back=[3,1,2,4,5,2,1,3];
  for(let f=0;f<8;f++){board[f]=[back[f],1];board[f+8]=[0,1];board[f+48]=[0,0];board[f+56]=[back[f],0];}return board;}
const fixtures=[],counts={all_square_states:0,insert_then_decode:0,initial:0,random_valid:0};
let occupiedRows=0,emptyRows=0;
function add(category,op,kind,color,sq,cells){
  const after=cells.slice();if(op===1){assert.equal(cells[sq],null,'fresh insertion fixture');after[sq]=[kind,color];}
  const reference=op===2?initial():after,cell=reference[sq];
  const row=Array(8).fill(0);if(cell){row[cell[0]]=1;row[cell[1]===1?6:7]=1;occupiedRows++;}else emptyRows++;
  const expected=[sq,cell?1:0,cell?cell[0]:5,...row];
  fixtures.push({category,input:[op,kind,color,sq,...encode(cells,[random(),random(),random()])],expected});counts[category]++;
}
for(let sq=0;sq<64;sq++)for(let state=0;state<13;state++){
  const b=randomBoard();b[sq]=state===0?null:[Math.floor((state-1)/2),(state-1)%2];add('all_square_states',0,0,0,sq,b);
}
for(let sq=0;sq<64;sq++)for(let kind=0;kind<6;kind++)for(let color=0;color<2;color++){
  const b=randomBoard();b[sq]=null;add('insert_then_decode',1,kind,color,sq,b);
}
for(let sq=0;sq<64;sq++)add('initial',2,0,0,sq,Array(64).fill(null));
for(let n=0;n<256;n++)add('random_valid',0,0,0,random()%64,randomBoard());
assert.equal(fixtures.length,1920);assert.equal(counts.all_square_states,832);assert.equal(counts.insert_then_decode,768);
assert.ok(occupiedRows>0&&emptyRows>0);
function compare(raw,cases,offset){
  const lines=raw.trimEnd().split('\n');assert.equal(lines.length,cases.length,'row count');
  lines.forEach((line,i)=>{
    const got=line.split(' ').map(Number);assert.equal(got.length,11,'complete square observation');
    got.forEach((v,j)=>assert.equal(v,cases[i].expected[j],`row ${offset+i}, field ${j}: ${cases[i].category}`));
    if(got[1]){
      const reconstructed=Array(8).fill(0);reconstructed[got[2]]=1;reconstructed[got[9]===1?6:7]=1;
      assert.deepEqual(got.slice(3),reconstructed,'decoded occupied square reconstructs all eight plane bits');
    }else assert.deepEqual(got.slice(3),Array(8).fill(0),'empty square has no kind/color bits');
  });
}
function execute(exe){let out='';for(let i=0;i<fixtures.length;i+=64){const chunk=fixtures.slice(i,i+64);
  const raw=run(exe,['--threads','1',...chunk.flatMap(f=>f.input)]);compare(raw,chunk,i);out+=raw;}return out;}
const base=[1,0,0,0,...encode(Array(64).fill(null),[1,0,64])];
const altered=(i,x)=>base.map((v,j)=>j===i?x:v);
const invalid=[altered(0,3),altered(1,6),altered(2,2),altered(3,64),altered(4,'4294967296'),
  altered(4,'-1'),altered(4,'x'),base.slice(0,-1),Array.from({length:65},()=>base).flat()];
try{
  const generated=path.join(temp,'probe.c');
  run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',generated]);
  const modes=[];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],
    ['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,generated,'-pthread','-lm','-o',exe]);
    const raw=execute(exe);
    for(const bad of invalid){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid decoder/);}
    modes.push({mode,rows:fixtures.length,fields:fixtures.length*11,invalid_rejections:invalid.length,output_sha256:sha(raw)});
    console.error(`PASS native ${mode}: ${fixtures.length} square observations`);
  }
  const mutations=[];
  for(const [name,from,to] of [
    ['pawn-tag','U64.test_bit(get_pawns(b), U32.to_nat(sq)), 0,','U64.test_bit(get_pawns(b), U32.to_nat(sq)), 1,'],
    ['king-fallback','U64.test_bit(get_queens(b), U32.to_nat(sq)), 4, 5)','U64.test_bit(get_queens(b), U32.to_nat(sq)), 4, 4)'],
    ['occupied-intersection','U64.or(get_white(b), get_black(b))','U64.and(get_white(b), get_black(b))']]){
    const m=path.join(temp,name+'-source');fs.cpSync(engine,m,{recursive:true});
    const f=path.join(m,'legal_probe/Chess.bend'),s=fs.readFileSync(f,'utf8');assert.equal(s.split(from).length,2);
    fs.writeFileSync(f,s.replace(from,to));const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'-mutant');
    run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(m,'standalone/proofs/decoder/probe.bend'),'-o',mc]);
    run(cc,['-std=c11','-O1',mc,'-pthread','-lm','-o',exe]);
    // Restrict regression witnesses to occupied inputs inside the raw-decoder theorem.
    const chunk=fixtures.filter(f=>f.category==='all_square_states'&&f.expected[1]===1).slice(0,64);
    const raw=run(exe,['--threads','1',...chunk.flatMap(f=>f.input)]);let diagnostic='';
    try{compare(raw,chunk,0);}catch(error){assert.ok(error instanceof assert.AssertionError);diagnostic=error.message;}
    assert.match(diagnostic,/row \d+, field \d+: all_square_states/);
    mutations.push({name,rejected:true,compiled_and_executed:true,occupied_inputs_only:true,diagnostic,
      scope:'first generic-mode mismatch, not four-mode mutated qualification'});
  }
  assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,rows_per_mode:fixtures.length,
    square_observation_fields_per_mode:fixtures.length*11,cases_by_category:counts,occupied_rows:occupiedRows,empty_rows:emptyRows,
    all_square_state_combinations:832,all_fresh_insert_combinations:768,modes,mutations,source_sha256s:before,
    fixture_sha256:sha(JSON.stringify(fixtures)),cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
    oracle:'Independent 64-square optional (piece-kind,color) values. Candidate uses real Chess.occupied/piece/getters, Position.put/start. No proof predicates or answer labels execute in candidate.',
    scope:'Pointwise decoder observations on consistent boards, including raw empty fallback as an explicitly outside-occupied-domain observation. Not legal-game, full-board roundtrip, parser or move qualification.'};
  const out=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
