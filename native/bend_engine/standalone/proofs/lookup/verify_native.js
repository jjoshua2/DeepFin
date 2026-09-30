// Real full builder + repeated actual lookup; reference values are never candidate inputs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),engine=path.resolve(import.meta.dirname,'../../..');
const suite=import.meta.dirname,cc=process.env.CC||'clang',cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-lookup-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),max=(1n<<64n)-1n;
const tracked=['legal_probe/Chess.bend','bitboard_probe/Sliders.bend','standalone/Tables.bend','standalone/Subsets.bend',
  'standalone/Text.bend','standalone/proofs/lookup/probe.bend','standalone/proofs/lookup/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(command,argv){const r=spawnSync(command,argv,{encoding:'utf8',timeout:180000,maxBuffer:32<<20,
  env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(command,argv){const r=invoke(command,argv);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-3000));assert.equal(r.stderr,'','unexpected diagnostics');return r.stdout;}
let seed=0x12fedcab89127345n;
function random(){seed^=(seed<<13n)&max;seed^=seed>>7n;seed^=(seed<<17n)&max;seed&=max;return seed;}
function attacks(key,occ){const square=key%64,x=square%8,y=Math.floor(square/8);
  const directions=key>=64?[[1,1],[1,-1],[-1,1],[-1,-1]]:[[1,0],[-1,0],[0,1],[0,-1]];
  let result=0n;
  for(const [dx,dy] of directions){for(let file=x+dx,rank=y+dy;file>=0&&file<8&&rank>=0&&rank<8;file+=dx,rank+=dy){
    const bit=1n<<BigInt(rank*8+file);result|=bit;if(occ&bit)break;}}
  return result;}
function compare(actual,wanted){const lines=actual.trimEnd().split('\n');assert.equal(lines.length,wanted.length);
  for(let i=0;i<lines.length;i++)assert.equal(lines[i],wanted[i],'lookup row '+i);}
try{
  const fixtures=[];
  for(let key=0;key<128;key++)for(const occ of [0n,max,1n<<63n,1n<<BigInt(key%64),0xaaaaaaaaaaaaaaaan,0x5555555555555555n,random(),random()])fixtures.push([key,occ]);
  assert.equal(fixtures.length,1024);
  const expected=fixtures.map(([k,v])=>{const a=attacks(k,v);return `${k} ${a>>32n} ${a&0xffffffffn}`;});
  const inputs=fixtures.flatMap(([k,v])=>[String(k),String(v>>32n),String(v&0xffffffffn)]);
  assert.throws(()=>compare(expected.join('\n').replace(/^0 /,'1 '),expected),/lookup row 0/);
  const c=path.join(temp,'probe.c');run(process.execPath,[cli,path.join(suite,'probe.bend'),'-o',c]);
  const modes=[];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
    const output=run(exe,['--threads','1',...inputs]);compare(output,expected);
    const invalid=[['128','0','0'],['0','4294967296','0'],['x','0','0'],['0','0'],['0','0','-1'],Array(1025).fill(['0','0','0']).flat()];
    for(const bad of invalid){const r=invoke(exe,['--threads','1',...bad]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid lookup|lookup input budget/);}
    modes.push({mode,rows:fixtures.length,invalid_rejections:invalid.length,output_sha256:sha(output)});
    console.error('PASS native '+mode+': 1024 queries, six invalid requests');
  }
  const mutant=path.join(temp,'mutant');fs.cpSync(engine,mutant,{recursive:true});
  const chess=path.join(mutant,'legal_probe/Chess.bend'),original=fs.readFileSync(chess,'utf8');
  const needle='Sliders.lookup(table, U64.low(offset), index)';assert.equal(original.split(needle).length,2);
  fs.writeFileSync(chess,original.replace(needle,'Sliders.lookup(table, U32.inc(U64.low(offset)), index)'));
  const mc=path.join(temp,'mutant.c'),me=path.join(temp,'mutant-bin');
  run(process.execPath,[cli,path.join(mutant,'standalone/proofs/lookup/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',mc,'-pthread','-lm','-o',me]);
  const bad=run(me,['--threads','1',...inputs]);
  assert.throws(()=>compare(bad,expected),/lookup row 0/);
  assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,rows_per_mode:1024,
    distinct_input_pairs:new Set(fixtures.map(([k,v])=>k+':'+v)).size,all_keys:128,modes,
    actual_lookup_mutation:{rejected:true,kind:'generic wrong-value comparison after successful compilation and execution',row:0,
      observed:bad.split('\n')[0],expected:expected[0],mutated_source_sha256:sha(fs.readFileSync(chess))},
    source_sha256s:before,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
    scope:'Actual Tables.build and threaded Chess.slide. Selected queries, not full-buffer comparisons, exhaustive occupancies or independent source geometry.'};
  const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
