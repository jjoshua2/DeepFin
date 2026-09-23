// Actual full-block fill frames. The candidate reads its metadata from Tables.build.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const root=path.resolve(suite,'../..'),engine=path.dirname(root),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fill-frame-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
function invoke(cmd,argv,timeout=120000){
 const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,2000));assert.equal(r.stderr,'');return r.stdout;}
const inputs=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
 'standalone/proofs/fill_frame/probe.bend','standalone/proofs/fill_frame/verify_native.js'];
const hashes=()=>Object.fromEntries(inputs.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes(),all=(1n<<64n)-1n;
const halves=x=>[String(x>>32n),String(x&0xffffffffn)];
function rays(key){
 const sq=key%64,x=sq%8,y=Math.floor(sq/8),dirs=key<64?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,1],[1,-1],[-1,1],[-1,-1]];
 return dirs.map(([dx,dy])=>{const out=[];for(let f=x+dx,r=y+dy;f>=0&&f<8&&r>=0&&r<8;f+=dx,r+=dy)out.push(r*8+f);return out;});
}
const masks=Array.from({length:128},(_,k)=>rays(k).flatMap(r=>r.slice(0,-1)).reduce((a,s)=>a|(1n<<BigInt(s)),0n));
const bits=m=>Array.from({length:64},(_,i)=>1n<<BigInt(i)).filter(b=>(m&b)!==0n);
const selected=masks.map(bits),sizes=selected.map(b=>2**b.length),prefix=[512];
for(const size of sizes)prefix.push(prefix.at(-1)+size);
assert.equal(prefix[128],108160);
function attack(k,occ){let a=0n;for(const ray of rays(k))for(const s of ray){const b=1n<<BigInt(s);a|=b;if(occ&b)break;}return a;}
function scatter(v,bs){return bs.reduce((a,b,i)=>a|((v&(1n<<BigInt(i)))?b:0n),0n);}
let rng=0x87ed83219ac4b5d1n;
function random(){rng^=(rng<<13n)&all;rng^=rng>>7n;rng=(rng^(rng<<17n))&all;return rng;}
const fixtures=[];
for(let k=0;k<128;k++){
 for(const q of [prefix[k]-1,prefix[k+1],0,131071,prefix[k],prefix[k+1]-1])fixtures.push({k,mode:0,q,seed:random()});
 fixtures.push({k,mode:1,q:prefix[k]+1,seed:random()});
 fixtures.push({k,mode:2,q:prefix[k],seed:random()});
}
assert.equal(fixtures.length,1024);
let protectedRows=0,overwrittenRows=0,totalWrites=0;
const expected=fixtures.map(({k,mode,q,seed},id)=>{
 const count=mode===0?sizes[k]:mode===1?1:0,at=prefix[k];totalWrites+=count;
 const value=i=>i>=at&&i<at+count?attack(k,scatter(BigInt(i-at),selected[k])):seed;
 const after=value(q);
 if(q<at||q>=at+count){assert.equal(after,seed);protectedRows++;}else{assert.notEqual(after,seed);overwrittenRows++;}
 return [id,k,at,sizes[k],count,q,...halves(masks[k]),...halves(seed),...halves(after),...halves(value(at)),...halves(value(prefix[k+1]-1)),131072].join(' ');
}).join('\n')+'\n';
const operands=fixtures.flatMap(({k,mode,q,seed})=>[String(k),String(mode),String(q),...halves(seed)]);
function compare(actual,wanted){const a=actual.trimEnd().split('\n'),b=wanted.trimEnd().split('\n');assert.equal(a.length,b.length,'row count');a.forEach((s,i)=>assert.equal(s,b[i],`fill frame row ${i}`));}
assert.throws(()=>compare(expected.replace(/^0 /,'1 '),expected),/row 0/);
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const actual=run(exe,['--threads','1',...operands]);compare(actual,expected);
  const invalid=[['128','0','0','0','0'],['0','3','0','0','0'],['0','0','131072','0','0'],
   ['0','0','0','4294967296','0'],['0','0','0','-1','0'],['x','0','0','0','0'],['0','0']];
  for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid fill frame|expected five fill frame/);}
  modes.push({mode,rows:fixtures.length,invalid_rejections:invalid.length,output_sha256:sha(actual)});
  console.error(`PASS ${mode}: ${fixtures.length} actual fill/query observations`);
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:fixtures.length,chess_keys:128,
  full_block_rows:768,one_write_rows:128,zero_write_rows:128,protected_query_rows:protectedRows,
  overwritten_query_rows:overwrittenRows,actual_fill_writes_per_mode:totalWrites,allocation_cells:131072,exclusive_logical_end:prefix[128],
  observation:'Actual table metadata, full/one/zero actual fill, before/after query and first/last block values, capacity; not every returned array cell',
  oracle:'Independent signed-coordinate masks/rays, BigInt deposition and integer prefix sums. Candidate receives only key/mode/query/seed, not expected masks, offsets or values',
  scope:'Full fills for every chess block. Source theorem concerns explicit certified prefixes; complete stored-header/final-table lookup refinement is separate.',
  modes,source_sha256s:before,bun:process.versions.bun,cc:run(cc,['--version']).split('\n')[0]};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
