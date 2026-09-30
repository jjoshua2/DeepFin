// Opt-in real fill-loop frame observations, not execution of proof predicates.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fill-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
function invoke(cmd,argv,timeout=120000){
 const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,2000));assert.equal(r.stderr,'');return r.stdout;}
const inputs=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
 'standalone/proofs/fill_interval/probe.bend','standalone/proofs/fill_interval/verify_native.js'];
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
let rng=0x798465de1985abcdn;
function random(){rng^=(rng<<13n)&all;rng^=rng>>7n;rng=(rng^(rng<<17n))&all;return rng;}
const fixtures=[];
function add(key,mode,code){fixtures.push({key,mode,code,seed:random()});}
for(let key=0;key<128;key++)for(const code of [0,1,2])add(key,0,code);
for(const key of [0,63,64,91,127])for(const mode of [1,2,3,4])for(const code of [3,4,5,6])add(key,mode,code);
// Force overlong loops to overwrite the exclusive endpoint, exposing a false frame premise.
for(const key of [0,63,64,91,127])add(key,4,1);
function scatter(i,b){return b.reduce((a,x,j)=>a|((BigInt(i)&(1n<<BigInt(j)))?x:0n),0n);}
let protectedRows=0,overlapRows=0,fullBlocks=0,zeroRows=0,overlongRows=0;
const expected=fixtures.map(({key,mode,code,seed},id)=>{
 const size=sizes[key],at=prefix[key],end=at+size;
 const n=[size,0,1,size-1,size+1][mode];
 const q=[at-1,end,at+size/2,end-1,0,131071,end+1][code];
 if(mode===0)fullBlocks++;if(n===0)zeroRows++;if(n>size)overlongRows++;
 const overwritten=x=>at<=x&&x<at+n;
 const actualAttack=x=>attack(key,scatter((x-at)%size,selected[key]));
 const after=overwritten(q)?actualAttack(q):seed;
 if(overwritten(q)){overlapRows++;assert.notEqual(after,seed);}else{protectedRows++;assert.equal(after,seed);}
 const first=overwritten(at)?actualAttack(at):(q===at?seed:attack(key,0n));
 const last=overwritten(end-1)?actualAttack(end-1):(q===end-1?seed:attack(key,all));
 return [id,at,size,n,q,...halves(seed),...halves(after),...halves(first),...halves(last),131072].join(' ');
}).join('\n')+'\n';
const operands=fixtures.flatMap(({key,mode,code,seed})=>[String(key),String(mode),String(code),...halves(seed)]);
function compare(actual,wanted){const a=actual.trimEnd().split('\n'),b=wanted.trimEnd().split('\n');assert.equal(a.length,b.length,'row count');a.forEach((s,i)=>assert.equal(s,b[i],`fill interval row ${i}`));}
assert.throws(()=>compare(expected.replace(/^0 /,'1 '),expected),/row 0/);
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const actual=run(exe,['--threads','1',...operands]);compare(actual,expected);
  const invalid=[['128','0','0','0','0'],['0','5','0','0','0'],['0','0','7','0','0'],
   ['0','0','0','4294967296','0'],['0','0','0','0','-1'],['x','0','0','0','0'],['0','0','0']];
  for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid fill interval|expected five fill interval/);}
  modes.push({mode,rows:fixtures.length,invalid_rejections:invalid.length,output_sha256:sha(actual)});
  console.error(`PASS ${mode}: ${fixtures.length} actual fill/query observations`);
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:fixtures.length,chess_keys:128,
  full_block_rows:fullBlocks,zero_count_rows:zeroRows,overlong_count_rows:overlongRows,
  protected_query_rows:protectedRows,overwritten_query_rows:overlapRows,allocation_cells:131072,
  exclusive_logical_end:prefix[128],
  observation:'Actual table mask/prefix headers and complete fill loops; query, first and last values and capacity, not every array cell',
  oracle:'Independent signed-coordinate rays, relevant masks, direct BigInt scatter and ordinary integer prefix/address sums',
  scope:'Real Tables.build supplies metadata. Query is poisoned before the actual fill. Includes zero/partial/full/overlong loops; no proof predicates executed. Stored metadata and final full geometric lookup still require source refinement.',
  modes,source_sha256s:before,bun:process.versions.bun,cc:run(cc,['--version']).split('\n')[0]};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
