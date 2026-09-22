// Opt-in end-to-end address observations. The candidate reads real table metadata.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-relative-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
function invoke(cmd,argv,timeout=120000){
 const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,2000));assert.equal(r.stderr,'');return r.stdout;}
const inputs=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
 'bitboard_probe/Sliders.bend','standalone/proofs/relative/probe.bend','standalone/proofs/relative/verify_native.js'];
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
function gather(v,bs){return Number(bs.reduce((a,b,i)=>a|((v&b)?1n<<BigInt(i):0n),0n));}
function attack(k,occ){let a=0n;for(const ray of rays(k))for(const s of ray){const b=1n<<BigInt(s);a|=b;if(occ&b)break;}return a;}
let rng=0x798465de1985abcdn;
function random(){rng^=(rng<<13n)&all;rng^=rng>>7n;rng=(rng^(rng<<17n))&all;return rng;}
const fixtures=[];
function add(i,j,oi,oj,v){fixtures.push({i,j,oi,oj,v});}
// Every key is observed at both zero and maximum compact index; both orderings.
for(let k=0;k<128;k++){
 const j=(k+1)%128;
 add(k,j,0n,all,random());
 add(k,j,all,0n,random());
}
// Same-key writes are deliberate outside-domain controls for cross-block framing.
for(let k=0;k<16;k++){add(k,k,random(),random(),random());add(k,k,0n,0n,1n<<63n);}
assert.equal(fixtures.length,288);
let protectedRows=0,sameAddressRows=0,orderedRows=0;
const expected=fixtures.map(({i,j,oi,oj,v},id)=>{
 const ri=gather(oi,selected[i]),rj=gather(oj,selected[j]),ai=prefix[i]+ri,aj=prefix[j]+rj;
 assert.ok(prefix[i]<=ai&&ai<prefix[i+1]&&ai<131072);
 assert.ok(prefix[j]<=aj&&aj<prefix[j+1]&&aj<131072);
 assert.equal(ai&131071,ai);assert.equal(aj&131071,aj);
 if(i<j){assert.ok(ai<aj);orderedRows++;}
 const old=attack(j,oj),after=ai===aj?v:old;
 if(ai===aj)sameAddressRows++;else protectedRows++;
 return [id,prefix[i],ri,ai,prefix[j],rj,aj,...halves(old),...halves(after),...halves(v),131072].join(' ');
}).join('\n')+'\n';
const operands=fixtures.flatMap(({i,j,oi,oj,v})=>[String(i),String(j),...halves(oi),...halves(oj),...halves(v)]);
function compare(actual,wanted){const a=actual.trimEnd().split('\n'),b=wanted.trimEnd().split('\n');assert.equal(a.length,b.length,'row count');a.forEach((s,i)=>assert.equal(s,b[i],`relative address row ${i}`));}
assert.throws(()=>compare(expected.replace(/^0 /,'1 '),expected),/row 0/);
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const actual=run(exe,['--threads','1',...operands]);compare(actual,expected);
  const invalid=[['128','0','0','0','0','0','0','0'],['0','128','0','0','0','0','0','0'],
   ['0','0','4294967296','0','0','0','0','0'],['0','0','0','0','-1','0','0','0'],
   ['x','0','0','0','0','0','0','0'],['0','0','0']];
  for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid relative address|expected eight relative address/);}
  modes.push({mode,rows:fixtures.length,invalid_rejections:invalid.length,output_sha256:sha(actual)});
  console.error(`PASS ${mode}: ${fixtures.length} actual table/address/read-write observations`);
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:fixtures.length,chess_keys:128,
  cross_block_rows:256,ordered_cross_block_rows:orderedRows,same_block_rows:32,protected_query_rows:protectedRows,
  same_address_rows:sameAddressRows,allocation_cells:131072,exclusive_logical_end:prefix[128],
  observation:'Actual table metadata, PEXT indices, U32 addresses and public read/write returned values; not every array cell',
  oracle:'Independent signed-coordinate rays, relevant masks, BigInt gather and ordinary integer prefix sums; candidate receives no expected tables or indices',
  scope:'Fresh actual Tables.build per bounded request. Source prefix correspondence to stored header values and final computed contents remain separate proof obligations.',
  modes,source_sha256s:before,bun:process.versions.bun,cc:run(cc,['--version']).split('\n')[0]};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
