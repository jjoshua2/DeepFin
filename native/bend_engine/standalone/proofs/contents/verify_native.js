// Opt-in computed-entry tests. Only real Tables/Base code runs in the probe.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fill-contents-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),max=(1n<<64n)-1n;
const halves=x=>[String(x>>32n),String(x&0xffffffffn)];
const inputs=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
 'standalone/proofs/contents/probe.bend','standalone/proofs/contents/verify_native.js'];
const hashes=()=>Object.fromEntries(inputs.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(cmd,argv,timeout=120000){
 const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,1000));assert.equal(r.stderr,'','unexpected diagnostics');return r.stdout;}
function rays(key){const sq=key%64,f=sq%8,r=Math.floor(sq/8),dirs=key<64?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,1],[1,-1],[-1,1],[-1,-1]];
 return dirs.map(([df,dr])=>{const a=[];for(let x=f+df,y=r+dr;x>=0&&x<8&&y>=0&&y<8;x+=df,y+=dr)a.push(8*y+x);return a;});}
const masks=Array.from({length:128},(_,k)=>rays(k).flatMap(r=>r.slice(0,-1)).reduce((a,s)=>a|(1n<<BigInt(s)),0n));
const bits=m=>Array.from({length:64},(_,i)=>1n<<BigInt(i)).filter(x=>(x&m)!==0n);
const sizes=masks.map(m=>2**bits(m).length),prefixes=[];let end=512;
for(const size of sizes){prefixes.push(end);end+=size;}assert.equal(end,108160);
function scatter(i,m){return bits(m).reduce((a,b,k)=>a|((i&(1n<<BigInt(k)))?b:0n),0n);}
function attack(key,occ){let a=0n;for(const ray of rays(key)){for(const sq of ray){const b=1n<<BigInt(sq);a|=b;if(occ&b)break;}}return a;}
function compare(actual,expected){const a=actual.trimEnd().split('\n'),e=expected.trimEnd().split('\n');assert.equal(a.length,e.length,'row count');a.forEach((s,i)=>assert.equal(s,e[i],`computed entry row ${i}`));}
try{
 const fixtures=[],seen=new Set();let rng=0x34ad124ee81231ban;
 function random(){rng^=(rng<<13n)&max;rng^=rng>>7n;rng=(rng^(rng<<17n))&max;return rng;}
 function add(k,mode,i){const id=[k,mode,i].join(':');if(!seen.has(id)){seen.add(id);fixtures.push([k,mode,i,...halves(random())]);}}
 for(let k=0;k<128;k++)for(const i of [0,Math.floor(sizes[k]/2),sizes[k]-1])add(k,0,i);
 for(let k=0;k<128;k+=16)for(const i of [0,sizes[k]/2-1,sizes[k]/2,sizes[k]-1])add(k,1,i);
 for(const k of [0,27,63,64,91,127]){
  for(const i of [0,1])add(k,2,i);
  for(const i of [0,sizes[k]-1])add(k,3,i);
  for(const i of [sizes[k],sizes[k]+1])add(k,4,i);
 }
 let written=0,untouched=0,full=0,tailProtected=0;
 const expected=fixtures.map(([k,mode,i,h,l],id)=>{
  const size=sizes[k],n=[size,size/2,1,0,size+1][mode],seed=(BigInt(h)<<32n)|BigInt(l);
  let value=seed;
  if(i<n){value=attack(k,scatter(BigInt(i%size),masks[k]));written++;if(i+1<n)tailProtected++;assert.notEqual(value,seed,'poison must differ from computed value');}else untouched++;
  if(mode===0)full++;
  return [id,k,...halves(BigInt(prefixes[k])),...halves(masks[k]),size,n,i,...halves(value),131072].join(' ');
 }).join('\n')+'\n';
 assert.ok(written>0&&untouched>0&&tailProtected>0);
 assert.throws(()=>compare(expected.replace(/^0 /,'1 '),expected),/computed entry row/);
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 const invalid=[['128','0','0','0','0'],['0','5','0','0','0'],['0','0','8192','0','0'],
  ['0','0','0','4294967296','0'],['x','0','0','0','0'],['0','0']];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe],180000);
  const actual=run(exe,['--threads','1',...fixtures.flat().map(String)],180000);compare(actual,expected);
  for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid fill contents|expected five fill contents/);}
  modes.push({mode,fixture_rows:fixtures.length,invalid_rejections:invalid.length,output_sha256:sha(actual)});
  console.error(`PASS ${mode}: ${fixtures.length} computed-entry/control rows`);
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:fixtures.length,chess_keys:128,
  full_block_rows:full,written_entry_rows:written,untouched_control_rows:untouched,entries_followed_by_later_writes:tailProtected,
  all_seed_values_differ_from_expected_computed_values:true,observed:'Selected entry value, full mask/prefix metadata, size/count/index/capacity; not every output cell',
  candidate:'Actual Tables.build supplies persistent metadata; actual fill runs on a fresh seeded complete allocation for each case. No proof model or host answers enter candidate.',
  oracle:'Independent signed-coordinate rays, direct compact-bit deposition and mathematical prefix sums',modes,source_sha256s:before,
  cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
  scope:'Native evidence covers zero-start chess masks and selected offsets, not every arbitrary starting subset/array or a quantified independent geometry theorem'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
