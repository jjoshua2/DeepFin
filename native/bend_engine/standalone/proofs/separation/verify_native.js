// Opt-in frame tests. Reuse the unchanged real Array/Tables probe; no proof image executes.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=path.join(root,'proofs/separation'),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-separation-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),max=(1n<<64n)-1n;
const seed=0xdeadbeef89abcdefn,halves=x=>[String(x>>32n),String(x&0xffffffffn)];
const inputs=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
  'standalone/proofs/storage/probe.bend','standalone/proofs/separation/verify_native.js'];
const hashes=()=>Object.fromEntries(inputs.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(cmd,argv,timeout=120000){
  const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,`${cmd}: ${r.error}`);assert.equal(r.signal,null);return r;
}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,2000));
  assert.equal(r.stderr,'','unexpected native/tool diagnostics');return r.stdout;}
function directions(key){return key<64?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,1],[1,-1],[-1,1],[-1,-1]];}
function rays(key){const s=key%64,x=s%8,y=Math.floor(s/8);return directions(key).map(([dx,dy])=>{
  const out=[];for(let f=x+dx,r=y+dy;f>=0&&f<8&&r>=0&&r<8;f+=dx,r+=dy)out.push(r*8+f);return out;});}
function mask(key){return rays(key).flatMap(r=>r.slice(0,-1)).reduce((v,s)=>v|(1n<<BigInt(s)),0n);}
function bits(m){return Array.from({length:64},(_,i)=>1n<<BigInt(i)).filter(v=>(v&m)!==0n);}
function gather(v,b){return b.reduce((a,x,i)=>a|((v&x)?1n<<BigInt(i):0n),0n);}
function scatter(v,b){return b.reduce((a,x,i)=>a|((v&(1n<<BigInt(i)))?x:0n),0n);}
function slider(key,occ){let out=0n;for(const r of rays(key)){for(const s of r){const b=1n<<BigInt(s);out|=b;if(occ&b)break;}}return out;}
function compare(a,b){const rows=a.trimEnd().split('\n'),wanted=b.trimEnd().split('\n');assert.equal(rows.length,wanted.length,'rows');
  rows.forEach((s,i)=>assert.equal(s,wanted[i],`row ${i} full array`));}
try{
  const fixtures=[],seen=new Set();
  function add(op,d,k,at,count,q,v){const f=[op,d,k,at>>>0,count,q>>>0,...halves(v)],id=f.join(':');if(!seen.has(id)){seen.add(id);fixtures.push(f);}}
  let rng=0xb94f618825dcf107n;
  function random(){rng^=(rng<<13n)&max;rng^=rng>>7n;rng=(rng^(rng<<17n))&max;return rng;}
  for(let d=0;d<=7;d++){const n=2**d;
    for(const i of [0,1,n-1,n,n+1,0xfffffffe,0xffffffff])
      for(const q of [i,i+1,i+n,i+n+1])
        for(const v of [0n,1n<<63n,random()])add(0,d,0,i,0,q,v);
  }
  for(let key=0;key<128;key++){
    add(1,3,key,0xffffffff,3,3,random()); // wraps; query separated
    add(1,3,key,0xffffffff,3,8,random()); // distinct integer, aliases written zero
    add(1,7,key,126,4,130,random()); // normalized query 2 untouched
    add(1,7,key,126,4,128,random()); // normalized query 0 overwritten
    add(1,3,key,3,8,2,random()); // a full small-buffer cycle has no protected slot
  }
  add(1,3,0,0,0,0,0n); // explicit zero-count base case
  let cellsChecked=0,aliasRows=0,protectedReads=0,interferenceWitnesses=0;
  const counts={array_set:0,fill:0};
  const expected=fixtures.map((f,id)=>{
    const [op,d,key,at,count,q,h,l]=f,value=(BigInt(h)<<32n)|BigInt(l),n=2**d,a=Array(n).fill(seed);
    const set=(i,v)=>{a[(i>>>0)&(n-1)]=v;};
    if(op===0){counts.array_set++;set(at,value);if(at!==q&&(at&(n-1))===(q&(n-1)))aliasRows++;}
    else if(op===1){counts.fill++;const selected=bits(mask(key)),capacity=1n<<BigInt(selected.length),start=gather(value,selected);
      for(let j=0;j<count;j++)set(at+j,slider(key,scatter((start+BigInt(j))%capacity,selected)));}
    const writes=op===0?[at]:Array.from({length:count},(_,j)=>(at+j)>>>0);
    const clear=writes.every(i=>(i&(n-1))!==(q&(n-1)));
    if(clear){assert.equal(a[q&(n-1)],seed);protectedReads++;}
    else if(a[q&(n-1)]!==seed){interferenceWitnesses++;}
    cellsChecked+=a.length;
    return ['v',id,n,...halves(seed),...halves(a[at&(n-1)]),...halves(a[q&(n-1)]),...a.flatMap(halves)].join(' ');
  }).join('\n')+'\n';
  assert.ok(fixtures.length<4096);assert.ok(aliasRows>0);assert.ok(protectedReads>0);assert.ok(interferenceWitnesses>0);
  assert.throws(()=>compare(expected.replace('v 0 ','v 1 '),expected),/row 0/);
  const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(root,'proofs/storage/probe.bend'),'-o',c]);
  const modes=[];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],
    ['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe],180000);
    const actual=run(exe,['--threads','1',...fixtures.flat().map(String)],180000);compare(actual,expected);
    const invalid=[['4','0','0','0','0','0','0','0'],['0','8','0','0','0','0','0','0'],
      ['1','3','128','0','1','0','0','0'],['1','3','0','0','9','0','0','0'],
      ['0','1','0','4294967296','0','0','0','0'],['x','0','0','0','0','0','0','0'],['0','1']];
    for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid storage|expected eight storage/);}
    modes.push({mode,fixture_rows:fixtures.length,full_array_cells:cellsChecked,invalid_rejections:invalid.length,output_sha256:sha(actual)});
    console.error('PASS native '+mode+': '+fixtures.length+' rows, '+cellsChecked+' cells');
  }
  assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:fixtures.length,
    cases_by_operation:counts,chess_fill_keys:128,full_array_cells:cellsChecked,distinct_indices_aliasing_rows:aliasRows,
    protected_query_rows:protectedReads,overlap_counterexample_rows:interferenceWitnesses,
    input_array_shapes:'Complete binary arrays only; arbitrary/ragged arrays covered by source laws, not native fixtures',
    oracle:'Independent flat array, unsigned index masking, signed-coordinate rays and compact scatter successor',
    modes,source_sha256s:before,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
    scope:'Real set/get and fill frame outcomes under independent normalized-index classification; no native execution of proof predicates, numeric prefix proof or final lookup-geometry theorem'};
  const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
