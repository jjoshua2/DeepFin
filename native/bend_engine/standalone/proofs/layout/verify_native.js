// Opt-in native mask/size/lookup-index agreement. No perft/model/engine build.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1 || (args.length===3 && args[1]==='--report'),
  'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler);
const root=path.resolve(import.meta.dirname,'../..'), engine=path.dirname(root);
const cc=process.env.CC || 'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-layout-native-'));
const max64=(1n<<64n)-1n, sha=b=>createHash('sha256').update(b).digest('hex');
const halves=x=>[String(x>>32n),String(x&0xffffffffn)];
const files=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
  'standalone/proofs/layout/Spec.bend','standalone/proofs/layout/probe.bend',
  'standalone/proofs/layout/verify_native.js','bitboard_probe/Sliders.bend'];
const hashes=()=>Object.fromEntries(files.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(command,argv,timeout=60000){
  const r=spawnSync(command,argv,{encoding:'utf8',timeout,maxBuffer:16<<20,
    env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,`${command}: ${r.error}`);assert.equal(r.signal,null);
  return r;
}
function run(command,argv,timeout){
  const r=invoke(command,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,3000));
  assert.equal(r.stderr,'','unexpected diagnostics');return r.stdout;
}
function maskFor(key){
  const square=key%64, x=square%8,y=Math.floor(square/8);
  const dirs=key<64?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,1],[1,-1],[-1,1],[-1,-1]];
  let mask=0n;
  for(const [dx,dy] of dirs){
    const ray=[];
    for(let f=x+dx,r=y+dy; f>=0&&f<8&&r>=0&&r<8; f+=dx,r+=dy)ray.push(r*8+f);
    // A relevant occupancy mask excludes the final board-edge square of each ray.
    for(const square of ray.slice(0,-1))mask|=1n<<BigInt(square);
  }
  return mask;
}
function positions(mask){return Array.from({length:64},(_,i)=>1n<<BigInt(i)).filter(bit=>mask&bit);}
function extract(x,bits){let result=0n;bits.forEach((bit,i)=>{if(x&bit)result|=1n<<BigInt(i);});return result;}
function compare(actual,expected){
  const a=actual.trimEnd().split('\n'),b=expected.trimEnd().split('\n');
  assert.equal(a.length,b.length,'row count');a.forEach((s,i)=>assert.equal(s,b[i],`row ${i}`));
}
try{
  const fixtures=[],seen=new Set();
  function add(key,occ){const id=key+':'+occ;if(!seen.has(id)){seen.add(id);fixtures.push({key,occ});}}
  let rng=0x865f33076248aefdn;
  function random(){rng^=(rng<<13n)&max64;rng^=rng>>7n;rng=(rng^(rng<<17n))&max64;return rng;}
  const populations={rook:new Set(),bishop:new Set()};
  const regions=[];let prefix=512;
  for(let key=0;key<128;key++){
    const mask=maskFor(key),k=positions(mask).length,size=2**k;
    populations[key<64?'rook':'bishop'].add(k);
    assert.ok(k<=(key<64?12:9));assert.ok(size>0&&size<=(key<64?4096:512));
    regions.push({key,offset:prefix,size});prefix+=size;
    for(const occ of [0n,max64,mask,max64^mask,0xffffffffn,0xffffffff00000000n,
      1n<<63n,1n<<BigInt(key%64),0xaaaaaaaaaaaaaaaan,0x5555555555555555n])add(key,occ);
    for(let n=0;n<8;n++)add(key,random());
  }
  assert.ok(fixtures.length<=4096);assert.equal(prefix,108160);
  // These offsets are an external numeric observation, not source-proved Array facts.
  const expected=fixtures.map(({key,occ},id)=>{
    const mask=maskFor(key),bits=positions(mask),index=extract(occ,bits),size=1n<<BigInt(bits.length);
    assert.ok(index<size);assert.equal(index>>32n,0n);
    return ['v',id,key,...halves(mask),bits.length,String(size),...halves(index),String(index)].join(' ');
  }).join('\n')+'\n';
  assert.throws(()=>compare(expected.replace('v 0 ','v 1 '),expected),/row 0/);
  const cfile=path.join(temp,'probe.c');
  run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(root,'proofs/layout/probe.bend'),'-o',cfile],120000);
  const operands=fixtures.flatMap(({key,occ})=>[String(key),...halves(occ)]), reports=[];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],
    ['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const exe=path.join(temp,mode);
    run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,cfile,'-pthread','-lm','-o',exe],120000);
    const actual=run(exe,['--threads','1',...operands]);compare(actual,expected);
    const invalid=[['128','0','0'],['4294967295','0','0'],['0','4294967296','0'],['0','0'],['junk','0','0']];
    for(const words of invalid){const r=invoke(exe,['--threads','1',...words]);
      assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid layout|expected key/);}
    reports.push({mode,operand_pairs:fixtures.length,keys:128,malformed_or_domain_rejections:invalid.length,output_sha256:sha(actual)});
  }
  assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const result={native_gate:'PASS',compiler_revision:PIN.revision,...identity,source_sha256s:before,
    distinct_operand_pairs:fixtures.length,keys:128,
    populations:Object.fromEntries(Object.entries(populations).map(([p,s])=>[p,[...s].sort((a,b)=>a-b)])),
    oracle:'Independent signed-coordinate ray walks; direct BigInt bit-position gather',
    reference_only_logical_end:prefix,reference_offsets_are_not_source_proofs:true,
    modes:reports,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
    scope:'Actual native mask geometry, sizes, full PEXT values and lookup indices on bounded fixtures; NOT actual affine table contents, performance or full engine'};
  const text=JSON.stringify(result,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
