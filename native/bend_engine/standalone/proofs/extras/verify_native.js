// Full-buffer external oracle for the real final extras stage. Opt-in only.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]), identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const cc=process.env.CC||'clang',tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-extras-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const files=['Tables.bend','Text.bend','Subsets.bend','proofs/extras/probe.bend','proofs/extras/verify_native.js','proofs/extras/unique_probe.bend'];
const hashes=()=>Object.fromEntries(files.map(f=>[f,sha(fs.readFileSync(path.join(root,f)))]));
const before=hashes();
const fixtures=[[0,64,0xdeadbeef89abcdefn],[1,0,0x81234567fedcba98n],[1,63,0xffffffff12345678n],
  [2,62,0xaaaaaaaa55555555n],[64,0,0xbad0cafe12345678n]];
const locations=[0,255,512,108159,131071];
const moves={knight:[[1,2],[2,1],[-1,2],[-2,1],[1,-2],[2,-1],[-1,-2],[-2,-1]],
  king:[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],
  white:[[1,1],[-1,1]],black:[[1,-1],[-1,-1]]};
function attack(sq,steps){const x=sq%8,y=Math.floor(sq/8);let out=0n;for(const [dx,dy] of steps){
  const f=x+dx,r=y+dy;if(f>=0&&f<8&&r>=0&&r<8)out|=1n<<BigInt(8*r+f);}return out;}
const expected=fixtures.map(([n,k,seed])=>{const a=Array(131072).fill(seed);locations.forEach((j,i)=>a[j]=seed^BigInt(i+1));
  for(let s=k;s<k+n;s++){a[256+s]=attack(s,moves.knight);a[320+s]=attack(s,moves.king);a[384+s]=attack(s,moves.white);a[448+s]=attack(s,moves.black);}return a;});
// Uniform seeds cannot detect a copy between two ordinary protected cells.
// These fixed native fixtures initialize every cell differently before extras.
// Keep the original five buffers and their malformed-input cases unchanged.
const uniqueFixtures=[[0,64],[2,31],[64,0]];
const uniqueExpected=uniqueFixtures.map(([n,k])=>{
  const a=Array.from({length:131072},(_,j)=>
    ((0xdeadbeefn^BigInt(j))<<32n)|((0x89abcdefn+BigInt(j))&0xffffffffn));
  assert.equal(new Set(a).size,a.length,'initial cell identities must be distinct');
  for(let s=k;s<k+n;s++){a[256+s]=attack(s,moves.knight);a[320+s]=attack(s,moves.king);
    a[384+s]=attack(s,moves.white);a[448+s]=attack(s,moves.black);}return a;
});
function compareUnique(out){const lines=out.trimEnd().split('\n');assert.equal(lines.length,uniqueFixtures.length);
  lines.forEach((line,id)=>{const x=line.split(' ');assert.equal(x.length,2+2*131072);
    assert.deepEqual(x.slice(0,2),[String(id),'131072']);
    for(let j=0;j<131072;j++){const v=(BigInt(x[2+2*j])<<32n)|BigInt(x[3+2*j]);
      assert.equal(v,uniqueExpected[id][j],`unique case ${id} cell ${j}`);}});}
function invoke(cmd,argv,timeout=180000){const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:64<<20,
  env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stderr+r.stdout).slice(0,3000));assert.equal(r.stderr,'');return r.stdout;}
function compare(out){const lines=out.trimEnd().split('\n');assert.equal(lines.length,fixtures.length);
  lines.forEach((line,id)=>{const x=line.split(' ');assert.equal(x.length,4+2*131072);
    assert.deepEqual(x.slice(0,4),[String(id),String(fixtures[id][0]),String(fixtures[id][1]),'131072']);
    for(let j=0;j<131072;j++){const v=(BigInt(x[4+2*j])<<32n)|BigInt(x[5+2*j]);assert.equal(v,expected[id][j],`case ${id} cell ${j}`);}});}
try{
  const c=path.join(tmp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(import.meta.dirname,'probe.bend'),'-o',c]);
  const uniqueC=path.join(tmp,'unique.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(import.meta.dirname,'unique_probe.bend'),'-o',uniqueC]);
  const uniqueModes=[];
  const modes=[];const cli=fixtures.flatMap(([n,k,v])=>[String(n),String(k),String(v>>32n),String(v&0xffffffffn)]);
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],
    ['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const exe=path.join(tmp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
    const out=run(exe,['--threads','1',...cli]);compare(out);
    const invalid=[['65','0','1','2'],['1','64','1','2'],['1','4294967295','1','2'],['x','0','1','2'],['1','0','4294967296','2'],['1','0'],Array(9).fill(['0','0','1','2']).flat()];
    for(const f of invalid){const r=invoke(exe,['--threads','1',...f]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid extras|expected four extras/);}
    modes.push({mode,full_buffers:fixtures.length,cell_comparisons:fixtures.length*131072,invalid_rejections:invalid.length,output_sha256:sha(out)});
    const uniqueExe=path.join(tmp,mode+'-unique');
    run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,uniqueC,'-pthread','-lm','-o',uniqueExe]);
    const uniqueOut=run(uniqueExe,['--threads','1']);compareUnique(uniqueOut);
    uniqueModes.push({mode,full_buffers:uniqueFixtures.length,cell_comparisons:uniqueFixtures.length*131072,
      output_sha256:sha(uniqueOut)});
    console.error(`PASS ${mode}: ${fixtures.length} original + ${uniqueFixtures.length} distinct-cell complete buffers`);
  }
  assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,full_buffers_per_mode:5,cell_comparisons_per_mode:655360,
    source_shapes:'complete depth-17 allocations; five nonzero seeds with five distinct sentinel cells',
    scopes:'all header cells, all active extras cells, all logical slider cells and all allocation slack compared',
    oracle:'flat arrays and independent signed-coordinate knight/king/pawn steps',modes,source_sha256s:before,
    unique_fixture_ranges:uniqueFixtures,unique_full_buffers_per_mode:uniqueFixtures.length,
    unique_cell_comparisons_per_mode:uniqueFixtures.length*131072,
    total_full_buffers_per_mode:fixtures.length+uniqueFixtures.length,
    total_cell_comparisons_per_mode:(fixtures.length+uniqueFixtures.length)*131072,
    unique_initialization:'Every source cell j has high32=0xdeadbeef XOR j and low32=(0x89abcdef+j) modulo 2^32; no shared seed can hide a copy between protected cells',
    unique_modes:uniqueModes,
    cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun};
  const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
