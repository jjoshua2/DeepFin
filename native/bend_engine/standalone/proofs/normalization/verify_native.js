// Independent arithmetic expectations for the supported public Base APIs.
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
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-normalization-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
function invoke(cmd,argv,timeout=90000){const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return r;}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,r.stderr+'\n'+r.stdout.slice(0,3000));assert.equal(r.stderr,'');return r.stdout;}
const tracked=[path.join(suite,'probe.bend'),path.join(suite,'verify_native.js'),path.join(root,'Text.bend')];
const hashes=()=>Object.fromEntries(tracked.map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));
const before=hashes(),seed=0xdeadbeef89abcdefn,half=x=>[String(x>>32n),String(x&0xffffffffn)];
try{
 const rows=[],cap=131072;let x=0x47103b2a;
 const rand=()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return x>>>0;};
 const indices=[0,1,511,512,4095,4096,65535,65536,70437,108159,108160,131070,131071,...Array.from({length:19},()=>rand()%cap)];
 for(const i of indices)for(const j of [i,(i+1)%cap]){
   const v=(BigInt(rand())<<32n)|BigInt(rand());rows.push([i,j,...half(v)]);
 }
 for(const i of [cap,cap+1,2*cap-1,0x80000000,0xffffffff])for(const j of [i%cap,(i+1)%cap])rows.push([i,j,...half(0x8000000100000001n)]);
 assert.equal(rows.length,74);
 const expected=rows.map(([i,j,hi,lo],id)=>[id,i%cap,cap,...half(seed),...half(i%cap===j%cap?(BigInt(hi)<<32n)|BigInt(lo):seed)].join(' ')).join('\n')+'\n';
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags]of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe],180000);
  const actual=run(exe,['--threads','1',...rows.flat().map(String)],120000);assert.equal(actual,expected,'all observed public API values, capacity and normalization');
  const invalid=[['4294967296','0','0','0'],['0','4294967296','0','0'],['0','0','4294967296','0'],['0','0','0','4294967296'],['x','0','0','0'],['0','1']];
  for(const q of invalid){const r=invoke(exe,['--threads','1',...q]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid normalization|expected four normalization/);}
  modes.push({mode,rows:rows.length,invalid_rejections:invalid.length,output_sha256:sha(actual)});console.error('PASS '+mode);
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fixture_rows:74,in_range_rows:64,out_of_range_wrapper_rows:10,
  raw_internal_routines:"Source-checked only; exploratory direct native lowering failed with open Array element type. Public wrappers are tested here.",
  allocation_cells:131072,observation:'initial and post-write query, reported capacity and masked write address; not every returned cell',
  oracle:'Independent unsigned arithmetic modulo 131072 and expected seed/write value',modes,source_sha256s:before,bun:process.versions.bun,cc:run(cc,['--version']).split('\n')[0]};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
