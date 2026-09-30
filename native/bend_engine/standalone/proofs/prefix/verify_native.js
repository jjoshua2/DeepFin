// Opt-in full-buffer comparison. Candidate computes all offsets and tables itself.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),engine=path.resolve(import.meta.dirname,'../../..');
const root=path.join(engine,'standalone'),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-prefix-native-'));
const sha=s=>createHash('sha256').update(s).digest('hex'),half=x=>`${x>>32n} ${x&0xffffffffn}`;
const paths=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
  'standalone/proofs/prefix/probe.bend','standalone/proofs/prefix/verify_native.js'];
const hashes=()=>Object.fromEntries(paths.map(p=>[p,sha(fs.readFileSync(path.join(engine,p)))])),before=hashes();
function invoke(cmd,argv,timeout=120000){const r=spawnSync(cmd,argv,{encoding:'utf8',timeout,maxBuffer:32<<20,
  env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,`${cmd}: ${r.error}`);assert.equal(r.signal,null);return r;}
function run(cmd,argv,timeout){const r=invoke(cmd,argv,timeout);assert.equal(r.status,0,`${r.stderr}\n${r.stdout.slice(0,2000)}`);assert.equal(r.stderr,'');return r.stdout;}
function rays(sq,bishop){const ds=bishop?[[1,1],[1,-1],[-1,1],[-1,-1]]:[[1,0],[-1,0],[0,1],[0,-1]];
 return ds.map(([dx,dy])=>{const out=[];for(let x=sq%8+dx,y=Math.floor(sq/8)+dy;x>=0&&x<8&&y>=0&&y<8;x+=dx,y+=dy)out.push(1n<<BigInt(y*8+x));return out;});}
function attacks(rs,occ){let out=0n;for(const r of rs)for(const b of r){out|=b;if(b&occ)break;}return out;}
function deposit(index,bits){return bits.reduce((v,b,i)=>v|((index&(1<<i))?b:0n),0n);}
function leaper(sq,ds){let v=0n;for(const[dx,dy]of ds){const x=sq%8+dx,y=Math.floor(sq/8)+dy;if(x>=0&&x<8&&y>=0&&y<8)v|=1n<<BigInt(y*8+x);}return v;}
const king=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],knight=[[1,2],[2,1],[-1,2],[-2,1],[1,-2],[2,-1],[-1,-2],[-2,-1]];
function reference(op,count,seed){const a=Array(131072).fill(seed),written=new Set(),blocks=[];let at=512;
 const put=(i,v)=>{assert.ok(i>=0&&i<a.length,'out-of-allocation reference write');assert.ok(!written.has(i),'overlapping logical regions');a[i]=v;written.add(i);};
 for(let key=0;key<count;key++){const rs=rays(key%64,key>=64),bits=rs.flatMap(r=>r.slice(0,-1)).sort((a,b)=>a<b?-1:1),mask=bits.reduce((x,b)=>x|b,0n),size=2**bits.length;
  put(key,mask);put(128+key,BigInt(at));assert.ok(at+size<2**32);blocks.push({key,start:at,end:at+size,size});
  for(let i=0;i<size;i++)put(at+i,attacks(rs,deposit(i,bits)));at+=size;
 }
 if(op===1)for(let s=0;s<64;s++){put(256+s,leaper(s,knight));put(320+s,leaper(s,king));put(384+s,leaper(s,[[1,1],[-1,1]]));put(448+s,leaper(s,[[1,-1],[-1,-1]]));}
 return{cells:a,text:a.map(half).join('\n')+'\n',blocks,end:at,written:written.size,untouched:a.length-written.size};
}
function same(actual,expected){const a=actual.trimEnd().split('\n'),b=expected.trimEnd().split('\n');assert.equal(a.length,b.length,'buffer length');for(let i=0;i<a.length;i++)assert.equal(a[i],b[i],`array cell ${i}`);}
try{
 const cases=[[0,0,0xdeadbeef89abcdefn],[0,1,0xdeadbeef89abcdefn],[0,64,0x8000000000000001n],[0,128,0x123456789abcdef0n],[1,128,0n]];
 const expected=cases.map(([op,n,seed])=>reference(op,n,seed));assert.equal(expected.at(-1).end,108160);assert.equal(expected.at(-1).written,108160);
 assert.throws(()=>same(expected[0].text.replace(/^\d+ /,'0 '),expected[0].text),/array cell 0/);
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(import.meta.dirname,'probe.bend'),'-o',c]);
 const modes=[];
 for(const[mode,flags]of[['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe],180000);
  const outputs=[];
  for(let i=0;i<cases.length;i++){const[op,n,seed]=cases[i],actual=run(exe,['--threads','1',String(op),String(n),...half(seed).split(' ')],180000);same(actual,expected[i].text);outputs.push(sha(actual));}
  const invalid=[['2','0','0','0'],['0','129','0','0'],['1','127','0','0'],['1','128','0','1'],['0','0','4294967296','0'],['x','0','0','0'],['0','1']];
  for(const x of invalid){const r=invoke(exe,['--threads','1',...x]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid prefix probe|expected four prefix/);}
  modes.push({mode,executions:cases.length,full_array_cells:cases.length*131072,invalid_rejections:invalid.length,output_sha256s:outputs});console.error('PASS '+mode+': '+cases.length+' full buffers');
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,
  executions_per_mode:cases.length,full_array_cells_per_mode:cases.length*131072,all_chess_keys:128,
  logical_end:108160,allocation_capacity:131072,cases:cases.map(([op,n,seed],i)=>({op,count:n,seed:seed.toString(),end:expected[i].end,distinct_writes:expected[i].written,untouched_cells:expected[i].untouched})),
  modes,source_sha256s:before,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
  scope:'Original Tables.tables partial/full prefixes and Tables.build complete returned buffers; not native execution of a proof schedule or general geometric-refinement theorem'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
