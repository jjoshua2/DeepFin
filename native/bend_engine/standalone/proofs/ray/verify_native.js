// Opt-in actual ray/slider checks against independent signed-coordinate traversal.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {verifyCompiler,PIN} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),engine=path.resolve(import.meta.dirname,'../../..');
const cli=path.join(compiler,'bend2/main.ts'),suite=import.meta.dirname,cc=process.env.CC||'clang';
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-ray-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex'),max=(1n<<64n)-1n;
const tracked=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend','standalone/proofs/ray/probe.bend','standalone/proofs/ray/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(cmd,argv){const p=spawnSync(cmd,argv,{encoding:'utf8',timeout:180000,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(p.error,undefined,String(p.error));assert.equal(p.signal,null);return p;}
function run(cmd,argv){const p=invoke(cmd,argv);assert.equal(p.status,0,(p.stderr+p.stdout).slice(-3000));assert.equal(p.stderr,'','unexpected diagnostics');return p.stdout;}
const dirs=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];
function coordinates(s,d){const [dx,dy]=dirs[d],xs=[];for(let x=s%8+dx,y=Math.floor(s/8)+dy;x>=0&&x<8&&y>=0&&y<8;x+=dx,y+=dy)xs.push(y*8+x);return xs;}
function attack(xs,occ){let v=0n;for(const x of xs){const b=1n<<BigInt(x);v|=b;if(occ&b)break;}return v;}
function slider(k,occ){let v=0n;for(let d=(k<64?0:4);d<(k<64?4:8);d++)v|=attack(coordinates(k%64,d),occ);return v;}
let seed=0x5db26de965d313a7n;
function random(){seed^=(seed<<13n)&max;seed^=seed>>7n;seed^=(seed<<17n)&max;seed&=max;return seed;}
const rows=[],seen=new Set();
function add(kind,key,dir,occ,acc=0n){const row=[kind,key,dir,occ,acc],id=row.join(':');if(!seen.has(id)){seen.add(id);rows.push(row);}}
let blockerConfigurations=0;
for(let s=0;s<64;s++)for(let d=0;d<8;d++){
 const xs=coordinates(s,d),mask=xs.reduce((m,x)=>m|(1n<<BigInt(x)),0n);
 // Every first-blocker position, with all subsequent blockers also set.
 add(0,s,d,0n);add(0,s,d,max^mask);
 for(let i=0;i<xs.length;i++){blockerConfigurations++;const suffix=xs.slice(i).reduce((m,x)=>m|(1n<<BigInt(x)),0n);add(0,s,d,suffix);add(0,s,d,suffix|(max^mask));}
 add(0,s,d,random());add(0,s,d,random(),random());
}
for(let k=0;k<128;k++){
 const ds=dirs.slice(k<64?0:4,k<64?4:8),edge=ds.reduce((m,_d,j)=>{const xs=coordinates(k%64,j+(k<64?0:4));return xs.length?m|(1n<<BigInt(xs.at(-1))):m;},0n);
 for(const occ of [0n,max,edge,max^edge,0xaaaaaaaaaaaaaaaan,0x5555555555555555n,random(),random()]){add(1,k,0,occ);add(2,k,0,occ);}
}
const expected=rows.map(([kind,k,d,occ,acc])=>{const v=kind===0?(acc|attack(coordinates(k,d),occ)):slider(k,occ);return `${v>>32n} ${v&0xffffffffn}`;});
const encode=rs=>rs.flatMap(([kind,k,d,o,a])=>[kind,k,d,String(o>>32n),String(o&0xffffffffn),String(a>>32n),String(a&0xffffffffn)]).map(String);
function compare(exe,wanted=expected){let output='';for(let i=0;i<rows.length;i+=512){const raw=run(exe,['--threads','1',...encode(rows.slice(i,i+512))]),ls=raw.trimEnd().split('\n');
 assert.equal(ls.length,Math.min(512,rows.length-i));ls.forEach((v,j)=>assert.equal(v,wanted[i+j],`ray/slider row ${i+j}`));output+=raw;}return output;}
try{
 const c=path.join(tmp,'probe.c');run(process.execPath,[cli,path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(tmp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const output=compare(exe);
  const invalid=[['0','64','0','0','0','0','0'],['0','0','8','0','0','0','0'],['1','128','0','0','0','0','0'],['3','0','0','0','0','0','0'],['1','0','1','0','0','0','0'],['0','0','0','4294967296','0','0','0'],['x','0','0','0','0','0','0'],['0','0'],Array(1025).fill(['0','0','0','0','0','0','0']).flat()];
  for(const bad of invalid){const p=invoke(exe,['--threads','1',...bad]);assert.equal(p.status,2);assert.match(p.stdout+p.stderr,/invalid ray|ray input budget/);}
  modes.push({mode,rows:rows.length,invalid_rejections:invalid.length,output_sha256:sha(output)});
  console.error(`PASS ${mode}: ${rows.length} distinct ray/slider rows`);
 }
 const mutations=[];
 for(const [name,old,value] of [
  ['ignore-blocker','stop = Bool.or(U32.is_eq(next, 64), U64.test_bit(occ, U32.to_nat(next)))','stop = U32.is_eq(next, 64)'],
  ['omit-blocker','acc = U64.or(acc, Bool.pick(U64, add, U64.bit(U32.to_nat(next)), U64.zero()))','acc = U64.or(acc, Bool.pick(U64, Bool.and(add,Bool.not(U64.test_bit(occ,U32.to_nat(next)))), U64.bit(U32.to_nat(next)), U64.zero()))']]){
  const m=path.join(tmp,name);fs.cpSync(engine,m,{recursive:true});const table=path.join(m,'standalone/Tables.bend'),source=fs.readFileSync(table,'utf8');assert.equal(source.split(old).length,2);fs.writeFileSync(table,source.replace(old,value));
  const mc=path.join(tmp,name+'.c'),exe=path.join(tmp,name+'.bin');run(process.execPath,[cli,path.join(m,'standalone/proofs/ray/probe.bend'),'-o',mc]);run(cc,['-std=c11','-O1',mc,'-pthread','-lm','-o',exe]);
  let message='';try{compare(exe);}catch(e){message=e.message;}
  assert.match(message,/ray\/slider row/,'must reject actual wrong values, not a process failure');mutations.push({name,rejected:true,kind:'generic wrong-value rejection after successful compilation and execution',diagnostic:message.slice(0,1000)});
 }
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,distinct_rows:rows.length,ray_rows:rows.filter(x=>x[0]===0).length,unmasked_slider_rows:rows.filter(x=>x[0]===1).length,masked_slider_rows:rows.filter(x=>x[0]===2).length,ray_domains:512,chess_keys:128,first_blocker_positions:blockerConfigurations,modes,mutations,source_sha256s:before,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,scope:'Actual Tables.ray and masked/unmasked Tables.slider; no proof predicates or candidate input from reference. All first-blocker positions plus off-ray noise, nonzero accumulators and selected full occupancies. Repeated modes, not exhaustive U64 pairs or new allocation/lifetime evidence.'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
