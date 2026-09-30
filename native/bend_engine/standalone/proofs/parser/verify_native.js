// Actual layout parser versus an external per-square-set cursor reference.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname,engine=path.resolve(suite,'../../..');
const cc=process.env.CC||'clang',temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-parser-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['standalone/Position.bend','standalone/Text.bend','legal_probe/Chess.bend','bitboard_probe/Sliders.bend',
 'standalone/proofs/parser/probe.bend','standalone/proofs/parser/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const original=hashes();
function invoke(cmd,argv,timeout=90000){const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout,maxBuffer:32<<20,
 env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stdout+r.stderr).slice(-3000));assert.equal(r.stderr,'');return r.stdout;}
const empty=()=>Array.from({length:64},()=>({k:new Set(),c:new Set()}));
const clone=cells=>cells.map(x=>({k:new Set(x.k),c:new Set(x.c)}));
function initial(mode){const cells=empty();
 if(mode===1){const back=[3,1,2,4,5,2,1,3];for(let f=0;f<8;f++){
  cells[f].k.add(back[f]);cells[f].c.add(1);cells[8+f].k.add(0);cells[8+f].c.add(1);
  cells[48+f].k.add(0);cells[48+f].c.add(0);cells[56+f].k.add(back[f]);cells[56+f].c.add(0);}}
 if(mode===2){const limbs=[[305419896,2864434397],[2271560481,4275878552],[1431655765,2863311530],[252645135,4042322160],
  [858993459,3435973836],[16711935,4278255360],[2147483648,1],[4294967295,4294967295]];
  limbs.forEach(([h,l],j)=>{const v=(BigInt(h)<<32n)|BigInt(l);for(let sq=0;sq<64;sq++)if(v&(1n<<BigInt(sq))){
   if(j<6)cells[sq].k.add(j);else cells[sq].c.add(j===6?1:0);}});}
 return cells;
}
function encode(s){const planes=Array(8).fill(0n);s.cells.forEach((x,sq)=>{const bit=1n<<BigInt(sq);
 for(const k of x.k)planes[k]|=bit;for(const c of x.c)planes[c===1?6:7]|=bit;});
 return [s.file,s.rank,Number(s.live),Number(s.live&&s.file===8&&s.rank===0),
  ...planes.flatMap(x=>[Number(x>>32n),Number(x&0xffffffffn)]),...s.metadata];}
function parse(text,start){const s={...start,cells:clone(start.cells),metadata:[...start.metadata]};
 for(const ch of text){const c=ch.codePointAt(0);
  if(c===47){s.live=s.live&&s.file===8&&s.rank>0;s.file=0;s.rank=(s.rank-1)>>>0;}
  else if(c>=49&&c<=56){s.file=(s.file+c-48)>>>0;s.live=s.live&&s.file<=8;}
  else{const p='pnbrqk'.indexOf(String.fromCodePoint(c|32));const sq=(s.file+8*s.rank)>>>0;
   // The actual bit constructor uses a saturating shift, not sq modulo64.
   if(sq<64){if(p>=0)s.cells[sq].k.add(p);s.cells[sq].c.add(c<97?1:0);}
   s.live=s.live&&p>=0&&s.file<8;s.file=(s.file+1)>>>0;}}
 return s;}
let rng=0x94519ac3;function random(){rng^=rng<<13;rng^=rng>>>17;rng^=rng<<5;return rng>>>0;}
const fixtures=[],seen=new Set(),counts={};
function add(category,a,b,file=0,rank=7,live=1,mode=0){const metadata=[random(),random(),random()];
 const input=[a,b,file,rank,live,...metadata,mode],id=JSON.stringify(input);assert.ok(!seen.has(id));seen.add(id);
 const start={cells:initial(mode),file,rank,live:Boolean(live),metadata},prefix=parse(a,start),last=parse(b,prefix),joined=parse(a+b,start);
 const expected=[...encode(prefix),...encode(last),...encode(joined)];assert.equal(expected.length,69);
 assert.deepEqual(encode(last),encode(joined));if(!prefix.live)assert.equal(last.live,false);
 if(encode(last)[3])assert.equal(prefix.live,true);
 fixtures.push({category,input,expected,prefix_live:prefix.live,accepted:Boolean(encode(last)[3])});counts[category]=(counts[category]||0)+1;
}
// Repair-shaped suffixes exercise the initially invalid state before the main corpus.
for(const mode of [0,1,2])add('invalid_seed','', '8',0,0,0,mode);
const clear='8/8/8/8/8/8/8/8',start='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
for(const [category,text] of [['empty_layout',clear],['start_layout',start]])
 for(let split=0;split<=text.length;split++)add(category,text.slice(0,split),text.slice(split));
const chars='PNBRQKpnbrqk';
for(let sq=0;sq<64;sq++)for(const ch of chars){const rank=Math.floor(sq/8),file=sq%8;
 const row=(file?String(file):'')+ch+(file<7?String(7-file):'');const rows=Array(8).fill('8');rows[7-rank]=row;
 const text=rows.join('/'),split=random()%(text.length+1);add('one_piece',text.slice(0,split),text.slice(split));}
const bad=['','/','8/8','88/8/8/8/8/8/8/8','0/8/8/8/8/8/8/8','9/8/8/8/8/8/8/8',
 'x7/8/8/8/8/8/8/8',clear+'/',clear+'/8','8//8/8/8/8/8/8','PPPPPPPPP/8/8/8/8/8/8/8',
 '8/8/8/8/8/8/8/8 ','8/8/8/8/8/8/8/8\n','８/8/8/8/8/8/8/8','♟7/8/8/8/8/8/8/8'];
for(const text of bad)for(const mode of [0,1,2])for(const split of new Set([0,Math.floor(text.length/2),text.length]))
 add('malformed_layout',text.slice(0,split),text.slice(split),0,7,1,mode);
for(const c of [...Array.from({length:127},(_,i)=>i+1),128,160,945,9823,128512])
 for(const live of [0,1])add('character_boundary',String.fromCodePoint(c),'8/8',c%10,c%9,live,c%3);
for(const mode of [0,1,2])for(const f of [0,7,8,9])for(const r of [0,7,8])
 for(const live of [0,1])add('cursor_boundary','P/','8',f,r,live,mode);
for(const mode of [1,2])add('arbitrary_seed',clear.slice(0,6),clear.slice(6),0,7,1,mode);
function compare(raw,chunk,offset){const lines=raw.trimEnd().split('\n');assert.equal(lines.length,chunk.length,'output rows');
 lines.forEach((line,i)=>{const got=line.split(' ').map(Number);assert.equal(got.length,69,'complete three-state observation');
 got.forEach((x,j)=>assert.equal(x,chunk[i].expected[j],`row ${offset+i}, field ${j}, ${chunk[i].category}`));});}
function execute(exe){let raw='';for(let i=0;i<fixtures.length;i+=64){const chunk=fixtures.slice(i,i+64);
 const text=run(exe,['--threads','1',...chunk.flatMap(x=>x.input)]);compare(text,chunk,i);raw+=text;}return raw;}
const good=['','8',0,0,0,1,0,64,0],change=(i,v)=>good.map((x,j)=>i===j?v:x);
const invalid=[change(2,10),change(3,9),change(4,2),change(8,3),change(5,'4294967296'),change(5,'x'),good.slice(0,-1),
 change(0,'P'.repeat(513)),Array.from({length:65},()=>good).flat()];
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],
 ['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){const exe=path.join(temp,mode);
 run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-lm','-pthread','-o',exe]);
 const raw=execute(exe);for(const input of invalid){const bad=invoke(exe,['--threads','1',...input]);assert.equal(bad.status,2);assert.match(bad.stdout+bad.stderr,/invalid parser probe/);}
 modes.push({mode,rows:fixtures.length,observed_fields:fixtures.length*69,invalid_rejections:invalid.length,output_sha256:sha(raw)});
 console.error(`PASS native ${mode}: ${fixtures.length} complete three-state rows`);}
 const mutations=[];for(const [name,from,to] of [
 ['revive-invalid-digit','Bool.and(valid, U32.is_le(next, 8))','U32.is_le(next, 8)'],
 ['discard-characters','case SCon{Chr{c}, rest}: layout(rest, layout_char(c, l))','case SCon{Chr{c}, rest}: layout(rest, l)']]){
 const e=path.join(temp,name);fs.cpSync(engine,e,{recursive:true});const p=path.join(e,'standalone/Position.bend'),text=fs.readFileSync(p,'utf8');
 assert.equal(text.split(from).length,2);fs.writeFileSync(p,text.replace(from,to));const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'-exe');
 console.error('START mutation compilation: '+name);
 run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/parser/probe.bend'),'-o',mc]);
 console.error('PASS mutation C generation: '+name);
 console.error('START mutation C build: '+name);
 run(cc,['-std=c11','-O1','-ffp-contract=off',mc,'-lm','-pthread','-o',exe]);
 console.error('PASS mutation C build: '+name);
 console.error('START mutation execution: '+name);
 let diagnostic='';try{execute(exe);}catch(error){assert.ok(error instanceof assert.AssertionError);diagnostic=error.message;}
 assert.match(diagnostic,/row \d+, field \d+/);mutations.push({name,rejected:true,compiled_and_executed:true,diagnostic,
 scope:'First generic-mode mismatched actual state; remaining corrupted modes not run'});}
 assert.deepEqual(hashes(),original);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,rows_per_mode:fixtures.length,fields_per_mode:fixtures.length*69,
 distinct_inputs:seen.size,categories:counts,accepted_final_rows:fixtures.filter(x=>x.accepted).length,
 invalid_prefix_rows:fixtures.filter(x=>!x.prefix_live).length,modes,mutations,source_sha256s:original,
 fixture_sha256:sha(JSON.stringify(fixtures)),native_domain:'IO Unicode strings without embedded NUL; combined length<=512, initial file0..9/rank0..8; arbitrary U32 metadata; three fixed Board seeds. Source laws cover arbitrary Strings/Layout values.',
 oracle:'Independent square kind/color sets plus character cursor interpreter; compares entire prefix, split suffix and concatenated Layout states, result acceptance and metadata. No proof code runs in candidate.',
 scope:'Placement-layout traversal and acceptance safety, not all-field FEN validation, parser-wide freshness, legal boards or rollback of rejected raw layouts.',cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun};
 const out=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
