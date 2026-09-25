// Public six-field FEN outputs versus an external rank grammar and square-array model.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname,engine=path.resolve(suite,'../../..');
const cc=process.env.CC||'clang',temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fen-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['standalone/Position.bend','standalone/Text.bend','legal_probe/Chess.bend','bitboard_probe/Sliders.bend',
 'standalone/proofs/fen/probe.bend','standalone/proofs/fen/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))])),before=hashes();
function invoke(cmd,argv,timeout=90000){const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout,maxBuffer:64<<20,
 env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-3000));assert.equal(r.stderr,'');return r.stdout;}
function placement(text){const ranks=text.split('/');if(ranks.length!==8)return null;const cells=Array(64).fill(null);
 for(let row=0;row<8;row++){let file=0;for(const ch of ranks[row]){
  if(/^[1-8]$/.test(ch)){file+=Number(ch);if(file>8)return null;}
  else {if(!/^[PNBRQKpnbrqk]$/.test(ch)||file>=8)return null;
   cells[(7-row)*8+file]={kind:'pnbrqk'.indexOf(ch.toLowerCase()),white:ch===ch.toUpperCase()};file++;}}
  if(file!==8)return null;}return cells;}
function number(s){if(!/^[0-9]+$/.test(s))return null;const n=BigInt(s);return n<=0xffffffffn?Number(n):null;}
function fen(input){const [text,side,castle,target,hs,fs]=input,cells=placement(text);
 if(!cells||!['w','b'].includes(side))return 'F 0';
 let rights=0;if(castle!=='-'){if(!castle)return 'F 0';for(const c of castle){const bit={K:1,Q:2,k:4,q:8}[c];if(!bit||(rights&bit))return 'F 0';rights|=bit;}}
 let ep=64;if(target!=='-'){if(!/^[a-h][1-8]$/.test(target))return 'F 0';ep=target.charCodeAt(0)-97+(Number(target[1])-1)*8;}
 const h=number(hs),f=number(fs);if(h===null||f===null||h>1000000||f<1||f>1000000)return 'F 0';
 const planes=Array(8).fill(0n);cells.forEach((cell,sq)=>{if(cell){const bit=1n<<BigInt(sq);planes[cell.kind]|=bit;planes[cell.white?6:7]|=bit;}});
 return ['F',1,...planes.flatMap(v=>[Number(v>>32n),Number(v&0xffffffffn)]),side==='w'?1:0,rights,ep,h,f,0].join(' ');}
const cases=[],seen=new Set(),categories={};
function add(input,category){assert.equal(input.length,6);const id=JSON.stringify(input);if(seen.has(id))return;seen.add(id);
 assert.ok(input.every(x=>!x.includes('\0')));assert.ok(input.reduce((n,s)=>n+[...s].length,0)<=1024);
 cases.push({input,expected:fen(input),category});categories[category]=(categories[category]||0)+1;}
const clear='8/8/8/8/8/8/8/8',initial='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
const specimen=['P7/8/8/8/8/8/8/7n','b','Kq','h8','0017','00042'];
add(specimen,'exact_fields');
for(const text of [clear,initial,'11111111/11111111/11111111/11111111/11111111/11111111/11111111/11111111'])add([text,'w','-','-','0','1'],'initials');
for(let sq=0;sq<64;sq++)for(const ch of 'PNBRQKpnbrqk'){
 const rank=Math.floor(sq/8),file=sq%8,rows=Array(8).fill('8');rows[7-rank]=(file?String(file):'')+ch+(file<7?String(7-file):'');
 add([rows.join('/'),sq%2?'b':'w','-','-',String(sq),String(sq+1)],'single_piece');}
let rng=0x531bc967;function random(){rng^=rng<<13;rng^=rng>>>17;rng^=rng<<5;return rng>>>0;}
for(let j=0;j<128;j++){const rows=[];for(let row=0;row<8;row++){let r='',gap=0;for(let f=0;f<8;f++){
 if(random()%3===0)gap++;else{if(gap){r+=gap;gap=0;}r+='PNBRQKpnbrqk'[random()%12];}}
 if(gap)r+=gap;rows.push(r);}add([rows.join('/'),j%2?'w':'b',j%3?'Kq':'-',j%4?'-':'a1',String(j),String(j+1)],'mixed');}
function permutations(prefix,remaining){if(prefix)add([specimen[0],'b',prefix,'h8','17','42'],'rights_permutations');
 for(let i=0;i<remaining.length;i++)permutations(prefix+remaining[i],remaining.slice(0,i)+remaining.slice(i+1));}
permutations('','KQkq');
for(let sq=0;sq<64;sq++)add([specimen[0],sq%2?'w':'b','-',String.fromCharCode(97+sq%8)+String(1+Math.floor(sq/8)),'5','9'],'ep_coordinates');
for(const h of ['0','1','17','999999','1000000','000001'])for(const f of ['1','42','999999','1000000','0001'])add([initial,'b','qkQK','a1',h,f],'clock_bounds');
const badPlacements=['','8','8/8','/','x7/8/8/8/8/8/8/8','88/8/8/8/8/8/8/8','8//8/8/8/8/8/8','PPPPPPPPP/8/8/8/8/8/8/8',clear+'/',clear+'/8','0/8/8/8/8/8/8/8','9/8/8/8/8/8/8/8','éΩ🙂/8','８/8/8/8/8/8/8/8',clear+' ',clear+'\n'];
for(const text of badPlacements)add([text,'w','-','-','0','1'],'bad_placement');
for(const c of [...Array.from({length:127},(_,i)=>i+1),128,160,945,9823,128512])add([String.fromCodePoint(c)+'7/8/8/8/8/8/8/8','w','-','-','0','1'],'character_boundary');
for(const [field,values] of [[1,['','W','B','wb',' w','w ','0','🙂']],
 [2,['','KK','QQ','kk','qq','KQK','-K','K-','-Q','none','🙂']],
 [3,['','a0','i1','a9','A1','a11','-a',' a1','🙂']],
 [4,['','-1','+1','1.0','1000001','4294967295','4294967296','9999999999999999999999','1 ',' 1','١','１']],
 [5,['','0','0000','-1','+1','1.0','1000001','4294967295','4294967296','9999999999999999999999','1 ','١']]]){
 for(const value of values){const input=[...specimen];input[field]=value;add(input,'bad_field_'+field);}}
// Output comparison is deliberately sensitive to every numeric field, not just acceptance.
function compare(raw,rows,offset){const lines=raw.trimEnd().split('\n');assert.equal(lines.length,rows.length,'FEN row count');
 for(let i=0;i<lines.length;i++)assert.equal(lines[i],rows[i].expected,`FEN row ${offset+i}`);}
function execute(exe){let raw='';for(let i=0;i<cases.length;i+=64){const batch=cases.slice(i,i+64),out=run(exe,['--threads','1','fen',...batch.flatMap(x=>x.input)]);compare(out,batch,i);raw+=out;}return raw;}
const invalid=[[],['other'],['fen',clear],['fen',...specimen.slice(0,5)],['fen',...specimen,'extra'],['fen','P'.repeat(1025),'w','-','-','0','1'],['fen',clear,'w'.repeat(1025),'-','-','0','1'],['fen',...Array.from({length:65},()=>specimen).flat()]];
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const exe=path.join(temp,mode);run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-pthread','-lm','-o',exe]);
  const raw=execute(exe);for(const input of invalid){const r=invoke(exe,['--threads','1',...input]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid fen-result/);}
  modes.push({mode,rows:cases.length,invalid_rejections:invalid.length,output_sha256:sha(raw)});console.error('PASS '+mode);}
 const mutations=[];const returned='return Game{metadata(b, t, r, e), h, f, Nil{}}';
 for(const [name,from,to] of [
  ['lost-halfmove-clock',returned,'return Game{metadata(b,t,r,e),0,f,Nil{}}'],
  ['ignored-placement-failure','b : Chess.Board <- layout_result(layout(pieces, Layout{empty(), 0, 7, True{}}))','b : Chess.Board <- Some{empty()}'],
  ['fabricated-inconsistent-board',returned,'return Game{metadata(put(6,True{},0,empty()),t,r,e),h,f,Nil{}}']]){
  const dir=path.join(temp,name);fs.cpSync(engine,dir,{recursive:true});const p=path.join(dir,'standalone/Position.bend'),old=fs.readFileSync(p,'utf8');
  assert.equal(old.split(from).length,2,'unique actual mutation');fs.writeFileSync(p,old.replace(from,to));
  const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'.bin');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(dir,'standalone/proofs/fen/probe.bend'),'-o',mc]);
  run(cc,['-std=c11','-O1','-ffp-contract=off',mc,'-pthread','-lm','-o',exe]);
  let error='';try{
   if(name==='ignored-placement-failure'){
    const i=cases.findIndex(c=>c.category==='bad_placement'&&c.input[0].startsWith('x'));
    assert.ok(i>=0);compare(run(exe,['--threads','1','fen',...cases[i].input]),[cases[i]],i);
   }else execute(exe);
  }catch(e){assert.ok(e instanceof assert.AssertionError);error=e.message;}
  assert.match(error,/FEN row \d+/);mutations.push({name,rejected:true,compiled_and_executed:true,diagnostic:error,
   scope:'First generic-mode full-Game disagreement; later corrupted modes not run'});}
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const accepted=cases.filter(x=>x.expected!=='F 0').length;
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,fen_cases:cases.length,fen_accepted:accepted,fen_rejected:cases.length-accepted,
  numeric_output_fields:23*accepted+(cases.length-accepted),single_piece_cases:categories.single_piece,rights_permutations:categories.rights_permutations,ep_coordinates:categories.ep_coordinates,
  categories,modes,mutations,fixture_sha256:sha(JSON.stringify(cases)),source_sha256s:before,
  native_domain:'Six raw Unicode strings, no embedded NUL, total length<=1024, at most64 rows per invocation.',
  oracle:'Independent rank-token grammar, 64-square typed-array board, castling-set and coordinate grammar, BigInt decimal bounds; complete Board/clock/history-length outputs.',
  scope:'Actual public FEN sampled acceptance and complete returned fields; not universal grammar completeness, exact placement theorem, semantic metadata legality or reachability. No proof code in candidate. Modes repeat fixtures.',cc:run(cc,['--version']).split('\n')[0]};
 const out=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
