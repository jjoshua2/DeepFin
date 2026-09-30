// Actual fixed-initial-state traversal at every character boundary; independent squares.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),cc=process.env.CC||'clang';
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-frontier-native-'));
const sha=x=>createHash('sha256').update(x).digest('hex');
const tracked=['standalone/Position.bend','standalone/Text.bend','legal_probe/Chess.bend','bitboard_probe/Sliders.bend',
 'standalone/proofs/frontier/probe.bend','standalone/proofs/frontier/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(p=>[p,sha(fs.readFileSync(path.join(engine,p)))]));
const before=hashes();
function invoke(cmd,argv,timeout=90000){const r=spawnSync(cmd,argv.map(String),{encoding:'utf8',timeout,maxBuffer:64<<20,
 env:{...process.env,BEND_NO_TELEMETRY:'1',TERM:'dumb'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
function run(cmd,argv){const r=invoke(cmd,argv);assert.equal(r.status,0,(r.stdout+r.stderr).slice(-2500));assert.equal(r.stderr,'');return r.stdout;}
const empty=()=>Array.from({length:64},()=>({k:new Set(),c:new Set()}));
const clone=s=>({...s,cells:s.cells.map(x=>({k:new Set(x.k),c:new Set(x.c)}))});
const fresh=()=>({cells:empty(),file:0,rank:7,live:true});
function step(s,ch){s=clone(s);const code=ch.codePointAt(0);
 if(ch==='/'){s.live=s.live&&s.file===8&&s.rank>0;s.file=0;s.rank=(s.rank-1)>>>0;}
 else if(code>=49&&code<=56){s.file=(s.file+code-48)>>>0;s.live=s.live&&s.file<=8;}
 else{const k='pnbrqk'.indexOf(String.fromCodePoint(code|32)),at=(s.file+8*s.rank)>>>0;
  if(at<64){if(k>=0)s.cells[at].k.add(k);s.cells[at].c.add(code<97?1:0);}
  s.live=s.live&&k>=0&&s.file<8;s.file=(s.file+1)>>>0;}
 return s;
}
function encode(s){const planes=Array(8).fill(0n);s.cells.forEach((x,i)=>{const bit=1n<<BigInt(i);
 for(const k of x.k)planes[k]|=bit;for(const c of x.c)planes[c===1?6:7]|=bit;});
 return [s.file,s.rank,Number(s.live),Number(s.live&&s.file===8&&s.rank===0),
  ...planes.flatMap(v=>[Number(v>>32n),Number(v&0xffffffffn)]),1,0,64];}
function inspect(got){const [file,rank,live,accepted]=got;const planes=[];
 for(let k=0;k<8;k++)planes.push((BigInt(got[4+2*k])<<32n)|BigInt(got[5+2*k]));
 if(live){assert.ok(file<=8&&rank<8,'live cursor is bounded');
  for(let at=0;at<64;at++){const bit=1n<<BigInt(at),row=planes.map(v=>Boolean(v&bit)),k=row.slice(0,6).filter(Boolean).length,c=row.slice(6).filter(Boolean).length;
   assert.ok((k===0&&c===0)||(k===1&&c===1),'live board partition');
   if(Math.floor(at/8)<rank||(Math.floor(at/8)===rank&&at%8>=file))assert.equal(c,0,'unvisited region must be empty');}}
 if(accepted)assert.ok(live&&file===8&&rank===0);
 return planes;
}
const fixtures=[],seen=new Set(),categories={};
function add(category,text){if(seen.has(text))return;seen.add(text);let s=fresh();const states=[encode(s)],chars=[...text];
 for(const ch of chars){s=step(s,ch);states.push(encode(s));}
 fixtures.push({category,text,states,chars});categories[category]=(categories[category]||0)+1;}
const emptyFen='8/8/8/8/8/8/8/8',startFen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
add('collision_sensitive','pn6/8/8/8/8/8/8/8');add('empty',emptyFen);add('start',startFen);
for(let sq=0;sq<64;sq++)for(const ch of 'PNBRQKpnbrqk'){
 const rows=Array(8).fill('8'),f=sq%8;rows[7-Math.floor(sq/8)]=(f?String(f):'')+ch+(f<7?String(7-f):'');add('single_piece',rows.join('/'));}
let seed=0x31957abc;function random(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;}
for(let j=0;j<128;j++){const rows=[];for(let r=0;r<8;r++){let row='',gap=0;for(let f=0;f<8;f++){
 if(random()%3===0){gap++;continue;}if(gap){row+=String(gap);gap=0;}row+='PNBRQKpnbrqk'[random()%12];}if(gap)row+=String(gap);rows.push(row);}add('mixed',rows.join('/'));}
for(const text of ['', '/', '88', 'PPPPPPPPP', '8//8', emptyFen+'/', emptyFen+'/8', 'x7/8/8/8/8/8/8/8',
 'P7/8/8/8/8/8/8/7kP', '0/8/8/8/8/8/8/8', '9/8/8/8/8/8/8/8', '８/8/8/8/8/8/8/8', '♟7/8/8/8/8/8/8/8'])add('malformed',text);
for(const prefix of ['', 'P', 'P7/', '8/8/', emptyFen])for(const cp of [...Array.from({length:127},(_,i)=>i+1),128,160,945,9823,128512])
 add('character_boundary',prefix+String.fromCodePoint(cp)+'8/8');
for(let i=0;i<=startFen.length;i++)add('incomplete',startFen.slice(0,i));
let prefixStates=0,liveStates=0,acceptedRows=0,invalidStates=0,pieceTransitions=0;
for(const f of fixtures){prefixStates+=f.states.length;for(const s of f.states){if(s[2])liveStates++;else invalidStates++;}
 if(f.states.at(-1)[3])acceptedRows++;
 f.chars.forEach((ch,i)=>{if('PNBRQKpnbrqk'.includes(ch)&&f.states[i+1][2])pieceTransitions++;});}
function execute(exe){let raw='';for(let offset=0;offset<fixtures.length;offset+=32){const chunk=fixtures.slice(offset,offset+32);
 const text=run(exe,['--threads','1',...chunk.map(x=>x.text)]),lines=text.trimEnd().split('\n');let index=0;
 for(let i=0;i<chunk.length;i++){const f=chunk[i];let previous=null;
  f.states.forEach((expected,j)=>{assert.ok(index<lines.length,'missing state');const got=lines[index++].split(' ').map(Number);
   assert.equal(got.length,23,'complete Layout fields');got.forEach((v,k)=>assert.equal(v,expected[k],`case ${offset+i}, prefix ${j}, field ${k}, ${f.category}`));
   const planes=inspect(got);
   if(j>0&&got[2]&&'PNBRQKpnbrqk'.includes(f.chars[j-1])){const at=previous[0]+8*previous[1];assert.ok(at<64);
    const occupied=((BigInt(previous[16])<<32n)|BigInt(previous[17]))|((BigInt(previous[18])<<32n)|BigInt(previous[19]));
    assert.equal(occupied&(1n<<BigInt(at)),0n,'actual accepted piece target must have been fresh');}
   previous=got;void planes;});}
 assert.equal(index,lines.length,'unexpected states');raw+=text;}return raw;}
try{
 const c=path.join(temp,'probe.c');run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(suite,'probe.bend'),'-o',c]);
 const modes=[];for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],
 ['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){const exe=path.join(temp,mode);
 run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,c,'-lm','-pthread','-o',exe]);
 const output=execute(exe);const invalid=[['P'.repeat(257)],Array(33).fill('')];
 for(const input of invalid){const r=invoke(exe,['--threads','1',...input]);assert.equal(r.status,2);assert.match(r.stdout+r.stderr,/invalid frontier probe/);}
 modes.push({mode,placement_strings:fixtures.length,prefix_states:prefixStates,fields:prefixStates*23,invalid_rejections:invalid.length,output_sha256:sha(output)});
 console.error(`PASS ${mode}: ${fixtures.length} strings, ${prefixStates} complete prefix states`);}
 const mutations=[];for(const [name,from,to] of [
 ['piece-does-not-advance','b), U32.inc(file), rank,','b), file, rank,'],
 ['slash-does-not-descend','Layout{b, 0, U32.sub(rank, 1),','Layout{b, 0, rank,']]){
 const e=path.join(temp,name);fs.cpSync(engine,e,{recursive:true});const p=path.join(e,'standalone/Position.bend'),s=fs.readFileSync(p,'utf8');
 assert.equal(s.split(from).length,2);fs.writeFileSync(p,s.replace(from,to));const mc=path.join(temp,name+'.c'),exe=path.join(temp,name+'-exe');
 run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/frontier/probe.bend'),'-o',mc]);
 run(cc,['-std=c11','-O1','-ffp-contract=off',mc,'-lm','-pthread','-o',exe]);
 let diagnostic='';try{execute(exe);}catch(error){assert.ok(error instanceof assert.AssertionError);diagnostic=error.message;}
 assert.match(diagnostic,/case \d+, prefix \d+, field \d+/);
 mutations.push({name,rejected:true,compiled_and_executed:true,diagnostic,scope:'First generic-mode wrong-state rejection; later mutated modes not run'});
 console.error('PASS mutation '+name);}
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const result={native_gate:'PASS',compiler_revision:PIN.revision,...identity,placement_strings:fixtures.length,distinct_strings:seen.size,
 prefix_states:prefixStates,fields_per_mode:prefixStates*23,live_prefix_states:liveStates,invalid_prefix_states:invalidStates,
 accepted_final_strings:acceptedRows,live_piece_transitions:pieceTransitions,categories,modes,mutations,
 source_sha256s:before,fixture_sha256:sha(JSON.stringify(fixtures)),
 scope:'Every character boundary of actual initialized layout traversal. Complete states compared to independent square sets; cursor, partition and empty-ahead invariant checked only while live. Not arbitrary strings, six-field FEN validation, rollback or legal positions.',
 native_domain:'Unicode CLI strings without embedded NUL, length<=256, at most32 strings per process; actual fixed empty initialization.',
 cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun};
 const out=JSON.stringify(result,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
