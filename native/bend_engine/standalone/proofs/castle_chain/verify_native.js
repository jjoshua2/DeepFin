// Independent square-set oracle for actual final filtering and full-generator castling.
// Full legal-list output is inspected, but only the castling subset has a move-set oracle.
// Every returned child Board is checked against independent raw square-set updates.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-chain-native-'));
const cc=process.env.CC||'clang',sha=b=>createHash('sha256').update(b).digest('hex');
function cmd(argv,timeout=120000){const r=spawnSync(argv[0],argv.slice(1),{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);assert.equal(r.status,0,r.stderr.slice(-3000));assert.equal(r.stderr,'',r.stderr);return r.stdout;}
const planes=()=>Array.from({length:64},()=>({k:new Set(),c:new Set()}));
function copy(b){return {sq:b.sq.map(s=>({k:new Set(s.k),c:new Set(s.c)})),turn:b.turn,rights:b.rights,ep:b.ep};}
function put(b,s,k,c){assert.ok(s>=0&&s<64);b.sq[s].k.add(k);b.sq[s].c.add(c);return b;}
function clear(b,s){b.sq[s]={k:new Set(),c:new Set()};return b;}
function occupied(s){return s.c.size!==0;}
function consistent(b){return b.sq.every(s=>(s.k.size===0&&s.c.size===0)||(s.k.size===1&&s.c.size===1));}
function encoded(b){const bits=Array(8).fill(0n);for(let s=0;s<64;s++){for(const k of b.sq[s].k)bits[k]|=1n<<BigInt(s);for(const c of b.sq[s].c)bits[c===1?6:7]|=1n<<BigInt(s);}return [...bits.flatMap(n=>[Number(n>>32n),Number(n&0xffffffffn)]),b.turn,b.rights,b.ep];}
function bitadd(b,s,k,c){if(s<0||s>=64)return;if(k>=0&&k<6)b.sq[s].k.add(k);b.sq[s].c.add(c);}
function corner(s){return ({0:2,7:1,56:8,63:4})[s]||0;}
// Reference is square-set manipulation; no bitboard operation or candidate answer.
function apply(b,m){const [src,dst,promotion,flag]=m,side=b.turn===1?1:0;
  let kind=5;for(let k=0;k<5;k++){if(b.sq[src].k.has(k)){kind=k;break;}}
  const cap=flag===1?(Math.floor(dst/8)%2===0?dst+8:dst-8):dst;
  const rook=dst%8===6?dst+1:dst-2,landing=Math.floor((src+dst)/2),r=copy(b);
  for(const s of [src,cap,dst,...(flag===2?[rook]:[])])if(s>=0&&s<64)clear(r,s);
  bitadd(r,dst,promotion===0?kind:promotion,side);
  if(flag===2)bitadd(r,landing,3,side);
  const lost=corner(src)|corner(dst)|(kind===5?(side===1?3:12):0);
  r.turn=(b.turn^1)>>>0;r.rights=(b.rights&~lost)>>>0;
  r.ep=kind===0&&(src^dst)===16?Math.floor((src+dst)/2):64;
  return r;
}
function attacked(b,target,by){const color=by===1?1:0,tx=target%8,ty=Math.floor(target/8);
  for(let s=0;s<64;s++){const cell=b.sq[s];if(!cell.c.has(color))continue;const x=s%8,y=Math.floor(s/8),dx=tx-x,dy=ty-y;
    for(const k of cell.k){if(k===0&&Math.abs(dx)===1&&dy===(color===1?1:-1))return true;
      if(k===1&&Math.abs(dx)*Math.abs(dy)===2)return true;
      if(k===5&&Math.max(Math.abs(dx),Math.abs(dy))===1)return true;
      const diag=dx!==0&&Math.abs(dx)===Math.abs(dy),orth=(dx===0)!==(dy===0);
      if(!((k===2&&diag)||(k===3&&orth)||(k===4&&(diag||orth))))continue;
      const sx=Math.sign(dx),sy=Math.sign(dy);let xx=x+sx,yy=y+sy,blocked=false;
      while(xx!==tx||yy!==ty){if(occupied(b.sq[8*yy+xx])){blocked=true;break;}xx+=sx;yy+=sy;}
      if(!blocked)return true;
    }
  }return false;
}
function inCheck(b,side){const c=side===1?1:0,s=b.sq.findIndex(z=>z.c.has(c)&&z.k.has(5));assert.ok(s>=0,'reference guard admitted missing king');return attacked(b,s,(side^1)>>>0);}
const tails=[[],[[17,25,0,0]],[[4,6,0,2],[17,25,0,0]],[[60,58,0,2],[4,6,0,2]]];
function castle(b,ks){
  const w=b.turn===1,h=w?0:56,c=w?1:0,src=h+4,dst=h+(ks?6:2),tr=h+(ks?5:3),rook=h+(ks?7:0),right=(w?1:4)*(ks?1:2);
  const cells=(ks?[5,6]:[1,2,3]).map(x=>h+x),move=[src,dst,0,2];
  const guard=(b.rights&right)!==0&&b.sq[src].k.has(5)&&b.sq[src].c.has(c)&&b.sq[rook].k.has(3)&&b.sq[rook].c.has(c)&&cells.every(x=>!occupied(b.sq[x]));
  const emit=guard&&!inCheck(b,b.turn)&&!inCheck(apply(b,[src,tr,0,0]),b.turn);
  return {guard,emit,move};
}
// Independent coordinate first-blocker observation, not the candidate's attack table.
function sensitive(b){
  const side=b.turn===1?1:0,k=b.sq.findIndex(z=>z.c.has(side)&&z.k.has(5));assert.ok(k>=0);
  const result=new Set();for(let s=0;s<64;s++)if(b.sq[s].c.has(side)&&b.sq[s].k.has(5))result.add(s);
  for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]){
    let x=k%8+dx,y=Math.floor(k/8)+dy;
    while(x>=0&&x<8&&y>=0&&y<8){const s=y*8+x;if(occupied(b.sq[s])){if(b.sq[s].c.has(side))result.add(s);break;}x+=dx;y+=dy;}
  }return result;
}
function filtered(b,ms){
  const all=inCheck(b,b.turn),ss=sensitive(b),result=[];
  for(const m of ms){const required=all||m[3]===1||ss.has(m[0]);if(!required||!inCheck(apply(b,m),b.turn))result.unshift(m);}
  return result;
}
function expected(f){
  const k=castle(f.b,true),q=castle(f.b,false),base=[...(q.emit?[q.move]:[]),...(k.emit?[k.move]:[])];
  const input=f.op===0?tails[f.mode]:[...base,...(f.op===1?tails[f.mode]:[])];
  return {king:k,queen:q,moves:filtered(f.b,input),input};
}
function base(w){const h=w?0:56,b={sq:planes(),turn:w?1:0,rights:15,ep:64};put(b,h+4,5,w?1:0);put(b,h,3,w?1:0);put(b,h+7,3,w?1:0);put(b,w?57:1,5,w?0:1);return b;}
const fixtures=[],seen=new Set();let duplicates=0;
function add(label,b,op,mode){
  assert.ok(consistent(b));for(const c of [0,1])assert.equal(b.sq.filter(s=>s.k.has(5)&&s.c.has(c)).length,1,'reference requires one king per color');
  const input=[op,mode,...encoded(b)],key=JSON.stringify(input);if(seen.has(key)){duplicates++;return;}
  seen.add(key);const f={label,b,op,mode,input};f.expected=expected(f);fixtures.push(f);
}
function batch(label,b){for(const op of [0,1,2])add(label,b,op,op===2?0:3);}
for(const w of [true,false])for(let rights=0;rights<16;rights++){
  const b=base(w);b.rights=rights;
  for(const mode of [0,1,2,3]){add('rights-tail',b,0,mode);add('rights-tail',b,1,mode);}add('rights-full',b,2,0);
}
for(const w of [true,false]){
  const h=w?0:56,c=w?1:0;
  for(const file of [1,2,3,5,6])for(let k=0;k<5;k++)for(const color of [0,1]){const b=base(w);put(b,h+file,k,color);batch('path-blocker',b);}
  for(const ks of [true,false])for(const fault of ['missing','wrong-color','wrong-kind']){const b=base(w),s=h+(ks?7:0);clear(b,s);if(fault!=='missing')put(b,s,fault==='wrong-kind'?1:3,fault==='wrong-color'?1-c:c);batch('rook-'+fault,b);}
  for(const file of [2,3,4,5,6]){const b=base(w);put(b,(w?56:0)+file,3,1-c);batch('attack-file-'+file,b);}
  for(const ep of [0,27,64,0xffffffff]){const b=base(w);b.rights=0xffffffff;b.ep=ep;batch('raw-metadata',b);}
}
let seed=0x326ac71;const rng=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;};
for(const w of [true,false])for(let i=0;i<64;i++){
  const b=base(w),h=w?0:56;
  for(let j=0;j<2+i%20;j++){const s=rng()%64;if(occupied(b.sq[s]))continue;if(i%2===0&&[1,2,3,5,6].includes(s-h))continue;put(b,s,rng()%5,rng()%2);}
  b.rights=i%3===0?rng():15;b.ep=rng();batch('mixed-coordinate',b);
}
const destCases=fixtures.filter(f=>f.op===2&&['attack-file-2','attack-file-6'].includes(f.label));assert.equal(destCases.length,4);
assert.ok(destCases.every(f=>{const c=f.label==='attack-file-6'?f.expected.king:f.expected.queen;return c.emit&&inCheck(apply(f.b,c.move),f.b.turn)&&!f.expected.moves.some(m=>JSON.stringify(m)===JSON.stringify(c.move));}));
const both=fixtures.filter(f=>f.op===2&&f.expected.moves.length===2);assert.ok(both.length>0);
function read(output,part){
  const lines=output.trim().split('\n');let p=0,records=0,castles=0,noncastle=0;
  for(const f of part){
    const header=lines[p++];assert.match(header,/^begin \d+$/);const count=Number(header.slice(6));assert.ok(count>=0&&count<1024);
    const moves=[];
    for(let i=0;i<count;i++){
      const line=lines[p++];assert.match(line,/^move(?: \d+){23}$/);const row=line.slice(5).split(' ').map(Number);assert.ok(row.every(x=>Number.isInteger(x)&&x>=0&&x<=0xffffffff));
      const m=row.slice(0,4);assert.ok(m[0]<64&&m[1]<64);assert.deepEqual(row.slice(4),encoded(apply(f.b,m)),'complete child Board: '+f.label);
      moves.push(m);records++;if(m[3]===2)castles++;else noncastle++;
    }
    assert.equal(lines[p++],'end');
    if(f.op===2){
      assert.ok(moves.every(m=>[0,1,2].includes(m[3])),'unexpected raw flag');
      const cm=moves.filter(m=>m[3]===2);assert.deepEqual(cm,f.expected.moves,'full legal_moves castling subset: '+f.label);
      for(const m of cm){const c=m[1]%8===6?f.expected.king:f.expected.queen;assert.ok(c.guard);assert.deepEqual(m,c.move);assert.ok(consistent(apply(f.b,m)));assert.ok(!inCheck(apply(f.b,m),f.b.turn));}
    }else{
      assert.deepEqual(moves,f.expected.moves,'exact filter/chain ordered list: '+f.label);
      const supply=f.expected.input.map(m=>JSON.stringify(m));for(const m of moves){const i=supply.indexOf(JSON.stringify(m));assert.ok(i>=0,'filter invented or duplicated a move');supply.splice(i,1);}
    }
  }
  assert.equal(p,lines.length);return {records,castles,noncastle};
}
function run(binary,fsx){let all='',records=0,castles=0,noncastle=0;for(let i=0;i<fsx.length;i+=64){const part=fsx.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000),r=read(out,part);records+=r.records;castles+=r.castles;noncastle+=r.noncastle;all+=out;}return {records,castles,noncastle,output_sha256:sha(all)};}
function generate(e,label){const c=path.join(tmp,label+'.c');cmd([process.execPath,path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/castle_chain/probe.bend'),'-o',c],180000);return c;}
function build(c,label,flags){const binary=path.join(tmp,label);cmd([cc,'-std=c11','-O2',...flags,c,'-pthread','-lm','-o',binary],180000);return binary;}
function change(p,a,b){const s=fs.readFileSync(p,'utf8');assert.equal(s.split(a).length,2,'nonunique mutation');fs.writeFileSync(p,s.replace(a,b));}
const tracked=()=>Object.fromEntries(fs.readdirSync(suite).filter(f=>f.endsWith('.bend')||f.endsWith('.js')).sort().map(f=>[f,sha(fs.readFileSync(path.join(suite,f)))]));
try{
  const before=tracked(),sourceChess=sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),generated=generate(engine,'baseline'),modes=[];
  const sample=fixtures[0].input.map(String),bad=[['x'],sample.slice(1),['3',...sample.slice(1)],[sample[0],'4',...sample.slice(2)],['-1',...sample.slice(1)],['4294967296',...sample.slice(1)],Array.from({length:65},()=>sample).flat()];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    console.error('BUILD '+mode);const binary=build(generated,mode,flags),out=run(binary,fixtures);
    for(const input of bad){const r=spawnSync(binary,['--threads','1',...input],{encoding:'utf8',timeout:30000,maxBuffer:4<<20});assert.equal(r.error,undefined);assert.equal(r.signal,null);assert.equal(r.status,2);assert.match(r.stderr,/invalid chain/);}
    modes.push({mode,...out,invalid_rejections:bad.length});console.error('PASS '+mode+' '+JSON.stringify(out));
  }
  const mutations=[];
  for(const [name,from,to] of [
    ['forge-scan-tail-castle','scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Nil{}))','scan(bit_squares(64n, U64.is_zero(own), own, Nil{}), b, (table, Con{Ply{0,0,0,2},Nil{}}))'],
    ['duplicate-kingside-stage','filter_prepare(b, castle_side(b, False{}, castle_side(b, True{},','filter_prepare(b, castle_side(b, True{}, castle_side(b, True{},'],
    ['disable-destination-rejection','retain_move(check, m, acc)','retain_move(False{}, m, acc)']]){
    console.error('MUTANT '+name);const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});change(path.join(e,'legal_probe/Chess.bend'),from,to);const binary=build(generate(e,name),name+'-bin',[]);
    let mismatch=null;
    for(let i=0;i<fixtures.length;i+=64){const part=fixtures.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000);try{read(out,part);}catch(error){mismatch={first_batch_index:i,diagnostic:String(error),output_sha256:sha(out)};break;}}
    assert.ok(mismatch,'mutation escaped reference');mutations.push({name,compiled_and_executed:true,rejected:true,...mismatch});
  }
  assert.deepEqual(tracked(),before);assert.equal(sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),sourceChess);assert.deepEqual(verifyCompiler(compiler),identity);
  const byOperation={filter:fixtures.filter(f=>f.op===0).length,filtered_castling_suffix:fixtures.filter(f=>f.op===1).length,full_legal_moves:fixtures.filter(f=>f.op===2).length};
  const report={native_gate:'PASS',compiler_identity:identity,cc:cmd([cc,'--version']).split('\n')[0],fixture_count:fixtures.length,distinct_inputs:seen.size,deduplicated_candidates:duplicates,fixture_sha256:sha(JSON.stringify(fixtures.map(f=>f.input))),cases_by_operation:byOperation,destination_attacked_castles_rejected:destCases.length,both_castles_retained_cases:both.length,complete_child_boards_per_mode:modes[0].records,board_fields_per_mode:modes[0].records*19,modes,mutations,source_sha256s:before,chess_sha256:sourceChess,generated_c_sha256:sha(fs.readFileSync(generated)),scope:'Real current tables and actual filter_prepare, both castle_side stages and full legal_moves. Exact ordered lists for filter/suffix; independent ordered castling subset for full legal_moves, not its complete non-castling move-set correctness. All returned raw child Boards compared. No native compiler proof, general king-safety proof, historical rights or generation completeness claim.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
