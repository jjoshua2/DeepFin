// Bounded, independent coordinate/set reference for actual castle_side output.
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
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-emission-native-'));
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
function expected(f){const {b,ks,mode}=f,w=b.turn===1,home=w?0:56,side=w?1:0,src=home+4,dst=home+(ks?6:2),transit=home+(ks?5:3),rook=home+(ks?7:0),right=(w?1:4)*(ks?1:2);
  const cells=(ks?[5,6]:[1,2,3]).map(s=>s+home);
  const guard=(b.rights&right)!==0&&b.sq[src].k.has(5)&&b.sq[src].c.has(side)&&b.sq[rook].k.has(3)&&b.sq[rook].c.has(side)&&cells.every(s=>!occupied(b.sq[s]));
  const initialCheck=guard?inCheck(b,b.turn):null;
  const transitCheck=guard?inCheck(apply(b,[src,transit,0,0]),b.turn):null;
  const emit=guard&&!initialCheck&&!transitCheck,move=[src,dst,0,2];
  return {guard,initialCheck,transitCheck,emit,move,rows:[...(emit?[move]:[]),...tails[mode]].map(m=>[...m,...encoded(apply(b,m))])};
}
function base(w,ks){const h=w?0:56,b={sq:planes(),turn:w?1:0,rights:15,ep:64};put(b,h+4,5,w?1:0);put(b,h+(ks?7:0),3,w?1:0);put(b,w?56:0,5,w?0:1);return b;}
const fixtures=[],seen=new Set();let duplicates=0;
function add(label,b,ks,mode){assert.ok(consistent(b));const input=[ks?1:0,mode,...encoded(b)],key=JSON.stringify(input);if(seen.has(key)){duplicates++;return;}seen.add(key);const f={label,b,ks,mode,input};f.expected=expected(f);fixtures.push(f);}
// Rights, all four accumulator shapes, every route. Missing rights is first mutation witness.
for(const w of [true,false])for(const ks of [true,false])for(let rights=0;rights<16;rights++)for(let mode=0;mode<4;mode++){const b=base(w,ks);b.rights=rights;add('rights-tail',b,ks,mode);}
for(const w of [true,false])for(const ks of [true,false]){const h=w?0:56,c=w?1:0;
  for(const role of ['king','rook'])for(const fault of ['missing','wrong-color','wrong-kind']){const b=base(w,ks),s=h+(role==='king'?4:(ks?7:0));clear(b,s);if(fault!=='missing')put(b,s,fault==='wrong-kind'?1:(role==='king'?5:3),fault==='wrong-color'?1-c:c);add(`${role}-${fault}`,b,ks,2);}
  for(const file of (ks?[5,6]:[1,2,3]))for(let k=0;k<6;k++)for(const color of [c,1-c]){const b=base(w,ks);put(b,h+file,k,color);add('path-blocker',b,ks,3);}
  // Remote rook attacks separately target initial, transit, destination, or b-file.
  for(const file of [4,ks?5:3,ks?6:2,1]){const b=base(w,ks);put(b,(w?56:0)+file,3,1-c);add('attack-file-'+file,b,ks,1);}
  for(const rights of [0xffffffff,0x80000000,0xfffffff0])for(const ep of [0,27,64,0xffffffff]){const b=base(w,ks);b.rights=rights;b.ep=ep;add('raw-metadata',b,ks,2);}
  // Knight transit attacks: the attack reference does not require enemy legal moves.
  const b=base(w,ks),target=h+(ks?5:3),tx=target%8,ty=Math.floor(target/8);
  const a=(ty+(w?2:-2))*8+tx-1;put(b,a,1,1-c);add('knight-transit',b,ks,0);
}
let seed=0x526ca571;const rng=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;};
for(const w of [true,false])for(const ks of [true,false])for(let i=0;i<96;i++){const b=base(w,ks),h=w?0:56;for(let j=0;j<2+i%18;j++){const s=rng()%64;if(occupied(b.sq[s]))continue;if(i%2===0&&(ks?[5,6]:[1,2,3]).includes(s-h))continue;put(b,s,rng()%5,rng()%2);}b.rights=i%3===0?rng():15;b.ep=rng();add('mixed-coordinate',b,ks,i%4);}
// Demonstrate separation of castle_side from legal_moves' final destination check.
const destinationOnly=fixtures.filter(f=>f.label==='attack-file-'+(f.ks?6:2));assert.equal(destinationOnly.length,4);assert.ok(destinationOnly.every(f=>f.expected.emit&&inCheck(apply(f.b,f.expected.move),f.b.turn)));
const attackedTransit=fixtures.filter(f=>f.label==='attack-file-'+(f.ks?5:3));assert.ok(attackedTransit.every(f=>!f.expected.emit&&f.expected.guard));
function read(text,fsx){const lines=text.trim().split('\n');let p=0,total=0;for(const f of fsx){const e=f.expected;assert.equal(lines[p++],`begin ${e.rows.length}`,'wrong output-list length: '+f.label);for(const row of e.rows){assert.equal(lines[p++],'move '+row.join(' '),'wrong move or complete Board: '+f.label);total++;}assert.equal(lines[p++],'end');}assert.equal(p,lines.length);return total;}
function run(binary,fsx){let all='',records=0;for(let i=0;i<fsx.length;i+=64){const part=fsx.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000);records+=read(out,part);all+=out;}return {records,output_sha256:sha(all)};}
function generate(e,label){const c=path.join(tmp,label+'.c');cmd([process.execPath,path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/castle_emission/probe.bend'),'-o',c]);return c;}
function build(c,label,flags){const binary=path.join(tmp,label);cmd([cc,'-std=c11','-O2',...flags,c,'-pthread','-lm','-o',binary]);return binary;}
function change(p,a,b){const s=fs.readFileSync(p,'utf8');assert.equal(s.split(a).length,2,'nonunique mutation');fs.writeFileSync(p,s.replace(a,b));}
const tracked=()=>Object.fromEntries(fs.readdirSync(suite).filter(f=>f.endsWith('.bend')||f.endsWith('.js')).sort().map(f=>[f,sha(fs.readFileSync(path.join(suite,f)))]));
try{
  const before=tracked(),sourceChess=sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),generated=generate(engine,'baseline'),modes=[];
  const sample=fixtures[0].input.map(String),bad=[['x'],sample.slice(1),['2',...sample.slice(1)],[sample[0],'4',...sample.slice(2)],['-1',...sample.slice(1)],['4294967296',...sample.slice(1)],Array.from({length:65},()=>sample).flat()];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    const binary=build(generated,mode,flags),out=run(binary,fixtures);for(const input of bad){const r=spawnSync(binary,['--threads','1',...input],{encoding:'utf8',timeout:30000,maxBuffer:4<<20});assert.equal(r.error,undefined);assert.equal(r.signal,null);assert.equal(r.status,2);assert.match(r.stderr,/invalid emission/);}
    modes.push({mode,...out,invalid_rejections:bad.length});
  }
  const mutations=[];
  for(const [name,from,to] of [
    ['bypass-producer-guard','  castle_checked(ok, table, b, src, dst, transit, acc)','  castle_checked(True{}, table, b, src, dst, transit, acc)'],
    ['omit-transit-from-path-mask','Bool.pick(U32, king_side, 96, 14)','Bool.pick(U32, king_side, 64, 14)'],
    ['wrong-king-destination','dst = U32.add(home, Bool.pick(U32, king_side, 6, 2))','dst = U32.add(home, Bool.pick(U32, king_side, 7, 2))']]){
    const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});change(path.join(e,'legal_probe/Chess.bend'),from,to);const binary=build(generate(e,name),name+'-bin',[]);
    let mismatch=null;
    for(const f of fixtures){const out=cmd([binary,'--threads','1',...f.input.map(String)],30000);try{read(out,[f]);}catch(error){mismatch={label:f.label,input:f.input,actual:out,expected_rows:f.expected.rows,diagnostic:String(error)};break;}}
    assert.ok(mismatch,'mutation escaped independent oracle');mutations.push({name,compiled_and_executed:true,rejected:true,...mismatch});
  }
  assert.deepEqual(tracked(),before);assert.equal(sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),sourceChess);assert.deepEqual(verifyCompiler(compiler),identity);
  const categories={};for(const f of fixtures)categories[f.label]=(categories[f.label]||0)+1;
  const report={native_gate:'PASS',compiler_identity:identity,cc:cmd([cc,'--version']).split('\n')[0],fixture_count:fixtures.length,distinct_inputs:seen.size,deduplicated_candidates:duplicates,fixture_sha256:sha(JSON.stringify(fixtures.map(f=>f.input))),cases_by_category:categories,guard_true_cases:fixtures.filter(f=>f.expected.guard).length,emitted_cases:fixtures.filter(f=>f.expected.emit).length,destination_attacked_still_retained:destinationOnly.length,complete_child_boards_per_mode:modes[0].records,modes,mutations,source_sha256s:before,chess_sha256:sourceChess,generated_c_sha256:sha(fs.readFileSync(generated)),scope:'Actual castle_side with current Tables.build, exact ordered output including inherited tails and complete raw child Boards. Independent coordinate/set guard and start/transit attack oracle. Not final legal_moves, complete attack proof, legal reachability, or performance.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
