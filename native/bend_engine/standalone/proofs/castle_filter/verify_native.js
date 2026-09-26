// Actual paired producers, final filter and whole legal_moves vs coordinate/set reference.
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
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-filter-native-'));
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
function castle(b,ks){const w=b.turn===1,h=w?0:56,c=w?1:0,src=h+4,dst=h+(ks?6:2),transit=h+(ks?5:3),rook=h+(ks?7:0),right=(w?1:4)*(ks?1:2);
  const guard=(b.rights&right)!==0&&b.sq[src].k.has(5)&&b.sq[src].c.has(c)&&b.sq[rook].k.has(3)&&b.sq[rook].c.has(c)&&(ks?[5,6]:[1,2,3]).every(f=>!occupied(b.sq[h+f]));
  return guard&&!inCheck(b,b.turn)&&!inCheck(apply(b,[src,transit,0,0]),b.turn)?[src,dst,0,2]:null;
}
function both(b,tail){const k=castle(b,true),q=castle(b,false);return [...(q?[q]:[]),...(k?[k]:[]),...tail];}
const dirs=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]];
function sensitive(b){const side=b.turn===1?1:0,king=b.sq.findIndex(z=>z.k.has(5)&&z.c.has(side));assert.ok(king>=0);const res=new Set([king]);
  for(const [dx,dy] of dirs){let x=king%8+dx,y=Math.floor(king/8)+dy;while(x>=0&&x<8&&y>=0&&y<8){const s=y*8+x;if(occupied(b.sq[s])){if(b.sq[s].c.has(side))res.add(s);break;}x+=dx;y+=dy;}}
  return res;
}
function filter(b,ms){const checked=inCheck(b,b.turn),need=sensitive(b),out=[];for(const m of ms){if(!(checked||m[3]===1||need.has(m[0]))||!inCheck(apply(b,m),b.turn))out.unshift(m);}return out;}
function attackSquares(b,s,kind){const x=s%8,y=Math.floor(s/8),out=[];
  if(kind===1){for(const [dx,dy] of [[1,2],[2,1],[-1,2],[-2,1],[1,-2],[2,-1],[-1,-2],[-2,-1]]){const xx=x+dx,yy=y+dy;if(xx>=0&&xx<8&&yy>=0&&yy<8)out.push(yy*8+xx);}return out;}
  if(kind===5){for(const [dx,dy] of dirs){const xx=x+dx,yy=y+dy;if(xx>=0&&xx<8&&yy>=0&&yy<8)out.push(yy*8+xx);}return out;}
  for(const [dx,dy] of dirs){if(kind===2&&(!dx||!dy))continue;if(kind===3&&dx&&dy)continue;let xx=x+dx,yy=y+dy;while(xx>=0&&xx<8&&yy>=0&&yy<8){const target=8*yy+xx;out.push(target);if(occupied(b.sq[target]))break;xx+=dx;yy+=dy;}}return out;
}
function pseudo(b){const side=b.turn===1?1:0,out=[];for(let src=63;src>=0;src--){const cell=b.sq[src];if(!cell.c.has(side))continue;const kind=[...cell.k][0];let targets=[];
  if(kind===0){const x=src%8,y=Math.floor(src/8),dy=side===1?1:-1,first=src+8*dy;
    if(first>=0&&first<64&&!occupied(b.sq[first])){targets.push(first);const second=src+16*dy;if(y===(side===1?1:6)&&!occupied(b.sq[second]))targets.push(second);}
    for(const dx of [-1,1]){const xx=x+dx,yy=y+dy;if(xx<0||xx>7||yy<0||yy>7)continue;const dst=8*yy+xx,other=b.sq[dst];
      const victim=dst+(side===1?-8:8),ep=dst===b.ep&&yy===(side===1?5:2)&&!occupied(other)&&b.sq[victim].k.has(0)&&b.sq[victim].c.has(1-side);
      if((other.c.has(1-side)&&!other.k.has(5))||ep)targets.push(dst);
    }
  }else targets=attackSquares(b,src,kind).filter(t=>!b.sq[t].c.has(side)&&!b.sq[t].k.has(5));
  targets.sort((a,b)=>a-b);for(const dst of targets){const promo=kind===0&&(Math.floor(dst/8)===0||Math.floor(dst/8)===7),flag=kind===0&&dst===b.ep?1:0;if(promo)out.unshift(...[1,2,3,4].map(k=>[src,dst,k,0]));else out.unshift([src,dst,0,flag]);}
 }return out;}
function expected(f){const initial=both(f.b,tails[f.mode]),filtered=filter(f.b,initial),p=pseudo(f.b),full=filter(f.b,both(f.b,p));
  // Independent slow king-check reference agrees with the optimized full-list filter
  // on these finite one-king-per-color fixtures. This is not a universal proof.
  const slow=both(f.b,p).filter(m=>!inCheck(apply(f.b,m),f.b.turn)).reverse();assert.deepEqual(full,slow,'fast/slow external reference divergence');
  return [initial,filtered,full].map(ms=>({moves:ms,rows:ms.map(m=>[...m,...encoded(apply(f.b,m))])}));
}
function base(w){const h=w?0:56,b={sq:planes(),turn:w?1:0,rights:15,ep:64};put(b,h+4,5,w?1:0);put(b,h,3,w?1:0);put(b,h+7,3,w?1:0);put(b,w?57:1,5,w?0:1);return b;}
const fixtures=[],seen=new Set();let duplicates=0;
function add(label,b,mode){assert.ok(consistent(b));for(const color of [0,1])assert.equal(b.sq.filter(s=>s.k.has(5)&&s.c.has(color)).length,1);
 const input=[mode,...encoded(b)],key=JSON.stringify(input);if(seen.has(key)){duplicates++;return;}seen.add(key);const f={label,b,mode,input};f.expected=expected(f);fixtures.push(f);}
for(const w of [true,false])for(let r=0;r<16;r++)for(let mode=0;mode<4;mode++){const b=base(w);b.rights=r;add('rights-and-tails',b,mode);}
for(const w of [true,false]){const h=w?0:56,c=w?1:0;
 for(const file of [1,2,3,5,6])for(const k of [0,1,2,3,4])for(const color of [c,1-c]){const b=base(w);put(b,h+file,k,color);add('blocked-path',b,2);}
 for(const file of [2,3,4,5,6]){const b=base(w);put(b,(w?56:0)+file,3,1-c);add('attacked-file-'+file,b,0);}
 for(const file of [0,7])for(const fault of ['missing','wrong-kind','wrong-color']){const b=base(w);clear(b,h+file);if(fault!=='missing')put(b,h+file,fault==='wrong-kind'?1:3,fault==='wrong-color'?1-c:c);add('rook-'+fault,b,3);}
 for(const rights of [0xffffffff,0x80000000,0xfffffff0])for(const ep of [0,27,64,0xffffffff]){const b=base(w);b.rights=rights;b.ep=ep;add('raw-metadata',b,1);}
}
let seed=0x6c25f317;const rng=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;};
for(const w of [true,false])for(let i=0;i<64;i++){const b=base(w);for(let j=0;j<2+i%12;j++){const s=rng()%64;if(occupied(b.sq[s]))continue;put(b,s,rng()%5,rng()%2);}b.rights=i%2?15:rng();b.ep=rng();add('mixed-board',b,i%4);}
// Every destination-only attacker retained before the final filter must disappear
// from the filtered pair and actual full legal_moves.
const destination=fixtures.filter(f=>['attacked-file-2','attacked-file-6'].includes(f.label));assert.equal(destination.length,4);
for(const f of destination){const dst=(f.b.turn===1?0:56)+Number(f.label.at(-1));assert.ok(f.expected[0].moves.some(m=>m[1]===dst&&m[3]===2));assert.ok(!f.expected[1].moves.some(m=>m[1]===dst&&m[3]===2));assert.ok(!f.expected[2].moves.some(m=>m[1]===dst&&m[3]===2));}
function read(text,fsx){const lines=text.trim().split('\n');let p=0,records=0,castles=0;for(const f of fsx)for(let stage=0;stage<3;stage++){const e=f.expected[stage];assert.equal(lines[p++],`begin ${e.rows.length}`,'wrong stage length: '+f.label+'/'+stage);for(const row of e.rows){assert.equal(lines[p++],'move '+row.join(' '),'wrong complete record: '+f.label+'/'+stage);records++;if(stage===2&&row[3]===2){castles++;assert.ok(consistent(apply(f.b,row.slice(0,4))));}}assert.equal(lines[p++],'end');}assert.equal(p,lines.length);return {records,castles};}
function run(binary,fsx){let all='',records=0,castles=0;for(let i=0;i<fsx.length;i+=64){const part=fsx.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000),r=read(out,part);records+=r.records;castles+=r.castles;all+=out;}return {records,castles,output_sha256:sha(all)};}
function generate(e,label){const c=path.join(tmp,label+'.c');cmd([process.execPath,path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/castle_filter/probe.bend'),'-o',c]);return c;}
function build(c,label,flags){const binary=path.join(tmp,label);cmd([cc,'-std=c11','-O2',...flags,c,'-pthread','-lm','-o',binary]);return binary;}
function change(p,a,b){const s=fs.readFileSync(p,'utf8');assert.equal(s.split(a).length,2,'nonunique mutation');fs.writeFileSync(p,s.replace(a,b));}
const tracked=()=>Object.fromEntries(fs.readdirSync(suite).filter(f=>f.endsWith('.bend')||f.endsWith('.js')).sort().map(f=>[f,sha(fs.readFileSync(path.join(suite,f)))]));
try{
 const before=tracked(),sourceChess=sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),generated=generate(engine,'baseline'),modes=[];
 const sample=fixtures[0].input.map(String),bad=[['x'],sample.slice(1),['4',...sample.slice(1)],['-1',...sample.slice(1)],['4294967296',...sample.slice(1)],Array.from({length:65},()=>sample).flat(),[...sample,'0']];
 for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
  const binary=build(generated,mode,flags),out=run(binary,fixtures);for(const input of bad){const r=spawnSync(binary,['--threads','1',...input],{encoding:'utf8',timeout:30000,maxBuffer:4<<20});assert.equal(r.error,undefined);assert.equal(r.signal,null);assert.equal(r.status,2);assert.match(r.stderr,/invalid castle filter/);}modes.push({mode,...out,invalid_rejections:bad.length});console.error('PASS native '+mode);
 }
 const mutations=[];
 for(const [name,from,to,checkSource] of [
  ['scan-invents-castle-flag','Bool.to_u32(Bool.and(pawn, U32.is_eq(dst, ep_sq)))','2',false],
  ['fast-filter-invents-move','(table, Con{m, acc})','(table, Con{Ply{0,1,0,2}, acc})',false],
  ['bypass-final-check','(table, retain_move(check, m, acc))','(table, retain_move(False{}, m, acc))',true]]){
  const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});change(path.join(e,'legal_probe/Chess.bend'),from,to);
  let sourceRejection=null;if(checkSource){const r=spawnSync(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/castle_filter/consumer.bend')],{encoding:'utf8',timeout:180000,maxBuffer:32<<20,env:{...process.env,TERM:'dumb',BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined);assert.equal(r.signal,null);assert.equal(r.status,1);const diagnostic=(r.stdout+r.stderr).trim();assert.match(diagnostic,/expected[\s\S]*observed/);assert.match(diagnostic,/Location: Filter\.after_then\b/);sourceRejection={rejected:true,location:'Filter.after_then',diagnostic_sha256:sha(diagnostic),diagnostic};}
  const binary=build(generate(e,name),name+'-bin',[]);let mismatch=null;
  for(const f of fixtures){const out=cmd([binary,'--threads','1',...f.input.map(String)],30000);try{read(out,[f]);}catch(error){mismatch={label:f.label,input:f.input,actual:out,expected_rows:f.expected.map(x=>x.rows),diagnostic:String(error)};break;}}
  assert.ok(mismatch,'mutation escaped independent oracle');mutations.push({name,compiled_and_executed:true,rejected:true,source_rejection:sourceRejection,...mismatch});console.error('PASS mutation '+name);
 }
 assert.deepEqual(tracked(),before);assert.equal(sha(fs.readFileSync(path.join(engine,'legal_probe/Chess.bend'))),sourceChess);assert.deepEqual(verifyCompiler(compiler),identity);
 const categories={};for(const f of fixtures)categories[f.label]=(categories[f.label]||0)+1;
 const report={native_gate:'PASS',compiler_identity:identity,cc:cmd([cc,'--version']).split('\n')[0],fixture_count:fixtures.length,distinct_inputs:seen.size,deduplicated_candidates:duplicates,fixture_sha256:sha(JSON.stringify(fixtures.map(f=>f.input))),cases_by_category:categories,stages_per_request:3,destination_attacked_removed_cases:destination.length,complete_child_boards_per_mode:modes[0].records,complete_board_fields_per_mode:modes[0].records*19,full_list_castles_per_mode:modes[0].castles,modes,mutations,source_sha256s:before,chess_sha256:sourceChess,generated_c_sha256:sha(fs.readFileSync(generated)),scope:'Actual paired castle_side, filter_prepare and complete legal_moves with current Tables.build. Ordered outputs and full child Boards versus independent coordinate/set model; finite slow/fast reference comparison. Exact continuation refinement rejects bypassed filter branching; the provenance result is not an independent semantic king-safety theorem.'};
 const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
