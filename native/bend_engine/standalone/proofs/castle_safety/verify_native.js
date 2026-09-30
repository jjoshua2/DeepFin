// Mandatory-check native verifier. Independent square-set reference reused from the saved chain suite.
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
const engine=path.resolve(suite,'../../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'castle-safety-native-'));
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

function base(w){const h=w?0:56,b={sq:planes(),turn:w?1:0,rights:15,ep:64};put(b,h+4,5,w?1:0);put(b,h,3,w?1:0);put(b,h+7,3,w?1:0);put(b,w?57:1,5,w?0:1);return b;}
function expected(b,m,rays,mode){
  const c=b.turn===1?1:0,ss=b.sq.map((z,i)=>z.c.has(c)&&(z.k.has(5)||Boolean((rays>>BigInt(i))&1n)));
  const mask=ss.reduce((n,on,i)=>on?n|(1n<<BigInt(i)):n,0n),required=m[3]===1||ss[m[0]],check=inCheck(apply(b,m),b.turn);
  return {mask,required,check,owned_king:b.sq[m[0]].k.has(5)&&b.sq[m[0]].c.has(c),
    blockers:(!required||!check)?[m]:[],fast:(!required||!check)?[m,...tails[mode]]:tails[mode],full:!check?[m,...tails[mode]]:tails[mode]};
}
const fixtures=[],seen=new Set();let duplicates=0;
function add(label,b,m,rays,mode){
  assert.ok(consistent(b));for(const c of [0,1])assert.equal(b.sq.filter(s=>s.k.has(5)&&s.c.has(c)).length,1);
  const input=[...m,Number(rays>>32n),Number(rays&0xffffffffn),mode,...encoded(b)],key=JSON.stringify(input);
  if(seen.has(key)){duplicates++;return;}seen.add(key);
  fixtures.push({label,b,m,rays,mode,input,expected:expected(b,m,rays,mode)});
}
const masks=[0n,0xffffffffffffffffn,0xaaaaaaaaaaaaaaaan,0x5555555555555555n];
for(const c of [0,1])for(let src=0;src<64;src++)for(const flag of [0,2])for(const rays of masks){
  const dst=src^1,b={sq:planes(),turn:c,rights:15,ep:64};
  put(b,src,5,c);put(b,(src+32)%64,5,1-c);
  const attacker=(dst+8)%64;if(!occupied(b.sq[attacker]))put(b,attacker,3,1-c);
  add('owned-king-all-squares',b,[src,dst,0,flag],rays,(src+flag)%4);
}
for(const w of [true,false])for(const ks of [true,false])for(const rays of masks)for(const mode of [0,1,2,3]){
  const h=w?0:56,c=w?1:0,src=h+4,dst=h+(ks?6:2),transit=h+(ks?5:3),m=[src,dst,0,2];
  add('route-safe',base(w),m,rays,mode);
  const dest=base(w);put(dest,(w?56:0)+(dst%8),3,1-c);add('destination-attacked',dest,m,rays,mode);
  const through=base(w);put(through,(w?56:0)+(transit%8),3,1-c);add('transit-attacked',through,m,rays,mode);
  const absent=base(w);clear(absent,src);put(absent,h+(w?12:-4),5,c);add('raw-empty-source',absent,m,rays,mode);
  const wrong=copy(absent);put(wrong,src,2,1-c);add('raw-unowned-source',wrong,m,rays,mode);
}
for(const w of [true,false])for(const mode of [0,1,2,3])for(const flag of [0,1]){
  const h=w?0:56,c=w?1:0,b=base(w),src=w?16:40,dst=w?24:32;
  put(b,src,0,c);put(b,(w?56:0)+4,3,1-c);add('nonking-shortcut-boundary',b,[src,dst,0,flag],0n,mode);
}
assert.ok(fixtures.filter(f=>f.expected.owned_king).every(f=>f.expected.required));
assert.ok(fixtures.some(f=>!f.expected.required&&f.expected.check));
assert.ok(fixtures.some(f=>f.label==='destination-attacked'&&f.expected.required&&f.expected.check));
function read(output,part){
  const lines=output.trim().split('\n');let p=0,records=0,owned=0,checked=0,skipped=0;
  for(const f of part){
    const e=f.expected,line=lines[p++];assert.match(line,/^case(?: \d+){4}$/);
    const values=line.slice(5).split(' ').map(Number);
    assert.deepEqual(values,[Number(e.mask>>32n),Number(e.mask&0xffffffffn),Number(e.required),Number(e.check)],'sensitive/required/check '+f.label);
    if(e.owned_king)owned++;if(e.required)checked++;else skipped++;
    const got={};
    for(const name of ['blockers','fast','full']){
      const head=lines[p++];assert.match(head,new RegExp('^'+name+' \\d+$'));
      const count=Number(head.split(' ')[1]);assert.ok(count>=0&&count<=3);const moves=[];
      for(let i=0;i<count;i++){
        const rowtext=lines[p++];assert.match(rowtext,/^move(?: \d+){23}$/);const row=rowtext.slice(5).split(' ').map(Number);
        assert.ok(row.every(x=>Number.isInteger(x)&&x>=0&&x<=0xffffffff));
        const m=row.slice(0,4);assert.deepEqual(row.slice(4),encoded(apply(f.b,m)),'full child Board '+name+' '+f.label);moves.push(m);records++;
      }
      assert.equal(lines[p++],'end');assert.deepEqual(moves,e[name],name+' list '+f.label);got[name]=moves;
    }
    if(e.owned_king){assert.equal(values[2],1);assert.deepEqual(got.fast,got.full,'owned-king path must equal full check');}
  }
  assert.equal(p,lines.length);return {records,owned,checked,skipped};
}
function run(binary,rows){let all='',records=0,owned=0,checked=0,skipped=0;
  for(let i=0;i<rows.length;i+=64){const part=rows.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000),v=read(out,part);all+=out;records+=v.records;owned+=v.owned;checked+=v.checked;skipped+=v.skipped;}
  return {records,owned,checked,skipped,output_sha256:sha(all)};
}
function generate(e,label){const c=path.join(tmp,label+'.c');cmd([process.execPath,path.join(compiler,'bend2/main.ts'),path.join(e,'standalone/proofs/castle_safety/probe.bend'),'-o',c],180000);return c;}
function build(c,label,flags){const binary=path.join(tmp,label);cmd([cc,'-std=c11','-O2',...flags,c,'-pthread','-lm','-o',binary],180000);return binary;}
function change(p,a,b){const text=fs.readFileSync(p,'utf8');assert.equal(text.split(a).length,2,'nonunique mutation');fs.writeFileSync(p,text.replace(a,b));}
function dependencies(){const result={};const walk=p=>{if(p in result)return;const text=fs.readFileSync(p,'utf8');result[p]=sha(text);for(const m of text.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\//);walk(path.resolve(path.dirname(p),m[1]));}};walk(path.join(suite,'probe.bend'));result[path.join(suite,'verify_native.js')]=sha(fs.readFileSync(path.join(suite,'verify_native.js')));return Object.fromEntries(Object.entries(result).map(([p,h])=>[path.relative(engine,p),h]));}
try{
  const before=dependencies(),generated=generate(engine,'baseline'),modes=[];
  const sample=fixtures[0].input.map(String),bad=[['x'],sample.slice(1),['64',...sample.slice(1)],[sample[0],'64',...sample.slice(2)], [...sample.slice(0,6),'4',...sample.slice(7)],['-1',...sample.slice(1)],['4294967296',...sample.slice(1)],Array.from({length:65},()=>sample).flat()];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]){
    console.error('BUILD '+mode);const binary=build(generated,mode,flags),out=run(binary,fixtures);
    for(const input of bad){const rr=spawnSync(binary,['--threads','1',...input],{encoding:'utf8',timeout:30000,maxBuffer:4<<20});assert.equal(rr.error,undefined);assert.equal(rr.signal,null);assert.equal(rr.status,2);assert.match(rr.stderr,/invalid safety/);}
    modes.push({mode,...out,invalid_rejections:bad.length});console.error('PASS '+mode+' '+JSON.stringify(out));
  }
  const mutations=[];
  for(const [name,from,to] of [
    ['omit-kings-from-blocker-mask','sensitive = U64.and(own, U64.or(rays, get_kings(b)))','sensitive = U64.and(own, rays)'],
    ['bypass-required-check','case True{}: filter_step(b, m, r)','case True{}:\n      (table, acc) = r\n      (table, Con{m, acc})'],
    ['check-opponent-instead','filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))','filter_after(m, acc, in_check(table, make_move(b, m), U32.xor(get_turn(b), 1)))']]){
    console.error('MUTANT '+name);const e=path.join(tmp,name);fs.cpSync(engine,e,{recursive:true});change(path.join(e,'legal_probe/Chess.bend'),from,to);const binary=build(generate(e,name),name+'-bin',[]);
    let mismatch=null;for(let i=0;i<fixtures.length;i+=64){const part=fixtures.slice(i,i+64),out=cmd([binary,'--threads','1',...part.flatMap(f=>f.input.map(String))],30000);try{read(out,part);}catch(error){mismatch={first_batch_index:i,diagnostic:String(error),output_sha256:sha(out)};break;}}
    assert.ok(mismatch,'mutation escaped');mutations.push({name,compiled_and_executed:true,rejected:true,...mismatch});
  }
  assert.deepEqual(dependencies(),before);assert.deepEqual(verifyCompiler(compiler),identity);
  const report={native_gate:'PASS',compiler_identity:identity,cc:cmd([cc,'--version']).split('\n')[0],fixture_count:fixtures.length,distinct_inputs:seen.size,deduplicated_candidates:duplicates,fixture_sha256:sha(JSON.stringify(fixtures.map(f=>f.input))),cases_by_label:Object.fromEntries([...new Set(fixtures.map(f=>f.label))].map(label=>[label,fixtures.filter(f=>f.label===label).length])),owned_king_sources:new Set(fixtures.filter(f=>f.expected.owned_king).map(f=>f.m[0])).size,owned_king_cases:modes[0].owned,required_cases:modes[0].checked,shortcut_cases:modes[0].skipped,shortcut_full_disagreement_cases:fixtures.filter(f=>!f.expected.required&&f.expected.check).length,complete_child_boards_per_mode:modes[0].records,board_fields_per_mode:modes[0].records*19,mask_required_check_fields_per_mode:fixtures.length*4,modes,mutations,source_sha256s:before,generated_c_sha256:sha(fs.readFileSync(generated)),scope:'Current Tables.build, actual filter_requires/blockers/fast_step/filter_step/in_check. External square-set mask and geometric attack oracle checks ordered lists and all child Board fields. Direct raw-step domain includes outside-pipeline shortcut counterexamples. No proof of independent attack correctness or full legal-move set completeness.'};
  const json=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],json);console.log(json.trimEnd());
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
