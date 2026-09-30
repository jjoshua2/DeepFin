// Three computed-content laws. The full verify.js additionally retains the parent gate.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),controlsOnly=args.includes('--controls-only'),cleanArgs=args.filter(x=>x!=='--controls-only');
assert.equal(args.filter(x=>x==='--controls-only').length,controlsOnly?1:0);
assert.ok(cleanArgs.length===1||(cleanArgs.length===3&&cleanArgs[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler=path.resolve(cleanArgs[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const root=path.resolve(suite,'../..'),engine=path.dirname(root),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-fill-contents-laws-'));
const required=['bounded_fill_entry','bounded_zero_fill_read','full_block_entry'],negatives=[];
const sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
function invoke(file,timeout=120000){const r=spawnSync(process.execPath,[cli,file],{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return {status:r.status,text:(r.stdout+r.stderr).trim()};}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/prefix\/PROOF\.bend as PrefixProof/);
 assert.match(code(path.join(s,'Schedule.bend')),/import \.\.\/PROOF\.bend as SequenceProof/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function policy(file,boundary,seen=new Set()){
 file=path.resolve(file);const rel=path.relative(boundary,file);
 assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='bitboard_probe/Sliders.bend'),'escaped proof boundary');
 assert.ok(fs.lstatSync(file).isFile(),'nonregular proof');assert.equal(fs.realpathSync(file),file,'symlinked proof');
 if(seen.has(file))return seen;seen.add(file);const s=code(file);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){
  if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign or unexpected proof import');
  policy(path.resolve(path.dirname(file),m[1]),boundary,seen);
 }return seen;
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,a,b){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(a).length,2,`nonunique mutation: ${a}`);fs.writeFileSync(f,s.replace(a,b));}
function section(f,start,end,edit){const s=fs.readFileSync(f,'utf8'),l=s.indexOf(start),r=s.indexOf(end,l);assert.ok(l>=0&&r>l);const old=s.slice(l,r),changed=edit(old);assert.notEqual(old,changed);fs.writeFileSync(f,s.slice(0,l)+changed+s.slice(r));}
function copy(name){const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});fs.mkdirSync(path.join(e,'bitboard_probe'));
 fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
 return {e,r,s:path.join(r,'proofs/contents')};}
function semantic(name,entry,edit){const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry));
 assert.equal(r.status,1,`${name}: must reject\n${r.text}`);assert.match(r.text,/expected[\s\S]*observed/,`${name}: not ordinary proof rejection\n${r.text}`);
 assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault|no such file|TODOs found/,'crash or missing dependency is not semantic rejection');
 const location=r.text.match(/Location:\s*([^\n]+)/)?.[1]||null;assert.ok(location,'missing failure location');
 assert.doesNotMatch(r.text,/consumed more than once|unbound variable|unknown variable/,'typing harness failure is not semantic rejection');
 const expected={
  'actual-fill-noop-write':'../storage/Build.fill','actual-fill-wrong-value':'../storage/Build.fill',
  'address-schedule-stuck':'address','state-schedule-stuck':'model','state-start-discarded':'model',
  'inclusive-entry-bound':'bounded','returned-original-buffer':'pair_entry_model',
  'claimed-first-value-zero':'model','claimed-next-state-instead-of-selected':'model'};
 assert.equal(location,expected[name],`wrong proof layer for ${name}\n${r.text}`);
 negatives.push({name,rejected:true,kind:name.startsWith('actual-fill')?'inherited implementation dependency':'new contents refinement',entry,location,diagnostic_sha256:sha(r.text),diagnostic_bytes:Buffer.byteLength(r.text),diagnostic_excerpt:r.text.length<=2400?r.text:r.text.slice(0,1600)+'\n[expanded diagnostic elided; SHA-256 and length retained]\n'+r.text.slice(-800)});console.error('PASS rejection '+name+' at '+location);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
 const before=hashes(tracked);
 if(!controlsOnly)clean(invoke(path.join(suite,'consumer.bend'),900000));
 semantic('actual-fill-noop-write','First.bend',d=>replace(path.join(d.r,'Tables.bend'),'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))','a = a'));
 semantic('actual-fill-wrong-value','First.bend',d=>replace(path.join(d.r,'Tables.bend'),'a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))','a = Array.set(U64, a, at, U64.zero())'));
 semantic('address-schedule-stuck','Schedule.bend',d=>replace(path.join(d.s,'Spec.bend'),'case 1n+p: address(p,U32.inc(at))','case 1n+p: address(p,at)'));
 semantic('state-schedule-stuck','Entries.bend',d=>replace(path.join(d.s,'Spec.bend'),'case 1n+p: state(p,mask,Subsets.next(subset,mask))','case 1n+p: state(p,mask,subset)'));
 semantic('state-start-discarded','Entries.bend',d=>replace(path.join(d.s,'Spec.bend'),'case 0n: subset','case 0n: U64.zero()'));
 semantic('inclusive-entry-bound','Entries.bend',d=>section(path.join(d.s,'Entries.bend'),'def bounded(','\n  +count =',s=>s.replace('Nat.is_lt(i,n)','Nat.is_le(i,n)')));
 semantic('returned-original-buffer','Actual.bend',d=>section(path.join(d.s,'Actual.bend'),'def pair_entry_model(','def pair_entry_lift(',s=>s.replace(
  '(Tables.fill(n,at,sq,bishop,mask,U64.zero(),O.pack(c)),Tables.slider','(O.pack(c),Tables.slider')));
 semantic('claimed-first-value-zero','First.bend',d=>section(path.join(d.s,'First.bend'),'def model(',"  +v = Tables.slider",s=>s.replace('== Tables.slider(sq,bishop,subset,False{}) : U64}','== U64.zero() : U64}')));
 semantic('claimed-next-state-instead-of-selected','Entries.bend',d=>section(path.join(d.s,'Entries.bend'),'def model(','  match i:',s=>s.replace('Spec.state(i,mask,subset)','Spec.state(1n+i,mask,subset)')));
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law full_block_entry:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.full_block_entry(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-prefix-proof',d=>replace(path.join(d.s,'Certified.bend'),'import ../prefix/PROOF.bend as PrefixProof\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Entries.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-proof',d=>fs.appendFileSync(path.join(d.s,'Entries.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign or unexpected/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Entries.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Entries.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 const d=copy('unsafe-proof');replace(path.join(d.s,'Entries.bend'),'def bounded(','@unsafe\ndef bounded(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe dependency/);
 const warned=invoke(path.join(d.s,'Actual.bend'));assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 negatives.push({name:'unsafe-proof',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(negatives.length,17);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={...(controlsOnly?{controls_gate:'PASS'}:{focused_gate:'PASS'}),compiler_revision:PIN.revision,...identity,
 new_laws:required,new_law_count:3,inherited_gate_run:false,consumer:controlsOnly?'not run by this command':'PASS',negative_controls:negatives,
 source_sha256s:before,scope:'Actual bounded fill computed-entry values and full returned pairs, including certified chess blocks; not later table/header/extras preservation or independent ray equality'};
 const text=JSON.stringify(report,null,2)+'\n';if(cleanArgs[2])fs.writeFileSync(cleanArgs[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
