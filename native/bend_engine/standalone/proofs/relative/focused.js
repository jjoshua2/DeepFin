// Six relative-address contracts. verify.js additionally requires the unchanged parent.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2),controlsOnly=args.includes('--controls-only');
const cleanArgs=args.filter(x=>x!=='--controls-only');
assert.ok(cleanArgs.length===1||(cleanArgs.length===3&&cleanArgs[1]==='--report'),'usage: focused.js COMPILER [--controls-only] [--report FILE]');
assert.equal(args.filter(x=>x==='--controls-only').length,controlsOnly?1:0);
const compiler=path.resolve(cleanArgs[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const root=path.resolve(suite,'../..'),engine=path.dirname(root),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-relative-proofs-'));
const required=['relative_address_bounds','relative_addition_exact','lookup_address_bounds',
 'ordered_block_addresses','ordered_block_address_paths','lookup_other_block_write'];
const sha=b=>createHash('sha256').update(b).digest('hex');
const code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
const negatives=[];
function invoke(file,checker=cli,timeout=900000){
 const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return {status:r.status,text:(r.stdout+r.stderr).trim()};
}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
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
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/prefix\/PROOF\.bend as PrefixProof/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/layout\/PROOF\.bend as LayoutProof/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,a,b){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(a).length,2,`nonunique mutation ${a}`);fs.writeFileSync(f,s.replace(a,b));}
function copy(name,withCompiler=false){
 const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});fs.mkdirSync(path.join(e,'bitboard_probe'));
 fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));
 const d={e,r,s:path.join(r,'proofs/relative'),checker:cli};
 if(withCompiler){const c=path.join(e,'compiler');fs.mkdirSync(c);for(const f of ['bend.ts','comp.ts','base.bend','main.ts'])fs.copyFileSync(path.join(compiler,'bend2',f),path.join(c,f));d.checker=path.join(c,'main.ts');d.base=path.join(c,'base.bend');}
 return d;
}
function semantic(name,entry,edit,withCompiler=false,expectedLocation=null){
 const d=copy(name,withCompiler);edit(d);const r=invoke(path.join(d.s,entry),d.checker,60000);
 assert.equal(r.status,1,`${name}: must reject\n${r.text}`);assert.match(r.text,/expected[\s\S]*observed/,`${name}: not ordinary semantic rejection\n${r.text}`);
 assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault|cannot infer|an annotated term|parse error|no such file|unfilled law|unbound variable|consumed twice/i,'invalid program or crash is not semantic rejection');
 const location=r.text.match(/^Location:\s*(.*)$/m)?.[1];assert.ok(location,'missing rejection location');
 if(expectedLocation)assert.match(location,expectedLocation,'wrong rejection layer');
 const diagnostic=r.text.replaceAll(d.e,'<disposable-source>');
 negatives.push({name,rejected:true,kind:'affected source refinement',entry,failure_location:location,diagnostic_sha256:sha(diagnostic)});console.error('PASS rejection '+name);
}
function guard(name,edit,check){const d=copy(name);edit(d);assert.throws(()=>check(d));negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),path.join(root,'verify_compiler.js'),path.join(root,'toolchain.json')]);
 const before=hashes(tracked);
 if(!controlsOnly){clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS six public laws and importing consumer');}
 semantic('relative-address-one-ahead','Bounds.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.add(Prefix.prefix(k),r)','U32.inc(U32.add(Prefix.prefix(k),r))'));
 semantic('relative-address-ignores-offset','Bounds.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.add(Prefix.prefix(k),r)','Prefix.prefix(k)'));
 semantic('incorrect-incoming-addition-carry','Add.bend',d=>replace(path.join(d.s,'Add.bend'),'case U32{x} U32{y}: carry(32n,x,y,False{})','case U32{x} U32{y}: carry(32n,x,y,True{})'));
 semantic('inclusive-relative-order','Add.bend',d=>replace(path.join(d.s,'Add.bend'),
  'lt: {Cmp.is_lt(Word.cmp(n,x,y)) == True{} : Bool}) ->','lt: {Cmp.is_le(Word.cmp(n,x,y)) == True{} : Bool}) ->'));
 semantic('wrong-order-coherence','Order.bend',d=>replace(path.join(d.s,'Order.bend'),'case LT{} LT{} GT{}: False{}','case LT{} LT{} GT{}: True{}'));
 semantic('endpoint-upper-replaced-by-lower','Bounds.bend',d=>replace(path.join(d.s,'Bounds.bend'),'  (lo,hi) = pair\n  hi','  (lo,hi) = pair\n  lo'));
 semantic('query-routes-to-write-address','Frame.bend',d=>replace(path.join(d.s,'Frame.bend'),
  'Actual.paths(a,S.address(i,ri),S.address(j,rj),shape','Actual.paths(a,S.address(i,ri),S.address(i,ri),shape'));
 semantic('frame-returns-original-buffer','Frame.bend',d=>replace(path.join(d.s,'Frame.bend'),
  '(Array.set(U64,a,S.address(i,ri),v),Separation.value','(a,Separation.value'));
 semantic('actual-wide-add-discards-carry','Add.bend',d=>replace(d.base,
  'carry = Bool.to_u32(U32.is_lt(lo, alo))','carry = U32.from_nat(0n)'),true,/safe_from_wide/);
 semantic('actual-wide-add-shifts-low-half','Add.bend',d=>replace(d.base,
  'U64{lo, U32.add(U32.add(ahi, bhi), carry)}','U64{U32.inc(lo), U32.add(U32.add(ahi, bhi), carry)}'),true,/widening/);
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law lookup_other_block_write:','def omitted:'),d=>manifest(d.s));
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.lookup_other_block_write(','def omitted('),d=>manifest(d.s));
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s));
 guard('missing-prefix-certificate-producer',d=>replace(path.join(d.s,'Certified.bend'),'import ../prefix/PROOF.bend as PrefixProof\n',''),d=>manifest(d.s));
 guard('missing-lookup-certificate-producer',d=>replace(path.join(d.s,'Certified.bend'),'import ../layout/PROOF.bend as LayoutProof\n',''),d=>manifest(d.s));
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s));
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Add.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e));
 guard('foreign-witness',d=>fs.appendFileSync(path.join(d.s,'Add.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e));
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Add.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Add.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e));
 const d=copy('unsafe-frame');replace(path.join(d.s,'Frame.bend'),'def write(','@unsafe\ndef write(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e));const warned=invoke(path.join(d.s,'Frame.bend'),cli,60000);
 assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 negatives.push({name:'unsafe-frame',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(negatives.length,20);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={...(controlsOnly?{control_gate:'PASS'}:{focused_gate:'PASS'}),compiler_revision:PIN.revision,...identity,
  new_laws:required,new_law_count:6,inherited_gate_run:false,consumer:controlsOnly?'NOT RUN':'PASS',
  negative_controls:negatives,source_sha256s:before,
  scope:'Actual U32 prefix-plus-relative-index arithmetic, arbitrary-occupancy PEXT bounds and cross-block complete-array framing; not stored-header/final contents or full fill-clear certificates'};
 const text=JSON.stringify(report,null,2)+'\n';if(cleanArgs[2])fs.writeFileSync(cleanArgs[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
