// Focused development gate. verify.js also requires the unchanged parent aggregate.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),suite=import.meta.dirname,cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-prefix-proofs-'));
const required=['prefix_widening','prefix_in_allocation','step_no_overflow','blocks_ordered','tables_follow_prefix','logical_end'];
const sha=b=>createHash('sha256').update(b).digest('hex'),code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
const negatives=[];
function invoke(file,timeout=600000){const r=spawnSync(process.execPath,[cli,file],{encoding:'utf8',timeout,maxBuffer:16<<20,
 env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return {status:r.status,text:(r.stdout+r.stderr).trim()};}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
 file=path.resolve(file);const rel=path.relative(boundary,file);
 assert.ok(!rel.split(path.sep).includes('..')&&(rel.startsWith('standalone/')||rel==='bitboard_probe/Sliders.bend'),'proof outside boundary');
 assert.ok(fs.lstatSync(file).isFile(),'nonregular proof');assert.equal(fs.realpathSync(file),file,'symlinked proof');
 if(seen.has(file))return seen;seen.add(file);const s=code(file);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');policy(path.resolve(path.dirname(file),m[1]),boundary,seen);}return seen;
}
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),required);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),required);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'Certified.bend')),/import \.\.\/layout\/PROOF\.bend as LayoutProof/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,a,b){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(a).length,2,`nonunique anchor: ${a}`);fs.writeFileSync(f,s.replace(a,b));}
function copy(name){const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});fs.mkdirSync(path.join(e,'bitboard_probe'));
 fs.copyFileSync(path.join(engine,'bitboard_probe/Sliders.bend'),path.join(e,'bitboard_probe/Sliders.bend'));return{e,r,s:path.join(r,'proofs/prefix')};}
function semantic(name,entry,edit,timeout=150000){const d=copy(name);edit(d);const r=invoke(path.join(d.s,entry),timeout);
 assert.equal(r.status,1,`${name}: must reject\n${r.text}`);assert.match(r.text,/expected[\s\S]*observed/,`${name}: not ordinary proof rejection\n${r.text}`);
 assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault/,'crash is not rejection');
 negatives.push({name,rejected:true,kind:'affected source refinement',entry});console.error('PASS rejection: '+name);
}
function guard(name,edit,check,reason){const d=copy(name);edit(d);assert.throws(()=>check(d),reason);negatives.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js','probe.bend'].map(f=>path.join(suite,f)),path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
 const before=hashes(tracked);clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS: six public prefix laws and importing consumer');
 semantic('reserved-prefix-overlaps-headers','Facts.bend',d=>replace(path.join(d.s,'Spec.bend'),'case 0n: 512','case 0n: 256'));
 semantic('prefix-skips-block-size','Facts.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.add(prefix(p),size(p))','prefix(p)'));
 semantic('widened-prefix-skips-block','Facts.bend',d=>replace(path.join(d.s,'Spec.bend'),'U64.add(wide(p),U64.from_u32(size(p)))','wide(p)'));
 semantic('wrong-geometric-block-size','Size.bend',d=>replace(path.join(d.s,'Spec.bend'),'U32.shln(1,Geometry.population(k))','U32.shln(1,1n+Geometry.population(k))'));
 semantic('actual-next-prefix-increments-one','Builder.bend',d=>replace(path.join(d.r,'Tables.bend'),'U32.add(at, size), fill','U32.inc(at), fill'));
 semantic('actual-offset-header-one-ahead','Builder.bend',d=>replace(path.join(d.r,'Tables.bend'),'U64.from_u32(at))','U64.from_u32(U32.inc(at)))'));
 semantic('actual-mask-header-wrong-key','Builder.bend',d=>replace(path.join(d.r,'Tables.bend'),'a, key, mask)','a, U32.inc(key), mask)'));
 semantic('actual-header-region-shifted','Builder.bend',d=>replace(path.join(d.r,'Tables.bend'),'U32.add(128, key)','U32.add(129, key)'));
 semantic('actual-fill-count-one-too-many','Builder.bend',d=>replace(path.join(d.r,'Tables.bend'),'fill(U32.to_nat(size), at','fill(1n+U32.to_nat(size), at'));
 semantic('scheduled-writes-ignore-prefix','Builder.bend',d=>replace(path.join(d.s,'Spec.bend'),'+at = prefix(k)','+at = 512'));
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law blocks_ordered:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.blocks_ordered(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-size-certificate-producer',d=>replace(path.join(d.s,'Certified.bend'),'import ../layout/PROOF.bend as LayoutProof\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Builder.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-witness',d=>fs.appendFileSync(path.join(d.s,'Builder.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Builder.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Builder.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 const d=copy('unsafe-refinement');replace(path.join(d.s,'Builder.bend'),'def refine(','@unsafe\ndef refine(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe/);
 const warned=invoke(path.join(d.s,'Builder.bend'),60000);assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 negatives.push({name:'unsafe-refinement',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(negatives.length,19);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={focused_gate:'PASS',compiler_revision:PIN.revision,...identity,new_laws:required,new_law_count:6,inherited_gate_run:false,
 negative_controls:negatives,consumer:'PASS',source_sha256s:before,
 scope:'Actual table-loop/full-array equality to an independently bounded prefix schedule, widened step arithmetic and ordered block endpoints; not all normalized-path certificates or final attack geometry'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
