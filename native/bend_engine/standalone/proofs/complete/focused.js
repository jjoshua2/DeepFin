// Focused development check only. verify.js retains the entire parent gate.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: focused.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),suite=import.meta.dirname;
const root=path.resolve(suite,'../..'),engine=path.dirname(root),cli=path.join(compiler,'bend2/main.ts');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-complete-routes-'));
const names=['allocation_complete','table_pipeline_complete','bounded_complete_routes','bounded_other_index_write'];
const sha=b=>createHash('sha256').update(b).digest('hex'),code=f=>fs.readFileSync(f,'utf8').split('\n').map(l=>l.split('#')[0]).join('\n');
const controls=[];
function invoke(file,checker=cli){const r=spawnSync(process.execPath,[checker,file],{encoding:'utf8',timeout:60000,maxBuffer:16<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);return{status:r.status,text:(r.stdout+r.stderr).trim()};}
function clean(r){assert.equal(r.status,0,r.text);assert.equal(r.text,'All terms check.',r.text);}
function policy(file,boundary,seen=new Set()){
 file=path.resolve(file);const rel=path.relative(boundary,file);
 assert.ok(!rel.split(path.sep).includes('..')&&rel.startsWith('standalone/'),'escaped proof boundary');
 assert.ok(fs.lstatSync(file).isFile(),'nonregular proof');assert.equal(fs.realpathSync(file),file,'symlinked proof');
 if(seen.has(file))return seen;seen.add(file);const s=code(file);assert.doesNotMatch(s,/@unsafe|\?/,'unsafe dependency or proof hole');
 for(const m of s.matchAll(/^\s*import\s+(\S+)/gm)){if(m[1]==='Base')continue;assert.match(m[1],/^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/,'foreign import');policy(path.resolve(path.dirname(file),m[1]),boundary,seen);}return seen;
}
function manifest(s){
 assert.deepEqual([...code(path.join(s,'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m=>m[1]),names);
 assert.deepEqual([...code(path.join(s,'PROOF.bend')).matchAll(/^def Laws\.(\w+)\(/gm)].map(m=>m[1]),names);
 assert.match(code(path.join(s,'PROOF.bend')),/import \.\/LAWS\.bend as Laws/);
 assert.match(code(path.join(s,'consumer.bend')),/import \.\/PROOF\.bend as Proof/);
}
function hashes(files){return Object.fromEntries([...files].sort().map(f=>[path.relative(engine,f),sha(fs.readFileSync(f))]));}
function replace(f,a,b){const s=fs.readFileSync(f,'utf8');assert.equal(s.split(a).length,2,`nonunique mutation ${a}`);fs.writeFileSync(f,s.replace(a,b));}
function copy(name,base=false){const e=path.join(temp,name),r=path.join(e,'standalone');fs.mkdirSync(r,{recursive:true});fs.cpSync(path.join(root,'proofs'),path.join(r,'proofs'),{recursive:true});
 for(const f of ['Tables.bend','Subsets.bend'])fs.copyFileSync(path.join(root,f),path.join(r,f));
 const d={e,r,s:path.join(r,'proofs/complete'),cli};
 if(base){fs.cpSync(path.join(compiler,'bend2'),path.join(e,'compiler/bend2'),{recursive:true});d.cli=path.join(e,'compiler/bend2/main.ts');d.base=path.join(e,'compiler/bend2/base.bend');}return d;}
function reject(name,entry,edit,base=false){const d=copy(name,base);edit(d);const r=invoke(path.join(d.s,entry),d.cli);
 assert.equal(r.status,1,`${name}\n${r.text}`);assert.match(r.text,/expected[\s\S]*observed/,`${name}: not semantic rejection\n${r.text}`);assert.doesNotMatch(r.text,/RangeError|Maximum call stack|Segmentation fault/,'crash is not rejection');
 controls.push({name,rejected:true,kind:'affected source refinement',entry});console.error('PASS rejection '+name);}
function guard(name,edit,test,reason){const d=copy(name);edit(d);assert.throws(()=>test(d),reason);controls.push({name,rejected:true,kind:'manifest/import policy'});}
try{
 manifest(suite);const graph=policy(path.join(suite,'consumer.bend'),engine);
 const tracked=new Set([...graph,...['focused.js','verify.js','verify_native.js'].map(f=>path.join(suite,f)),path.join(root,'proofs/normalization/probe.bend'),path.join(root,'toolchain.json'),path.join(root,'verify_compiler.js')]);
 const before=hashes(tracked);clean(invoke(path.join(suite,'consumer.bend')));console.error('PASS four laws and importing consumer');
 reject('incomplete-shape-specification','Actual.bend',d=>replace(path.join(d.s,'Spec.bend'),'O.Node{child,child}','O.Node{child,O.Leaf{}}'));
 reject('cross-branch-is-not-separated','Core.bend',d=>replace(path.join(d.r,'proofs/separation/Spec.bend'),'case True{} False{}: True{}','case True{} False{}: False{}'));
 reject('right-query-not-rebased','Core.bend',d=>replace(path.join(d.r,'proofs/separation/Spec.bend'),
   'apart(ys,U32.shr(n),U32.sub(i,U32.shr(n)),U32.sub(j,U32.shr(n)))',
   'apart(ys,U32.shr(n),U32.sub(i,U32.shr(n)),j)'));
 function law_edit(d,from,to){const f=path.join(d.s,'LAWS.bend');const s=fs.readFileSync(f,'utf8');const start=s.indexOf('law bounded_complete_routes:');const end=s.indexOf('law bounded_other_index_write:');const part=s.slice(start,end);assert.equal(part.split(from).length,2);fs.writeFileSync(f,s.slice(0,start)+part.replace(from,to)+s.slice(end));}
 reject('equal-indices-admitted','PROOF.bend',d=>law_edit(d,'U32.is_eq(i,j) == False{}','U32.is_eq(i,j) == True{}'));
 reject('inclusive-bound-admits-alias','PROOF.bend',d=>law_edit(d,'U32.is_lt(i,131072)','U32.is_le(i,131072)'));
 reject('capacity-alone-is-insufficient','PROOF.bend',d=>law_edit(d,'O.shape(a) == Spec.full(17n) : O.Shape','S.size(O.shape(a)) == 131072 : U32'));
 reject('actual-allocation-ragged','Actual.bend',d=>replace(d.base,
   'ANode{[v : T^p], [v : T^p]}','ANode{[v : T^p], ALeaf{v}}'),true);
 reject('actual-table-zero-case-destroys-shape','Actual.bend',d=>{
   const f=path.join(d.r,'Tables.bend'),s=fs.readFileSync(f,'utf8'),i=s.indexOf('def tables(');assert.ok(i>=0);
   const rest=s.slice(i);assert.ok(rest.includes('case 0n: a'));fs.writeFileSync(f,s.slice(0,i)+rest.replace('case 0n: a','case 0n: ALeaf{U64.zero()}'));
 });
 guard('omitted-law',d=>replace(path.join(d.s,'LAWS.bend'),'law allocation_complete:','def omitted:'),d=>manifest(d.s),/./);
 guard('omitted-proof',d=>replace(path.join(d.s,'PROOF.bend'),'def Laws.allocation_complete(','def omitted('),d=>manifest(d.s),/./);
 guard('missing-laws-import',d=>replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''),d=>manifest(d.s),/./);
 guard('missing-consumer-proof',d=>replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''),d=>manifest(d.s),/./);
 guard('proof-hole',d=>fs.appendFileSync(path.join(d.s,'Arithmetic.bend'),'\n?TODO\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/proof hole/);
 guard('foreign-witness',d=>fs.appendFileSync(path.join(d.s,'Arithmetic.bend'),'\nimport "./oracle.c"\n'),d=>policy(path.join(d.s,'consumer.bend'),d.e),/foreign/);
 guard('symlinked-proof',d=>{const f=path.join(d.s,'Arithmetic.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Arithmetic.bend.original',f);},d=>policy(path.join(d.s,'consumer.bend'),d.e),/nonregular/);
 const d=copy('unsafe-paths');replace(path.join(d.s,'Actual.bend'),'def paths(','@unsafe\ndef paths(');
 assert.throws(()=>policy(path.join(d.s,'consumer.bend'),d.e),/unsafe/);const warned=invoke(path.join(d.s,'PROOF.bend'));assert.equal(warned.status,0);assert.match(warned.text,/unsafe or foreign/);assert.throws(()=>clean(warned));
 controls.push({name:'unsafe-paths',rejected:true,kind:'policy and exact output',raw_cli_status:warned.status});
 assert.equal(controls.length,16);assert.deepEqual(hashes(tracked),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={focused_gate:'PASS',compiler_revision:PIN.revision,...identity,new_law_count:4,new_laws:names,inherited_gate_run:false,negative_controls:controls,consumer:'PASS',source_sha256s:before,
  scope:'Complete depth-17 shape plus distinct bounded indices proves actual normalized-route separation and full returned-pair write framing. Real allocation and table/extras loops supply shape; prefix-plus-index bounds, fill-clear certificates and final geometry remain separate.'};
 const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
