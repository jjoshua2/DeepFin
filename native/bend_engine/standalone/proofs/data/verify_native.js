// Reuse the unchanged independent full-buffer verifier; add a targeted late corruption.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN,verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify_native.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler),root=path.resolve(import.meta.dirname,'../..');
const engine=path.dirname(root),temp=fs.mkdtempSync(path.join(os.tmpdir(),'deepfin-data-native-'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const tracked=['standalone/Tables.bend','standalone/Subsets.bend','standalone/Text.bend',
 'standalone/proofs/headers/probe.bend','standalone/proofs/headers/verify_native.js','standalone/proofs/data/verify_native.js'];
const hashes=()=>Object.fromEntries(tracked.map(f=>[f,sha(fs.readFileSync(path.join(engine,f)))]));
const before=hashes();
function invoke(file,argv=[]){const r=spawnSync(process.execPath,[file,compiler,...argv],{encoding:'utf8',timeout:600000,maxBuffer:64<<20,
 env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);return r;}
try{
 const result=invoke(path.join(root,'proofs/headers/verify_native.js'),['--report',path.join(temp,'native.json')]);
 assert.equal(result.status,0,(result.stderr+result.stdout).slice(-4000));
 const n=JSON.parse(fs.readFileSync(path.join(temp,'native.json'),'utf8'));
 assert.equal(n.native_gate,'PASS');assert.equal(n.modes.length,4);
 const mutant=path.join(temp,'mutant');fs.cpSync(engine,mutant,{recursive:true});
 const table=path.join(mutant,'standalone/Tables.bend'),old=fs.readFileSync(table,'utf8');
 const needle='      tables(p, U32.inc(key), U32.add(at, size),';
 assert.equal(old.split(needle).length,2,'unique mutation site');
 fs.writeFileSync(table,old.replace(needle,'      a = Array.set(U64,a,512,U64.zero())\n'+needle));
 const bad=invoke(path.join(mutant,'standalone/proofs/headers/verify_native.js'));
 assert.equal(bad.status,1,'corrupted later metadata must fail');
 const text=(bad.stdout+bad.stderr).trim();
 assert.match(text,/cell 512|cell512|cell=512/,'must fail as an earlier data-cell value mismatch');
 assert.doesNotMatch(text,/error:|Segmentation fault|Maximum call stack|no such file/,'compile or execution failure is not value rejection');
 assert.deepEqual(hashes(),before);assert.deepEqual(verifyCompiler(compiler),identity);
 const report={native_gate:'PASS',compiler_revision:PIN.revision,...identity,
   verifier:'Unchanged headers/verify_native.js newly executed; no duplicate oracle or proof representation in candidate',
   baseline:n,late_metadata_corruption:{rejected:true,exit_code:bad.status,diagnostic_sha256:sha(text),
     diagnostic_bytes:Buffer.byteLength(text),excerpt:text.slice(-2400),
     scope:'generic-mode wrong-value rejection after compilation; driver stops before further corrupted modes'},
   source_sha256s:before,scope:'Complete native buffers and late-write regression, not a new independent source ray theorem or native ownership/lifetime proof'};
 const encoded=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],encoded);console.log(encoded.trimEnd());
}finally{fs.rmSync(temp,{recursive:true,force:true});}
