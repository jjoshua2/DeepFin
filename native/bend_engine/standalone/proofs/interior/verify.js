// Full opt-in aggregate: check the new laws first, then retain the unchanged 74-law prefix gate.
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler);
function check(file,timeout){
 const r=spawnSync(process.execPath,[path.resolve(import.meta.dirname,file),compiler],{encoding:'utf8',timeout,maxBuffer:32<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,`${r.error}`);assert.equal(r.signal,null);assert.equal(r.status,0,r.stdout+r.stderr);return JSON.parse(r.stdout);
}
const focused=check('./focused.js',1500000);
assert.equal(focused.focused_gate,'PASS');assert.equal(focused.consumer,'PASS');assert.equal(focused.new_law_count,5);assert.equal(focused.negative_controls.length,21);assert.ok(focused.negative_controls.every(c=>c.rejected));
const parent=check('../prefix/verify.js',3300000);
assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,74);
assert.equal(parent.inherited_negative_controls,107);assert.equal(parent.negative_controls.length,19);assert.ok(parent.negative_controls.every(c=>c.rejected));
console.error('PASS: unchanged 74-law parent and all 126 inherited controls');
assert.deepEqual(verifyCompiler(compiler),identity);
const report={proof_gate:'PASS',compiler_revision:focused.compiler_revision,...identity,aggregate_law_count:79,new_law_count:5,
 new_laws:focused.new_laws,inherited_law_count:74,inherited_negative_controls:126,negative_controls:focused.negative_controls,
 source_sha256s:focused.source_sha256s,inherited_source_sha256s:parent.source_sha256s,consumer:focused.consumer,scope:focused.scope};
const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
