// Opt-in full aggregate; retained parent receipts are separately labeled modular.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler);
function run(file,timeout){const r=spawnSync(process.execPath,[path.resolve(import.meta.dirname,file),compiler],
 {encoding:'utf8',timeout,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-4000));return JSON.parse(r.stdout);}
const parent=run('../decoder/verify.js',10800000);
assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,127);
assert.equal(parent.inherited_negative_controls,332);assert.equal(parent.negative_controls.length,18);
assert.ok(parent.negative_controls.every(x=>x.rejected));
const focused=run('./focused.js',1800000);
assert.equal(focused.focused_gate,'PASS');assert.equal(focused.new_law_count,5);
assert.equal(focused.negative_controls.length,17);assert.ok(focused.negative_controls.every(x=>x.rejected));
assert.deepEqual(verifyCompiler(compiler),identity);
const result={proof_gate:'PASS',aggregate_law_count:132,new_law_count:5,inherited_law_count:127,
 inherited_negative_controls:350,negative_controls:focused.negative_controls,
 source_sha256s:focused.source_sha256s,inherited_source_sha256s:parent.source_sha256s,
 compiler_revision:focused.compiler_revision,...identity};
const out=JSON.stringify(result,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],out);console.log(out.trimEnd());
