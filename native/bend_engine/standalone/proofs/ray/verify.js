// Explicitly opt-in full inherited chain; modular receipts must say it was not run.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);assert.ok(args.length===1||(args.length===3&&args[1]==='--report'));
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler);
function check(file,timeout){const p=spawnSync(process.execPath,[path.resolve(import.meta.dirname,file),compiler],{encoding:'utf8',timeout,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(p.error,undefined,String(p.error));assert.equal(p.signal,null);assert.equal(p.status,0,(p.stderr+p.stdout).slice(-4000));return JSON.parse(p.stdout);}
const parent=check('../lookup/verify.js',18000000);assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,110);
assert.equal(parent.inherited_negative_controls,260);assert.equal(parent.negative_controls.length,16);assert.ok(parent.negative_controls.every(x=>x.rejected));
const p=check('./focused.js',7200000);assert.equal(p.focused_gate,'PASS');assert.equal(p.controls_gate,'PASS');assert.equal(p.consumer,'PASS');assert.equal(p.new_law_count,5);assert.equal(p.negative_controls.length,19);
assert.ok(p.negative_controls.every(x=>x.rejected));assert.deepEqual(verifyCompiler(compiler),identity);
const report={proof_gate:'PASS',compiler_revision:p.compiler_revision,...identity,aggregate_law_count:115,new_law_count:5,inherited_law_count:110,inherited_negative_controls:276,new_laws:p.new_laws,negative_controls:p.negative_controls,source_sha256s:p.source_sha256s,inherited_source_sha256s:parent.source_sha256s,consumer:p.consumer,scope:p.scope};
const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
