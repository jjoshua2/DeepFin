// Opt-in complete aggregate. Modular qualification may retain exact parent evidence.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler);
function check(file,timeout){const r=spawnSync(process.execPath,[path.resolve(import.meta.dirname,file),compiler],
 {encoding:'utf8',timeout,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
 assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-5000));return JSON.parse(r.stdout);}
const parent=check('../extras/verify.js',5400000);
assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,100);
assert.equal(parent.inherited_negative_controls,212);assert.equal(parent.negative_controls.length,14);assert.ok(parent.negative_controls.every(x=>x.rejected));
console.error('PASS: unchanged 100-law parent and all 226 inherited rejection controls');
const p=check('./focused.js',4800000);assert.equal(p.focused_gate,'PASS');assert.equal(p.controls_gate,'PASS');assert.equal(p.consumer,'PASS');
assert.equal(p.new_law_count,3);assert.equal(p.negative_controls.length,17);assert.ok(p.negative_controls.every(x=>x.rejected));
assert.deepEqual(verifyCompiler(compiler),identity);
const report={proof_gate:'PASS',compiler_revision:p.compiler_revision,...identity,aggregate_law_count:103,new_law_count:3,inherited_law_count:100,
 inherited_negative_controls:226,new_laws:p.new_laws,promoted_prior_supplementary_laws:p.promoted_prior_supplementary_laws,negative_controls:p.negative_controls,
 consumer:p.consumer,source_sha256s:p.source_sha256s,inherited_source_sha256s:parent.source_sha256s,scope:p.scope};
const text=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
