import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'),'usage: verify.js COMPILER [--report FILE]');
const compiler=path.resolve(args[0]),identity=verifyCompiler(compiler);
function check(file,timeout){const r=spawnSync(process.execPath,[path.join(import.meta.dirname,file),compiler],
  {encoding:'utf8',timeout,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});
  assert.equal(r.error,undefined,String(r.error));assert.equal(r.signal,null);assert.equal(r.status,0,(r.stderr+r.stdout).slice(-4000));return JSON.parse(r.stdout);}
const parent=check('../contents/verify.js',4200000);
assert.equal(parent.proof_gate,'PASS');assert.equal(parent.aggregate_law_count,98);assert.equal(parent.inherited_negative_controls,195);
assert.equal(parent.negative_controls.length,17);assert.ok(parent.negative_controls.every(c=>c.rejected));
console.error('PASS unchanged 98-law parent and 212 inherited controls');
const child=check('focused.js',1200000);
assert.equal(child.focused_gate,'PASS');assert.equal(child.new_law_count,2);assert.equal(child.negative_controls.length,14);
assert.ok(child.negative_controls.every(c=>c.rejected));assert.deepEqual(verifyCompiler(compiler),identity);
const r={proof_gate:'PASS',compiler_revision:child.compiler_revision,...identity,aggregate_law_count:100,new_law_count:2,
  inherited_law_count:98,inherited_negative_controls:212,new_laws:child.new_laws,negative_controls:child.negative_controls,
  source_sha256s:child.source_sha256s,inherited_source_sha256s:parent.source_sha256s,consumer:child.consumer,scope:child.scope};
const text=JSON.stringify(r,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
