// Opt-in chain extension; no old gate is removed or changed.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const args=process.argv.slice(2);
assert.ok(args.length===1||(args.length===3&&args[1]==='--report'));
function run(name){const r=spawnSync(process.execPath,[path.resolve(import.meta.dirname,name),path.resolve(args[0])],{encoding:'utf8',timeout:12000000,maxBuffer:64<<20,env:{...process.env,BEND_NO_TELEMETRY:'1'}});assert.equal(r.error,undefined);assert.equal(r.signal,null);assert.equal(r.status,0,r.stderr.slice(-3000));return JSON.parse(r.stdout);}
const p=run('../move_update/verify.js');assert.equal(p.proof_gate,'PASS');assert.equal(p.aggregate_law_count,146);
const n=run('./focused.js');assert.equal(n.focused_gate,'PASS');assert.equal(n.new_law_count,4);assert.equal(n.negative_controls.length,17);
const out={proof_gate:'PASS',aggregate_law_count:150,new_law_count:4,inherited_law_count:146,inherited_negative_controls:419,new_laws:n.new_laws,negative_controls:n.negative_controls,source_sha256s:n.source_sha256s,inherited_source_sha256s:p.source_sha256s,consumer:n.consumer,compiler_revision:n.compiler_revision,scope:n.scope};
const text=JSON.stringify(out,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],text);console.log(text.trimEnd());
