// Full opt-in chain; modular publication receipts must not claim this command ran.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {verifyCompiler} from '../../verify_compiler.js';
const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'));
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
function check(file, timeout) {
  const r = spawnSync(process.execPath, [path.resolve(import.meta.dirname, file), compiler], {
    encoding: 'utf8', timeout, maxBuffer: 64 << 20, env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  assert.equal(r.status, 0, (r.stderr + r.stdout).slice(-4000)); return JSON.parse(r.stdout);
}
const parent = check('../builder/verify.js', 25200000);
assert.equal(parent.proof_gate, 'PASS'); assert.equal(parent.aggregate_law_count, 117);
assert.equal(parent.inherited_negative_controls, 295); assert.equal(parent.negative_controls.length, 20);
assert.ok(parent.negative_controls.every(x => x.rejected));
const result = check('./focused.js', 1800000);
assert.equal(result.focused_gate, 'PASS'); assert.equal(result.controls_gate, 'PASS');
assert.equal(result.consumer, 'PASS'); assert.equal(result.new_law_count, 6);
assert.equal(result.negative_controls.length, 17); assert.ok(result.negative_controls.every(x => x.rejected));
assert.deepEqual(verifyCompiler(compiler), identity);
const report = {proof_gate: 'PASS', compiler_revision: result.compiler_revision, ...identity,
  aggregate_law_count: 123, new_law_count: 6, inherited_law_count: 117, inherited_negative_controls: 315,
  new_laws: result.new_laws, negative_controls: result.negative_controls, source_sha256s: result.source_sha256s,
  inherited_source_sha256s: parent.source_sha256s, consumer: result.consumer,
  scope: result.scope};
const out = JSON.stringify(report, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], out); console.log(out.trimEnd());
