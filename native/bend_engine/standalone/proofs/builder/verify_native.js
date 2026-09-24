// Reuse real build/lookup qualification; deliberately shorten the public builder only.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
import {readRecipe, checkRecipe} from './recipe.js';
const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'));
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
const suite = import.meta.dirname, engine = path.resolve(suite, '../../..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-builder-native-'));
const sha = b => createHash('sha256').update(b).digest('hex');
const tracked = ['standalone/Tables.bend', 'standalone/proofs/lookup/verify_native.js',
  'standalone/proofs/lookup/probe.bend', 'standalone/proofs/builder/verify_native.js',
  'standalone/proofs/builder/recipe.js', 'standalone/proofs/builder/Spec.bend'];
const hashes = () => Object.fromEntries(tracked.map(f => [f, sha(fs.readFileSync(path.join(engine, f)))]));
const original = hashes();
function run(file, rest = []) {
  const p = spawnSync(process.execPath, [file, compiler, ...rest], {
    encoding: 'utf8', timeout: 600000, maxBuffer: 32 << 20, env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(p.error, undefined, String(p.error)); assert.equal(p.signal, null); return p;
}
try {
  const recipe = readRecipe(path.join(engine, 'standalone/Tables.bend'), path.join(suite, 'Spec.bend'));
  const good = run(path.join(suite, '../lookup/verify_native.js'), ['--report', path.join(temporary, 'native.json')]);
  assert.equal(good.status, 0, (good.stderr + good.stdout).slice(-3000));
  const native = JSON.parse(fs.readFileSync(path.join(temporary, 'native.json'), 'utf8'));
  assert.equal(native.native_gate, 'PASS'); assert.equal(native.modes.length, 4);
  const mutation = path.join(temporary, 'short-builder'); fs.cpSync(engine, mutation, {recursive: true});
  const table = path.join(mutation, 'standalone/Tables.bend'), source = fs.readFileSync(table, 'utf8');
  const needle = 'tables(128n, 0, 512, Array.new'; assert.equal(source.split(needle).length, 2);
  const changed = source.replace(needle, 'tables(127n, 0, 512, Array.new'); fs.writeFileSync(table, changed);
  assert.throws(() => checkRecipe(changed, fs.readFileSync(path.join(suite, 'Spec.bend'), 'utf8')), /public build recipe/);
  const bad = run(path.join(mutation, 'standalone/proofs/lookup/verify_native.js'));
  const diagnostic = (bad.stdout + bad.stderr).trim();
  assert.equal(bad.status, 1, 'short builder must be rejected');
  assert.match(diagnostic, /lookup row 1016/, 'must fail on the first final-key lookup, not compilation');
  assert.doesNotMatch(diagnostic, /error:|no such file|Segmentation fault|Maximum call stack/);
  assert.deepEqual(hashes(), original); assert.deepEqual(verifyCompiler(compiler), identity);
  const report = {native_gate: 'PASS', compiler_revision: PIN.revision, ...identity, public_recipe: recipe,
    baseline: native, public_builder_mutation: {rejected: true, source_guard_rejected: true, status: bad.status,
      kind: 'generic wrong-value rejection after compiling actual 127-block builder', row: 1016,
      diagnostic_sha256: sha(diagnostic), diagnostic_bytes: Buffer.byteLength(diagnostic), excerpt: diagnostic.slice(-2000)},
    source_sha256s: original,
    scope: 'Newly executed unchanged four-mode builder/lookup verifier and one public-wrapper mutation. No full-buffer or lifetime claim.'};
  const out = JSON.stringify(report, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], out); console.log(out.trimEnd());
} finally { fs.rmSync(temporary, {recursive: true, force: true}); }
