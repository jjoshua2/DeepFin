// Uniform full-table laws, small semantic controls and explicit public-recipe linkage.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
import {checkRecipe, readRecipe} from './recipe.js';
const args = process.argv.slice(2), only = args.includes('--controls-only');
const a = args.filter(x => x !== '--controls-only');
assert.ok(a.length === 1 || (a.length === 3 && a[1] === '--report'),
  'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler = path.resolve(a[0]), identity = verifyCompiler(compiler);
const suite = import.meta.dirname, engine = path.resolve(suite, '../../..');
const table = path.join(engine, 'standalone/Tables.bend');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-builder-'));
const names = ['full_table_lookup', 'u32_full_table_lookup'], controls = [];
const sha = b => createHash('sha256').update(b).digest('hex');
const text = f => fs.readFileSync(f, 'utf8');
const code = f => text(f).split('\n').map(l => l.split('#')[0]).join('\n');
function manifest(s) {
  assert.deepEqual([...code(path.join(s, 'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m => m[1]), names);
  const p = code(path.join(s, 'PROOF.bend'));
  assert.deepEqual([...p.matchAll(/^def Laws\.(\w+)\(/gm)].map(m => m[1]), names);
  assert.match(p, /import \.\/LAWS\.bend as Laws/);
  assert.match(code(path.join(s, 'consumer.bend')), /import \.\/PROOF\.bend as Proof/);
}
function graph(f, boundary, seen = new Set()) {
  f = path.resolve(f);
  const rel = path.relative(boundary, f);
  assert.ok(!rel.split(path.sep).includes('..') &&
    (rel.startsWith('standalone/') || rel === 'legal_probe/Chess.bend' || rel === 'bitboard_probe/Sliders.bend'), 'escaped scope');
  assert.ok(fs.lstatSync(f).isFile(), 'nonregular proof input');
  assert.equal(fs.realpathSync(f), f, 'symlinked proof input');
  if (seen.has(f)) return seen;
  seen.add(f);
  const s = code(f);
  assert.doesNotMatch(s, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const m of s.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (m[1] === 'Base') continue;
    assert.match(m[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign import');
    graph(path.resolve(path.dirname(f), m[1]), boundary, seen);
  }
  return seen;
}
function invoke(file, timeout = 1500000) {
  const r = spawnSync(process.execPath, [path.join(compiler, 'bend2/main.ts'), file], {
    encoding: 'utf8', timeout, maxBuffer: 32 << 20, env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error));
  assert.equal(r.signal, null, 'crash/timeout is not a semantic rejection');
  return {status: r.status, output: (r.stdout + r.stderr).trim()};
}
function clean(r) { assert.equal(r.status, 0, r.output.slice(-2000)); assert.equal(r.output, 'All terms check.'); }
function copy(name) {
  const e = path.join(temporary, name); fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/builder')};
}
function replace(f, from, to) {
  const s = text(f); assert.equal(s.split(from).length, 2, 'nonunique mutation site');
  fs.writeFileSync(f, s.replace(from, to));
}
function semantic(name, file, edit, location) {
  const d = copy(name); edit(d); const r = invoke(path.join(d.s, file), 60000);
  assert.equal(r.status, 1, name);
  assert.match(r.output, /expected[\s\S]*observed/, 'not an ordinary refinement failure');
  assert.doesNotMatch(r.output, /no such file|a defined name|more than once|RangeError|Maximum call stack|Segmentation fault/);
  assert.match(r.output, location);
  controls.push({name, kind: 'source semantic/refinement', rejected: true, entry: file,
    diagnostic_sha256: sha(r.output), diagnostic_bytes: Buffer.byteLength(r.output), excerpt: r.output.slice(-700)});
  fs.rmSync(d.e, {recursive: true, force: true});
}
function policyControl(name, edit, check, pattern) {
  const d = copy(name); edit(d); assert.throws(() => check(d), pattern);
  controls.push({name, kind: 'manifest/import policy', rejected: true});
  fs.rmSync(d.e, {recursive: true, force: true});
}
try {
  manifest(suite);
  const files = graph(path.join(suite, 'consumer.bend'), engine);
  for (const name of ['focused.js', 'recipe.js', 'verify.js', 'verify_native.js']) files.add(path.join(suite, name));
  const hashes = () => Object.fromEntries([...files].sort().map(f => [path.relative(engine, f), sha(fs.readFileSync(f))]));
  const original = hashes(), link = readRecipe(table, path.join(suite, 'Spec.bend'));
  if (!only) clean(invoke(path.join(suite, 'consumer.bend')));
  semantic('wrong-full-count', 'Domain.bend', d => replace(path.join(d.s, 'Domain.bend'),
    'Nat.sub(127n,k)', 'Nat.sub(126n,k)'), /Location: full_configuration\b/);
  semantic('wrong-selected-key', 'Domain.bend', d => replace(path.join(d.s, 'Domain.bend'),
    '{Nat.add(k,0n) == k : Nat}', '{Nat.add(k,0n) == 1n+k : Nat}'), /Location: full_configuration\b/);
  semantic('shifted-symbolic-prefix', 'Binding.bend', d => replace(path.join(d.s, 'Spec.bend'),
    'Tables.tables(n,0,512,', 'Tables.tables(n,0,513,'), /Location: image\b/);
  semantic('accept-127-blocks', 'Binding.bend', d => replace(path.join(d.s, 'Binding.bend'),
    'full: {n == 128n : Nat}', 'full: {n == 127n : Nat}'), /Location: image\b/);
  const actual = text(table), spec = text(path.join(suite, 'Spec.bend'));
  const recipeMutations = [
    ['public-short-table', 'tables(128n, 0, 512,', 'tables(127n, 0, 512,'],
    ['public-start-key', 'tables(128n, 0, 512,', 'tables(128n, 1, 512,'],
    ['public-shifted-prefix', 'tables(128n, 0, 512,', 'tables(128n, 0, 513,'],
    ['public-wrong-depth', 'Array.new(U64, 17n, U64.zero())', 'Array.new(U64, 16n, U64.zero())'],
    ['public-short-extras', 'extras(64n, 0, tables(', 'extras(63n, 0, tables('],
    ['public-start-square', 'extras(64n, 0, tables(', 'extras(64n, 1, tables('],
    ['public-changed-seed', 'Array.new(U64, 17n, U64.zero())', 'Array.new(U64, 17n, U64.from_u32(1))']
  ];
  for (const [name, from, to] of recipeMutations) {
    assert.equal(actual.split(from).length, 2);
    assert.throws(() => checkRecipe(actual.replace(from, to), spec), /public build recipe/);
    controls.push({name, kind: 'public source-recipe guard, not source theorem rejection', rejected: true});
  }
  assert.deepEqual(checkRecipe(actual.replace('def build()', '# harmless comment\ndef build()'), spec), link);
  policyControl('omitted-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law full_table_lookup:', 'def omitted:'), d => manifest(d.s), /./);
  policyControl('omitted-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.full_table_lookup(', 'def omitted('), d => manifest(d.s), /./);
  policyControl('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  policyControl('missing-consumer-proof', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  policyControl('proof-hole', d => fs.appendFileSync(path.join(d.s, 'Binding.bend'), '\n?hole\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /proof hole/);
  policyControl('unsafe-proof', d => replace(path.join(d.s, 'Binding.bend'), 'def image(', '@unsafe\ndef image('), d => graph(path.join(d.s, 'consumer.bend'), d.e), /unsafe dependency/);
  policyControl('foreign-proof', d => fs.appendFileSync(path.join(d.s, 'Binding.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /foreign import/);
  policyControl('symlinked-proof', d => {const f = path.join(d.s, 'Binding.bend'); fs.renameSync(f, f + '.old'); fs.symlinkSync('Binding.bend.old', f);}, d => graph(path.join(d.s, 'consumer.bend'), d.e), /nonregular/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name: 'warning-with-zero', kind: 'synthetic exact-output wrapper unit; not compiler execution', rejected: true});
  assert.equal(controls.length, 20);
  assert.deepEqual(hashes(), original); assert.deepEqual(verifyCompiler(compiler), identity);
  const report = {focused_gate: only ? 'NOT_RUN' : 'PASS', consumer: only ? 'NOT_RUN' : 'PASS', controls_gate: 'PASS',
    compiler_revision: PIN.revision, ...identity, new_law_count: 2, new_laws: names, inherited_gate_run: false,
    public_recipe: link, negative_controls: controls, source_sha256s: original,
    scope: 'Uniform full-table Nat/U32 lookup contracts plus explicit token linkage to public defaults. Not a normalized closed Tables.build equality.'};
  const out = JSON.stringify(report, null, 2) + '\n'; if (a[2]) fs.writeFileSync(a[2], out); console.log(out.trimEnd());
} finally { fs.rmSync(temporary, {recursive: true, force: true}); }
