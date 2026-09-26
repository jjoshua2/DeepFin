// Opt-in successor/ordinal gate. Parent laws, gates and compiler remain unchanged.
// Usage: bun native/bend_engine/standalone/proofs/successor/verify.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';

const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]);
const identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '../..');
const proofRoot = path.join(root, 'proofs');
const suite = path.join(proofRoot, 'successor');
const cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-successor-proofs-'));
const required = ['subtraction_refinement', 'step_successor', 'sequence_ordinal',
  'sequence_nonduplicating', 'sequence_coverage', 'cycle_endpoint', 'sequence_periodic'];
const negatives = [];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const code = f => fs.readFileSync(f, 'utf8').split('\n').map(s => s.split('#')[0]).join('\n');
const names = f => [...code(f).matchAll(/^law ([A-Za-z0-9_]+):/gm)].map(m => m[1]);
function proofFiles(dir, prefix = 'proofs') {
  return fs.readdirSync(dir).sort().flatMap(name => {
    const file = path.join(dir, name), rel = prefix + '/' + name;
    const stat = fs.lstatSync(file);
    assert.ok(!stat.isSymbolicLink(), 'symlinked proof input');
    return stat.isDirectory() ? proofFiles(file, rel) : /\.(bend|js|json)$/.test(name) ? [rel] : [];
  });
}
const tracked = ['Subsets.bend', 'Tables.bend', ...proofFiles(proofRoot)];
const sourceHashes = () => Object.fromEntries(tracked.map(f => [f, sha(fs.readFileSync(path.join(root, f)))]));
const sources = sourceHashes();

function invoke(file, checker = cli) {
  const r = spawnSync(process.execPath, [checker, file], {encoding: 'utf8',
    timeout: 30000, maxBuffer: 8 << 20, env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, `checker execution error: ${r.error}`);
  assert.equal(r.signal, null, `checker signal: ${r.signal}`);
  return {status: r.status, text: (r.stdout + r.stderr).trim()};
}
function clean(r) {
  assert.equal(r.status, 0, r.text);
  assert.equal(r.text, 'All terms check.', r.text);
}
function policy(file, boundary, seen = new Set()) {
  file = path.resolve(file);
  assert.ok(file.startsWith(boundary + path.sep), 'proof import escaped standalone source');
  assert.ok(fs.lstatSync(file).isFile(), 'proof input must be a regular non-symlink file');
  if (seen.has(file)) return;
  seen.add(file);
  const s = code(file);
  assert.doesNotMatch(s, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const m of s.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (m[1] === 'Base') continue;
    assert.match(m[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign or unexpected import');
    policy(path.resolve(path.dirname(file), m[1]), boundary, seen);
  }
}
function replace(file, before, after) {
  const s = fs.readFileSync(file, 'utf8');
  assert.equal(s.split(before).length, 2, `mutation anchor must be unique: ${before}`);
  fs.writeFileSync(file, s.replace(before, after));
}
function copy(name) {
  const d = path.join(temp, name);
  fs.mkdirSync(d);
  fs.copyFileSync(path.join(root, 'Subsets.bend'), path.join(d, 'Subsets.bend'));
  fs.cpSync(proofRoot, path.join(d, 'proofs'), {recursive: true});
  return d;
}
function variant(name, edit, reason = /expected[\s\S]*observed|TODO found/) {
  const d = copy(name);
  edit(d);
  const r = invoke(path.join(d, 'proofs/successor/PROOF.bend'));
  assert.equal(r.status, 1, `${name}: must reject\n${r.text}`);
  assert.match(r.text, reason, `${name}: wrong rejection\n${r.text}`);
  negatives.push({name, rejected: true});
}
function mutatedBase(name, before, after) {
  // Only a disposable copy of Base is altered. The protected checker and real
  // compiler checkout are never edited; the normal gate still verifies the pin.
  const d = path.join(temp, name);
  fs.mkdirSync(d);
  for (const f of ['bend.ts', 'comp.ts', 'base.bend', 'main.ts']) {
    fs.copyFileSync(path.join(compiler, 'bend2', f), path.join(d, f));
  }
  replace(path.join(d, 'base.bend'), before, after);
  const r = invoke(path.join(suite, 'Borrow.bend'), path.join(d, 'main.ts'));
  assert.equal(r.status, 1, `${name}: corrupt Base must fail\n${r.text}`);
  assert.match(r.text, /expected[\s\S]*observed/, r.text);
  negatives.push({name, rejected: true, disposable_base_mutation: true});
}

try {
  const inherited = spawnSync(process.execPath, [path.join(proofRoot, 'index/verify.js'), compiler],
    {encoding: 'utf8', timeout: 120000, maxBuffer: 8 << 20,
      env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(inherited.error, undefined, `${inherited.error}`);
  assert.equal(inherited.signal, null);
  assert.equal(inherited.status, 0, inherited.stderr + inherited.stdout);
  const parent = JSON.parse(inherited.stdout);
  assert.equal(parent.proof_gate, 'PASS');
  assert.equal(parent.new_law_count, 9);
  assert.equal(parent.parent_law_count, 8);
  assert.equal(parent.inherited_u64_law_count, 16);
  assert.equal(parent.negative_controls.length, 15);
  assert.equal(parent.parent_negative_controls, 11);
  assert.ok(parent.negative_controls.every(c => c.rejected));
  assert.deepEqual(names(path.join(suite, 'LAWS.bend')), required);
  policy(path.join(suite, 'PROOF.bend'), root);
  policy(path.join(suite, 'consumer.bend'), root);
  clean(invoke(path.join(suite, 'PROOF.bend')));
  clean(invoke(path.join(suite, 'consumer.bend')));

  const rel = 'proofs/successor/';
  variant('missing-new-proof', d => replace(path.join(d, rel+'PROOF.bend'),
    'def Laws.sequence_periodic(i,mask):\n  Ordinal.periodic(i,mask)\n', ''), /TODO found/);
  variant('missing-laws-import', d => fs.writeFileSync(path.join(d, rel+'PROOF.bend'),
    'import Base\n'), /PROOF\.bend must import/);
  variant('proof-hole', d => replace(path.join(d, rel+'PROOF.bend'),
    '  Ordinal.periodic(i,mask)', '  ?TODO'), /TODO found/);
  variant('inclusive-ordinal-bound', d => replace(path.join(d, rel+'LAWS.bend'),
    'for +bounded: {Nat.is_lt(i,Bits.capacity(mask))', 'for +bounded: {Nat.is_le(i,Bits.capacity(mask))'));
  variant('removed-step-membership', d => replace(path.join(d, rel+'LAWS.bend'),
    'law step_successor:\n  for +s: U64\n  for +mask: U64\n  for member: {U64.and(s,mask) == s : U64}',
    'law step_successor:\n  for +s: U64\n  for +mask: U64\n  for member: {True{} == True{} : Bool}'));
  variant('wrong-wrap-value', d => replace(path.join(d, rel+'Spec.bend'),
    'Bool.pick(Nat,Nat.is_lt(1n+v,cap),1n+v,0n)', 'Bool.pick(Nat,Nat.is_lt(1n+v,cap),1n+v,1n)'));
  variant('reversed-actual-subtraction', d => replace(path.join(d, 'Subsets.bend'),
    'U64.sub(subset, mask)', 'U64.sub(mask, subset)'));
  variant('unmasked-actual-step', d => replace(path.join(d, 'Subsets.bend'),
    'U64.and(difference(subset, mask), mask)', 'difference(subset, mask)'));
  variant('stuck-actual-recurrence', d => replace(path.join(d, 'proofs/LAWS.bend'),
    'Subsets.next(at(p, mask), mask)', 'at(p, mask)'));
  variant('missing-inherited-index-proof', d => replace(path.join(d, 'proofs/index/PROOF.bend'),
    'def Laws.sequence_index_bound(n, mask):\n  Laws.extraction_bound(SubsetLaws.at(n, mask), mask)\n', ''), /TODO found/);

  const unsafe = copy('unsafe-subtraction-bridge');
  replace(path.join(unsafe, rel+'Borrow.bend'), 'def u64_sub(', '@unsafe\ndef u64_sub(');
  assert.throws(() => policy(path.join(unsafe, rel+'PROOF.bend'), unsafe), /unsafe dependency/);
  const warning = invoke(path.join(unsafe, rel+'PROOF.bend'));
  assert.match(warning.text, /unsafe or foreign/);
  assert.throws(() => clean(warning));
  negatives.push({name: 'unsafe-subtraction-bridge', rejected: true, raw_cli_status: warning.status});
  const foreign = copy('foreign-proof');
  fs.appendFileSync(path.join(foreign, rel+'Borrow.bend'), '\nimport "./oracle.c"\n');
  assert.throws(() => policy(path.join(foreign, rel+'PROOF.bend'), foreign), /foreign or unexpected import/);
  negatives.push({name: 'foreign-proof', rejected: true});
  const omitted = copy('omitted-ordinal-law');
  replace(path.join(omitted, rel+'LAWS.bend'), 'law sequence_ordinal:', 'def not_a_law:');
  assert.throws(() => assert.deepEqual(names(path.join(omitted, rel+'LAWS.bend')), required));
  negatives.push({name: 'omitted-ordinal-law', rejected: true});

  // These specifically check the new subtraction bridge, not an old boundary law.
  mutatedBase('dropped-low-half-borrow', 'borrow = Bool.to_u32(U32.is_lt(alo, blo))', 'borrow = 0');
  mutatedBase('borrow-on-equal-halves', 'borrow = Bool.to_u32(U32.is_lt(alo, blo))',
    'borrow = Bool.to_u32(U32.is_le(alo, blo))');
  mutatedBase('truncated-high-half-result',
    'U64{U32.sub(alo, blo), U32.sub(U32.sub(ahi, bhi), borrow)}', 'U64{U32.sub(alo, blo), 0}');

  assert.deepEqual(verifyCompiler(compiler), identity, 'compiler changed during tests');
  assert.deepEqual(sourceHashes(), sources, 'sources changed during tests');
  const report = {proof_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_universal_laws: required, new_law_count: required.length,
    prior_engine_law_count: 17, inherited_u64_law_count: 16, aggregate_law_count: 40,
    prior_negative_controls: 26,
    negative_controls: negatives, consumer: 'PASS', source_sha256s: sources,
    scope: 'Full-width subtraction, actual carry-rippler compact successor, Nat ordinal, coverage, nonduplication and period; NOT affine table/lookup refinement or native correctness'};
  const json = JSON.stringify(report, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
