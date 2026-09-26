// Additive, opt-in aggregate gate. Original proofs/gate and compiler stay unchanged.
// Usage: bun native/bend_engine/standalone/proofs/index/verify.js COMPILER [--report FILE]
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
const suite = path.join(proofRoot, 'index');
const cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-compact-proofs-'));
const required = ['capacity_matches_popcount', 'extraction_bound', 'compact_projection',
  'compact_roundtrip', 'deposit_in_mask', 'deposit_injective', 'masked_extract_injective',
  'compact_coverage', 'sequence_index_bound'];
const negatives = [];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const code = f => fs.readFileSync(f, 'utf8').split('\n').map(s => s.split('#')[0]).join('\n');
const names = f => [...code(f).matchAll(/^law ([A-Za-z0-9_]+):/gm)].map(m => m[1]);
const tracked = ['Subsets.bend', 'Tables.bend', 'proofs/LAWS.bend', 'proofs/PROOF.bend',
  'proofs/Mask.bend', 'proofs/verify.js', 'proofs/u64/LAWS.bend', 'proofs/u64/PROOF.bend',
  'proofs/u64/Words.bend', 'proofs/u64/provenance.json',
  ...['LAWS.bend', 'PROOF.bend', 'Bits.bend', 'Nats.bend', 'Words.bend', 'Count.bend',
    'consumer.bend', 'verify.js'].map(f => 'proofs/index/' + f)];
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
  const r = invoke(path.join(d, 'proofs/index/PROOF.bend'));
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
  const r = invoke(path.join(suite, 'PROOF.bend'), path.join(d, 'main.ts'));
  assert.equal(r.status, 1, `${name}: corrupt Base must fail\n${r.text}`);
  assert.match(r.text, /expected[\s\S]*observed/, r.text);
  negatives.push({name, rejected: true, disposable_base_mutation: true});
}

try {
  const inherited = spawnSync(process.execPath, [path.join(proofRoot, 'verify.js'), compiler],
    {encoding: 'utf8', timeout: 120000, maxBuffer: 8 << 20,
      env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(inherited.error, undefined, `${inherited.error}`);
  assert.equal(inherited.signal, null);
  assert.equal(inherited.status, 0, inherited.stderr + inherited.stdout);
  const parent = JSON.parse(inherited.stdout);
  assert.equal(parent.proof_gate, 'PASS');
  assert.equal(parent.new_source_laws.length, 8);
  assert.equal(parent.inherited_source_laws.length, 16);
  assert.equal(parent.negative_controls.length, 11);
  assert.ok(parent.negative_controls.every(c => c.rejected));
  assert.deepEqual(names(path.join(suite, 'LAWS.bend')), required);
  policy(path.join(suite, 'PROOF.bend'), root);
  policy(path.join(suite, 'consumer.bend'), root);
  clean(invoke(path.join(suite, 'PROOF.bend')));
  clean(invoke(path.join(suite, 'consumer.bend')));

  variant('missing-new-proof', d => replace(path.join(d, 'proofs/index/PROOF.bend'),
    'def Laws.sequence_index_bound(n, mask):\n  Laws.extraction_bound(SubsetLaws.at(n, mask), mask)\n', ''), /TODO found/);
  variant('missing-new-laws-import', d => fs.writeFileSync(path.join(d, 'proofs/index/PROOF.bend'),
    'import Base\n'), /PROOF\.bend must import/);
  variant('missing-parent-proof', d => replace(path.join(d, 'proofs/PROOF.bend'),
    'def Laws.full_word_wrap():\n  {==}\n', ''), /TODO found/);
  variant('new-proof-hole', d => replace(path.join(d, 'proofs/index/PROOF.bend'),
    '  Laws.extraction_bound(SubsetLaws.at(n, mask), mask)', '  ?TODO'), /TODO found/);
  variant('zero-capacity', d => replace(path.join(d, 'proofs/index/Bits.bend'),
    'case 0n: 1n', 'case 0n: 0n'));
  variant('wrong-capacity-growth', d => replace(path.join(d, 'proofs/index/Bits.bend'),
    'Nat.double(power(p))', 'power(p)'));
  variant('inclusive-roundtrip-bound', d => replace(path.join(d, 'proofs/index/LAWS.bend'),
    'law compact_roundtrip:\n  for +x: U64\n  for +mask: U64\n  for bounded: {Nat.is_lt',
    'law compact_roundtrip:\n  for +x: U64\n  for +mask: U64\n  for bounded: {Nat.is_le'));
  variant('unrestricted-roundtrip', d => replace(path.join(d, 'proofs/index/LAWS.bend'),
    'law compact_roundtrip:\n  for +x: U64\n  for +mask: U64\n  for bounded: {Nat.is_lt(Bits.value(x), Bits.capacity(mask)) == True{} : Bool}',
    'law compact_roundtrip:\n  for +x: U64\n  for +mask: U64\n  for bounded: {True{} == True{} : Bool}'));
  variant('wrong-zero-width-projection', d => replace(path.join(d, 'proofs/index/Bits.bend'),
    'case 0n: Word.zero(1n+p)', 'case 0n: w'));

  const unsafe = copy('unsafe-helper');
  replace(path.join(unsafe, 'proofs/index/Words.bend'), 'def reverse(', '@unsafe\ndef reverse(');
  assert.throws(() => policy(path.join(unsafe, 'proofs/index/PROOF.bend'), unsafe), /unsafe dependency/);
  const warning = invoke(path.join(unsafe, 'proofs/index/PROOF.bend'));
  assert.match(warning.text, /unsafe or foreign/);
  assert.throws(() => clean(warning));
  negatives.push({name: 'unsafe-helper', rejected: true, raw_cli_status: warning.status});

  const foreign = copy('foreign-import');
  fs.appendFileSync(path.join(foreign, 'proofs/index/Words.bend'), '\nimport "./oracle.c"\n');
  assert.throws(() => policy(path.join(foreign, 'proofs/index/PROOF.bend'), foreign), /foreign or unexpected import/);
  negatives.push({name: 'foreign-import', rejected: true});
  const omitted = copy('omitted-new-law');
  replace(path.join(omitted, 'proofs/index/LAWS.bend'), 'law extraction_bound:', 'def not_a_law:');
  assert.throws(() => assert.deepEqual(names(path.join(omitted, 'proofs/index/LAWS.bend')), required));
  negatives.push({name: 'omitted-new-law', rejected: true});

  mutatedBase('corrupt-extract-padding', 'WCon{False{}, w}\n    case 1n+p:\n      match w:',
    'WCon{True{}, w}\n    case 1n+p:\n      match w:');
  mutatedBase('corrupt-deposit-selected-bit', 'WCon{b, Word.pdep(p, m, t)}',
    'WCon{False{}, Word.pdep(p, m, t)}');
  mutatedBase('truncate-public-u64-extraction',
    'U64.from_word(Word.pext(64n, U64.to_word(a), U64.to_word(mask)))',
    'U64.from_u32(U64.low(U64.from_word(Word.pext(64n, U64.to_word(a), U64.to_word(mask)))))');

  assert.deepEqual(verifyCompiler(compiler), identity, 'compiler changed during tests');
  assert.deepEqual(sourceHashes(), sources, 'sources changed during tests');
  const report = {proof_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_universal_laws: required, new_law_count: required.length,
    parent_law_count: 8, inherited_u64_law_count: 16, parent_negative_controls: 11,
    negative_controls: negatives, consumer: 'PASS', source_sha256s: sources,
    scope: 'Bounded PEXT/PDEP bijection and exact extraction range; NOT carry-rippler order, table lookup refinement or native correctness'};
  const json = JSON.stringify(report, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
