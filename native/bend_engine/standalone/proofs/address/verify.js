// Opt-in exact-index gate. Existing gates, accepted laws and compiler are unchanged.
// Usage: bun native/bend_engine/standalone/proofs/address/verify.js COMPILER [--report FILE]
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
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '../..'), engine = path.dirname(root);
const suite = path.join(root, 'proofs/address'), cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-exact-indices-'));
const required = ['low_projection_exact', 'compact_index_exact', 'chess_lookup_exact',
  'chess_lookup_word', 'chess_sequence_ordinal', 'chess_sequence_coverage',
  'chess_lookup_injective', 'chess_lookup_redeposit'];
const sha = b => createHash('sha256').update(b).digest('hex');
const code = f => fs.readFileSync(f, 'utf8').split('\n').map(s => s.split('#')[0]).join('\n');
const negatives = [];

function invoke(file, timeout = 600000) {
  const r = spawnSync(process.execPath, [cli, file], {encoding: 'utf8', timeout,
    maxBuffer: 16 << 20, env: {...process.env, BEND_NO_TELEMETRY: '1'}});
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
  const rel = path.relative(boundary, file);
  assert.ok(!rel.split(path.sep).includes('..'), 'proof path escapes boundary');
  assert.ok(rel.startsWith('standalone/') || rel === 'bitboard_probe/Sliders.bend',
    'proof outside approved implementation boundary');
  assert.ok(fs.lstatSync(file).isFile(), 'proof input must be a regular file');
  assert.equal(fs.realpathSync(file), file, 'symlinked proof path');
  if (seen.has(file)) return seen;
  seen.add(file);
  const s = code(file);
  assert.doesNotMatch(s, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const m of s.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (m[1] === 'Base') continue;
    assert.match(m[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign or unexpected import');
    policy(path.resolve(path.dirname(file), m[1]), boundary, seen);
  }
  return seen;
}
function manifest(s) {
  assert.deepEqual([...code(path.join(s, 'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m => m[1]), required);
  const proof = code(path.join(s, 'PROOF.bend'));
  assert.deepEqual([...proof.matchAll(/^def Laws\.(\w+)\(/gm)].map(m => m[1]), required);
  assert.match(proof, /import \.\/LAWS\.bend as Laws/);
  assert.match(proof, /import \.\.\/layout\/PROOF\.bend as LayoutProof/);
  assert.match(code(path.join(s, 'consumer.bend')), /import \.\/PROOF\.bend as Proof/);
}
function snapshot(files) {
  return Object.fromEntries([...files].sort().map(f => [path.relative(engine, f), sha(fs.readFileSync(f))]));
}
function replace(f, before, after) {
  const s = fs.readFileSync(f, 'utf8');
  assert.equal(s.split(before).length, 2, `mutation anchor must be unique: ${before}`);
  fs.writeFileSync(f, s.replace(before, after));
}
function copy(name) {
  const e = path.join(temp, name), r = path.join(e, 'standalone');
  fs.mkdirSync(r, {recursive: true});
  for (const f of ['Tables.bend', 'Subsets.bend']) fs.copyFileSync(path.join(root, f), path.join(r, f));
  fs.cpSync(path.join(root, 'proofs'), path.join(r, 'proofs'), {recursive: true});
  fs.mkdirSync(path.join(e, 'bitboard_probe'));
  fs.copyFileSync(path.join(engine, 'bitboard_probe/Sliders.bend'), path.join(e, 'bitboard_probe/Sliders.bend'));
  return {e, r, s: path.join(r, 'proofs/address')};
}
function semantic(name, edit, entry = 'Correspondence.bend') {
  const d = copy(name); edit(d);
  const result = invoke(path.join(d.s, entry), 60000);
  assert.equal(result.status, 1, `${name}: must reject\n${result.text}`);
  assert.match(result.text, /expected[\s\S]*observed/, `${name}: not a proof rejection\n${result.text}`);
  negatives.push({name, rejected: true, kind: 'checker rejects affected new lemma', entry});
  console.error('PASS rejection: ' + name);
}
function rejectedPolicy(name, edit, check, reason) {
  const d = copy(name); edit(d); assert.throws(() => check(d), reason);
  negatives.push({name, rejected: true, kind: 'manifest/import policy'});
}

try {
  manifest(suite);
  const graph = policy(path.join(suite, 'consumer.bend'), engine);
  const tracked = new Set([...graph, path.join(suite, 'verify.js'),
    path.join(suite, 'verify_native.js'), path.join(suite, 'probe.bend'),
    path.join(root, 'verify_compiler.js'), path.join(root, 'toolchain.json')]);
  const before = snapshot(tracked);
  const r = spawnSync(process.execPath, [path.join(root, 'proofs/layout/verify.js'), compiler],
    {encoding: 'utf8', timeout: 900000, maxBuffer: 24 << 20,
      env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, `${r.error}`); assert.equal(r.signal, null);
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const parent = JSON.parse(r.stdout);
  assert.equal(parent.proof_gate, 'PASS'); assert.equal(parent.aggregate_law_count, 48);
  assert.equal(parent.negative_controls.length, 13); assert.equal(parent.inherited_negative_controls, 42);
  assert.ok(parent.negative_controls.every(n => n.rejected));
  console.error('PASS: unchanged 48-law parent and 55 inherited controls');
  clean(invoke(path.join(suite, 'consumer.bend')));
  console.error('PASS: all 56 laws and importing consumer');

  const lookup = d => path.join(d.e, 'bitboard_probe/Sliders.bend');
  const expression = 'U64.low(U64.pext(occupied, mask))';
  semantic('in-range-but-wrong-zero-index', d => replace(lookup(d), expression, '0'));
  semantic('wrong-high-half-projection', d => replace(lookup(d), expression, 'U64.high(U64.pext(occupied, mask))'));
  semantic('off-by-one-index', d => replace(lookup(d), expression, 'U32.inc(U64.low(U64.pext(occupied, mask)))'));
  semantic('index-truncated-to-one-bit', d => replace(lookup(d), expression, 'U32.and(U64.low(U64.pext(occupied, mask)),1)'));
  semantic('swapped-pext-operands', d => replace(lookup(d), expression, 'U64.low(U64.pext(mask, occupied))'));
  semantic('allows-one-more-bit-than-retained', d => replace(path.join(d.s, 'Projection.bend'),
    'retained: {Nat.is_le(k,n) == True{} : Bool}', 'retained: {Nat.is_le(k,1n+n) == True{} : Bool}'), 'Projection.bend');
  semantic('removed-recovery-membership', d => replace(path.join(d.s, 'Correspondence.bend'),
    'member: {U64.and(s,Layout.mask(key)) == s : U64}', 'member: {True{} == True{} : Bool}'));
  semantic('inclusive-ordinal-bound', d => replace(path.join(d.s, 'Correspondence.bend'),
    'bounded: {Nat.is_lt(i,Bits.capacity(Layout.mask(key)))', 'bounded: {Nat.is_le(i,Bits.capacity(Layout.mask(key)))'));

  rejectedPolicy('omitted-law', d => replace(path.join(d.s, 'LAWS.bend'),
    'law chess_lookup_exact:', 'def omitted:'), d => manifest(d.s), /./);
  rejectedPolicy('omitted-proof-definition', d => replace(path.join(d.s, 'PROOF.bend'),
    'def Laws.chess_lookup_exact(', 'def omitted('), d => manifest(d.s), /./);
  rejectedPolicy('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'),
    'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  rejectedPolicy('missing-inherited-layout-proof', d => replace(path.join(d.s, 'PROOF.bend'),
    'import ../layout/PROOF.bend as LayoutProof\n', ''), d => manifest(d.s), /./);
  rejectedPolicy('missing-consumer-proof-import', d => replace(path.join(d.s, 'consumer.bend'),
    'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  rejectedPolicy('proof-hole', d => fs.appendFileSync(path.join(d.s, 'Projection.bend'), '\n?TODO\n'),
    d => policy(path.join(d.s, 'consumer.bend'), d.e), /proof hole/);
  rejectedPolicy('foreign-proof-import', d => fs.appendFileSync(path.join(d.s, 'Projection.bend'), '\nimport "./oracle.c"\n'),
    d => policy(path.join(d.s, 'consumer.bend'), d.e), /foreign or unexpected import/);
  rejectedPolicy('symlinked-proof-input', d => {
    const file = path.join(d.s, 'Projection.bend'); fs.renameSync(file, file + '.original');
    fs.symlinkSync('Projection.bend.original', file);
  }, d => policy(path.join(d.s, 'consumer.bend'), d.e), /regular file/);
  const unsafe = copy('unsafe-projection');
  replace(path.join(unsafe.s, 'Projection.bend'), 'def low_exact(', '@unsafe\ndef low_exact(');
  assert.throws(() => policy(path.join(unsafe.s, 'consumer.bend'), unsafe.e), /unsafe dependency/);
  const warned = invoke(path.join(unsafe.s, 'Correspondence.bend'), 60000);
  assert.match(warned.text, /unsafe or foreign/); assert.throws(() => clean(warned));
  negatives.push({name: 'unsafe-projection', rejected: true,
    kind: 'policy and exact checker-output guard', raw_cli_status: warned.status});

  assert.equal(negatives.length, 17);
  assert.deepEqual(snapshot(tracked), before, 'sources changed during verification');
  assert.deepEqual(verifyCompiler(compiler), identity, 'compiler changed during verification');
  const result = {proof_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_universal_laws: required, new_law_count: required.length,
    inherited_law_count: 48, aggregate_law_count: 56, inherited_negative_controls: 55,
    negative_controls: negatives, consumer: 'PASS', source_sha256s: before,
    general_compact_width: 'mathematical Nat k <= 32', chess_key_domain: [0, 127],
    proof_method: 'Structural retained-width induction plus original chess-mask count certificates and actual recurrence ordinal',
    scope: 'Exact low-U32/full-PEXT correspondence, ordinal, reconstruction and injectivity; NOT prefix offsets, affine storage, attack geometry or native correctness'};
  const text = JSON.stringify(result, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], text);
  console.log(text.trimEnd());
} finally { fs.rmSync(temp, {recursive: true, force: true}); }
