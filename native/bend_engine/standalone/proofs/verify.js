// Opt-in source proof gate; no model, native compilation, network, or live edits.
// Usage: bun native/bend_engine/standalone/proofs/verify.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../verify_compiler.js';

const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]);
const identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '..');
const suite = path.join(root, 'proofs');
const cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-subset-proofs-'));
const required = ['step_in_mask', 'sequence_in_mask', 'sequence_extract_deposit',
  'empty_mask_sequence', 'cross_low_half', 'cross_bit63', 'mask_cycle_end', 'full_word_wrap'];
const negatives = [];
const blob = data => createHash('sha1').update(`blob ${data.length}\0`).update(data).digest('hex');
const code = file => fs.readFileSync(file, 'utf8').split('\n').map(s => s.split('#')[0]).join('\n');
const names = file => [...code(file).matchAll(/^law ([a-zA-Z0-9_]+):/gm)].map(m => m[1]);

function invoke(file) {
  const r = spawnSync(process.execPath, [cli, file], {
    encoding: 'utf8', timeout: 30000, maxBuffer: 8 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'},
  });
  assert.equal(r.error, undefined, `checker failed to execute: ${r.error}`);
  assert.equal(r.signal, null, `checker terminated by ${r.signal}`);
  return {status: r.status, text: (r.stdout + r.stderr).trim()};
}
function clean(r) {
  assert.equal(r.status, 0, r.text);
  // Unsafe/foreign warnings may accompany exit zero: output is part of the gate.
  assert.equal(r.text, 'All terms check.', r.text);
}
function policy(file, seen = new Set()) {
  file = path.resolve(file);
  if (seen.has(file)) return;
  seen.add(file);
  const source = code(file);
  assert.doesNotMatch(source, /@unsafe|\?/, `${file}: unsafe code or proof hole`);
  for (const m of source.matchAll(/^\s*import\s+(\S+)/gm)) {
    const dep = m[1];
    if (dep === 'Base') continue;
    assert.match(dep, /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, `${file}: foreign/unexpected import`);
    const target = path.resolve(path.dirname(file), dep);
    assert.ok(fs.existsSync(target), `${file}: missing import ${dep}`);
    policy(target, seen);
  }
}
function inherited(dir) {
  const provenance = JSON.parse(fs.readFileSync(path.join(dir, 'u64/provenance.json')));
  assert.equal(provenance.repository, 'https://github.com/jjoshua2/bend');
  assert.equal(provenance.revision, PIN.revision, 'inherited laws must match the compiler pin');
  assert.equal(provenance.directory, 'demos/proof_u64');
  const expected = {
    'LAWS.bend': '158f45a68c7f24bc79c8cb7e557424a317d315de',
    'PROOF.bend': 'e08dad20df90b9e542aee34b7e775e4402339b9c',
    'Words.bend': '503f1e5f49c4ef01ca45a1a3a010d5b52551a3df',
  };
  assert.deepEqual(provenance.git_blobs, expected, 'inherited manifest changed');
  for (const [name, sha] of Object.entries(expected)) {
    assert.equal(blob(fs.readFileSync(path.join(dir, 'u64', name))), sha, `inherited ${name} changed`);
  }
  return names(path.join(dir, 'u64/LAWS.bend'));
}
function replace(file, before, after) {
  const source = fs.readFileSync(file, 'utf8');
  assert.equal(source.split(before).length, 2, `mutation anchor must occur once: ${before}`);
  fs.writeFileSync(file, source.replace(before, after));
}
function copy(name) {
  const dir = path.join(temp, name);
  fs.mkdirSync(dir);
  fs.copyFileSync(path.join(root, 'Subsets.bend'), path.join(dir, 'Subsets.bend'));
  fs.cpSync(suite, path.join(dir, 'proofs'), {recursive: true});
  return dir;
}
function variant(name, edit, reason) {
  const dir = copy(name);
  edit(dir);
  const result = invoke(path.join(dir, 'proofs/PROOF.bend'));
  assert.equal(result.status, 1, `${name}: checker must reject\n${result.text}`);
  assert.match(result.text, reason, `${name}: wrong rejection reason\n${result.text}`);
  negatives.push({name, rejected: true});
}

try {
  const inheritedLaws = inherited(suite);
  assert.equal(inheritedLaws.length, 16);
  assert.deepEqual(names(path.join(suite, 'LAWS.bend')), required);
  policy(path.join(suite, 'PROOF.bend'));
  clean(invoke(path.join(suite, 'PROOF.bend')));

  variant('missing-proof', d => replace(path.join(d, 'proofs/PROOF.bend'),
    'def Laws.full_word_wrap():\n  {==}\n', ''), /TODO found/);
  variant('missing-laws-import', d => fs.writeFileSync(path.join(d, 'proofs/PROOF.bend'),
    'import Base\n'), /PROOF\.bend must import/);
  variant('false-boundary-law', d => replace(path.join(d, 'proofs/LAWS.bend'),
    '== U64.from_parts(1, 0) : U64}', '== U64.from_parts(1, 1) : U64}'), /Location:.*cross_low_half/);
  variant('proof-hole', d => replace(path.join(d, 'proofs/PROOF.bend'),
    'def Laws.full_word_wrap():\n  {==}', 'def Laws.full_word_wrap():\n  ?TODO'), /TODO found/);
  variant('unmasked-implementation', d => replace(path.join(d, 'Subsets.bend'),
    'U64.and(difference(subset, mask), mask)', 'difference(subset, mask)'), /Location:.*step_in_mask/);
  variant('reversed-subtraction', d => replace(path.join(d, 'Subsets.bend'),
    'U64.sub(subset, mask)', 'U64.sub(mask, subset)'), /Location:.*cross_low_half/);
  variant('stuck-at-zero', d => replace(path.join(d, 'Subsets.bend'),
    'U64.sub(subset, mask)', 'U64.zero()'), /Location:.*cross_low_half/);
  variant('truncated-subtraction', d => replace(path.join(d, 'Subsets.bend'),
    'U64.sub(subset, mask)', 'U64.from_u32(U32.sub(U64.low(subset), U64.low(mask)))'), /Location:.*cross_low_half/);

  const unsafe = copy('unsafe-proof');
  replace(path.join(unsafe, 'proofs/Mask.bend'), 'def u64_absorb(', '@unsafe\ndef u64_absorb(');
  assert.throws(() => policy(path.join(unsafe, 'proofs/PROOF.bend')), /unsafe code/);
  const warned = invoke(path.join(unsafe, 'proofs/PROOF.bend'));
  assert.match(warned.text, /unsafe or foreign/);
  assert.throws(() => clean(warned));
  negatives.push({name: 'unsafe-dependency', rejected: true, raw_cli_status: warned.status});

  const corrupt = copy('inherited-proof-corruption');
  fs.appendFileSync(path.join(corrupt, 'proofs/u64/Words.bend'), '\n# changed\n');
  assert.throws(() => inherited(path.join(corrupt, 'proofs')), /inherited Words.bend changed/);
  negatives.push({name: 'inherited-provenance-change', rejected: true});

  // Missing proof of an inherited law must not disappear from the checked graph.
  variant('missing-inherited-proof', d => replace(path.join(d, 'proofs/u64/PROOF.bend'),
    'def Laws.popcount_zero():\n  {==}\n', ''), /TODO found/);
  const sourceFiles = ['Subsets.bend', 'Tables.bend', 'proofs/LAWS.bend', 'proofs/PROOF.bend',
    'proofs/Mask.bend', 'proofs/u64/LAWS.bend', 'proofs/u64/PROOF.bend', 'proofs/u64/Words.bend'];
  const result = {
    proof_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_source_laws: required, universal_laws: 4, closed_boundary_equalities: 4,
    inherited_source_laws: inheritedLaws, negative_controls: negatives,
    source_blobs: Object.fromEntries(sourceFiles.map(f => [f, blob(fs.readFileSync(path.join(root, f)))])),
    scope: 'Source mask membership and extract/deposit recovery; NOT complete ordering or affine-table refinement',
  };
  assert.deepEqual(verifyCompiler(compiler), identity, 'compiler changed during checks');
  const json = JSON.stringify(result, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
