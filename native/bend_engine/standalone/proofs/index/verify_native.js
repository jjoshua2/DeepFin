// Bounded opt-in native qualification of the exact PEXT/PDEP contracts.
// No Python, model, perft, runtime mutation, benchmark, or engine changes.
// Usage: bun native/bend_engine/standalone/proofs/index/verify_native.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';

const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify_native.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]);
const identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '../..');
const cli = path.join(compiler, 'bend2/main.ts');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-compact-native-'));
const cc = process.env.CC || 'clang';
const max64 = (1n << 64n) - 1n;
const sha = x => createHash('sha256').update(x).digest('hex');
const halves = x => `${x >> 32n} ${x & 0xffffffffn}`;
const files = ['Text.bend', 'proofs/index/probe.bend', 'proofs/index/verify_native.js'];
const sourceHashes = () => Object.fromEntries(files.map(f => [f, sha(fs.readFileSync(path.join(root, f)))]));
const sources = sourceHashes();
const modes = [['generic', []], ['portable', ['-DBEND_U64_PORTABLE']],
  ['native', ['-march=native']], ['ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all']]];

function invoke(command, argv, timeout = 60000) {
  const r = spawnSync(command, argv, {encoding: 'utf8', timeout, maxBuffer: 8 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, `${command}: ${r.error}`);
  assert.equal(r.signal, null, `${command}: signal ${r.signal}`);
  return r;
}
function run(command, argv, timeout) {
  const r = invoke(command, argv, timeout);
  assert.equal(r.status, 0, `${command}: ${r.stderr}\n${r.stdout.slice(0, 4000)}`);
  assert.equal(r.stderr, '', `${command}: unexpected diagnostics`);
  return r.stdout;
}
function positions(mask) {
  const result = [];
  for (let b = 0n; b < 64n; ++b) if (mask & (1n << b)) result.push(1n << b);
  return result;
}
// Direct scatter/gather by bit positions, independent of structural Base code
// and of the carry-rippler arithmetic. BigInt bounds retain 2^64 exactly.
function extract(x, bits) {
  let result = 0n;
  bits.forEach((bit, i) => { if (x & bit) result |= 1n << BigInt(i); });
  return result;
}
function deposit(x, bits) {
  let result = 0n;
  bits.forEach((bit, i) => { if (x & (1n << BigInt(i))) result |= bit; });
  return result;
}
function compare(actual, expected) {
  const a = actual.trimEnd().split('\n'), e = expected.trimEnd().split('\n');
  assert.equal(a.length, e.length, 'native output row count');
  a.forEach((line, i) => assert.equal(line, e[i], `native row ${i}`));
}

try {
  const fixtures = [], seen = new Set();
  function add(mask, x) {
    assert.ok(mask >= 0n && mask <= max64 && x >= 0n && x <= max64);
    const key = `${mask}:${x}`;
    if (!seen.has(key)) { seen.add(key); fixtures.push({mask, x}); }
  }
  for (let k = 0; k <= 64; ++k) {
    const size = 1n << BigInt(k);
    const mask = (size - 1n) << BigInt(64 - k);
    for (const x of [0n, 1n, size - 1n, max64, 1n << 63n, 0xaaaaaaaaaaaaaaaan]) add(mask, x);
    if (size <= max64) { add(mask, size); add(mask, max64 - size + 1n); }
  }
  for (let b = 0n; b < 64n; ++b) {
    const bit = 1n << b;
    for (const x of [0n, 1n, 2n, 3n, bit, max64]) add(bit, x);
    for (const mask of [max64, 0xaaaaaaaaaaaaaaaan, 0x5555555555555555n]) add(mask, bit);
  }
  let state = 0x3cb52f7048493ecbn;
  function random() {
    state ^= (state << 13n) & max64;
    state ^= state >> 7n;
    state = (state ^ (state << 17n)) & max64;
    return state;
  }
  for (let i = 0; i < 256; ++i) {
    const mask = random(), x = random();
    add(mask, x);
    add(mask, x & ((1n << BigInt(positions(mask).length)) - 1n));
  }
  assert.ok(fixtures.length > 1000 && fixtures.length <= 2048);
  const populations = new Set();
  let bounded = 0, truncated = 0;
  const expected = fixtures.map(({mask, x}, id) => {
    const bits = positions(mask), size = 1n << BigInt(bits.length);
    populations.add(bits.length);
    const ext = extract(x, bits), dep = deposit(x, bits), reverse = extract(dep, bits);
    assert.ok(ext >= 0n && ext < size);
    assert.equal(reverse, x & (size - 1n));
    assert.equal(dep & mask, dep);
    assert.equal(deposit(ext, bits), x & mask);
    if (x < size) { ++bounded; assert.equal(reverse, x); }
    else { ++truncated; assert.notEqual(reverse, x); }
    return `v ${id} ${bits.length} ${halves(ext)} ${halves(dep)} ${halves(reverse)} ${halves(deposit(ext, bits))}`;
  }).join('\n') + '\n';
  assert.deepEqual([...populations].sort((a,b) => a-b), Array.from({length: 65}, (_,i) => i));
  assert.ok(bounded > 0 && truncated > 0);
  // Ensure comparison cannot silently ignore a corrupted numeric observation.
  assert.throws(() => compare(expected.replace('v 0 ', 'v 1 '), expected), /native row/);
  assert.equal(extract(deposit(1n, []), []), 0n, 'unrestricted reverse counterexample');
  assert.equal(extract(deposit(2n, [1n << 63n]), [1n << 63n]), 0n, 'inclusive-bound counterexample');
  const operands = fixtures.flatMap(f => [...halves(f.mask).split(' '), ...halves(f.x).split(' ')]);
  const cfile = path.join(temp, 'probe.c');
  run(process.execPath, [cli, path.join(root, 'proofs/index/probe.bend'), '-o', cfile], 120000);
  const reports = [];
  for (const [mode, flags] of modes) {
    const exe = path.join(temp, 'probe-' + mode);
    run(cc, ['-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', ...flags,
      cfile, '-pthread', '-lm', '-o', exe], 120000);
    const actual = run(exe, ['--threads', '1', ...operands]);
    compare(actual, expected);
    const invalid = [['junk', '0', '0', '0'], ['4294967296', '0', '0', '0'],
      ['0', '0', '0'], Array(2049 * 4).fill('0')];
    for (const test of invalid) {
      const r = invoke(exe, ['--threads', '1', ...test]);
      assert.equal(r.status, 2, `${mode}: invalid/budget case accepted`);
      assert.match(r.stderr + r.stdout, /invalid compact test integer|expected mask_hi|compact test budget exceeded/);
    }
    reports.push({mode, operand_pairs: fixtures.length, bounded_inputs: bounded,
      out_of_range_inputs: truncated, malformed_or_budget_rejections: invalid.length,
      output_sha256: sha(actual)});
  }
  assert.deepEqual(sourceHashes(), sources);
  assert.deepEqual(verifyCompiler(compiler), identity);
  const result = {native_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    source_sha256s: sources, population_range: [0,64], distinct_operand_pairs: fixtures.length,
    arbitrary_masks: 'bounded seeded cases, not exhaustive U64 pairs',
    oracle: 'BigInt direct bit-position scatter/gather with exact mathematical 2^k',
    modes: reports, cc: run(cc, ['--version']).split('\n')[0], bun: process.versions.bun,
    scope: 'Native PEXT/PDEP range, both compositions and truncation; NOT enumeration order or complete engine qualification'};
  const json = JSON.stringify(result, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
