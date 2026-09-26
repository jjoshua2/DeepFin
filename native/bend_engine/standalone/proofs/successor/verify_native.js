// Bounded opt-in native qualification of the actual U64 subtraction and masked successor.
// No Python, model, perft, runtime mutation, benchmark, or engine changes.
// Usage: bun native/bend_engine/standalone/proofs/successor/verify_native.js COMPILER [--report FILE]
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
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-successor-native-'));
const cc = process.env.CC || 'clang';
const max64 = (1n << 64n) - 1n;
const sha = x => createHash('sha256').update(x).digest('hex');
const halves = x => `${x >> 32n} ${x & 0xffffffffn}`;
const files = ['Subsets.bend', 'Text.bend', 'proofs/successor/probe.bend', 'proofs/successor/verify_native.js'];
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
  // Add specific equal-low-half and cross-half borrow cases.
  for (const mask of [0x100000000n,0xffffffffn,0x100000001n,max64])
    for (const x of [0n,1n,0xffffffffn,0x100000000n,0x200000000n,max64]) add(mask,x);
  const populations = new Set();
  let borrows = 0, noBorrows = 0, wraps = 0, advances = 0;
  const expected = fixtures.map(({mask,x},id) => {
    const bits = positions(mask), size = 1n << BigInt(bits.length);
    populations.add(bits.length);
    const subset = x & mask, before = extract(subset,bits);
    const after = (before+1n) % size;
    const next = deposit(after,bits);
    const diff = (x-mask) & max64;
    if ((x & 0xffffffffn) < (mask & 0xffffffffn)) ++borrows; else ++noBorrows;
    if (after === 0n) ++wraps; else ++advances;
    assert.equal(extract(next,bits),after);
    assert.equal(next & mask,next);
    return `s ${id} ${bits.length} ${halves(diff)} ${halves(next)} ${halves(before)} ${halves(after)}`;
  }).join('\n') + '\n';
  assert.ok(fixtures.length <= 2048);
  assert.deepEqual([...populations].sort((a,b) => a-b), Array.from({length:65},(_,i)=>i));
  assert.ok(borrows > 0 && noBorrows > 0 && wraps > 0 && advances > 0);
  assert.throws(() => compare(expected.replace('s 0 ', 's 1 '), expected), /native row/);
  // Explicit counterexample to dropping the source law's membership precondition.
  assert.equal(extract((2n-5n)&5n,positions(5n)),3n);
  assert.equal((extract(2n,positions(5n))+1n)%4n,1n);
  const operands = fixtures.flatMap(f => [...halves(f.mask).split(' '), ...halves(f.x).split(' ')]);
  const cfile = path.join(temp, 'probe.c');
  run(process.execPath, [cli, path.join(root, 'proofs/successor/probe.bend'), '-o', cfile], 120000);
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
      assert.match(r.stderr + r.stdout, /invalid successor test integer|expected mask_hi|successor test budget exceeded/);
    }
    reports.push({mode, operand_pairs: fixtures.length, low_half_borrows: borrows, low_half_no_borrows: noBorrows,
      successor_wraps: wraps, successor_advances: advances, malformed_or_budget_rejections: invalid.length,
      output_sha256: sha(actual)});
  }
  assert.deepEqual(sourceHashes(), sources);
  assert.deepEqual(verifyCompiler(compiler), identity);
  const result = {native_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    source_sha256s: sources, population_range: [0,64], distinct_operand_pairs: fixtures.length,
    arbitrary_masks: 'bounded seeded cases, not exhaustive U64 pairs',
    oracle: 'BigInt full-width subtraction and independently scattered compact successor modulo exact 2^k',
    modes: reports, cc: run(cc, ['--version']).split('\n')[0], bun: process.versions.bun,
    scope: 'Native public U64.sub and actual Subsets.next/PEXT across full-width masks; NOT affine-table or engine/model qualification'};
  const json = JSON.stringify(result, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
