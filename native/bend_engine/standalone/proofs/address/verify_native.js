// Bounded native exact-index/ordinal probe. No table-array, engine or model build.
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
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '../..'), engine = path.dirname(root);
const cc = process.env.CC || 'clang', limit64 = (1n << 64n) - 1n, low32 = 0xffffffffn;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-exact-indices-native-'));
const sha = b => createHash('sha256').update(b).digest('hex');
const halves = x => [String(x >> 32n), String(x & low32)];
const files = ['standalone/Tables.bend', 'standalone/Subsets.bend', 'standalone/Text.bend',
  'standalone/proofs/LAWS.bend', 'standalone/proofs/PROOF.bend', 'standalone/proofs/Mask.bend',
  'standalone/proofs/u64/LAWS.bend', 'standalone/proofs/u64/PROOF.bend', 'standalone/proofs/u64/Words.bend',
  'standalone/proofs/layout/Spec.bend', 'bitboard_probe/Sliders.bend',
  'standalone/proofs/address/probe.bend', 'standalone/proofs/address/verify_native.js'];
const hashes = () => Object.fromEntries(files.map(f => [f, sha(fs.readFileSync(path.join(engine, f)))]));
const before = hashes();
function invoke(command, argv, timeout = 60000) {
  const r = spawnSync(command, argv, {encoding: 'utf8', timeout, maxBuffer: 24 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, `${command}: ${r.error}`); assert.equal(r.signal, null);
  return r;
}
function run(command, argv, timeout) {
  const r = invoke(command, argv, timeout); assert.equal(r.status, 0, r.stderr + '\n' + r.stdout.slice(0, 3000));
  assert.equal(r.stderr, '', 'unexpected diagnostics'); return r.stdout;
}
function maskFor(key) {
  const square = key % 64, x = square % 8, y = Math.floor(square / 8);
  const dirs = key < 64 ? [[1, 0], [-1, 0], [0, 1], [0, -1]] : [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  let result = 0n;
  for (const [dx, dy] of dirs) {
    const ray = [];
    for (let f = x + dx, r = y + dy; f >= 0 && f < 8 && r >= 0 && r < 8; f += dx, r += dy) ray.push(r * 8 + f);
    for (const square of ray.slice(0, -1)) result |= 1n << BigInt(square);
  }
  return result;
}
function positions(mask) { return Array.from({length: 64}, (_, i) => 1n << BigInt(i)).filter(bit => mask & bit); }
function extract(x, bits) { let result = 0n; bits.forEach((bit, i) => { if (x & bit) result |= 1n << BigInt(i); }); return result; }
function deposit(x, bits) { let result = 0n; bits.forEach((bit, i) => { if (x & (1n << BigInt(i))) result |= bit; }); return result; }
function compare(actual, expected) {
  const a = actual.trimEnd().split('\n'), b = expected.trimEnd().split('\n');
  assert.equal(a.length, b.length, 'row count'); a.forEach((s, i) => assert.equal(s, b[i], `row ${i}`));
}
try {
  const fixtures = [], seen = new Set();
  function add(key, mask, occ, n) {
    const id = [key, mask, occ, n].join(':');
    assert.ok(n >= 0 && n <= 4096);
    if (!seen.has(id)) { seen.add(id); fixtures.push({key, mask, occ, n}); }
  }
  let rng = 0x59feb2c3a617904dn;
  function random() { rng ^= (rng << 13n) & limit64; rng ^= rng >> 7n; rng = (rng ^ (rng << 17n)) & limit64; return rng; }
  for (let key = 0; key < 128; key++) {
    const mask = maskFor(key), size = 2 ** positions(mask).length;
    for (const n of [0, 1, size >> 1, size - 1, size]) add(key, mask, random(), n);
    for (const occ of [0n, limit64, mask, limit64 ^ mask, 1n << 63n, low32, low32 << 32n]) {
      add(key, mask, occ, Number(random() % BigInt(size)));
    }
  }
  // General <=32-bit masks, and explicit >32-bit out-of-domain counterexamples.
  for (let k = 0; k <= 64; k++) {
    const consecutive = (1n << BigInt(k)) - 1n;
    let scattered = 0n;
    for (let j = 0; j < k; j++) scattered |= 1n << BigInt((j * 17 + 31) % 64);
    for (const mask of [consecutive, scattered]) {
      for (const [occ, n] of [[limit64, 0], [random(), 1], [1n << 63n, 31], [mask, 4096]]) add(128, mask, occ, n);
    }
  }
  for (const mask of [low32, low32 << 32n, 0x5555555555555555n, 0xaaaaaaaaaaaaaaaan]) {
    for (const occ of [0n, limit64, 1n << 31n, 1n << 32n, 1n << 63n]) add(128, mask, occ, 17);
  }
  assert.ok(fixtures.length < 4096);
  const counts = {chess_rows: 0, raw_rows: 0, width_32_rows: 0, wider_than_32_rows: 0,
    truncation_counterexamples: 0, first_cycle_rows: 0, cycle_or_later_rows: 0};
  const populations = new Set();
  const expected = fixtures.map(({key, mask, occ, n}, id) => {
    const bits = positions(mask), k = bits.length, capacity = 1n << BigInt(k);
    populations.add(k); counts[key < 128 ? 'chess_rows' : 'raw_rows']++;
    if (k === 32) counts.width_32_rows++;
    if (k > 32) counts.wider_than_32_rows++;
    const full = extract(occ, bits), index = full & low32;
    if (full !== index) { assert.ok(k > 32); counts.truncation_counterexamples++; }
    const ordinal = BigInt(n) % capacity, state = deposit(ordinal, bits), restored = deposit(index, bits);
    counts[BigInt(n) < capacity ? 'first_cycle_rows' : 'cycle_or_later_rows']++;
    if (k <= 32) { assert.equal(index, full); assert.equal(restored, occ & mask); }
    assert.equal(extract(state, bits), ordinal);
    return ['v', id, key, ...halves(mask), k, n, String(index), ...halves(full),
      ...halves(state), String(ordinal & low32), ...halves(restored)].join(' ');
  }).join('\n') + '\n';
  assert.equal(populations.size, 65); assert.ok(counts.truncation_counterexamples > 0);
  assert.throws(() => compare(expected.replace('v 0 ', 'v 1 '), expected), /row 0/);
  const cfile = path.join(temp, 'probe.c');
  run(process.execPath, [path.join(compiler, 'bend2/main.ts'), path.join(root, 'proofs/address/probe.bend'), '-o', cfile], 120000);
  const operands = fixtures.flatMap(({key, mask, occ, n}) => [String(key), ...halves(key === 128 ? mask : 0n), ...halves(occ), String(n)]);
  const modes = [];
  for (const [mode, flags] of [['generic', []], ['portable', ['-DBEND_U64_PORTABLE']],
    ['native', ['-march=native']], ['ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all']]]) {
    const exe = path.join(temp, mode);
    run(cc, ['-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', ...flags, cfile, '-pthread', '-lm', '-o', exe], 120000);
    const actual = run(exe, ['--threads', '1', ...operands], 120000); compare(actual, expected);
    const invalid = [['129', '0', '0', '0', '0', '0'], ['0', '0', '0', '0', '0', '4097'],
      ['128', '4294967296', '0', '0', '0', '0'], ['0', '0', '0', '0', 'junk', '0'], ['0', '0', '0']];
    for (const words of invalid) {
      const r = invoke(exe, ['--threads', '1', ...words]);
      assert.equal(r.status, 2); assert.match(r.stdout + r.stderr, /invalid exact-index|expected key/);
    }
    modes.push({mode, operand_rows: fixtures.length, malformed_or_budget_rejections: invalid.length, output_sha256: sha(actual)});
    console.error('PASS native: ' + mode);
  }
  assert.deepEqual(hashes(), before); assert.deepEqual(verifyCompiler(compiler), identity);
  const result = {native_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    source_sha256s: before, distinct_operand_rows: fixtures.length, chess_keys: 128,
    population_range: [0, 64], ...counts, modes, cc: run(cc, ['--version']).split('\n')[0], bun: process.versions.bun,
    oracle: 'Independent signed-coordinate chess masks, direct BigInt gather/scatter, and mathematical ordinal modulo capacity',
    candidate_recurrence: 'The imported proofs/LAWS.at calls actual Subsets.next',
    scope: 'Exact scalar index, actual recurrence states and reconstruction on bounded fixtures; wide-mask truncation controls are outside the <=32-bit theorem; NOT affine arrays, full engine or performance'};
  const text = JSON.stringify(result, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], text); console.log(text.trimEnd());
} finally { fs.rmSync(temp, {recursive: true, force: true}); }
