// Opt-in bounded native tests; no Python, models, perft or benchmarks.
// Usage: bun native/bend_engine/standalone/proofs/verify_native.js COMPILER [--report FILE]
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../verify_compiler.js';

const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify_native.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]);
const identity = verifyCompiler(compiler);
const root = path.resolve(import.meta.dirname, '..');
const repo = path.resolve(root, '../../..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-subset-native-'));
const cc = process.env.CC || 'clang';
const cli = path.join(compiler, 'bend2/main.ts');
const max64 = (1n << 64n) - 1n;
const halves = n => `${n >> 32n} ${n & 0xffffffffn}`;
const sha = text => createHash('sha256').update(text).digest('hex');
const sourcePaths = ['Subsets.bend', 'Tables.bend', 'Text.bend', 'table_dump.bend',
  'table_reference.c', '../legal_probe/support.c', '../legal_probe/support.h',
  'proofs/subset_probe.bend', 'proofs/verify_native.js'];
const sourceHashes = () => Object.fromEntries(sourcePaths.map(f =>
  [f, sha(fs.readFileSync(path.join(root, f)))]));
const sources = sourceHashes();
const modes = [['generic', []], ['portable', ['-DBEND_U64_PORTABLE']],
  ['native', ['-march=native']], ['ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all']]];

function invoke(command, argv, timeout = 60000) {
  const p = spawnSync(command, argv, {encoding: 'utf8', timeout, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(p.error, undefined, `${command}: ${p.error}`);
  assert.equal(p.signal, null, `${command}: signal ${p.signal}`);
  return p;
}
function run(command, argv, timeout) {
  const p = invoke(command, argv, timeout);
  if (p.status !== 0) throw new Error(`${command} ${argv.join(' ')}: exit ${p.status}\n${p.stderr}\n${p.stdout.slice(0, 4000)}`);
  assert.equal(p.stderr, '', `${command}: unexpected diagnostics`);
  return p.stdout;
}
function equalLines(actual, expected, label) {
  if (actual === expected) return;
  const a = actual.trimEnd().split('\n'), b = expected.trimEnd().split('\n');
  const at = a.findIndex((line, i) => line !== b[i]);
  throw new Error(`${label}: mismatch at line ${at + 1}; actual=${a[at]}, expected=${b[at]}; lengths ${a.length}/${b.length}`);
}
function positions(mask) {
  const bits = [];
  for (let b = 0n; b < 64n; ++b) if (mask & (1n << b)) bits.push(1n << b);
  return bits;
}
// Direct compact-bit deposition, independent of the subtraction recurrence.
function deposit(index, bits) {
  let value = 0n;
  for (let b = 0; b < bits.length; ++b) if (index & (1n << BigInt(b))) value |= bits[b];
  return value;
}
function extract(value, bits) {
  let index = 0n;
  for (let b = 0; b < bits.length; ++b) if (value & bits[b]) index |= 1n << BigInt(b);
  return index;
}
function rays(sq, bishop) {
  const directions = bishop ? [[1,1], [1,-1], [-1,1], [-1,-1]] : [[1,0], [-1,0], [0,1], [0,-1]];
  return directions.map(([dx, dy]) => {
    const ray = [];
    for (let x = sq % 8 + dx, y = Math.floor(sq / 8) + dy;
         x >= 0 && x < 8 && y >= 0 && y < 8; x += dx, y += dy) ray.push(1n << BigInt(x + y * 8));
    return ray;
  });
}
function attacks(lines, occupied) {
  let result = 0n;
  for (const ray of lines) for (const bit of ray) {
    result |= bit;
    if (occupied & bit) break;
  }
  return result;
}

try {
  const fixtures = [];
  const table = new Array(108160).fill(null);
  let offset = 512;
  const populations = {rook: new Set(), bishop: new Set()};
  for (let key = 0; key < 128; ++key) {
    const bishop = key >= 64, lines = rays(key % 64, bishop);
    const mask = lines.flatMap(ray => ray.slice(0, -1)).reduce((x, b) => x | b, 0n);
    const bits = positions(mask), count = 2 ** bits.length;
    populations[bishop ? 'bishop' : 'rook'].add(bits.length);
    assert.ok(bits.length <= (bishop ? 9 : 12));
    table[key] = mask;
    table[128 + key] = BigInt(offset);
    fixtures.push({mask, start: 0n, count, bits, label: `chess-${key}`});
    for (let i = 0; i < count; ++i) table[offset + i] = attacks(lines, deposit(BigInt(i), bits));
    offset += count;
  }
  assert.equal(offset, 108160, 'independent geometry must cover the existing logical layout');
  const chessRows = offset - 512;
  const custom = [];
  function add(mask, startIndex, count, label) {
    const bits = positions(mask);
    custom.push({mask, start: deposit(startIndex, bits), count, bits, label});
  }
  add(0n, 0n, 1, 'empty');
  for (let b = 0n; b < 64n; ++b) add(1n << b, 0n, 2, `single-${b}`);
  // All populations including 64. High-index suffixes test wrap without 2^64 steps.
  for (let k = 1; k <= 64; ++k) {
    const size = 1n << BigInt(k), mask = ((size - 1n) << BigInt(64 - k)) & max64;
    add(mask, 0n, Number(size < 65n ? size : 65n), `prefix-pop-${k}`);
    add(mask, size - 2n, 3, `suffix-pop-${k}`);
  }
  let seed = 0x123456789abcdef1n;
  for (let n = 0; n < 16; ++n) {
    seed ^= (seed << 13n) & max64;
    seed ^= seed >> 7n;
    seed = (seed ^ (seed << 17n)) & max64;
    const size = 1n << BigInt(positions(seed).length);
    add(seed, 0n, Number(size < 65n ? size : 65n), `mixed-prefix-${n}`);
    add(seed, size - 2n, 3, `mixed-suffix-${n}`);
  }
  assert.ok(custom.length <= 256);
  fixtures.push(...custom);
  const commandArgs = custom.flatMap(f => [...halves(f.mask).split(' '), ...halves(f.start).split(' '), String(f.count)]);
  const expectedLines = [];
  let stateRows = 0;
  fixtures.forEach((f, id) => {
    const modulus = 1n << BigInt(f.bits.length), start = extract(f.start, f.bits);
    expectedLines.push(`begin ${id} ${halves(f.mask)} ${halves(f.start)} ${f.count} ${f.bits.length}`);
    for (let i = 0; i < f.count; ++i) {
      const compact = (start + BigInt(i)) % modulus;
      expectedLines.push(`v ${i} ${halves(deposit(compact, f.bits))} ${halves(compact)}`);
    }
    expectedLines.push(`end ${halves(deposit((start + BigInt(f.count)) % modulus, f.bits))}`);
    stateRows += f.count;
  });
  const expected = expectedLines.join('\n') + '\n';
  assert.throws(() => equalLines(expected.replace('\nv 0 ', '\nv 1 '), expected, 'mutated ordinal'), /mismatch/);

  for (const [name, source] of [['subset', path.join(root, 'proofs/subset_probe.bend')],
                                ['tables', path.join(root, 'table_dump.bend')]]) {
    run(process.execPath, [cli, source, '-o', path.join(temp, name + '.c')], 120000);
  }
  const reference = path.join(temp, 'table-reference');
  run(cc, ['-std=c11', '-O1', '-I', repo, path.join(root, 'table_reference.c'),
    path.join(root, '../legal_probe/support.c'), '-pthread', '-lm', '-o', reference], 120000);
  const referenceTable = run(reference, []);
  const cTable = referenceTable.trimEnd().split('\n');
  assert.equal(cTable.length, 108160);
  for (let i = 0; i < table.length; ++i) if (table[i] !== null) {
    assert.equal(cTable[i], halves(table[i]), `independent ray/deposition reference at ${i}`);
  }
  const results = [];
  for (const [mode, flags] of modes) {
    for (const name of ['subset', 'tables']) {
      run(cc, ['-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', ...flags,
        path.join(temp, name + '.c'), '-pthread', '-lm', '-o', path.join(temp, `${name}-${mode}`)], 120000);
    }
    const executable = path.join(temp, `subset-${mode}`);
    const actual = run(executable, ['--threads', '1', ...commandArgs]);
    equalLines(actual, expected, `subset ${mode}`);
    const actualTable = run(path.join(temp, `tables-${mode}`), ['--threads', '1']);
    equalLines(actualTable, referenceTable, `actual Tables.build ${mode}`);
    const failures = [
      ['junk', '0', '0', '0', '1'], ['4294967296', '0', '0', '0', '1'],
      ['0', '0', '0', '0', '0'], ['0', '0', '0', '0', '4097'], ['0'],
    ];
    for (const invalid of failures) {
      const p = invoke(executable, ['--threads', '1', ...invalid]);
      assert.equal(p.status, 2, `invalid input must be rejected: ${invalid}`);
      assert.match(p.stderr + p.stdout, /invalid subset integer|test budget|expected mask_hi/);
    }
    results.push({mode, state_rows: stateRows, table_entries: 108160,
      invalid_requests_rejected: failures.length, subset_sha256: sha(actual), table_sha256: sha(actualTable)});
  }
  assert.deepEqual(verifyCompiler(compiler), identity, 'compiler changed during checks');
  assert.deepEqual(sourceHashes(), sources, 'source changed during checks');
  const result = {
    native_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    source_sha256s: sources,
    chess_masks: 128, exhaustive_chess_states: chessRows, synthetic_cases: custom.length,
    synthetic_states: stateRows - chessRows, total_state_rows: stateRows,
    populations: Object.fromEntries(Object.entries(populations).map(([k, v]) => [k, [...v].sort((a,b) => a-b)])),
    oracle: 'Independent geometric rays and direct compact-bit deposition; separate unchanged CBoard table reference',
    modes: results, cc: run(cc, ['--version']).split('\n')[0], bun: process.versions.bun,
    scope: 'Native enumeration and all existing logical table entries; NOT a quantified source proof of ordering/geometry',
  };
  const json = JSON.stringify(result, null, 2) + '\n';
  if (args[2]) fs.writeFileSync(args[2], json);
  console.log(json.trimEnd());
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
