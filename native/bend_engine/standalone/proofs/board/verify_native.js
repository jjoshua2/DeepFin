// Opt-in actual Position operations versus an independent per-square set model.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: verify_native.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
const suite = import.meta.dirname, engine = path.resolve(suite, '../../..');
const cc = process.env.CC || 'clang';
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-board-native-'));
const sha = x => createHash('sha256').update(x).digest('hex');
const names = ['standalone/Position.bend', 'standalone/Text.bend', 'legal_probe/Chess.bend',
  'bitboard_probe/Sliders.bend', 'standalone/proofs/board/probe.bend', 'standalone/proofs/board/verify_native.js'];
const hashes = () => Object.fromEntries(names.map(f => [f, sha(fs.readFileSync(path.join(engine, f)))]));
const before = hashes();
function invoke(command, argv, timeout = 90000) {
  const r = spawnSync(command, argv.map(String), {encoding: 'utf8', timeout, maxBuffer: 32 << 20,
    env: {...process.env, TERM: 'dumb', BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  return r;
}
function run(command, argv, timeout) {
  const r = invoke(command, argv, timeout);
  assert.equal(r.status, 0, (r.stderr + r.stdout).slice(-3000));
  assert.equal(r.stderr, '', 'unexpected tool/native diagnostic'); return r.stdout;
}
const empty = () => Array.from({length: 64}, () => ({kinds: new Set(), colors: new Set()}));
const copy = cells => cells.map(x => ({kinds: new Set(x.kinds), colors: new Set(x.colors)}));
const put = (cells, square, kind, color) => {
  const result = copy(cells); result[square].kinds.add(kind); result[square].colors.add(color); return result;
};
const consistent = cells => cells.every(x => (x.kinds.size === 0 && x.colors.size === 0) ||
  (x.kinds.size === 1 && x.colors.size === 1));
function initial() {
  let cells = empty(); const back = [3, 1, 2, 4, 5, 2, 1, 3];
  for (let file = 0; file < 8; file++) {
    cells = put(cells, file, back[file], 1); cells = put(cells, 8 + file, 0, 1);
    cells = put(cells, 48 + file, 0, 0); cells = put(cells, 56 + file, back[file], 0);
  }
  return cells;
}
function encode(cells, metadata) {
  const planes = Array(8).fill(0n);
  cells.forEach((x, square) => {
    const bit = 1n << BigInt(square);
    for (const kind of x.kinds) planes[kind] |= bit;
    for (const color of x.colors) planes[color === 1 ? 6 : 7] |= bit;
  });
  return [...planes.flatMap(v => [Number(v >> 32n), Number(v & 0xffffffffn)]), ...metadata];
}
function decode(values) {
  const cells = empty(); const planes = Array.from({length: 8}, (_, i) =>
    (BigInt(values[2 * i]) << 32n) | BigInt(values[2 * i + 1]));
  cells.forEach((x, square) => {
    const bit = 1n << BigInt(square);
    for (let kind = 0; kind < 6; kind++) if (planes[kind] & bit) x.kinds.add(kind);
    if (planes[6] & bit) x.colors.add(1); if (planes[7] & bit) x.colors.add(0);
  });
  return cells;
}
let randomState = 0x5b1846f2;
function random() {
  randomState ^= randomState << 13; randomState ^= randomState >>> 17;
  randomState ^= randomState << 5; return randomState >>> 0;
}
function randomBoard() {
  let cells = empty();
  for (let sq = 0; sq < 64; sq++) if (random() % 3 !== 0) cells = put(cells, sq, random() % 6, random() % 2);
  return cells;
}
const fixtures = [], counts = {fresh_insert: 0, repeated_insert: 0, conflicting_insert: 0,
  metadata_valid: 0, metadata_invalid: 0, empty: 0, initial: 0};
function add(category, op, kind, color, square, cells, metadata) {
  let expectedCells = cells, expectedMetadata = metadata;
  if (op === 0) expectedCells = put(cells, square, kind, color);
  if (op === 1) expectedMetadata = [kind, color, square];
  if (op === 2) { expectedCells = empty(); expectedMetadata = [1, 0, 64]; }
  if (op === 3) { expectedCells = initial(); expectedMetadata = [1, 15, 64]; }
  const validBefore = consistent(cells), fresh = op === 0 && cells[square].colors.size === 0;
  if (category === 'fresh_insert') assert.ok(validBefore && fresh && consistent(expectedCells));
  if (category === 'conflicting_insert') assert.equal(consistent(expectedCells), false);
  fixtures.push({category, input: [op, kind, color, square, ...encode(cells, metadata)],
    expected: encode(expectedCells, expectedMetadata), expectedValid: consistent(expectedCells), validBefore, fresh});
  counts[category]++;
}
for (let sq = 0; sq < 64; sq++) for (let kind = 0; kind < 6; kind++) for (let color = 0; color < 2; color++) {
  const cells = randomBoard(); cells[sq] = {kinds: new Set(), colors: new Set()};
  add('fresh_insert', 0, kind, color, sq, cells, [random(), random(), random()]);
}
for (let sq = 0; sq < 64; sq++) {
  const kind = sq % 6, color = sq % 2;
  const cells = put(randomBoard(), sq, kind, color);
  // Reset the selected cell to exactly one occupant before each collision test.
  cells[sq] = {kinds: new Set([kind]), colors: new Set([color])};
  add('repeated_insert', 0, kind, color, sq, cells, [1, 15, 64]);
  add('conflicting_insert', 0, (kind + 1) % 6, 1 - color, sq, cells, [0, 0xffffffff, 0xffffffff]);
  add('metadata_valid', 1, random(), random(), random(), randomBoard(), [random(), random(), random()]);
  const malformed = randomBoard(); malformed[sq] = {kinds: new Set([0, 5]), colors: new Set([0, 1])};
  add('metadata_invalid', 1, random(), random(), random(), malformed, [random(), random(), random()]);
}
add('empty', 2, 0, 0, 0, randomBoard(), [0xffffffff, 0xffffffff, 0xffffffff]);
add('initial', 3, 0, 0, 0, empty(), [0, 0, 0]);
assert.equal(fixtures.length, 1026);
const allKinds = new Set(fixtures.filter(x => x.category === 'fresh_insert').map(x => `${x.input[3]}:${x.input[1]}:${x.input[2]}`));
assert.equal(allKinds.size, 768);
function compare(raw, cases, offset) {
  const lines = raw.trimEnd().split('\n'); assert.equal(lines.length, cases.length, 'row count');
  lines.forEach((line, i) => {
    const got = line.split(' ').map(Number), expected = cases[i].expected;
    assert.equal(got.length, 19, 'complete Board output');
    got.forEach((x, field) => assert.equal(x, expected[field], `row ${offset + i}, field ${field}: ${cases[i].category}`));
    assert.equal(consistent(decode(got)), cases[i].expectedValid, 'independent representation classification');
  });
}
const badBase = [0, 0, 0, 0, ...encode(empty(), [1, 0, 64])];
const changed = (index, value) => badBase.map((x, i) => i === index ? value : x);
const invalid = [changed(0, 4), changed(1, 6), changed(2, 2), changed(3, 64), changed(4, '4294967296'),
  changed(4, '-1'), changed(4, 'x'), badBase.slice(0, -1), Array.from({length: 65}, () => badBase).flat()];
function executeAll(exe) {
  let raw = '';
  for (let start = 0; start < fixtures.length; start += 64) {
    const chunk = fixtures.slice(start, start + 64);
    const out = run(exe, ['--threads', '1', ...chunk.flatMap(x => x.input)]);
    compare(out, chunk, start); raw += out;
  }
  return raw;
}
try {
  const c = path.join(temp, 'probe.c');
  run(process.execPath, [path.join(compiler, 'bend2/main.ts'), path.join(suite, 'probe.bend'), '-o', c]);
  const modes = [];
  for (const [name, flags] of [['generic', []], ['portable', ['-DBEND_U64_PORTABLE']], ['native', ['-march=native']],
    ['ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all']]]) {
    const exe = path.join(temp, name);
    run(cc, ['-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', ...flags, c, '-lm', '-pthread', '-o', exe]);
    const raw = executeAll(exe);
    for (const bad of invalid) {
      const r = invoke(exe, ['--threads', '1', ...bad]);
      assert.equal(r.status, 2); assert.match(r.stdout + r.stderr, /invalid board/);
    }
    modes.push({mode: name, rows: fixtures.length, output_fields: fixtures.length * 19,
      invalid_rejections: invalid.length, output_sha256: sha(raw)});
    console.error(`PASS native ${name}: ${fixtures.length} complete Board rows`);
  }
  const mutations = [];
  for (const [name, from, to] of [['noop', '+bit = U64.bit(U32.to_nat(sq))', '+bit = U64.zero()'],
    ['wrong-color', 'U64.or(w, Chess.select_u64(white, bit, U64.zero()))', 'U64.or(w, bit)']]) {
    const e = path.join(temp, name + '-source'); fs.cpSync(engine, e, {recursive: true});
    const p = path.join(e, 'standalone/Position.bend'), source = fs.readFileSync(p, 'utf8');
    assert.equal(source.split(from).length, 2); fs.writeFileSync(p, source.replace(from, to));
    const mc = path.join(temp, name + '.c'), exe = path.join(temp, name + '-mutant');
    run(process.execPath, [path.join(compiler, 'bend2/main.ts'), path.join(e, 'standalone/proofs/board/probe.bend'), '-o', mc]);
    run(cc, ['-std=c11', '-O1', '-ffp-contract=off', mc, '-lm', '-pthread', '-o', exe]);
    const chunk = fixtures.slice(0, 64), raw = run(exe, ['--threads', '1', ...chunk.flatMap(x => x.input)]);
    let diagnostic = '';
    try { compare(raw, chunk, 0); } catch (error) {
      assert.ok(error instanceof assert.AssertionError); diagnostic = error.message;
    }
    assert.match(diagnostic, /row \d+, field \d+: fresh_insert/);
    mutations.push({name, rejected: true, compiled_and_executed: true, diagnostic,
      scope: 'first generic-mode full-board mismatch; other corrupted modes not run'});
  }
  assert.deepEqual(hashes(), before); assert.deepEqual(verifyCompiler(compiler), identity);
  const report = {native_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    rows_per_mode: fixtures.length, cases_by_operation: counts, fresh_square_kind_color_cases: allKinds.size,
    complete_board_fields_per_mode: fixtures.length * 19, malformed_requests_per_mode: invalid.length,
    native_query_square_domain: '0..63; broader source bit-mask invariant does not make out-of-board squares meaningful',
    modes, mutations, source_sha256s: before, fixture_sha256: sha(JSON.stringify(fixtures)),
    oracle: 'Independent 64-square sets of piece kinds/colors; encode all eight bitboards and all three metadata fields. No proof predicates execute in candidate.',
    scope: 'Representation insertion and metadata only, including explicit collision/outside-premise cases; not legal chess positions, FEN reachability or legal moves.',
    cc: run(cc, ['--version']).split('\n')[0], bun: process.versions.bun};
  const output = JSON.stringify(report, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], output); console.log(output.trimEnd());
} finally { fs.rmSync(temp, {recursive: true, force: true}); }
