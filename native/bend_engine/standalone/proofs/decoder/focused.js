// Opt-in occupied-square decoding; unchanged parent chain is retained in verify.js.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';
const args = process.argv.slice(2);
assert.ok(args.length === 1 || (args.length === 3 && args[1] === '--report'),
  'usage: focused.js COMPILER [--report FILE]');
const compiler = path.resolve(args[0]), identity = verifyCompiler(compiler);
const suite = import.meta.dirname, engine = path.resolve(suite, '../../..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-decoder-laws-'));
const names = ['square_projection_valid', 'abstract_square_roundtrip', 'guarded_decoder_matches_square', 'occupied_decoder_matches_square'];
const sha = x => createHash('sha256').update(x).digest('hex');
const text = p => fs.readFileSync(p, 'utf8');
const code = p => text(p).split('\n').map(l => l.split('#')[0]).join('\n');
const controls = [];
function graph(f, boundary, seen = new Set()) {
  f = path.resolve(f);
  const rel = path.relative(boundary, f);
  assert.ok(!rel.split(path.sep).includes('..') &&
    (rel.startsWith('standalone/') || rel === 'legal_probe/Chess.bend' || rel === 'bitboard_probe/Sliders.bend'), 'escaped scope');
  assert.ok(fs.lstatSync(f).isFile(), 'nonregular proof input');
  assert.equal(fs.realpathSync(f), f, 'symlinked proof input');
  if (seen.has(f)) return seen;
  seen.add(f);
  const source = code(f);
  assert.doesNotMatch(source, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const m of source.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (m[1] === 'Base') continue;
    assert.match(m[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign import');
    graph(path.resolve(path.dirname(f), m[1]), boundary, seen);
  }
  return seen;
}
function manifest(s) {
  assert.deepEqual([...code(path.join(s, 'LAWS.bend')).matchAll(/^law (\w+):/gm)].map(m => m[1]), names);
  const p = code(path.join(s, 'PROOF.bend'));
  assert.deepEqual([...p.matchAll(/^def Laws\.(\w+)\(/gm)].map(m => m[1]), names);
  assert.match(p, /import \.\/LAWS\.bend as Laws/);
  assert.match(code(path.join(s, 'consumer.bend')), /import \.\/PROOF\.bend as Proof/);
}
function invoke(f) {
  const start = performance.now();
  const r = spawnSync(process.execPath, [path.join(compiler, 'bend2/main.ts'), f], {
    encoding: 'utf8', timeout: 90000, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  return {status: r.status, output: (r.stdout + r.stderr).trim(), elapsed_seconds: (performance.now() - start) / 1000};
}
function clean(r) { assert.equal(r.status, 0, r.output.slice(-2500)); assert.equal(r.output, 'All terms check.'); }
function copy(name) {
  const e = path.join(temp, name); fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/decoder'), chess: path.join(e, 'legal_probe/Chess.bend')};
}
function replace(f, from, to) {
  const s = text(f); assert.equal(s.split(from).length, 2, 'nonunique mutation'); fs.writeFileSync(f, s.replace(from, to));
}
function reject(name, entry, edit, location) {
  console.error('START semantic: ' + name); const d = copy(name); edit(d); const r = invoke(path.join(d.s, entry));
  assert.equal(r.status, 1, name + ': must reject');
  assert.match(r.output, /expected[\s\S]*observed/, name + ': not semantic');
  assert.doesNotMatch(r.output, /no such file|RangeError|Maximum call stack|Segmentation fault|more than once|a decreasing self-call/);
  assert.match(r.output, location, name + ': wrong location');
  controls.push({name, rejected: true, kind: 'source semantic/refinement', entry,
    diagnostic_sha256: sha(r.output), diagnostic_bytes: Buffer.byteLength(r.output), excerpt: r.output.slice(-1600)});
  console.error('PASS rejection: ' + name + ' in ' + r.elapsed_seconds.toFixed(3) + 's');
}
function guard(name, edit, check, reason) {
  const d = copy(name); edit(d); assert.throws(() => check(d), reason);
  controls.push({name, rejected: true, kind: 'manifest/import policy'});
}
try {
  manifest(suite); const closure = graph(path.join(suite, 'consumer.bend'), engine);
  for (const f of ['focused.js', 'verify.js', 'verify_native.js', 'probe.bend']) closure.add(path.join(suite, f));
  const hashes = () => Object.fromEntries([...closure].sort().map(f => [path.relative(engine, f), sha(fs.readFileSync(f))]));
  const before = hashes(); const consumer = invoke(path.join(suite, 'consumer.bend')); clean(consumer);
  console.error('PASS: four square-projection/decoder laws and importing consumer');
  reject('actual-pawn-tag', 'Actual.bend', d => replace(d.chess,
    'U64.test_bit(get_pawns(b), U32.to_nat(sq)), 0,', 'U64.test_bit(get_pawns(b), U32.to_nat(sq)), 1,'), /Location: bridge\b/);
  reject('actual-king-fallback', 'Actual.bend', d => replace(d.chess,
    'U64.test_bit(get_queens(b), U32.to_nat(sq)), 4, 5)', 'U64.test_bit(get_queens(b), U32.to_nat(sq)), 4, 4)'), /Location: bridge\b/);
  reject('actual-bishop-skipped', 'Actual.bend', d => replace(d.chess,
    'U64.test_bit(get_bishops(b), U32.to_nat(sq)), 2,', 'False{}, 2,'), /Location: bridge\b/);
  reject('actual-occupied-intersection', 'Actual.bend', d => replace(d.chess,
    'U64.or(get_white(b), get_black(b))', 'U64.and(get_white(b), get_black(b))'), /Location: bridge\b/);
  reject('actual-white-plane-swap', 'Projection.bend', d => replace(d.chess,
    'def get_white(b: Board) -> U64:\n  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b\n  white',
    'def get_white(b: Board) -> U64:\n  Board{pawns, knights, bishops, rooks, queens, kings, white, black, turn, rights, ep} = b\n  black'), /Location: structural\b/);
  reject('incorrect-independent-classification', 'Rows.bend', d => replace(path.join(d.s,'Spec.bend'),
    'case B.Row{True{},False{},False{},False{},False{},False{},True{},False{}}: Occupant{0,True{}}',
    'case B.Row{True{},False{},False{},False{},False{},False{},True{},False{}}: Occupant{1,True{}}'), /Location: checked\b/);
  reject('lost-king-plane', 'Rows.bend', d => replace(path.join(d.s,'Spec.bend'),
    'U32.is_eq(kind,5),white,Bool.not(white)', 'False{},white,Bool.not(white)'), /Location: checked\b/);
  reject('empty-square-raw-decoding-rejected', 'consumer.bend', d => fs.appendFileSync(path.join(d.s,'consumer.bend'), `

def negative_empty_decode() ->
  {S.square(Position.empty(),0) == S.Occupant{Chess.piece(Position.empty(),0),False{}} : S.Cell}:
  Laws.occupied_decoder_matches_square(Position.empty(),0,{==},{==},{==})
`), /Location: negative_empty_decode\b/);
  reject('inconsistent-board-excluded', 'consumer.bend', d => fs.appendFileSync(path.join(d.s,'consumer.bend'), `

def negative_collision() ->
  {B.row_valid(S.row(Position.put(0,True{},0,Position.put(1,False{},0,Position.empty())),0)) == True{} : Bool}:
  Laws.square_projection_valid(Position.put(0,True{},0,Position.put(1,False{},0,Position.empty())),0,{==},{==})
`), /Location: negative_collision\b/);
  guard('missing-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law square_projection_valid:', 'def missing:'), d => manifest(d.s), /./);
  guard('missing-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.square_projection_valid(', 'def missing('), d => manifest(d.s), /./);
  guard('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  guard('missing-consumer-proof', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  guard('hole', d => fs.appendFileSync(path.join(d.s, 'Actual.bend'), '\n?hole\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /proof hole/);
  guard('foreign-import', d => fs.appendFileSync(path.join(d.s, 'Actual.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /foreign import/);
  guard('symlink', d => { const p = path.join(d.s, 'Actual.bend'); fs.renameSync(p, p + '.old'); fs.symlinkSync('Actual.bend.old', p); }, d => graph(path.join(d.s, 'consumer.bend'), d.e), /nonregular/);
  guard('unsafe', d => replace(path.join(d.s, 'Actual.bend'), 'def decode(', '@unsafe\ndef decode('), d => graph(path.join(d.s, 'consumer.bend'), d.e), /unsafe dependency/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name: 'unsafe-warning-zero-exit', rejected: true, kind: 'synthetic exact-output unit, not compiler execution'});
  assert.equal(controls.length, 18); assert.deepEqual(hashes(), before); assert.deepEqual(verifyCompiler(compiler), identity);
  const result = {focused_gate: 'PASS', controls_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_law_count: 4, universal_laws: 4, closed_laws: 0, new_laws: names, inherited_gate_run: false,
    consumer: 'PASS', consumer_seconds: consumer.elapsed_seconds, negative_controls: controls, source_sha256s: before,
    scope: 'Actual occupied decoding and pointwise abstract-square correspondence under global partition and sq<64. No abstract-board bijection, parser freshness, metadata legality or move correctness.'};
  const output = JSON.stringify(result, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], output); console.log(output.trimEnd());
} finally { fs.rmSync(temp, {recursive: true, force: true}); }
