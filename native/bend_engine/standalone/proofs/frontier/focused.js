// Opt-in parser frontier; actual initialized paths, not merely a true input flag.
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
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-parser-frontier-laws-'));
const names = ['layout_preserves_frontier', 'accepted_prefix_has_frontier', 'live_typed_insertion_is_fresh', 'accepted_placement_has_consistent_board'];
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
    encoding: 'utf8', timeout: 180000, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  return {status: r.status, output: (r.stdout + r.stderr).trim(), elapsed_seconds: (performance.now() - start) / 1000};
}
function clean(r) { assert.equal(r.status, 0, r.output.slice(-2500)); assert.equal(r.output, 'All terms check.'); }
function copy(name) {
  const e = path.join(temp, name); fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/frontier'), position: path.join(e, 'standalone/Position.bend')};
}
function replace(f, from, to) {
  const s = text(f); assert.equal(s.split(from).length, 2, 'nonunique mutation'); fs.writeFileSync(f, s.replace(from, to));
}
function reject(name, entry, edit, location) {
  console.error('START semantic: ' + name); const d = copy(name); edit(d); const r = invoke(path.join(d.s, entry));
  assert.equal(r.status, 1, name + ': must reject');
  assert.match(r.output, /expected[\s\S]*observed/, name + ': not semantic');
  assert.doesNotMatch(r.output, /a defined name|no such file|RangeError|Maximum call stack|Segmentation fault|more than once|a decreasing self-call/);
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
  console.error('PASS: four initialized parser frontier contracts and importing consumer');
  reject('current-square-omitted-from-frontier', 'Facts.bend', d => replace(path.join(d.s,'Spec.bend'),
    'Nat.is_le(file,at)', 'Nat.is_lt(file,at)'), /Location: piece_nat\b/);
  reject('lower-ranks-omitted-from-frontier', 'Facts.bend', d => replace(path.join(d.s,'Spec.bend'),
    'Nat.is_lt(r,rank)', 'False{}'), /Location: slash_nat\b/);
  reject('inclusive-file-eight-insertion', 'Facts.bend', d => {
    const p=path.join(d.s,'Facts.bend');
    replace(p,'fit: {U32.is_lt(U32.from_nat(f),8) == True{} : Bool}',
      'fit: {U32.is_le(U32.from_nat(f),8) == True{} : Bool}');
  }, /Location: piece_nat\b/);
  reject('assume-no-freshness', 'Fresh.bend', d => replace(path.join(d.s,'Fresh.bend'),
    'old: {Bool.not(Bool.and(t,Bool.or(w,b))) == True{} : Bool}', 'old: {True{} == True{} : Bool}'), /Location: narrow_row\b/);
  reject('actual-piece-reuses-file', '../parser/Typed.bend', d => replace(d.position,
    'b), U32.inc(file), rank,', 'b), file, rank,'), /Location: step\b/);
  reject('actual-slash-reuses-rank', 'Step.bend', d => replace(d.position,
    'Layout{b, 0, U32.sub(rank, 1),', 'Layout{b, 0, rank,'), /Location: .*bits\b/);
  reject('initial-board-not-empty', 'Traversal.bend', d => replace(path.join(d.s,'Spec.bend'),
    'P.Layout{P.empty(),0,7,True{}}','P.Layout{P.put(0,True{},56,P.empty()),0,7,True{}}'), /Location: initialized\b/);
  reject('unconditional-prefix-frontier', 'PROOF.bend', d => replace(path.join(d.s,'LAWS.bend'),
    'for accepted: {Parser.accepted(Position.layout_result(Position.layout(String.append(prefix,suffix),Spec.initial()))) == True{} : Bool}',
    'for accepted: {True{} == True{} : Bool}'), /Location: LAWS\.accepted_prefix_has_frontier\b/);
  guard('missing-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law layout_preserves_frontier:', 'def missing:'), d => manifest(d.s), /./);
  guard('missing-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.layout_preserves_frontier(', 'def missing('), d => manifest(d.s), /./);
  guard('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  guard('missing-consumer-proof', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  guard('hole', d => fs.appendFileSync(path.join(d.s, 'Traversal.bend'), '\n?hole\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /proof hole/);
  guard('foreign-import', d => fs.appendFileSync(path.join(d.s, 'Traversal.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s, 'consumer.bend'), d.e), /foreign import/);
  guard('symlink', d => { const p = path.join(d.s, 'Traversal.bend'); fs.renameSync(p, p + '.old'); fs.symlinkSync('Traversal.bend.old', p); }, d => graph(path.join(d.s, 'consumer.bend'), d.e), /nonregular/);
  guard('unsafe', d => replace(path.join(d.s, 'Traversal.bend'), 'def all(', '@unsafe\ndef all('), d => graph(path.join(d.s, 'consumer.bend'), d.e), /unsafe dependency/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name: 'unsafe-warning-zero-exit', rejected: true, kind: 'synthetic exact-output unit, not compiler execution'});
  assert.equal(controls.length, 17); assert.deepEqual(hashes(), before); assert.deepEqual(verifyCompiler(compiler), identity);
  const result = {focused_gate: 'PASS', controls_gate: 'PASS', compiler_revision: PIN.revision, ...identity,
    new_law_count: 4, universal_laws: 4, closed_laws: 0, new_laws: names, inherited_gate_run: false,
    consumer: 'PASS', consumer_seconds: consumer.elapsed_seconds, negative_controls: controls, source_sha256s: before,
    scope: 'Accepted actual initialized placement paths have bounded consistent empty-ahead frontiers and fresh typed insertion addresses. The actual optional placement result is consistent when present; not six-field FEN semantics, rollback, metadata legality or reachability.'};
  const output = JSON.stringify(result, null, 2) + '\n'; if (args[2]) fs.writeFileSync(args[2], output); console.log(output.trimEnd());
} finally { fs.rmSync(temp, {recursive: true, force: true}); }
