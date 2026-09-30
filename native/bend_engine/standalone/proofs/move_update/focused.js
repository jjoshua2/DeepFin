// Opt-in representation proofs. No engine, perft, model or training workload.
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
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-move-laws-'));
const names = ['clear_mask_exact_update', 'clear_mask_preserves_representation', 'cleared_target_is_fresh', 'ordinary_make_move_exact_update', 'ordinary_make_move_preserves_representation'];
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
    encoding: 'utf8', timeout: 60000, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  return {status: r.status, output: (r.stdout + r.stderr).trim(), elapsed_seconds: (performance.now() - start) / 1000};
}
function clean(r) { assert.equal(r.status, 0, r.output.slice(-2500)); assert.equal(r.output, 'All terms check.'); }
function copy(name) {
  const e = path.join(temp, name); fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/move_update'), chess: path.join(e, 'legal_probe/Chess.bend')};
}
function replace(f, from, to) {
  const s = text(f); assert.equal(s.split(from).length, 2, 'nonunique mutation'); fs.writeFileSync(f, s.replace(from, to));
}
function reject(name, entry, edit, location) {
  const d = copy(name); edit(d); const r = invoke(path.join(d.s, entry));
  assert.equal(r.status, 1, name + ': must reject');
  assert.match(r.output, /expected[\s\S]*observed/, name + ': not semantic');
  assert.doesNotMatch(r.output, /no such file|RangeError|Maximum call stack|Segmentation fault|more than once|a decreasing self-call|a defined name|a pattern \(a binder/);
  assert.match(r.output, location, name + ': wrong location');
  controls.push({name, rejected: true, kind: 'source semantic/refinement', entry,
    diagnostic_sha256: sha(r.output), diagnostic_bytes: Buffer.byteLength(r.output), excerpt: r.output.slice(-1600)});
  console.error('PASS rejection: ' + name);
}
function guard(name, edit, check, reason) {
  const d = copy(name); edit(d); assert.throws(() => check(d), reason);
  controls.push({name, rejected: true, kind: 'manifest/import policy'});
}
try {
  manifest(suite); const closure = graph(path.join(suite, 'consumer.bend'), engine);
  for (const f of ['focused.js', 'verify.js', 'verify_native.js', 'probe.bend']) closure.add(path.join(suite, f));
  closure.add(path.resolve(suite, '../../toolchain.json'));
  closure.add(path.resolve(suite, '../../verify_compiler.js'));
  const hashes = () => Object.fromEntries([...closure].sort().map(f => [path.relative(engine, f), sha(fs.readFileSync(f))]));
  const before = hashes(); const consumer = invoke(path.join(suite, 'consumer.bend')); clean(consumer);
  console.error('PASS: five move-update laws and importing consumer');
  reject('actual-kernel-ignores-removal', 'ClearActual.bend', d => replace(d.chess,
    'U64.or(U64.and_not(bb, remove), select_u64(put, target, U64.zero()))',
    'U64.or(bb, select_u64(put, target, U64.zero()))'), /Location: scalar\b/);
  reject('actual-kernel-ignores-put-flag', 'ClearActual.bend', d => replace(d.chess,
    'U64.or(U64.and_not(bb, remove), select_u64(put, target, U64.zero()))',
    'U64.or(U64.and_not(bb, remove), target)'), /Location: scalar\b/);
  reject('actual-move-keeps-captured-piece', 'Ordinary.bend', d => replace(d.chess,
    '+remove = U64.or(U64.or(U64.bit(U32.to_nat(src)), U64.bit(U32.to_nat(cap))),\n    U64.or(target, select_u64(castle, U64.bit(U32.to_nat(rook_from)), U64.zero())))',
    '+remove = U64.bit(U32.to_nat(src))'), /Location: unfold\b/);
  reject('actual-move-does-not-toggle-turn', 'Ordinary.bend', d => replace(d.chess,
    'U32.xor(side, 1), U32.and(get_rights(b), U32.not(rights_lost)), ep}',
    'side, U32.and(get_rights(b), U32.not(rights_lost)), ep}'), /Location: unfold\b/);
  reject('actual-move-noop', 'Ordinary.bend', d => {
    const original = text(d.chess), start = original.indexOf('def make_move('), end = original.indexOf('\ndef put_move(', start);
    assert.ok(start >= 0 && end > start);
    fs.writeFileSync(d.chess, original.slice(0,start) + 'def make_move(+b: Board, m: Ply) -> Board:\n  b\n' + original.slice(end));
  }, /Location: unfold\b/);
  reject('model-forgets-black-plane', 'ClearActual.bend', d => replace(path.join(d.s,'Spec.bend'),
    'U64.and_not(bl,mask),turn,rights,ep}', 'bl,turn,rights,ep}'), /Location: .*clear\b/);
  reject('target-not-in-removal-mask', 'ClearActual.bend', d => replace(path.join(d.s,'Spec.bend'),
    'U64.or(U64.or(source,target),U64.or(target,U64.zero()))', 'source'), /Location: .*fresh\b/);
  reject('omit-board-consistency-premise', 'PROOF.bend', d => {
    const f = path.join(d.s,'LAWS.bend'), p = text(f), anchor = 'law ordinary_make_move_preserves_representation:';
    const start = p.indexOf(anchor); assert.ok(start >= 0);
    const tail = p.slice(start); assert.equal(tail.split('{Board.valid(b) == True{} : Bool}').length,2);
    fs.writeFileSync(f,p.slice(0,start)+tail.replace('{Board.valid(b) == True{} : Bool}','{True{} == True{} : Bool}'));
  }, /Location: LAWS\.ordinary_make_move_preserves_representation\b/);
  reject('typed-decoder-mislabels-fallback', 'Selection.bend', d => {
    const f=path.join(d.s,'Selection.bend'), source=text(f);
    const needle='case False{} False{} False{} False{} False{}: B.King{}';
    assert.equal(source.split(needle).length,2);
    fs.writeFileSync(f,source.replace(needle,'case False{} False{} False{} False{} False{}: B.Queen{}'));
  }, /Location: tag\b/);
  guard('missing-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law clear_mask_exact_update:', 'def missing:'), d => manifest(d.s), /./);
  guard('missing-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.clear_mask_exact_update(', 'def missing('), d => manifest(d.s), /./);
  guard('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  guard('missing-consumer-import', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  guard('proof-hole', d => fs.appendFileSync(path.join(d.s, 'Ordinary.bend'), '\n?missing\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /proof hole/);
  guard('foreign-proof', d => fs.appendFileSync(path.join(d.s, 'Ordinary.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /foreign import/);
  guard('unsafe-proof', d => replace(path.join(d.s, 'Ordinary.bend'), 'def actual(', '@unsafe\ndef actual('), d => graph(path.join(d.s,'consumer.bend'),d.e), /unsafe dependency/);
  guard('symlink-proof', d => {const f=path.join(d.s,'Ordinary.bend');fs.renameSync(f,f+'.orig');fs.symlinkSync('Ordinary.bend.orig',f);}, d => graph(path.join(d.s,'consumer.bend'),d.e), /nonregular/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name:'warning-on-zero-exit',rejected:true,kind:'synthetic output-wrapper unit; not compiler execution'});
  assert.equal(controls.length,18); assert.deepEqual(hashes(),before); assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',controls_gate:'PASS',consumer:'PASS',compiler_revision:PIN.revision,...identity,
    new_law_count:5,new_laws:names,inherited_gate_run:false,consumer_seconds:consumer.elapsed_seconds,
    negative_controls:controls,source_sha256s:before,
    scope:'Actual scalar-kernel deletion lift and flag0/promotion0 make_move exact update/partition; no legal-move, special-move or valid-metadata theorem.'};
  const encoded=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],encoded);console.log(encoded.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
