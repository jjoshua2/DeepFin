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
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-promotion-laws-'));
const names = ['promotion_choice_roundtrip', 'promotion_choices_exact', 'promotion_make_move_exact_update', 'promotion_make_move_preserves_representation'];
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
    encoding: 'utf8', timeout: 120000, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, String(r.error)); assert.equal(r.signal, null);
  return {status: r.status, output: (r.stdout + r.stderr).trim(), elapsed_seconds: (performance.now() - start) / 1000};
}
function clean(r) { assert.equal(r.status, 0, r.output.slice(-2500)); assert.equal(r.output, 'All terms check.'); }
function copy(name) {
  const e = path.join(temp, name); fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/promotion'), chess: path.join(e, 'legal_probe/Chess.bend')};
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
  for (const f of ['focused.js', 'verify.js', 'verify_native.js', 'probe.bend', 'choices_probe.bend']) closure.add(path.join(suite, f));
  closure.add(path.resolve(suite, '../../toolchain.json'));
  closure.add(path.resolve(suite, '../../verify_compiler.js'));
  const hashes = () => Object.fromEntries([...closure].sort().map(f => [path.relative(engine, f), sha(fs.readFileSync(f))]));
  const before = hashes(); const consumer = invoke(path.join(suite, 'consumer.bend')); clean(consumer);
  console.error('PASS: four promotion laws and importing consumer');
  reject('actual-move-ignores-promotion','Actual.bend',d=>replace(d.chess,
    '+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)',
    '+put = kind'), /Location: unfold\b/);
  reject('actual-move-forces-queen','Actual.bend',d=>replace(d.chess,
    '+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)',
    '+put = Bool.pick(U32, U32.is_zero(promotion), kind, 4)'), /Location: unfold\b/);
  reject('actual-generator-omits-queen','Encoding.bend',d=>replace(d.chess,
    'Con{Ply{src, dst, 3, 0}, Con{Ply{src, dst, 4, 0}, tail}}}}',
    'Con{Ply{src, dst, 3, 0}, tail}}}'), /Location: generated\b/);
  reject('actual-generator-duplicates-knight','Encoding.bend',d=>replace(d.chess,
    'Con{Ply{src, dst, 1, 0}, Con{Ply{src, dst, 2, 0},',
    'Con{Ply{src, dst, 1, 0}, Con{Ply{src, dst, 1, 0},'), /Location: generated\b/);
  reject('actual-generator-drops-tail','Encoding.bend',d=>replace(d.chess,
    'Con{Ply{src, dst, 4, 0}, tail}}}}',
    'Con{Ply{src, dst, 4, 0}, Nil{}}}}}'), /Location: generated\b/);
  reject('rook-choice-misencoded','Encoding.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'case Rook{}: Board.Rook{}','case Rook{}: Board.Bishop{}'), /Location: roundtrip\b/);
  reject('promotion-does-not-toggle-turn','Actual.bend',d=>replace(d.chess,
    'U32.xor(side, 1), U32.and(get_rights(b), U32.not(rights_lost)), ep}',
    'Bool.pick(U32,U32.is_zero(promotion),U32.xor(side,1),side), U32.and(get_rights(b), U32.not(rights_lost)), ep}'), /Location: unfold\b/);
  reject('omit-consistency-premise','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    'for good: {Board.valid(b) == True{} : Bool}',
    'for good: {True{} == True{} : Bool}'), /Location: LAWS\.promotion_make_move_preserves_representation\b/);
  guard('missing-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law promotion_choice_roundtrip:', 'def missing:'), d => manifest(d.s), /./);
  guard('missing-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.promotion_choice_roundtrip(', 'def missing('), d => manifest(d.s), /./);
  guard('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  guard('missing-consumer-import', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  guard('proof-hole', d => fs.appendFileSync(path.join(d.s, 'Actual.bend'), '\n?missing\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /proof hole/);
  guard('foreign-proof', d => fs.appendFileSync(path.join(d.s, 'Actual.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /foreign import/);
  guard('unsafe-proof', d => replace(path.join(d.s, 'Actual.bend'), 'def exact(', '@unsafe\ndef exact('), d => graph(path.join(d.s,'consumer.bend'),d.e), /unsafe dependency/);
  guard('symlink-proof', d => {const f=path.join(d.s,'Actual.bend');fs.renameSync(f,f+'.orig');fs.symlinkSync('Actual.bend.orig',f);}, d => graph(path.join(d.s,'consumer.bend'),d.e), /nonregular/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name:'warning-on-zero-exit',rejected:true,kind:'synthetic output-wrapper unit; not compiler execution'});
  assert.equal(controls.length,17); assert.deepEqual(hashes(),before); assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',controls_gate:'PASS',consumer:'PASS',compiler_revision:PIN.revision,...identity,
    new_law_count:4,new_laws:names,inherited_gate_run:false,consumer_seconds:consumer.elapsed_seconds,
    negative_controls:controls,source_sha256s:before,
    scope:'Typed knight/bishop/rook/queen promotions: exact generator list and complete flag-zero make_move update/partition. No legal-move, EP/castling, or independently valid-metadata theorem.'};
  const encoded=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],encoded);console.log(encoded.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
