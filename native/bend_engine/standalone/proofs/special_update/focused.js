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
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-special-laws-'));
const names = ["typed_promotion_exact_update", "typed_promotion_preserves_representation", "en_passant_exact_update", "en_passant_preserves_representation"];
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
  return {e, s: path.join(e, 'standalone/proofs/special_update'), chess: path.join(e, 'legal_probe/Chess.bend')};
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
  console.error('PASS: four special-update laws and importing consumer');
  reject('actual-promotion-ignored','Promotion.bend',d=>replace(d.chess,
    '+put = Bool.pick(U32, U32.is_zero(promotion), kind, promotion)', '+put = kind'), /Location: unfold\b/);
  reject('actual-en-passant-victim-is-destination','EnPassant.bend',d=>replace(d.chess,
    'cap = Bool.pick(U32, U32.is_eq(flag, 1), U32.xor(dst, 8), dst)', 'cap = dst'), /Location: unfold\b/);
  reject('model-omits-en-passant-source-removal','EnPassant.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'U64.or(U64.or(target(src),target(U32.xor(dst,8))),U64.or(target(dst),U64.zero()))',
    'U64.or(target(U32.xor(dst,8)),U64.or(target(dst),U64.zero()))'), /Location: unfold\b/);
  reject('omit-promotion-consistency','PROOF.bend',d=>{
    const f=path.join(d.s,'LAWS.bend'),p=text(f);const start=p.indexOf('law typed_promotion_preserves_representation:');
    const end=p.indexOf('law en_passant_exact_update:',start);assert.ok(start>=0&&end>start);
    fs.writeFileSync(f,p.slice(0,start)+p.slice(start,end).replace('{Board.valid(b) == True{} : Bool}','{True{} == True{} : Bool}')+p.slice(end));
  }, /Location: LAWS\.typed_promotion_preserves_representation\b/);
  reject('return-original-board-for-promotion','PROOF.bend',d=>replace(path.join(d.s,'LAWS.bend'),
    '== Spec.promotion_expected(b,src,dst,p) : Chess.Board}', '== b : Chess.Board}'), /Location: LAWS\.typed_promotion_exact_update\b/);
  reject('typed-knight-label-drift','consumer.bend',d=>replace(path.join(d.s,'Spec.bend'),
    'case Knight{}: B.Knight{}', 'case Knight{}: B.Bishop{}'), /Location: tags\b/);
  reject('fresh-target-without-target-clear','Fresh.bend',d=>replace(path.join(d.s,'Fresh.bend'),
    'M.erase(board,U64.or(source,U64.or(dest,U64.zero())))', 'M.erase(board,source)'), /Location: target\b/);
  reject('unrestricted-promotion-claim','consumer.bend',d=>replace(path.join(d.s,'consumer.bend'),
    '{B.valid(Chess.make_move(promotion_board(),Chess.Ply{48,57,6,0})) == False{} : Bool}',
    '{B.valid(Chess.make_move(promotion_board(),Chess.Ply{48,57,6,0})) == True{} : Bool}'), /Location: invalid_promotion_is_not_covered\b/);
  guard('missing-law', d => replace(path.join(d.s, 'LAWS.bend'), 'law typed_promotion_exact_update:', 'def missing:'), d => manifest(d.s), /./);
  guard('missing-proof', d => replace(path.join(d.s, 'PROOF.bend'), 'def Laws.typed_promotion_exact_update(', 'def missing('), d => manifest(d.s), /./);
  guard('missing-laws-import', d => replace(path.join(d.s, 'PROOF.bend'), 'import ./LAWS.bend as Laws\n', ''), d => manifest(d.s), /./);
  guard('missing-consumer-import', d => replace(path.join(d.s, 'consumer.bend'), 'import ./PROOF.bend as Proof\n', ''), d => manifest(d.s), /./);
  guard('proof-hole', d => fs.appendFileSync(path.join(d.s, 'EnPassant.bend'), '\n?missing\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /proof hole/);
  guard('foreign-proof', d => fs.appendFileSync(path.join(d.s, 'EnPassant.bend'), '\nimport "./oracle.c"\n'), d => graph(path.join(d.s,'consumer.bend'),d.e), /foreign import/);
  guard('unsafe-proof', d => replace(path.join(d.s, 'EnPassant.bend'), 'def actual(', '@unsafe\ndef actual('), d => graph(path.join(d.s,'consumer.bend'),d.e), /unsafe dependency/);
  guard('symlink-proof', d => {const f=path.join(d.s,'EnPassant.bend');fs.renameSync(f,f+'.orig');fs.symlinkSync('EnPassant.bend.orig',f);}, d => graph(path.join(d.s,'consumer.bend'),d.e), /nonregular/);
  assert.throws(() => clean({status: 0, output: 'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name:'warning-on-zero-exit',rejected:true,kind:'synthetic output-wrapper unit; not compiler execution'});
  assert.equal(controls.length,17); assert.deepEqual(hashes(),before); assert.deepEqual(verifyCompiler(compiler),identity);
  const report={focused_gate:'PASS',controls_gate:'PASS',consumer:'PASS',compiler_revision:PIN.revision,...identity,
    new_law_count:4,new_laws:names,inherited_gate_run:false,consumer_seconds:consumer.elapsed_seconds,
    negative_controls:controls,source_sha256s:before,
    scope:'Actual typed promotion with flag0 and flag1/promotion0 updates; exact full Boards and representation preservation, not legal moves or castling.'};
  const encoded=JSON.stringify(report,null,2)+'\n';if(args[2])fs.writeFileSync(args[2],encoded);console.log(encoded.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
