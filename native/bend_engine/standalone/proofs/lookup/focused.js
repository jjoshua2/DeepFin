// Opt-in source gate; inherited results are not implied by a focused pass.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {PIN, verifyCompiler} from '../../verify_compiler.js';

const args = process.argv.slice(2);
const controlsOnly = args.includes('--controls-only');
const filtered = args.filter(a => a !== '--controls-only');
assert.ok(filtered.length === 1 || (filtered.length === 3 && filtered[1] === '--report'),
  'usage: focused.js COMPILER [--controls-only] [--report FILE]');
const compiler = path.resolve(filtered[0]);
const identity = verifyCompiler(compiler);
const engine = path.resolve(import.meta.dirname, '../../..');
const root = path.join(engine, 'standalone');
const suite = import.meta.dirname;
const cli = path.join(compiler, 'bend2/main.ts');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-lookup-laws-'));
const names = ['certified_header_route', 'selected_state_is_masked',
  'initialized_indexed_lookup', 'initialized_masked_lookup'];
const controls = [];
const sha = b => createHash('sha256').update(b).digest('hex');
const code = f => fs.readFileSync(f, 'utf8').split('\n').map(x => x.split('#')[0]).join('\n');
function invoke(file) {
  const r = spawnSync(process.execPath, [cli, file], {
    encoding: 'utf8', timeout: 900000, maxBuffer: 64 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'},
  });
  assert.equal(r.error, undefined, String(r.error));
  assert.equal(r.signal, null, 'a terminated checker is not a result');
  return {status: r.status, text: (r.stdout + r.stderr).trim()};
}
function clean(r) {
  assert.equal(r.status, 0, r.text.slice(-2000));
  assert.equal(r.text, 'All terms check.', 'warnings or extra output do not pass');
}
function manifest(s) {
  const laws = code(path.join(s, 'LAWS.bend'));
  const proof = code(path.join(s, 'PROOF.bend'));
  const consumer = code(path.join(s, 'consumer.bend'));
  assert.deepEqual([...laws.matchAll(/^law (\w+):/gm)].map(m => m[1]), names, 'law manifest');
  assert.deepEqual([...proof.matchAll(/^def Laws\.(\w+)\(/gm)].map(m => m[1]), names, 'proof manifest');
  assert.match(proof, /^import \.\/LAWS\.bend as Laws$/m, 'required laws import');
  assert.match(consumer, /^import \.\/PROOF\.bend as Proof$/m, 'required consumer proof');
  for (const name of names) assert.ok(consumer.includes('Laws.' + name + '('), 'consumer omits ' + name);
}
function policy(file, boundary, seen = new Set()) {
  file = path.resolve(file);
  const relative = path.relative(boundary, file);
  assert.ok(!relative.split(path.sep).includes('..') && !path.isAbsolute(relative), 'escaped source boundary');
  assert.ok(fs.lstatSync(file).isFile(), 'nonregular proof input');
  assert.equal(fs.realpathSync(file), file, 'symlinked proof dependency');
  if (seen.has(file)) return seen;
  seen.add(file);
  const text = code(file);
  assert.doesNotMatch(text, /@unsafe|\?/, 'unsafe dependency or proof hole');
  for (const match of text.matchAll(/^\s*import\s+(\S+)/gm)) {
    if (match[1] === 'Base') continue;
    assert.match(match[1], /^\.{1,2}\/[A-Za-z0-9_/.]+\.bend$/, 'foreign import');
    policy(path.resolve(path.dirname(file), match[1]), boundary, seen);
  }
  return seen;
}
function replace(file, from, to) {
  const text = fs.readFileSync(file, 'utf8');
  assert.equal(text.split(from).length, 2, 'mutation must match exactly once: ' + from);
  fs.writeFileSync(file, text.replace(from, to));
}
function copy(name) {
  const e = path.join(tmp, name);
  fs.cpSync(engine, e, {recursive: true});
  return {e, s: path.join(e, 'standalone/proofs/lookup'), chess: path.join(e, 'legal_probe/Chess.bend')};
}
function reject(name, entry, edit, location, kind = 'actual lookup routing') {
  const d = copy(name); edit(d);
  const r = invoke(path.join(d.s, entry));
  assert.equal(r.status, 1, name + ': not a source rejection');
  assert.match(r.text, /expected[\s\S]*observed/, name + ': not a semantic diagnostic');
  assert.doesNotMatch(r.text, /no such file|a defined name|more than once|a decreasing self-call|RangeError|Maximum call stack|Segmentation fault/,
    name + ': malformed fixture or crash is not proof rejection');
  assert.ok(location.test(r.text), name + ': unexpected location\n' + r.text.slice(-1500));
  controls.push({name, rejected: true, kind, entry, diagnostic_sha256: sha(r.text),
    diagnostic_bytes: Buffer.byteLength(r.text), excerpt: r.text.slice(0,160) + '\n...\n' + r.text.slice(-900)});
  console.error('PASS rejection: ' + name);
}
function guard(name, edit, check, reason) {
  const d = copy(name); edit(d);
  assert.throws(() => check(d), reason, name);
  controls.push({name, rejected: true, kind: 'manifest/import policy'});
}
try {
  manifest(suite);
  const graph = policy(path.join(suite, 'consumer.bend'), engine);
  for (const name of ['focused.js', 'verify.js', 'verify_native.js', 'probe.bend']) graph.add(path.join(suite,name));
  const hashes = () => Object.fromEntries([...graph].sort().map(f => [path.relative(engine,f),sha(fs.readFileSync(f))]));
  const before = hashes();
  if (!controlsOnly) { clean(invoke(path.join(suite,'consumer.bend'))); console.error('PASS: four public lookup laws and importing consumer'); }
  reject('actual-offset-plus-one','Route.bend', d => replace(d.chess,
    'Sliders.lookup(table, U64.low(offset), index)', 'Sliders.lookup(table, U32.inc(U64.low(offset)), index)'), /Location: model\b/);
  reject('actual-wrong-mask-slot','Route.bend', d => replace(d.chess,
    'slide_mask(sq, occ, Array.get(U64, table, sq))', 'slide_mask(sq, occ, Array.get(U64, table, U32.inc(sq)))'), /Location: model\b/);
  reject('actual-wrong-prefix-slot','Route.bend', d => replace(d.chess,
    'Array.get(U64, table, U32.add(128, sq))', 'Array.get(U64, table, U32.add(127, sq))'), /Location: model\b/);
  reject('actual-swapped-extraction','Route.bend', d => replace(d.chess,
    'slide_offset(Sliders.pext_index(occ, mask),', 'slide_offset(Sliders.pext_index(mask, occ),'), /Location: model\b/);
  reject('actual-discarded-index','Route.bend', d => replace(d.chess,
    'Sliders.lookup(table, U64.low(offset), index)', 'Sliders.lookup(table, U64.low(offset), 0)'), /Location: model\b/);
  reject('unmasked-state-claim','Canonical.bend', d => replace(path.join(d.s,'Canonical.bend'),
    '== U64.and(occ,Layout.mask(key)) : U64}:\n', '== occ : U64}:\n'), /Location: state\b/, 'derived state contract');
  reject('public-return-original-array','PROOF.bend', d => replace(path.join(d.s,'LAWS.bend'),
    '(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Masked.expected(Nat.add(before,k),occ))',
    '(Array.new(U64,d,seed),Masked.expected(Nat.add(before,k),occ))'), /Location: LAWS\.initialized_masked_lookup\b/, 'public complete-buffer contract');
  guard('missing-law', d => replace(path.join(d.s,'LAWS.bend'),'law initialized_masked_lookup:','def omitted:'), d => manifest(d.s), /law manifest/);
  guard('missing-proof', d => replace(path.join(d.s,'PROOF.bend'),'def Laws.initialized_masked_lookup(','def omitted('), d => manifest(d.s), /proof manifest/);
  guard('missing-laws-import', d => replace(path.join(d.s,'PROOF.bend'),'import ./LAWS.bend as Laws\n',''), d => manifest(d.s), /required laws import/);
  guard('missing-consumer-import', d => replace(path.join(d.s,'consumer.bend'),'import ./PROOF.bend as Proof\n',''), d => manifest(d.s), /required consumer proof/);
  guard('proof-hole', d => fs.appendFileSync(path.join(d.s,'Route.bend'),'\n?hole\n'), d => policy(path.join(d.s,'consumer.bend'),d.e), /proof hole/);
  guard('foreign-proof', d => fs.appendFileSync(path.join(d.s,'Route.bend'),'\nimport "./oracle.c"\n'), d => policy(path.join(d.s,'consumer.bend'),d.e), /foreign import/);
  guard('symlinked-proof', d => {const f=path.join(d.s,'Route.bend');fs.renameSync(f,f+'.original');fs.symlinkSync('Route.bend.original',f);},
    d => policy(path.join(d.s,'consumer.bend'),d.e), /nonregular/);
  guard('unsafe-proof', d => replace(path.join(d.s,'Route.bend'),'def actual(','@unsafe\ndef actual('), d => policy(path.join(d.s,'consumer.bend'),d.e), /unsafe dependency/);
  assert.throws(() => clean({status:0,text:'All terms check.\nWARNING: unsafe or foreign dependency'}));
  controls.push({name:'zero-with-warning',rejected:true,kind:'exact-output wrapper unit test, not a new compiler run'});
  assert.equal(controls.length,16);
  assert.deepEqual(hashes(),before); assert.deepEqual(verifyCompiler(compiler),identity);
  const report = {focused_gate:controlsOnly?'NOT_RUN':'PASS',controls_gate:'PASS',consumer:controlsOnly?'NOT_RUN':'PASS',
    compiler_revision:PIN.revision,...identity,new_law_count:4,new_laws:names,negative_controls:controls,
    inherited_gate_run:false,source_sha256s:before,
    scope:'Promoted source-to-source actual lookup contracts; independent mask/blocker-ray geometry remains separate.'};
  const text=JSON.stringify(report,null,2)+'\n';if(filtered[2])fs.writeFileSync(filtered[2],text);console.log(text.trimEnd());
} finally { fs.rmSync(tmp,{recursive:true,force:true}); }
