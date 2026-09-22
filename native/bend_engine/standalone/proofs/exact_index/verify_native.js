// Opt-in recurrence-to-actual-index check. No engine/perft/model work.
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
const cc = process.env.CC || 'clang';
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'deepfin-exact-index-native-'));
const sha = b => createHash('sha256').update(b).digest('hex');
const files = ['standalone/Tables.bend', 'standalone/Subsets.bend', 'standalone/Text.bend',
  'standalone/proofs/layout/Spec.bend', 'bitboard_probe/Sliders.bend',
  'standalone/proofs/exact_index/probe.bend', 'standalone/proofs/exact_index/verify_native.js'];
const snapshot = () => Object.fromEntries(files.map(f => [f, sha(fs.readFileSync(path.join(engine, f)))]));
const before = snapshot();
function invoke(command, argv, timeout = 120000) {
  const r = spawnSync(command, argv, {encoding: 'utf8', timeout, maxBuffer: 32 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: '1'}});
  assert.equal(r.error, undefined, `${command}: ${r.error}`);
  assert.equal(r.signal, null, 'unexpected signal');
  return r;
}
function run(command, argv) {
  const r = invoke(command, argv);
  assert.equal(r.status, 0, r.stderr + r.stdout.slice(0, 1000));
  assert.equal(r.stderr, '', 'unexpected native diagnostics');
  return r.stdout;
}
// Independent signed-coordinate geometry and compact-bit scatter. No recurrence.
function bitsFor(key) {
  const square = key % 64, x = square % 8, y = Math.floor(square / 8);
  const dirs = key < 64 ? [[1,0],[-1,0],[0,1],[0,-1]] : [[1,1],[1,-1],[-1,1],[-1,-1]];
  const squares = new Set();
  for (const [dx,dy] of dirs) {
    const ray = [];
    for (let f=x+dx, r=y+dy; f>=0 && f<8 && r>=0 && r<8; f+=dx, r+=dy) ray.push(r*8+f);
    for (const sq of ray.slice(0,-1)) squares.add(sq);
  }
  return [...squares].sort((a,b)=>a-b).map(sq=>1n<<BigInt(sq));
}
function deposit(index, bits) {
  return bits.reduce((value, bit, i)=>value | ((BigInt(index)>>BigInt(i))&1n ? bit : 0n), 0n);
}
function compare(actual, expected) {
  const a=actual.trimEnd().split('\n'), b=expected.trimEnd().split('\n');
  assert.equal(a.length,b.length,'row count');
  for(let i=0;i<a.length;i++) assert.equal(a[i],b[i],`row ${i}`);
}
try {
  const expectedRows=[]; let states=0;
  for(let key=0;key<128;key++) {
    const bits=bitsFor(key), size=2**bits.length;
    for(let i=0;i<size;i++) {
      const state=deposit(i,bits);
      expectedRows.push(['r',key,i,String(state>>32n),String(state&0xffffffffn),i,i,0,i].join(' '));
    }
    expectedRows.push(`e ${key} ${size} 0 0`);
    states+=size;
  }
  assert.equal(states,107648);
  const expected=expectedRows.join('\n')+'\n';
  assert.throws(()=>compare(expected.replace('r 0 0 ','r 0 1 '),expected),/row 0/);
  const cfile=path.join(temp,'probe.c');
  run(process.execPath,[path.join(compiler,'bend2/main.ts'),path.join(root,'proofs/exact_index/probe.bend'),'-o',cfile]);
  const reports=[];
  for(const [mode,flags] of [['generic',[]],['portable',['-DBEND_U64_PORTABLE']],
    ['native',['-march=native']],['ubsan',['-fsanitize=undefined','-fno-sanitize-recover=all']]]) {
    const exe=path.join(temp,mode);
    run(cc,['-std=c11','-O1','-ffp-contract=off','-Werror=shift-count-overflow',...flags,cfile,'-pthread','-lm','-o',exe]);
    const actual=run(exe,['--threads','1',...Array.from({length:128},(_,i)=>String(i))]);
    compare(actual,expected);
    for(const bad of [['128'],['4294967295'],['4294967296'],['-1'],['junk']]) {
      const r=invoke(exe,['--threads','1',...bad]);
      assert.equal(r.status,2); assert.match(r.stdout+r.stderr,/invalid key|invalid exact-index/);
    }
    // Budget overflow is checked using the cheapest valid key (32 states), not rook corners.
    const budget=invoke(exe,['--threads','1',...Array(129).fill('73')]);
    assert.equal(budget.status,2); assert.match(budget.stdout+budget.stderr,/exact-index budget/);
    reports.push({mode,keys:128,enumerated_states:states,indexed_occupancies:2*states,
      cycle_endpoints:128,invalid_requests:6,output_sha256:sha(actual)});
  }
  assert.deepEqual(snapshot(),before); assert.deepEqual(verifyCompiler(compiler),identity);
  const result={native_gate:'PASS',compiler_revision:PIN.revision,...identity,source_sha256s:before,
    keys:128,enumerated_states:states,indexed_occupancies:2*states,cycle_endpoints:128,
    modes:reports,cc:run(cc,['--version']).split('\n')[0],bun:process.versions.bun,
    oracle:'Independent signed-coordinate masks and direct BigInt compact-bit scatter; no carry-rippler',
    scope:'Actual scalar index and recurrence for all chess-mask subsets, plus all-irrelevant-bits-set contrast; NOT prefix arithmetic, Array storage, attack values or full-engine qualification'};
  const text=JSON.stringify(result,null,2)+'\n';
  if(args[2]) fs.writeFileSync(args[2],text);
  console.log(text.trimEnd());
} finally {fs.rmSync(temp,{recursive:true,force:true});}
