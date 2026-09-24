// Fail-closed source linkage, not a substitute for a discharged source equality.
import assert from 'node:assert/strict';
import fs from 'node:fs';

function tokens(text) {
  const token = /\s+|#[^\n]*|[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*|[0-9]+n?|->|[()<>:,+]/gy;
  const result = [];
  let at = 0;
  while (at < text.length) {
    token.lastIndex = at;
    const match = token.exec(text);
    assert.ok(match, `unsupported recipe syntax at byte ${at}`);
    if (!/^\s|^#/.test(match[0])) result.push(match[0]);
    at = token.lastIndex;
  }
  return result;
}

function definition(text, name) {
  const lines = text.split('\n');
  const indices = lines.flatMap((line, index) => line.startsWith(`def ${name}(`) ? [index] : []);
  assert.equal(indices.length, 1, `expected exactly one ${name} definition`);
  const start = indices[0];
  let end = start + 1;
  while (end < lines.length && (/^\s/.test(lines[end]) || !lines[end] || lines[end].startsWith('#'))) end++;
  return tokens(lines.slice(start, end).join('\n'));
}

export function checkRecipe(table, spec) {
  assert.deepEqual(definition(table, 'build'), tokens(`def build() -> Array<U64>:
    extras(64n, 0, tables(128n, 0, 512, Array.new(U64, 17n, U64.zero())))`),
    'public build recipe differs from the qualified defaults');
  assert.deepEqual(definition(spec, 'run'), tokens(`def run(+d: Nat,seed: U64,n: Nat,extra: Nat,start: Nat) -> Array<U64>:
    Tables.extras(extra,U32.from_nat(start),Tables.tables(n,0,512,Array.new(U64,d,seed)))`),
    'symbolic recipe no longer instantiates the public build expression');
  return {method: 'exact signature/body tokens, whitespace/comments ignored', depth: 17,
    blocks: 128, first_key: 0, first_offset: 512, extras: 64, first_square: 0,
    seed: 'U64.zero()', closed_builder_equality_proved: false};
}

export function readRecipe(tablePath, specPath) {
  for (const file of [tablePath, specPath]) {
    assert.ok(fs.lstatSync(file).isFile(), 'recipe must be a regular file');
    assert.equal(fs.realpathSync(file), file, 'recipe must not traverse symlinks');
  }
  return checkRecipe(fs.readFileSync(tablePath, 'utf8'), fs.readFileSync(specPath, 'utf8'));
}
