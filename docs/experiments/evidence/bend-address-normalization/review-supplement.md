# Address-normalization self-review supplement

This record concerns source commit `a26e59f7168519c7de9720b9edb7967ef0650d97`
on PR #827, with pinned compiler
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2 and local Clang 17.
It is self-review, not independent review. The checks below are additional
bounded diagnostics, not extra public laws or additional full-aggregate passes.
They do not modify the candidate, compiler, previous proofs or ordinary test budget.

## Capacity and numeric injectivity do not prove leaf-path separation

The following source example passed with status zero and exactly
`All terms check.`. Its left subtree has depth 16 and its right subtree is a
single leaf. The source Array.size operation observes the left spine, so the
reported capacity is 131072. Two unequal in-range addresses nevertheless reach
the same right leaf:

```bend
import Base
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

def ragged(+left: U64,+right: U64) -> Array<U64>:
  ANode{Array.new(U64,16n,left),ALeaf{right}}

def capacity(+x: U64,+y: U64) ->
  {S.size(O.shape(ragged(x,y))) == 131072 : U32}:
  {==}

def both_bounded() ->
  {Bool.and(U32.is_lt(65536,131072),U32.is_lt(65537,131072)) == True{} : Bool}:
  {==}

def unequal() -> {U32.is_eq(65536,65537) == False{} : Bool}:
  {==}

def not_separate(+x: U64,+y: U64) ->
  {S.separate(ragged(x,y),65536,65537) == False{} : Bool}:
  {==}

def aliased_write(+x: U64,+y: U64,+v: U64) ->
  {S.value(Array.get(U64,Array.set(U64,ragged(x,y),65536,v),65537)) == v : U64}:
  {==}
```

Relative imports are from the normalization suite directory. The full executed
fixture, including comments, has SHA-256
`9dddfec159f265055c48cada6f60f2a8f4d9aed87db507883b966b54455de20d`.
The raw command/result report is `ragged-capacity-counterexample.json`, SHA-256
`0c4f716d9e30c6b35e052c79ce103163ede1134bd57617da56d619e9494de549`,
in the conversation's review evidence package.

This is not a demonstrated bug in the actual complete depth-17 table allocation.
It disproves the stronger claim that observed capacity and unequal bounded
integers suffice for arbitrary constructor shapes. The accepted normalization
laws do not assert that claim. The next source proof must use the complete-tree
shape, preserved by the existing builder-shape theorems.

## Minimal direct-internal-routine C-emission failure

```bend
import Base

def value(r: Array<U64> & U64) -> U64:
  (a,v) = r
  v

def main() -> U64:
  value(Array.get.go(U64,Array.new(U64,1n,U64.zero()),2,0))
```

The original CLI normalizes this program to `U64{0, 0}` with status zero.
Adding `-o direct.c` fails with status one and
`Error: an open Array element type`. Replacing the expression with
`Array.get(U64,Array.new(U64,1n,U64.zero()),0)` normalizes successfully and emits C.
The corresponding Array.swap.go/direct versus Array.swap/public pair has the
same normalization/emission outcomes. These four minimized checks did not run
their generated executables. Their source bytes, commands, outputs and hashes
are retained in `native-minimal/report.json`, SHA-256
`db911c3478973823f40195602cafbca459e85b510eea0f7fc7c0bf132d82a7ae`.
The original larger probe failure remains committed separately, not replaced by
these smaller examples. No compiler fix is claimed.

## Native construction rejects deliberately unbalanced arrays

Two further tiny fixtures use this shape:

```bend
def ragged() -> Array<U64>:
  ANode{ALeaf{U64.from_u32(1)},ANode{ALeaf{U64.from_u32(2)},ALeaf{U64.from_u32(3)}}}
```

Source evaluation returns size **2** and value **3** for a public read at index 1.
Both programs emit C and compile under Clang 17 with undefined-behavior
sanitization, but their executables exit **1** with `bend: runtime fail-stop`.
This is consistent with the pinned native `blk_node` guard rejecting children
with different block classes; the JavaScript implementation also explicitly
rejects unequal child lengths. The guard was inspected, not changed.

These are intentionally nonuniform source trees, not the complete allocation
used by Tables.build. They are not added to the passing native-wrapper fixtures
and are not classified as successful native execution. The source law's ability
to quantify over such arrays does not establish their native constructibility.
Raw source, command-stage outcomes and outputs are in `ragged-native/report.json`,
SHA-256 `4320ac98f5a612028135633ac57e90db0fab7c84dbd32eb2b601090073f8893b`.

## Independent native expectations reject a shifted write

In a disposable copy of the final supported public probe, replace only
`Array.set(U64,a,i,v)` with `Array.set(U64,a,U32.inc(i),v)`. The unchanged native
driver compiles the generic-mode candidate but rejects its returned values with
status one and the assertion `all observed public API values, capacity and normalization`.
This verifies that the oracle can detect a wrong-location write; it is not merely
checking successful process exit. The candidate and compiler remain unchanged.
The raw diagnostic is retained in `native-shifted-write-rejection.json`, SHA-256
`3a7d49e59d24d721b53e44edb71954091f00c4e094568fa223c1adbd065c9667`.
This extra diagnostic does not change the published gate's 14 source controls
or six malformed native requests per mode.

## Interpretation

The five new laws remove address normalization from bounded source expressions.
They do not make the internal raw APIs native-supported, turn ragged source arrays
into native complete arrays, establish physical lifetime safety, derive all
prefix-plus-interior bounds, or prove final computed attack-table contents.
Those distinctions and the pinned-compiler trust boundary remain explicit.
No training, model, GPU, perft or performance work was performed by these checks.

## Downloaded qualification evidence

Hosted run **35759797335**, job **106854671729**, completed successfully with
79 accepted laws, 140 rejection controls, four supported-public-API native modes,
original compiler source/pin tests and unchanged whole-repository lint.
Artifact **10709989407** has ZIP SHA-256
`0c38ccc154617de7e52006739632a7d214e050ddb9b5b2070afc49ea77aecec3`.
After downloading it, all **228 source-manifest entries** matched local file bytes.
Every focused source-report field agrees with the local report, and the complete
native report agrees except for the separately recorded Clang version.
The hosted evidence commit is `c25f656f39d2e6cf2862a1a58fc64ff612c079b6`.
This supplement adds no executable changes after that qualification.

Full local diagnostic reports, their sources and command outputs are retained in
`bend-address-normalization-review.zip` in the conversation, alongside the new
source patch and the unchanged downloaded qualification ZIP. Those local diagnostics
are separate from the exact hosted aggregate and native fixture counts above.
