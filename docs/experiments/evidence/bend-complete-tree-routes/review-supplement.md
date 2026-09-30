# Complete-tree route self-review supplement

Candidate source `868335129a501b677ca916d2bbdc779a1b2e22e9`, tree
`2e6428e87a347679b2f39b5eacc0a0ea58e421fc`, parent PR #827
`e680c126908d1e816f31c38b7e3255e196e6f174`. Compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, local Bun 1.4.2/Clang 17.
These are additional self-review diagnostics, not independent review, additional
public laws or a second full-aggregate run. This file changes documentation only.

## Composition with the real initialized pipeline

The additional consumer below checked with status zero and exactly
`All terms check.`. The caller supplies only address bounds and numeric inequality.
The actual Array.new -> Tables.tables -> Tables.extras pipeline supplies its
complete-shape certificate, which the real get-after-set theorem consumes.
The result includes the complete returned array/value pair. Counts, scalar
parameters, arbitrary seed and write value remain symbolic source values.

```bend
import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../../Tables.bend as Tables
import ../separation/Spec.bend as S

# Both shape and path certificates are derived, not provided by the caller.
def pipeline_write(+n: Nat,+extra: Nat,+key: U32,+at: U32,+sq: U32,+seed: U64,
  +i: U32,+j: U32,+v: U64,
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) ->
  {Array.get(U64,Array.set(U64,Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),i,v),j) ==
    (Array.set(U64,Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),i,v),
     S.value(Array.get(U64,Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),j))) : Array<U64> & U64}:
  Laws.bounded_other_index_write(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),i,j,v,
    Laws.table_pipeline_complete(17n,n,extra,key,at,sq,seed),ib,jb,ne)
```

Imports are relative to `proofs/complete/`. Source SHA-256:
`5270deb68c5d014f34c1fa990c58007382dada7dcb8f9d7e21bf6e090cddc448`.
Raw command/report SHA-256:
`0d6617a512cfc51cb87f4771cd275c6e4ca409cb1652e53f37682dc7ee552c72`.
This is a source proof, not native execution at arbitrary counts, a computed
attack-content theorem or a prefix-plus-interior bound.

## Independent native reference rejects a protected-slot overwrite

A disposable copy changes only the public probe's `Array.set(U64,a,i,v)` to
`Array.set(U64,a,j,v)`. The candidate writes directly into the query that should
remain protected for different bounded indices. C emission and generic native
execution complete; the unchanged reference rejects the query values with
status 1 and `all observed public API values, capacity and normalization`.
The driver stops at this generic failure; no four-mode mutant pass is claimed.
The original probe is restored in that disposable copy. Accepted source and
compiler files were not changed.

Mutated source SHA-256:
`e1e1eb42b410a52ab8aad5c7140bb6867a0070487beeaa930e35ae0ffbce3ffd`.
Full source/command/report SHA-256:
`c1474cba80968ff5fe2c139b86e813b3bf8fce8e69632a6f9ec9c11443f4b399`.
This extra diagnostic is separate from the 16 published source controls and the
six malformed native requests per mode.

## The whole returned buffer is checked

Another disposable mutation changes the last law's expected returned pair to
contain the original array rather than the actual updated array. PROOF rejects
with status 1 and ordinary expected/observed diagnostics. The actual result
retains `Array.set.fin(...)`, whereas the deliberately false expected pair
contains original `a`. It is not a crash, missing import or timeout. This shows
the contract is not merely about the scalar query result.

Mutated LAWS SHA-256:
`386c450e40ade4f098c588ae5df88eacba8ebb409f6d96637b97af8b2553b56f`.
Raw report SHA-256:
`3461685ea0fc547a31d44c950a89f9c5a716689eec3e89f895435d3cc25a978e`.
The large diagnostic expands the complete depth-17 shape in its context; it is
compressed in the conversation's review ZIP, not committed as a large raw log.
This is additional self-review, not another public law or gate-control count.

## Exact source and retained draft failure

The source-only patch reapplied to a fresh baseline worktree, producing the
complete published source tree with normal Git whitespace checks passing.
Patch SHA-256:
`b1cb7869c6ebf6264ecc4c38dd85167aebeee7d43c12f9c57cc5f9ee5995c141`.
This add-only patch contains the source commit, not later hosted documentation.

An initial mutation-driver draft searched for expanded Array.new syntax rather
than actual `[v : T^p]` notation. Its uniqueness assertion failed before a valid
mutated proof check. It is not a semantic rejection. The corrected driver uses
the actual source spelling and its entire focused gate passed again. Original
failed log SHA-256:
`84f6bee585f573c620397a57e4a0bdc3c60b66fe7e03b95064eb0d6c7b501fd7`.

## Completed qualification, with failures kept separate

- Source run **35764846623** passed the full **83-law/156-control** aggregate.
  Its overall verdict remains red because default Clang's native-target feature
  warnings were rejected after generic and portable modes passed.
- Run **35766677592** verified the retained source report and passed the focused
  recheck, all four native modes with explicit Clang 18.1.3, the original compiler
  source suite and all 12 pin tests. Its overall verdict remains red: job-wide
  `CC=clang-18` also affected Python's editable build, which lacked `omp.h`.
- Attempt **35767241042** stopped while fetching ANSI-bearing historical job logs
  with gh; it did not run lint. The retry saves raw log bytes directly to a file
  without terminal rendering, rather than disabling the terminal guard.
- Final run **35767511451**, job **106880672697**, completed successfully. It
  verified retained reports and all **239 source hashes**, installed the locked
  CPU development environment with its normal GCC compiler, passed unchanged
  Ruff/Basedpyright/Vulture lint, and published documentation/evidence commit
  `a37baa1df7af89c55dc1f6529b45c449a299b7c4`.

All focused source-report fields match the local JSON exactly. Complete native
reports match except C compiler identity. The full source JSON is unchanged from
the first successful source step. Proofs/native checks were not redundantly rerun
during final lint publication; exact source verification ties the stages together.
No earlier failed workflow is relabeled green and no warning policy is relaxed.

Downloaded native-recovery artifact **10712168622** ZIP SHA-256:
`c91f200ed814f33886da5ed3eab7da579198f8e42de1ac8ab16657b81a69332b`.
Downloaded final artifact **10713215339** ZIP SHA-256:
`7bc93f32b88213e0265abc7ac74a450d66641288c680a6eb49e870cf1efd9e7e`.
Both were checked locally, including all candidate source entries and report
agreement. This verifies transport and recorded results, not independent review.

Scope the native compiler override to its command, not Python environment setup:

```bash
CC=clang-18 BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/complete/verify_native.js "$BEND_DIR" --report /tmp/complete-native.json
```

The first runner's default-compiler/CPU feature issue is not claimed fixed.
The public native probe observes query values/capacity/normalization, not all
array cells, raw internal calls or a fresh complete table-builder execution.
Earlier raw-internal lowering, ragged-native, TypeScript and checker-depth limits
remain separate. No source theorem proves native allocation/lifetime or hardware.
The next P2 targets remain actual prefix/interior arithmetic, all fill-clear
certificates, final computed contents and independent blocker-ray lookup equality.

Full local diagnostic sources and outputs, final source patch and hosted evidence
are retained in `bend-complete-routes-review.zip` in the conversation. No production
source, prior accepted law, compiler input, model/GPU/perft/training workload or
performance claim was changed or added by this supplement.
