# Explicit bounded search arenas

## Preregistration

Continue the collections work on #870 at
`5f7991bc91906440f3baa66f732fa46a5f05bf49`. All three workflows on that head
completed successfully before this change. Its earlier start-position callback
screen stopped at 184 of 256 requested simulations because the 4,096-node arena
was full. This is an allocation/admission problem, not evidence for another
queue substitution or a faster hash map.

Intervention: specialize the existing `Base.Array<Node>` allocation to a requested
logical node capacity, with power-of-two physical storage and a fixed upper bound.
The opt-in cohort reads `DEEPFIN_COHORT_ARENA_NODES` once at startup. Absence keeps
4,096; an explicit ASCII decimal in 1..65,536 selects a fixed arena for each root.
No automatic growth, in-search copying, String map, third-party source, compiler
update or production setting is introduced. Normal standalone/UCI and the session
transport keep their existing 4,096-node configuration limits.

Compatibility: for logical capacities <=4,096 keep the old physical 4,096 slots
and pending sentinel. Larger logical capacities use the next power of two (up to
65,536). `no_pending(cap) = max(4096, cap)` is always outside valid logical node
IDs; completion, cancellation and deadline halt paths use it consistently. Resume
also requires the ticket node to be below both `used` and `cap` before any array
access. Idle sentinels and unallocated slots must not be accepted as replies.

`Search.start` rejects invalid capacities by returning a stopped invalid tree
(code 4) with one physical slot; it does not allocate an arbitrary large arena or
silently continue with a different request. These are constructor/API invariants,
not a claim that exposed internal Tree constructors are safe deserialization.
The cohort rejects invalid environment values with exit 2 before position loading
or evaluation. Values have at most five characters; leading zeros within that
limit are allowed, whitespace/signs/nondecimal notation/Unicode digits are not.
Nondefault capacities produce `info string cohort_arena LOGICAL PHYSICAL` before
search output. Default output is unchanged. Parsers must opt into a nondefault
capacity and reconcile that declaration, not silently relax all old bounds.

## Gates and budget

The existing source-only completion CI is extended, reusing its freshly generated
coordinator and callback matrix. Existing default-capacity oracles, control tests,
normal/UBSan configurations and numerical tolerances remain. The independent Python
reference takes the declared logical capacity; its only sentinel change is the
same out-of-range value at capacities above 4,096.

Small compiled probes exercise logical capacities 0/1/20/4095/4096/4097/8191/8192/
8193/16384/16385/32768/32769/65536/65537/U32_MAX, writes across node 4,096 and to
65,535, exact U64 payloads, physical size, idle/unallocated/out-of-range replies,
legitimate high-ID resumes, and the actual cancellation/deadline helper functions.
Run normal and UBSan. Independently checked output must reject three compiled and
normally exiting mutants: fixed-small allocation, colliding sentinel, and removed
live-node bound. A compile failure or crash is not accepted as an oracle rejection.

For each input width 146/175 and synchronous/asynchronous mode, run the current
actual coordinator with absent/default/8,192 settings, exact small boundaries
1/20/21, a non-power-of-two 4,097, a 65,536-node allocation, and a mixed three-root
16,384-node cohort. Every diagnostics-on run checks full trees, leaf encodings and
non-time work against the unchanged CBoard/Python-chess reference calculations;
a diagnostics-off repeat must preserve root summaries and work. One full 8,192
case also runs the UBSan coordinator. The decisive 8,192-node start-position gate
requires 256 completed simulations, more than 4,096 actual used nodes and no
capacity stop, while the default still stops early. This prevents a parsed but
ignored setting from passing. Larger searches doing more work are not speedups.

Hosted budget: locked CPU environment, two Torch threads, one compiler job, the
existing default matrix plus 37 added diagnostic cases and their quiet repeats.
No model export, GPU, live process, training run, timing-performance experiment or
Elo claim. The maximum allocation is opt-in: 16 roots can reserve 1,048,576 node
slots, plus other application memory. Node count is not a measured byte/RSS budget
or a guarantee that this is safe alongside training. Smaller arbitrary capacities
still retain at least 4,096 physical slots for compatibility. No dynamic growth
pauses are added, but increasing preallocation costs initialization and memory.

Recovery: leave the default unset/4,096 or revert this source-only PR. No merge or
deployment is part of qualification. Self-review only; no independent review or
formal proof. The upstream collection work remains a design reference, not copied
code. Dynamic growth and numeric-key maps remain separate decisions.

## Local checks and setup limitations

The pinned compiler and local Clang 17 compiled the storage/transaction probe;
76 exact rows matched an independent expectation in normal and UBSan builds.
Five explicit valid settings and fourteen invalid inputs passed. All three
compiled semantic mutants were executed and rejected. The initial live-node
mutant needed a type-inferable always-true expression before it could compile;
that initial compilation failure is not counted as mutation rejection.

The local standard pytest entry point could not run because the container lacks
python-chess required by the repository's global conftest. Portable parser tests
were run with `--noconftest`; this is not the locked hosted regression result.
Full coordinator and reference qualification remains the hosted gate below.

Source snapshot setup initially archived a missing shallow-history commit behind
a pipeline without pipefail (run 35995907813). Its tiny empty archive was rejected
by inspection and never used. Run 35996043610 explicitly checked out the exact
parent, enabled pipefail and required an archive inventory. This setup correction
changed no application, test oracle or compiler. Source archive artifact 10806730321
ZIP SHA-256: `2d62fb30f02abda849ba0f49621d82c4bdd83a4f1615bcf203a449c4e02b65cd`.

## Readout

The completed hosted qualification is recorded below.


### Completed hosted qualification

Run https://github.com/jjoshua2/DeepFin/actions/runs/35998021794 passed all preceding gates. 345 focused Python cases passed without skips, including 38 new cases. Focused static checks and whole-repository Ruff/Basedpyright/Vulture passed. The unchanged native completion/worker tests and negative controls passed. Fresh source generation passed the existing 33 default-capacity reports: ten batch/width configurations in both modes, coordinator UBSan, held cancellation/stop/quit and deadline gates.

The native arena probe matched 76 exact output rows per normal/UBSan mode, including IDs 4,096 and 65,535 and actual cancellation/deadline helper calls. Fourteen invalid environment settings were rejected; all three compiled-and-executed semantic mutations failed the independent output oracle. All 37 additional diagnostics-on search cases passed independent leaf input, numerical, complete-tree and accounting checks; their 37 quiet repeats preserved summaries/work. The larger-arena UBSan coordinator case is included in that count. Repeated modes reuse fixtures, not independent games.

| Width | Async | Default simulations | Default nodes | 8,192 simulations | 8,192 nodes |
|---|---|---:|---:|---:|---:|
| 146 | False | 184 | 4095 | 256 | 5663 |
| 146 | True | 184 | 4095 | 256 | 5663 |
| 175 | False | 184 | 4095 | 256 | 5663 |
| 175 | True | 184 | 4095 | 256 | 5663 |

This establishes actual search beyond the old node limit, not a throughput or Elo improvement. Default behavior remains capacity-limited and explicit default output/state matches absent configuration. Maximum-capacity validation is a small smoke plus high-ID storage/transaction tests, not an exhaustive search filling every slot. Fixed preallocation trades memory/startup work for room; no live setting changed. Source/readout evidence is retained under evidence/bounded-search-arenas/. Its source manifest describes the tested preregistration before this documentation-only readout. Self-review only; no independent review or formal proof. Nothing merged or deployed.
