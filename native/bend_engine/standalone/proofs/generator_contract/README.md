# Complete legal-generator contract, work in progress

The finish line is [the complete contract](../../../../../docs/bend_legal_generator_contract.md).
This directory does not register an accepted law or increment the proof inventory.

The first candidate targets the exact public-builder boundary:
`Tables.build() == Init.run(17n,U64.zero(),128n,64n)`.
Both sides are the actual production construction calls, and the only proof term
is reflexivity. The importing consumer repeats the complete proposition.
There is no assumed table, desired attack answer, token-guard substitution,
new axiom, proof hole, compiler modification or runtime change.

The earlier symbolic builder result is already qualified. Its public-body token
guard is useful drift detection but is not this closed equality. The pinned
checker's conversion routine weak-head-normalizes both sides; the earlier closed
attempt exceeded 30 seconds. This candidate is deliberately subject to a bounded
hosted resource check before anyone calls it proved.

Run on a disposable CPU environment:

```sh
BUN=bun python -m native.bend_engine.standalone.proofs.generator_contract.qualify \
  build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --report artifacts/bend-generator-contract/closed-builder.json
```

The new GitHub workflow uses the existing compiler installer and pin checker.
The positive compiler child defaults to an 86,400-second (24-hour) wall limit.
Use `--consumer-timeout-seconds` for an explicit job override. A time-based warning
is emitted after one hour; it reports elapsed time only and does not imply semantic
progress. The checker log is retained as an artifact; exceeding the 16 MiB output
threshold terminates qualification. The report includes the last 1 MiB and captured-output digest, and refuses to classify truncated output as success or semantic rejection.
Each of two semantic controls retains its explicit 180-second wall limit.

The hosted workflow retains its existing 22-minute job limit and 1,100-second
cgroup runtime ceiling, with 200% CPU quota, 6 GiB RAM and no swap. These outer
limits bind before the positive child's 24-hour default; a longer child timeout
does not extend the hosted job. Any separately admitted local run remains subject
to its own outer CPU, memory and wall limits. No larger retry follows automatically.
These are qualification resource bounds, not theorem premises. Timeout,
out-of-memory, signals, unsafe output, parser/import/ownership errors or a missing
report are failures, never a proof or a successful negative control.

On a positive consumer pass, two disposable mutations must receive a semantic
checker rejection at the candidate theorem: the actual public builder writes a
different value to unused slot 131071, and the proposition asks for a nonzero seed
(the uninitialized slots preserve the seed). Full array equality matters, even
when every attack query still returns the expected answer. Any resource failure
during a control leaves qualification incomplete. Existing semantic, recipe,
native, compiler and corruption gates are unchanged.

The runner first invalidates an older report, binds transitive Bend and host-helper
source identities, verifies compiler identity before and after, and rejects
warnings even with a zero exit code. Host unit tests exercise those result
classifications under normal Python, -O and -OO. They do not count as source laws
or semantic checker controls.

A pass would close only this initialization bridge. It would not prove castling,
ordinary/en-passant generation, fast-filter equivalence, completeness, uniqueness,
or native execution. A failure will be retained with its exact identity rather
than relabeled a theorem. No local executor, GPU, model or training work is needed.

## Observed result

The initial candidate in run [36807294025](https://github.com/jjoshua2/DeepFin/actions/runs/36807294025)
exceeded the 600-second positive-consumer bound after 600.272 seconds with no
checker output. Its receipt is NOT_COMPLETED: zero accepted laws and neither
semantic control executed. See the [dated failure record](../../../../../docs/experiments/2026-10-01-bend-generator-contract.md).
A timeout is not a theorem rejection or proof of falsehood. No larger retry or
checker/runtime change follows automatically.

## Source reconstruction boundary (2026-10-14)

This four-file wrapper reconstruction changes no Bend theorem, table, count or
training source. The default and explicit integer override reach the positive
checker call; the controls retain 180 seconds and compiler identity calls retain
30 seconds. A parsed nonpositive override invalidates an earlier PASS report
before rejection, and the direct invoke API rejects nonfinite wall bounds before
opening a log or creating a child. The integer CLI parser does not admit NaN or
infinity strings. Host propagation and boundary regression sources use inert
child receipts; they are not compiler qualification or evidence of a new theorem.
No long proof or runtime validation was run while preparing this source packet.
The hosted runner ceiling is documented at
https://docs.github.com/en/actions/reference/limits.

## Shared capture failure envelope

`process_exit_code` retains the unmodified child return code. `exit_code` is
qualification status: it is `None` for an elapsed timeout, output-limit breach
or a truncated report tail. The separate `timed_out`, `output_limit_exceeded`
and `output_truncated` fields preserve the actual reasons; a capture failure
is not falsely labelled a timeout. Full retained logs and the observed-output
digest remain available. This extends the existing timeout envelope so unchanged
callers requiring exact exit 0 or 1 cannot credit an incomplete diagnostic.

The tracked shared callers are the closed-builder wrapper, castle projection,
king-away, bitboard inventory, destination factorization and forward attacks.
The inert regression sources call their real unchanged acceptance functions with
clean receipts, greater-than-1-MiB diagnostics that hide a forbidden prefix, and
output-limit failures. This does not carry forward historical proof/count source
qualification: their helper-source bindings changed and remain tied to the old
revision. No castling, count, theorem or unpublished owner source is adopted.

## Persistent log ownership and completion

Each main invocation creates a private directory under its report parent and
records its actual persistent checker log path before starting the child. Distinct
report files in the same directory remain safe to run concurrently. Supplied
invoke log paths are created exclusively; existing evidence is never overwritten.
Temporary logs keep their original owned descriptor. Tail and retained-size
classification read the same open descriptor after the drain reaches EOF and
joins, never by reopening a potentially replaced pathname.

A parent that exits while another writer retains stdout has not sealed its output.
If the pipe completion grace expires, cleanup is forced and `output_incomplete`
is true; qualification `exit_code` is `None`, even if the parent exited cleanly.
The actual parent status remains in `process_exit_code`; `timed_out` is only set
when the wall deadline elapsed. An uncloseable pipe or failed capture raises and
never produces a successful qualification. Process and reader cleanup waits are
bounded; a failed drain cannot keep the interpreter alive by a non-daemon thread.
The inert host-pipe control and distinct-report control do not create proof
children or send real signals. Runtime tests remain unexecuted in this packet.
