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
The complete hosted qualifier runs in a transient cgroup limited to two CPU
cores of quota, 6 GiB RAM, no swap and 1,100 seconds. The positive compiler child
has a 600-second wall limit; each of two controls has 180 seconds.
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
