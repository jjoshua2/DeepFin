# Control partition and strict tablebase CPU controls

Two distinct reviewed CPU gates completed on 2026-10-21. Neither ran a model or
changed a running owner, queue or control marker. Ordinary productive generation
retains priority over candidate GPU experiments.

## Explicit validation contract

The experimental [candidate and contract](lightweight-validation-candidate-v1/LIGHTWEIGHT-CONTRACT-v1.json)
separate launch validation, runtime controls and closure preservation. They reuse
the original guard constructor/charge behavior and retain the production guard.
All unclassified controls remain runtime-checked. A complete, disjoint partition
is required; a hash-checked reference is evidence identity, not semantic approval
of an exemption.

Thirteen synthetic mechanism controls passed in 0.3967181 seconds native elapsed
time, independently accepted with no findings. They include omission/overlap
refusal, active STOP/PAUSE/CANCEL/YIELD refusal, deadline ordering, propagated
guard failures, inherited charge ordering and changed detection coverage. A
declared preservation-only mutation is intentionally invisible during work but
fails the full closure sweep. That is a new contract, not legacy equivalence.

No concrete operational exemptions or live integration are approved. The private
authority map still has zero exemptions. The eventual adapter must invoke the
full closure sweep after authenticated quiescence and preserve the primary
failure. These fake-I/O controls do not establish filesystem performance,
canonical custody behavior or a successful optimized experiment.

Receipt SHA256: `2968594047f857df8e4cc5986f1b3fcb9e5fb9479a78a753e304e5c9b223c748`.
Independent actual review: `f23f8437a250bad44b98936fbfb45287587052987a91798ac29163ad779e6571`.

## Shared strict tablebase handle

The strict match opener now supplies a low-dependency handle with a per-handle
reentrant gate around complete high-level WDL/DTZ probes, their pair and close.
The theoretical tablebase cache and installed dependency are unchanged. Default
file-descriptor bounds remain unchanged. Opening and final close still require
the owner's lifecycle discipline; this does not authorize arbitrary concurrent
low-level access or changes to table files.

Five actual component tests passed with zero errors, failures or skips, in
1.4068156 seconds native elapsed time. The bounded dependency schedule exposed
the original reader/LRU lock inversion and succeeded with serialization. Other
cases covered pair reentrancy, exceptions, close ordering and serial conversion
behavior. The tests used actual reader counting, eviction and close methods,
with synthetic decoding and a deliberately small fixture file-descriptor limit.
They did not read real table files or establish a cause for prior model deadlines.

The exact production component and normal chess import origins were verified.
Test threads joined and the helper observed ECHILD. The separate eight pytest
integration cases remain unrun; these five controls do not qualify the opener's
full caller, native encoding, real decoding, model inference, memory fit or speed.

Receipt SHA256: `e7af6c54be1f6dbe75b4ad2d4a6b41bdf130fb5303839a8ffd976cbeaf414b36`.
Independent actual review: `32be23b0bd864f7c05dd1509cb0588e0196e1464dd92bc5b2d48f1ba0227474e`.

## Test-convention follow-up

CI subsequently reported assertion-style and unused-argument lint findings in
the two new test files. The repair preserves all oracles and exception identity
and ordering without adding a pytest dependency to the low-dependency component.
The override parameter retains its original keyword-compatible name and
explicitly discards its unused value. No lint rule or fixture was suppressed.

Pinned Ruff 0.15.20 passes on the two files. Pinned Basedpyright 1.39.9, using the
repository's configuration and the same two-file scope, reports zero errors and
warnings. These are focused checks, not a whole-repository static result.

The five existing component cases passed again after the final repair, with
zero errors, failures or skips, in 1.4059826 seconds native elapsed time. This is
a regression rerun of the same five cases, not five new cases. The separate
eight-case pytest integration scope remains unrun locally. No manual CI rerun
was requested.

Final regression receipt SHA256:
`c57bbf32a83dfbcdda413a6a3e07a7be4e49ef66b17d97d012095110a785a9a7`.
Independent actual review:
`c3cf98ed7ed5ece0bb807940702ed5396a28b3947700ac232cfb9bfa33792a50`.

## Actual finite-caller wiring component follow-up

A distinct unarmed child, adapter and parent successor now bind the validation
policy and source origins explicitly. The child performs the full preservation
sweep after authenticated quiescence; the parent verifies preservation after
owned-child settlement. Cleanup rendering and verification failures preserve the
primary exception, and verification failure cannot produce a successful closure
record. The production OwnerGuard implementation is unchanged.

The source-bound CPU gate passed 18 observations: eight wiring groups, nine
settlement groups and one actual-callsite AST group. Native elapsed time was
0.8980103 seconds, with complete stdout and empty stderr. The gate exercised the
actual OwnerGuard class body through explicit fake custody, process, memory,
disk and output seams. It did not establish normal production import behavior,
real file-lock custody, native quiescence, performance or model qualification.
Canonical custody was observed only; no lease was acquired by this gate.

All current-authority paths remain runtime checked by default. No concrete
exemptions are approved. Controls demonstrate that an endpoint-only mutation
which is restored before closure can be missed entirely; that coverage change
is explicit rather than claimed equivalent to continuous checking. Frozen
source versions rejected during review remain retained and were never run.

Actual receipt SHA256:
`593e4ffcd6469df217cc9295f683314ac3cfa1ba993388c96813b1e086d0c877`.
Independent actual review:
`d8d2fa020a59cb01b70788d194a1dd61a6befb9e61c60ecfae630c2a1843c42f`.
