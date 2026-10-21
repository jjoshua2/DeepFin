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
