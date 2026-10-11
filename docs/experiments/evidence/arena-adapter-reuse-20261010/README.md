# Arena adapter reuse source packet

This archived operational-source draft belongs with the
[dated record](../../2026-10-10-local-arena-adapter-reuse.md), not in the general
runtime API. Candidate production bytes match reviewed local revision
`870d9aa84d58e48a90e44fafac24bf263fb65167`. No active runtime is edited or adopted.

From this directory, run only the isolated contract tests:

```sh
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 28 nice -n 19 /usr/bin/python3.10 -B -m unittest -v test_adapter_reuse
```

All subprocess workload and canonical-lease interactions are mocked. The tests
read private temporary inputs and the two original hash-pinned baseline fixtures;
they do not require the local prototype's Git history. The original adapters are
stored as `.py.txt` reference inputs, not selectable launch implementations.
`.gitattributes` preserves exact source/fixture bytes across checkouts because the
shared source hash is an executable contract.

The 11 tests cover unchanged shared functions, exact legacy commands for both
variants, default/explicit bindings and unsupported CLI rejection, captured pinned
source and fresh wrapper identity, source tampering, shim/attempt bounds, inherited
descriptors and create-only outputs, finalizer refusal of code 3, missing inherited
descriptors, authority/process/canonical inode identity, and manifest tampering.

`local-adapter-reuse.patch` and `local-verification.json` retain the original local
implementation/review record. `packet-verification.json` records the portable
packet validation separately. The historical patch's README reflects the earlier
placement investigation; the dated record explains the resolved evidence placement.

[ADOPTION.md](ADOPTION.md) is preparation guidance for a separately authorized,
reviewed future launch. It does not authorize an arena or repinning the current
armed evaluation. Bulk models, panels, logs, caches, queues and outputs are absent.
