# Owner checkpoint ACK and refill component readout

Four distinct CPU component cases passed on 2026-10-21. The original V3 run, V5 style-successor retest and V6 typed-successor retest each ran the same four cases once, with no failures, errors or skips. Independent saved-result reviews accepted all three scopes. The two retests add zero distinct cases.

| Run | Test source SHA256 | Native seconds | Guest seconds |
|---|---|---:|---:|
| Original V3 |1146180e30c25421ef430ec239c3aab782272c5c924814a8be46d8f9dd7a3efc|1.8349155|1.40775648|
| Style successor V5 |b2ee4a50e579b77f4b1c9327f46854623dcc2cdc69d63a152608edf7d4a0afbe|1.9130471|1.46728825|
| Typed successor V6 |e32a17edcd5c2fea6a3b57a339527e25019b01d07caf20b0d594d7a19fc7be93|1.7664801|1.38172765|

All three commands exited 0 with complete stdout/stderr EOF. V1, V2 and V4 remain unrun source versions. A V6 transport refusal occurred before native process creation; saved intent/output absence established that source had not started. That refusal is preserved separately from the one subsequent confirmed V6 execution. The companion JSON records full source and evidence hashes for each completed run.

The positive case used the actual DurableWriter queue, retained immutable raw verifier, bind_owner_checkpoints and retained Checkpoint.record_game transaction. A writer receipt alone left the checkpoint empty. Owner-thread ACK committed the receipt before the actual pool/driver publication path released the slot and constructed its replacement. A partial fixture could not earn full 128-game completion.

The other cases passed: a writer failure produced neither ACK nor replacement; a mismatched immutable receipt was rejected without replacement; and STOP while a writer was pending retained its slot and later durable raw receipt without admitting ACK or replacement. STOP state, actors, custody identity and the zero-row NPZ publication are explicit fixture seams. Production NPZ serialization, native moves and roster admission were not exercised.

V5 replaced unittest assertion methods with plain assertions and an explicit stdlib contextmanager/regex helper, plus dict literals. The helper invokes its body once, preserves expected exception types and regex searches, and propagates other exceptions. Its regex assertion is outside the exception handler. V6 adds resource-free constructor defaults, precise ordinary field types and checked accessors for actual runtime objects. Dynamic types remain limited to the declared retained-reference and inert-fixture boundaries. Actual resources are still created in setUp in their original order. The controller requires optimization 0. No pytest dependency, suppression, runtime change or game-semantic change was added.

The final V6 source passed pinned Ruff for twelve explicitly selected integration production/test files, including this test. Focused Basedpyright for this test under the existing project configuration reported 0 errors, 0 warnings and 0 notes. These are scoped static checks, not whole-repository qualification.

Each run bound all sixteen copied source files, checked 119 normal module-origin hash joins before and after the methods, and hashed 277 inventoried NumPy/chess dependency files twice. The retained checkpoint and raw verifier use the original bodies. Three native-checkout files had CRLF bytes where the canonical checkout had LF; normalized text equality was explicitly checked while exact copied hashes remain separate.

After every method, owned Python writer/producer joins and an empty retention registry were asserted. Temporary fixture cleanup returned each run to its source-copy census: 16 files and 8 directories, with 146733 bytes for V3, 146881 for V5 and 148409 for V6. No success receipt contains a separate numeric retention-counter field. This is no whole-process or ECHILD proof. Unknown cleanup remains fail-closed in source; none of these runs injected an indefinitely stuck thread.

These results establish four CPU component contracts. They provide no model, native gameplay, GPU, five-unit roster, full-unit, physical memory-fit, throughput, training or production-adoption qualification. The test source's historical UNRUN header is preserved; later scoped results are recorded here. Raw operational evidence remains private. This publication contains the test and summarized source/evidence identities, not operational helpers or absolute paths.