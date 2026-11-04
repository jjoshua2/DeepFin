# Generation runtime recovery checkpoint

**INCOMPLETE, UNQUALIFIED, NOT THE FINAL V8 CANDIDATE. DO NOT MERGE OR DEPLOY.**

This draft checkpoints 19 recoverable Python modules after the complete working source and qualification artifacts became unavailable. Each module is byte-identical in two retained runtime fixture copies captured around 2026-10-20 18:35 UTC. The adjacent manifest pins every recovered file. This is source preservation, not a tested integration or throughput result.

The original final v8 patch was recorded against base `1fdd4c854ace0d63ac08e18e87513741ea0eefec`, SHA256 `a15010e7743ac7f9fe770f62f772c51345cba8c5d6851745474ac5d4db275c81`. Its exact bytes are not present here. No prior v8 or v7 pass count, review approval or batch-supply measurement transfers to this recovery checkpoint.

## Recovered pieces

- Common generation runtime, exclusive producer executor, completion/refill, FIFO request routing and staged backend machinery.
- Teacher dispatcher, BT4 prepared-root/evaluator/stepper/worker and canonical durable writer/owner integration.
- Ceres backend/pipelined-client/service adapters used by the generation path.

Ten retained external operational-owner helpers are excluded because their test-runtime paths do not authenticate them as repository additions. They require the existing owner's original source, configuration and custody, not guessed replacement files.

Some modules may equal base. The checkpoint preserves the recovered module set at its actual repository paths without inventing missing edits. No fixture configurations, interpreter/native binaries, models, data, private logs or unrelated packets are included.

## Known incomplete interfaces and behavior

- Selfplay manager/domain adoption and the final Torch dispatcher adoption are missing. Preserving common modules alone does not establish a fully shared execution path.
- Native batch encoding must be reconciled with base Cython/native source, rebuilt in an isolated environment and checked. No copied native image is supplied.
- Final guard thread affinity, acceptance ordering and ownership/ACK corrections must be recovered or reimplemented and independently reviewed. Do not run this checkpoint through production callbacks.
- Eight later backlog scopes are not recovered here: STOP teardown recycling, writer-lock inode validation, checkpoint architecture selection, quarantine restore containment, dropped-value warning privacy, watcher session handoff, replay base-plane admission, and PID node invariants.
- The final regression suite, execution ledger and review artifacts are missing. Static correctness, imports and runtime compatibility are unqualified. Existing tests on base are not evidence for the recovered files.

## Completion and qualification plan

1. Prefer recovering exact final source if a verified copy exists. Otherwise continue as a new candidate, with its own identity, tests and review. Reconcile existing PR/owner work before integration; do not overwrite a live checkout.
2. Restore the missing actual callsite integrations and behavioral corrections, preserving RNG, source/history semantics, STOP, cancellation and durable ACK custody. Check all dependencies on current base rather than adding another runtime.
3. In an isolated checkout, follow current AGENTS/CLAUDE/development guidance for the locked Python/native build. Keep CPU and accelerator build variants separate; do not rebuild anything used by active jobs. Run meaningful actual-import CPU controls and configured static gates.
4. After the generation owner establishes safe custody and resource availability, qualify actual BT4 provider/model/input/output and tablebase identities, first-root CUDA profile and canonical output receipts. Start with bounded correctness, then compare matched baseline/candidate real end-to-end execution. Keep evaluation STOP markers and live work untouched.
5. Measure dispatched logical rows, startup/steady/drain tails, completed games/roots, ACK/refill correctness and resource use. Never infer GPU performance from CPU stubs or manufacture full batches with unbounded waits. Ceres physical-batch qualification is separate.

Publication of this draft is a durable recovery checkpoint only, not a request to merge, deploy, launch work or bypass outstanding gates.
