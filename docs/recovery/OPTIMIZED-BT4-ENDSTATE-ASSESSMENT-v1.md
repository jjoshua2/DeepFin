# Optimized BT4: source-grounded end-state assessment

Status: design and ownership assessment only. No implementation, imports, tests, model run, control migration or production adoption in this assessment. The negative lexical-only CPU screen at commit 2a378f2 remains rejected.

## Two separate limitations

The finite diagnostic is not a test of the intended high-supply end-state. Its 64 actors, one producer, native chunk32 and zero fill wait explain the observed32-row ready-prefix dispatches. TeacherDispatcher target_rows is a fill target and upper size for a dispatched route, not a promise that enough rows exist. Raising the target alone cannot create ready work.

Separately, the measured qualification wrapper repeatedly checks externally mutable authoritative paths. The CPU screen kept all seven authenticated present-marker reads,557 absent-marker checks and seven abort-marker checks per full call. Mounted-Windows path boundaries dominated that small measurement. This is not a unique filesystem/syscall diagnosis, and it is not unwrapped production throughput.

## Coherent execution path already present

Reuse run_owned_roster -> run_pooled_units -> GenerationSessionRunner/BT4GenerationDriver, one shared TeacherDispatcher and existing staged backend, ProducerExecutor, ExclusiveLaneRegistry, ReservationCredits, canonical writer and owner checkpoint ACK. SelfplayDomain remains a distinct owner-thread domain; it is not another BT4 producer scheduler.

A320-row reservoir requires at least320 independent active game lanes. Canonical research WorkerSpec permits at most64 parallel games per unit: five64-actor units provide320 lanes in principle. With128-game unit quotas that is a640-game roster, not the old128-game diagnostic. Stalls, terminal games and pending publications can leave fewer than320 ready rows.

TeacherDispatcher max_rows counts queued plus running rows until retirement. target320/max320 cannot retain two full320-row waves. Two full waves need max_rows at least640 and enough independent ready actors (at least640 for both at once); ten64-actor units provide that logical reservoir, not a readiness guarantee. A larger tail-safe reservoir, fill wait and producer geometry require measured tuning, not an optimality claim.

Native CPU microbatches are bounded to256. Multiple completed native tasks must supply a320-row route. More producers can help only with disjoint leased game fields, independent backend/writer progress, and preserved admission/STOP/RNG ownership. No producer may mutate a leased game or replace it before durable writer receipt and owner checkpoint ACK. Preserve preparation/execution/accepted-finish ordering and owner-thread rendezvous.

For400-ply units the source reserves24,244,736 bytes per game.320 trajectories reserve7,758,315,520 bytes (about7.23GiB);640 reserve about14.45GiB, before progress reserves, interpreter, model, native allocations and telemetry. These are conservative schema reservations, not measured RSS. The old memory envelope cannot be assumed to fit.

Full control is called at runner advances, backend and publication boundaries. The per-actor _fill stop predicate checks the producer event and private roster STOP paths; it is not source evidence of a full564-marker guard scan per actor. Larger geometry may amortize some work, but it neither eliminates mandatory checks nor proves a throughput gain.

## Safety-preserving options and material changes

Within the unchanged contract, retain every synchronous dynamic path check at its existing boundary, original error/alias order, deadline, STOP, resource and canonical-custody observation. Pure lexical/schema allocation cleanup does not avoid the fresh filesystem operations and failed its measured benefit screen. A native loop issuing the same operations might reduce Python overhead, but has no evidence of removing the dominant cost.

A mirror, watcher, asynchronous receipt, directory snapshot, negative cache, persistent descriptor or parallel/bulk lookup changes observation timing, replacement/error behavior or mutable-alias ordering. Those are not equivalent optimizations. Moving controls or changing mounts/security is outside scope.

A concrete material research proposal is a synchronous Windows-native verifier reading the SAME authoritative files, while canonical WSL custody and retained guard remain in WSL and execution waits for a fresh owner-bound verdict. It is not currently a supported or equivalent interface. It requires explicit path/reparse-point/dangling-link/error mapping, paired file identity/stamp/byte contracts, bounded authenticated rendezvous, current-alias semantics and failure propagation before model execution. Wrong mapping or stale verdicts could miss a STOP/replacement or authorize work after a control changed. No such verifier is implemented or authorized by this document.

The alternative policy change would be to observe selected preservation-only records only at experiment endpoints. That explicitly loses intermediate-change detection and is NOT proposed as an equivalent fix; it cannot be done under the present every-call contract. No record is reclassified or omitted here.

## Smallest credible next evidence, and blockers

First prove the existing actual driver/dispatcher can gather and route320 independent fake rows with multiple disjoint producer completions, bounded fill/tail behavior, FIFO/scatter, STOP, failures and writer->ownerACK->refill. This can establish logical supply and ownership without another GPU run. It cannot establish native thread safety, full import fit, model correctness, host/VRAM fit or throughput. fake_cpu alone does not make full worker imports lightweight: an exact selected import closure and unchanged CPU/resource gate must be reviewed before execution. No executable packet or reservation is created here.

Before real320 qualification, require a freshly authorized multi-unit roster, exact source/provider/model/native identities, aggregate trajectory/progress/output budgets, unique outputs, finite deadline and genuine all-three custody/restoration. A single-unit GO or consumed diagnostic grant cannot authorize this roster. There is currently no qualified cross-filesystem-avoiding checker and no admitted320 roster. More GPU diagnostics on the old geometry would not resolve those gaps.

Decision for parent: retain the current checking contract and qualify only the logical supply path first, or explicitly authorize research into the synchronous same-path Windows verifier with the risks above. Neither choice merges, deploys, resumes evaluation or changes existing STOP/PAUSE controls.