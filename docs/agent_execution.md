# Agent execution cadence

Use this procedure for delegated work and finite tests. The task owner remains
responsible for progress and resource ownership. It is not a new scheduler,
background watcher, deployment mechanism or source of execution permission.

## Keep a next action, not just a status

At each useful boundary, keep a short task note with the current owner, any active
process/session, the next concrete action and any exact blocker. Reuse the task's
existing record; do not create a new approval packet or artifact for every poll.
Distinguish `running`, `terminal awaiting reconciliation`, `ready` and `blocked`.
An agent message, process exit and accepted qualification are different events.

When a tool returns a live session, use its supported wait/poll interface until
terminal or attention is required. Do not wait for a user prompt to inspect a
completed session. Use progress-aware waits and concise updates; neither busy
polling nor an arbitrary delay improves completion detection. Never claim a
process, descendant or custody lease closed solely because a transport returned.

When a terminal or worker result arrives, act in the same work cycle:

1. Preserve the result and reconcile the evidence needed by the affected action:
   exit/EOF, source and output identities, remaining work and custody where relevant.
   Use fail-fast wrapper error handling and check each native command's exit,
   expected receipt schema/hash and complete capture/EOF. A tool transport exit0
   is not success if the wrapper logged an error or produced a null/invalid receipt;
   PowerShell can otherwise continue after a nonterminating error. Preserve that
   record as failed and correct it through an authorized successor, not an overwrite.
2. Select the prepared outcome branch below. A failed or partial result is not a
   passed qualification; review and artifact acceptance may remain outstanding.
3. Take the next already-authorized action, or dispatch the next bounded task to
   its existing owner. If a routine readable prerequisite is missing, collect it
   through the authorized route rather than stopping at a source-only proposal.
4. If no action is currently allowed, state the exact blocker, its owner and what
   evidence or decision resolves it. Continue independent work that does not rely
   on the blocked state. Do not replace a specific blocker with a blanket request
   for user approval when authorization is already present.

A new action still needs all applicable source, review, expiry, identity, custody
and resource checks. A consumed or expired action is not refreshed by this rule.
Honor actual access/approval denials. Do not change controls or retry an uncertain
launch merely to keep the task moving.

## Restart the worker, not its mailbox

Where the team tools are available, use the operation appropriate to worker state:

| Worker state | Next assignment or correction |
| --- | --- |
| Running | Send a message to steer the current task; retain its ownership. |
| Completed/idle, and more work is authorized | Use `followup_task` (or the platform's explicit task-resume operation) to assign and start the next bounded task. |
| Completed/idle, information only | `send_message` is sufficient; do not describe the worker as running again. |
| Failed or disconnected | Preserve the failure and reconcile uncertain actions before a new assignment or retry. Restart only through a supported operation and within current authority. |

Check the returned worker state. Do not send an assignment to a completed worker,
assume a mailbox message restarted it, then wait for work that never began.
Starting a new worker is unnecessary when the existing owner can safely continue.

## Prepare three branches before a scarce-resource test

Record the next decision for each outcome before launch, using the existing task
record and reviewed owner/settlement interfaces:

- **Success or informative partial:** bank raw results and exact scope, obtain the
  required independent acceptance, then select the next authorized comparison or
  implementation. One batch or one partial receipt does not prove a full unit,
  sustained throughput, adoption or training credit.
- **Startup failure:** retain the exact error, provenance and any actual work or
  partial files. Select a bounded source diagnosis or correction. Do not infer
  zero work from an exception alone or replay a consumed one-use action.
- **Timeout, resource abort or disconnection:** retain diagnostics and checkpoints;
  reconcile actual process/custody state with the supported settlement path.
  Unknown closure stays a hold on conflicting resource work. Do not invent
  ECHILD, kill outside the authorized cleanup contract, widen caps or drop checks.

A transport failure is not proof the remote action never ran. A recovery probe,
if authorized, establishes transport only; reconcile the prior action before any
replay. An actual approval/access denial is not a transport retry opportunity.

## Keep independent lanes useful

Separate implementation/CPU prerequisites, independent review and productive
fallback/recovery preparation when they can progress independently. Give writing
agents isolated worktrees and bounded ownership; reviewers may share frozen inputs.
Use available workers for distinct work, not duplicate implementations or launchers.

Keep one launch owner for a shared scarce resource. A fallback lane may prepare and
review a qualified successor while CPU work continues. Its actual admission must
be coordinated: known prior custody, current controls/identity and the existing
resume/yield route are required. Do not replay an old launcher, remove PAUSE/STOP,
create a second scheduler or resume generation solely because this document says
"fallback." Do not leave an authorized productive fallback idle merely because a
separate candidate is still being prepared; identify and resolve its concrete gates.

Prepare alternative diagnostics/configurations before the current test ends where
possible. Their source preparation is not launch permission. Do not start a
conflicting timing window while another owner holds it.

## Checkpoint and communicate

Commit useful reviewed source and compact evidence at meaningful boundaries so a
transport interruption does not erase the only copy. Preserve originals, failed
attempts and honest unrun labels. Push through the existing authorized branch/PR;
publication, merge and live adoption remain separate actions. Do not manufacture
extra commits or rerun passed checks without a changed reason.

Report a useful change, actual result or exact blocker during sustained work.
When asked about delay, distinguish job execution from preparation, review,
transport failure and avoidable serial work using recorded timestamps. A promise
to go faster is not a substitute for changing the workflow or resolving a gate.

## Scenario walkthroughs

These are decision-rule examples for review, not automated test results or evidence
that platform notification delivery changed. None authorizes GPU work or control
mutation by itself.

| Simulated event | Required next action | Must not happen |
| --- | --- | --- |
| Worker completes source work; its review assignment is authorized | Preserve the frozen diff, then start the existing reviewer with `followup_task`. | Send-only to an idle reviewer and assume review started. |
| Tool returns a live test session | Wait/poll that session using supported tools while independent source work progresses. | Wait for the user to ask whether it finished. |
| Test returns success; acceptance and next CPU step are authorized | Reconcile scope, obtain required acceptance, then run or dispatch the ready step after its gates pass. | Count unaccepted work or ask the user to repeat existing authorization. |
| Readable dependency identities are missing and read-only collection is authorized | Collect the bounded evidence, review it and continue binding the test. | Stop at a recoverable inventory gap without taking the authorized read. |
| Test fails during startup | Preserve the exact failure and select the prepared diagnosis/correction branch. | Blind retry, discarded evidence or automatic new resource authority. |
| Test times out with custody unresolved | Keep conflicting runtime held and use the authorized settlement/closure path. | Launch fallback on process-exit or transport evidence alone. |
| Connection drops after a launch request | Treat the action as uncertain and reconcile before replay; continue unrelated permitted source work. | Assume no process exists or restart the executor/change access. |
| Explicit access/approval denial occurs | Stop that operation, preserve the stated reason and pursue unaffected work. | Reclassify the denial as transport and route around it. |
| Candidate remains unready; productive fallback has a qualified authorized successor | Coordinate the sole resource handoff and admit only after fresh gates pass. | Idle indefinitely, duplicate the owner or remove controls blindly. |

The repository has no agent-completion event consumer in the inspected `.agents/`,
`.claude/`, `tools/` or `scripts/` paths. These instructions improve the behavior of
an agent that receives a result; they cannot change platform scheduling, deliver a
missing event, wake a finished agent automatically or guarantee real-time latency.
An actual event-delivery defect must be reported through supported platform tools,
not papered over by a repository daemon or a new production queue.

## Durable backlog handoff audit

Use `scripts/backlog_handoff.py` with the **existing** finding ledger. The canonical
tracker remains [Review Atlas](https://deepfin-review-atlas.new-gecko-0210.chatgpt.site/#queue).
Keep finding IDs, repair-group membership, dispositions, old failures and evidence
intact. Do not replace them with a second tracker, new totals or a count of started
turns. The historical Atlas population, a newer local inventory and Security Cloud
occurrences are separate populations. Neither source review nor selected test-case
counts establish full finding closure.

The local inventory shape is `findings: [{finding_id: ...}, ...]`. The recovered
canonical crosswalk v22 declares schema `backlog-assignment-crosswalk/v18` with
`records: [{id: "commit:...", ...}, ...]`; the audit also reads that exact shape and
retains the `commit:` prefix. It refuses an ambiguous crosswalk with a separate
`findings` member. Both adapters add only a `handoff` member. A different source
shape needs a reviewed adapter preserving IDs; do not guess a mapping. Never mutate
a sealed historical inventory: make a successor through its existing producer, bind
the predecessor identity, and regenerate the seal after adding handoff state.

At a terminal worker/tool result, dispatch decision, useful cycle boundary, restart
or report, record one complete observed cycle in `handoff.cycles`. Use the existing
ledger's writer and atomic update route; do not introduce a polling scheduler. Each
cycle has a unique `event_id`, timezone-aware `at`, pinned observation `evidence`,
`items` keyed by existing finding ID, `selected` finding IDs and optional `hold`.
Each evidence reference has a nonempty `ref` and SHA256; hashes are declarations
until the normal evidence review authenticates their contents. An audit does not
fetch the referenced evidence or independently observe worker/platform state.

Each open item records:

- `state`: `open`, `ready`, `running`, `blocked` or `terminal`; `updated_at`; and an
  `evidence` list. `terminal` means returned work awaiting reconciliation, not done.
- `owner`: `{task_id, status: "active", evidence}`. A role label, null owner or
  completed task does not count as an executable assignment. `next_action` has a
  concrete `description` and its own active `owner` using that same shape.
- `permission`: `{state, reason, evidence}`, where state is explicitly `granted`,
  `pending`, `denied` or `uncertain`. Also record Boolean `stop` and `custody` as
  `settled`, `held` or `unknown`. These are observations, never grants or waivers.
- `depends_on`: existing finding IDs; `blockers`: objects with exact `reason`,
  active `owner`, resolving action in `resolution`, and pinned `evidence`.
- A `worker` for terminal items (optional otherwise): `{task_id, state, evidence}`, with state `running`,
  `completed`, `failed`, `timeout` or `disconnected`. Every non-running result on
  an unresolved item needs `continuation`: `{kind, owner, description, evidence,
  worker_result}`. `worker_result` equals the SHA256 of that worker result;
  `kind` is `review`, `dispatch`, `reconcile` or `blocked`. Record an actual resume
  operation and its returned state as evidence before claiming a worker restarted.
  A restart cannot erase an earlier unreconciled terminal result: later continuation
  or terminal resolution must explicitly bind its result hash, even after a new worker.

An evidence-backed `done` disposition may omit active ownership and next action;
it requires `resolution: {ref, sha256, reason}` and settled custody. If reconciling a
prior worker, also bind its hash in `resolution.worker_result`. This permits reviewed false positives
and accepted terminal decisions without inventing active workers. It does not itself
prove that tests ran. Optional `credits` require unique `claim_id`, `kind` (`test` or
`finding`), pinned `evidence` and independent acceptance reference in `review`.
Test credits additionally require exact `source_sha256` and stable `case_id`, with
evidence `kind: "test", outcome: "passed"`. Repeated snapshots retain the same claim
ID. Rerunning an already credited source/case cannot create a second completion.
Finding credit is keyed by the original finding ID. Scope review remains mandatory.

The audit detects missing ownership/actions, stale items, terminal results without
continuation, unowned blockers, repeated unchanged result/evidence across three
cycles, unsupported/duplicate credit and unmet selection gates. Progress evidence
kinds are `implementation`, `result`, `test`, `source_review`, `disposition` and
`settlement`; source review stays distinct from test credit. Changing status,
next-action prose, timestamps or observation-only zero tallies is not progress. A global `hold` records
`reason`, `evidence` and `blocked_by` IDs. Independent candidates need selection or
an item-specific `defer_reason` with `defer_evidence`; an unrelated hold cannot hide
them. Candidates are planning observations, never authorization to execute work.

```text
python scripts/backlog_handoff.py <existing-ledger.json>
python scripts/backlog_handoff.py <existing-ledger.json> --event <observed-cycle.json> --expected-sha256 <current-ledger-sha256>
python -m unittest discover -s tools/backlog_handoff -p test_handoff.py -v
```

CLI exit 0 means no declared-state defects, 1 means audit findings, 2 means read/write
or input failure. An exact replay of a cycle is idempotent; changed reuse is refused.
The optional writer appends immutable cycles under an exclusive cooperating-writer
lock, compares the expected ledger hash, fsyncs a temporary file and atomically
replaces the ledger. Reopen state after restart. A leftover `.handoff.lock` is a hold
for owner reconciliation; never steal it by age. This is not an adversarial-filesystem
or power-loss durability guarantee. Other ledger writers must honor the same lock
or use their existing transaction system. Failed attempts and prior cycles remain.

Report producers can call `report_hook(ledger, now)` immediately before serialization,
seal construction and reporting completion counts. A failure raises and must prevent
a clean report; preserve its findings and continue independent authorized work.
The coordinator confirmed that previous chat reports were manually assembled; its
first `current_backlog_report.py` was an unexecuted proposal. The integrated local
successor is now `scripts/current_backlog_report.py`. Use a captured successor input,
its expected hash and a new output directory:

```text
python scripts/current_backlog_report.py <captured-ledger.json> <new-output-directory> --expected-sha256 <input-sha256>
python -m unittest discover -s tools/backlog_handoff -p test_report_consumer.py -v
```

The consumer calls `report_hook` before counting or serialization. It additionally
checks the receipt overlay's exact source-ID coverage, every nonempty missing-join
entry (including explanatory/global gaps), source-owner
assignment match, pending status and unmapped publications. A metadata reconciliation
owner cannot silently substitute for product ownership. A failed audit writes
`pending-report.json`, `audit.json`, an exact `ledger.json` snapshot and `result.json`
with every original ID and unresolved work visible. It writes **no `seal.json`** or
clean `report.json`. An accepted declared-state audit writes the clean report and
seal only after the source snapshot, report and audit are persisted; the seal pins
their hashes and both producer/audit source identities. Existing output directories
are refused. A clean handoff may still contain unresolved findings and is not closure.

CLI exits are 0 for a clean declared-state report, 1 for an explicit pending report,
2 for input/write failure. Source imports alone and synthetic consumer tests are not
evidence that real reports used this route. Record an actual invocation and output
identities when adopting it for a report. Manual parent chat and the hosted Atlas
remain outside mechanical enforcement; this consumer is not a dispatch hook and
cannot wake a terminal worker. No permissions, workload controller, STOP change,
scheduler or guaranteed completion is provided. Tests and actual report generation
retain their own resource/owner coordination requirements.
