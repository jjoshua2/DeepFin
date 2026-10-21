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