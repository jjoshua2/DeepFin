# Per-search observation only. This module never selects, encodes or backs up.
# CPU F32 singleton observer. Dispatch, physical completion and acceptance differ.
# A fatal backend/output error produces no successful report: the benchmark
# must treat missing bestmove/metrics as a failed run, not zero-cost success.
import Base

type Run is Data:
  Run{started: Nat, eval_limit: U32, milliseconds: U32, profile: Bool,
    executed: U32, accepted: U32, selection_ms: Nat, encoding_ms: Nat,
    backend_ms: Nat, post_ms: Nat, backup_ms: Nat,
    dispatched: U32, cancelled: U32, failed: U32, rejected: U32, wasted: U32,
    asynchronous: Bool, epoch: U32}

def start(now: Nat, evals: U32, ms: U32, profile: Bool) -> Run:
  Run{now, evals, ms, profile, 0, 0, 0n, 0n, 0n, 0n, 0n, 0, 0, 0, 0, 0, False{}, 0}

def wall(now: Nat, run: Run) -> Nat:
  Run{+started, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Nat.sub(now, started)

def profiling(run: Run) -> Bool:
  Run{+started, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  profile

def has_limit(run: Run) -> Bool:
  Run{+started, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  U32.is_gt(limit, 0)

def exhausted(run: Run) -> Bool:
  Run{+started, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Bool.and(U32.is_gt(limit, 0), U32.is_ge(dispatched, limit))

# Clock reads inside the search are opt-in. Always-on counters only add bounded
# integer operations; IO.now is still read by the existing deadline check.
def clock(profile: Bool) -> IO(Nat):
  match profile:
    case True{}: IO.now()
    case False{}: IO.pure(Nat, 0n)

def selected(run: Run, elapsed: Nat) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Run{began, limit, ms, profile, executed, accepted, Nat.add(selection, elapsed), encoding, backend, post, backup, dispatched, cancelled, failed, rejected, wasted, asynchronous, epoch}

# Call once for a returned model evaluation, and only mark accepted when
# Search.resume actually increments completed. Cached/rule/material replies
# may complete simulations but must pass neural=False here.
def returned(run: Run, +neural: Bool, +before: U32, after: U32,
  enc: Nat, back: Nat, norm: Nat, update: Nat) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  +ok = Bool.and(neural, U32.is_eq(after, U32.inc(before)))
  +bad = Bool.and(neural, Bool.not(ok))
  Run{began, limit, ms, profile, U32.add(executed, Bool.to_u32(neural)), U32.add(accepted, Bool.to_u32(ok)),
    selection, Nat.add(encoding, enc), Nat.add(backend, back), Nat.add(post, norm), Nat.add(backup, update),
    U32.add(dispatched, Bool.to_u32(Bool.and(neural, Bool.not(asynchronous)))), cancelled, failed,
    U32.add(rejected, Bool.to_u32(bad)), U32.add(wasted, Bool.to_u32(bad)), asynchronous, epoch}

# These transitions are driven exactly once by the application's owned state.
# admitted calls are irrevocable callback reservations, not confirmed execution.
def submitted(run: Run, enc: Nat) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Run{began, limit, ms, profile, executed, accepted, selection, Nat.add(encoding, enc), backend, post, backup,
    U32.inc(dispatched), cancelled, failed, rejected, wasted, True{}, epoch}

def cancel_pending(run: Run) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Run{began, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup,
    dispatched, U32.inc(cancelled), failed, rejected, wasted, asynchronous, epoch}

def retired(run: Run, +success: Bool) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  Run{began, limit, ms, profile, U32.add(executed, Bool.to_u32(success)), accepted, selection, encoding, backend, post, backup,
    dispatched, cancelled, U32.add(failed, Bool.to_u32(Bool.not(success))), rejected,
    U32.add(wasted, Bool.to_u32(success)), asynchronous, epoch}

def identified(run: Run, epoch: U32) -> Run:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +old} = run
  Run{began, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup,
    dispatched, cancelled, failed, rejected, wasted, asynchronous, epoch}

# Exact fixed-point JSON; the pinned Base has no F32 string formatter.
def digits(s: String) -> String:
  match s:
    case SNil{}: ""
    case SCon{head, tail}: tail

def decimal_parts(scale: Nat, parts: Nat & Nat) -> String:
  (whole, fraction) = parts
  Nat.show(whole) ++ "." ++ digits(Nat.show(Nat.add(scale, fraction)))

def decimal(+scale: Nat, value: Nat) -> String:
  decimal_parts(scale, Nat.divmod(value, scale))

def seconds(ms: Nat) -> String:
  decimal(1000n, ms)

def rate(rows: U32, elapsed: Nat) -> String:
  match elapsed:
    case 0n: "null"
    case 1n+p: decimal(1000000n, Nat.div(Nat.mul(U32.to_nat(rows), 1000000000n), Nat.add(1n, p)))

def measured(enabled: Bool, ms: Nat) -> String:
  match enabled:
    case True{}: seconds(ms)
    case False{}: "null"

def histogram(rows: U32) -> String:
  match rows:
    case 0: "{}"
    case _: "{\"1\":" ++ U32.show(rows) ++ "}"

def json(now: Nat, completed: U32, stop: U32, run: Run) -> String:
  Run{+began, +limit, +ms, +profile, +executed, +accepted, +selection, +encoding, +backend, +post, +backup, +dispatched, +cancelled, +failed, +rejected, +wasted, +asynchronous, +epoch} = run
  +elapsed = Nat.sub(now, began)
  "{\"schema\":\"deepfin.neural-work.v1\",\"wall_seconds\":" ++ seconds(elapsed)
    ++ ",\"completed_simulations\":" ++ U32.show(completed)
    ++ ",\"forward_calls\":" ++ U32.show(dispatched)
    ++ ",\"dispatched_real_rows\":" ++ U32.show(dispatched)
    ++ ",\"executed_real_rows\":" ++ U32.show(executed)
    ++ ",\"accepted_neural_rows\":" ++ U32.show(accepted)
    ++ ",\"padded_rows\":0,\"failed_forward_rows\":" ++ U32.show(failed)
    ++ ",\"cancelled_rows\":" ++ U32.show(cancelled) ++ ",\"stale_rows\":0,\"failed_rows\":0"
    ++ ",\"rejected_rows\":" ++ U32.show(rejected)
    ++ ",\"executed_wasted_rows\":" ++ U32.show(wasted)
    ++ ",\"unresolved_rows\":" ++ U32.show(U32.sub(dispatched, U32.add(accepted, U32.add(cancelled, rejected))))
    ++ ",\"unconfirmed_forward_rows\":" ++ U32.show(U32.sub(dispatched, U32.add(executed, failed)))
    ++ ",\"search_epoch\":" ++ U32.show(epoch) ++ ",\"useful_eps\":" ++ rate(accepted, elapsed)
    ++ ",\"executed_eps\":" ++ rate(executed, elapsed)
    ++ ",\"padding_fraction\":0,\"real_batch_histogram\":" ++ histogram(dispatched)
    ++ ",\"physical_batch_histogram\":" ++ histogram(dispatched)
    ++ ",\"neural_budget\":" ++ U32.show(limit) ++ ",\"movetime_ms\":" ++ U32.show(ms)
    ++ ",\"stop_code\":" ++ U32.show(stop)
    ++ ",\"clock_resolution_seconds\":0.001,\"phase_seconds\":{\"selection\":null,\"encoding\":" ++ measured(profile, encoding)
    ++ ",\"queue_wait\":null,\"h2d\":null,\"gpu\":null,\"d2h\":null,\"backup\":" ++ measured(profile, backup)
    ++ "},\"selection_and_cached_backup_seconds\":" ++ measured(profile, selection)
    ++ ",\"backend_and_transport_seconds\":" ++ measured(Bool.and(profile, Bool.not(asynchronous)), backend)
    ++ ",\"normalization_and_diagnostics_seconds\":" ++ measured(profile, post) ++ "}"


def render(now: Nat, completed: U32, stop: U32, run: Run) -> String:
  "info string neural_work " ++ json(now, completed, stop, run)

# Distinct record name: never masquerade as the next search's decision report.
# wall_seconds includes physical retirement here, not just time to bestmove.
def retirement(now: Nat, completed: U32, stop: U32, run: Run) -> String:
  "info string neural_retired " ++ json(now, completed, stop, run)
