# Per-search observation only. This module never selects, encodes or backs up.
# The current native backend is synchronous, CPU F32 and exactly batch one.
# A fatal backend/output error produces no successful report: the benchmark
# must treat missing bestmove/metrics as a failed run, not zero-cost success.
import Base

type Run is Data:
  Run{started: Nat, eval_limit: U32, milliseconds: U32, profile: Bool,
    executed: U32, accepted: U32, selection_ms: Nat, encoding_ms: Nat,
    backend_ms: Nat, post_ms: Nat, backup_ms: Nat}

def start(now: Nat, evals: U32, ms: U32, profile: Bool) -> Run:
  Run{now, evals, ms, profile, 0, 0, 0n, 0n, 0n, 0n, 0n}

def wall(now: Nat, run: Run) -> Nat:
  Run{started, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  Nat.sub(now, started)

def profiling(run: Run) -> Bool:
  Run{started, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  profile

def has_limit(run: Run) -> Bool:
  Run{started, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  U32.is_gt(limit, 0)

def exhausted(run: Run) -> Bool:
  Run{started, +limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  Bool.and(U32.is_gt(limit, 0), U32.is_ge(executed, limit))

# Clock reads inside the search are opt-in. Always-on counters only add bounded
# integer operations; IO.now is still read by the existing deadline check.
def clock(profile: Bool) -> IO(Nat):
  match profile:
    case True{}: IO.now()
    case False{}: IO.pure(Nat, 0n)

def selected(run: Run, elapsed: Nat) -> Run:
  Run{began, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  Run{began, limit, ms, profile, executed, accepted, Nat.add(selection, elapsed), encoding, backend, post, backup}

# Call once for a returned model evaluation, and only mark accepted when
# Search.resume actually increments completed. Cached/rule/material replies
# may complete simulations but must pass neural=False here.
def returned(run: Run, +neural: Bool, +before: U32, after: U32,
  enc: Nat, back: Nat, norm: Nat, update: Nat) -> Run:
  Run{began, limit, ms, profile, executed, accepted, selection, encoding, backend, post, backup} = run
  accepted = U32.add(accepted, Bool.to_u32(Bool.and(neural, U32.is_eq(after, U32.inc(before)))))
  Run{began, limit, ms, profile, U32.add(executed, Bool.to_u32(neural)), accepted,
    selection, Nat.add(encoding, enc), Nat.add(backend, back), Nat.add(post, norm), Nat.add(backup, update)}

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

def render(now: Nat, completed: U32, stop: U32, run: Run) -> String:
  Run{began, limit, ms, +profile, +executed, +accepted, selection, encoding, backend, post, backup} = run
  +elapsed = Nat.sub(now, began)
  "info string neural_work {\"schema\":\"deepfin.neural-work.v1\",\"wall_seconds\":" ++ seconds(elapsed)
    ++ ",\"completed_simulations\":" ++ U32.show(completed)
    ++ ",\"forward_calls\":" ++ U32.show(executed)
    ++ ",\"dispatched_real_rows\":" ++ U32.show(executed)
    ++ ",\"executed_real_rows\":" ++ U32.show(executed)
    ++ ",\"accepted_neural_rows\":" ++ U32.show(accepted)
    ++ ",\"padded_rows\":0,\"failed_forward_rows\":0,\"cancelled_rows\":0,\"stale_rows\":0,\"failed_rows\":0"
    ++ ",\"rejected_rows\":" ++ U32.show(U32.sub(executed, accepted))
    ++ ",\"executed_wasted_rows\":" ++ U32.show(U32.sub(executed, accepted))
    ++ ",\"unresolved_rows\":0,\"useful_eps\":" ++ rate(accepted, elapsed)
    ++ ",\"executed_eps\":" ++ rate(executed, elapsed)
    ++ ",\"padding_fraction\":0,\"real_batch_histogram\":" ++ histogram(executed)
    ++ ",\"physical_batch_histogram\":" ++ histogram(executed)
    ++ ",\"neural_budget\":" ++ U32.show(limit) ++ ",\"movetime_ms\":" ++ U32.show(ms)
    ++ ",\"stop_code\":" ++ U32.show(stop)
    ++ ",\"clock_resolution_seconds\":0.001,\"phase_seconds\":{\"selection\":null,\"encoding\":" ++ measured(profile, encoding)
    ++ ",\"queue_wait\":null,\"h2d\":null,\"gpu\":null,\"d2h\":null,\"backup\":" ++ measured(profile, backup)
    ++ "},\"selection_and_cached_backup_seconds\":" ++ measured(profile, selection)
    ++ ",\"backend_and_transport_seconds\":" ++ measured(profile, backend)
    ++ ",\"normalization_and_diagnostics_seconds\":" ++ measured(profile, post) ++ "}"
