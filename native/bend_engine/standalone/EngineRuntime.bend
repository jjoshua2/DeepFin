# Application-owned metadata around the existing linear neural cache. The native
# worker owns only a copied tensor; it never sees the root, tree or legal entries.
import Base
import ../session_probe/Search.bend as Search
import ./NativeEvaluation.bend as Native
import ./Position.bend as Pos
import ./Policy.bend as Policy
import ./EvaluationInput.bend as Evaluation
import ./ModelInput.bend as Input
import ./LogitReply.bend as Logits
import ./NeuralWork.bend as Work
import ./Async.bend as Async

type Retirement is Data:
  NoRetirement{}
  Retiring{id: Async.Identity, run: Work.Run, completed: U32, stop: U32}

type Control is Data:
  Control{epoch: U32, asynchronous: Bool, retirement: Retirement}

type Cache is Type:
  Cache{native: Native.Cache, control: Control}

type Result is Type:
  Result{cache: Cache, table: Array<U64>, reply: Search.Reply, encoding_ms: Nat, backend_ms: Nat, post_ms: Nat}

type Pending is Data:
  Pending{id: Async.Identity, ticket: Search.Ticket, game: Pos.Game, entries: List<&2, Policy.Entry>}

type Submission is Type:
  Submission{cache: Cache, table: Array<U64>, pending: Pending, encoding_ms: Nat}

def cache(model: U32, diagnostics: Bool, asynchronous: Bool) -> Cache:
  Cache{Native.cache(model, diagnostics), Control{0, asynchronous, NoRetirement{}}}

def enabled(cache: Cache) -> Cache & Bool:
  Cache{native, Control{epoch, +asynchronous, retirement}} = cache
  (Cache{native, Control{epoch, asynchronous, retirement}}, asynchronous)

# A new search owns a fresh epoch, including zero-forward and queued searches.
# No wraparound and no reset on position/ucinewgame. Tokens are separate IDs.
def next(cache: Cache) -> IO(Cache & U32):
  Cache{native, Control{epoch, asynchronous, retirement}} = cache
  match epoch:
    case 4294967295: IO.die(Cache & U32, 2, "search epoch exhausted; restart required")
    case _:
      +fresh = U32.inc(epoch)
      IO.pure(Cache & U32, (Cache{native, Control{fresh, asynchronous, retirement}}, fresh))

def wrap(control: Control, r: Native.Result) -> Result:
  Native.Result{native, table, reply, enc, back, post} = r
  Result{Cache{native, control}, table, reply, enc, back, post}

def material(cache: Cache, r: Array<U64> & Search.Reply) -> Result:
  (table, reply) = r
  Result{cache, table, reply, 0n, 0n, 0n}

def evaluate(cache: Cache, timing: Bool, model: U32, root: Pos.Game, game: Pos.Game,
  ticket: Search.Ticket, table: Array<U64>) -> IO(Result):
  Cache{native, control} = cache
  do IO<Result>:
    r : Native.Result <- Native.evaluate(native, timing, model, root, game, ticket, table)
    return wrap(control, r)

def submitted(control: Control, diagnostics: Bool, began: Nat, ended: Nat,
  +ticket: Search.Ticket, game: Pos.Game, entries: List<&2, Policy.Entry>, maps: Policy.Maps,
  table: Array<U64>, output: Array<F32>, r: Async.AsyncSubmitted) -> Submission:
  Search.Ticket{epoch, request, node, b, moves, needs} = ticket
  Async.AsyncSubmitted{input, token} = r
  Submission{Cache{Native.Ready{maps, input, output, diagnostics}, control}, table,
    Pending{Async.Identity{token, epoch, request, node}, ticket, game, entries}, Nat.sub(ended, began)}

def prepared(control: Control, timing: Bool, diagnostics: Bool, began: Nat, game: Pos.Game,
  +ticket: Search.Ticket, output: Array<F32>, p: Evaluation.Preparation) -> IO(Submission):
  Search.Ticket{epoch, request, node, b, moves, needs} = ticket
  match p:
    case Evaluation.Invalid{table, maps}: IO.die(Submission, 2, "async input preparation failed")
    case Evaluation.Ready{Evaluation.Prepared{maps, Input.Input{table, values, channels}, entries}}:
      do IO<Submission>:
        ended : Nat <- Work.clock(timing)
        r : Async.AsyncSubmitted <- Async.Async.submit(epoch, request, node, U32.mul(channels, 64), values)
        return submitted(control, diagnostics, began, ended, ticket, game, entries, maps, table, output, r)

def submit(cache: Cache, +timing: Bool, +model: U32, +game: Pos.Game,
  ticket: Search.Ticket, table: Array<U64>) -> IO(Submission):
  Cache{native, control} = cache
  match native:
    case Native.Disabled{}: IO.die(Submission, 2, "async requires native model")
    case Native.Ready{maps, input, output, diagnostics}:
      do IO<Submission>:
        began : Nat <- Work.clock(timing)
        prepared(control, timing, diagnostics, began, game, ticket, output,
          Evaluation.prepare_into(Native.layout(model), Native.version(model), game, table, maps, Array.size(F32, input)))

def polled(control: Control, maps: Policy.Maps, input: Array<F32>, +diagnostics: Bool,
  r: Async.AsyncPolled) -> Cache & U32:
  Async.AsyncPolled{output, status} = r
  (Cache{Native.Ready{maps, input, output, diagnostics}, control}, status)

def poll(cache: Cache, id: Async.Identity) -> IO(Cache & U32):
  Cache{native, control} = cache
  match native:
    case Native.Disabled{}: IO.die(Cache & U32, 2, "missing async model cache")
    case Native.Ready{maps, input, output, diagnostics}:
      Async.Identity{token, epoch, request, node} = id
      do IO<Cache & U32>:
        r : Async.AsyncPolled <- Async.Async.poll(token, epoch, request, node, output)
        return polled(control, maps, input, diagnostics, r)

def converted(control: Control, maps: Policy.Maps, input: Array<F32>, +diagnostics: Bool,
  +timing: Bool, +began: Nat, root: Pos.Game, game: Pos.Game, entries: List<&2, Policy.Entry>,
  table: Array<U64>, r: Logits.Converted) -> IO(Result):
  Logits.Converted{output, reply} = r
  do IO<Result>:
    r : Array<U64> & Search.Reply <- Native.notice(root, game, entries, table, diagnostics, reply)
    ended : Nat <- Work.clock(timing)
    return wrap(control, Native.timed_result(Native.Ready{maps, input, output, diagnostics}, began, began, began, ended, r))

def complete(cache: Cache, +timing: Bool, root: Pos.Game, pending: Pending, table: Array<U64>) -> IO(Result):
  Cache{native, control} = cache
  match native:
    case Native.Disabled{}: IO.die(Result, 2, "missing async completion cache")
    case Native.Ready{maps, input, output, diagnostics}:
      Pending{id, ticket, game, +entries} = pending
      do IO<Result>:
        began : Nat <- Work.clock(timing)
        converted(control, maps, input, diagnostics, timing, began, root, game, entries, table,
          Logits.convert_buffer(ticket, entries, 1861, Array.size(F32, output)))

def retiring(cache: Cache, id: Async.Identity, run: Work.Run, completed: U32, stop: U32) -> Cache:
  Cache{native, Control{epoch, asynchronous, retirement}} = cache
  Cache{native, Control{epoch, asynchronous, Retiring{id, run, completed, stop}}}

def retirement_done(native: Native.Cache, epoch: U32, asynchronous: Bool, id: Async.Identity,
  run: Work.Run, completed: U32, stop: U32, status: U32) -> IO(Cache & Bool):
  match status:
    case 1: IO.pure(Cache & Bool, (Cache{native, Control{epoch, asynchronous, Retiring{id, run, completed, stop}}}, True{}))
    case 3:
      do IO<Cache & Bool>:
        now : Nat <- IO.now()
        IO.print(Work.retirement(now, completed, stop, Work.retired(run, True{})))
        return (Cache{native, Control{epoch, asynchronous, NoRetirement{}}}, False{})
    case 4:
      do IO<Cache & Bool>:
        now : Nat <- IO.now()
        IO.print(Work.retirement(now, completed, stop, Work.retired(run, False{})))
        IO.die(Cache & Bool, 2, "cancelled native execution failed; worker poisoned")
    case _: IO.die(Cache & Bool, 2, "invalid native retirement identity or disposition")

# Nonblocking; a replacement search can own its own root/clock while this drain
# prevents dispatch. Late output can never reach either the old or new tree.
def retire(cache: Cache) -> IO(Cache & Bool):
  Cache{native, Control{epoch, asynchronous, retirement}} = cache
  match retirement:
    case NoRetirement{}: IO.pure(Cache & Bool, (Cache{native, Control{epoch, asynchronous, NoRetirement{}}}, False{}))
    case Retiring{+id, run, completed, stop}:
      do IO<Cache & Bool>:
        status : U32 <- Async.drain_identity(id)
        retirement_done(native, epoch, asynchronous, id, run, completed, stop, status)

def shutdown_checked(r: Cache & Bool) -> IO(Unit):
  (discarded, busy) = r
  match busy:
    case True{}: IO.die(Unit, 2, "native shutdown did not retire physical work")
    case False{}: IO.pure(Unit, Unit{})

def shutdown(cache: Cache) -> IO(Unit):
  do IO<Unit>:
    Async.Async.shutdown()
    r : Cache & Bool <- retire(cache)
    shutdown_checked(r)
