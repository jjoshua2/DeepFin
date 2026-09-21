# Native-model entry point. Reuse the standalone UCI parser/state, chess, tree,
# histories and finish logic; replace ONLY the selected-leaf evaluation step.
# Model calls are synchronous. No readiness/stop guarantee during a native call.
import Base
import ../../session_probe/Search.bend as Search
import ../main.bend as Engine
import ../Position.bend as Pos
import ../Rules.bend as Rules
import ../SearchHistory.bend as History
import ../Tables.bend as Tables
import ./Backend.bend as Backend


def accepted(root: Pos.Game, tree: Search.Tree, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, r: Backend.Outcome) -> Engine.State & Backend.Context:
  Backend.Outcome{table, context, reply} = r
  (Engine.ready_state(root, table, Search.resume(tree, ticket, reply), budget, inf, started, duration, fallback), context)

def adjudicated(root: Pos.Game, game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, context: Backend.Context,
  r: Array<U64> & Rules.Facts) -> IO(Engine.State & Backend.Context):
  Search.Ticket{epoch, request, id, board, moves, needed} = ticket
  (table, Rules.Facts{+code, count, half}) = r
  match code:
    case 0:
      do IO<Engine.State & Backend.Context>:
        result : Backend.Outcome <- Backend.evaluate(game, ticket, table, context)
        return accepted(root, tree, ticket, budget, inf, started, duration, fallback, result)
    case _:
      do IO<Engine.State & Backend.Context>:
        Engine.rule_notice(code, id, count, half)
        return (Engine.ready_state(root, table, Engine.rule_result(code, tree, ticket), budget, inf, started, duration, fallback), context)

def checked(ok: Bool, root: Pos.Game, +game: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Nat, duration: U32,
  fallback: U32, context: Backend.Context) -> IO(Engine.State & Backend.Context):
  match ok:
    case False{}: IO.die(Engine.State & Backend.Context, 2, "invalid native search history")
    case True{}:
      Search.Ticket{epoch, request, id, board, moves, needed} = ticket
      adjudicated(root, game, tree, ticket, budget, inf, started, duration, fallback, context,
        Rules.automatic(game, moves, table))

def rebuilt(root: Pos.Game, table: Array<U64>, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, context: Backend.Context,
  h: History.Rebuilt) -> IO(Engine.State & Backend.Context):
  History.Rebuilt{tree, game, valid} = h
  checked(valid, root, game, table, tree, ticket, budget, inf, started, duration, fallback, context)

def needed(yes: Bool, +root: Pos.Game, table: Array<U64>, tree: Search.Tree, +ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32,
  context: Backend.Context) -> IO(Engine.State & Backend.Context):
  match yes:
    case False{}: IO.pure(Engine.State & Backend.Context,
      (Engine.ready_state(root, table, tree, budget, inf, started, duration, fallback), context))
    case True{}:
      Search.Ticket{epoch, request, id, board, moves, wanted} = ticket
      rebuilt(root, table, ticket, budget, inf, started, duration, fallback, context, History.rebuild(root, id, board, tree))

def prepared(root: Pos.Game, budget: U32, infinite: Bool, started: Nat, duration: U32,
  fallback: U32, context: Backend.Context, p: Search.Prepared) -> IO(Engine.State & Backend.Context):
  Search.Prepared{table, tree, +ticket} = p
  Search.Ticket{epoch, request, id, board, moves, yes} = ticket
  needed(yes, root, table, tree, ticket, budget, infinite, started, duration, fallback, context)

def progress(done: Bool, root: Pos.Game, table: Array<U64>, tree: Search.Tree, budget: U32,
  infinite: Bool, started: Nat, duration: U32, fallback: U32, context: Backend.Context) -> IO(Engine.State & Backend.Context):
  match done:
    case True{}:
      do IO<Engine.State & Backend.Context>:
        state : Engine.State <- Engine.finish(infinite, root, table, fallback, tree)
        return (state, context)
    case False{}: prepared(root, budget, infinite, started, duration, fallback, context, Search.prepare(table, tree))

def elapsed(now: Nat, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +budget: U32, inf: Bool, +started: Nat, +duration: U32, fallback: U32,
  context: Backend.Context) -> IO(Engine.State & Backend.Context):
  Search.Tree{pool, used, cap, +n, pending, epoch, request, +code, limit} = tree
  done = Bool.or(U32.is_ge(n, budget), Bool.or(U32.is_gt(code, 0),
    Bool.and(U32.is_gt(duration, 0), Nat.is_ge(Nat.sub(now, started), U32.to_nat(duration)))))
  progress(done, root, table, Search.Tree{pool, used, cap, n, pending, epoch, request, code, limit},
    budget, inf, started, duration, fallback, context)

def tick(root: Pos.Game, table: Array<U64>, mode: Engine.Mode, context: Backend.Context) -> IO(Engine.State & Backend.Context):
  match mode:
    case Engine.Running{tree, budget, inf, started, duration, fallback}:
      do IO<Engine.State & Backend.Context>:
        now : Nat <- IO.now()
        elapsed(now, root, table, tree, budget, inf, started, duration, fallback, context)
    case _:
      do IO<Engine.State & Backend.Context>:
        s : Engine.State <- Engine.tick_mode(root, table, mode)
        return (s, context)

def packet(p: Engine.Packet, state: Engine.State, context: Backend.Context) -> IO(Engine.State & Backend.Context):
  Engine.Packet{kind, text} = p
  match kind:
    case 0:
      Engine.State{root, table, mode, quit} = state
      tick(root, table, mode, context)
    case _:
      do IO<Engine.State & Backend.Context>:
        next : Engine.State <- Engine.packet(Engine.Packet{kind, text}, state)
        return (next, context)

def serve(fuel: Nat, r: Engine.State & Backend.Context) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string event budget exhausted")
    case 1n+p:
      (Engine.State{root, table, mode, quit}, context) = r
      match quit:
        case True{}: IO.pure(Unit, Unit{})
        case False{}:
          do IO<Unit>:
            input : Engine.Packet <- Engine.Wire.poll()
            next : Engine.State & Backend.Context <- packet(input, Engine.State{root, table, mode, False{}}, context)
            serve(p, next)

def main() -> IO(Unit):
  do IO<Unit>:
    context : Backend.Context <- Backend.start()
    IO.print("info string backend native AOTI CPU; Bend input and legal softmax; synchronous inference")
    serve(1000000000n, (Engine.State{Pos.start(), Tables.build(), Engine.Idle{}, False{}}, context))
    Backend.DfModel.close()
