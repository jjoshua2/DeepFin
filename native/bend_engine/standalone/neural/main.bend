# Opt-in direct native inference. Bend owns every chess/evaluation decision.
# The original material executable is unchanged. Synchronous forward calls are
# NOT interruptible: this gate claims no responsive stop during a model kernel.
import Base
import ../main.bend as App
import ../Position.bend as Pos
import ../Protocol.bend as Protocol
import ../Text.bend as Text
import ../Tables.bend as Tables
import ../Rules.bend as Rules
import ../SearchHistory.bend as History
import ../HistoryEncoding.bend as H
import ../ModelInput.bend as Input
import ../EvaluationInput.bend as Evaluation
import ../Policy.bend as Policy
import ../../session_probe/Search.bend as Search
import ./Native.bend as Native
import ./Reply.bend as Reply
import ./Trace.bend as Trace

type Model is Type:
  Model{layout: H.Layout, version: Input.Version, maps: Policy.Maps, sequence: U32, trace: Bool}

type Next is Type:
  Next{engine: App.State, model: Model}

def ticket_node(ticket: Search.Ticket) -> U32:
  Search.Ticket{e, q, n, b, ms, needs} = ticket
  n

def paired(model: Model, engine: App.State) -> Next:
  Next{engine, model}

def applied(root: Pos.Game, table: Array<U64>, tree: Search.Tree, ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model,
  r: Search.Reply) -> IO(Next):
  Search.Reply{e, q, n, +status, w, d, l, count, policy} = r
  match status:
    case 0:
      IO.pure(Next, Next{App.ready_state(root, table,
        Search.resume(tree, ticket, Search.Reply{e, q, n, status, w, d, l, count, policy}),
        budget, inf, started, duration, fallback), model})
    case _: IO.die(Next, 2, "native model output rejected; no diagnostic fallback")

def forwarded(root: Pos.Game, game: Pos.Game, table: Array<U64>, tree: Search.Tree, +ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model,
  entries: List<&2, Policy.Entry>, +logits: List<&2, F32>) -> IO(Next):
  Model{layout, version, maps, sequence, +trace} = model
  do IO<Next>:
    Trace.logits(trace, logits, 0)
    response : Search.Reply <- Trace.reply(trace, Reply.convert(ticket, entries, logits))
    applied(root, table, tree, ticket, budget, inf, started, duration, fallback,
      Model{layout, version, maps, U32.inc(sequence), trace}, response)

def execute(root: Pos.Game, +game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32,
  layout: H.Layout, version: Input.Version, +sequence: U32, +trace: Bool,
  p: Evaluation.Preparation) -> IO(Next):
  match p:
    case Evaluation.Invalid{table, maps}: IO.die(Next, 2, "native evaluation preparation rejected")
    case Evaluation.Ready{Evaluation.Prepared{maps, Input.Input{table, values, +channels}, +entries}}:
      do IO<Next>:
        values : Array<F32> <- Trace.input(trace, sequence, ticket, game, entries, channels, values)
        logits : List<&2, F32> <- Native.Native.forward(sequence, Native.input_words(U32.to_nat(U32.mul(channels, 64)), 0, (values, Nil{})))
        forwarded(root, game, table, tree, ticket, budget, inf, started, duration, fallback,
          Model{layout, version, maps, sequence, trace}, entries, logits)

def adjudicated(root: Pos.Game, +game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32,
  model: Model, r: Array<U64> & Rules.Facts) -> IO(Next):
  Model{+layout, +version, maps, sequence, trace} = model
  (table, Rules.Facts{code, count, half}) = r
  match code:
    case 0: execute(root, game, tree, ticket, budget, inf, started, duration, fallback,
      layout, version, sequence, trace, Evaluation.prepare(layout, version, game, table, maps))
    case _:
      do IO<Next>:
        App.rule_notice(code, ticket_node(ticket), count, half)
        return Next{App.ready_state(root, table, Search.resume(tree, ticket, App.rule_reply(ticket)),
          budget, inf, started, duration, fallback), Model{layout, version, maps, sequence, trace}}

def valid_history(ok: Bool, root: Pos.Game, +game: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model) -> IO(Next):
  match ok:
    case False{}: IO.die(Next, 2, "native search history mismatch")
    case True{}:
      Search.Ticket{e, q, n, b, moves, needs} = ticket
      adjudicated(root, game, tree, ticket, budget, inf, started, duration, fallback,
        model, Rules.automatic(game, moves, table))

def rebuilt(root: Pos.Game, table: Array<U64>, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model, r: History.Rebuilt) -> IO(Next):
  History.Rebuilt{tree, game, ok} = r
  valid_history(ok, root, game, table, tree, ticket, budget, inf, started, duration, fallback, model)

def evaluate(needed: Bool, +root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model) -> IO(Next):
  match needed:
    case False{}: IO.pure(Next, Next{App.ready_state(root, table, tree, budget, inf, started, duration, fallback), model})
    case True{}:
      Search.Ticket{e, q, id, board, ms, needs} = ticket
      rebuilt(root, table, ticket, budget, inf, started, duration, fallback, model, History.rebuild(root, id, board, tree))

def prepared(root: Pos.Game, budget: U32, inf: Bool, started: Nat, duration: U32,
  fallback: U32, model: Model, p: Search.Prepared) -> IO(Next):
  Search.Prepared{table, tree, +ticket} = p
  Search.Ticket{e, q, id, b, ms, needs} = ticket
  evaluate(needs, root, table, tree, ticket, budget, inf, started, duration, fallback, model)

def result_label(n: U32) -> String:
  match n:
    case 0: "unsearched legal fallback; native model PUCT cpu"
    case _: "native model PUCT cpu"

def output(root: Pos.Game, table: Array<U64>, key: U32, +n: U32, stop: U32, model: Model) -> IO(Next):
  do IO<Next>:
    IO.print("info nodes " ++ U32.show(n) ++ " string " ++ result_label(n) ++ "; stop_code " ++ U32.show(stop))
    IO.print("bestmove " ++ App.move_key(key))
    return Next{App.State{root, table, App.Idle{}, False{}}, model}

def output_or_hold(hold: Bool, root: Pos.Game, table: Array<U64>, key: U32, n: U32, stop: U32, model: Model) -> IO(Next):
  match hold:
    case True{}: IO.pure(Next, Next{App.State{root, table, App.Held{key, n, stop}, False{}}, model})
    case False{}: output(root, table, key, n, stop, model)

def picked(hold: Bool, root: Pos.Game, table: Array<U64>, +fallback: U32, n: U32, stop: U32,
  model: Model, p: App.Pick) -> IO(Next):
  App.Pick{pool, +key, visits} = p
  +missing = U32.is_eq(key, 4294967295)
  do IO<Next>:
    App.fallback_notice(Bool.and(missing, Bool.not(U32.is_eq(fallback, 4294967295))))
    output_or_hold(hold, root, table, Bool.pick(U32, missing, fallback, key), n, stop, model)

def finished(hold: Bool, root: Pos.Game, table: Array<U64>, fallback: U32, model: Model, t: Search.Tree) -> IO(Next):
  Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = t
  picked(hold, root, table, fallback, n, stop, model, App.pick(U32.to_nat(used), 0, App.Pick{pool, 4294967295, 0}))

def finish(hold: Bool, root: Pos.Game, table: Array<U64>, fallback: U32, model: Model, t: Search.Tree) -> IO(Next):
  Model{layout, version, maps, sequence, +trace} = model
  do IO<Next>:
    t : Search.Tree <- Trace.tree(trace, t)
    finished(hold, root, table, fallback, Model{layout, version, maps, sequence, trace}, t)

def progress(done: Bool, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32, model: Model) -> IO(Next):
  match done:
    case True{}: finish(inf, root, table, fallback, model, tree)
    case False{}: prepared(root, budget, inf, started, duration, fallback, model, Search.prepare(table, tree))

def elapsed(now: Nat, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +budget: U32, inf: Bool, +started: Nat, +duration: U32, fallback: U32, model: Model) -> IO(Next):
  Search.Tree{pool, used, cap, +n, pending, epoch, request, +code, limit} = tree
  done = Bool.or(U32.is_ge(n, budget), Bool.or(U32.is_gt(code, 0),
    Bool.and(U32.is_gt(duration, 0), Nat.is_ge(Nat.sub(now, started), U32.to_nat(duration)))))
  progress(done, root, table, Search.Tree{pool, used, cap, n, pending, epoch, request, code, limit},
    budget, inf, started, duration, fallback, model)

def tick(engine: App.State, model: Model) -> IO(Next):
  App.State{root, table, mode, quit} = engine
  match mode:
    case App.Running{tree, budget, inf, started, duration, fallback}:
      do IO<Next>:
        now : Nat <- IO.now()
        elapsed(now, root, table, tree, budget, inf, started, duration, fallback, model)
    case _:
      do IO<Next>:
        IO.sleep(10)
        return Next{App.State{root, table, mode, quit}, model}

def stop_search(engine: App.State, model: Model) -> IO(Next):
  App.State{root, table, mode, quit} = engine
  match mode:
    case App.Idle{}: IO.pure(Next, Next{App.State{root, table, App.Idle{}, quit}, model})
    case App.Held{key, n, code}: output(root, table, key, n, code, model)
    case App.Running{tree, budget, inf, started, duration, fallback}: finish(False{}, root, table, fallback, model, tree)

def command(code: U32, args: List<&2, String>, engine: App.State, model: Model) -> IO(Next):
  match code:
    case 1:
      do IO<Next>:
        IO.print("id name DeepFin Bend standalone native neural experimental")
        IO.print("id author DeepFin contributors")
        IO.print("uciok")
        return Next{engine, model}
    case 5: stop_search(engine, model)
    case _:
      do IO<Next>:
        engine : App.State <- App.handle(code, args, engine)
        return Next{engine, model}

def dispatch(words: List<&2, String>, engine: App.State, model: Model) -> IO(Next):
  match words:
    case Nil{}: IO.pure(Next, Next{engine, model})
    case Con{head, args}: command(Protocol.command(head), args, engine, model)

def packet(p: App.Packet, engine: App.State, model: Model) -> IO(Next):
  App.Packet{kind, text} = p
  match kind:
    case 0: tick(engine, model)
    case 1: dispatch(Text.words(text), engine, model)
    case _:
      do IO<Next>:
        engine : App.State <- App.packet(App.Packet{kind, text}, engine)
        return Next{engine, model}

def serve(fuel: Nat, s: Next) -> IO(Unit):
  match fuel:
    case 0n: IO.die(Unit, 2, "neural event budget exhausted")
    case 1n+p:
      Next{App.State{root, table, mode, quit}, model} = s
      match quit:
        case True{}: Native.Native.close()
        case False{}:
          do IO<Unit>:
            input : App.Packet <- App.Wire.poll()
            next : Next <- packet(input, App.State{root, table, mode, False{}}, model)
            serve(p, next)

def launch(layout: H.Layout, version: Input.Version, trace: Bool) -> IO(Unit):
  serve(1000000000n, Next{App.State{Pos.start(), Tables.build(), App.Idle{}, False{}},
    Model{layout, version, Policy.build(), 1, trace}})

def spec(words: List<&2, U32>, trace: Bool) -> IO(Unit):
  match words:
    case Con{0, Con{0, Con{146, Con{batch, Nil{}}}}}: launch(H.Root{}, Input.V1{}, trace)
    case Con{0, Con{1, Con{175, Con{batch, Nil{}}}}}: launch(H.Root{}, Input.V2Threats{}, trace)
    case Con{1, Con{0, Con{146, Con{batch, Nil{}}}}}: launch(H.RootLegacyMeta{}, Input.V1{}, trace)
    case Con{1, Con{1, Con{175, Con{batch, Nil{}}}}}: launch(H.RootLegacyMeta{}, Input.V2Threats{}, trace)
    case _: IO.die(Unit, 2, "invalid compiled model encoding contract")

def opened(code: U32, trace: Bool) -> IO(Unit):
  match code:
    case 0:
      do IO<Unit>:
        words : List<&2, U32> <- Native.Native.spec()
        spec(words, trace)
    case _: IO.die(Unit, 2, "native model startup failed; no diagnostic fallback")

def trace_value(r: Result<&1, &1, U32 & String, String>) -> Bool:
  match r:
    case Done{text}: String.eq(text, "1")
    case Fail{err}: False{}

def main() -> IO(Unit):
  do IO<Unit>:
    path : String <- IO.try(String, IO.get_env("DEEPFIN_BEND_PACKAGE"))
    trace : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_BEND_TRACE")
    code : U32 <- Native.Native.open(path)
    opened(code, trace_value(trace))
