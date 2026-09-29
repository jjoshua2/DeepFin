# Diagnostic engine application in Bend, not a wrapper around a Python engine.
import Base
import ../legal_probe/Chess.bend as Chess
import ../session_probe/Search.bend as Search
import ./Tables.bend as Tables
import ./Text.bend as Text
import ./Position.bend as Pos
import ./Protocol.bend as Protocol
import ./Rules.bend as Rules
import ./SearchHistory.bend as History
import ./EncodingDiagnostic.bend as EncodingDiagnostic
import ./ModelInputDiagnostic.bend as ModelInputDiagnostic
import ./PolicyDiagnostic.bend as PolicyDiagnostic
import ./EvaluationDiagnostic.bend as EvaluationDiagnostic
import ./EngineRuntime.bend as NativeEvaluation
import ./Async.bend as Async
import ./NeuralWork.bend as Work
import ./ModelBuffers.bend as Buffers

type Packet is Data:
  Packet{kind: U32, text: String}

def Wire.poll() -> IO(Packet):
  import "./poll.c"

type Mode is Type:
  Idle{}
  Running{tree: Search.Tree, budget: U32, infinite: Bool, started: Work.Run, duration: U32, fallback: U32}
  Awaiting{tree: Search.Tree, pending: NativeEvaluation.Pending, budget: U32, infinite: Bool, started: Work.Run, duration: U32, fallback: U32}
  Held{key: U32, completed: U32, stop: U32, report: String}

type State is Type:
  State{cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, mode: Mode, quit: Bool}

def message(s: State, text: String) -> IO(State):
  do IO<State>:
    IO.print("info string " ++ text)
    return s

def word(+b: U64) -> String:
  U32.show(U64.high(b)) ++ " " ++ U32.show(U64.low(b))

def board_text(b: Chess.Board) -> String:
  Chess.Board{p, n, bi, r, q, k, w, bl, turn, rights, ep} = b
  word(p) ++ " " ++ word(n) ++ " " ++ word(bi) ++ " " ++ word(r) ++ " " ++ word(q) ++ " " ++ word(k)
    ++ " " ++ word(w) ++ " " ++ word(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def history_text(hist: List<&2, Pos.Past>) -> IO(Unit):
  match hist:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{Pos.Past{b, half, full, move}, rest}:
      do IO<Unit>:
        IO.print("info string history " ++ board_text(b) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full) ++ " " ++ Pos.move_text(move))
        history_text(rest)

def debug(s: State) -> IO(State):
  State{cache, g, table, mode, quit} = s
  Pos.Game{+b, +half, +full, +hist} = g
  do IO<State>:
    IO.print("info string state " ++ board_text(b) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full)
      ++ " " ++ Nat.show(List.length(&2, Pos.Past, hist)))
    history_text(hist)
    IO.print("info string state_end")
    return State{cache, Pos.Game{b, half, full, hist}, table, mode, quit}

def material(+b: Chess.Board, side: U32) -> U32:
  +own = Chess.color(b, side)
  U32.add(U64.popcount(U64.and(Chess.get_pawns(b), own)),
    U32.add(U32.mul(3, U64.popcount(U64.and(Chess.get_knights(b), own))),
      U32.add(U32.mul(3, U64.popcount(U64.and(Chess.get_bishops(b), own))),
        U32.add(U32.mul(5, U64.popcount(U64.and(Chess.get_rooks(b), own))),
          U32.mul(9, U64.popcount(U64.and(Chess.get_queens(b), own)))))))

def win_probability(+b: Chess.Board) -> F32:
  +us = F32.from_nat(U32.to_nat(material(b, Chess.get_turn(b))))
  +them = F32.from_nat(U32.to_nat(material(b, U32.xor(Chess.get_turn(b), 1))))
  F32.mul(0.5, F32.add(1.0, F32.div(F32.sub(us, them), F32.add(1.0, F32.add(us, them)))))

def reply_for(ticket: Search.Ticket) -> Search.Reply:
  Search.Ticket{epoch, request, id, b, moves, reply} = ticket
  +win = win_probability(b)
  +count = U32.from_nat(List.length(&2, Chess.Ply, moves))
  prior = F32.div(1.0, F32.from_nat(U32.to_nat(U32.max(1, count))))
  Search.Reply{epoch, request, id, 0, win, 0.0, F32.sub(1.0, win), count, Array.new(F32, 8n, prior)}

# A canonical draw reply is constructed here in Bend, not supplied by a host.
def rule_reply(ticket: Search.Ticket) -> Search.Reply:
  Search.Ticket{epoch, request, id, b, moves, reply} = ticket
  Search.Reply{epoch, request, id, 3, 0.0, 1.0, 0.0, 0, Array.new(F32, 8n, 0.0)}

def model_reply(cache: NativeEvaluation.Cache, timing: Bool, model: U32, root: Pos.Game, game: Pos.Game, ticket: Search.Ticket,
  table: Array<U64>) -> IO(NativeEvaluation.Result):
  match model:
    case 0: IO.pure(NativeEvaluation.Result, NativeEvaluation.material(cache, (table, reply_for(ticket))))
    case _: NativeEvaluation.evaluate(cache, timing, model, root, game, ticket, table)

def evaluate_reply(cache: NativeEvaluation.Cache, timing: Bool, model: U32, reason: U32, root: Pos.Game, game: Pos.Game,
  ticket: Search.Ticket, table: Array<U64>) -> IO(NativeEvaluation.Result):
  match reason:
    case 0: model_reply(cache, timing, model, root, game, ticket, table)
    case _: IO.pure(NativeEvaluation.Result, NativeEvaluation.material(cache, (table, rule_reply(ticket))))

def ready_state(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, tree: Search.Tree, budget: U32,
  inf: Bool, started: Work.Run, duration: U32, fallback: U32) -> State:
  State{cache, root, table, Running{tree, budget, inf, started, duration, fallback}, False{}}

def rule_notice(code: U32, node: U32, count: U32, half: U32) -> IO(Unit):
  match code:
    case 0: IO.pure(Unit, Unit{})
    case _:
      IO.print("info string rule_draw " ++ U32.show(node) ++ " " ++ Rules.name(code)
        ++ " " ++ U32.show(count) ++ " " ++ U32.show(half))

# Read only scalar completion counts; never duplicate the mutable node arena.
def resumed_reply(cache: NativeEvaluation.Cache, neural: Bool, before: U32, enc: Nat, back: Nat, post: Nat, t0: Nat,
  root: Pos.Game, table: Array<U64>, tree: Search.Tree, budget: U32,
  inf: Bool, +started: Work.Run, duration: U32, fallback: U32) -> IO(State):
  Search.Tree{pool, used, cap, +n, pending, epoch, request, stop, limit} = tree
  do IO<State>:
    t1 : Nat <- Work.clock(Work.profiling(started))
    return ready_state(cache, root, table, Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit}, budget, inf,
      Work.returned(started, neural, before, n, enc, back, post, Nat.sub(t1, t0)), duration, fallback)

def resume_result(neural: Bool, before: U32, root: Pos.Game, tree: Search.Tree, ticket: Search.Ticket, budget: U32,
  inf: Bool, +started: Work.Run, duration: U32, fallback: U32, r: NativeEvaluation.Result) -> IO(State):
  NativeEvaluation.Result{cache, table, reply, enc, back, post} = r
  do IO<State>:
    t0 : Nat <- Work.clock(Work.profiling(started))
    resumed_reply(cache, neural, before, enc, back, post, t0, root, table,
      Search.resume(tree, ticket, reply), budget, inf, started, duration, fallback)

def resume_reply(neural: Bool, root: Pos.Game, tree: Search.Tree, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Work.Run, duration: U32, fallback: U32, r: NativeEvaluation.Result) -> IO(State):
  Search.Tree{pool, used, cap, +before, pending, epoch, request, stop, limit} = tree
  resume_result(neural, before, root, Search.Tree{pool, used, cap, before, pending, epoch, request, stop, limit},
    ticket, budget, inf, started, duration, fallback, r)

def await_submitted(root: Pos.Game, tree: Search.Tree, budget: U32, inf: Bool,
  started: Work.Run, duration: U32, fallback: U32, r: NativeEvaluation.Submission) -> IO(State):
  NativeEvaluation.Submission{cache, table, pending, enc} = r
  IO.pure(State, State{cache, root, table, Awaiting{tree, pending, budget, inf, Work.submitted(started, enc), duration, fallback}, False{}})

def adjudicated_mode(asynchronous: Bool, cache: NativeEvaluation.Cache, +model: U32, +code: U32,
  +root: Pos.Game, game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket, table: Array<U64>,
  budget: U32, inf: Bool, +started: Work.Run, duration: U32, fallback: U32) -> IO(State):
  match asynchronous:
    case True{}:
      do IO<State>:
        r : NativeEvaluation.Submission <- NativeEvaluation.submit(cache, Work.profiling(started), model, game, ticket, table)
        await_submitted(root, tree, budget, inf, started, duration, fallback, r)
    case False{}:
      do IO<State>:
        r : NativeEvaluation.Result <- evaluate_reply(cache, Work.profiling(started), model, code, root, game, ticket, table)
        resume_reply(Bool.and(U32.is_gt(model, 0), U32.is_zero(code)), root, tree, ticket, budget, inf, started, duration, fallback, r)

def adjudicated_cache(+model: U32, +code: U32, root: Pos.Game, game: Pos.Game, tree: Search.Tree,
  ticket: Search.Ticket, table: Array<U64>, budget: U32, inf: Bool, started: Work.Run,
  duration: U32, fallback: U32, r: NativeEvaluation.Cache & Bool) -> IO(State):
  (cache, enabled) = r
  adjudicated_mode(Bool.and(enabled, Bool.and(U32.is_gt(model, 0), U32.is_zero(code))), cache, model, code,
    root, game, tree, ticket, table, budget, inf, started, duration, fallback)

def adjudicated(cache: NativeEvaluation.Cache, +model: U32, root: Pos.Game, game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Work.Run, duration: U32, fallback: U32, r: Array<U64> & Rules.Facts) -> IO(State):
  Search.Ticket{epoch, request, id, b, moves, reply} = ticket
  (table, Rules.Facts{+code, count, half}) = r
  do IO<State>:
    rule_notice(code, id, count, half)
    adjudicated_cache(model, code, root, game, tree, ticket, table, budget, inf, started, duration, fallback, NativeEvaluation.enabled(cache))

def history_checked(cache: NativeEvaluation.Cache, +model: U32, ok: Bool, root: Pos.Game, +game: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Work.Run, duration: U32, fallback: U32) -> IO(State):
  match ok:
    case False{}: IO.die(State, 2, "invalid native search history")
    case True{}:
      Search.Ticket{epoch, request, id, b, moves, reply} = ticket
      adjudicated(cache, model, root, game, tree, ticket, budget, inf, started, duration, fallback, Rules.automatic(game, moves, table))

def rebuilt(cache: NativeEvaluation.Cache, +model: U32, root: Pos.Game, table: Array<U64>, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Work.Run, duration: U32, fallback: U32, h: History.Rebuilt) -> IO(State):
  History.Rebuilt{tree, game, valid} = h
  history_checked(cache, model, valid, root, game, table, tree, ticket, budget, inf, started, duration, fallback)

def evaluated(cache: NativeEvaluation.Cache, +model: U32, needed: Bool, +root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Work.Run, duration: U32, fallback: U32) -> IO(State):
  match needed:
    case False{}: IO.pure(State, ready_state(cache, root, table, tree, budget, inf, started, duration, fallback))
    case True{}:
      Search.Ticket{epoch, request, id, b, moves, reply} = ticket
      rebuilt(cache, model, root, table, ticket, budget, inf, started, duration, fallback, History.rebuild(root, id, b, tree))

def prepared(cache: NativeEvaluation.Cache, +model: U32, root: Pos.Game, budget: U32, infinite: Bool, started: Work.Run, duration: U32, fallback: U32,
  p: Search.Prepared) -> IO(State):
  Search.Prepared{table, tree, +ticket} = p
  Search.Ticket{epoch, request, id, b, moves, needs} = ticket
  evaluated(cache, model, needs, root, table, tree, ticket, budget, infinite, started, duration, fallback)

type Pick is Type:
  Pick{pool: Array<Search.Node>, key: U32, visits: U32}

def pick_seen(+i: U32, +best: U32, +visits: U32, r: Array<Search.Node> & Search.Node) -> Pick:
  (pool, +a) = r
  +key = Search.key(Search.node_move(a))
  +n = Search.node_n(a)
  +better = Bool.and(U32.is_gt(i, 0), Bool.and(U32.is_zero(Search.node_parent(a)),
    Bool.or(U32.is_gt(n, visits), Bool.and(U32.is_eq(n, visits), U32.is_lt(key, best)))))
  Pick{pool, Bool.pick(U32, better, key, best), Bool.pick(U32, better, n, visits)}

def pick_step(+i: U32, p: Pick) -> Pick:
  Pick{pool, key, visits} = p
  pick_seen(i, key, visits, Array.get(Search.Node, pool, i))

def pick(n: Nat, +i: U32, p: Pick) -> Pick:
  match n:
    case 0n: p
    case 1n+j: pick(j, U32.inc(i), pick_step(i, p))

def move_key(+key: U32) -> String:
  match key:
    case 4294967295: "0000"
    case _:
      Pos.move_text(Chess.Ply{U32.and(key, 63), U32.and(U32.shrn(key, 6n), 63), U32.and(U32.shrn(key, 12n), 7), 0})

def result_label(n: U32) -> String:
  match n:
    case 0: "unsearched legal fallback; Bend PUCT"
    case _: "Bend PUCT"

def output(cache: NativeEvaluation.Cache, report: String, root: Pos.Game, table: Array<U64>, key: U32, +n: U32, stop: U32) -> IO(State):
  do IO<State>:
    IO.print(report)
    IO.print("info nodes " ++ U32.show(n) ++ " string " ++ result_label(n) ++ "; stop_code " ++ U32.show(stop))
    IO.print("bestmove " ++ move_key(key))
    return State{cache, root, table, Idle{}, False{}}

def output_or_hold(cache: NativeEvaluation.Cache, report: String, hold: Bool, root: Pos.Game, table: Array<U64>, key: U32, +n: U32, stop: U32) -> IO(State):
  match hold:
    case True{}: IO.pure(State, State{cache, root, table, Held{key, n, stop, report}, False{}})
    case False{}: output(cache, report, root, table, key, n, stop)

def fallback_notice(used: Bool) -> IO(Unit):
  match used:
    case False{}: IO.pure(Unit, Unit{})
    case True{}: IO.print("info string legal protocol fallback; no searched continuation; GUI owns adjudication")

def picked(cache: NativeEvaluation.Cache, run: Work.Run, hold: Bool, root: Pos.Game, table: Array<U64>, +fallback: U32, +n: U32, +stop: U32, p: Pick) -> IO(State):
  Pick{pool, +key, visits} = p
  +missing = U32.is_eq(key, 4294967295)
  do IO<State>:
    now : Nat <- IO.now()
    fallback_notice(Bool.and(missing, Bool.not(U32.is_eq(fallback, 4294967295))))
    output_or_hold(cache, Work.render(now, n, stop, run), hold, root, table, Bool.pick(U32, missing, fallback, key), n, stop)

def finish(cache: NativeEvaluation.Cache, run: Work.Run, hold: Bool, root: Pos.Game, table: Array<U64>, fallback: U32, tree: Search.Tree) -> IO(State):
  Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
  picked(cache, run, hold, root, table, fallback, n, stop, pick(U32.to_nat(used), 0, Pick{pool, 4294967295, 0}))

def cancelled_output(quitting: Bool, cache: NativeEvaluation.Cache, run: Work.Run, root: Pos.Game,
  table: Array<U64>, fallback: U32, tree: Search.Tree) -> IO(State):
  match quitting:
    case True{}: IO.pure(State, State{cache, root, table, Idle{}, True{}})
    case False{}: finish(cache, run, False{}, root, table, fallback, tree)

def cancel_wait(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  pending: NativeEvaluation.Pending, run: Work.Run, fallback: U32, +code: U32, quitting: Bool) -> IO(State):
  Search.Tree{pool, used, cap, +n, ignored, epoch, request, stop, limit} = tree
  NativeEvaluation.Pending{+id, ticket, game, entries} = pending
  +cancelled = Work.cancel_pending(run)
  do IO<State>:
    Async.cancel_identity(id)
    cancelled_output(quitting, NativeEvaluation.retiring(cache, id, cancelled, n, code), cancelled,
      root, table, fallback, Search.Tree{pool, used, cap, n, 4096, epoch, request, code, limit})

def stop(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Idle{}: IO.pure(State, State{cache, root, table, Idle{}, False{}})
    case Held{key, n, code, report}: output(cache, report, root, table, key, n, code)
    case Running{tree, budget, inf, started, duration, fallback}: finish(cache, started, False{}, root, table, fallback, tree)
    case Awaiting{tree, pending, budget, inf, started, duration, fallback}:
      cancel_wait(cache, root, table, tree, pending, started, fallback, 2, False{})

def request_quit(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Awaiting{tree, pending, budget, inf, started, duration, fallback}:
      cancel_wait(cache, root, table, tree, pending, started, fallback, 2, True{})
    case _: IO.pure(State, State{cache, root, table, Idle{}, True{}})

def selected(cache: NativeEvaluation.Cache, model: U32, root: Pos.Game, budget: U32, inf: Bool, +started: Work.Run,
  duration: U32, fallback: U32, t0: Nat, p: Search.Prepared) -> IO(State):
  do IO<State>:
    t1 : Nat <- Work.clock(Work.profiling(started))
    prepared(cache, model, root, budget, inf, Work.selected(started, Nat.sub(t1, t0)), duration, fallback, p)

def progress(cache: NativeEvaluation.Cache, +model: U32, done: Bool, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  budget: U32, inf: Bool, +started: Work.Run, duration: U32, fallback: U32) -> IO(State):
  match done:
    case True{}: finish(cache, started, inf, root, table, fallback, tree)
    case False{}:
      do IO<State>:
        t0 : Nat <- Work.clock(Work.profiling(started))
        selected(cache, model, root, budget, inf, started, duration, fallback, t0, Search.prepare(table, tree))

def elapsed(cache: NativeEvaluation.Cache, +model: U32, now: Nat, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +budget: U32, inf: Bool, +started: Work.Run, +duration: U32, fallback: U32) -> IO(State):
  Search.Tree{pool, used, cap, +n, pending, epoch, request, +code, limit} = tree
  done = Bool.or(Work.exhausted(started), Bool.or(Bool.and(U32.is_zero(model), Work.has_limit(started)),
    Bool.or(U32.is_ge(n, budget), Bool.or(U32.is_gt(code, 0),
      Bool.and(U32.is_gt(duration, 0), Nat.is_ge(Work.wall(now, started), U32.to_nat(duration)))))))
  progress(cache, model, done, root, table, Search.Tree{pool, used, cap, n, pending, epoch, request, code, limit}, budget, inf, started, duration, fallback)

def idle(s: State) -> IO(State):
  do IO<State>:
    IO.sleep(10)
    return s

def await_polled(+root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +pending: NativeEvaluation.Pending, budget: U32, inf: Bool, +run: Work.Run, duration: U32,
  fallback: U32, r: NativeEvaluation.Cache & U32) -> IO(State):
  NativeEvaluation.Pending{id, ticket, game, entries} = pending
  (cache, status) = r
  match status:
    case 1:
      do IO<State>:
        IO.sleep(1)
        return State{cache, root, table, Awaiting{tree, pending, budget, inf, run, duration, fallback}, False{}}
    case 2:
      do IO<State>:
        r : NativeEvaluation.Result <- NativeEvaluation.complete(cache, Work.profiling(run), root, pending, table)
        resume_reply(True{}, root, tree, ticket, budget, inf, run, duration, fallback, r)
    case _: IO.die(State, 2, "native async poll failed; no result applied to search")

def await_live(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +pending: NativeEvaluation.Pending, budget: U32, inf: Bool, run: Work.Run,
  duration: U32, fallback: U32) -> IO(State):
  NativeEvaluation.Pending{id, ticket, game, entries} = pending
  do IO<State>:
    r : NativeEvaluation.Cache & U32 <- NativeEvaluation.poll(cache, id)
    await_polled(root, table, tree, pending, budget, inf, run, duration, fallback, r)

def await_deadline(expired: Bool, cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  pending: NativeEvaluation.Pending, budget: U32, inf: Bool, run: Work.Run,
  duration: U32, fallback: U32) -> IO(State):
  match expired:
    case True{}: cancel_wait(cache, root, table, tree, pending, run, fallback, 0, False{})
    case False{}: await_live(cache, root, table, tree, pending, budget, inf, run, duration, fallback)

def past_deadline(now: Nat, +duration: U32, run: Work.Run) -> Bool:
  Bool.and(U32.is_gt(duration, 0), Nat.is_ge(Work.wall(now, run), U32.to_nat(duration)))

# A replacement go is represented by Running with its own epoch and clock, but
# cannot select/dispatch until the old slot retires. Its timer still expires.
def blocked_running(expired: Bool, cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  budget: U32, inf: Bool, run: Work.Run, duration: U32, fallback: U32) -> IO(State):
  match expired:
    case True{}: finish(cache, run, False{}, root, table, fallback, tree)
    case False{}:
      do IO<State>:
        IO.sleep(1)
        return ready_state(cache, root, table, tree, budget, inf, run, duration, fallback)

def tick_running(busy: Bool, cache: NativeEvaluation.Cache, model: U32, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  budget: U32, inf: Bool, +run: Work.Run, +duration: U32, fallback: U32) -> IO(State):
  match busy:
    case True{}:
      do IO<State>:
        now : Nat <- IO.now()
        blocked_running(past_deadline(now, duration, run), cache, root, table, tree, budget, inf, run, duration, fallback)
    case False{}:
      do IO<State>:
        now : Nat <- IO.now()
        elapsed(cache, model, now, root, table, tree, budget, inf, run, duration, fallback)

def tick_retired(r: NativeEvaluation.Cache & Bool, model: U32, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  (cache, busy) = r
  match mode:
    case Idle{}: idle(State{cache, root, table, Idle{}, False{}})
    case Held{k, n, code, report}: idle(State{cache, root, table, Held{k, n, code, report}, False{}})
    case Running{tree, budget, inf, started, duration, fallback}:
      tick_running(busy, cache, model, root, table, tree, budget, inf, started, duration, fallback)
    case Awaiting{tree, pending, budget, inf, +started, +duration, fallback}:
      do IO<State>:
        now : Nat <- IO.now()
        await_deadline(past_deadline(now, duration, started), cache, root, table, tree, pending, budget, inf, started, duration, fallback)

def tick_mode(cache: NativeEvaluation.Cache, model: U32, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  do IO<State>:
    r : NativeEvaluation.Cache & Bool <- NativeEvaluation.retire(cache)
    tick_retired(r, model, root, table, mode)

def fallback(ms: List<&2, Chess.Ply>, best: U32) -> U32:
  match ms:
    case Nil{}: best
    case Con{m, rest}: fallback(rest, U32.min(best, Search.key(m)))

def start_identified(+root: Pos.Game, lim: Protocol.Limits, table: Array<U64>, ms: List<&2, Chess.Ply>,
  r: NativeEvaluation.Cache & U32) -> IO(State):
  Protocol.Limits{budget, depth, +duration, inf, seen, valid, evals} = lim
  (cache, +epoch) = r
  do IO<State>:
    now : Nat <- IO.now()
    return State{cache, root, table, Running{Search.start(Pos.board(root), 4096, epoch, depth), budget, inf,
      Work.identified(Work.start(now, evals, duration, U32.is_gt(U32.and(seen, 32), 0)), epoch), duration, fallback(ms, 4294967295)}, False{}}

def start_search(cache: NativeEvaluation.Cache, root: Pos.Game, lim: Protocol.Limits, r: Array<U64> & List<&2, Chess.Ply>) -> IO(State):
  (table, ms) = r
  do IO<State>:
    r : NativeEvaluation.Cache & U32 <- NativeEvaluation.next(cache)
    start_identified(root, lim, table, ms, r)

def checked_go(cache: NativeEvaluation.Cache, ok: Bool, +root: Pos.Game, table: Array<U64>, lim: Protocol.Limits) -> IO(State):
  match ok:
    case False{}: message(State{cache, root, table, Idle{}, False{}}, "unsupported or invalid go limits")
    case True{}: start_search(cache, root, lim, Chess.legal_moves(table, Pos.board(root)))

def go_limits(cache: NativeEvaluation.Cache, root: Pos.Game, table: Array<U64>, lim: Protocol.Limits) -> IO(State):
  Protocol.Limits{budget, depth, +duration, +inf, seen, +valid, evals} = lim
  checked_go(cache, Bool.and(valid, Bool.not(Bool.and(inf, U32.is_gt(duration, 0)))), root, table,
    Protocol.Limits{budget, depth, duration, inf, seen, valid, evals})

def replace(cache: NativeEvaluation.Cache, old: Pos.Game, r: Array<U64> & Maybe<&2, Pos.Game>) -> IO(State):
  (table, g) = r
  match g:
    case None{}: message(State{cache, old, table, Idle{}, False{}}, "invalid position; previous root preserved")
    case Some{g}: IO.pure(State, State{cache, g, table, Idle{}, False{}})

def perft_done(cache: NativeEvaluation.Cache, root: Pos.Game, depth: U32, r: Array<U64> & U64) -> IO(State):
  (table, count) = r
  do IO<State>:
    IO.print("info string perft " ++ U32.show(depth) ++ " " ++ word(count))
    return State{cache, root, table, Idle{}, False{}}

def perft_limit(cache: NativeEvaluation.Cache, valid: Bool, root: Pos.Game, table: Array<U64>, depth: U32) -> IO(State):
  match valid:
    case False{}: message(State{cache, root, table, Idle{}, False{}}, "perft depth must be 0..5")
    case True{}:
      +root = root
      +depth = depth
      perft_done(cache, root, depth, Chess.perft(U32.to_nat(depth), Con{Pos.board(root), Nil{}}, table, U64.zero()))

def perft_number(cache: NativeEvaluation.Cache, n: Maybe<&2, U32>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match n:
    case None{}: message(State{cache, root, table, Idle{}, False{}}, "invalid perft depth")
    case Some{+n}: perft_limit(cache, U32.is_le(n, 5), root, table, n)

def perft_args(cache: NativeEvaluation.Cache, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match args:
    case Con{n, Nil{}}: perft_number(cache, Text.number(n), root, table)
    case _: message(State{cache, root, table, Idle{}, False{}}, "perft requires one depth")

# Read-only diagnostics for an external oracle; all facts are computed in Bend.
def rules_done(cache: NativeEvaluation.Cache, root: Pos.Game, r: Array<U64> & Rules.Facts) -> IO(State):
  (table, Rules.Facts{code, count, half}) = r
  do IO<State>:
    IO.print("info string rules " ++ Rules.name(code) ++ " " ++ U32.show(count) ++ " " ++ U32.show(half))
    return State{cache, root, table, Idle{}, False{}}

def rules_args(cache: NativeEvaluation.Cache, args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  match args:
    case Nil{}: rules_done(cache, root, Rules.inspect(root, table))
    case _: message(State{cache, root, table, Idle{}, False{}}, "rules takes no arguments")

def encoding_args(cache: NativeEvaluation.Cache, args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- EncodingDiagnostic.run(args, root, table)
    return State{cache, root, table, Idle{}, False{}}

def input_args(cache: NativeEvaluation.Cache, args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- ModelInputDiagnostic.run(args, root, table)
    return State{cache, root, table, Idle{}, False{}}

def policy_args(cache: NativeEvaluation.Cache, args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- PolicyDiagnostic.run(args, root, table)
    return State{cache, root, table, Idle{}, False{}}

def evaluation_args(cache: NativeEvaluation.Cache, args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- EvaluationDiagnostic.run(args, root, table)
    return State{cache, root, table, Idle{}, False{}}

def idle_command(cache: NativeEvaluation.Cache, code: U32, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match code:
    case 3: replace(cache, root, Protocol.position(Protocol.parse_position(args), table))
    case 4: go_limits(cache, root, table, Protocol.go(args, Protocol.GoState{Protocol.Limits{64, 4, 0, False{}, 0, True{}, 0}, 0}))
    case 7: IO.pure(State, State{cache, Pos.start(), table, Idle{}, False{}})
    case 9: perft_args(cache, args, root, table)
    case 10: rules_args(cache, args, root, table)
    case 11: encoding_args(cache, args, root, table)
    case 12: input_args(cache, args, root, table)
    case 13: policy_args(cache, args, root, table)
    case 14: evaluation_args(cache, args, root, table)
    case _: message(State{cache, root, table, Idle{}, False{}}, "unsupported command")

def available(cache: NativeEvaluation.Cache, code: U32, args: List<&2, String>, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Idle{}: idle_command(cache, code, args, root, table)
    case Running{t, n, inf, start, ms, fb}: message(State{cache, root, table, Running{t, n, inf, start, ms, fb}, False{}}, "busy; stop before changing position or search")
    case Awaiting{t, pending, n, inf, start, ms, fb}: message(State{cache, root, table, Awaiting{t, pending, n, inf, start, ms, fb}, False{}}, "busy; stop before changing position or search")
    case Held{k, n, stop, report}: message(State{cache, root, table, Held{k, n, stop, report}, False{}}, "busy; stop before changing position or search")

def handle(code: U32, args: List<&2, String>, s: State) -> IO(State):
  match code:
    case 1:
      do IO<State>:
        IO.print("id name DeepFin Bend standalone diagnostic")
        IO.print("id author DeepFin contributors")
        IO.print("uciok")
        return s
    case 2:
      do IO<State>:
        IO.print("readyok")
        return s
    case 5:
      State{cache, root, table, mode, quit} = s
      stop(cache, root, table, mode)
    case 6:
      State{cache, root, table, mode, quit} = s
      request_quit(cache, root, table, mode)
    case 8: debug(s)
    case _:
      State{cache, root, table, mode, quit} = s
      available(cache, code, args, root, table, mode)

def dispatch(words: List<&2, String>, s: State) -> IO(State):
  match words:
    case Nil{}: IO.pure(State, s)
    case Con{h, t}: handle(Protocol.command(h), t, s)

def command_deadline(expired: Bool, s: State) -> IO(State):
  match expired:
    case False{}: IO.pure(State, s)
    case True{}:
      State{cache, root, table, mode, quit} = s
      match mode:
        case Awaiting{tree, pending, budget, inf, run, duration, fallback}:
          cancel_wait(cache, root, table, tree, pending, run, fallback, 0, False{})
        case Running{tree, budget, inf, run, duration, fallback}: finish(cache, run, False{}, root, table, fallback, tree)
        case _: IO.pure(State, State{cache, root, table, mode, quit})

def command_checked(s: State) -> IO(State):
  State{cache, root, table, mode, +quit} = s
  match mode:
    case Running{tree, budget, inf, +run, +duration, fallback}:
      do IO<State>:
        now : Nat <- IO.now()
        command_deadline(past_deadline(now, duration, run), State{cache, root, table, Running{tree, budget, inf, run, duration, fallback}, quit})
    case Awaiting{tree, pending, budget, inf, +run, +duration, fallback}:
      do IO<State>:
        now : Nat <- IO.now()
        command_deadline(past_deadline(now, duration, run), State{cache, root, table, Awaiting{tree, pending, budget, inf, run, duration, fallback}, quit})
    case _: IO.pure(State, State{cache, root, table, mode, quit})

def command_packet(words: List<&2, String>, s: State) -> IO(State):
  do IO<State>:
    next : State <- dispatch(words, s)
    command_checked(next)

def packet(+model: U32, p: Packet, s: State) -> IO(State):
  Packet{kind, text} = p
  match kind:
    case 0:
      State{cache, root, table, mode, quit} = s
      tick_mode(cache, model, root, table, mode)
    case 1: command_packet(Text.words(text), s)
    case 2: handle(6, Nil{}, s)
    case _: message(s, "invalid or overlong input record")

def serve(+model: U32, fuel: Nat, s: State) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string event budget exhausted")
    case 1n+p:
      State{cache, root, table, mode, quit} = s
      match quit:
        case True{}: NativeEvaluation.shutdown(cache)
        case False{}:
          do IO<Unit>:
            input : Packet <- Wire.poll()
            next : State <- packet(model, input, State{cache, root, table, mode, False{}})
            serve(model, p, next)

def launch(+model: U32) -> IO(Unit):
  do IO<Unit>:
    diagnostics : Bool <- Buffers.Native.diagnostics()
    asynchronous : Bool <- Async.Async.enabled()
    serve(model, 1000000000n, State{NativeEvaluation.cache(model, diagnostics, asynchronous), Pos.start(), Tables.build(), Idle{}, False{}})

def native_launch(+model: U32) -> IO(Unit):
  do IO<Unit>:
    IO.print("info string evaluator bound native CPU F32 batch=1; Bend input/policy/softmax; profile=" ++ U32.show(model))
    launch(model)

def opened(+model: U32) -> IO(Unit):
  match model:
    case 0:
      do IO<Unit>:
        IO.print("info string evaluator diagnostic material; no neural model")
        launch(model)
    case 1: native_launch(model)
    case 2: native_launch(model)
    case 3: native_launch(model)
    case 4: native_launch(model)
    case _: IO.die(Unit, 2, "unsupported bound native model profile")

def main() -> IO(Unit):
  do IO<Unit>:
    model : U32 <- Buffers.Native.open()
    opened(model)
