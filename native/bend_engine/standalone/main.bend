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
import ./NativeEvaluation.bend as NativeEvaluation

type Packet is Data:
  Packet{kind: U32, text: String}

def Wire.poll() -> IO(Packet):
  import "./poll.c"

type Mode is Type:
  Idle{}
  Running{tree: Search.Tree, budget: U32, infinite: Bool, started: Nat, duration: U32, fallback: U32}
  Held{key: U32, completed: U32, stop: U32}

type State is Type:
  State{root: Pos.Game, table: Array<U64>, mode: Mode, quit: Bool}

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
  State{g, table, mode, quit} = s
  Pos.Game{+b, +half, +full, +hist} = g
  do IO<State>:
    IO.print("info string state " ++ board_text(b) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full)
      ++ " " ++ Nat.show(List.length(&2, Pos.Past, hist)))
    history_text(hist)
    IO.print("info string state_end")
    return State{Pos.Game{b, half, full, hist}, table, mode, quit}

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

def model_reply(model: U32, root: Pos.Game, game: Pos.Game, ticket: Search.Ticket,
  table: Array<U64>) -> IO(Array<U64> & Search.Reply):
  match model:
    case 0: IO.pure(Array<U64> & Search.Reply, (table, reply_for(ticket)))
    case _: NativeEvaluation.evaluate(model, root, game, ticket, table)

def evaluate_reply(model: U32, reason: U32, root: Pos.Game, game: Pos.Game,
  ticket: Search.Ticket, table: Array<U64>) -> IO(Array<U64> & Search.Reply):
  match reason:
    case 0: model_reply(model, root, game, ticket, table)
    case _: IO.pure(Array<U64> & Search.Reply, (table, rule_reply(ticket)))

def ready_state(root: Pos.Game, table: Array<U64>, tree: Search.Tree, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32) -> State:
  State{root, table, Running{tree, budget, inf, started, duration, fallback}, False{}}

def rule_notice(code: U32, node: U32, count: U32, half: U32) -> IO(Unit):
  match code:
    case 0: IO.pure(Unit, Unit{})
    case _:
      IO.print("info string rule_draw " ++ U32.show(node) ++ " " ++ Rules.name(code)
        ++ " " ++ U32.show(count) ++ " " ++ U32.show(half))

def resume_reply(root: Pos.Game, tree: Search.Tree, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, r: Array<U64> & Search.Reply) -> State:
  (table, reply) = r
  ready_state(root, table, Search.resume(tree, ticket, reply), budget, inf, started, duration, fallback)

def adjudicated(+model: U32, +root: Pos.Game, game: Pos.Game, tree: Search.Tree, +ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, r: Array<U64> & Rules.Facts) -> IO(State):
  Search.Ticket{epoch, request, id, b, moves, reply} = ticket
  (table, Rules.Facts{+code, count, half}) = r
  do IO<State>:
    rule_notice(code, id, count, half)
    r : Array<U64> & Search.Reply <- evaluate_reply(model, code, root, game, ticket, table)
    return resume_reply(root, tree, ticket, budget, inf, started, duration, fallback, r)

def history_checked(+model: U32, ok: Bool, root: Pos.Game, +game: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32) -> IO(State):
  match ok:
    case False{}: IO.die(State, 2, "invalid native search history")
    case True{}:
      Search.Ticket{epoch, request, id, b, moves, reply} = ticket
      adjudicated(model, root, game, tree, ticket, budget, inf, started, duration, fallback, Rules.automatic(game, moves, table))

def rebuilt(+model: U32, root: Pos.Game, table: Array<U64>, ticket: Search.Ticket, budget: U32,
  inf: Bool, started: Nat, duration: U32, fallback: U32, h: History.Rebuilt) -> IO(State):
  History.Rebuilt{tree, game, valid} = h
  history_checked(model, valid, root, game, table, tree, ticket, budget, inf, started, duration, fallback)

def evaluated(+model: U32, needed: Bool, +root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +ticket: Search.Ticket, budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32) -> IO(State):
  match needed:
    case False{}: IO.pure(State, ready_state(root, table, tree, budget, inf, started, duration, fallback))
    case True{}:
      Search.Ticket{epoch, request, id, b, moves, reply} = ticket
      rebuilt(model, root, table, ticket, budget, inf, started, duration, fallback, History.rebuild(root, id, b, tree))

def prepared(+model: U32, root: Pos.Game, budget: U32, infinite: Bool, started: Nat, duration: U32, fallback: U32,
  p: Search.Prepared) -> IO(State):
  Search.Prepared{table, tree, +ticket} = p
  Search.Ticket{epoch, request, id, b, moves, needs} = ticket
  evaluated(model, needs, root, table, tree, ticket, budget, infinite, started, duration, fallback)

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

def output(root: Pos.Game, table: Array<U64>, key: U32, +n: U32, stop: U32) -> IO(State):
  do IO<State>:
    IO.print("info nodes " ++ U32.show(n) ++ " string " ++ result_label(n) ++ "; stop_code " ++ U32.show(stop))
    IO.print("bestmove " ++ move_key(key))
    return State{root, table, Idle{}, False{}}

def output_or_hold(hold: Bool, root: Pos.Game, table: Array<U64>, key: U32, +n: U32, stop: U32) -> IO(State):
  match hold:
    case True{}: IO.pure(State, State{root, table, Held{key, n, stop}, False{}})
    case False{}: output(root, table, key, n, stop)

def fallback_notice(used: Bool) -> IO(Unit):
  match used:
    case False{}: IO.pure(Unit, Unit{})
    case True{}: IO.print("info string legal protocol fallback; no searched continuation; GUI owns adjudication")

def picked(hold: Bool, root: Pos.Game, table: Array<U64>, +fallback: U32, n: U32, stop: U32, p: Pick) -> IO(State):
  Pick{pool, +key, visits} = p
  +missing = U32.is_eq(key, 4294967295)
  do IO<State>:
    fallback_notice(Bool.and(missing, Bool.not(U32.is_eq(fallback, 4294967295))))
    output_or_hold(hold, root, table, Bool.pick(U32, missing, fallback, key), n, stop)

def finish(hold: Bool, root: Pos.Game, table: Array<U64>, fallback: U32, tree: Search.Tree) -> IO(State):
  Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
  picked(hold, root, table, fallback, n, stop, pick(U32.to_nat(used), 0, Pick{pool, 4294967295, 0}))

def stop(root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Idle{}: IO.pure(State, State{root, table, Idle{}, False{}})
    case Held{key, n, code}: output(root, table, key, n, code)
    case Running{tree, budget, inf, started, duration, fallback}: finish(False{}, root, table, fallback, tree)

def progress(+model: U32, done: Bool, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  budget: U32, inf: Bool, started: Nat, duration: U32, fallback: U32) -> IO(State):
  match done:
    case True{}: finish(inf, root, table, fallback, tree)
    case False{}: prepared(model, root, budget, inf, started, duration, fallback, Search.prepare(table, tree))

def elapsed(+model: U32, now: Nat, root: Pos.Game, table: Array<U64>, tree: Search.Tree,
  +budget: U32, inf: Bool, +started: Nat, +duration: U32, fallback: U32) -> IO(State):
  Search.Tree{pool, used, cap, +n, pending, epoch, request, +code, limit} = tree
  done = Bool.or(U32.is_ge(n, budget), Bool.or(U32.is_gt(code, 0),
    Bool.and(U32.is_gt(duration, 0), Nat.is_ge(Nat.sub(now, started), U32.to_nat(duration)))))
  progress(model, done, root, table, Search.Tree{pool, used, cap, n, pending, epoch, request, code, limit}, budget, inf, started, duration, fallback)

def idle(s: State) -> IO(State):
  do IO<State>:
    IO.sleep(10)
    return s

def tick_mode(+model: U32, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Idle{}: idle(State{root, table, Idle{}, False{}})
    case Held{k, n, code}: idle(State{root, table, Held{k, n, code}, False{}})
    case Running{tree, budget, inf, started, duration, fallback}:
      do IO<State>:
        now : Nat <- IO.now()
        elapsed(model, now, root, table, tree, budget, inf, started, duration, fallback)

def fallback(ms: List<&2, Chess.Ply>, best: U32) -> U32:
  match ms:
    case Nil{}: best
    case Con{m, rest}: fallback(rest, U32.min(best, Search.key(m)))

def start_search(+root: Pos.Game, lim: Protocol.Limits, r: Array<U64> & List<&2, Chess.Ply>) -> IO(State):
  Protocol.Limits{budget, depth, duration, inf, seen, valid} = lim
  (table, ms) = r
  do IO<State>:
    now : Nat <- IO.now()
    return State{root, table, Running{Search.start(Pos.board(root), 4096, 1, depth), budget, inf, now, duration, fallback(ms, 4294967295)}, False{}}

def checked_go(ok: Bool, +root: Pos.Game, table: Array<U64>, lim: Protocol.Limits) -> IO(State):
  match ok:
    case False{}: message(State{root, table, Idle{}, False{}}, "unsupported or invalid go limits")
    case True{}: start_search(root, lim, Chess.legal_moves(table, Pos.board(root)))

def go_limits(root: Pos.Game, table: Array<U64>, lim: Protocol.Limits) -> IO(State):
  Protocol.Limits{budget, depth, +duration, +inf, seen, +valid} = lim
  checked_go(Bool.and(valid, Bool.not(Bool.and(inf, U32.is_gt(duration, 0)))), root, table,
    Protocol.Limits{budget, depth, duration, inf, seen, valid})

def replace(old: Pos.Game, r: Array<U64> & Maybe<&2, Pos.Game>) -> IO(State):
  (table, g) = r
  match g:
    case None{}: message(State{old, table, Idle{}, False{}}, "invalid position; previous root preserved")
    case Some{g}: IO.pure(State, State{g, table, Idle{}, False{}})

def perft_done(root: Pos.Game, depth: U32, r: Array<U64> & U64) -> IO(State):
  (table, count) = r
  do IO<State>:
    IO.print("info string perft " ++ U32.show(depth) ++ " " ++ word(count))
    return State{root, table, Idle{}, False{}}

def perft_limit(valid: Bool, root: Pos.Game, table: Array<U64>, depth: U32) -> IO(State):
  match valid:
    case False{}: message(State{root, table, Idle{}, False{}}, "perft depth must be 0..5")
    case True{}:
      +root = root
      +depth = depth
      perft_done(root, depth, Chess.perft(U32.to_nat(depth), Con{Pos.board(root), Nil{}}, table, U64.zero()))

def perft_number(n: Maybe<&2, U32>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match n:
    case None{}: message(State{root, table, Idle{}, False{}}, "invalid perft depth")
    case Some{+n}: perft_limit(U32.is_le(n, 5), root, table, n)

def perft_args(args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match args:
    case Con{n, Nil{}}: perft_number(Text.number(n), root, table)
    case _: message(State{root, table, Idle{}, False{}}, "perft requires one depth")

# Read-only diagnostics for an external oracle; all facts are computed in Bend.
def rules_done(root: Pos.Game, r: Array<U64> & Rules.Facts) -> IO(State):
  (table, Rules.Facts{code, count, half}) = r
  do IO<State>:
    IO.print("info string rules " ++ Rules.name(code) ++ " " ++ U32.show(count) ++ " " ++ U32.show(half))
    return State{root, table, Idle{}, False{}}

def rules_args(args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  match args:
    case Nil{}: rules_done(root, Rules.inspect(root, table))
    case _: message(State{root, table, Idle{}, False{}}, "rules takes no arguments")

def encoding_args(args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- EncodingDiagnostic.run(args, root, table)
    return State{root, table, Idle{}, False{}}

def input_args(args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- ModelInputDiagnostic.run(args, root, table)
    return State{root, table, Idle{}, False{}}

def policy_args(args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- PolicyDiagnostic.run(args, root, table)
    return State{root, table, Idle{}, False{}}

def evaluation_args(args: List<&2, String>, +root: Pos.Game, table: Array<U64>) -> IO(State):
  do IO<State>:
    table : Array<U64> <- EvaluationDiagnostic.run(args, root, table)
    return State{root, table, Idle{}, False{}}

def idle_command(code: U32, args: List<&2, String>, root: Pos.Game, table: Array<U64>) -> IO(State):
  match code:
    case 3: replace(root, Protocol.position(Protocol.parse_position(args), table))
    case 4: go_limits(root, table, Protocol.go(args, Protocol.GoState{Protocol.Limits{64, 4, 0, False{}, 0, True{}}, 0}))
    case 7: IO.pure(State, State{Pos.start(), table, Idle{}, False{}})
    case 9: perft_args(args, root, table)
    case 10: rules_args(args, root, table)
    case 11: encoding_args(args, root, table)
    case 12: input_args(args, root, table)
    case 13: policy_args(args, root, table)
    case 14: evaluation_args(args, root, table)
    case _: message(State{root, table, Idle{}, False{}}, "unsupported command")

def available(code: U32, args: List<&2, String>, root: Pos.Game, table: Array<U64>, mode: Mode) -> IO(State):
  match mode:
    case Idle{}: idle_command(code, args, root, table)
    case Running{t, n, inf, start, ms, fb}: message(State{root, table, Running{t, n, inf, start, ms, fb}, False{}}, "busy; stop before changing position or search")
    case Held{k, n, stop}: message(State{root, table, Held{k, n, stop}, False{}}, "busy; stop before changing position or search")

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
      State{root, table, mode, quit} = s
      stop(root, table, mode)
    case 6:
      State{root, table, mode, quit} = s
      IO.pure(State, State{root, table, Idle{}, True{}})
    case 8: debug(s)
    case _:
      State{root, table, mode, quit} = s
      available(code, args, root, table, mode)

def dispatch(words: List<&2, String>, s: State) -> IO(State):
  match words:
    case Nil{}: IO.pure(State, s)
    case Con{h, t}: handle(Protocol.command(h), t, s)

def packet(+model: U32, p: Packet, s: State) -> IO(State):
  Packet{kind, text} = p
  match kind:
    case 0:
      State{root, table, mode, quit} = s
      tick_mode(model, root, table, mode)
    case 1: dispatch(Text.words(text), s)
    case 2: handle(6, Nil{}, s)
    case _: message(s, "invalid or overlong input record")

def serve(+model: U32, fuel: Nat, s: State) -> IO(Unit):
  match fuel:
    case 0n: IO.print("info string event budget exhausted")
    case 1n+p:
      State{root, table, mode, quit} = s
      match quit:
        case True{}: IO.pure(Unit, Unit{})
        case False{}:
          do IO<Unit>:
            input : Packet <- Wire.poll()
            next : State <- packet(model, input, State{root, table, mode, False{}})
            serve(model, p, next)

def launch(model: U32) -> IO(Unit):
  serve(model, 1000000000n, State{Pos.start(), Tables.build(), Idle{}, False{}})

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
    model : U32 <- NativeEvaluation.Native.open()
    opened(model)
