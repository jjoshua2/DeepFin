import Base
import ../legal_probe/Chess.bend as Chess
import ./Search.bend as Search

type Job is Type:
  Job{board: Chess.Board, table: Array<U64>, depth: U32, mode: U32, seed: U32}

def Job.load() -> IO(Job):
  import "../legal_probe/input.c"

type Config is Data:
  Config{epoch: U32, simulations: U32, capacity: U32, depth: U32}

# Transport only decodes commands. Bend checks the epoch and move semantics.
# kind: 0=config/quit, 1=advance. The old config wire is unchanged.
type Command is Data:
  Command{kind: U32, expected: U32, epoch: U32, simulations: U32,
    capacity: U32, depth: U32, key: U32}

def Command.read() -> IO(Command):
  import "./transport.c"

def Reply.read() -> IO(Search.Reply):
  import "./transport.c"

def word(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def float_word(x: F32) -> String:
  F32{bits} = x
  U32.show(U32{bits})

def board_text(+b: Chess.Board) -> String:
  word(Chess.get_pawns(b)) ++ " " ++ word(Chess.get_knights(b)) ++ " " ++ word(Chess.get_bishops(b))
  ++ " " ++ word(Chess.get_rooks(b)) ++ " " ++ word(Chess.get_queens(b)) ++ " " ++ word(Chess.get_kings(b))
  ++ " " ++ word(Chess.get_white(b)) ++ " " ++ word(Chess.get_black(b)) ++ " " ++ U32.show(Chess.get_turn(b))
  ++ " " ++ U32.show(Chess.get_rights(b)) ++ " " ++ U32.show(Chess.get_ep(b))

def moves_print(ms: List<&2, Chess.Ply>) -> IO(Unit):
  match ms:
    case Nil{}: IO.print("end_eval")
    case Con{m, rest}:
      do IO<Unit>:
        IO.print("action " ++ U32.show(Search.key(m)))
        moves_print(rest)

# Root-to-leaf moves, not just the final board, define a neural input.
# Walking the actual parent chain preserves transposition-specific history.
type Trace is Type:
  Trace{pool: Array<Search.Node>, path: List<&2, U32>, id: U32, valid: Bool}

def trace_read(+id: U32, path: List<&2, U32>, valid: Bool, r: Array<Search.Node> & Search.Node) -> Trace:
  (pool, +node) = r
  +parent = Search.node_parent(node)
  Trace{pool, Con{Search.key(Search.node_move(node)), path}, parent,
    Bool.and(valid, U32.is_lt(parent, id))}

def trace_step(stopped: Bool, t: Trace) -> Trace:
  match stopped:
    case True{}: t
    case False{}:
      Trace{pool, path, +id, valid} = t
      trace_read(id, path, valid, Array.get(Search.Node, pool, id))

def trace_dispatch(t: Trace) -> Trace:
  Trace{pool, path, +id, +valid} = t
  trace_step(Bool.or(U32.is_zero(id), Bool.not(valid)), Trace{pool, path, id, valid})

def trace(fuel: Nat, t: Trace) -> Trace:
  match fuel:
    case 0n: t
    case 1n+p: trace(p, trace_dispatch(t))

def path_text(path: List<&2, U32>) -> String:
  match path:
    case Nil{}: ""
    case Con{k, rest}: " " ++ U32.show(k) ++ path_text(rest)

def ticket_send(ticket: Search.Ticket, +path: List<&2, U32>) -> IO(Unit):
  Search.Ticket{epoch, request, node, board, +moves, needed} = ticket
  do IO<Unit>:
    IO.print("eval " ++ U32.show(epoch) ++ " " ++ U32.show(request) ++ " " ++ U32.show(node)
      ++ " " ++ U32.show(U32.from_nat(List.length(&2, Chess.Ply, moves))))
    IO.print("board " ++ board_text(board))
    IO.print("path " ++ U32.show(U32.from_nat(List.length(&2, U32, path))) ++ path_text(path))
    moves_print(moves)

type Live is Type:
  Live{table: Array<U64>, tree: Search.Tree}

def live_flag(table: Array<U64>, tree: Search.Tree) -> Live & Bool:
  Search.Tree{pool, used, cap, n, pending, epoch, request, +stop, limit} = tree
  (Live{table, Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit}}, Bool.not(U32.is_zero(stop)))

def ticket_node(ticket: Search.Ticket) -> U32:
  Search.Ticket{epoch, request, node, board, moves, needed} = ticket
  node

def step_path_checked(valid: Bool, table: Array<U64>, +ticket: Search.Ticket,
  path: List<&2, U32>, tree: Search.Tree) -> IO(Live & Bool):
  match valid:
    case False{}: IO.die(Live & Bool, 2, "invalid search parent chain")
    case True{}:
      do IO<Live & Bool>:
        ticket_send(ticket, path)
        reply : Search.Reply <- Reply.read()
        return live_flag(table, Search.resume(tree, ticket, reply))

def step_traced(table: Array<U64>, ticket: Search.Ticket, used: U32, cap: U32,
  n: U32, pending: U32, epoch: U32, request: U32, stop: U32, limit: U32, t: Trace) -> IO(Live & Bool):
  Trace{pool, path, id, valid} = t
  step_path_checked(Bool.and(valid, U32.is_zero(id)), table, ticket, path,
    Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit})

def step_read(needed: Bool, table: Array<U64>, tree: Search.Tree, +ticket: Search.Ticket) -> IO(Live & Bool):
  match needed:
    case False{}: IO.pure(Live & Bool, live_flag(table, tree))
    case True{}:
      Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
      step_traced(table, ticket, used, cap, n, pending, epoch, request, stop, limit,
        trace(32n, Trace{pool, Nil{}, ticket_node(ticket), True{}}))

def ticket_needed(t: Search.Ticket) -> Bool:
  Search.Ticket{epoch, request, node, board, moves, needed} = t
  needed

def step_prepared(r: Search.Prepared) -> IO(Live & Bool):
  Search.Prepared{table, tree, +ticket} = r
  step_read(ticket_needed(ticket), table, tree, ticket)

def step(s: Live) -> IO(Live & Bool):
  Live{table, tree} = s
  step_prepared(Search.prepare(table, tree))

def run(n: Nat, r: Live & Bool) -> IO(Live):
  match n:
    case 0n:
      (s, stop) = r
      IO.pure(Live, s)
    case 1n+p:
      (s, stop) = r
      match stop:
        case True{}: IO.pure(Live, s)
        case False{}:
          do IO<Live>:
            next : Live & Bool <- step(s)
            run(p, next)

type Printed is Type:
  Printed{pool: Array<Search.Node>, best: U32, visits: U32,
    claim: Bool, positive: U32, positive_visits: U32}

def node_print(+i: U32, +best: U32, +visits: U32, claim: Bool,
  +positive: U32, +positive_visits: U32, r: Array<Search.Node> & Search.Node) -> IO(Printed):
  (pool, +a) = r
  +key = Search.key(Search.node_move(a))
  +n = Search.node_n(a)
  +root_child = Bool.and(Bool.not(U32.is_zero(i)), U32.is_zero(Search.node_parent(a)))
  claim = Bool.or(claim, Bool.and(root_child, Search.is_claim(a)))
  +good = Bool.and(root_child, Bool.and(Bool.not(Search.is_claim(a)),
    Bool.and(U32.is_gt(n, 0), F32.is_gt(F32.neg(Search.mean(a)), 0.0))))
  +better_positive = Bool.and(good, Bool.or(U32.is_gt(n, positive_visits),
    Bool.and(U32.is_eq(n, positive_visits), U32.is_lt(key, positive))))
  +better = Bool.and(Bool.not(U32.is_zero(i)), Bool.and(U32.is_zero(Search.node_parent(a)),
    Bool.or(U32.is_gt(n, visits), Bool.and(U32.is_eq(n, visits), U32.is_lt(key, best)))))
  do IO<Printed>:
    IO.print("node " ++ U32.show(i) ++ " " ++ U32.show(Search.node_parent(a)) ++ " " ++ U32.show(key)
      ++ " " ++ U32.show(Search.node_depth(a)) ++ " " ++ U32.show(Search.node_status(a))
      ++ " " ++ U32.show(n) ++ " " ++ float_word(Search.node_w(a))
      ++ " " ++ float_word(Search.node_prior(a)) ++ " " ++ float_word(Search.node_value(a))
      ++ " " ++ U32.show(Search.node_first(a)) ++ " " ++ U32.show(Search.node_count(a))
      ++ " " ++ board_text(Search.node_board(a)))
    return Printed{pool, Bool.pick(U32, better, key, best), Bool.pick(U32, better, n, visits),
      claim, Bool.pick(U32, better_positive, key, positive),
      Bool.pick(U32, better_positive, n, positive_visits)}

def print_step(+i: U32, state: Printed) -> IO(Printed):
  Printed{pool, best, visits, claim, positive, positive_visits} = state
  node_print(i, best, visits, claim, positive, positive_visits, Array.get(Search.Node, pool, i))

def print_nodes(n: Nat, +i: U32, state: Printed) -> IO(Unit):
  match n:
    case 0n:
      Printed{pool, best, visits, claim, +positive, positive_visits} = state
      option = Bool.pick(U32, U32.is_eq(positive, 4294967295), Search.claim_key(), positive)
      IO.print("best " ++ U32.show(Bool.pick(U32, claim, option, best)))
    case 1n+p:
      do IO<Unit>:
        next : Printed <- print_step(i, state)
        print_nodes(p, U32.inc(i), next)

type Connection is Type:
  Connection{board: Chess.Board, table: Array<U64>, last_epoch: U32}

def finish(board: Chess.Board, r: Live) -> IO(Connection):
  Live{table, tree} = r
  Search.Tree{pool, +used, cap, n, pending, +epoch, request, stop, limit} = tree
  do IO<Connection>:
    IO.print("result " ++ U32.show(epoch) ++ " " ++ U32.show(n) ++ " " ++ U32.show(used)
      ++ " " ++ U32.show(stop) ++ " " ++ U32.show(pending) ++ " " ++ U32.show(request))
    print_nodes(U32.to_nat(used), 0, Printed{pool, 4294967295, 0, False{}, 4294967295, 0})
    return Connection{board, table, epoch}

def begin_valid(valid: Bool, +board: Chess.Board, table: Array<U64>, config: Config) -> IO(Connection):
  match valid:
    case False{}: IO.die(Connection, 2, "session epoch must increase")
    case True{}:
      Config{epoch, simulations, capacity, depth} = config
      do IO<Connection>:
        done : Live <- run(U32.to_nat(simulations), (Live{table, Search.start(board, capacity, epoch, depth)}, False{}))
        finish(board, done)

def config_epoch(c: Config) -> U32:
  Config{epoch, simulations, capacity, depth} = c
  epoch

def begin(conn: Connection, +config: Config) -> IO(Connection):
  Connection{board, table, last} = conn
  begin_valid(U32.is_gt(config_epoch(config), last), board, table, config)

def command(quit: Bool, conn: Connection, config: Config) -> IO(Connection & Bool):
  match quit:
    case True{}: IO.pure(Connection & Bool, (conn, True{}))
    case False{}:
      do IO<Connection & Bool>:
        next : Connection <- begin(conn, config)
        return (next, False{})

def dispatch(conn: Connection, +config: Config) -> IO(Connection & Bool):
  command(U32.is_zero(config_epoch(config)), conn, config)

# Advance only between epochs, after the previous tree/pending reply is gone.
# Match the exact generated key (including promotion/flags), never a truncated
# or host-decoded move. The host retains pre-root neural history separately.
type AdvanceMove is Data:
  NoMove{}
  ChosenMove{move: Chess.Ply}

def advance_choose(equal: Bool, move: Chess.Ply, rest: AdvanceMove) -> AdvanceMove:
  match equal:
    case True{}: ChosenMove{move}
    case False{}: rest

def advance_find(ms: List<&2, Chess.Ply>, +key: U32) -> AdvanceMove:
  match ms:
    case Nil{}: NoMove{}
    case Con{+move, rest}:
      advance_choose(U32.is_eq(Search.key(move), key), move, advance_find(rest, key))

def advance_apply(board: Chess.Board, choice: AdvanceMove) -> Chess.Board & Bool:
  match choice:
    case NoMove{}: (board, False{})
    case ChosenMove{move}: (Chess.make_move(board, move), True{})

def advance_emit(expected: U32, epoch: U32, key: U32, status: U32, c: Connection) -> IO(Connection & Bool):
  Connection{+board, table, +last} = c
  do IO<Connection & Bool>:
    IO.print("advance_result " ++ U32.show(expected) ++ " " ++ U32.show(epoch)
      ++ " " ++ U32.show(key) ++ " " ++ U32.show(status) ++ " " ++ U32.show(last))
    IO.print("board " ++ board_text(board))
    return (Connection{board, table, last}, False{})

def advance_selected(expected: U32, +epoch: U32, key: U32, last: U32,
  table: Array<U64>, r: Chess.Board & Bool) -> IO(Connection & Bool):
  (board, +accepted) = r
  advance_emit(expected, epoch, key, Bool.pick(U32, accepted, 0, 2),
    Connection{board, table, Bool.pick(U32, accepted, epoch, last)})

def advance_generated(expected: U32, epoch: U32, +key: U32, last: U32,
  board: Chess.Board, r: Array<U64> & List<&2, Chess.Ply>) -> IO(Connection & Bool):
  (table, moves) = r
  advance_selected(expected, epoch, key, last, table, advance_apply(board, advance_find(moves, key)))

def advance_valid(valid: Bool, expected: U32, epoch: U32, key: U32, c: Connection) -> IO(Connection & Bool):
  match valid:
    case False{}: advance_emit(expected, epoch, key, 1, c)
    case True{}:
      Connection{+board, table, last} = c
      advance_generated(expected, epoch, key, last, board, Chess.legal_moves(table, board))

def advance(+expected: U32, +epoch: U32, key: U32, c: Connection) -> IO(Connection & Bool):
  Connection{board, table, +last} = c
  advance_valid(Bool.and(U32.is_eq(expected, last), U32.is_gt(epoch, last)),
    expected, epoch, key, Connection{board, table, last})

def command_dispatch(conn: Connection, command: Command) -> IO(Connection & Bool):
  Command{kind, expected, epoch, simulations, capacity, depth, key} = command
  match kind:
    case 0: dispatch(conn, Config{epoch, simulations, capacity, depth})
    case 1: advance(expected, epoch, key, conn)
    case _: IO.die(Connection & Bool, 2, "unknown session command")

def serve(n: Nat, r: Connection & Bool) -> IO(Unit):
  match n:
    case 0n: IO.print("session_limit")
    case 1n+p:
      (conn, stop) = r
      match stop:
        case True{}: IO.print("bye")
        case False{}:
          do IO<Unit>:
            IO.print("ready")
            command : Command <- Command.read()
            next : Connection & Bool <- command_dispatch(conn, command)
            serve(p, next)

def validated(board: Chess.Board, r: Array<U64> & Bool) -> IO(Unit):
  (table, invalid) = r
  match invalid:
    case True{}: IO.die(Unit, 2, "invalid input: nonmoving king is attacked")
    case False{}: serve(1024n, (Connection{board, table, 0}, False{}))

def start(job: Job) -> IO(Unit):
  Job{+board, table, depth, mode, seed} = job
  validated(board, Chess.in_check(table, board, U32.xor(Chess.get_turn(board), 1)))

def main() -> IO(Unit):
  do IO<Unit>:
    job : Job <- Job.load()
    start(job)
