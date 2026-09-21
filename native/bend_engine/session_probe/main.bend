import Base
import ../legal_probe/Chess.bend as Chess
import ./Search.bend as Search

type Job is Type:
  Job{board: Chess.Board, table: Array<U64>, depth: U32, mode: U32, seed: U32}

def Job.load() -> IO(Job):
  import "../legal_probe/input.c"

type Config is Data:
  Config{epoch: U32, simulations: U32, capacity: U32, depth: U32}

def Config.read() -> IO(Config):
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

def ticket_send(ticket: Search.Ticket) -> IO(Unit):
  Search.Ticket{epoch, request, node, board, +moves, needed} = ticket
  do IO<Unit>:
    IO.print("eval " ++ U32.show(epoch) ++ " " ++ U32.show(request) ++ " " ++ U32.show(node)
      ++ " " ++ U32.show(U32.from_nat(List.length(&2, Chess.Ply, moves))))
    IO.print("board " ++ board_text(board))
    moves_print(moves)

type Live is Type:
  Live{table: Array<U64>, tree: Search.Tree}

def live_flag(table: Array<U64>, tree: Search.Tree) -> Live & Bool:
  Search.Tree{pool, used, cap, n, pending, epoch, request, +stop, limit} = tree
  (Live{table, Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit}}, Bool.not(U32.is_zero(stop)))

def step_read(needed: Bool, table: Array<U64>, tree: Search.Tree, +ticket: Search.Ticket) -> IO(Live & Bool):
  match needed:
    case False{}: IO.pure(Live & Bool, live_flag(table, tree))
    case True{}:
      do IO<Live & Bool>:
        ticket_send(ticket)
        reply : Search.Reply <- Reply.read()
        return live_flag(table, Search.resume(tree, ticket, reply))

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
  Printed{pool: Array<Search.Node>, best: U32, visits: U32}

def node_print(+i: U32, +best: U32, +visits: U32, r: Array<Search.Node> & Search.Node) -> IO(Printed):
  (pool, +a) = r
  +key = Search.key(Search.node_move(a))
  +n = Search.node_n(a)
  +better = Bool.and(Bool.not(U32.is_zero(i)), Bool.and(U32.is_zero(Search.node_parent(a)),
    Bool.or(U32.is_gt(n, visits), Bool.and(U32.is_eq(n, visits), U32.is_lt(key, best)))))
  do IO<Printed>:
    IO.print("node " ++ U32.show(i) ++ " " ++ U32.show(Search.node_parent(a)) ++ " " ++ U32.show(key)
      ++ " " ++ U32.show(Search.node_depth(a)) ++ " " ++ U32.show(Search.node_status(a))
      ++ " " ++ U32.show(n) ++ " " ++ float_word(Search.node_w(a))
      ++ " " ++ float_word(Search.node_prior(a)) ++ " " ++ float_word(Search.node_value(a))
      ++ " " ++ U32.show(Search.node_first(a)) ++ " " ++ U32.show(Search.node_count(a))
      ++ " " ++ board_text(Search.node_board(a)))
    return Printed{pool, Bool.pick(U32, better, key, best), Bool.pick(U32, better, n, visits)}

def print_step(+i: U32, state: Printed) -> IO(Printed):
  Printed{pool, best, visits} = state
  node_print(i, best, visits, Array.get(Search.Node, pool, i))

def print_nodes(n: Nat, +i: U32, state: Printed) -> IO(Unit):
  match n:
    case 0n:
      Printed{pool, best, visits} = state
      IO.print("best " ++ U32.show(best))
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
    print_nodes(U32.to_nat(used), 0, Printed{pool, 4294967295, 0})
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
            config : Config <- Config.read()
            next : Connection & Bool <- dispatch(conn, config)
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
