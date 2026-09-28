import Base
import ./Chess.bend as Chess

type Job is Type:
  Job{board: Chess.Board, table: Array<U64>, depth: U32, mode: U32, seed: U32}

def Job.load() -> IO(Job):
  import "./input.c"

def word(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def board_text(+b: Chess.Board) -> String:
  word(Chess.get_pawns(b)) ++ " " ++ word(Chess.get_knights(b)) ++ " " ++ word(Chess.get_bishops(b))
  ++ " " ++ word(Chess.get_rooks(b)) ++ " " ++ word(Chess.get_queens(b)) ++ " " ++ word(Chess.get_kings(b))
  ++ " " ++ word(Chess.get_white(b)) ++ " " ++ word(Chess.get_black(b)) ++ " " ++ U32.show(Chess.get_turn(b))
  ++ " " ++ U32.show(Chess.get_rights(b)) ++ " " ++ U32.show(Chess.get_ep(b))

def move_text(m: Chess.Ply) -> String:
  Chess.Ply{src, dst, promotion, flag} = m
  U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(promotion) ++ " " ++ U32.show(flag)

def total_report(r: Array<U64> & U64) -> IO(Unit):
  (table, count) = r
  IO.print("nodes " ++ word(count))

def divide_print(m: Chess.Ply, total: U64, r: Array<U64> & U64) -> IO(Array<U64> & U64):
  (table, +count) = r
  do IO<Array<U64> & U64>:
    IO.print("divide " ++ move_text(m) ++ " " ++ word(count))
    return (table, U64.add(total, count))

def divide_step(b: Chess.Board, depth: Nat, +m: Chess.Ply, r: Array<U64> & U64) -> IO(Array<U64> & U64):
  (table, total) = r
  divide_print(m, total, Chess.perft(depth, [Chess.make_move(b, m)], table, U64.zero()))

def divide(ms: List<&2, Chess.Ply>, +b: Chess.Board, +depth: Nat, r: Array<U64> & U64) -> IO(Unit):
  match ms:
    case Nil{}: total_report(r)
    case Con{m, rest}:
      do IO<Unit>:
        next : Array<U64> & U64 <- divide_step(b, depth, m, r)
        divide(rest, b, depth, next)

def perft_start(depth: Nat, b: Chess.Board, r: Array<U64> & List<&2, Chess.Ply>) -> IO(Unit):
  match depth:
    case 0n: IO.print("nodes 0 1")
    case 1n+p:
      (table, ms) = r
      divide(ms, b, p, (table, U64.zero()))

def moves_print(ms: List<&2, Chess.Ply>, +b: Chess.Board) -> IO(Unit):
  match ms:
    case Nil{}: IO.print("end")
    case Con{+m, rest}:
      do IO<Unit>:
        IO.print("move " ++ move_text(m) ++ " " ++ board_text(Chess.make_move(b, m)))
        moves_print(rest, b)

def moves_start(b: Chess.Board, r: Array<U64> & List<&2, Chess.Ply>) -> IO(Unit):
  (table, ms) = r
  moves_print(ms, b)

type RandomState is Type:
  RandomState{board: Chess.Board, table: Array<U64>, seed: U32, stopped: Bool}

def random_seed(+x: U32) -> U32:
  +x = U32.xor(x, U32.shln(x, 13n))
  +x = U32.xor(x, U32.shrn(x, 17n))
  U32.xor(x, U32.shln(x, 5n))

def random_picked(+b: Chess.Board, seed: U32, table: Array<U64>, m: Maybe<&2, Chess.Ply>) -> IO(RandomState):
  match m:
    case None{}: IO.die(RandomState, 1, "random move selection failed")
    case Some{+m}:
      +child = Chess.make_move(b, m)
      do IO<RandomState>:
        IO.print("ply " ++ move_text(m) ++ " " ++ board_text(child))
        return RandomState{child, table, seed, False{}}

def random_terminal(b: Chess.Board, seed: U32, r: Array<U64> & Bool) -> IO(RandomState):
  (table, check) = r
  do IO<RandomState>:
    IO.print(Bool.pick(String, check, "checkmate", "stalemate"))
    return RandomState{b, table, seed, True{}}

def random_moves(+b: Chess.Board, seed: U32, r: Array<U64> & List<&2, Chess.Ply>) -> IO(RandomState):
  (table, ms) = r
  match ms:
    case Nil{}: random_terminal(b, seed, Chess.in_check(table, b, Chess.get_turn(b)))
    case Con{first, rest}:
      +ms = {Con{first, rest} : List<&2, Chess.Ply>}
      +seed = random_seed(seed)
      size = U32.from_nat(List.length(&2, Chess.Ply, ms))
      random_picked(b, seed, table, List.get(&2, Chess.Ply, ms, U32.to_nat(U32.mod(seed, size))))

def random_step(s: RandomState) -> IO(RandomState):
  RandomState{+b, table, seed, stopped} = s
  random_moves(b, seed, Chess.legal_moves(table, b))

def random_finish(s: RandomState) -> IO(Unit):
  RandomState{board, table, seed, stopped} = s
  match stopped:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.print("ply_limit")

def random_play(n: Nat, s: RandomState) -> IO(Unit):
  match n:
    case 0n: random_finish(s)
    case 1n+p:
      match s:
        case RandomState{b, table, seed, stopped}:
          match stopped:
            case True{}: IO.pure(Unit, Unit{})
            case False{}:
              do IO<Unit>:
                next : RandomState <- random_step(RandomState{b, table, seed, False{}})
                random_play(p, next)

def run_mode(mode: U32, depth: U32, +b: Chess.Board, table: Array<U64>, seed: U32) -> IO(Unit):
  match mode:
    case 0: perft_start(U32.to_nat(depth), b, Chess.legal_moves(table, b))
    case 1: moves_start(b, Chess.legal_moves(table, b))
    case _: random_play(U32.to_nat(depth), RandomState{b, table, seed, False{}})

def validated(mode: U32, depth: U32, b: Chess.Board, seed: U32, r: Array<U64> & Bool) -> IO(Unit):
  (table, bad) = r
  match bad:
    case True{}: IO.die(Unit, 2, "invalid input: nonmoving king is attacked")
    case False{}: run_mode(mode, depth, b, table, seed)

def run_job(job: Job) -> IO(Unit):
  Job{+b, table, depth, mode, seed} = job
  validated(mode, depth, b, seed, Chess.in_check(table, b, U32.xor(Chess.get_turn(b), 1)))

def main() -> IO(Unit):
  do IO<Unit>:
    job : Job <- Job.load()
    run_job(job)
