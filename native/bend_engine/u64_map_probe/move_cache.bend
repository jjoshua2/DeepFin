# Source-only consumer of MoveCache using the real native position/move path.
import Base
import ../legal_probe/Chess.bend as C
import ../standalone/Position.bend as Pos
import ../standalone/Protocol.bend as Protocol
import ../standalone/Text.bend as Text
import ../standalone/Tables.bend as Tables
import ./MoveCache.bend as M

type State is Type:
  State{tables: Array<U64>, cache: M.Cache}

def word(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def board_text(+b: C.Board) -> String:
  word(C.get_pawns(b)) ++ " " ++ word(C.get_knights(b)) ++ " " ++ word(C.get_bishops(b))
    ++ " " ++ word(C.get_rooks(b)) ++ " " ++ word(C.get_queens(b)) ++ " " ++ word(C.get_kings(b))
    ++ " " ++ word(C.get_white(b)) ++ " " ++ word(C.get_black(b)) ++ " " ++ U32.show(C.get_turn(b))
    ++ " " ++ U32.show(C.get_rights(b)) ++ " " ++ U32.show(C.get_ep(b))

def move_text(m: C.Ply) -> String:
  C.Ply{src, dst, promotion, flag} = m
  U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(promotion) ++ " " ++ U32.show(flag)

def access_text(access: M.Access) -> String:
  match access:
    case M.Hit{}: "hit"
    case M.Filled{}: "filled"
    case M.Bypassed{}: "bypassed"

def moves_text(+n: U32, +b: C.Board, moves: List<&2, C.Ply>) -> IO(Unit):
  match moves:
    case Nil{}: IO.print("end " ++ U32.show(n))
    case Con{+move, rest}:
      do IO<Unit>:
        IO.print("move " ++ U32.show(n) ++ " " ++ move_text(move) ++ " " ++ board_text(C.make_move(b, move)))
        moves_text(n, b, rest)

def report_game(+n: U32, tables: Array<U64>, cache: M.Cache, +moves: List<&2, C.Ply>,
  access: M.Access, game: Pos.Game) -> IO(State):
  Pos.Game{board, half, full, history} = game
  do IO<State>:
    IO.print("request " ++ U32.show(n) ++ " " ++ access_text(access) ++ " " ++ Nat.show(List.length(&2, C.Ply, moves)))
    IO.print("context " ++ U32.show(n) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full)
      ++ " " ++ Nat.show(List.length(&2, Pos.Past, history)))
    done : Unit <- moves_text(n, board, moves)
    return State{tables, cache}

def direct_moves(+n: U32, moves: List<&2, C.Ply>) -> IO(Unit):
  match moves:
    case Nil{}: IO.print("reference-end " ++ U32.show(n))
    case Con{m, rest}:
      do IO<Unit>:
        IO.print("reference " ++ U32.show(n) ++ " " ++ move_text(m))
        direct_moves(n, rest)

def checked(+n: U32, game: Pos.Game, cache: M.Cache, moves: List<&2, C.Ply>, access: M.Access,
  baseline: Array<U64> & List<&2, C.Ply>) -> IO(State):
  (tables, reference) = baseline
  do IO<State>:
    # Qualification only: independently call the unmodified native generator
    # to check that cached ordering is unchanged. This is NOT a timing driver.
    done : Unit <- direct_moves(n, reference)
    report_game(n, tables, cache, moves, access, game)

def report(n: U32, +game: Pos.Game, result: M.Result) -> IO(State):
  M.Result{tables, cache, moves, access} = result
  checked(n, game, cache, moves, access, C.legal_moves(tables, Pos.board(game)))

def position(n: U32, cache: M.Cache, result: Array<U64> & Maybe<&2, Pos.Game>) -> IO(State):
  (tables, game) = result
  match game:
    case None{}: IO.die(State, 2, "invalid move-cache position")
    case Some{+g}: report(n, g, M.legal_moves(tables, cache, Pos.board(g)))

def run(lines: List<&2, String>, +n: U32, state: State) -> IO(Unit):
  match lines state:
    case Nil{} State{tables, cache}: IO.print("done " ++ U32.show(n))
    case Con{line, rest} State{tables, cache}:
      do IO<Unit>:
        next : State <- position(n, cache, Protocol.position(Protocol.parse_position(Text.words(line)), tables))
        run(rest, U32.inc(n), next)

def created(lines: List<&2, String>, result: Maybe<&1, M.Cache>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "invalid move-cache capacity")
    case Some{cache}: run(lines, 0, State{Tables.build(), cache})

def bounded(bits: U32, lines: List<&2, String>, okay: Bool) -> IO(Unit):
  match okay:
    case False{}: IO.die(Unit, 2, "move-cache input exceeds limit")
    case True{}: created(lines, M.new(bits))

def configured(bits: Maybe<&2, U32>, +lines: List<&2, String>) -> IO(Unit):
  match bits:
    case None{}: IO.die(Unit, 2, "invalid move-cache capacity")
    case Some{value}: bounded(value, lines, Nat.is_le(List.length(&2, String, lines), 128n))

def input(lines: List<&2, String>) -> IO(Unit):
  match lines:
    case Nil{}: IO.die(Unit, 2, "missing move-cache input")
    case Con{bits, rest}: configured(Text.number(bits), rest)

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_MOVE_CACHE")
    case Done{text}: input(String.split(text, ';'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_MOVE_CACHE")
    environment(text)
