# Opt-in matched-work driver. Parsing and priming precede the operation timer.
# No independent regeneration in the timed loop; direct and cached arms share it.
import Base
import ../legal_probe/Chess.bend as C
import ../standalone/Position.bend as Pos
import ../standalone/Protocol.bend as Protocol
import ../standalone/Text.bend as Text
import ../standalone/Tables.bend as Tables
import ./MoveCache.bend as M

type Store is Type:
  Direct{}
  Cached{cache: M.Cache}

type Stats is Data:
  Stats{requests: U32, moves: U32, checksum: U32, hits: U32, fills: U32, bypasses: U32}

type State is Type:
  State{tables: Array<U64>, store: Store, stats: Stats}

def empty() -> Stats:
  Stats{0, 0, 0, 0, 0, 0}

def create(result: Maybe<&1, M.Cache>) -> Store:
  match result:
    case None{}: Direct{}
    case Some{cache}: Cached{cache}

def store(arm: U32, bits: U32) -> Store:
  match arm:
    case 0: Direct{}
    case _: create(M.new(bits))

def pack(ply: C.Ply) -> U32:
  C.Ply{src, dst, promotion, flag} = ply
  U32.or(src, U32.or(U32.shln(dst, 6n), U32.or(U32.shln(promotion, 12n), U32.shln(flag, 15n))))

def fold(sum: U32, word: U32) -> U32:
  U32.add(U32.mul(sum, 1664525), word)

def checked(ms: List<&2, C.Ply>, sum: U32, +count: U32) -> U32 & U32:
  match ms:
    case Nil{}: (fold(sum, U32.add(131072, count)), count)
    case Con{m, tail}: checked(tail, fold(sum, U32.inc(pack(m))), U32.inc(count))

def routed(route: M.Access, requests: U32, moves: U32, sum: U32,
  hits: U32, fills: U32, bypasses: U32, count: U32) -> Stats:
  match route:
    case M.Hit{}: Stats{U32.inc(requests), U32.add(moves, count), sum, U32.inc(hits), fills, bypasses}
    case M.Filled{}: Stats{U32.inc(requests), U32.add(moves, count), sum, hits, U32.inc(fills), bypasses}
    case M.Bypassed{}: Stats{U32.inc(requests), U32.add(moves, count), sum, hits, fills, U32.inc(bypasses)}

def updated(route: M.Access, old: Stats, result: U32 & U32) -> Stats:
  Stats{requests, moves, ignored, hits, fills, bypasses} = old
  (sum, count) = result
  routed(route, requests, moves, sum, hits, fills, bypasses, count)

def move_text(ms: List<&2, C.Ply>) -> String:
  match ms:
    case Nil{}: ""
    case Con{m, rest}: " " ++ U32.show(pack(m)) ++ move_text(rest)

def trace(diag: U32, number: U32, ms: List<&2, C.Ply>) -> IO(Unit):
  match diag:
    case 0: IO.pure(Unit, Unit{})
    case _: IO.print("trace " ++ U32.show(number) ++ move_text(ms))

def accept(diag: U32, tables: Array<U64>, store: Store, +stats: Stats,
  +ms: List<&2, C.Ply>, route: M.Access) -> IO(State):
  Stats{requests, moves, sum, hits, fills, bypasses} = stats
  do IO<State>:
    done : Unit <- trace(diag, requests, ms)
    return State{tables, store, updated(route, stats, checked(ms, sum, 0))}

def cached_result(diag: U32, stats: Stats, result: M.Result) -> IO(State):
  M.Result{tables, cache, ms, access} = result
  accept(diag, tables, Cached{cache}, stats, ms, access)

def direct_result(diag: U32, stats: Stats, result: Array<U64> & List<&2, C.Ply>) -> IO(State):
  (tables, ms) = result
  accept(diag, tables, Direct{}, stats, ms, M.Bypassed{})

def request(diag: U32, board: C.Board, state: State) -> IO(State):
  State{tables, owner, stats} = state
  match owner:
    case Direct{}: direct_result(diag, stats, C.legal_moves(tables, board))
    case Cached{cache}: cached_result(diag, stats, M.legal_moves(tables, cache, board))

def tape(boards: List<&2, C.Board>, +diag: U32, state: State) -> IO(State):
  match boards:
    case Nil{}: IO.pure(State, state)
    case Con{b, tail}:
      do IO<State>:
        state : State <- request(diag, b, state)
        tape(tail, diag, state)

def restart(cold: U32, arm: U32, bits: U32, state: State) -> State:
  match cold:
    case 0: state
    case _:
      State{tables, owner, stats} = state
      State{tables, store(arm, bits), stats}

def repeat(n: Nat, +arm: U32, +bits: U32, +cold: U32, +diag: U32,
  +boards: List<&2, C.Board>, state: State) -> IO(State):
  match n:
    case 0n: IO.pure(State, state)
    case 1n+p:
      do IO<State>:
        state : State <- tape(boards, diag, restart(cold, arm, bits, state))
        repeat(p, arm, bits, cold, diag, boards, state)

def report_at(started: Nat, ended: Nat, state: State) -> IO(Unit):
  State{tables, owner, stats} = state
  Stats{requests, moves, sum, hits, fills, bypasses} = stats
  do IO<Unit>:
    IO.print("sample " ++ Nat.show(Nat.sub(ended, started)) ++ " " ++ U32.show(requests)
      ++ " " ++ U32.show(moves) ++ " " ++ U32.show(sum) ++ " " ++ U32.show(hits)
      ++ " " ++ U32.show(fills) ++ " " ++ U32.show(bypasses))

def report(started: Nat, state: State) -> IO(Unit):
  # Retain the owners until after the ending timestamp; final destruction is not timed.
  do IO<Unit>:
    ended : Nat <- IO.now()
    report_at(started, ended, state)

def timed(arm: U32, bits: U32, rounds: U32, cold: U32, diag: U32,
  boards: List<&2, C.Board>, state: State) -> IO(Unit):
  State{tables, owner, ignored} = state
  do IO<Unit>:
    started : Nat <- IO.now()
    result : State <- repeat(U32.to_nat(rounds), arm, bits, cold, diag, boards, State{tables, owner, empty()})
    report(started, result)

type Input is Type:
  Input{tables: Array<U64>, boards: List<&2, C.Board>}

def reversed(boards: List<&2, C.Board>, acc: List<&2, C.Board>) -> List<&2, C.Board>:
  match boards:
    case Nil{}: acc
    case Con{b, tail}: reversed(tail, Con{b, acc})

def decoded(boards: List<&2, C.Board>, result: Array<U64> & Maybe<&2, Pos.Game>) -> IO(Input):
  (tables, maybe) = result
  match maybe:
    case None{}: IO.die(Input, 2, "invalid benchmark board")
    case Some{game}: IO.pure(Input, Input{tables, Con{Pos.board(game), boards}})

def parse_one(line: String, state: Input) -> IO(Input):
  Input{tables, boards} = state
  decoded(boards, Protocol.position(Protocol.parse_position(Text.words(line)), tables))

def parse(lines: List<&2, String>, state: Input) -> IO(Input):
  match lines:
    case Nil{}:
      Input{tables, boards} = state
      IO.pure(Input, Input{tables, reversed(boards, Nil{})})
    case Con{line, rest}:
      do IO<Input>:
        state : Input <- parse_one(line, state)
        parse(rest, state)

def maybe_prime(text: String, tables: Array<U64>) -> IO(Input):
  match text:
    case SCon{'-', SNil{}}: IO.pure(Input, Input{tables, Nil{}})
    case _: parse(String.split(text, ';'), Input{tables, Nil{}})

def loaded(+arm: U32, +bits: U32, +rounds: U32, +cold: U32, +diag: U32,
  prime: List<&2, C.Board>, input: Input) -> IO(Unit):
  Input{tables, boards} = input
  do IO<Unit>:
    IO.print("configuration " ++ U32.show(arm) ++ " " ++ U32.show(bits) ++ " " ++ U32.show(rounds)
      ++ " " ++ U32.show(cold) ++ " " ++ U32.show(diag))
    state : State <- tape(prime, 0, State{tables, store(arm, bits), empty()})
    timed(arm, bits, rounds, cold, diag, boards, state)

def primed(arm: U32, bits: U32, rounds: U32, cold: U32, diag: U32, body: String, input: Input) -> IO(Unit):
  Input{tables, prime} = input
  do IO<Unit>:
    parsed : Input <- parse(String.split(body, ';'), Input{tables, Nil{}})
    loaded(arm, bits, rounds, cold, diag, prime, parsed)

def run(arm: U32, bits: U32, rounds: U32, cold: U32, diag: U32,
  prime: String, body: String, valid: Bool) -> IO(Unit):
  match valid:
    case False{}: IO.die(Unit, 2, "invalid benchmark bounds")
    case True{}:
      do IO<Unit>:
        input : Input <- maybe_prime(prime, Tables.build())
        primed(arm, bits, rounds, cold, diag, body, input)

def cold_prime(cold: U32, prime: String) -> Bool:
  match cold:
    case 0: True{}
    case _:
      match prime:
        case SCon{'-', SNil{}}: True{}
        case _: False{}

def text_bounds(+prime: String, body: String, cold: U32) -> Bool:
  Bool.and(cold_prime(cold, prime), Bool.and(
    Nat.is_le(List.length(&2, String, String.split(prime, ';')), 128n),
    Nat.is_le(List.length(&2, String, String.split(body, ';')), 128n)))

def numbers(a: Maybe<&2, U32>, b: Maybe<&2, U32>, r: Maybe<&2, U32>,
  c: Maybe<&2, U32>, d: Maybe<&2, U32>, +prime: String, +body: String) -> IO(Unit):
  match a b r c d:
    case Some{+arm} Some{+bits} Some{+rounds} Some{+cold} Some{+diag}:
      run(arm, bits, rounds, cold, diag, prime, body,
        Bool.and(U32.is_le(arm, 1), Bool.and(U32.is_ge(bits, 1), Bool.and(U32.is_le(bits, 16),
        Bool.and(U32.is_le(rounds, 65536), Bool.and(U32.is_le(cold, 1), Bool.and(U32.is_le(diag, 1),
        Bool.and(text_bounds(prime, body, cold), Bool.or(U32.is_zero(diag), U32.is_le(rounds, 3))))))))))
    case _ _ _ _ _: IO.die(Unit, 2, "invalid benchmark configuration")

def config(words: List<&2, String>, prime: String, body: String) -> IO(Unit):
  match words:
    case Con{a, Con{b, Con{r, Con{c, Con{d, Nil{}}}}}}:
      numbers(Text.number(a), Text.number(b), Text.number(r), Text.number(c), Text.number(d), prime, body)
    case _: IO.die(Unit, 2, "invalid benchmark configuration")

def input(parts: List<&2, String>) -> IO(Unit):
  match parts:
    case Con{header, Con{prime, Con{body, Nil{}}}}: config(Text.words(header), prime, body)
    case _: IO.die(Unit, 2, "invalid benchmark input")

def environment(value: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match value:
    case Fail{error}: IO.die(Unit, 2, "missing DEEPFIN_MOVE_CACHE_BENCHMARK")
    case Done{text}: input(String.split(text, '|'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_MOVE_CACHE_BENCHMARK")
    environment(text)
