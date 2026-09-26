# Parse and legally apply moves using the existing native Bend position path.
# Inputs are position commands, not precomputed hashes/identity records.
import Base
import ../legal_probe/Chess.bend as C
import ../standalone/Position.bend as Pos
import ../standalone/Protocol.bend as Protocol
import ../standalone/Text.bend as Text
import ../standalone/Tables.bend as Tables
import ./PositionIndex.bend as P
import ./BoardIndex.bend as B

type State is Type:
  State{tables: Array<U64>, index: P.Index}

def word(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def fields_text(f: P.Identity) -> String:
  P.Identity{p, n, b, r, q, k, w, x, turn, castling, ep} = f
  word(p) ++ " " ++ word(n) ++ " " ++ word(b) ++ " " ++ word(r) ++ " " ++ word(q)
    ++ " " ++ word(k) ++ " " ++ word(w) ++ " " ++ word(x) ++ " " ++ U32.show(turn)
    ++ " " ++ U32.show(castling) ++ " " ++ U32.show(ep)

def describe(+ordinal: U32, +g: Pos.Game) -> IO(Unit):
  Pos.Game{+b, half, full, history} = g
  +fields = B.identity(b)
  do IO<Unit>:
    IO.print("identity " ++ U32.show(ordinal) ++ " " ++ fields_text(fields))
    IO.print("hash " ++ U32.show(ordinal) ++ " " ++ word(B.hash(fields)))
    IO.print("context " ++ U32.show(ordinal) ++ " " ++ U32.show(C.get_ep(b)) ++ " "
      ++ U32.show(half) ++ " " ++ U32.show(full) ++ " " ++ Nat.show(List.length(&2, Pos.Past, history)))

def sized(text: String, result: P.Index & U32) -> IO(P.Index):
  (index, size) = result
  do IO<P.Index>:
    IO.print(text ++ " size " ++ U32.show(size))
    return index

def inserted(n: U32, result: P.Index & P.Outcome) -> IO(P.Index):
  (index, outcome) = result
  match outcome:
    case P.Assigned{id}: sized("intern " ++ U32.show(n) ++ " assigned " ++ U32.show(id), P.length(index))
    case P.Known{id}: sized("intern " ++ U32.show(n) ++ " known " ++ U32.show(id), P.length(index))
    case P.Full{}: sized("intern " ++ U32.show(n) ++ " full", P.length(index))

def looked(n: U32, result: P.Index & Maybe<&2, U32>) -> IO(P.Index):
  (index, found) = result
  match found:
    case None{}: sized("get " ++ U32.show(n) ++ " missing", P.length(index))
    case Some{id}: sized("get " ++ U32.show(n) ++ " value " ++ U32.show(id), P.length(index))

def position(+n: U32, index: P.Index, result: Array<U64> & Maybe<&2, Pos.Game>) -> IO(State):
  (tables, maybe) = result
  match maybe:
    case None{}: IO.die(State, 2, "invalid native board-index position")
    case Some{+g}:
      do IO<State>:
        done : Unit <- describe(n, g)
        index : P.Index <- inserted(n, B.intern(index, Pos.board(g)))
        index : P.Index <- looked(n, B.get(index, Pos.board(g)))
        return State{tables, index}

def finished(result: P.Index & U32) -> IO(Unit):
  (index, size) = result
  IO.print("end " ++ U32.show(size))

def run(lines: List<&2, String>, +n: U32, state: State) -> IO(Unit):
  match lines state:
    case Nil{} State{tables, index}: finished(P.length(index))
    case Con{line, rest} State{tables, index}:
      do IO<Unit>:
        next : State <- position(n, index,
          Protocol.position(Protocol.parse_position(Text.words(line)), tables))
        run(rest, U32.inc(n), next)

def created(lines: List<&2, String>, result: Maybe<&1, P.Index>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "invalid native board-index capacity")
    case Some{index}: run(lines, 0, State{Tables.build(), index})

def admitted(bits: U32, lines: List<&2, String>, valid: Bool) -> IO(Unit):
  match valid:
    case False{}: IO.die(Unit, 2, "native board-index input exceeds limit")
    case True{}: created(lines, P.new(bits))

def configured(bits: Maybe<&2, U32>, +lines: List<&2, String>) -> IO(Unit):
  match bits:
    case None{}: IO.die(Unit, 2, "invalid native board-index capacity")
    case Some{n}: admitted(n, lines, Nat.is_le(List.length(&2, String, lines), 128n))

def input(lines: List<&2, String>) -> IO(Unit):
  match lines:
    case Nil{}: IO.die(Unit, 2, "missing native board-index input")
    case Con{bits, tail}: configured(Text.number(bits), tail)

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_BOARD_INDEX")
    case Done{text}: input(String.split(text, ';'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_BOARD_INDEX")
    environment(text)
