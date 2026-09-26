# Test transport for canonical-field interning, not a chess FEN parser.
import Base
import ./PositionIndex.bend as P

def sized(prefix: String, result: P.Index & U32) -> IO(P.Index):
  (index, size) = result
  do IO<P.Index>:
    IO.print(prefix ++ " size " ++ U32.show(size))
    return index

def interned(prefix: String, result: P.Index & P.Outcome) -> IO(P.Index):
  (index, outcome) = result
  match outcome:
    case P.Assigned{id}: sized(prefix ++ " assigned " ++ U32.show(id), P.length(index))
    case P.Known{id}: sized(prefix ++ " known " ++ U32.show(id), P.length(index))
    case P.Full{}: sized(prefix ++ " full", P.length(index))

def found(prefix: String, result: P.Index & Maybe<&2, U32>) -> IO(P.Index):
  (index, value) = result
  match value:
    case None{}: sized(prefix ++ " missing", P.length(index))
    case Some{id}: sized(prefix ++ " value " ++ U32.show(id), P.length(index))

def operate(action: String, number: U32, hash: U64, identity: P.Identity, index: P.Index) -> IO(P.Index):
  match action:
    case SCon{'r', SNil{}}: interned("r " ++ U32.show(number), P.intern(index, hash, identity))
    case SCon{'g', SNil{}}: found("g " ++ U32.show(number), P.get(index, hash, identity))
    case _: IO.die(P.Index, 2, "invalid position-index action")

def parsed(action: String, number: U32, values: List<&2, U32>, index: P.Index) -> IO(P.Index):
  match values:
    case Con{hh, Con{hl, Con{ph, Con{pl, Con{nh, Con{nl, Con{bh, Con{bl, Con{rh, Con{rl, Con{qh, Con{ql, Con{kh, Con{kl, Con{wh, Con{wl, Con{xh, Con{xl, Con{turn, Con{castling, Con{ep, Nil{}}}}}}}}}}}}}}}}}}}}}}:
      operate(action, number, U64{hl, hh}, P.Identity{U64{pl, ph}, U64{nl, nh}, U64{bl, bh}, U64{rl, rh}, U64{ql, qh}, U64{kl, kh}, U64{wl, wh}, U64{xl, xh}, turn, castling, ep}, index)
    case _: IO.die(P.Index, 2, "invalid position-index fields")

def number(value: Maybe<&2, U32>) -> IO(U32):
  match value:
    case None{}: IO.die(U32, 2, "invalid position-index number")
    case Some{n}: IO.pure(U32, n)

def numbers(words: List<&2, String>) -> IO(List<&2, U32>):
  match words:
    case Nil{}: IO.pure(List<&2, U32>, Nil{})
    case Con{word, tail}:
      do IO<List<&2, U32>>:
        n : U32 <- number(U32.read(word))
        rest : List<&2, U32> <- numbers(tail)
        return Con{n, rest}

def op(words: List<&2, String>, n: U32, index: P.Index) -> IO(P.Index):
  match words:
    case Nil{}: IO.die(P.Index, 2, "missing position-index action")
    case Con{action, tail}:
      do IO<P.Index>:
        values : List<&2, U32> <- numbers(tail)
        parsed(action, n, values, index)

def finish(result: P.Index & U32) -> IO(Unit):
  (index, size) = result
  IO.print("end " ++ U32.show(size))

def run(lines: List<&2, String>, +n: U32, index: P.Index) -> IO(Unit):
  match lines:
    case Nil{}: finish(P.length(index))
    case Con{line, tail}:
      do IO<Unit>:
        index : P.Index <- op(String.split(line, ' '), n, index)
        run(tail, U32.inc(n), index)

def ready(bits: U32, lines: List<&2, String>, result: Maybe<&1, P.Index>) -> IO(Unit):
  match result:
    case None{}: IO.print("invalid " ++ U32.show(bits))
    case Some{index}:
      do IO<Unit>:
        IO.print("begin " ++ U32.show(bits))
        run(lines, 0, index)

def configured(+bits: U32, lines: List<&2, String>) -> IO(Unit):
  ready(bits, lines, P.new(bits))

def config(lines: List<&2, String>) -> IO(Unit):
  match lines:
    case Nil{}: IO.die(Unit, 2, "missing position-index configuration")
    case Con{word, tail}:
      do IO<Unit>:
        bits : U32 <- number(U32.read(word))
        configured(bits, tail)

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_POSITION_INDEX")
    case Done{text}: config(String.split(text, ';'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_POSITION_INDEX")
    environment(text)
