# Test-only full-record population. No direct access to PositionIndex storage.
# Synthetic field vectors, not legal boards or production hash distributions.
import Base
import ./PositionIndex.bend as P

def key(+group: U32) -> U64:
  U64{U32.xor(group, 2779096485), U32.mul(group, 2654435761)}

def word(+i: U32, +field: U32, changed: U32) -> U64:
  U64{U32.xor(U32.mul(i, 2654435761), U32.mul(field, 2246822519)),
    U32.xor(U32.add(i, field), Bool.pick(U32, U32.is_eq(field, changed), 0, 2147483648))}

def metadata(value: U32, field: U32, changed: U32) -> U32:
  U32.xor(value, Bool.to_u32(U32.is_eq(field, changed)))

def identity(+i: U32, +changed: U32) -> P.Identity:
  P.Identity{word(i, 0, changed), word(i, 1, changed), word(i, 2, changed),
    word(i, 3, changed), word(i, 4, changed), word(i, 5, changed),
    word(i, 6, changed), word(i, 7, changed),
    metadata(U32.and(i, 1), 8, changed), metadata(U32.and(i, 15), 9, changed),
    metadata(U32.and(i, 63), 10, changed)}

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

def operate(kind: U32, label: String, hash: U64, fields: P.Identity, index: P.Index) -> IO(P.Index):
  match kind:
    case 0: interned("r " ++ label, P.intern(index, hash, fields))
    case _: found("g " ++ label, P.get(index, hash, fields))

def op(kind: U32, +i: U32, +changed: U32, +group: U32, index: P.Index) -> IO(P.Index):
  operate(kind, U32.show(i) ++ " " ++ U32.show(changed) ++ " " ++ U32.show(group),
    key(group), identity(i, changed), index)

def loop(fuel: Nat, +kind: U32, +i: U32, +stride: U32, +width: U32, index: P.Index) -> IO(P.Index):
  match fuel:
    case 0n: IO.pure(P.Index, index)
    case 1n+p:
      do IO<P.Index>:
        index : P.Index <- op(kind, i, 11, U32.div(i, width), index)
        loop(p, kind, U32.add(i, stride), stride, width, index)

def mismatches(fuel: Nat, +field: U32, +i: U32, +group: U32, index: P.Index) -> IO(P.Index):
  match fuel:
    case 0n: IO.pure(P.Index, index)
    case 1n+p:
      do IO<P.Index>:
        index : P.Index <- op(1, i, field, group, index)
        index : P.Index <- op(0, i, field, group, index)
        mismatches(p, U32.inc(field), i, group, index)

def permuted(fuel: Nat, +ordinal: U32, +mask: U32, +width: U32, index: P.Index) -> IO(P.Index):
  match fuel:
    case 0n: IO.pure(P.Index, index)
    case 1n+p:
      +i = U32.and(U32.mul(ordinal, 73), mask)
      do IO<P.Index>:
        index : P.Index <- op(1, i, 11, U32.div(i, width), index)
        permuted(p, U32.inc(ordinal), mask, width, index)

def finish(result: P.Index & U32) -> IO(Unit):
  (index, size) = result
  IO.print("end " ++ U32.show(size))

def run(bits: U32, +width: U32, +n: U32, index: P.Index) -> IO(Unit):
  do IO<Unit>:
    IO.print("position-scale " ++ U32.show(bits) ++ " " ++ U32.show(width) ++ " " ++ U32.show(n))
    index : P.Index <- op(1, 0, 11, 0, index)
    index : P.Index <- loop(U32.to_nat(n), 0, 0, 1, width, index)
    index : P.Index <- loop(U32.to_nat(n), 1, U32.sub(n, 1), 4294967295, width, index)
    index : P.Index <- loop(U32.to_nat(n), 0, 0, 1, width, index)
    # Reject both a new identity in an existing chain and a completely new hash.
    index : P.Index <- op(0, n, 11, 0, index)
    index : P.Index <- op(1, n, 11, 0, index)
    index : P.Index <- op(0, U32.inc(n), 11, U32.inc(U32.div(n, width)), index)
    index : P.Index <- op(1, U32.inc(n), 11, U32.inc(U32.div(n, width)), index)
    index : P.Index <- mismatches(11n, 0, 0, 0, index)
    index : P.Index <- mismatches(11n, 0, U32.sub(n, 1), U32.div(U32.sub(n, 1), width), index)
    index : P.Index <- permuted(U32.to_nat(n), 0, U32.sub(n, 1), width, index)
    finish(P.length(index))

def created(bits: U32, width: U32, n: U32, result: Maybe<&1, P.Index>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "position scale constructor failed")
    case Some{index}: run(bits, width, n, index)

def valid(+bits: U32, width: U32, okay: Bool) -> IO(Unit):
  match okay:
    case False{}: IO.die(Unit, 2, "invalid position scale configuration")
    case True{}: created(bits, width, U32.shln(1, U32.to_nat(U32.sub(bits, 1))), P.new(bits))

def checked(+bits: U32, +width: U32) -> IO(Unit):
  valid(bits, width, Bool.and(
    Bool.or(U32.is_eq(bits, 2), Bool.or(U32.is_eq(bits, 10), U32.is_eq(bits, 16))),
    Bool.or(U32.is_eq(width, 1), Bool.or(U32.is_eq(width, 8),
      Bool.and(U32.is_eq(bits, 10), U32.is_eq(width, 512))))))

def numbers(b: Maybe<&2, U32>, w: Maybe<&2, U32>) -> IO(Unit):
  match b w:
    case Some{bits} Some{width}: checked(bits, width)
    case _ _: IO.die(Unit, 2, "invalid position scale configuration")

def config(words: List<&2, String>) -> IO(Unit):
  match words:
    case Con{bits, Con{width, Nil{}}}: numbers(U32.read(bits), U32.read(width))
    case _: IO.die(Unit, 2, "invalid position scale configuration")

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_POSITION_SCALE")
    case Done{text}: config(String.split(text, ' '))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_POSITION_SCALE")
    environment(text)
