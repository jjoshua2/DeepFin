# Opt-in pure output-conversion contracts; no model or host feature oracle.
import Base
import ./Reply.bend as R
import ../../session_probe/Search.bend as S
import ../../legal_probe/Chess.bend as C
import ../Policy.bend as P

def move(+i: U32) -> C.Ply:
  C.Ply{U32.mod(i, 64), U32.add(1, U32.div(i, 64)), 0, 0}

def size(mode: U32) -> U32:
  match mode:
    case 12: 0
    case 14: 1
    case 15: 256
    case 16: 257
    case _: 2

def moves(fuel: Nat, +i: U32) -> List<&2, C.Ply>:
  match fuel:
    case 0n: Nil{}
    case 1n+p: Con{move(i), moves(p, U32.inc(i))}

def slot(mode: U32, +i: U32) -> U32:
  match mode:
    case 9: 0
    case 10: 1858
    case 15: i
    case 16: i
    case _: Bool.pick(U32, U32.is_zero(i), 0, 1857)

def entries(fuel: Nat, +i: U32, +mode: U32) -> List<&2, P.Entry>:
  match fuel:
    case 0n: Nil{}
    case 1n+p:
      Con{P.Entry{move(Bool.pick(U32, U32.is_eq(mode, 11), U32.inc(i), i)),
        Bool.pick(U32, U32.is_eq(mode, 17), 4672, i), slot(mode, i)}, entries(p, U32.inc(i), mode)}

def normal(i: U32) -> F32:
  match i:
    case 0: F32.sub(0.0, 1.0)
    case 1857: 2.0
    case 1858: 1.0
    case 1859: 2.0
    case 1860: F32.sub(0.0, 3.0)
    case _: 0.0

def value(mode: U32, +i: U32) -> F32:
  match mode:
    case 1: Bool.pick(F32, U32.is_eq(i, 1), 1.0e20, normal(i))
    case 2: F32.sub(0.0, 3.0e38)
    case 3: Bool.pick(F32, U32.is_zero(i), F32.sub(0.0, 3.0e38), 3.0e38)
    case 4: 0.0
    case 5: Bool.pick(F32, U32.is_eq(i, 1), F32.div(0.0, 0.0), normal(i))
    case 6: Bool.pick(F32, U32.is_eq(i, 1859), F32.div(1.0, 0.0), normal(i))
    case _: normal(i)

def raw(fuel: Nat, +i: U32, +mode: U32) -> List<&2, F32>:
  match fuel:
    case 0n: Nil{}
    case 1n+p: Con{value(mode, i), raw(p, U32.inc(i), mode)}

def raw_size(mode: U32) -> Nat:
  match mode:
    case 7: 1860n
    case 8: 1862n
    case _: 1861n

def board() -> C.Board:
  C.Board{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), 1, 0, 64}

def converted(+mode: U32) -> S.Reply:
  +n = size(mode)
  R.convert(S.Ticket{9, 13, 31, board(), moves(U32.to_nat(n), 0), Bool.not(U32.is_eq(mode, 13))},
    entries(U32.to_nat(n), 0, mode), raw(raw_size(mode), 0, mode))

def append(text: String, r: Array<F32> & F32) -> Array<F32> & String:
  (a, x) = r
  (a, text ++ " " ++ U32.show(F32.bits(x)))

def prior_text(fuel: Nat, +i: U32, r: Array<F32> & String) -> String:
  match fuel:
    case 0n:
      (a, text) = r
      text
    case 1n+p:
      (a, text) = r
      prior_text(p, U32.inc(i), append(text, Array.get(F32, a, i)))

def output(mode: U32, r: S.Reply) -> IO(Unit):
  S.Reply{e, q, n, status, w, d, l, +count, policy} = r
  IO.print("reply_case " ++ U32.show(mode) ++ " " ++ U32.show(e) ++ " " ++ U32.show(q)
    ++ " " ++ U32.show(n) ++ " " ++ U32.show(status) ++ " " ++ U32.show(F32.bits(w))
    ++ " " ++ U32.show(F32.bits(d)) ++ " " ++ U32.show(F32.bits(l)) ++ " " ++ U32.show(count)
    ++ prior_text(U32.to_nat(count), 0, (policy, "")))

def cases(fuel: Nat, +i: U32) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        output(i, converted(i))
        cases(p, U32.inc(i))

def main() -> IO(Unit):
  cases(18n, 0)
