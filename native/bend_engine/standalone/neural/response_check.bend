# Opt-in conversion checks, not executed by the application or ordinary pytest.
import Base
import ../../legal_probe/Chess.bend as Chess
import ../../session_probe/Search.bend as Search
import ../Policy.bend as Policy
import ./Response.bend as R

def board() -> Chess.Board:
  Chess.Board{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(),
    U64.zero(), U64.zero(), 1, 0, 64}

def move1() -> Chess.Ply: Chess.Ply{12, 28, 0, 0}
def move2() -> Chess.Ply: Chess.Ply{11, 27, 0, 0}

def basic(index: U32, special: U32) -> U32:
  match index:
    case 0: Bool.pick(U32, U32.is_eq(special, 2), F32.bits(10000.0), 0)
    case 1: Bool.pick(U32, U32.is_eq(special, 1), F32.bits(1000.0), 0)
    case 2: Bool.pick(U32, U32.is_eq(special, 1), 3296329728, 0)
    case 1858: Bool.pick(U32, U32.is_eq(special, 1), F32.bits(1000.0), 0)
    case 1859: 0
    case 1860: Bool.pick(U32, U32.is_eq(special, 1), 3296329728, 0)
    case _: 0

def word(code: U32, +i: U32) -> U32:
  match code:
    case 3: Bool.pick(U32, U32.is_eq(i, 0), 2143289344, 0)
    case 4: Bool.pick(U32, U32.is_eq(i, 1858), 2139095040, 0)
    case 5: Bool.pick(U32, U32.is_eq(i, 700), 4286578688, 0)
    case _: basic(i, code)

def words(n: Nat, +i: U32, +code: U32) -> List<&2, U32>:
  match n:
    case 0n: Nil{}
    case 1n+p: Con{word(code, i), words(p, U32.inc(i), code)}

def entries(code: U32) -> List<&2, Policy.Entry>:
  match code:
    case 8: [Policy.Entry{move2(), 11, 2}, Policy.Entry{move1(), 10, 1}]
    case 9: [Policy.Entry{Chess.Ply{12, 28, 0, 1}, 10, 1}, Policy.Entry{move2(), 11, 2}]
    case 10: [Policy.Entry{move1(), 10, 1858}, Policy.Entry{move2(), 11, 2}]
    case 11: [Policy.Entry{move1(), 4672, 1}, Policy.Entry{move2(), 11, 2}]
    case 12: [Policy.Entry{move1(), 10, 1}]
    case 14: Nil{}
    case _: [Policy.Entry{move1(), 10, 1}, Policy.Entry{move2(), 11, 2}]

def moves(code: U32) -> List<&2, Chess.Ply>:
  match code:
    case 14: Nil{}
    case _: [move1(), move2()]

def length(code: U32) -> Nat:
  match code:
    case 6: 1860n
    case 7: 1862n
    case _: 1861n

def second(code: U32, epoch: U32, req: U32, node: U32, count: U32,
  w: F32, d: F32, l: F32, first: F32, r: Array<F32> & F32) -> IO(Unit):
  (a, v) = r
  IO.print("accepted " ++ U32.show(code) ++ " " ++ U32.show(epoch) ++ " " ++ U32.show(req)
    ++ " " ++ U32.show(node) ++ " " ++ U32.show(count) ++ " " ++ U32.show(F32.bits(w))
    ++ " " ++ U32.show(F32.bits(d)) ++ " " ++ U32.show(F32.bits(l))
    ++ " " ++ U32.show(F32.bits(first)) ++ " " ++ U32.show(F32.bits(v)))

def first(code: U32, epoch: U32, req: U32, node: U32, count: U32,
  w: F32, d: F32, l: F32, r: Array<F32> & F32) -> IO(Unit):
  (a, v) = r
  second(code, epoch, req, node, count, w, d, l, v, Array.get(F32, a, 1))

def emit(code: U32, r: R.Conversion) -> IO(Unit):
  match r:
    case R.Rejected{}: IO.print("rejected " ++ U32.show(code))
    case R.Accepted{Search.Reply{epoch, req, node, status, w, d, l, count, p}}:
      first(code, epoch, req, node, count, w, d, l, Array.get(F32, p, 0))

def check(+code: U32) -> IO(Unit):
  emit(code, R.convert(Search.Ticket{7, 9, 3, board(), moves(code), Bool.not(U32.is_eq(code, 13))},
    entries(code), words(length(code), 0, code)))

def run(n: Nat, +i: U32) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        check(i)
        run(p, U32.inc(i))

def main() -> IO(Unit): run(15n, 0)
