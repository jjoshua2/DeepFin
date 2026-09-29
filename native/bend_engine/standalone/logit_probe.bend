# Explicit opt-in native numerical boundary tests. Not a deployed evaluator.
import Base
import ../legal_probe/Chess.bend as Chess
import ../session_probe/Search.bend as Search
import ./Position.bend as Pos
import ./Policy.bend as Policy
import ./LogitReply.bend as Reply

def base(i: U32) -> F32:
  match i:
    case 0: F32.neg(2.0)
    case 1857: 2.0
    case 1858: F32.neg(2.0)
    case 1859: 0.0
    case 1860: 2.0
    case _: 100000.0

def exceptional(mode: U32, normal: F32) -> F32:
  match mode:
    case 3: F32.div(0.0, 0.0)
    case 4: F32.div(1.0, 0.0)
    case 5: F32.neg(F32.div(1.0, 0.0))
    case _: normal

def special(hit: Bool, mode: U32, value: F32) -> F32:
  match hit:
    case True{}: exceptional(mode, value)
    case False{}: value

def logit(mode: U32, +i: U32) -> F32:
  match mode:
    case 1: F32.add(base(i), 1000.0)
    case 2: Bool.pick(F32, U32.is_zero(i), F32.neg(3.0e38), 3.0e38)
    case _: special(U32.is_eq(i, 17), mode, base(i))

def raw(fuel: Nat, +mode: U32, +i: U32) -> List<F32>:
  match fuel:
    case 0n: Nil{}
    case 1n+p: Con{logit(mode, i), raw(p, mode, U32.inc(i))}

def entries(mode: U32) -> List<&2, Policy.Entry>:
  match mode:
    case 8: Con{Policy.Entry{Chess.Ply{12, 28, 0, 0}, 0, 1858}, Con{Policy.Entry{Chess.Ply{11, 27, 0, 0}, 1, 1857}, Nil{}}}
    case 9: Con{Policy.Entry{Chess.Ply{12, 28, 0, 1}, 0, 0}, Con{Policy.Entry{Chess.Ply{11, 27, 0, 0}, 1, 1857}, Nil{}}}
    case 10: Nil{}
    case _: Con{Policy.Entry{Chess.Ply{12, 28, 0, 0}, 0, 0}, Con{Policy.Entry{Chess.Ply{11, 27, 0, 0}, 1, 1857}, Nil{}}}

def length(mode: U32) -> Nat:
  match mode:
    case 6: 1860n
    case 7: 1862n
    case _: 1861n

def second(text: String, r: Array<F32> & F32) -> IO(Unit):
  (a, x) = r
  IO.print(text ++ " " ++ U32.show(F32.bits(x)))

def first(text: String, r: Array<F32> & F32) -> IO(Unit):
  (a, x) = r
  second(text ++ " " ++ U32.show(F32.bits(x)), Array.get(F32, a, 1))

def result(mode: U32, r: Search.Reply) -> IO(Unit):
  Search.Reply{epoch, request, node, status, win, draw, loss, count, policy} = r
  first(U32.show(mode) ++ " " ++ U32.show(status) ++ " " ++ U32.show(count)
    ++ " " ++ U32.show(F32.bits(win)) ++ " " ++ U32.show(F32.bits(draw)) ++ " " ++ U32.show(F32.bits(loss)), Array.get(F32, policy, 0))

def run(+mode: U32) -> IO(Unit):
  ticket : Search.Ticket = Search.Ticket{7, 8, 9, Pos.board(Pos.start()), Con{Chess.Ply{12, 28, 0, 0}, Con{Chess.Ply{11, 27, 0, 0}, Nil{}}}, True{}}
  result(mode, Reply.convert(ticket, entries(mode), raw(length(mode), mode, 0)))

def cases(fuel: Nat, +mode: U32) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        run(mode)
        cases(p, U32.inc(mode))

def main() -> IO(Unit):
  cases(11n, 0)
