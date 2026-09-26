# Same numeric cases as the list oracle, plus wrong physical capacity/needs.
# Spare capacity is poisoned: it is not part of the 1861-logit output shape.
import Base
import ../legal_probe/Chess.bend as Chess
import ../session_probe/Search.bend as Search
import ./Position.bend as Pos
import ./Policy.bend as Policy
import ./LogitReply.bend as Reply
import ./logit_probe.bend as Reference

def fill(fuel: Nat, +mode: U32, +i: U32, a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: fill(p, mode, U32.inc(i), Array.set(F32, a, i, Reference.logit(mode, i)))

def result(mode: U32, c: Reply.Converted) -> IO(Unit):
  Reply.Converted{values, reply} = c
  Reference.result(mode, reply)

def run_array(+mode: U32, a: Array<F32>) -> IO(Unit):
  ticket : Search.Ticket = Search.Ticket{7, 8, 9, Pos.board(Pos.start()), Con{Chess.Ply{12, 28, 0, 0}, Con{Chess.Ply{11, 27, 0, 0}, Nil{}}}, Bool.not(U32.is_eq(mode, 14))}
  result(mode, Reply.convert_buffer(ticket, Reference.entries(mode), U32.from_nat(Reference.length(mode)),
    Array.size(F32, fill(1861n, mode, 0, a))))

def run(+mode: U32) -> IO(Unit):
  match mode:
    case 11: run_array(mode, Array.new(F32, 10n, 0.0))
    case 12: run_array(mode, Array.new(F32, 12n, 0.0))
    case 13: run_array(mode, Array.new(F32, 0n, 0.0))
    case _: run_array(mode, Array.new(F32, 11n, F32.div(0.0, 0.0)))

def cases(fuel: Nat, +mode: U32) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        run(mode)
        cases(p, U32.inc(mode))

def main() -> IO(Unit):
  cases(15n, 0)
