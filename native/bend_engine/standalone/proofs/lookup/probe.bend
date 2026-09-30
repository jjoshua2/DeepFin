# Supplementary actual lookup execution; inputs are occupancies, never oracle values.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def emit(+key: U32,r: Array<U64> & U64) -> IO(Array<U64>):
  (a,+v) = r
  do IO<Array<U64>>:
    IO.print(U32.show(key) ++ " " ++ U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))
    return a

def checked(ok: Bool,+key: U32,hi: U32,lo: U32,a: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: emit(key,Chess.slide(a,key,U64.from_parts(hi,lo)))
    case False{}: IO.die(Array<U64>,2,"invalid lookup key")

def request(k: Maybe<&2,U32>,h: Maybe<&2,U32>,l: Maybe<&2,U32>,a: Array<U64>) -> IO(Array<U64>):
  match k h l:
    case Some{+key} Some{hi} Some{lo}: checked(U32.is_lt(key,128),key,hi,lo,a)
    case _ _ _: IO.die(Array<U64>,2,"invalid lookup numbers")

def go(fuel: Nat,xs: List<String>,a: Array<U64>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p k <> h <> l <> tail:
      do IO<Unit>:
        a : Array<U64> <- request(Text.number(k),Text.number(h),Text.number(l),a)
        go(p,tail,a)
    case _ _: IO.die(Unit,2,"lookup input budget or tuple error")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    go(1024n,xs,Tables.build())
