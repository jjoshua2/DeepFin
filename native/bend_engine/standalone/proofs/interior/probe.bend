# Opt-in native check of the public API, explicit masked index and masked aliases.
# Neither the interior specification nor any proof representation is executed.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def third(+i: U32, +v: U64, +w: U64, r: Array<U64> & U64) -> IO(Array<U64>):
  (a,z) = r
  do IO<Array<U64>>:
    IO.print(U32.show(i) ++ " " ++ show(v) ++ " " ++ show(w) ++ " " ++ show(z))
    return a

def second(+i: U32, +v: U64, r: Array<U64> & U64) -> IO(Array<U64>):
  (a,+w) = r
  third(i,v,w,Array.get(U64,a,U32.add(i,131072)))

def first(+i: U32, r: Array<U64> & U64) -> IO(Array<U64>):
  (a,+v) = r
  second(i,v,Array.get(U64,a,U32.and(i,131071)))

def dump(n: Nat, +i: U32, a: Array<U64>) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Array<U64> <- first(i,Array.get(U64,a,i))
        dump(p,U32.inc(i),a)

def checked(ok: Bool, count: U32) -> IO(Unit):
  match ok:
    case True{}: dump(U32.to_nat(count),0,Tables.build())
    case False{}: IO.die(Unit,2,"invalid interior probe count")

def parsed(n: Maybe<&2,U32>) -> IO(Unit):
  match n:
    case Some{+count}: checked(Bool.and(U32.is_gt(count,0),U32.is_le(count,131072)),count)
    case None{}: IO.die(Unit,2,"invalid interior probe count")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case n <> Nil{}: parsed(Text.number(n))
    case _: IO.die(Unit,2,"expected one interior probe argument")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
