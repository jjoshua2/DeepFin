# Opt-in native candidate: only original production Tables and Base operations.
# It does not import the prefix specification, proof certificates or any oracle.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def emit(r: Array<U64> & U64) -> IO(Array<U64>):
  (a,+v) = r
  do IO<Array<U64>>:
    IO.print(U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))
    return a

def dump(n: Nat, +i: U32, a: Array<U64>) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Array<U64> <- emit(Array.get(U64,a,i))
        dump(p,U32.inc(i),a)

def run(op: U32, n: U32, seed: U64) -> IO(Unit):
  match op:
    case 0: dump(131072n,0,Tables.tables(U32.to_nat(n),0,512,Array.new(U64,17n,seed)))
    case _: dump(131072n,0,Tables.build())

def checked(ok: Bool, op: U32, n: U32, seed: U64) -> IO(Unit):
  match ok:
    case True{}: run(op,n,seed)
    case False{}: IO.die(Unit,2,"invalid prefix probe request")

def parse(op: Maybe<&2,U32>, n: Maybe<&2,U32>, hi: Maybe<&2,U32>, lo: Maybe<&2,U32>) -> IO(Unit):
  match op n hi lo:
    case Some{+o} Some{+count} Some{+h} Some{+l}:
      checked(Bool.or(Bool.and(U32.is_eq(o,0),U32.is_le(count,128)),
        Bool.and(U32.is_eq(o,1),Bool.and(U32.is_eq(count,128),Bool.and(U32.is_eq(h,0),U32.is_eq(l,0))))),
        o,count,U64.from_parts(h,l))
    case _ _ _ _: IO.die(Unit,2,"invalid prefix probe request")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case op <> n <> hi <> lo <> Nil{}: parse(Text.number(op),Text.number(n),Text.number(hi),Text.number(lo))
    case _: IO.die(Unit,2,"expected four prefix probe arguments")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
