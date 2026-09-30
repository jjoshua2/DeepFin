# Real allocation, initialization, and query only; no proof/model imports.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def initial(mode: U32) -> Array<U64>:
  match mode:
    case 0: Tables.build()
    case 1: Tables.extras(64n,0,Array.new(U64,17n,U64.from_parts(4294967295,4294967295)))
    case 2: Tables.extras(64n,0,Tables.tables(3n,10,512,Array.new(U64,17n,U64.from_parts(305419896,2596069104))))
    case _: Tables.extras(64n,0,Array.set(U64,Array.new(U64,17n,U64.zero()),511,U64.from_parts(314159265,271828182)))

def tag(k: U32) -> U32:
  match k:
    case 0: 1
    case 1: 5
    case _: 0

def query(+k: U32,t: Array<U64>,sq: U32,occ: U64) -> Array<U64> & U64:
  Chess.attack(tag(k),t,sq,Bool.to_u32(U32.is_eq(k,2)),occ)

def output(r: Array<U64> & U64) -> IO(Array<U64>):
  (t,+v) = r
  do IO<Array<U64>>:
    IO.print("mask " ++ U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))
    return t

def valid(ok: Bool,k: U32,sq: U32,occ: U64,t: Array<U64>) -> IO(Array<U64>):
  match ok:
    case False{}: IO.die(Array<U64>,2,"invalid leaper domain")
    case True{}: output(query(k,t,sq,occ))

def rows(n: Nat,xs: List<&2,U32>,t: Array<U64>) -> IO(Unit):
  match n xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p +k <> +sq <> hi <> lo <> rest:
      do IO<Unit>:
        t : Array<U64> <- valid(Bool.and(U32.is_lt(k,4),U32.is_lt(sq,64)),k,sq,U64.from_parts(hi,lo),t)
        rows(p,rest,t)
    case _ _: IO.die(Unit,2,"invalid leaper tuple or budget")

def start(ok: Bool,mode: U32,xs: List<&2,U32>) -> IO(Unit):
  match ok:
    case True{}: rows(1024n,xs,initial(mode))
    case False{}: IO.die(Unit,2,"invalid initialization mode")

def parse(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        parse(t,v <> acc)

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case Some{+mode <> rest}: start(U32.is_lt(mode,4),mode,rest)
    case _: IO.die(Unit,2,"invalid leaper number or header")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    parsed(parse(args,Nil{}))
