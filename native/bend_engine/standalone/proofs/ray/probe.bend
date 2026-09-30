# Runtime probe imports production operations only, never proof/specification modules.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def calculate(kind: U32,+key: U32,+dir: U32,+occ: U64,acc: U64) -> U64:
  match kind:
    case 0: Tables.ray(7n,False{},key,dir,occ,False{},acc)
    case 1: Tables.slider(U32.mod(key,64),U32.is_ge(key,64),occ,False{})
    case _: Tables.slider(U32.mod(key,64),U32.is_ge(key,64),
      U64.and(occ,Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.zero(),True{})),False{})

def checked(ok: Bool,kind: U32,key: U32,dir: U32,hi: U32,lo: U32,ahi: U32,alo: U32) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"invalid ray domain")
    case True{}:
      +v = calculate(kind,key,dir,U64.from_parts(hi,lo),U64.from_parts(ahi,alo))
      IO.print(U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))

def valid(+kind: U32,+key: U32,+dir: U32) -> Bool:
  Bool.and(U32.is_lt(kind,3),Bool.pick(Bool,U32.is_eq(kind,0),
    Bool.and(U32.is_lt(key,64),U32.is_lt(dir,8)),Bool.and(U32.is_lt(key,128),U32.is_eq(dir,0))))

def request(a: Maybe<&2,U32>,b: Maybe<&2,U32>,c: Maybe<&2,U32>,d: Maybe<&2,U32>,e: Maybe<&2,U32>,f: Maybe<&2,U32>,g: Maybe<&2,U32>) -> IO(Unit):
  match a b c d e f g:
    case Some{+kind} Some{+key} Some{+dir} Some{hi} Some{lo} Some{ahi} Some{alo}:
      checked(valid(kind,key,dir),kind,key,dir,hi,lo,ahi,alo)
    case _ _ _ _ _ _ _: IO.die(Unit,2,"invalid ray numbers")

def go(fuel: Nat,xs: List<String>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p a <> b <> c <> d <> e <> f <> g <> tail:
      do IO<Unit>:
        request(Text.number(a),Text.number(b),Text.number(c),Text.number(d),Text.number(e),Text.number(f),Text.number(g))
        go(p,tail)
    case _ _: IO.die(Unit,2,"ray input budget or tuple error")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    go(1024n,xs)
