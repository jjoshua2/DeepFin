# Bounded opt-in probe of actual extras; full returned arrays, no proof imports.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def words(xs: List<U64>) -> String:
  match xs:
    case Nil{}: ""
    case h <> t: " " ++ show(h) ++ words(t)

def poison(+seed: U64,a: Array<U64>) -> Array<U64>:
  a = Array.set(U64,a,0,U64.xor(seed,U64.from_u32(1)))
  a = Array.set(U64,a,255,U64.xor(seed,U64.from_u32(2)))
  a = Array.set(U64,a,512,U64.xor(seed,U64.from_u32(3)))
  a = Array.set(U64,a,108159,U64.xor(seed,U64.from_u32(4)))
  Array.set(U64,a,131071,U64.xor(seed,U64.from_u32(5)))

def emit(id: U32,n: U32,k: U32,r: Array<U64> & U32) -> IO(Unit):
  (a,size) = r
  IO.print(U32.show(id) ++ " " ++ U32.show(n) ++ " " ++ U32.show(k) ++ " " ++ U32.show(size) ++ words(Array.to_list(U64,a)))

def checked(ok: Bool,id: U32,+n: U32,+k: U32,+seed: U64) -> IO(Unit):
  match ok:
    case True{}: emit(id,n,k,Array.size(U64,Tables.extras(U32.to_nat(n),k,poison(seed,Array.new(U64,17n,seed)))))
    case False{}: IO.die(Unit,2,"invalid extras fixture or budget")

def parse(+id: U32,n: Maybe<&2,U32>,k: Maybe<&2,U32>,hi: Maybe<&2,U32>,lo: Maybe<&2,U32>) -> IO(Unit):
  match n k hi lo:
    case Some{+n} Some{+k} Some{h} Some{l}:
      checked(Bool.and(U32.is_lt(id,8),Bool.and(U32.is_le(n,64),Bool.and(U32.is_le(k,64),U32.is_le(U32.add(n,k),64)))),id,n,k,U64.from_parts(h,l))
    case _ _ _ _: IO.die(Unit,2,"invalid extras integer")

def cases(xs: List<String>,+id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case n <> k <> h <> l <> rest:
      do IO<Unit>:
        parse(id,Text.number(n),Text.number(k),Text.number(h),Text.number(l))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected four extras operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
