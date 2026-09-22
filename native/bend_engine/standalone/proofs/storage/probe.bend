# Bounded external fixture driver, never linked into the engine. No proof model.
import Base
import ../../Text.bend as Text
import ../../Tables.bend as Tables

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def words(xs: List<U64>) -> String:
  match xs:
    case Nil{}: ""
    case h <> t: " " ++ show(h) ++ words(t)

def seed() -> U64:
  U64.from_parts(3735928559,2309737967)

def applied(op: U32, +key: U32, at: U32, count: U32, +value: U64, a: Array<U64>) -> Array<U64>:
  match op:
    case 0: Array.set(U64,a,at,value)
    case 1:
      +sq = U32.mod(key,64)
      +bishop = U32.is_ge(key,64)
      +mask = Tables.slider(sq,bishop,U64.zero(),True{})
      Tables.fill(U32.to_nat(count),at,sq,bishop,mask,U64.and(value,mask),a)
    case 2: Tables.tables(U32.to_nat(count),key,at,a)
    case _: Tables.extras(U32.to_nat(count),key,a)

def finish(id: U32, initial: U64, wrote: U64, queried: U64, r: Array<U64> & U32) -> IO(Unit):
  (a,size) = r
  IO.print("v " ++ U32.show(id) ++ " " ++ U32.show(size) ++ " " ++ show(initial)
    ++ " " ++ show(wrote) ++ " " ++ show(queried) ++ words(Array.to_list(U64,a)))

def queried(id: U32, initial: U64, wrote: U64, r: Array<U64> & U64) -> IO(Unit):
  (a,value) = r
  finish(id,initial,wrote,value,Array.size(U64,a))

def wrote(id: U32, initial: U64, q: U32, r: Array<U64> & U64) -> IO(Unit):
  (a,value) = r
  queried(id,initial,value,Array.get(U64,a,q))

def started(id: U32, op: U32, key: U32, +at: U32, count: U32, q: U32, value: U64, r: Array<U64> & U64) -> IO(Unit):
  (a,initial) = r
  wrote(id,initial,q,Array.get(U64,applied(op,key,at,count,value,a),at))

def observe(id: U32, op: U32, d: U32, key: U32, at: U32, count: U32, +q: U32, value: U64) -> IO(Unit):
  started(id,op,key,at,count,q,value,Array.get(U64,Array.new(U64,U32.to_nat(d),seed()),q))

def allowed(op: U32, +d: U32, +key: U32, +at: U32, +count: U32) -> Bool:
  match op:
    case 0: Bool.and(U32.is_le(d,7),Bool.and(U32.is_eq(key,0),U32.is_eq(count,0)))
    case 1: Bool.and(U32.is_le(d,7),Bool.and(U32.is_lt(key,128),U32.is_le(count,8)))
    case 2: Bool.and(U32.is_eq(d,14),Bool.and(U32.is_le(key,126),Bool.and(U32.is_eq(at,512),U32.is_le(count,2))))
    case 3: Bool.and(U32.is_eq(d,9),Bool.and(U32.is_eq(key,0),U32.is_le(count,64)))
    case _: False{}

def checked(ok: Bool, id: U32, op: U32, d: U32, key: U32, at: U32, count: U32, q: U32, value: U64) -> IO(Unit):
  match ok:
    case True{}: observe(id,op,d,key,at,count,q,value)
    case False{}: IO.die(Unit,2,"invalid storage fixture or budget")

def parse(+id: U32, op: Maybe<&2,U32>, depth: Maybe<&2,U32>, key: Maybe<&2,U32>, at: Maybe<&2,U32>, count: Maybe<&2,U32>, q: Maybe<&2,U32>, hi: Maybe<&2,U32>, lo: Maybe<&2,U32>) -> IO(Unit):
  match op depth key at count q hi lo:
    case Some{+o} Some{+d} Some{+k} Some{+a} Some{+c} Some{query} Some{h} Some{l}:
      checked(Bool.and(U32.is_lt(id,4096),allowed(o,d,k,a,c)),id,o,d,k,a,c,query,U64.from_parts(h,l))
    case _ _ _ _ _ _ _ _: IO.die(Unit,2,"invalid storage integer")

def cases(xs: List<String>, +id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case op <> d <> k <> a <> c <> q <> h <> l <> rest:
      do IO<Unit>:
        parse(id,Text.number(op),Text.number(d),Text.number(k),Text.number(a),Text.number(c),Text.number(q),Text.number(h),Text.number(l))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected eight storage operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
