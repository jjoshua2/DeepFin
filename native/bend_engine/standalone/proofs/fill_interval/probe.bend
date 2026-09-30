# Actual table headers and whole fill loops, with a poisoned query location.
# No proof predicates, expected tables, or external answer provider are imported.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def count(mode: U32,+size: U32) -> U32:
  match mode:
    case 0: size
    case 1: 0
    case 2: 1
    case 3: U32.sub(size,1)
    case _: U32.inc(size)

def query(code: U32,+at: U32,+size: U32) -> U32:
  match code:
    case 0: U32.sub(at,1)
    case 1: U32.add(at,size)
    case 2: U32.add(at,U32.shr(size))
    case 3: U32.sub(U32.add(at,size),1)
    case 4: 0
    case 5: 131071
    case _: U32.inc(U32.add(at,size))

def emit(id: U32,at: U32,size: U32,n: U32,q: U32,old: U64,after: U64,first: U64,last: U64,r: Array<U64> & U32) -> IO(Unit):
  (a,capacity) = r
  IO.print(U32.show(id) ++ " " ++ U32.show(at) ++ " " ++ U32.show(size) ++ " " ++ U32.show(n) ++ " " ++ U32.show(q)
    ++ " " ++ show(old) ++ " " ++ show(after) ++ " " ++ show(first) ++ " " ++ show(last) ++ " " ++ U32.show(capacity))

def last(id: U32,at: U32,size: U32,n: U32,q: U32,old: U64,after: U64,first: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  emit(id,at,size,n,q,old,after,first,v,Array.size(U64,a))

def first(id: U32,+at: U32,+size: U32,n: U32,q: U32,old: U64,after: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  last(id,at,size,n,q,old,after,v,Array.get(U64,a,U32.sub(U32.add(at,size),1)))

def after(id: U32,+at: U32,size: U32,n: U32,q: U32,old: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  first(id,at,size,n,q,old,v,Array.get(U64,a,at))

def old(+id: U32,+key: U32,+at: U32,+size: U32,+n: U32,+q: U32,mask: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  after(id,at,size,n,q,v,Array.get(U64,
    Tables.fill(U32.to_nat(n),at,U32.mod(key,64),U32.is_ge(key,64),mask,U64.zero(),a),q))

def mask(id: U32,key: U32,+at: U32,mode: U32,code: U32,seed: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,+m) = r
  +size = U32.shln(1,U32.to_nat(U64.popcount(m)))
  +n = count(mode,size)
  +q = query(code,at,size)
  old(id,key,at,size,n,q,m,Array.get(U64,Array.set(U64,a,q,seed),q))

def prefix(id: U32,+key: U32,mode: U32,code: U32,seed: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  mask(id,key,U64.low(v),mode,code,seed,Array.get(U64,a,key))

def checked(ok: Bool,id: U32,+key: U32,mode: U32,code: U32,seed: U64) -> IO(Unit):
  match ok:
    case True{}: prefix(id,key,mode,code,seed,Array.get(U64,Tables.build(),U32.add(128,key)))
    case False{}: IO.die(Unit,2,"invalid fill interval fixture or budget")

def parse(+id: U32,key: Maybe<&2,U32>,mode: Maybe<&2,U32>,code: Maybe<&2,U32>,hi: Maybe<&2,U32>,lo: Maybe<&2,U32>) -> IO(Unit):
  match key mode code hi lo:
    case Some{+k} Some{+m} Some{+c} Some{h} Some{l}:
      checked(Bool.and(U32.is_lt(id,1024),Bool.and(U32.is_lt(k,128),Bool.and(U32.is_lt(m,5),U32.is_lt(c,7)))),id,k,m,c,U64.from_parts(h,l))
    case _ _ _ _ _: IO.die(Unit,2,"invalid fill interval integer")

def cases(xs: List<String>,+id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> m <> c <> h <> l <> rest:
      do IO<Unit>:
        parse(id,Text.number(k),Text.number(m),Text.number(c),Text.number(h),Text.number(l))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected five fill interval operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
