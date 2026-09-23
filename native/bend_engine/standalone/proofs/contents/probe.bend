# Test only: real metadata and real fills on fresh poisoned complete storage.
# The persistent metadata cache is built by Tables.build, not a host oracle.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def count(mode: U32,+size: U32) -> U32:
  match mode:
    case 0: size
    case 1: U32.shr(size)
    case 2: 1
    case 3: 0
    case _: U32.inc(size)

def emit(cache: Array<U64>,id: U32,key: U32,prefix: U64,mask: U64,size: U32,n: U32,i: U32,value: U64,r: Array<U64> & U32) -> IO(Array<U64>):
  (a,capacity) = r
  do IO<Array<U64>>:
    IO.print(U32.show(id) ++ " " ++ U32.show(key) ++ " " ++ show(prefix) ++ " " ++ show(mask)
      ++ " " ++ U32.show(size) ++ " " ++ U32.show(n) ++ " " ++ U32.show(i) ++ " " ++ show(value) ++ " " ++ U32.show(capacity))
    IO.pure(Array<U64>,cache)

def result(cache: Array<U64>,id: U32,key: U32,prefix: U64,mask: U64,size: U32,n: U32,i: U32,r: Array<U64> & U64) -> IO(Array<U64>):
  (a,value) = r
  emit(cache,id,key,prefix,mask,size,n,i,value,Array.size(U64,a))

def mask(id: U32,+key: U32,+prefix: U64,mode: U32,+i: U32,seed: U64,r: Array<U64> & U64) -> IO(Array<U64>):
  (cache,+m) = r
  +size = U32.shln(1,U32.to_nat(U64.popcount(m)))
  +n = count(mode,size)
  result(cache,id,key,prefix,m,size,n,i,Array.get(U64,
    Tables.fill(U32.to_nat(n),U64.low(prefix),U32.mod(key,64),U32.is_ge(key,64),m,U64.zero(),Array.new(U64,17n,seed)),
    U32.add(U64.low(prefix),i)))

def prefix(id: U32,+key: U32,mode: U32,i: U32,seed: U64,r: Array<U64> & U64) -> IO(Array<U64>):
  (cache,p) = r
  mask(id,key,p,mode,i,seed,Array.get(U64,cache,key))

def checked(ok: Bool,cache: Array<U64>,id: U32,+key: U32,mode: U32,i: U32,seed: U64) -> IO(Array<U64>):
  match ok:
    case True{}: prefix(id,key,mode,i,seed,Array.get(U64,cache,U32.add(128,key)))
    case False{}: IO.die(Array<U64>,2,"invalid fill contents fixture or budget")

def parse(cache: Array<U64>,+id: U32,key: Maybe<&2,U32>,mode: Maybe<&2,U32>,index: Maybe<&2,U32>,hi: Maybe<&2,U32>,lo: Maybe<&2,U32>) -> IO(Array<U64>):
  match key mode index hi lo:
    case Some{+k} Some{+m} Some{+i} Some{h} Some{l}:
      checked(Bool.and(U32.is_lt(id,1024),Bool.and(U32.is_lt(k,128),Bool.and(U32.is_lt(m,5),U32.is_lt(i,8192)))),cache,id,k,m,i,U64.from_parts(h,l))
    case _ _ _ _ _: IO.die(Array<U64>,2,"invalid fill contents integer")

def cases(xs: List<String>,+id: U32,cache: Array<U64>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> m <> i <> h <> l <> rest:
      do IO<Unit>:
        next : Array<U64> <- parse(cache,id,Text.number(k),Text.number(m),Text.number(i),Text.number(h),Text.number(l))
        cases(rest,U32.inc(id),next)
    case _: IO.die(Unit,2,"expected five fill contents operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0,Tables.build())
