# Bounded probe of the unmodified actual fill, using actual builder metadata.
# No proof model, expected table data or external address/attack oracle is imported.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def emit(id: U32,k: U32,at: U32,size: U32,count: U32,q: U32,mask: U64,old: U64,after: U64,first: U64,last: U64,r: Array<U64> & U32) -> IO(Unit):
  (a,cap)=r
  IO.print(U32.show(id) ++ " " ++ U32.show(k) ++ " " ++ U32.show(at) ++ " " ++ U32.show(size)
    ++ " " ++ U32.show(count) ++ " " ++ U32.show(q) ++ " " ++ show(mask)
    ++ " " ++ show(old) ++ " " ++ show(after) ++ " " ++ show(first) ++ " " ++ show(last) ++ " " ++ U32.show(cap))

def last(id: U32,k: U32,at: U32,size: U32,count: U32,q: U32,mask: U64,old: U64,after: U64,first: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v)=r
  emit(id,k,at,size,count,q,mask,old,after,first,v,Array.size(U64,a))

def first(id: U32,k: U32,+at: U32,+size: U32,count: U32,q: U32,mask: U64,old: U64,after: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v)=r
  last(id,k,at,size,count,q,mask,old,after,v,Array.get(U64,a,U32.sub(U32.add(at,size),1)))

def after(id: U32,k: U32,+at: U32,size: U32,count: U32,q: U32,mask: U64,old: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v)=r
  first(id,k,at,size,count,q,mask,old,v,Array.get(U64,a,at))

def before(id: U32,+k: U32,+at: U32,size: U32,+count: U32,+q: U32,+mask: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v)=r
  after(id,k,at,size,count,q,mask,v,
    Array.get(U64,Tables.fill(U32.to_nat(count),at,U32.mod(k,64),U32.is_ge(k,64),mask,U64.zero(),a),q))

def count(mode: U32,size: U32) -> U32:
  match mode:
    case 0: size
    case 1: 1
    case _: 0

def mask(id: U32,k: U32,mode: U32,at: U32,+q: U32,seed: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,+m)=r
  +size=U32.shln(1,U32.to_nat(U64.popcount(m)))
  before(id,k,at,size,count(mode,size),q,m,Array.get(U64,Array.new(U64,17n,seed),q))

def prefix(id: U32,+k: U32,mode: U32,q: U32,seed: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,p)=r
  mask(id,k,mode,U64.low(p),q,seed,Array.get(U64,a,k))

def run(id: U32,+k: U32,mode: U32,q: U32,seed: U64) -> IO(Unit):
  prefix(id,k,mode,q,seed,Array.get(U64,Tables.build(),U32.add(128,k)))

def checked(ok: Bool,id: U32,k: U32,mode: U32,q: U32,seed: U64) -> IO(Unit):
  match ok:
    case True{}: run(id,k,mode,q,seed)
    case False{}: IO.die(Unit,2,"invalid fill frame fixture or budget")

def parse(+id: U32,k: Maybe<&2,U32>,mode: Maybe<&2,U32>,q: Maybe<&2,U32>,sh: Maybe<&2,U32>,sl: Maybe<&2,U32>) -> IO(Unit):
  match k mode q sh sl:
    case Some{+k} Some{+mode} Some{+q} Some{sh} Some{sl}:
      checked(Bool.and(U32.is_lt(id,1024),Bool.and(U32.is_lt(k,128),Bool.and(U32.is_lt(mode,3),U32.is_lt(q,131072)))),id,k,mode,q,U64.from_parts(sh,sl))
    case _ _ _ _ _: IO.die(Unit,2,"invalid fill frame integer")

def cases(xs: List<String>,+id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> mode <> q <> sh <> sl <> rest:
      do IO<Unit>:
        parse(id,Text.number(k),Text.number(mode),Text.number(q),Text.number(sh),Text.number(sl))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected five fill frame operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
