# Candidate computes its own scalar start from production masks. No proof/oracle import.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def prefix(n: Nat,+key: U32,+at: U32) -> U32:
  match n:
    case 0n: at
    case 1n+p:
      +mask = Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.zero(),True{})
      prefix(p,U32.inc(key),U32.add(at,U32.shln(1,U32.to_nat(U64.popcount(mask)))))

def initial(n: Nat,+i: U32,+seed: U64,a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      initial(p,U32.inc(i),seed,Array.set(U64,a,i,U64.xor(seed,U64.from_u32(i))))

def emit(r: Array<U64> & U64) -> IO(Array<U64>):
  (a,+v) = r
  do IO<Array<U64>>:
    IO.print(U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))
    return a

def dump(n: Nat,+i: U32,a: Array<U64>) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Array<U64> <- emit(Array.get(U64,a,i))
        dump(p,U32.inc(i),a)

def run(+k: U32,n: U32,extra: U32,start: U32,seed: U64) -> IO(Unit):
  a = initial(131072n,0,seed,Array.new(U64,17n,U64.zero()))
  a = Tables.tables(U32.to_nat(n),k,prefix(U32.to_nat(k),0,512),a)
  dump(131072n,0,Tables.extras(U32.to_nat(extra),start,a))

def checked(ok: Bool,k: U32,n: U32,extra: U32,start: U32,seed: U64) -> IO(Unit):
  match ok:
    case True{}: run(k,n,extra,start,seed)
    case False{}: IO.die(Unit,2,"invalid header probe request")

def parse(k: Maybe<&2,U32>,n: Maybe<&2,U32>,extra: Maybe<&2,U32>,start: Maybe<&2,U32>,hi: Maybe<&2,U32>,lo: Maybe<&2,U32>) -> IO(Unit):
  match k n extra start hi lo:
    case Some{+key} Some{+count} Some{+en} Some{+ek} Some{h} Some{l}:
      checked(Bool.and(U32.is_le(key,128),Bool.and(U32.is_le(count,128),Bool.and(U32.is_le(U32.add(key,count),128),
        Bool.and(U32.is_le(en,64),Bool.and(U32.is_le(ek,64),U32.is_le(U32.add(en,ek),64)))))),key,count,en,ek,U64.from_parts(h,l))
    case _ _ _ _ _ _: IO.die(Unit,2,"invalid header probe request")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case k <> n <> en <> ek <> hi <> lo <> Nil{}: parse(Text.number(k),Text.number(n),Text.number(en),Text.number(ek),Text.number(hi),Text.number(lo))
    case _: IO.die(Unit,2,"expected six header probe arguments")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
