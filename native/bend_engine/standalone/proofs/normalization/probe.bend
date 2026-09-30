# Opt-in supported public Array APIs at the actual depth-17 capacity.
# No proof representation or oracle is linked into this executable.
import Base
import ../../Text.bend as Text

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def emit(id: U32,+i: U32,initial: U64,after: U64,r: Array<U64> & U32) -> IO(Unit):
  (a,n) = r
  IO.print(U32.show(id) ++ " " ++ U32.show(U32.and(i,U32.sub(131072,1))) ++ " " ++ U32.show(n)
    ++ " " ++ show(initial) ++ " " ++ show(after))

def after(id: U32,i: U32,initial: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  emit(id,i,initial,v,Array.size(U64,a))

def initial(id: U32,+i: U32,+j: U32,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,old) = r
  after(id,i,old,Array.get(U64,Array.set(U64,a,i,v),j))

def run(id: U32,+i: U32,+j: U32,v: U64) -> IO(Unit):
  initial(id,i,j,v,Array.get(U64,Array.new(U64,17n,U64.from_parts(3735928559,2309737967)),j))

def checked(ok: Bool,id: U32,i: U32,j: U32,v: U64) -> IO(Unit):
  match ok:
    case True{}: run(id,i,j,v)
    case False{}: IO.die(Unit,2,"invalid normalization fixture or budget")

def parse(+id: U32,i: Maybe<&2,U32>,j: Maybe<&2,U32>,hi: Maybe<&2,U32>,lo: Maybe<&2,U32>) -> IO(Unit):
  match i j hi lo:
    case Some{x} Some{y} Some{h} Some{l}:
      checked(U32.is_lt(id,256),id,x,y,U64.from_parts(h,l))
    case _ _ _ _: IO.die(Unit,2,"invalid normalization integer")

def cases(xs: List<String>,+id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case i <> j <> hi <> lo <> rest:
      do IO<Unit>:
        parse(id,Text.number(i),Text.number(j),Text.number(hi),Text.number(lo))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected four normalization operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
