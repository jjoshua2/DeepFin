# Fresh distinct-cell native fixtures; no expected attack values enter the candidate.
import Base
import ../../Tables.bend as Tables

def initial(n: Nat,+i: U32,a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      a = Array.set(U64,a,i,U64.from_parts(U32.xor(3735928559,i),U32.add(2309737967,i)))
      initial(p,U32.inc(i),a)

def word(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def words(xs: List<U64>) -> String:
  match xs:
    case Nil{}: ""
    case v <> t: " " ++ word(v) ++ words(t)

def show(+id: U32,r: Array<U64> & U32) -> IO(Unit):
  (a,n) = r
  IO.print(U32.show(id) ++ " " ++ U32.show(n) ++ words(Array.to_list(U64,a)))

def one(id: U32,+n: Nat,k: U32) -> IO(Unit):
  show(id,Array.size(U64,Tables.extras(n,k,initial(131072n,0,Array.new(U64,17n,U64.zero())))))

def main() -> IO(Unit):
  do IO<Unit>:
    one(0,0n,64)
    one(1,2n,31)
    one(2,64n,0)
