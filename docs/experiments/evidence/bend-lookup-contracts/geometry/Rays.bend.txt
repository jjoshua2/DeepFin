# Independent finite coordinate paths, trimmed before bit-set conversion.
import Base
import ./Grid.bend as Grid

def path(fuel: Nat,+sq: Nat,+dir: Nat) -> List<&2,Nat>:
  match fuel:
    case 0n: Nil{}
    case 1n+p:
      +next = Grid.next(sq,dir)
      Bool.pick(List<&2,Nat>,Nat.is_lt(next,64n),next <> path(p,next,dir),Nil{})

def nonempty(xs: List<&2,Nat>) -> Bool:
  match xs:
    case Nil{}: False{}
    case _ <> _: True{}

def interior(xs: List<&2,Nat>) -> List<&2,Nat>:
  match xs:
    case Nil{}: Nil{}
    case x <> +tail: Bool.pick(List<&2,Nat>,nonempty(tail),x <> interior(tail),Nil{})

def bits(xs: List<&2,Nat>) -> U64:
  match xs:
    case Nil{}: U64.zero()
    case x <> tail: U64.or(U64.bit(x),bits(tail))

def mask(+key: Nat) -> U64:
  +sq = Nat.mod(key,64n)
  +d = Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n)
  U64.or(U64.or(bits(interior(path(7n,sq,d))),bits(interior(path(7n,sq,1n+d)))),
    U64.or(bits(interior(path(7n,sq,2n+d))),bits(interior(path(7n,sq,3n+d)))))
