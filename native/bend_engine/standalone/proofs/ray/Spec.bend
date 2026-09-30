# Independent list traversal, with an internal accumulator observation.
import Base
import ./Path.bend as Path
import ./Rays.bend as Rays
import ../../Tables.bend as Tables

def scan(xs: List<&2,Nat>,+occ: U64,+acc: U64) -> U64:
  match xs:
    case Nil{}: acc
    case +x <> tail:
      +updated = U64.or(acc,U64.bit(x))
      Bool.pick(U64,U64.test_bit(occ,x),updated,scan(tail,occ,updated))

# Input-only route certificate: checks coordinate steps, never occupancy or attacks.
def trace(n: Nat,+sq: U32,+dir: U32,xs: List<&2,Nat>) -> Bool:
  match n:
    case 0n:
      match xs:
        case Nil{}: True{}
        case _ <> _: False{}
    case 1n+p:
      match xs:
        case Nil{}: Nat.is_eq(U32.to_nat(Tables.jump(sq,Tables.delta(dir))),64n)
        case +x <> tail:
          +q = Tables.jump(sq,Tables.delta(dir))
          Bool.and(U32.is_lt(q,64),Bool.and(Bool.not(U32.is_eq(q,64)),
            Bool.and(Nat.is_eq(U32.to_nat(q),x),trace(p,q,dir,tail))))

def rays(+key: Nat,+occ: U64) -> U64:
  +sq = Nat.mod(key,64n)
  +d = Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n)
  U64.or(U64.or(Path.attack(Rays.path(7n,sq,d),occ),Path.attack(Rays.path(7n,sq,1n+d),occ)),
    U64.or(Path.attack(Rays.path(7n,sq,2n+d),occ),Path.attack(Rays.path(7n,sq,3n+d),occ)))

def directional(+sq: Nat,+d: Nat,+occ: U64) -> U64:
  U64.or(U64.or(Path.attack(Rays.path(7n,sq,d),occ),Path.attack(Rays.path(7n,sq,1n+d),occ)),
    U64.or(Path.attack(Rays.path(7n,sq,2n+d),occ),Path.attack(Rays.path(7n,sq,3n+d),occ)))

def actual_slider(+k: Nat,+occ: U64) -> U64:
  Tables.slider(U32.mod(U32.from_nat(k),64),U32.is_ge(U32.from_nat(k),64),occ,False{})
