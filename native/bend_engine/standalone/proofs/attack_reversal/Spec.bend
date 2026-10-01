# Independent finite-coordinate relations over the accepted geometry.
import Base
import ../attack_geometry/Geometry.bend as G

def member(+x: Nat,xs: List<&2,Nat>) -> Type:
  match xs:
    case Nil{}: Empty
    case h <> t: {x == h : Nat} | member(x,t)

def every(-P: Nat -> Type,xs: List<&2,Nat>) -> Type:
  match xs:
    case Nil{}: Unit
    case h <> t: P(h) & every(P,t)

def neighbor(+kind: G.Leaper,+src: Nat,+dst: Nat) -> Type:
  {Nat.is_lt(dst,64n) == True{} : Bool} -> member(src,G.targets(G.reverse(kind),dst))

def certificate(+kind: G.Leaper,+src: Nat) -> Type:
  every(dst => neighbor(kind,src,dst),G.targets(kind,src))

def contains(+x: Nat,xs: List<&2,Nat>) -> Bool:
  match xs:
    case Nil{}: False{}
    case h <> t: Bool.or(Nat.is_eq(x,h),contains(x,t))
