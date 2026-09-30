# Independent unbounded file/rank geometry. No production/table imports.
import Base
import ../ray/Grid.bend as Grid

type Leaper is Data:
  Knight{}
  King{}
  WhitePawn{}
  BlackPawn{}

def advance(steps: Nat,+up: Bool,at: Maybe<&2,Nat>) -> Maybe<&2,Nat>:
  match steps:
    case 0n: at
    case 1n+p:
      match at:
        case None{}: None{}
        case Some{+x}:
          advance(p,up,Bool.pick(Maybe<&2,Nat>,up,Grid.positive(x),Grid.negative(x)))

def target(+sq: Nat,dx: Nat,px: Bool,dy: Nat,py: Bool) -> Nat:
  Grid.flatten(advance(dx,px,Some{Nat.mod(sq,8n)}),advance(dy,py,Some{Nat.div(sq,8n)}))

def targets(kind: Leaper,+sq: Nat) -> List<&2,Nat>:
  match kind:
    case Knight{}:
      target(sq,1n,True{},2n,True{}) <> target(sq,2n,True{},1n,True{}) <>
      target(sq,1n,False{},2n,True{}) <> target(sq,2n,False{},1n,True{}) <>
      target(sq,1n,True{},2n,False{}) <> target(sq,2n,True{},1n,False{}) <>
      target(sq,1n,False{},2n,False{}) <> target(sq,2n,False{},1n,False{}) <> Nil{}
    case King{}:
      Grid.next(sq,0n) <> Grid.next(sq,1n) <> Grid.next(sq,2n) <> Grid.next(sq,3n) <>
      Grid.next(sq,4n) <> Grid.next(sq,5n) <> Grid.next(sq,6n) <> Grid.next(sq,7n) <> Nil{}
    case WhitePawn{}:
      target(sq,1n,True{},1n,True{}) <> target(sq,1n,False{},1n,True{}) <> Nil{}
    case BlackPawn{}:
      target(sq,1n,True{},1n,False{}) <> target(sq,1n,False{},1n,False{}) <> Nil{}

def fold(xs: List<&2,Nat>,acc: U64) -> U64:
  match xs:
    case Nil{}: acc
    case h <> t: fold(t,U64.or(acc,U64.bit(h)))

def two(xs: List<&2,Nat>) -> U64:
  match xs:
    case a <> b <> _: U64.or(U64.bit(a),U64.bit(b))
    case _: U64.zero()

def combine(kind: Leaper,xs: List<&2,Nat>) -> U64:
  match kind:
    case Knight{}: fold(xs,U64.zero())
    case King{}: fold(xs,U64.zero())
    case _: two(xs)

def mask(+kind: Leaper,sq: Nat) -> U64:
  combine(kind,targets(kind,sq))

def reverse(kind: Leaper) -> Leaper:
  match kind:
    case Knight{}: Knight{}
    case King{}: King{}
    case WhitePawn{}: BlackPawn{}
    case BlackPawn{}: WhitePawn{}
