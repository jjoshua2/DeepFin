# Proof predicates over complete actual Ply values, not a replacement generator.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E

def every(-P: Chess.Ply -> Type,xs: List<&2,Chess.Ply>) -> Type:
  match xs:
    case Nil{}: Unit
    case Con{m,ms}: P(m) & every(P,ms)

def flag(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f} = m
  f

def ordinary(m: Chess.Ply) -> Type:
  {U32.is_eq(flag(m),2) == False{} : Bool}

def side(-b: Chess.Board,-ks: Bool,-m: Chess.Ply) -> Type:
  {E.guard(b,ks) == True{} : Bool} & {m == E.move(b,ks) : Chess.Ply}

def castle(-b: Chess.Board,-m: Chess.Ply) -> Type:
  side(b,True{},m) | side(b,False{},m)

def classified(-b: Chess.Board,-m: Chess.Ply) -> Type:
  ordinary(m) | castle(b,m)

def candidates(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  +own = Chess.color(b,Chess.get_turn(b))
  Chess.scan(Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}))

def suffix(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.filter_prepare(b,Chess.castle_side(b,False{},Chess.castle_side(b,True{},r)))
