# List predicates preserve origin without assuming either attack correctness or valid metadata.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E

def all(-P: @-m: Chess.Ply -> Type,xs: List<&2,Chess.Ply>) -> Type:
  match xs:
    case Nil{}: Unit
    case Con{m,rest}: P(m) & all(P,rest)

def flag(m: Chess.Ply) -> U32:
  Chess.Ply{src,dst,promotion,f} = m
  f

def not_castle(-m: Chess.Ply) -> Type:
  {U32.is_eq(flag(m),2) == False{} : Bool}

def certified(-b: Chess.Board,-m: Chess.Ply) -> Type:
  ({E.guard(b,True{}) == True{} : Bool} & {m == E.move(b,True{}) : Chess.Ply}) |
  ({E.guard(b,False{}) == True{} : Bool} & {m == E.move(b,False{}) : Chess.Ply})

def origin(-b: Chess.Board,-tail: List<&2,Chess.Ply>,-m: Chess.Ply) -> Type:
  E.member(m,tail) | certified(b,m)

def classified(-b: Chess.Board,-m: Chess.Ply) -> Type:
  not_castle(m) | certified(b,m)

def both(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.castle_side(b,False{},Chess.castle_side(b,True{},r))

def filtered(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.filter_prepare(b,both(b,r))

# The exact pre-castling scan occurring inside actual legal_moves.
def scanned(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  +own = Chess.color(b,Chess.get_turn(b))
  Chess.scan(Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}))
