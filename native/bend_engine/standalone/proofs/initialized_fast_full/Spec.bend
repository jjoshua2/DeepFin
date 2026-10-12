# Exact production pairs and occurrence predicates over one input array.
import Base
import ../accepted_castle/Spec.bend as Prepared
import ../accepted_castle/Generated.bend as Actual
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def origins(-a: Array<U64>,-b: Chess.Board,+xs: List<&2,Chess.Ply>) -> Type:
  C.every(m => E.member(m,E.moves(Actual.candidates(a,b))),xs)
def initial(-a: Array<U64>,-b: Chess.Board) -> Type:
  {Chess.in_check(a,b,Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}
def list(-a: Array<U64>,-b: Chess.Board,-xs: List<&2,Chess.Ply>,-tail: List<&2,Chess.Ply>) -> Type:
  {Chess.filter_fast(xs,b,
    U64.and(Chess.color(b,Chess.get_turn(b)),
      U64.or(Prepared.mask(Chess.attack(4,a,
        U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)))),
        Chess.get_turn(b),Chess.occupied(b))),Chess.get_kings(b))),(a,tail)) ==
    Chess.filter_legal(xs,b,(a,tail)) : Array<U64> & List<&2,Chess.Ply>}
def legal(-a: Array<U64>,-b: Chess.Board) -> Type:
  {Chess.legal_moves(a,b) ==
    Chess.filter_legal(E.moves(Actual.candidates(a,b)),b,(a,Nil{})) :
    Array<U64> & List<&2,Chess.Ply>}
