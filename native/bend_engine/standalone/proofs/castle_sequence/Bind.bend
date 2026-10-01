# Connect the actual side producer to its checked call and valid-side coordinates.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_kings/Projection.bend as P
import ../castle_emission/Spec.bend as E

def side(table: Array<U64>,+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.castle_side(b,ks,(table,tail)) == Chess.castle_checked(True{},table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Bool,E.guard(b,ks),True{},guard) :
    {Chess.castle_checked(_,table,b,E.src(b),E.dst(b,ks),E.transit(b,ks),tail) == Chess.castle_checked(True{},table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) : Array<U64> & List<&2,Chess.Ply>}
  Equal.cong(Bool,Array<U64> & List<&2,Chess.Ply>,w => Chess.castle_checked(True{},table,b,K.src(w),K.dst(w,ks),K.transit(w,ks),tail),U32.is_eq(Chess.get_turn(b),1),white,P.white_turn(b,white,turn))
