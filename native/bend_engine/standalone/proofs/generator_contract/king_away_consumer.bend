# Independent importing caller for the complete actual-generator proposition.
import Base
import ../castle_sequence/PROOF.bend as InheritedProof
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ./CastleSpec.bend as P
import ./KingAway.bend as Away

def use_exact_king_away(table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {P.result(Chess.legal_moves(table,b)) == (table,Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  Away.exact(table,b,white,sq,turn,one,bound,away)
