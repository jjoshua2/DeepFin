import Base
import ./Source.bend as Source
import ../board/Spec.bend as Board
import ../backrank_pawns/Mask.bend as M
import ../color_counts/Actual.bend as A
import ../../../legal_probe/Chess.bend as Chess

def put(+b: Chess.Board,+m: Chess.Ply) -> Bool: U32.is_eq(M.put(b,m),5)
def same(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {put(b,m) == U64.test_bit(Chess.get_kings(b),U32.to_nat(A.source(m))) : Bool}
def ordinary(+b: Chess.Board,+s: U32,+d: U32,+f: U32,rows: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool}) -> same(b,Chess.Ply{s,d,0,f}):
  Source.decoded(b,s,rows,sb,owned)
def promoted(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,rows: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  pawn: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(s)) == True{} : Bool},
  nonzero: {U32.is_zero(p) == False{} : Bool},no: {U32.is_eq(p,5) == False{} : Bool}) -> same(b,Chess.Ply{s,d,p,f}):
  %Equal.sym(Bool,U32.is_zero(p),False{},nonzero) :
    {U32.is_eq(Bool.pick(U32,_,Chess.piece(b,s),p),5) == U64.test_bit(Chess.get_kings(b),U32.to_nat(s)) : Bool}
  %Equal.sym(Bool,U64.test_bit(Chess.get_kings(b),U32.to_nat(s)),False{},Source.pawn_clear(b,s,rows,sb,pawn)) :
    {U32.is_eq(p,5) == _ : Bool}
  no
