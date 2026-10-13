# Actual full-Ply legal occurrence, exact input array, explicit parent premises.
import Base
import ./Rows.bend as Rows
import ./Parent.bend as Parent
import ./Frame.bend as F
import ../ep_occurrence_tag/consumer.bend as Tag
import ../ep_occurrence_tag/Spec.bend as S
import ../child_metadata/Range.bend as R
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def preservation(+b: C.Board,-m: C.Ply) -> Type:
  {C.get_kings(C.make_move(b,m)) == C.get_kings(b) : U64} &
  {F.plane(C.make_move(b,m),C.get_turn(b)) == F.plane(b,C.get_turn(b)) : U64}

def cleared(+b: C.Board,+s: U32,+d: U32,+sc: {U64.and(U64.bit(U32.to_nat(s)),C.get_kings(b)) == U64.zero() : U64},+kind: {C.piece(b,s) == 0 : U32},cert: {U64.and(U64.bit(U32.to_nat(d)),C.get_kings(b)) == U64.zero() : U64} & {U64.and(U64.bit(U32.to_nat(Parent.victim(d))),C.get_kings(b)) == U64.zero() : U64}) -> preservation(b,C.Ply{s,d,0,1}):
  (+dc,+vc)=cert
  (F.kings(b,s,d,sc,dc,vc,kind),F.selected_plane(b,s,d,sc,dc,vc,kind))

def known(+b: C.Board,+s: U32,+d: U32,+good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},cert: S.facts(b,C.Ply{s,d,0,1})) -> preservation(b,C.Ply{s,d,0,1}):
  (promo,rest)=cert
  (+pawn,tail)=rest
  (same,raw_bound)=tail
  +below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool}=raw_bound
  +source_bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool}=Rows.true_bound(C.get_pawns(b),U32.to_nat(s),pawn)
  +sc=Rows.pawn_square(b,s,good,source_bound,pawn)
  +valid=Equal.trans(Bool,Position.valid_ep(d,b),Position.valid_ep(C.get_ep(b),b),True{},Equal.cong(U32,Bool,x => Position.valid_ep(x,b),d,C.get_ep(b),same),coherent)
  cleared(b,s,d,sc,Tag.pawn_kind(b,s,pawn),Parent.clear(b,d,good,below,valid))

def shape(+b: C.Board,m: C.Ply,+good: {Board.valid(b) == True{} : Bool},+coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},tag: {S.flag(m) == 1 : U32},cert: S.facts(b,m)) -> preservation(b,m):
  match m:
    case C.Ply{+s,+d,+p,+f}:
      (+promo,rest)=cert
      %Equal.sym(U32,f,1,tag) : preservation(b,C.Ply{s,d,p,_})
      %Equal.sym(U32,p,0,promo) : preservation(b,C.Ply{s,d,_,1})
      known(b,s,d,good,coherent,({==},rest))

# One affine array and one full-Ply membership certificate flow through the
# actual producer consumer. Source range follows from its actual pawn bit.
def legal_member(table: Array<U64>,+b: C.Board,+m: C.Ply,+good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {S.flag(m) == 1 : U32}) -> preservation(b,m):
  shape(b,m,good,coherent,tag,Tag.legal_member(table,b,m,here,tag))

def selected_known(+b: C.Board,+m: C.Ply,cert: preservation(b,m)) -> {U64.ctz(F.plane(C.make_move(b,m),C.get_turn(b))) == U64.ctz(F.plane(b,C.get_turn(b))) : U32}:
  (kings,plane)=cert
  Equal.cong(U64,U32,U64.ctz,F.plane(C.make_move(b,m),C.get_turn(b)),F.plane(b,C.get_turn(b)),plane)

def selected(table: Array<U64>,+b: C.Board,+m: C.Ply,good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},here: E.member(m,E.moves(C.legal_moves(table,b))),tag: {S.flag(m) == 1 : U32}) -> {U64.ctz(F.plane(C.make_move(b,m),C.get_turn(b))) == U64.ctz(F.plane(b,C.get_turn(b))) : U32}:
  selected_known(b,m,legal_member(table,b,m,good,coherent,here,tag))

