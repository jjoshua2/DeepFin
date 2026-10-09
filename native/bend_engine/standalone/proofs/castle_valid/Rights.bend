import Base
import ../castle_rights/Words.bend as U
import ../castle_rights/Rights.bend as R
import ../bitboard_membership/IndexEquality.bend as Index
import ../bitboard_membership/AllEmittedSound.bend as All
import ../u64/Words.bend as Old
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def zero_mask(+a: U32,+b: U32,+bit: U32,sub: U.subset(a,b),zero: {U32.and(a,bit) == 0 : U32}) ->
  {U32.and(b,bit) == 0 : U32}:
  Equal.trans(U32,U32.and(b,bit),U32.and(U32.and(b,a),bit),0,
    Equal.cong(U32,U32,x => U32.and(x,bit),b,U32.and(b,a),Equal.sym(U32,U32.and(b,a),b,sub)),
    Equal.trans(U32,U32.and(U32.and(b,a),bit),U32.and(b,U32.and(a,bit)),0,U.assoc(b,a,bit),
      Equal.trans(U32,U32.and(b,U32.and(a,bit)),U32.and(b,0),0,
        Equal.cong(U32,U32,x => U32.and(b,x),U32.and(a,bit),0,zero),
        Equal.trans(U32,U32.and(b,0),U32.and(0,b),0,U.comm(b,0),U.zero(b)))))
def smaller_zero(+a: U32,+big: U32,+small: U32,sub: U.subset(big,small),zero: {U32.and(a,big) == 0 : U32}) ->
  {U32.and(a,small) == 0 : U32}:
  Equal.trans(U32,U32.and(a,small),U32.and(a,U32.and(small,big)),0,
    Equal.cong(U32,U32,x => U32.and(a,x),small,U32.and(small,big),Equal.sym(U32,U32.and(small,big),small,sub)),
    Equal.trans(U32,U32.and(a,U32.and(small,big)),U32.and(a,U32.and(big,small)),0,
      Equal.cong(U32,U32,x => U32.and(a,x),U32.and(small,big),U32.and(big,small),U.comm(small,big)),
      Equal.trans(U32,U32.and(a,U32.and(big,small)),U32.and(U32.and(a,big),small),0,
        Equal.sym(U32,U32.and(U32.and(a,big),small),U32.and(a,U32.and(big,small)),U.assoc(a,big,small)),
        Equal.trans(U32,U32.and(U32.and(a,big),small),U32.and(0,small),0,
          Equal.cong(U32,U32,x => U32.and(x,small),U32.and(a,big),0,zero),U.zero(small)))))
def empty_valid(+b: Chess.Board,+bit: U32,+rook: U32,+king: U32,+side: U32,
  empty: {U32.and(Chess.get_rights(b),bit) == 0 : U32}) -> {Pos.valid_right(b,bit,rook,king,side) == True{} : Bool}:
  %Equal.sym(U32,U32.and(Chess.get_rights(b),bit),0,empty) :
    {Bool.or(U32.is_zero(_),Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side))) == True{} : Bool}
  {==}
def contradict(+b: Chess.Board,+m: Chess.Ply,+bit: U32,
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),bit)) == False{} : Bool},
  empty: {U32.and(Chess.get_rights(Chess.make_move(b,m)),bit) == 0 : U32}) -> {False{} == True{} : Bool}:
  Equal.trans(Bool,False{},U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),bit)),True{},
    Equal.sym(Bool,U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),bit)),False{},active),
    Equal.cong(U32,Bool,U32.is_zero,U32.and(Chess.get_rights(Chess.make_move(b,m)),bit),0,empty))
def parent_when(+b: Chess.Board,+m: Chess.Ply,+bit: U32,+rook: U32,+king: U32,+side: U32,old_empty: Bool,
  same: {U32.is_zero(U32.and(Chess.get_rights(b),bit)) == old_empty : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),bit)) == False{} : Bool},
  parent: {Pos.valid_right(b,bit,rook,king,side) == True{} : Bool}) ->
  {Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side)) == True{} : Bool}:
  match old_empty:
    case False{}:
      Equal.trans(Bool,Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side)),Pos.valid_right(b,bit,rook,king,side),True{},
        Equal.cong(Bool,Bool,x => Bool.or(x,Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side))),
          False{},U32.is_zero(U32.and(Chess.get_rights(b),bit)),Equal.sym(Bool,U32.is_zero(U32.and(Chess.get_rights(b),bit)),False{},same)),parent)
    case True{}:
      Old.false_true({Bool.and(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side)) == True{} : Bool},
        contradict(b,m,bit,active,zero_mask(Chess.get_rights(b),Chess.get_rights(Chess.make_move(b,m)),bit,R.subset(b,m),
          Index.u32_eq_true_same(U32.and(Chess.get_rights(b),bit),0,same))))
def parents(+b: Chess.Board,+m: Chess.Ply,+bit: U32,+rook: U32,+king: U32,+side: U32,
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,m)),bit)) == False{} : Bool},
  parent: {Pos.valid_right(b,bit,rook,king,side) == True{} : Bool}) ->
  {Pos.occupied_by(b,rook,3,side) == True{} : Bool} & {Pos.occupied_by(b,king,5,side) == True{} : Bool}:
  +both=parent_when(b,m,bit,rook,king,side,U32.is_zero(U32.and(Chess.get_rights(b),bit)),{==},active,parent)
  (All.bool_and_true_left(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side),both),
   All.bool_and_true_right(Pos.occupied_by(b,rook,3,side),Pos.occupied_by(b,king,5,side),both))
def kind(+b: Chess.Board,+sq: U32,+p: U32,+side: U32,
  yes: {Pos.occupied_by(b,sq,p,side) == True{} : Bool}) -> {Chess.piece(b,sq) == p : U32}:
  Index.u32_eq_true_same(Chess.piece(b,sq),p,All.bool_and_true_right(U64.test_bit(Chess.color(b,side),U32.to_nat(sq)),U32.is_eq(Chess.piece(b,sq),p),yes))
def color(+b: Chess.Board,+sq: U32,+p: U32,+side: U32,
  yes: {Pos.occupied_by(b,sq,p,side) == True{} : Bool}) -> {U64.test_bit(Chess.color(b,side),U32.to_nat(sq)) == True{} : Bool}:
  All.bool_and_true_left(U64.test_bit(Chess.color(b,side),U32.to_nat(sq)),U32.is_eq(Chess.piece(b,sq),p),yes)
