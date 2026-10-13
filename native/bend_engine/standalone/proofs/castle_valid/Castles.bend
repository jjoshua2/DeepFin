import Base
import ./Rights.bend as R
import ./Rows.bend as Rows
import ../castle_rights/Rights.bend as Prior
import ../castle_emission/Spec.bend as E
import ../decoder/Spec.bend as Row
import ../board/Spec.bend as Board
import ../bitboard_membership/IndexEquality.bend as Index
import ../u64/Words.bend as Old
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def or_true(a: Bool) -> {Bool.or(a,True{}) == True{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}
def model(white: Bool,+ks: Bool) -> Chess.Ply:
  +home=Bool.pick(U32,white,0,56)
  Chess.Ply{U32.add(home,4),U32.add(home,Bool.pick(U32,ks,6,2)),0,2}
def move_same(+b: Chess.Board,+ks: Bool,+white: Bool,same: {U32.is_eq(Chess.get_turn(b),1) == white : Bool}) ->
  {E.move(b,ks) == model(white,ks) : Chess.Ply}:
  Equal.cong(Bool,Chess.Ply,x => model(x,ks),U32.is_eq(Chess.get_turn(b),1),white,same)
def white_clear(+b: Chess.Board,+d: U32,+white: {U32.is_eq(Chess.get_turn(b),1) == True{} : Bool},
  home: {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{4,d,0,2})),1) == 0 : U32}:
  +cleared=Prior.king(b,4,d,0,2,R.kind(b,4,5,1,home))
  R.smaller_zero(Chess.get_rights(Chess.make_move(b,Chess.Ply{4,d,0,2})),3,1,{==},
    Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{4,d,0,2})),3),
      U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{4,d,0,2})),Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12)),0,
      Equal.cong(Bool,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{4,d,0,2})),Bool.pick(U32,x,3,12)),True{},U32.is_eq(Chess.get_turn(b),1),Equal.sym(Bool,U32.is_eq(Chess.get_turn(b),1),True{},white)),cleared))
def white_empty(+b: Chess.Board,ks: Bool,+white: {U32.is_eq(Chess.get_turn(b),1) == True{} : Bool},
  home: {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,model(True{},ks))),1) == 0 : U32}:
  match ks:
    case True{}: white_clear(b,6,white,home)
    case False{}: white_clear(b,2,white,home)
def black_rows(+b: Chess.Board,ks: Bool) ->
  {Row.row(Chess.make_move(b,model(False{},ks)),4) == Row.row(b,4) : Board.Row} &
  {Row.row(Chess.make_move(b,model(False{},ks)),7) == Row.row(b,7) : Board.Row}:
  match ks:
    case True{}:
      (Rows.row(b,Chess.Ply{60,62,0,2},4,{==},{==},{==},{==},{==}),
       Rows.row(b,Chess.Ply{60,62,0,2},7,{==},{==},{==},{==},{==}))
    case False{}:
      (Rows.row(b,Chess.Ply{60,58,0,2},4,{==},{==},{==},{==},{==}),
       Rows.row(b,Chess.Ply{60,58,0,2},7,{==},{==},{==},{==},{==}))
def black_post(+b: Chess.Board,+ks: Bool,
  rows: {Row.row(Chess.make_move(b,model(False{},ks)),4) == Row.row(b,4) : Board.Row} &
    {Row.row(Chess.make_move(b,model(False{},ks)),7) == Row.row(b,7) : Board.Row},
  homes: {Pos.occupied_by(b,7,3,1) == True{} : Bool} & {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,model(False{},ks)),1,7,4,1) == True{} : Bool}:
  (r4,r7)=rows
  (rook,king)=homes
  +r=Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,model(False{},ks)),7,3,1),Pos.occupied_by(b,7,3,1),True{},Rows.occupant(b,model(False{},ks),7,3,1,r7),rook)
  +k=Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,model(False{},ks)),4,5,1),Pos.occupied_by(b,4,5,1),True{},Rows.occupant(b,model(False{},ks),4,5,1,r4),king)
  Equal.trans(Bool,Pos.valid_right(Chess.make_move(b,model(False{},ks)),1,7,4,1),
    Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,model(False{},ks))),1)),True{}),True{},
    Equal.cong(Bool,Bool,x => Bool.or(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,model(False{},ks))),1)),x),
      Bool.and(Pos.occupied_by(Chess.make_move(b,model(False{},ks)),7,3,1),Pos.occupied_by(Chess.make_move(b,model(False{},ks)),4,5,1)),True{},
      Equal.trans(Bool,Bool.and(Pos.occupied_by(Chess.make_move(b,model(False{},ks)),7,3,1),Pos.occupied_by(Chess.make_move(b,model(False{},ks)),4,5,1)),
        Pos.occupied_by(Chess.make_move(b,model(False{},ks)),4,5,1),True{},
        Equal.cong(Bool,Bool,x => Bool.and(x,Pos.occupied_by(Chess.make_move(b,model(False{},ks)),4,5,1)),Pos.occupied_by(Chess.make_move(b,model(False{},ks)),7,3,1),True{},r),k)),or_true(U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,model(False{},ks))),1))))
def active(+b: Chess.Board,+ks: Bool,white: Bool,+chosen: {U32.is_eq(Chess.get_turn(b),1) == white : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,model(white,ks))),1)) == False{} : Bool},
  homes: {Pos.occupied_by(b,7,3,1) == True{} : Bool} & {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,model(white,ks)),1,7,4,1) == True{} : Bool}:
  match white:
    case False{}: black_post(b,ks,black_rows(b,ks),homes)
    case True{}:
      (rook,king)=homes
      Old.false_true({Pos.valid_right(Chess.make_move(b,model(True{},ks)),1,7,4,1) == True{} : Bool},
        R.contradict(b,model(True{},ks),1,active,white_empty(b,ks,chosen,king)))
def when(+b: Chess.Board,+ks: Bool,empty: Bool,
  +same: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),1)) == empty : Bool},
  parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),1,7,4,1) == True{} : Bool}:
  match empty:
    case True{}: R.empty_valid(Chess.make_move(b,E.move(b,ks)),1,7,4,1,Index.u32_eq_true_same(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),1),0,same))
    case False{}: active(b,ks,U32.is_eq(Chess.get_turn(b),1),{==},same,R.parents(b,E.move(b,ks),1,7,4,1,same,parent))
def post(+b: Chess.Board,+ks: Bool,parent: {Pos.valid_right(b,1,7,4,1) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),1,7,4,1) == True{} : Bool}:
  when(b,ks,U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),1)),{==},parent)
