import Base
import ./Right.bend as A
import ./Rights.bend as R
import ./GeneralHome.bend as Home
import ./Rows.bend as Rows
import ./CastleModel.bend as OldCastle
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
def homes_valid(+right: A.Right,+b: Chess.Board,
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  (+rook,+king)=homes
  Equal.trans(Bool,Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)),Bool.or(U32.is_zero(U32.and(Chess.get_rights(b),A.bit(right))),True{}),True{},
    Equal.cong(Bool,Bool,x => Bool.or(U32.is_zero(U32.and(Chess.get_rights(b),A.bit(right))),x),Bool.and(Pos.occupied_by(b,A.rook(right),3,A.side(right)),Pos.occupied_by(b,A.king(right),5,A.side(right))),True{},
      Equal.trans(Bool,Bool.and(Pos.occupied_by(b,A.rook(right),3,A.side(right)),Pos.occupied_by(b,A.king(right),5,A.side(right))),Pos.occupied_by(b,A.king(right),5,A.side(right)),True{},
        Equal.cong(Bool,Bool,x => Bool.and(x,Pos.occupied_by(b,A.king(right),5,A.side(right))),Pos.occupied_by(b,A.rook(right),3,A.side(right)),True{},rook),king)),or_true(U32.is_zero(U32.and(Chess.get_rights(b),A.bit(right)))))
def own_empty(+right: A.Right,+b: Chess.Board,ks: Bool,+chosen: {U32.is_eq(Chess.get_turn(b),1) == A.white(right) : Bool},
  +home: {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,OldCastle.model(A.white(right),ks))),A.bit(right)) == 0 : U32}:
  match right ks:
    case A.Wk{} True{}: Home.clear_own(A.Wk{},b,4,6,0,2,chosen,R.kind(b,4,5,1,home))
    case A.Wk{} False{}: Home.clear_own(A.Wk{},b,4,2,0,2,chosen,R.kind(b,4,5,1,home))
    case A.Wq{} True{}: Home.clear_own(A.Wq{},b,4,6,0,2,chosen,R.kind(b,4,5,1,home))
    case A.Wq{} False{}: Home.clear_own(A.Wq{},b,4,2,0,2,chosen,R.kind(b,4,5,1,home))
    case A.Bk{} True{}: Home.clear_own(A.Bk{},b,60,62,0,2,chosen,R.kind(b,60,5,0,home))
    case A.Bk{} False{}: Home.clear_own(A.Bk{},b,60,58,0,2,chosen,R.kind(b,60,5,0,home))
    case A.Bq{} True{}: Home.clear_own(A.Bq{},b,60,62,0,2,chosen,R.kind(b,60,5,0,home))
    case A.Bq{} False{}: Home.clear_own(A.Bq{},b,60,58,0,2,chosen,R.kind(b,60,5,0,home))
def other_rows(+right: A.Right,+b: Chess.Board,ks: Bool) ->
  {Row.row(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.king(right)) == Row.row(b,A.king(right)) : Board.Row} &
  {Row.row(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.rook(right)) == Row.row(b,A.rook(right)) : Board.Row}:
  match right ks:
    case A.Wk{} True{}: (Rows.row(b,Chess.Ply{60,62,0,2},4,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{60,62,0,2},7,{==},{==},{==},{==},{==}))
    case A.Wk{} False{}: (Rows.row(b,Chess.Ply{60,58,0,2},4,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{60,58,0,2},7,{==},{==},{==},{==},{==}))
    case A.Wq{} True{}: (Rows.row(b,Chess.Ply{60,62,0,2},4,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{60,62,0,2},0,{==},{==},{==},{==},{==}))
    case A.Wq{} False{}: (Rows.row(b,Chess.Ply{60,58,0,2},4,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{60,58,0,2},0,{==},{==},{==},{==},{==}))
    case A.Bk{} True{}: (Rows.row(b,Chess.Ply{4,6,0,2},60,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{4,6,0,2},63,{==},{==},{==},{==},{==}))
    case A.Bk{} False{}: (Rows.row(b,Chess.Ply{4,2,0,2},60,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{4,2,0,2},63,{==},{==},{==},{==},{==}))
    case A.Bq{} True{}: (Rows.row(b,Chess.Ply{4,6,0,2},60,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{4,6,0,2},56,{==},{==},{==},{==},{==}))
    case A.Bq{} False{}: (Rows.row(b,Chess.Ply{4,2,0,2},60,{==},{==},{==},{==},{==}),Rows.row(b,Chess.Ply{4,2,0,2},56,{==},{==},{==},{==},{==}))
def other_rows_post(+right: A.Right,+b: Chess.Board,+ks: Bool,
  observed: {Row.row(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.king(right)) == Row.row(b,A.king(right)) : Board.Row} &
    {Row.row(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.rook(right)) == Row.row(b,A.rook(right)) : Board.Row},
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  (king_row,rook_row)=observed
  (rook,king)=homes
  homes_valid(right,Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),
    (Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.rook(right),3,A.side(right)),Pos.occupied_by(b,A.rook(right),3,A.side(right)),True{},Rows.occupant(b,OldCastle.model(Bool.not(A.white(right)),ks),A.rook(right),3,A.side(right),rook_row),rook),
     Equal.trans(Bool,Pos.occupied_by(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.king(right),5,A.side(right)),Pos.occupied_by(b,A.king(right),5,A.side(right)),True{},Rows.occupant(b,OldCastle.model(Bool.not(A.white(right)),ks),A.king(right),5,A.side(right),king_row),king)))
def other_post(+right: A.Right,+b: Chess.Board,+ks: Bool,
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,OldCastle.model(Bool.not(A.white(right)),ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  other_rows_post(right,b,ks,other_rows(right,b,ks),homes)
def own_active(+right: A.Right,+b: Chess.Board,+ks: Bool,+chosen: {U32.is_eq(Chess.get_turn(b),1) == A.white(right) : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right))) == False{} : Bool},
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  (rook,king)=homes
  Old.false_true({Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool},
    R.contradict(b,E.move(b,ks),A.bit(right),active,
      Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right)),U32.and(Chess.get_rights(Chess.make_move(b,OldCastle.model(A.white(right),ks))),A.bit(right)),0,
        Equal.cong(Chess.Ply,U32,m => U32.and(Chess.get_rights(Chess.make_move(b,m)),A.bit(right)),E.move(b,ks),OldCastle.model(A.white(right),ks),OldCastle.move_same(b,ks,A.white(right),chosen)),own_empty(right,b,ks,chosen,king))))
def other_active(+right: A.Right,+b: Chess.Board,+ks: Bool,chosen: {U32.is_eq(Chess.get_turn(b),1) == Bool.not(A.white(right)) : Bool},
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  %Equal.sym(Chess.Ply,E.move(b,ks),OldCastle.model(Bool.not(A.white(right)),ks),OldCastle.move_same(b,ks,Bool.not(A.white(right)),chosen)) :
    {Pos.valid_right(Chess.make_move(b,_),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}
  other_post(right,b,ks,homes)
def active(+right: A.Right,+b: Chess.Board,+ks: Bool,white: Bool,chosen: {U32.is_eq(Chess.get_turn(b),1) == white : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right))) == False{} : Bool},
  homes: {Pos.occupied_by(b,A.rook(right),3,A.side(right)) == True{} : Bool} & {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  match right white:
    case A.Wk{} True{}: own_active(A.Wk{},b,ks,chosen,active,homes)
    case A.Wk{} False{}: other_active(A.Wk{},b,ks,chosen,homes)
    case A.Wq{} True{}: own_active(A.Wq{},b,ks,chosen,active,homes)
    case A.Wq{} False{}: other_active(A.Wq{},b,ks,chosen,homes)
    case A.Bk{} False{}: own_active(A.Bk{},b,ks,chosen,active,homes)
    case A.Bk{} True{}: other_active(A.Bk{},b,ks,chosen,homes)
    case A.Bq{} False{}: own_active(A.Bq{},b,ks,chosen,active,homes)
    case A.Bq{} True{}: other_active(A.Bq{},b,ks,chosen,homes)
def when(+right: A.Right,+b: Chess.Board,+ks: Bool,empty: Bool,+same: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right))) == empty : Bool},
  parent: {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  match empty:
    case True{}: R.empty_valid(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right),Index.u32_eq_true_same(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right)),0,same))
    case False{}: active(right,b,ks,U32.is_eq(Chess.get_turn(b),1),{==},same,R.parents(b,E.move(b,ks),A.bit(right),A.rook(right),A.king(right),A.side(right),same,parent))
def post(+right: A.Right,+b: Chess.Board,+ks: Bool,parent: {Pos.valid_right(b,A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}) ->
  {Pos.valid_right(Chess.make_move(b,E.move(b,ks)),A.bit(right),A.rook(right),A.king(right),A.side(right)) == True{} : Bool}:
  when(right,b,ks,U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,E.move(b,ks))),A.bit(right))),{==},parent)
