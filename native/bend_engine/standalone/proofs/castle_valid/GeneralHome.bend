import Base
import ./Right.bend as A
import ./Rights.bend as R
import ../castle_rights/Rights.bend as Prior
import ../king_counts/Source.bend as Source
import ../king_counts/Count.bend as Count
import ../decoder/consumer.bend as Decoder
import ../decoder/Spec.bend as Row
import ../board/Spec.bend as Board
import ../occupancy_insertion/Lattice.bend as L
import ../bitboard_membership/IndexEquality.bend as Index
import ../bitboard_membership/AllEmittedSound.bend as All
import ../ep_king_frame/Parent.bend as EP
import ../u64/Words.bend as Old
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def source_corner(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(s,A.rook(right)) == yes : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right))) == False{} : Bool}) ->
  {U32.is_eq(s,A.rook(right)) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(s,A.rook(right)) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},A.bit(right),active,
        Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right)),
          U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{A.rook(right),d,p,f})),A.bit(right)),0,
          Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{x,d,p,f})),A.bit(right)),s,A.rook(right),Index.u32_eq_true_same(s,A.rook(right),same)),
          Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{A.rook(right),d,p,f})),A.bit(right)),
            U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{A.rook(right),d,p,f})),Chess.corner_right(A.rook(right))),0,
            Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{A.rook(right),d,p,f})),x),A.bit(right),Chess.corner_right(A.rook(right)),Equal.sym(U32,Chess.corner_right(A.rook(right)),A.bit(right),A.corner(right))),
            Prior.source(b,A.rook(right),d,p,f)))))
def destination_corner(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(d,A.rook(right)) == yes : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right))) == False{} : Bool}) ->
  {U32.is_eq(d,A.rook(right)) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(d,A.rook(right)) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},A.bit(right),active,
        Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right)),
          U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,A.rook(right),p,f})),A.bit(right)),0,
          Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,x,p,f})),A.bit(right)),d,A.rook(right),Index.u32_eq_true_same(d,A.rook(right),same)),
          Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,A.rook(right),p,f})),A.bit(right)),
            U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,A.rook(right),p,f})),Chess.corner_right(A.rook(right))),0,
            Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,A.rook(right),p,f})),x),A.bit(right),Chess.corner_right(A.rook(right)),Equal.sym(U32,Chess.corner_right(A.rook(right)),A.bit(right),A.corner(right))),
            Prior.destination(b,s,A.rook(right),p,f)))))

def color_match(+w: Bool,+bl: Bool,+turn: Bool,+which: Bool,
  colors: {w == turn : Bool} & {bl == Bool.not(turn) : Bool},home: {Bool.pick(Bool,which,w,bl) == True{} : Bool}) ->
  {turn == which : Bool}:
  match which:
    case True{}:
      (cw,cb)=colors
      Equal.trans(Bool,turn,w,True{},Equal.sym(Bool,w,turn,cw),home)
    case False{}:
      (cw,cb)=colors
      EP.not_true(turn,Equal.trans(Bool,Bool.not(turn),bl,True{},Equal.sym(Bool,bl,Bool.not(turn),cb),home))
def clear_own(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,
  white: {U32.is_eq(Chess.get_turn(b),1) == A.white(right) : Bool},decoded: {Chess.piece(b,s) == 5 : U32}) ->
  {U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right)) == 0 : U32}:
  R.smaller_zero(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.own(right),A.bit(right),A.own_bit(right),
    Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.own(right)),
      U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,A.white(right),3,12)),0,
      Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),x),A.own(right),Bool.pick(U32,A.white(right),3,12),A.own_pick(right)),
      Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,A.white(right),3,12)),
        U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12)),0,
        Equal.cong(Bool,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,x,3,12)),A.white(right),U32.is_eq(Chess.get_turn(b),1),Equal.sym(Bool,U32.is_eq(Chess.get_turn(b),1),A.white(right),white)),Prior.king(b,s,d,p,f,decoded))))
def source_king(+right: A.Right,+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(s,A.king(right)) == yes : Bool},+rows: {Board.valid(b) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +home: {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),A.bit(right))) == False{} : Bool}) ->
  {U32.is_eq(s,A.king(right)) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      +fixed=Index.u32_eq_true_same(s,A.king(right),same)
      +turn=Equal.trans(Bool,U32.is_eq(Chess.get_turn(b),1),U32.is_eq(A.side(right),1),A.white(right),
        color_match(U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right))),U32.is_eq(Chess.get_turn(b),1),U32.is_eq(A.side(right),1),
          Source.colors(b,A.king(right),rows,A.king_bound(right),Equal.trans(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(A.king(right))),U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),True{},
            Equal.sym(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(A.king(right))),Equal.cong(U32,Bool,x => U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(x)),s,A.king(right),fixed)),owned)),
          Equal.trans(Bool,Bool.pick(Bool,U32.is_eq(A.side(right),1),U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right)))),U64.test_bit(Chess.color(b,A.side(right)),U32.to_nat(A.king(right))),True{},
            Equal.sym(Bool,U64.test_bit(Chess.color(b,A.side(right)),U32.to_nat(A.king(right))),Bool.pick(Bool,U32.is_eq(A.side(right),1),U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right)))),Source.pick_test(Chess.get_white(b),Chess.get_black(b),U32.to_nat(A.king(right)),U32.is_eq(A.side(right),1))),R.color(b,A.king(right),5,A.side(right),home))),
        Equal.sym(Bool,A.white(right),U32.is_eq(A.side(right),1),A.white_side(right)))
      Old.false_true({U32.is_eq(s,A.king(right)) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},A.bit(right),active,
        clear_own(right,b,s,d,p,f,turn,Equal.trans(U32,Chess.piece(b,s),Chess.piece(b,A.king(right)),5,Equal.cong(U32,U32,x => Chess.piece(b,x),s,A.king(right),fixed),R.kind(b,A.king(right),5,A.side(right),home)))))

def king_test(+right: A.Right,+b: Chess.Board,+rows: {Board.valid(b) == True{} : Bool},+home: {Pos.occupied_by(b,A.king(right),5,A.side(right)) == True{} : Bool}) ->
  {U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))) == True{} : Bool}:
  +present=Equal.trans(Bool,U64.test_bit(Chess.occupied(b),U32.to_nat(A.king(right))),Bool.or(U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right)))),True{},L.test_or(Chess.get_white(b),Chess.get_black(b),U32.to_nat(A.king(right))),
    Source.picked_or(U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right))),U32.is_eq(A.side(right),1),
      Equal.trans(Bool,Bool.pick(Bool,U32.is_eq(A.side(right),1),U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right)))),U64.test_bit(Chess.color(b,A.side(right)),U32.to_nat(A.king(right))),True{},
        Equal.sym(Bool,U64.test_bit(Chess.color(b,A.side(right)),U32.to_nat(A.king(right))),Bool.pick(Bool,U32.is_eq(A.side(right),1),U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right))),U64.test_bit(Chess.get_black(b),U32.to_nat(A.king(right)))),Source.pick_test(Chess.get_white(b),Chess.get_black(b),U32.to_nat(A.king(right)),U32.is_eq(A.side(right),1))),R.color(b,A.king(right),5,A.side(right),home))))
  Equal.trans(Bool,U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))),U32.is_eq(Chess.piece(b,A.king(right)),5),True{},
    Equal.sym(Bool,U32.is_eq(Chess.piece(b,A.king(right)),5),U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))),
      Equal.cong(Board.Row,Bool,Source.king,Row.encode(Row.Occupant{Chess.piece(b,A.king(right)),U64.test_bit(Chess.get_white(b),U32.to_nat(A.king(right)))}),Row.row(b,A.king(right)),Decoder.reconstruct_occupied_row(b,A.king(right),rows,A.king_bound(right),present))),
    All.bool_and_true_right(U64.test_bit(Chess.color(b,A.side(right)),U32.to_nat(A.king(right))),U32.is_eq(Chess.piece(b,A.king(right)),5),home))
def destination_king(+right: A.Right,+b: Chess.Board,+d: U32,yes: Bool,same: {U32.is_eq(d,A.king(right)) == yes : Bool},
  clear: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64},
  home: {U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))) == True{} : Bool}) -> {U32.is_eq(d,A.king(right)) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(d,A.king(right)) == False{} : Bool},
        Equal.trans(Bool,False{},U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))),True{},
          Equal.sym(Bool,U64.test_bit(Chess.get_kings(b),U32.to_nat(A.king(right))),False{},Count.clear_test(Chess.get_kings(b),U32.to_nat(A.king(right)),
            Equal.trans(U64,U64.and(U64.bit(U32.to_nat(A.king(right))),Chess.get_kings(b)),U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)),U64.zero(),
              Equal.sym(U64,U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)),U64.and(U64.bit(U32.to_nat(A.king(right))),Chess.get_kings(b)),
                Equal.cong(U32,U64,x => U64.and(U64.bit(U32.to_nat(x)),Chess.get_kings(b)),d,A.king(right),Index.u32_eq_true_same(d,A.king(right),same))),clear))),home))
