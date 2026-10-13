import Base
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
import ../u64/Words.bend as Old
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def source_corner(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(s,7) == yes : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1)) == False{} : Bool}) ->
  {U32.is_eq(s,7) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(s,7) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},1,active,
        Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1),U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{7,d,p,f})),1),0,
          Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{x,d,p,f})),1),s,7,Index.u32_eq_true_same(s,7,same)),Prior.source(b,7,d,p,f))))
def destination_corner(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(d,7) == yes : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1)) == False{} : Bool}) ->
  {U32.is_eq(d,7) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(d,7) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},1,active,
        Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1),U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,7,p,f})),1),0,
          Equal.cong(U32,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,x,p,f})),1),d,7,Index.u32_eq_true_same(d,7,same)),Prior.destination(b,s,7,p,f))))
def source_king(+b: Chess.Board,+s: U32,+d: U32,+p: U32,+f: U32,yes: Bool,
  same: {U32.is_eq(s,4) == yes : Bool},+rows: {Board.valid(b) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +home: {Pos.occupied_by(b,4,5,1) == True{} : Bool},
  active: {U32.is_zero(U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),1)) == False{} : Bool}) ->
  {U32.is_eq(s,4) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      +fixed=Index.u32_eq_true_same(s,4,same)
      +cw=Pair.fst({U64.test_bit(Chess.get_white(b),4n) == U32.is_eq(Chess.get_turn(b),1) : Bool},
        {U64.test_bit(Chess.get_black(b),4n) == Bool.not(U32.is_eq(Chess.get_turn(b),1)) : Bool},
        Source.colors(b,4,rows,{==},Equal.trans(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),4n),U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),True{},
        Equal.sym(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)),U64.test_bit(Chess.color(b,Chess.get_turn(b)),4n),
          Equal.cong(U32,Bool,x => U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(x)),s,4,fixed)),owned)))
      +white=Equal.trans(Bool,U32.is_eq(Chess.get_turn(b),1),U64.test_bit(Chess.get_white(b),4n),True{},
        Equal.sym(Bool,U64.test_bit(Chess.get_white(b),4n),U32.is_eq(Chess.get_turn(b),1),cw),R.color(b,4,5,1,home))
      +decoded=Equal.trans(U32,Chess.piece(b,s),Chess.piece(b,4),5,Equal.cong(U32,U32,x => Chess.piece(b,x),s,4,fixed),R.kind(b,4,5,1,home))
      +cleared=Prior.king(b,s,d,p,f,decoded)
      +own_empty=Equal.trans(U32,U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),3),
        U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),3,12)),0,
        Equal.cong(Bool,U32,x => U32.and(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),Bool.pick(U32,x,3,12)),True{},U32.is_eq(Chess.get_turn(b),1),Equal.sym(Bool,U32.is_eq(Chess.get_turn(b),1),True{},white)),cleared)
      Old.false_true({U32.is_eq(s,4) == False{} : Bool},R.contradict(b,Chess.Ply{s,d,p,f},1,active,R.smaller_zero(Chess.get_rights(Chess.make_move(b,Chess.Ply{s,d,p,f})),3,1,{==},own_empty)))
def king_test(+b: Chess.Board,+rows: {Board.valid(b) == True{} : Bool},+home: {Pos.occupied_by(b,4,5,1) == True{} : Bool}) ->
  {U64.test_bit(Chess.get_kings(b),4n) == True{} : Bool}:
  +white=R.color(b,4,5,1,home)
  +present=Equal.trans(Bool,U64.test_bit(Chess.occupied(b),4n),Bool.or(U64.test_bit(Chess.get_white(b),4n),U64.test_bit(Chess.get_black(b),4n)),True{},L.test_or(Chess.get_white(b),Chess.get_black(b),4n),
    Equal.cong(Bool,Bool,x => Bool.or(x,U64.test_bit(Chess.get_black(b),4n)),U64.test_bit(Chess.get_white(b),4n),True{},white))
  +row=Decoder.reconstruct_occupied_row(b,4,rows,{==},present)
  Equal.trans(Bool,U64.test_bit(Chess.get_kings(b),4n),U32.is_eq(Chess.piece(b,4),5),True{},
    Equal.sym(Bool,U32.is_eq(Chess.piece(b,4),5),U64.test_bit(Chess.get_kings(b),4n),
      Equal.cong(Board.Row,Bool,Source.king,Row.encode(Row.Occupant{Chess.piece(b,4),U64.test_bit(Chess.get_white(b),4n)}),Row.row(b,4),row)),
    All.bool_and_true_right(U64.test_bit(Chess.get_white(b),4n),U32.is_eq(Chess.piece(b,4),5),home))
def destination_king(+b: Chess.Board,+d: U32,yes: Bool,same: {U32.is_eq(d,4) == yes : Bool},
  clear: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64},
  home: {U64.test_bit(Chess.get_kings(b),4n) == True{} : Bool}) -> {U32.is_eq(d,4) == False{} : Bool}:
  match yes:
    case False{}: same
    case True{}:
      Old.false_true({U32.is_eq(d,4) == False{} : Bool},
        Equal.trans(Bool,False{},U64.test_bit(Chess.get_kings(b),4n),True{},
          Equal.sym(Bool,U64.test_bit(Chess.get_kings(b),4n),False{},Count.clear_test(Chess.get_kings(b),4n,
            Equal.trans(U64,U64.and(U64.bit(4n),Chess.get_kings(b)),U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)),U64.zero(),
              Equal.sym(U64,U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)),U64.and(U64.bit(4n),Chess.get_kings(b)),
                Equal.cong(U32,U64,x => U64.and(U64.bit(U32.to_nat(x)),Chess.get_kings(b)),d,4,Index.u32_eq_true_same(d,4,same))),clear))),home))
