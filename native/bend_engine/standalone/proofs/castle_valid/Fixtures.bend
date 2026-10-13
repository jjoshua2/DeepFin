# Concrete actual-update witnesses; producer fixtures retain complete Ply fields.
import Base
import ./Spec.bend as S
import ./consumer.bend as Consumer
import ../board/Spec.bend as Board
import ../decoder/Spec.bend as Row
import ../castle_emission/Spec.bend as E
import ../king_counts/Actual.bend as Kings
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def white_homes() -> U64: U64.or(U64.bit(0n),U64.or(U64.bit(4n),U64.bit(7n)))
def black_homes() -> U64: U64.or(U64.bit(56n),U64.or(U64.bit(60n),U64.bit(63n)))
def homes(turn: U32,+pawns: U64,+knights: U64,own: U64,enemy: U64) -> Chess.Board:
  Chess.Board{pawns,knights,U64.zero(),U64.or(U64.bit(0n),U64.or(U64.bit(7n),U64.or(U64.bit(56n),U64.bit(63n)))),
    U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),U64.or(white_homes(),own),U64.or(black_homes(),enemy),turn,15,64}
def quiet() -> Chess.Board: homes(1,U64.bit(8n),U64.zero(),U64.bit(8n),U64.zero())
def promotion() -> Chess.Board: homes(1,U64.bit(48n),U64.bit(57n),U64.bit(48n),U64.bit(57n))
def white_castle() -> Chess.Board: homes(1,U64.zero(),U64.zero(),U64.zero(),U64.zero())
def black_castle() -> Chess.Board: homes(0,U64.zero(),U64.zero(),U64.zero(),U64.zero())

def quiet_rows() -> {Board.valid(quiet()) == True{} : Bool}: {==}
def quiet_turn() -> S.canonical(quiet()): {==}
def quiet_coherent() -> {Pos.valid_ep(Chess.get_ep(quiet()),quiet()) == True{} : Bool}: {==}
def quiet_parent() -> S.parent(quiet()): ({==},({==},({==},{==})))
def quiet_nonzero() -> {Chess.get_rights(quiet()) == 15 : U32}: {==}
def quiet_rights() -> {Chess.get_rights(Chess.make_move(quiet(),Chess.Ply{8,16,0,0})) == 15 : U32}: {==}
def quiet_child() -> {Pos.valid_right(Chess.make_move(quiet(),Chess.Ply{8,16,0,0}),1,7,4,1) == True{} : Bool}: {==}
# Importing caller uses actual full-Ply legal membership; the concrete endpoint
# above is independently reduced by the checker.
def quiet_consumer(table: Array<U64>,here: E.member(Chess.Ply{8,16,0,0},E.moves(Chess.legal_moves(table,quiet())))) ->
  {Pos.valid_right(Chess.make_move(quiet(),Chess.Ply{8,16,0,0}),1,7,4,1) == True{} : Bool}:
  Consumer.preserves_white_kingside(table,quiet(),Chess.Ply{8,16,0,0},quiet_rows(),quiet_turn(),quiet_coherent(),quiet_parent(),here)

def promotion_rows() -> {Board.valid(promotion()) == True{} : Bool}: {==}
def promotion_parent() -> S.parent(promotion()): ({==},({==},({==},{==})))
def promotion_rights() -> {Chess.get_rights(Chess.make_move(promotion(),Chess.Ply{48,57,4,0})) == 15 : U32}: {==}
def promotion_child() -> {Pos.valid_right(Chess.make_move(promotion(),Chess.Ply{48,57,4,0}),1,7,4,1) == True{} : Bool}: {==}
def promotion_queen() -> {Chess.piece(Chess.make_move(promotion(),Chess.Ply{48,57,4,0}),57) == 4 : U32}: {==}
def promotion_knight_child() -> {Pos.valid_right(Chess.make_move(promotion(),Chess.Ply{48,57,1,0}),1,7,4,1) == True{} : Bool}: {==}
def promotion_bishop_child() -> {Pos.valid_right(Chess.make_move(promotion(),Chess.Ply{48,57,2,0}),1,7,4,1) == True{} : Bool}: {==}
def promotion_rook_child() -> {Pos.valid_right(Chess.make_move(promotion(),Chess.Ply{48,57,3,0}),1,7,4,1) == True{} : Bool}: {==}
def promotion_fields() -> {Chess.destinations(64n,False{},U64.bit(57n),48,True{},57,Nil{}) ==
  Con{Chess.Ply{48,57,1,0},Con{Chess.Ply{48,57,2,0},Con{Chess.Ply{48,57,3,0},Con{Chess.Ply{48,57,4,0},Nil{}}}}} : List<&2,Chess.Ply>}: {==}
def quiet_fields() -> {Chess.destinations(64n,False{},U64.bit(16n),8,True{},64,Nil{}) ==
  Con{Chess.Ply{8,16,0,0},Nil{}} : List<&2,Chess.Ply>}: {==}
def ep_fields() -> {Chess.destinations(64n,False{},U64.bit(43n),36,True{},43,Nil{}) ==
  Con{Chess.Ply{36,43,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}

def white_castle_rows() -> {Board.valid(white_castle()) == True{} : Bool}: {==}
def white_castle_parent() -> S.parent(white_castle()): ({==},({==},({==},{==})))
def white_castle_rights() -> {Chess.get_rights(Chess.make_move(white_castle(),Chess.Ply{4,6,0,2})) == 12 : U32}: {==}
def white_castle_clears_both() -> {U32.and(Chess.get_rights(Chess.make_move(white_castle(),Chess.Ply{4,6,0,2})),3) == 0 : U32}: {==}
def white_queenside_clears_both() -> {U32.and(Chess.get_rights(Chess.make_move(white_castle(),Chess.Ply{4,2,0,2})),3) == 0 : U32}: {==}
def white_castle_child() -> {Pos.valid_right(Chess.make_move(white_castle(),Chess.Ply{4,6,0,2}),1,7,4,1) == True{} : Bool}: {==}
def white_castle_rook() -> {Pos.occupied_by(Chess.make_move(white_castle(),Chess.Ply{4,6,0,2}),5,3,1) == True{} : Bool}: {==}
def white_castle_king() -> {Pos.occupied_by(Chess.make_move(white_castle(),Chess.Ply{4,6,0,2}),6,5,1) == True{} : Bool}: {==}

def black_castle_rows() -> {Board.valid(black_castle()) == True{} : Bool}: {==}
def black_castle_parent() -> S.parent(black_castle()): ({==},({==},({==},{==})))
def black_castle_rights() -> {Chess.get_rights(Chess.make_move(black_castle(),Chess.Ply{60,62,0,2})) == 3 : U32}: {==}
def black_castle_child() -> {Pos.valid_right(Chess.make_move(black_castle(),Chess.Ply{60,62,0,2}),1,7,4,1) == True{} : Bool}: {==}
def black_queenside_child() -> {Pos.valid_right(Chess.make_move(black_castle(),Chess.Ply{60,58,0,2}),1,7,4,1) == True{} : Bool}: {==}
def black_castle_rook_home_row() -> {Row.row(Chess.make_move(black_castle(),Chess.Ply{60,62,0,2}),7) == Row.row(black_castle(),7) : Board.Row}: {==}
def black_castle_king_home_row() -> {Row.row(Chess.make_move(black_castle(),Chess.Ply{60,62,0,2}),4) == Row.row(black_castle(),4) : Board.Row}: {==}

# Without coherent EP, an ordinary black capture22->15 is tagged flag1 by
# matching the metadata target, and also erases the white home rook7.
def malformed_ep() -> Chess.Board:
  Chess.Board{U64.bit(22n),U64.bit(15n),U64.zero(),U64.bit(7n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(15n),U64.or(U64.bit(4n),U64.bit(7n))),U64.or(U64.bit(22n),U64.bit(60n)),0,1,15}
def malformed_ep_rows() -> {Board.valid(malformed_ep()) == True{} : Bool}: {==}
def malformed_ep_turn() -> S.canonical(malformed_ep()): {==}
def malformed_ep_parent() -> S.parent(malformed_ep()): ({==},({==},({==},{==})))
def malformed_ep_parent_right() -> {Pos.valid_right(malformed_ep(),1,7,4,1) == True{} : Bool}: {==}
def malformed_ep_rejected() -> {Pos.valid_ep(Chess.get_ep(malformed_ep()),malformed_ep()) == False{} : Bool}: {==}
def malformed_ep_gate_empty() -> {Chess.ep_target(malformed_ep()) == U64.zero() : U64}: {==}
def malformed_ep_ordinary_target() -> {Pos.occupied_by(malformed_ep(),15,1,1) == True{} : Bool}: {==}
def malformed_ep_tagged() -> {Chess.destinations(64n,False{},U64.bit(15n),22,True{},15,Nil{}) ==
  Con{Chess.Ply{22,15,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def malformed_ep_surviving_right() -> {Chess.get_rights(Chess.make_move(malformed_ep(),Chess.Ply{22,15,0,1})) == 1 : U32}: {==}
def malformed_ep_rook_removed() -> {U64.test_bit(Chess.get_rooks(Chess.make_move(malformed_ep(),Chess.Ply{22,15,0,1})),7n) == False{} : Bool}: {==}
def malformed_ep_child_invalid() -> {Pos.valid_right(Chess.make_move(malformed_ep(),Chess.Ply{22,15,0,1}),1,7,4,1) == False{} : Bool}: {==}

# Preserve the exact prior malformed-EP board and conclusions locally. Importing
# its complete fixture module triggers a qualified tail-binder parser collision;
# the original suite is left unchanged.
def prior_malformed_ep() -> Chess.Board:
  Chess.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(24n),U64.bit(60n)),U64.or(U64.bit(8n),U64.bit(24n)),U64.bit(60n),1,0,16}
def prior_malformed_ep_rows() -> {Board.valid(prior_malformed_ep()) == True{} : Bool}: {==}
def prior_malformed_ep_rejected() -> {Pos.valid_ep(16,prior_malformed_ep()) == False{} : Bool}: {==}
def prior_malformed_ep_loses_king() -> {Kings.white(Chess.make_move(prior_malformed_ep(),Chess.Ply{8,16,0,1})) == 0n : Nat}: {==}
