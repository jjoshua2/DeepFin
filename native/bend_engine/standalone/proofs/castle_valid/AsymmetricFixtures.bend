# Independently reducible full legal-list EP counterexamples at the other three corners.
import Base
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def wq_parent() -> Chess.Board:
  Chess.Board{U64.bit(16n),U64.zero(),U64.zero(),U64.bit(0n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(4n),U64.bit(0n)),U64.or(U64.bit(60n),U64.bit(16n)),0,2,8}
def wq_rows() -> {Board.valid(wq_parent()) == True{} : Bool}: {==}
def wq_turn() -> S.canonical(wq_parent()): {==}
def wq_rights() -> S.parent(wq_parent()): ({==},({==},({==},{==})))
def wq_missing_ep() -> {Pos.valid_ep(Chess.get_ep(wq_parent()),wq_parent()) == False{} : Bool}: {==}
def wq_full_legal_list() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),wq_parent())) == Con{Chess.Ply{16,8,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def wq_member() -> E.member(Chess.Ply{16,8,0,1},E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),wq_parent()))):
  %Equal.sym(List<&2,Chess.Ply>,E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),wq_parent())),Con{Chess.Ply{16,8,0,1},Nil{}},wq_full_legal_list()) : E.member(Chess.Ply{16,8,0,1},_)
  Inl{{==}}
def wq_right_survives() -> {Chess.get_rights(Chess.make_move(wq_parent(),Chess.Ply{16,8,0,1})) == 2 : U32}: {==}
def wq_rook_removed() -> {U64.test_bit(Chess.get_rooks(Chess.make_move(wq_parent(),Chess.Ply{16,8,0,1})),0n) == False{} : Bool}: {==}
def wq_child_invalid() -> {Pos.valid_right(Chess.make_move(wq_parent(),Chess.Ply{16,8,0,1}),2,0,4,1) == False{} : Bool}: {==}

def bk_parent() -> Chess.Board:
  Chess.Board{U64.bit(47n),U64.zero(),U64.zero(),U64.bit(63n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(4n),U64.bit(47n)),U64.or(U64.bit(60n),U64.bit(63n)),1,4,55}
def bk_rows() -> {Board.valid(bk_parent()) == True{} : Bool}: {==}
def bk_turn() -> S.canonical(bk_parent()): {==}
def bk_rights() -> S.parent(bk_parent()): ({==},({==},({==},{==})))
def bk_missing_ep() -> {Pos.valid_ep(Chess.get_ep(bk_parent()),bk_parent()) == False{} : Bool}: {==}
def bk_full_legal_list() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bk_parent())) == Con{Chess.Ply{47,55,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def bk_member() -> E.member(Chess.Ply{47,55,0,1},E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bk_parent()))):
  %Equal.sym(List<&2,Chess.Ply>,E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bk_parent())),Con{Chess.Ply{47,55,0,1},Nil{}},bk_full_legal_list()) : E.member(Chess.Ply{47,55,0,1},_)
  Inl{{==}}
def bk_right_survives() -> {Chess.get_rights(Chess.make_move(bk_parent(),Chess.Ply{47,55,0,1})) == 4 : U32}: {==}
def bk_rook_removed() -> {U64.test_bit(Chess.get_rooks(Chess.make_move(bk_parent(),Chess.Ply{47,55,0,1})),63n) == False{} : Bool}: {==}
def bk_child_invalid() -> {Pos.valid_right(Chess.make_move(bk_parent(),Chess.Ply{47,55,0,1}),4,63,60,0) == False{} : Bool}: {==}

def bq_parent() -> Chess.Board:
  Chess.Board{U64.bit(40n),U64.zero(),U64.zero(),U64.bit(56n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(4n),U64.bit(40n)),U64.or(U64.bit(60n),U64.bit(56n)),1,8,48}
def bq_rows() -> {Board.valid(bq_parent()) == True{} : Bool}: {==}
def bq_turn() -> S.canonical(bq_parent()): {==}
def bq_rights() -> S.parent(bq_parent()): ({==},({==},({==},{==})))
def bq_missing_ep() -> {Pos.valid_ep(Chess.get_ep(bq_parent()),bq_parent()) == False{} : Bool}: {==}
def bq_full_legal_list() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bq_parent())) == Con{Chess.Ply{40,48,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def bq_member() -> E.member(Chess.Ply{40,48,0,1},E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bq_parent()))):
  %Equal.sym(List<&2,Chess.Ply>,E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),bq_parent())),Con{Chess.Ply{40,48,0,1},Nil{}},bq_full_legal_list()) : E.member(Chess.Ply{40,48,0,1},_)
  Inl{{==}}
def bq_right_survives() -> {Chess.get_rights(Chess.make_move(bq_parent(),Chess.Ply{40,48,0,1})) == 8 : U32}: {==}
def bq_rook_removed() -> {U64.test_bit(Chess.get_rooks(Chess.make_move(bq_parent(),Chess.Ply{40,48,0,1})),56n) == False{} : Bool}: {==}
def bq_child_invalid() -> {Pos.valid_right(Chess.make_move(bq_parent(),Chess.Ply{40,48,0,1}),8,56,60,0) == False{} : Bool}: {==}

# Nonzero15: own castles clear both own bits; cross-corner captures clear both touched bits.
def complete_homes(turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.or(U64.bit(7n),U64.or(U64.bit(56n),U64.bit(63n)))),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(0n),U64.or(U64.bit(4n),U64.bit(7n))),U64.or(U64.bit(56n),U64.or(U64.bit(60n),U64.bit(63n))),turn,15,64}
def white_ks_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(1),Chess.Ply{4,6,0,2})) == 12 : U32}: {==}
def white_ks_all_child_checks() -> S.parent(Chess.make_move(complete_homes(1),Chess.Ply{4,6,0,2})): ({==},({==},({==},{==})))
def white_qs_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(1),Chess.Ply{4,2,0,2})) == 12 : U32}: {==}
def white_qs_all_child_checks() -> S.parent(Chess.make_move(complete_homes(1),Chess.Ply{4,2,0,2})): ({==},({==},({==},{==})))
def black_ks_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(0),Chess.Ply{60,62,0,2})) == 3 : U32}: {==}
def black_ks_all_child_checks() -> S.parent(Chess.make_move(complete_homes(0),Chess.Ply{60,62,0,2})): ({==},({==},({==},{==})))
def black_qs_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(0),Chess.Ply{60,58,0,2})) == 3 : U32}: {==}
def black_qs_all_child_checks() -> S.parent(Chess.make_move(complete_homes(0),Chess.Ply{60,58,0,2})): ({==},({==},({==},{==})))
def white_ks_capture_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(1),Chess.Ply{7,63,0,0})) == 10 : U32}: {==}
def white_ks_capture_all_child_checks() -> S.parent(Chess.make_move(complete_homes(1),Chess.Ply{7,63,0,0})): ({==},({==},({==},{==})))
def white_qs_capture_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(1),Chess.Ply{0,56,0,0})) == 5 : U32}: {==}
def white_qs_capture_all_child_checks() -> S.parent(Chess.make_move(complete_homes(1),Chess.Ply{0,56,0,0})): ({==},({==},({==},{==})))
def black_ks_capture_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(0),Chess.Ply{63,7,0,0})) == 10 : U32}: {==}
def black_ks_capture_all_child_checks() -> S.parent(Chess.make_move(complete_homes(0),Chess.Ply{63,7,0,0})): ({==},({==},({==},{==})))
def black_qs_capture_rights() -> {Chess.get_rights(Chess.make_move(complete_homes(0),Chess.Ply{56,0,0,0})) == 5 : U32}: {==}
def black_qs_capture_all_child_checks() -> S.parent(Chess.make_move(complete_homes(0),Chess.Ply{56,0,0,0})): ({==},({==},({==},{==})))
