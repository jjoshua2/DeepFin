# Source-derived candidate, independently checker-validated with Fixtures.
# Actual full legal-list witness: EP coherence is the missing parent premise.
import Base
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def parent() -> Chess.Board:
  Chess.Board{U64.bit(23n),U64.zero(),U64.zero(),U64.bit(7n),U64.zero(),U64.or(U64.bit(4n),U64.bit(60n)),
    U64.or(U64.bit(4n),U64.bit(7n)),U64.or(U64.bit(23n),U64.bit(60n)),0,1,15}
def rows() -> {Board.valid(parent()) == True{} : Bool}: {==}
def turn() -> S.canonical(parent()): {==}
def rights() -> S.parent(parent()): ({==},({==},({==},{==})))
def missing_ep() -> {Pos.valid_ep(Chess.get_ep(parent()),parent()) == False{} : Bool}: {==}
def full_legal_list() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),parent())) ==
  Con{Chess.Ply{23,15,0,1},Nil{}} : List<&2,Chess.Ply>}: {==}
def member() -> E.member(Chess.Ply{23,15,0,1},E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),parent()))):
  %Equal.sym(List<&2,Chess.Ply>,E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),parent())),
    Con{Chess.Ply{23,15,0,1},Nil{}},full_legal_list()) : E.member(Chess.Ply{23,15,0,1},_)
  Inl{{==}}
def child_right_survives() -> {Chess.get_rights(Chess.make_move(parent(),Chess.Ply{23,15,0,1})) == 1 : U32}: {==}
def child_rook_removed() -> {U64.test_bit(Chess.get_rooks(Chess.make_move(parent(),Chess.Ply{23,15,0,1})),7n) == False{} : Bool}: {==}
def child_invalid() -> {Pos.valid_right(Chess.make_move(parent(),Chess.Ply{23,15,0,1}),1,7,4,1) == False{} : Bool}: {==}
