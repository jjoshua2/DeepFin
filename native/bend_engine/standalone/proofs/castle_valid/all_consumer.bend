# All four actual child checks under exactly the public parent premises.
import Base
import ./AllProof.bend as Proof
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def preserves_all_rights(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  rows: {Board.valid(b) == True{} : Bool},turn: S.canonical(b),
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},parent: S.parent(b),
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) ->
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool} &
  ({Pos.valid_right(Chess.make_move(b,m),2,0,4,1) == True{} : Bool} &
  ({Pos.valid_right(Chess.make_move(b,m),4,63,60,0) == True{} : Bool} &
  {Pos.valid_right(Chess.make_move(b,m),8,56,60,0) == True{} : Bool})):
  Proof.legal_member(table,b,m,rows,turn,coherent,parent,here)
