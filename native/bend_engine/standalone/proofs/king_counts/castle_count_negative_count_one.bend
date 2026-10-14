# MUST reject at reject_count_one: preservation does not force an initial count of one.
import Base
import ./CastleCount.bend as C
import ./Actual.bend as K
import ./castle_frame_fixtures.bend as F
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def white_result(proof: K.preserved(F.board(1),E.move(F.board(1),True{}))) ->
  {K.white(Chess.make_move(F.board(1),E.move(F.board(1),True{}))) == K.white(F.board(1)) : Nat}:
  (white,black)=proof
  white
def reject_count_one() ->
  {K.white(Chess.make_move(F.board(1),E.move(F.board(1),True{}))) == 1n : Nat}:
  Equal.trans(Nat,K.white(Chess.make_move(F.board(1),E.move(F.board(1),True{}))),K.white(F.board(1)),1n,white_result(C.preserved(F.board(1),True{},F.rows(1),{==})),{==})
