# Actual predicate acceptance does not establish physical attack geometry.
import Base
import ./Boards.bend as B
import ../accepted_castle/Spec.bend as S
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C
def accepted_observation() ->
  {C.in_check(ALeaf{U64.bit(43n)},C.make_move(B.discovered(),C.Ply{36,43,0,1}),1) ==
    (ALeaf{U64.bit(43n)},False{}) : Array<U64> & Bool}: {==}
def exposed_rook() ->
  {U64.and(U64.and(Tables.slider(32,False{},C.occupied(C.make_move(B.discovered(),C.Ply{36,43,0,1})),False{}),
    C.get_rooks(C.make_move(B.discovered(),C.Ply{36,43,0,1}))),C.color(C.make_move(B.discovered(),C.Ply{36,43,0,1}),0)) ==
    U64.bit(39n) : U64}: {==}
def moving_side_checked() ->
  {S.answer(C.in_check(ALeaf{U64.bit(39n)},C.make_move(B.discovered(),C.Ply{36,43,0,1}),C.get_turn(B.discovered()))) == True{} : Bool}: {==}
def child_side_unchecked() ->
  {S.answer(C.in_check(ALeaf{U64.bit(39n)},C.make_move(B.discovered(),C.Ply{36,43,0,1}),
    C.get_turn(C.make_move(B.discovered(),C.Ply{36,43,0,1})))) == False{} : Bool}: {==}
