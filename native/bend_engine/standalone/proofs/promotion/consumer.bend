import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../board/LAWS.bend as BL
import ../board/PROOF.bend as BP
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def arbitrary_promotion(+b: Chess.Board,+src: U32,+dst: U32,+c: S.Choice,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,S.ply(src,dst,c))) == True{} : Bool}:
  Laws.promotion_make_move_preserves_representation(b,src,dst,c,good)

def exact(+b: Chess.Board,+src: U32,+dst: U32,+c: S.Choice) ->
  {Chess.make_move(b,S.ply(src,dst,c)) == S.expected(b,src,dst,c) : Chess.Board}:
  Laws.promotion_make_move_exact_update(b,src,dst,c)

def choice(+c: S.Choice) -> {S.from_tag(S.tag(c)) == Some{c} : Maybe<&2,S.Choice>}:
  Laws.promotion_choice_roundtrip(c)

def generated(+src: U32,+dst: U32,flag: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.put_move(True{},src,dst,flag,tail) == List.append(&2,Chess.Ply,S.choices(src,dst),tail) : List<&2,Chess.Ply>}:
  Laws.promotion_choices_exact(src,dst,flag,tail)

# No pawn/king tags are present in the typed promotion interface.
def invalid_tags() -> {S.from_tag(0) == None{} : Maybe<&2,S.Choice>}:
  {==}
def king_not_choice() -> {S.from_tag(5) == None{} : Maybe<&2,S.Choice>}:
  {==}

# Real capture promotion: white a7 pawn captures b8 rook and becomes a knight.
def capture_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},48,Position.put(3,False{},57,Position.empty())),1,15,64)

def capture_result() ->
  {Chess.make_move(capture_board(),S.ply(48,57,S.Knight{})) ==
    Position.metadata(Position.put(1,True{},57,Position.empty()),0,15,64) : Chess.Board}:
  {==}

# Wrong-rank/empty-source raw calls are not rejected by make_move.
def raw_nonlegal_result() ->
  {Chess.make_move(Position.empty(),S.ply(0,1,S.Queen{})) ==
    Position.metadata(Position.put(4,True{},1,Position.empty()),0,0,64) : Chess.Board}:
  {==}

# Exact four distinct entries, and preserve the supplied nonempty tail.
def choices_keep_tail() ->
  {Chess.put_move(True{},48,56,99,Chess.Ply{9,10,0,0} <> Nil{}) ==
    Chess.Ply{48,56,1,0} <> Chess.Ply{48,56,2,0} <> Chess.Ply{48,56,3,0} <> Chess.Ply{48,56,4,0} <> Chess.Ply{9,10,0,0} <> Nil{} : List<&2,Chess.Ply>}:
  {==}
