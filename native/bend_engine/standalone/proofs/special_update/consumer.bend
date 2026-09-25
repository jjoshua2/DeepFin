import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def promote_any(+b: Chess.Board,+src: U32,+dst: U32,+p: S.Promotion,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0})) == True{} : Bool}:
  Laws.typed_promotion_preserves_representation(b,src,dst,p,good)

def ep_any(+b: Chess.Board,+src: U32,+dst: U32,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}:
  Laws.en_passant_preserves_representation(b,src,dst,good)

# Label meanings cannot drift together with both sides of an exact-update law.
def tags() -> {(S.tag(S.Knight{}),(S.tag(S.Bishop{}),(S.tag(S.Rook{}),S.tag(S.Queen{})))) == (1,(2,(3,4))) : U32 & (U32 & (U32 & U32))}:
  {==}

# White pawn on a7 captures a black rook on b8 and becomes a knight.
def promotion_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},48,Position.put(3,False{},57,Position.empty())),1,15,64)

def promotion_capture() ->
  {Chess.make_move(promotion_board(),Chess.Ply{48,57,1,0}) ==
    Position.metadata(Position.put(1,True{},57,Position.empty()),0,15,64) : Chess.Board}:
  {==}

# White e5xd6 en passant removes the pawn on d5, not merely destination d6.
def ep_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},36,Position.put(0,False{},35,Position.empty())),1,15,43)

def ep_capture() ->
  {Chess.make_move(ep_board(),Chess.Ply{36,43,0,1}) ==
    Position.metadata(Position.put(0,True{},43,Position.empty()),0,15,64) : Chess.Board}:
  {==}

# A tag outside the four promotion choices can leave color without any kind.
def invalid_promotion_is_not_covered() ->
  {B.valid(Chess.make_move(promotion_board(),Chess.Ply{48,57,6,0})) == False{} : Bool}:
  {==}
