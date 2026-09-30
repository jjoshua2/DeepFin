import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ./ClearActual.bend as Clear
import ./Ordinary.bend as Ordinary
import ../board/Spec.bend as B
import ../board/LAWS.bend as BL
import ../board/PROOF.bend as BP
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def erase_every_plane(+b: Chess.Board,+mask: U64) -> {Clear.run(b,mask) == Spec.erase(b,mask) : Chess.Board}:
  Laws.clear_mask_exact_update(b,mask)

def erase_consistent(+b: Chess.Board,+mask: U64,good: {B.valid(b) == True{} : Bool}) -> {B.valid(Clear.run(b,mask)) == True{} : Bool}:
  Laws.clear_mask_preserves_representation(b,mask,good)

def future_target(+b: Chess.Board,+removed: U64,+target: U64) -> {B.fresh_mask(Clear.run(b,Spec.removal(removed,target)),target) == True{} : Bool}:
  Laws.cleared_target_is_fresh(b,removed,target)

def exact_move(+b: Chess.Board,+src: U32,+dst: U32) -> {Chess.make_move(b,Chess.Ply{src,dst,0,0}) == Ordinary.expected(b,src,dst) : Chess.Board}:
  Laws.ordinary_make_move_exact_update(b,src,dst)

def consistent_move(+b: Chess.Board,+src: U32,+dst: U32,good: {B.valid(b) == True{} : Bool}) -> {B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,0})) == True{} : Bool}:
  Laws.ordinary_make_move_preserves_representation(b,src,dst,good)

# Nonempty capture, with complete expected Board including metadata.
def capture_input() -> Chess.Board:
  Position.put(1,True{},1,Position.put(3,False{},18,Position.empty()))

def capture_is_exact() ->
  {Chess.make_move(capture_input(),Chess.Ply{1,18,0,0}) == Position.metadata(Position.put(1,True{},18,Position.empty()),0,0,64) : Chess.Board}:
  {==}

# Representation safety is not permission to call make_move with illegal inputs.
# The raw decoder fallback can manufacture a king from an empty source.
def empty_source_is_not_legal_move_validation() ->
  {Chess.make_move(Position.empty(),Chess.Ply{0,1,0,0}) == Position.metadata(Position.put(5,True{},1,Position.empty()),0,0,64) : Chess.Board}:
  {==}

# Omitting a plane from clearing can break representation, even at one square.
def not_all_plane_deletions_preserve_partition() ->
  {B.valid(Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(0n),U64.zero(),1,0,64}) == False{} : Bool}:
  {==}

def arbitrary_square_input_not_legal_permission(+src: U32,+dst: U32) ->
  {B.valid(Chess.make_move(Position.empty(),Chess.Ply{src,dst,0,0})) == True{} : Bool}:
  Laws.ordinary_make_move_preserves_representation(Position.empty(),src,dst,BL.empty_representation())
