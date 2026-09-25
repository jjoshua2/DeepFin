import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

# The preservation theorem does not assume a desired result or limit other pieces.
def arbitrary_board(+b: Chess.Board,+sq: U32,+piece: S.Piece,+white: Bool,
  good: {S.valid(b) == True{} : Bool},fresh: {S.fresh(b,sq) == True{} : Bool}) ->
  {S.valid(Position.put(S.tag(piece),white,sq,b)) == True{} : Bool}:
  Laws.fresh_put_preserves_representation(b,sq,piece,white,good,fresh)

def empty() -> {S.valid(Position.empty()) == True{} : Bool}:
  Laws.empty_representation()

def start() -> {S.valid(Position.board(Position.start())) == True{} : Bool}:
  Laws.initial_representation()

# Non-vacuity includes the top U64 bit and all six typed pieces.
def top_square(+piece: S.Piece,+white: Bool) ->
  {S.valid(Position.put(S.tag(piece),white,63,Position.empty())) == True{} : Bool}:
  Laws.fresh_put_preserves_representation(Position.empty(),63,piece,white,Laws.empty_representation(),{==})

def two_pieces() ->
  {S.valid(Position.put(0,True{},0,Position.put(1,False{},63,Position.empty()))) == True{} : Bool}:
  Laws.fresh_put_preserves_representation(Position.put(1,False{},63,Position.empty()),0,S.Pawn{},True{},
    Laws.fresh_put_preserves_representation(Position.empty(),63,S.Knight{},False{},Laws.empty_representation(),{==}),{==})

# Additive insertion is NOT replacement; omitting freshness is false.
def collision() -> Chess.Board:
  Position.put(0,True{},0,Position.put(1,False{},0,Position.empty()))
def collision_invalid() -> {S.valid(collision()) == False{} : Bool}: {==}
def collision_not_fresh() -> {S.fresh(Position.put(1,False{},0,Position.empty()),0) == False{} : Bool}: {==}

# Metadata validity, king counts, reachability and move legality are not claimed.
def metadata_only(+b: Chess.Board,t: U32,r: U32,e: U32) ->
  {S.valid(Position.metadata(b,t,r,e)) == S.valid(b) : Bool}:
  Laws.metadata_preserves_representation(b,t,r,e)
def parser_put_metadata(+b: Chess.Board,k: U32,c: Bool,q: U32) ->
  {S.metadata(Position.put(k,c,q,b)) == S.metadata(b) : U32 & (U32 & U32)}:
  Laws.put_preserves_metadata(b,k,c,q)

def complete_update(+b: Chess.Board,+sq: U32,+piece: S.Piece,+white: Bool) ->
  {Position.put(S.tag(piece),white,sq,b) == S.insert_mask(b,U64.bit(U32.to_nat(sq)),piece,white) : Chess.Board}:
  Laws.put_exact_bitboard_update(b,sq,piece,white)
