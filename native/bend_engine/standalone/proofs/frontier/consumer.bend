import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../parser/Spec.bend as PS
import ../board/Spec.bend as B
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess

# All public statements are consumed, not just imported as uninstantiated laws.
def arbitrary_suffix(+text: String,+l: P.Layout,e: {S.safe(l) == True{} : Bool}) ->
  {S.safe(P.layout(text,l)) == True{} : Bool}:
  Laws.layout_preserves_frontier(text,l,e)

def accepted_prefix(+prefix: String,+suffix: String,
  e: {PS.accepted(P.layout_result(P.layout(String.append(prefix,suffix),S.initial()))) == True{} : Bool}) ->
  {S.ready(P.layout(prefix,S.initial())) == True{} : Bool}:
  Laws.accepted_prefix_has_frontier(prefix,suffix,e)

def accepted_piece(+prefix: String,+pc: B.Piece,+white: Bool,
  e: {PS.live(P.layout(S.typed(pc,white),P.layout(prefix,S.initial()))) == True{} : Bool}) ->
  {S.fresh_target(P.layout(prefix,S.initial())) == True{} : Bool}:
  Laws.live_typed_insertion_is_fresh(prefix,pc,white,e)

def arbitrary_placement(+text: String) ->
  {S.result_valid(P.layout_result(P.layout(text,S.initial()))) == True{} : Bool}:
  Laws.accepted_placement_has_consistent_board(text)

# Positive full-board/rank-boundary instances with nonzero piece data.
def mixed_placement_is_accepted() ->
  {PS.accepted(P.layout_result(P.layout("P7/8/8/8/8/8/8/7k",S.initial()))) == True{} : Bool}: {==}

def after_first_rank() ->
  {S.ready(P.layout("P7/",S.initial())) == True{} : Bool}:
  Laws.accepted_prefix_has_frontier("P7/","8/8/8/8/8/8/7k",mixed_placement_is_accepted())

def final_square_is_fresh() ->
  {S.fresh_target(P.layout("P7/8/8/8/8/8/8/7",S.initial())) == True{} : Bool}:
  Laws.live_typed_insertion_is_fresh("P7/8/8/8/8/8/8/7",B.King{},False{},{==})

def consistent_complete_placement() ->
  {S.result_valid(P.layout_result(P.layout("P7/8/8/8/8/8/8/7k",S.initial()))) == True{} : Bool}:
  Laws.accepted_placement_has_consistent_board("P7/8/8/8/8/8/8/7k")

# A full rank is still a live frontier, but no ninth piece is permitted there.
def ninth_piece_is_invalid() ->
  {PS.live(P.layout("PPPPPPPPP",S.initial())) == False{} : Bool}: {==}

# A true flag alone does not supply a safe frontier.
def wrapped_caller_is_not_safe() ->
  {S.safe(P.Layout{P.empty(),4294967295,0,True{}}) == False{} : Bool}: {==}

def occupied_ahead_is_not_safe() ->
  {S.safe(P.Layout{P.put(0,True{},56,P.empty()),0,7,True{}}) == False{} : Bool}: {==}

# Invalid raw states need not retain a consistent temporary board.
def invalid_character_can_corrupt_temporary_partition() ->
  {B.valid(PS.board(P.layout("x",S.initial()))) == False{} : Bool}: {==}

def invalid_result_is_not_a_board() ->
  {P.layout_result(P.layout("x",S.initial())) == None{} : Maybe<&2,Chess.Board>}: {==}
