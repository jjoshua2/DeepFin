import Base
import ./Post.bend as Post
import ./Children.bend as Children
import ./consumer.bend as Bridge
import ../board/Spec.bend as B
import ../candidate_bypass_classification/Generated.bend as G
import ../candidate_bypass_classification/Fields.bend as Fields
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../promotion/Spec.bend as P
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

# The arbitrary leaf arrays exercise implementation outputs, not attack geometry.
def ordinary_board() -> Chess.Board:
  Position.put(0,True{},8,Position.empty())

def ordinary_valid() -> {B.valid(ordinary_board()) == True{} : Bool}: {==}

def ordinary_order() ->
  {Chess.legal_moves(ALeaf{U64.zero()},ordinary_board()) ==
    (ALeaf{U64.zero()},Chess.Ply{8,16,0,0} <> Chess.Ply{8,24,0,0} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def ordinary_member_valid() -> Post.valid(ordinary_board(),Chess.Ply{8,24,0,0}):
  Bridge.legal_member(ALeaf{U64.zero()},ordinary_board(),Chess.Ply{8,24,0,0},ordinary_valid(),Inr{Inl{{==}}})

def ordinary_children_valid() ->
  Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},ordinary_board())),ordinary_board())):
  Bridge.legal_children(ALeaf{U64.zero()},ordinary_board(),ordinary_valid())

def ep_white_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},36,Position.put(0,False{},35,Position.empty())),1,0,43)

def ep_black_board() -> Chess.Board:
  Position.metadata(Position.put(0,False{},28,Position.put(0,True{},27,Position.empty())),0,0,19)

def ep_white_order() ->
  {Chess.legal_moves(ALeaf{U64.bit(43n)},ep_white_board()) ==
    (ALeaf{U64.bit(43n)},Chess.Ply{36,43,0,1} <> Chess.Ply{36,44,0,0} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def ep_black_order() ->
  {Chess.legal_moves(ALeaf{U64.bit(19n)},ep_black_board()) ==
    (ALeaf{U64.bit(19n)},Chess.Ply{28,19,0,1} <> Chess.Ply{28,20,0,0} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def ep_white_valid() -> Post.valid(ep_white_board(),Chess.Ply{36,43,0,1}):
  Bridge.legal_member(ALeaf{U64.bit(43n)},ep_white_board(),Chess.Ply{36,43,0,1},{==},Inl{{==}})

def ep_black_valid() -> Post.valid(ep_black_board(),Chess.Ply{28,19,0,1}):
  Bridge.legal_member(ALeaf{U64.bit(19n)},ep_black_board(),Chess.Ply{28,19,0,1},{==},Inl{{==}})

def ep_white_child_exact() ->
  {Chess.make_move(ep_white_board(),Chess.Ply{36,43,0,1}) ==
    Position.metadata(Position.put(0,True{},43,Position.empty()),0,0,64) : Chess.Board}: {==}

def ep_black_child_exact() ->
  {Chess.make_move(ep_black_board(),Chess.Ply{28,19,0,1}) ==
    Position.metadata(Position.put(0,False{},19,Position.empty()),1,0,64) : Chess.Board}: {==}

def mixed_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},49,
    Position.put(5,False{},60,Position.put(5,True{},4,
      Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))))),1,3,64)

def mixed_valid() -> {B.valid(mixed_board()) == True{} : Bool}: {==}

def mixed_prefilter_order() ->
  {G.produced(ALeaf{U64.zero()},mixed_board()) ==
    (ALeaf{U64.zero()},Chess.Ply{4,2,0,2} <> Chess.Ply{4,6,0,2} <> P.choices(49,57)) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def mixed_legal_order() ->
  {Chess.legal_moves(ALeaf{U64.zero()},mixed_board()) ==
    (ALeaf{U64.zero()},P.ply(49,57,P.Queen{}) <> P.ply(49,57,P.Rook{}) <> P.ply(49,57,P.Bishop{}) <>
      P.ply(49,57,P.Knight{}) <> Chess.Ply{4,6,0,2} <> Chess.Ply{4,2,0,2} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def promotion_member(+choice: P.Choice) ->
  E.member(P.ply(49,57,choice),E.moves(Chess.legal_moves(ALeaf{U64.zero()},mixed_board()))):
  match choice:
    case P.Queen{}: Inl{{==}}
    case P.Rook{}: Inr{Inl{{==}}}
    case P.Bishop{}: Inr{Inr{Inl{{==}}}}
    case P.Knight{}: Inr{Inr{Inr{Inl{{==}}}}}

def all_promotion_members_valid(+choice: P.Choice) -> Post.valid(mixed_board(),P.ply(49,57,choice)):
  Bridge.legal_member(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,choice),mixed_valid(),promotion_member(choice))

def mixed_prefilter_valid() -> C.every(m => Post.valid(mixed_board(),m),E.moves(G.produced(ALeaf{U64.zero()},mixed_board()))):
  Bridge.prefilter_all(ALeaf{U64.zero()},mixed_board(),mixed_valid())

def mixed_legal_valid() -> C.every(m => Post.valid(mixed_board(),m),E.moves(Chess.legal_moves(ALeaf{U64.zero()},mixed_board()))):
  Bridge.legal_all(ALeaf{U64.zero()},mixed_board(),mixed_valid())

def mixed_children_valid() ->
  Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},mixed_board())),mixed_board())):
  Bridge.legal_children(ALeaf{U64.zero()},mixed_board(),mixed_valid())

def black_castle_board() -> Chess.Board:
  Position.metadata(Position.put(5,False{},60,Position.put(3,False{},56,Position.put(3,False{},63,Position.empty()))),0,12,64)

def black_castle_order() ->
  {Chess.legal_moves(ALeaf{U64.zero()},black_castle_board()) ==
    (ALeaf{U64.zero()},Chess.Ply{60,62,0,2} <> Chess.Ply{60,58,0,2} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def black_castle_children_valid() ->
  Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},black_castle_board())),black_castle_board())):
  Bridge.legal_children(ALeaf{U64.zero()},black_castle_board(),{==})

# Each repeated occurrence has its own actual legal-member certificate.
def repeated_promotions() -> List<&2,Chess.Ply>:
  P.ply(49,57,P.Queen{}) <> P.ply(49,57,P.Knight{}) <> P.ply(49,57,P.Queen{}) <> Nil{}

def repeated_fields() -> C.every(m => Fields.generated(mixed_board(),m),repeated_promotions()):
  (G.member_legal(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Queen{}),promotion_member(P.Queen{})),
    (G.member_legal(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Knight{}),promotion_member(P.Knight{})),
      (G.member_legal(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Queen{}),promotion_member(P.Queen{})),Unit{})))

def repeated_children_valid() -> Children.every(Chess.children(repeated_promotions(),mixed_board())):
  Children.lift(mixed_board(),repeated_promotions(),Post.occurrences(mixed_board(),repeated_promotions(),mixed_valid(),repeated_fields()))

def promoted_child(+choice: P.Choice) -> Chess.Board:
  Position.metadata(Position.put(P.tag(choice),True{},57,
    Position.put(5,False{},60,Position.put(5,True{},4,
      Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))))),0,3,64)

def repeated_children_exact() ->
  {Chess.children(repeated_promotions(),mixed_board()) ==
    promoted_child(P.Queen{}) <> promoted_child(P.Knight{}) <> promoted_child(P.Queen{}) <> Nil{} : List<&2,Chess.Board>}:
  {==}

# Representation validity is weaker than legal-input or king existence contracts.
def malformed_promotion_breaks_partition() ->
  {B.valid(Chess.make_move(mixed_board(),Chess.Ply{49,57,6,0})) == False{} : Bool}: {==}

def unguarded_castle_breaks_partition() ->
  {B.valid(Chess.make_move(Position.put(2,True{},5,Position.empty()),Chess.Ply{4,6,0,2})) == False{} : Bool}: {==}

def invalid_parent_is_not_repaired() ->
  {B.valid(Chess.make_move(Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
      U64.bit(20n),U64.zero(),1,0,64},Chess.Ply{0,1,0,0})) == False{} : Bool}: {==}

def empty_children() -> Children.every(Chess.children(Nil{},Position.empty())):
  Children.lift(Position.empty(),Nil{},Unit{})
