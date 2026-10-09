import Base
import ./Range.bend as R
import ./Source.bend as Source
import ./Post.bend as Post
import ./Children.bend as Children
import ./consumer.bend as Bridge
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../promotion/Spec.bend as P
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

# Small local boards avoid importing and re-normalizing the prior fixture suite.
def ordinary_board() -> Chess.Board:
  Position.put(0,True{},8,Position.empty())

def ep_white_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},36,Position.put(0,False{},35,Position.empty())),1,0,43)

def ep_black_board() -> Chess.Board:
  Position.metadata(Position.put(0,False{},28,Position.put(0,True{},27,Position.empty())),0,0,19)

def mixed_board() -> Chess.Board:
  Position.metadata(Position.put(0,True{},49,
    Position.put(5,False{},60,Position.put(5,True{},4,
      Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))))),1,3,64)

def mixed_valid() -> {B.valid(mixed_board()) == True{} : Bool}: {==}

def black_castle_board() -> Chess.Board:
  Position.metadata(Position.put(5,False{},60,Position.put(3,False{},56,Position.put(3,False{},63,Position.empty()))),0,12,64)

def promotion_member(+choice: P.Choice) ->
  E.member(P.ply(49,57,choice),E.moves(Chess.legal_moves(ALeaf{U64.zero()},mixed_board()))):
  match choice:
    case P.Queen{}: Inl{{==}}
    case P.Rook{}: Inr{Inl{{==}}}
    case P.Bishop{}: Inr{Inr{Inl{{==}}}}
    case P.Knight{}: Inr{Inr{Inr{Inl{{==}}}}}

def repeated_promotions() -> List<&2,Chess.Ply>:
  P.ply(49,57,P.Queen{}) <> P.ply(49,57,P.Knight{}) <> P.ply(49,57,P.Queen{}) <> Nil{}

def promoted_child(+choice: P.Choice) -> Chess.Board:
  Position.metadata(Position.put(P.tag(choice),True{},57,
    Position.put(5,False{},60,Position.put(5,True{},4,
      Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))))),0,3,64)

def ordinary() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},ordinary_board())),ordinary_board()),False{}):
  Bridge.legal_children(ALeaf{U64.zero()},ordinary_board(),True{},{==},{==})

def ordinary_ep_midpoint() -> {Chess.get_ep(Chess.make_move(ordinary_board(),Chess.Ply{8,24,0,0})) == 16 : U32}: {==}

def ep_white() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.bit(43n)},ep_white_board())),ep_white_board()),False{}):
  Bridge.legal_children(ALeaf{U64.bit(43n)},ep_white_board(),True{},{==},{==})

def ep_black() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.bit(19n)},ep_black_board())),ep_black_board()),True{}):
  Bridge.legal_children(ALeaf{U64.bit(19n)},ep_black_board(),False{},{==},{==})

def mixed() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},mixed_board())),mixed_board()),False{}):
  Bridge.legal_children(ALeaf{U64.zero()},mixed_board(),True{},mixed_valid(),{==})

def black_castles() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},black_castle_board())),black_castle_board()),True{}):
  Bridge.legal_children(ALeaf{U64.zero()},black_castle_board(),False{},{==},{==})

def promotion_ready(+choice: P.Choice) -> Post.child(mixed_board(),P.ply(49,57,choice),True{}):
  Bridge.legal_member(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,choice),True{},mixed_valid(),{==},promotion_member(choice))

def repeated_fields() -> C.every(m => Source.generated(mixed_board(),m),repeated_promotions()):
  (Source.member(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Queen{}),promotion_member(P.Queen{})),
    (Source.member(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Knight{}),promotion_member(P.Knight{})),
      (Source.member(ALeaf{U64.zero()},mixed_board(),P.ply(49,57,P.Queen{}),promotion_member(P.Queen{})),Unit{})))

def repeated() -> Children.every(Chess.children(repeated_promotions(),mixed_board()),False{}):
  Children.lift(mixed_board(),repeated_promotions(),True{},
    Post.occurrences(mixed_board(),repeated_promotions(),True{},mixed_valid(),{==},repeated_fields()))

def repeated_exact() ->
  {Chess.children(repeated_promotions(),mixed_board()) ==
    promoted_child(P.Queen{}) <> promoted_child(P.Knight{}) <> promoted_child(P.Queen{}) <> Nil{} : List<&2,Chess.Board>}:
  {==}

# Actual stale metadata, with both parent kings, can erase the NEXT side's king.
def stale() -> Chess.Board:
  Position.metadata(Position.put(0,True{},8,Position.put(5,True{},0,Position.put(5,False{},24,Position.empty()))),1,0,16)

def stale_valid() -> {B.valid(stale()) == True{} : Bool}: {==}
def stale_own_nonempty() -> {U64.is_zero(K.plane(stale())) == False{} : Bool}: {==}
def stale_enemy_nonempty() -> {U64.is_zero(U64.and(Chess.get_kings(stale()),Chess.get_black(stale()))) == False{} : Bool}: {==}
def stale_ep_rejected() -> {Chess.ep_target(stale()) == U64.zero() : U64}: {==}

def stale_legal_order() ->
  {Chess.legal_moves(ALeaf{U64.zero()},stale()) ==
    (ALeaf{U64.zero()},Chess.Ply{8,16,0,1} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}: {==}

def stale_member_ready() -> Post.child(stale(),Chess.Ply{8,16,0,1},True{}):
  Bridge.legal_member(ALeaf{U64.zero()},stale(),Chess.Ply{8,16,0,1},True{},stale_valid(),{==},Inl{{==}})

def stale_child_missing_next_king() ->
  {U64.is_zero(K.plane(Chess.make_move(stale(),Chess.Ply{8,16,0,1}))) == True{} : Bool}: {==}

def stale_selected_64() -> {K.index(Chess.make_move(stale(),Chess.Ply{8,16,0,1})) == 64n : Nat}: {==}

# Arbitrary input attack arrays can emit a capture with src xor dst == 16.
# Its regenerated EP is in range but does not satisfy the next side's EP guard.
def jump() -> Chess.Board:
  Position.metadata(Position.put(0,True{},32,Position.put(5,True{},0,
    Position.put(1,False{},48,Position.put(5,False{},60,Position.empty())))),1,0,64)

def jump_valid() -> {B.valid(jump()) == True{} : Bool}: {==}
def jump_legal_order() ->
  {Chess.legal_moves(ALeaf{U64.bit(48n)},jump()) ==
    (ALeaf{U64.bit(48n)},Chess.Ply{32,48,0,0} <> Chess.Ply{0,48,0,0} <> Nil{}) : Array<U64> & List<&2,Chess.Ply>}: {==}

def jump_member_ready() -> Post.child(jump(),Chess.Ply{32,48,0,0},True{}):
  Bridge.legal_member(ALeaf{U64.bit(48n)},jump(),Chess.Ply{32,48,0,0},True{},jump_valid(),{==},Inl{{==}})

def jump_ep_exact() -> {Chess.get_ep(Chess.make_move(jump(),Chess.Ply{32,48,0,0})) == 40 : U32}: {==}
def jump_ep_incoherent() -> {Chess.ep_target(Chess.make_move(jump(),Chess.Ply{32,48,0,0})) == U64.zero() : U64}: {==}

# Canonical turn remains an explicit parent premise.
def invalid_turn() -> {Chess.get_turn(Chess.make_move(Position.metadata(Position.empty(),2,0,64),Chess.Ply{0,1,0,0})) == 3 : U32}: {==}

# Bounds are essential to the generic arithmetic helper with arbitrary kind input.
def outside_midpoint() -> {U32.div(U32.add(64,80),2) == 72 : U32}: {==}
def outside_bounded() -> {Nat.is_lt(U32.to_nat(U32.div(U32.add(64,80),2)),64n) == False{} : Bool}: {==}

def extreme_parent_ep() -> Children.every(Chess.children(E.moves(Chess.legal_moves(ALeaf{U64.zero()},
  Position.metadata(ordinary_board(),1,0,4294967295))),Position.metadata(ordinary_board(),1,0,4294967295)),False{}):
  Bridge.legal_children(ALeaf{U64.zero()},Position.metadata(ordinary_board(),1,0,4294967295),True{},{==},{==})

def empty_children() -> Children.every(Chess.children(Nil{},Position.empty()),False{}):
  Children.lift(Position.empty(),Nil{},True{},Unit{})
