import Base
import ./Kings.bend as Kings
import ./Validate.bend as V
import ./consumer.bend as Bridge
import ../prepared_king_selection/Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../board/Spec.bend as B
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def game(b: Chess.Board) -> Pos.Game:
  Pos.Game{b,0,1,Nil{}}

def kings(+w: U64,+bl: U64,turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(w,bl),w,bl,turn,0,64}

def simple() -> Chess.Board:
  kings(U64.bit(0n),U64.bit(63n),1)

def simple_validator_exact() ->
  {Protocol.validate(Some{game(simple())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},Some{game(simple())}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def simple_root() -> V.output(Protocol.validate(Some{game(simple())},ALeaf{U64.zero()})):
  Bridge.validated_root(ALeaf{U64.zero()},Some{game(simple())})

def simple_current_index() -> {K.index(simple()) == 0n : Nat}: {==}
def simple_opponent_index() -> {Kings.query_index(simple(),U32.xor(Chess.get_turn(simple()),1)) == 63n : Nat}: {==}

def actual_precheck() -> V.before_query(game(simple()),Pos.valid(simple())):
  Bridge.validator_lookup(game(simple()))

def actual_required() ->
  Bridge.filter_output(V.value(Protocol.validate(Some{game(simple())},ALeaf{U64.zero()})),U64.zero(),1,0,0):
  Bridge.required(ALeaf{U64.zero()},Some{game(simple())},U64.zero(),1,0,0)

def no_input() -> V.output(Protocol.validate(None{},ALeaf{U64.zero()})):
  Bridge.validated_root(ALeaf{U64.zero()},None{})

def empty_rejected() ->
  {Protocol.validate(Some{game(Pos.empty())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},None{}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def multiking() -> Chess.Board:
  kings(U64.or(U64.bit(0n),U64.bit(32n)),U64.bit(63n),1)

def multiking_rejected() ->
  {Protocol.validate(Some{game(multiking())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},None{}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

# These are arbitrary validate inputs, not claims about parse_position output.
def noncanonical() -> Chess.Board:
  kings(U64.bit(0n),U64.bit(63n),2)

def noncanonical_accepted() ->
  {Protocol.validate(Some{game(noncanonical())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},Some{game(noncanonical())}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def noncanonical_selects_black() -> {K.index(noncanonical()) == 63n : Nat}: {==}
def noncanonical_root() -> V.output(Protocol.validate(Some{game(noncanonical())},ALeaf{U64.zero()})):
  Bridge.validated_root(ALeaf{U64.zero()},Some{game(noncanonical())})

def collision() -> Chess.Board:
  Chess.Board{U64.bit(32n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(32n),U64.bit(63n)),
    U64.bit(32n),U64.bit(63n),1,0,64}

def collision_accepted() ->
  {Protocol.validate(Some{game(collision())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},Some{game(collision())}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def collision_partition_false() -> {B.valid(collision()) == False{} : Bool}: {==}
def collision_selected_kind() -> {Chess.piece(collision(),Q.king(collision())) == 0 : U32}: {==}
def collision_root() -> V.output(Protocol.validate(Some{game(collision())},ALeaf{U64.zero()})):
  Bridge.validated_root(ALeaf{U64.zero()},Some{game(collision())})

# Preserve the distinct PR1041 obstructions without rechecking its legal-output suite.
def stale() -> Chess.Board:
  Pos.metadata(Pos.put(0,True{},8,Pos.put(5,True{},0,Pos.put(5,False{},24,Pos.empty()))),1,0,16)

def stale_ep_invalid() -> {Pos.valid_ep(Chess.get_ep(stale()),stale()) == False{} : Bool}: {==}
def stale_rejected() ->
  {Protocol.validate(Some{game(stale())},ALeaf{U64.zero()}) ==
    (ALeaf{U64.zero()},None{}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def jump() -> Chess.Board:
  Pos.metadata(Pos.put(0,True{},32,Pos.put(5,True{},0,
    Pos.put(1,False{},48,Pos.put(5,False{},60,Pos.empty())))),1,0,64)

def jump_root_valid() -> {Pos.valid(jump()) == True{} : Bool}: {==}
def jump_accepted() ->
  {Protocol.validate(Some{game(jump())},ALeaf{U64.bit(48n)}) ==
    (ALeaf{U64.bit(48n)},Some{game(jump())}) : Array<U64> & Maybe<&2,Pos.Game>}:
  {==}

def jump_root() -> V.output(Protocol.validate(Some{game(jump())},ALeaf{U64.bit(48n)})):
  Bridge.validated_root(ALeaf{U64.bit(48n)},Some{game(jump())})
