# Small actual target/update witnesses, without normalizing a concrete P2 table.
import Base
import ./consumer.bend as Public
import ./Spec.bend as S
import ./Producer.bend as P
import ../../../legal_probe/Chess.bend as C

def advance_white() -> C.Board:
  C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(8n),U64.zero(),1,0,64}

def advance_black(turn: U32) -> C.Board:
  C.Board{U64.bit(48n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(48n),turn,0,64}

def white_guard() -> S.double(advance_white(),8,24):
  P.generated(advance_white(),8,24,{==},{==},{==})

def black_guard() -> S.double(advance_black(0),48,32):
  P.generated(advance_black(0),48,32,{==},{==},{==})

def noncanonical_guard() -> S.double(advance_black(2),48,32):
  P.generated(advance_black(2),48,32,{==},{==},{==})

def actual_white_target() -> {U64.test_bit(S.mask(C.pawn_targets(ALeaf{U64.zero()},advance_white(),8)),24n) == True{} : Bool}: {==}
def actual_black_target() -> {U64.test_bit(S.mask(C.pawn_targets(ALeaf{U64.zero()},advance_black(0),48)),32n) == True{} : Bool}: {==}
def white_ep() -> {C.get_ep(C.make_move(advance_white(),C.Ply{8,24,0,0})) == 16 : U32}: {==}
def black_ep() -> {C.get_ep(C.make_move(advance_black(0),C.Ply{48,32,0,0})) == 40 : U32}: {==}

def blocked_first() -> C.Board:
  C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(8n),U64.bit(16n)),U64.zero(),1,0,64}
def blocked_second() -> C.Board:
  C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(8n),U64.bit(24n)),U64.zero(),1,0,64}
def blocked_first_target() -> {U64.test_bit(S.mask(C.pawn_targets(ALeaf{U64.zero()},blocked_first(),8)),24n) == False{} : Bool}: {==}
def blocked_second_target() -> {U64.test_bit(S.mask(C.pawn_targets(ALeaf{U64.zero()},blocked_second(),8)),24n) == False{} : Bool}: {==}

# Arbitrary table geometry is essential to the existing jump counterexample.
def jump() -> C.Board:
  C.Board{U64.bit(32n),U64.bit(48n),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(60n)),
    U64.or(U64.bit(0n),U64.bit(32n)),U64.or(U64.bit(48n),U64.bit(60n)),1,0,64}
def arbitrary_jump_target() -> {U64.test_bit(S.mask(C.pawn_targets(ALeaf{U64.bit(48n)},jump(),32)),48n) == True{} : Bool}: {==}
def arbitrary_jump_not_double() -> {S.empty2(jump(),32) == False{} : Bool}: {==}
def initialized_jump_excluded() -> {U64.test_bit(S.targets(jump(),32),48n) == False{} : Bool}: {==}
def arbitrary_jump_ep() -> {C.get_ep(C.make_move(jump(),C.Ply{32,48,0,0})) == 40 : U32}: {==}

# Separate stale raw EP tag: an advance is tagged by the destination==ep rule.
# Parent root validation rejects it; this producer lemma does not repair the tag.
def stale() -> C.Board:
  C.Board{U64.bit(8n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(24n)),
    U64.or(U64.bit(0n),U64.bit(8n)),U64.bit(24n),1,0,16}
def stale_ep_unavailable() -> {C.ep_target(stale()) == U64.zero() : U64}: {==}
def stale_actual_scan() ->
  {C.scan_after(8,True{},16,Nil{},C.pawn_targets(ALeaf{U64.zero()},stale(),8)) ==
   (ALeaf{U64.zero()},C.Ply{8,16,0,1} <> Nil{}) : Array<U64> & List<&2,C.Ply>}: {==}
def stale_king_loss() -> {C.get_kings(C.make_move(stale(),C.Ply{8,16,0,1})) == U64.bit(0n) : U64}: {==}
