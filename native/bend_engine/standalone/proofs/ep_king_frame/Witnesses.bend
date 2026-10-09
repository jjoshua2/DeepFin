# Concrete witnesses retain both directions and the excluded counterexamples.
import Base
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../destination_factorization/Spec.bend as Count
import ../ep_occurrence_tag/Witnesses.bend as Tagged
import ../child_ep_coherence/Witnesses.bend as Prior
import ./Frame.bend as F
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def white() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(63n)),U64.or(U64.bit(0n),U64.bit(36n)),U64.or(U64.bit(63n),U64.bit(35n)),1,0,43}
def black() -> C.Board:
  C.Board{U64.or(U64.bit(27n),U64.bit(28n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(0n),U64.bit(63n)),U64.or(U64.bit(0n),U64.bit(28n)),U64.or(U64.bit(63n),U64.bit(27n)),0,0,20}
def white_partition() -> {Board.valid(white()) == True{} : Bool}: {==}
def white_coherent() -> {Position.valid_ep(43,white()) == True{} : Bool}: {==}
def black_partition() -> {Board.valid(black()) == True{} : Bool}: {==}
def black_coherent() -> {Position.valid_ep(20,black()) == True{} : Bool}: {==}
def white_kings() -> {C.get_kings(F.moved(white(),36,43)) == C.get_kings(white()) : U64}: {==}
def black_kings() -> {C.get_kings(F.moved(black(),27,20)) == C.get_kings(black()) : U64}: {==}
def white_old_plane() -> {F.plane(F.moved(white(),36,43),1) == U64.bit(0n) : U64}: {==}
def black_old_plane() -> {F.plane(F.moved(black(),27,20),0) == U64.bit(63n) : U64}: {==}
def flipped_plane() -> {F.plane(F.moved(white(),36,43),C.get_turn(F.moved(white(),36,43))) == U64.bit(63n) : U64}: {==}
def white_victim_removed() -> {U64.test_bit(C.get_pawns(F.moved(white(),36,43)),35n) == False{} : Bool}: {==}

# Partition and coherent EP do not imply accepted-root king uniqueness.
def multiple() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.or(U64.bit(0n),U64.bit(1n)),U64.bit(63n)),U64.or(U64.or(U64.bit(0n),U64.bit(1n)),U64.bit(36n)),U64.or(U64.bit(63n),U64.bit(35n)),1,0,43}
def multiple_partition() -> {Board.valid(multiple()) == True{} : Bool}: {==}
def multiple_coherent() -> {Position.valid_ep(43,multiple()) == True{} : Bool}: {==}
def multiple_root_invalid() -> {Position.valid(multiple()) == False{} : Bool}: {==}
def multiple_kings() -> {C.get_kings(F.moved(multiple(),36,43)) == C.get_kings(multiple()) : U64}: {==}

# Source pawn/king overlap is permitted by actual priority decoding, but violates
# the explicit partition premise and clears that source king in the raw update.
def overlap() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.or(U64.bit(0n),U64.bit(36n)),U64.bit(63n)),U64.or(U64.or(U64.bit(0n),U64.bit(36n)),U64.zero()),U64.or(U64.bit(63n),U64.bit(35n)),1,0,43}
def overlap_invalid() -> {Board.valid(overlap()) == False{} : Bool}: {==}
def overlap_actual_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},overlap()))): Inl{{==}}
def overlap_coherent() -> {Position.valid_ep(43,overlap()) == True{} : Bool}: {==}
def overlap_loss() -> {C.get_kings(F.moved(overlap(),36,43)) == U64.or(U64.bit(0n),U64.bit(63n)) : U64}: {==}

# Stale EP is an actual full-Ply legal occurrence for its specified arbitrary
# array, yet its incoherent parent allows opposing-king loss.
def stale_actual_member() -> E.member(C.Ply{8,16,0,1},E.moves(C.legal_moves(ALeaf{U64.zero()},Prior.stale()))): Inl{{==}}
def stale_partition() -> {Board.valid(Prior.stale()) == True{} : Bool}: {==}
def stale_incoherent() -> {Position.valid_ep(16,Prior.stale()) == False{} : Bool}: {==}
def stale_king_loss() -> {C.get_kings(F.moved(Prior.stale(),8,16)) == U64.bit(0n) : U64}: {==}
def arbitrary_jump_incoherent() -> {Prior.child(Prior.jump(),C.Ply{32,48,0,0}) == False{} : Bool}: {==}

# The inherited arbitrary multibit scan retains the same two full-Ply tail nodes.
def duplicate_count() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: {==}
def exact_table() -> {Tagged.result_table(Tagged.tagged()) == ALeaf{U64.bit(47n)} : Array<U64>}: {==}
