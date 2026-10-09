# Concrete calls distinguish emitted EP tags from safe EP captures.
import Base
import ../child_ep_coherence/Witnesses.bend as Prior
import ../destination_factorization/Spec.bend as Count
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position
import ../board/Spec.bend as Board

def repeated() -> List<&2,C.Ply>: C.Ply{8,16,0,1} <> C.Ply{8,16,0,1} <> Nil{}
def targets() -> U64: U64.or(U64.bit(16n),U64.bit(24n))
def tagged() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(8,True{},16,repeated(),(ALeaf{U64.bit(47n)},targets()))
def duplicate_count() -> {Count.count(E.moves(tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: {==}
def second_destination() -> {Count.count(E.moves(tagged()),C.Ply{8,24,0,0}) == 1n : Nat}: {==}
def wrong_ep_flag_absent() -> {Count.count(E.moves(tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: {==}
def wrong_promotion_absent() -> {Count.count(E.moves(tagged()),C.Ply{8,16,1,1}) == 0n : Nat}: {==}
def result_table(r: Array<U64> & List<&2,C.Ply>) -> Array<U64>:
  (table,moves)=r
  table
def exact_table() -> {result_table(tagged()) == ALeaf{U64.bit(47n)} : Array<U64>}: {==}
def nonpawn() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(8,False{},16,Nil{},(ALeaf{U64.bit(47n)},U64.bit(16n)))
def nonpawn_ep_absent() -> {Count.count(E.moves(nonpawn()),C.Ply{8,16,0,1}) == 0n : Nat}: {==}
def sentinel() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(8,True{},64,Nil{},(ALeaf{U64.bit(47n)},U64.or(U64.bit(16n),U64.bit(32n))))
def sentinel_ep_absent() -> {Count.count(E.moves(sentinel()),C.Ply{8,16,0,1}) == 0n : Nat}: {==}
def promoted() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(48,True{},56,Nil{},(ALeaf{U64.bit(47n)},U64.or(U64.bit(56n),U64.bit(57n))))
def promotion_raw_ep_absent() -> {Count.count(E.moves(promoted()),C.Ply{48,56,1,1}) == 0n : Nat}: {==}
def promotion_present() -> {Count.count(E.moves(promoted()),C.Ply{48,56,4,0}) == 1n : Nat}: {==}
def stale_parent_invalid() -> {Position.valid_ep(16,Prior.stale()) == False{} : Bool}: {==}
def stale_king_loss() -> {C.get_kings(C.make_move(Prior.stale(),C.Ply{8,16,0,1})) == U64.bit(0n) : U64}: {==}
def arbitrary_jump_incoherent() -> {Prior.child(Prior.jump(),C.Ply{32,48,0,0}) == False{} : Bool}: {==}
def stale_partition_valid() -> {Board.valid(Prior.stale()) == True{} : Bool}: {==}
