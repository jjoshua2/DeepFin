# Closed actual updates and the missing-victim-observation boundary.
import Base
import ./EP.bend as EP
import ../ep_king_frame/Witnesses.bend as Parent
import ../ep_occurrence_tag/Witnesses.bend as Tagged
import ../child_ep_coherence/Witnesses.bend as Prior
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../destination_factorization/Spec.bend as Count
import ../ray/Path.bend as Path
import ../../Position.bend as Position
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def white_path() -> List<&2,Nat>: 41n <> 42n <> 43n <> 44n <> 45n <> 46n <> 47n <> Nil{}
def black_path() -> List<&2,Nat>: 17n <> 18n <> 19n <> 20n <> 21n <> 22n <> 23n <> Nil{}
def white_old_ray() -> {Tables.ray(7n,False{},40,0,C.occupied(Parent.white()),False{},U64.zero()) == U64{U32.from_nat(0n),U32.from_nat(65024n)} : U64}: {==}
def white_new_ray() -> {Tables.ray(7n,False{},40,0,C.occupied(C.make_move(Parent.white(),C.Ply{36,43,0,1})),False{},U64.zero()) == U64{U32.from_nat(0n),U32.from_nat(3584n)} : U64}: {==}
def black_old_ray() -> {Tables.ray(7n,False{},16,0,C.occupied(Parent.black()),False{},U64.zero()) == U64.from_u32(16646144) : U64}: {==}
def black_new_ray() -> {Tables.ray(7n,False{},16,0,C.occupied(C.make_move(Parent.black(),C.Ply{27,20,0,1})),False{},U64.zero()) == U64.from_u32(1966080) : U64}: {==}

# Deliberately inconsistent piece/color rows and arbitrary turn still satisfy total occupancy.
def raw(+white: U64,+black: U64,turn: U32) -> C.Board:
  C.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),white,black,turn,0,64}
def raw_ep() -> {C.occupied(C.make_move(raw(U64.from_u32(2),U64.from_u32(4),7),C.Ply{1,10,0,1})) == U64.from_u32(1024) : U64}: {==}
def same_square() -> {C.occupied(C.make_move(raw(U64.from_u32(6),U64.from_u32(6),7),C.Ply{1,1,0,1})) == U64.from_u32(6) : U64}: {==}
def absent_source() -> {C.occupied(C.make_move(raw(U64.zero(),U64.zero(),9),C.Ply{4,10,0,1})) == U64.from_u32(1024) : U64}: {==}
def out_of_range() -> {C.occupied(C.make_move(raw(U64.from_u32(6),U64.zero(),7),C.Ply{64,65,0,1})) == U64.from_u32(6) : U64}: {==}
def same_source_victim() -> {C.occupied(C.make_move(raw(U64.from_u32(6),U64.zero(),7),C.Ply{2,10,0,1})) == U64.from_u32(1026) : U64}: {==}
def castle_not_ep() -> {U64.is_eq(C.occupied(C.make_move(raw(U64.from_u32(144),U64.zero(),1),C.Ply{4,6,0,2})),EP.normalized(raw(U64.from_u32(144),U64.zero(),1),4,6)) == False{} : Bool}: {==}

# Source36 is beyond the old first blocker35. Capturing35 opens the king32-to-rook39 ray.
def discovered() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.bit(39n),U64.zero(),U64.or(U64.bit(32n),U64.bit(63n)),
    U64.or(U64.bit(32n),U64.bit(36n)),U64.or(U64.or(U64.bit(63n),U64.bit(35n)),U64.bit(39n)),1,0,43}
def discovered_path() -> List<&2,Nat>: 33n <> 34n <> 35n <> 36n <> 37n <> 38n <> 39n <> Nil{}
def discovered_partition() -> {Board.valid(discovered()) == True{} : Bool}: {==}
def discovered_coherent() -> {Position.valid_ep(43,discovered()) == True{} : Bool}: {==}
def discovered_position_valid() -> {Position.valid(discovered()) == True{} : Bool}: {==}
def source_only_clear() -> {U64.and(U64.bit(36n),Path.attack(discovered_path(),C.occupied(discovered()))) == U64.zero() : U64}: {==}
def victim_observed() -> {U64.and(EP.removed(36,43),Path.attack(discovered_path(),C.occupied(discovered()))) == U64.bit(35n) : U64}: {==}
def discovered_old_ray() -> {Tables.ray(7n,False{},32,0,C.occupied(discovered()),False{},U64.zero()) == U64{U32.from_nat(0n),U32.from_nat(14n)} : U64}: {==}
def discovered_new_ray() -> {Tables.ray(7n,False{},32,0,C.occupied(C.make_move(discovered(),C.Ply{36,43,0,1})),False{},U64.zero()) == U64{U32.from_nat(0n),U32.from_nat(254n)} : U64}: {==}
def discovered_not_subset() -> {U64.is_eq(U64.and(Tables.ray(7n,False{},32,0,C.occupied(discovered()),False{},U64.zero()),Tables.ray(7n,False{},32,0,C.occupied(C.make_move(discovered(),C.Ply{36,43,0,1})),False{},U64.zero())),Tables.ray(7n,False{},32,0,C.occupied(C.make_move(discovered(),C.Ply{36,43,0,1})),False{},U64.zero())) == False{} : Bool}: {==}

# Existing excluded cases are retained, with their exact distinctions.
def stale_incoherent() -> {Position.valid_ep(16,Prior.stale()) == False{} : Bool}: Parent.stale_incoherent()
def stale_king_loss() -> {C.get_kings(C.make_move(Prior.stale(),C.Ply{8,16,0,1})) == U64.bit(0n) : U64}: Parent.stale_king_loss()
def overlap_invalid() -> {Board.valid(Parent.overlap()) == False{} : Bool}: Parent.overlap_invalid()
def overlap_loss() -> {C.get_kings(C.make_move(Parent.overlap(),C.Ply{36,43,0,1})) == U64.or(U64.bit(0n),U64.bit(63n)) : U64}: Parent.overlap_loss()
def multiple_root_invalid() -> {Position.valid(Parent.multiple()) == False{} : Bool}: Parent.multiple_root_invalid()
def arbitrary_jump_incoherent() -> {Prior.child(Prior.jump(),C.Ply{32,48,0,0}) == False{} : Bool}: Parent.arbitrary_jump_incoherent()
def duplicate_count() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: Tagged.duplicate_count()
def exact_table() -> {Tagged.result_table(Tagged.tagged()) == ALeaf{U64.bit(47n)} : Array<U64>}: Tagged.exact_table()

def shared_acc() -> {Tables.ray(7n,False{},40,0,C.occupied(C.make_move(Parent.white(),C.Ply{36,43,0,1})),False{},U64.bit(55n)) == U64.or(U64.bit(55n),U64{U32.from_nat(0n),U32.from_nat(3584n)}) : U64}: {==}
def wrong_promotion_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,1,1}) == 0n : Nat}: Tagged.wrong_promotion_absent()
def wrong_ep_flag_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: Tagged.wrong_ep_flag_absent()
