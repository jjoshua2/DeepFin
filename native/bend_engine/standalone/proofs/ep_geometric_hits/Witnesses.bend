# Concrete effect and excluded-generalization witnesses; imports the public fixtures.
import Base
import ./Fixtures.bend as F
import ./Hits.bend as Hits
import ./Plane.bend as Plane
import ../ep_occupancy_ray/Witnesses.bend as Prior
import ../ep_occupancy_ray/Fixtures.bend as Members
import ../ep_occurrence_tag/Witnesses.bend as Tagged
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../destination_factorization/Spec.bend as Count
import ../../../legal_probe/Chess.bend as C

def white_rook_old() -> {Hits.raw(7n,40,0,F.white(),False{},0,U64.zero()) == U64.bit(47n) : U64}: {==}
def white_rook_new() -> {Hits.raw(7n,40,0,Plane.moved(F.white(),36,43),False{},0,U64.zero()) == U64.zero() : U64}: {==}
def black_rook_old() -> {Hits.raw(7n,16,0,F.black(),False{},1,U64.zero()) == U64.bit(23n) : U64}: {==}
def black_rook_new() -> {Hits.raw(7n,16,0,Plane.moved(F.black(),27,20),False{},1,U64.zero()) == U64.zero() : U64}: {==}
def white_bishop_old() -> {Hits.raw(4n,25,4,F.white(),True{},0,U64.zero()) == U64.bit(61n) : U64}: {==}
def white_bishop_new() -> {Hits.raw(4n,25,4,Plane.moved(F.white(),36,43),True{},0,U64.zero()) == U64.zero() : U64}: {==}
def black_bishop_old() -> {Hits.raw(4n,38,7,F.black(),True{},1,U64.zero()) == U64.bit(2n) : U64}: {==}
def black_bishop_new() -> {Hits.raw(4n,38,7,Plane.moved(F.black(),27,20),True{},1,U64.zero()) == U64.zero() : U64}: {==}
def queen_old() -> {Hits.raw(1n,54,0,F.white(),False{},0,U64.zero()) == U64.bit(55n) : U64}: {==}
def queen_new() -> {Hits.raw(1n,54,0,Plane.moved(F.white(),36,43),False{},0,U64.zero()) == U64.bit(55n) : U64}: {==}
def shared_acc() -> {Hits.raw(7n,40,0,Plane.moved(F.white(),36,43),False{},0,U64.bit(55n)) == U64.bit(55n) : U64}: {==}

# Inconsistent rows are allowed by the low-level erasure identity.
# Pawn priority at source36; overlapping rook bits at source36/victim35/destination43.
def raw() -> C.Board:
  C.Board{U64.bit(36n),U64.zero(),U64.zero(),U64.or(U64.or(U64.or(U64.bit(36n),U64.bit(35n)),U64.bit(43n)),U64.bit(47n)),U64.zero(),U64.zero(),
    U64.or(U64.bit(36n),U64.bit(43n)),U64.or(U64.or(U64.or(U64.bit(36n),U64.bit(35n)),U64.bit(43n)),U64.bit(47n)),7,0,64}
def raw_kind() -> {C.piece(raw(),36) == 0 : U32}: {==}
def raw_rooks() -> {Plane.bits(Plane.moved(raw(),36,43),B.Rook{}) == U64.bit(47n) : U64}: {==}
def raw_black() -> {Plane.plane(Plane.moved(raw(),36,43),B.Rook{},7) == U64.bit(47n) : U64}: {==}
def raw_white() -> {Plane.plane(Plane.moved(raw(),36,43),B.Rook{},1) == U64.zero() : U64}: {==}
def raw_erasure() -> {Plane.plane(Plane.moved(raw(),36,43),B.Rook{},7) == U64.and_not(Plane.plane(raw(),B.Rook{},7),Plane.mask(36,43)) : U64}:
  Plane.colored_erasure(raw(),36,43,B.Rook{},7,raw_kind(),{==})
def missing_destination_false() -> {U64.is_eq(Plane.bits(Plane.moved(raw(),36,43),B.Rook{}),U64.and_not(Plane.bits(raw(),B.Rook{}),U64.or(U64.bit(36n),U64.bit(35n)))) == False{} : Bool}: {==}
def missing_victim_false() -> {U64.is_eq(Plane.bits(Plane.moved(raw(),36,43),B.Rook{}),U64.and_not(Plane.bits(raw(),B.Rook{}),U64.or(U64.bit(36n),U64.bit(43n)))) == False{} : Bool}: {==}
def pawn_put_false() -> {U64.is_eq(Plane.bits(Plane.moved(raw(),36,43),B.Pawn{}),U64.and_not(Plane.bits(raw(),B.Pawn{}),Plane.mask(36,43))) == False{} : Bool}: {==}

def nonpawn_source() -> C.Board:
  C.Board{U64.zero(),U64.zero(),U64.zero(),U64.bit(36n),U64.zero(),U64.zero(),U64.zero(),U64.bit(36n),2,0,64}
def nonpawn_kind() -> {C.piece(nonpawn_source(),36) == 3 : U32}: {==}
def nonpawn_put() -> {Plane.plane(Plane.moved(nonpawn_source(),36,43),B.Rook{},3) == U64.bit(43n) : U64}: {==}
def source_kind_needed() -> {U64.is_eq(Plane.plane(Plane.moved(nonpawn_source(),36,43),B.Rook{},3),U64.and_not(Plane.plane(nonpawn_source(),B.Rook{},3),Plane.mask(36,43))) == False{} : Bool}: {==}
def promotion_put() -> {Plane.plane(C.make_move(F.white(),C.Ply{36,43,3,1}),B.Rook{},1) == U64.bit(43n) : U64}: {==}
def promotion_zero_needed() -> {U64.is_eq(Plane.plane(C.make_move(F.white(),C.Ply{36,43,3,1}),B.Rook{},1),U64.and_not(Plane.plane(F.white(),B.Rook{},1),Plane.mask(36,43))) == False{} : Bool}: {==}
def wrong_side_false() -> {U64.is_eq(Plane.plane(Plane.moved(F.white(),36,43),B.Rook{},0),U64.and_not(Plane.plane(F.white(),B.Rook{},1),Plane.mask(36,43))) == False{} : Bool}: {==}

# Validated parent plus actual arbitrary-table legal occurrence does not imply
# the old victim observation condition. Its colored rook plane stays unchanged,
# but the actual king32-to-rook39 geometric gate grows from zero to bit39.
def discovered_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},Prior.discovered()))): Members.discovered_member()
def discovered_old_hit() -> {Hits.raw(7n,32,0,Prior.discovered(),False{},0,U64.zero()) == U64.zero() : U64}: {==}
def discovered_new_hit() -> {Hits.raw(7n,32,0,Plane.moved(Prior.discovered(),36,43),False{},0,U64.zero()) == U64.bit(39n) : U64}: {==}
def discovered_not_subset() -> {U64.is_eq(U64.and(Hits.raw(7n,32,0,Prior.discovered(),False{},0,U64.zero()),Hits.raw(7n,32,0,Plane.moved(Prior.discovered(),36,43),False{},0,U64.zero())),Hits.raw(7n,32,0,Plane.moved(Prior.discovered(),36,43),False{},0,U64.zero())) == False{} : Bool}: {==}

# Retain exact input storage and duplicate full-Ply tail multiplicity distinctions.
def duplicate_count() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: Tagged.duplicate_count()
def exact_table() -> {Tagged.result_table(Tagged.tagged()) == ALeaf{U64.bit(47n)} : Array<U64>}: Tagged.exact_table()
def wrong_promotion_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,1,1}) == 0n : Nat}: Tagged.wrong_promotion_absent()
def wrong_ep_flag_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: Tagged.wrong_ep_flag_absent()
