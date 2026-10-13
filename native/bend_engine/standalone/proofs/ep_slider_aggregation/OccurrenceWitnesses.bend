# Exact legal occurrence, full-Ply counts and input-table identity.
import Base
import ./Counterexample.bend as Counter
import ../ep_occupancy_ray/Fixtures.bend as Members
import ../ep_occurrence_tag/Witnesses.bend as Tagged
import ../destination_factorization/Spec.bend as Count
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C

def discovered_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},Counter.discovered()))): Members.discovered_member()

def duplicate_count() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: Tagged.duplicate_count()
def exact_table() -> {Tagged.result_table(Tagged.tagged()) == ALeaf{U64.bit(47n)} : Array<U64>}: Tagged.exact_table()
def wrong_promotion_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,1,1}) == 0n : Nat}: Tagged.wrong_promotion_absent()
def wrong_ep_flag_absent() -> {Count.count(E.moves(Tagged.tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: Tagged.wrong_ep_flag_absent()
