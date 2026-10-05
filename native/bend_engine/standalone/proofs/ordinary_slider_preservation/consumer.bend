import Base
import ./Algebra.bend as A
import ./Rays.bend as Rays
import ./Check.bend as Check
import ./Hits.bend as Hits
import ../candidate_king_destination/Provenance.bend as Provenance
import ../candidate_king_destination/Emission.bend as Destination
import ../candidate_king_destination/consumer.bend as King
import ../candidate_source_provenance/Source.bend as Source
import ../ordinary_king_frame/Frame.bend as F
import ../board/Spec.bend as B
import ../castle_safety/Spec.bend as Sensitive
import ../accepted_castle/Spec.bend as Prepared
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../prepared_king_selection/Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

# One actual candidate occurrence supplies source ownership/index and destination
# king exclusion. Table identity is shared by generation, preparation and queries.
# This result is the actual slider suffix; the non-slider accumulator is zero.
def from_certificate(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  certified: Source.source(b,Chess.Ply{src,dst,0,0}) & Destination.clear(b,Chess.Ply{src,dst,0,0}),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  new_r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  new_d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.attacked_rook(F.moved(b,src,dst),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),
      U32.xor(Chess.get_turn(b),1),U64.zero(),
      Chess.attack(3,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),
        U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  (source_cert,target_clear) = certified
  (raw_bound,raw_owned) = source_cert
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool} = raw_bound
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool} = raw_owned
  +q = K.index(b)
  +qb: {Nat.is_lt(q,64n) == True{} : Bool} = K.bound(b,nonempty)
  +chosen: {Q.king(b) == U32.from_nat(q) : U32} = K.chosen(b,nonempty)
  +selected = Equal.trans(U32,U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),Q.king(b),U32.from_nat(q),
    Equal.cong(U64,U32,U64.ctz,F.plane(F.moved(b,src,dst),Chess.get_turn(b)),F.plane(b,Chess.get_turn(b)),
      King.from_certificate(b,Prepared.rays(c,b),src,dst,((bound,owned),target_clear),good,bypass)),chosen)
  +initial: {Check.hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64} =
    Check.initial_clear(c,b,q,chosen,old_r,old_d,unchecked)
  +rook_clear = A.transfer(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),
    Tables.slider(U32.from_nat(q),False{},Chess.occupied(F.moved(b,src,dst)),False{}),
    Check.enemies(b,Chess.get_turn(b),B.Rook{}),Check.enemies(F.moved(b,src,dst),Chess.get_turn(b),B.Rook{}),
    Rays.rook(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d),
    Hits.enemies_subset(b,src,dst,B.Rook{},white,canonical),Check.rook_clear(c,b,q,initial))
  +bishop_clear = A.transfer(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),
    Tables.slider(U32.from_nat(q),True{},Chess.occupied(F.moved(b,src,dst)),False{}),
    Check.enemies(b,Chess.get_turn(b),B.Bishop{}),Check.enemies(F.moved(b,src,dst),Chess.get_turn(b),B.Bishop{}),
    Rays.bishop(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d),
    Hits.enemies_subset(b,src,dst,B.Bishop{},white,canonical),Check.bishop_clear(c,b,q,initial))
  Equal.trans(Array<U64> & Bool,
    Chess.attacked_rook(F.moved(b,src,dst),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),U64.zero(),
      Chess.attack(3,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))),
    Chess.attacked_rook(F.moved(b,src,dst),U32.from_nat(q),U32.xor(Chess.get_turn(b),1),U64.zero(),
      Chess.attack(3,O.pack(c),U32.from_nat(q),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))),(O.pack(c),False{}),
    Equal.cong(U32,Array<U64> & Bool,sq => Chess.attacked_rook(F.moved(b,src,dst),sq,U32.xor(Chess.get_turn(b),1),U64.zero(),
      Chess.attack(3,O.pack(c),sq,U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))),
      U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.from_nat(q),selected),
    Hits.safe(c,F.moved(b,src,dst),q,Chess.get_turn(b),new_r,new_d,
      Hits.clear(F.moved(b,src,dst),q,Chess.get_turn(b),rook_clear,bishop_clear)))

def use_slider(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(O.pack(c),b))),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  new_r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  new_d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.attacked_rook(F.moved(b,src,dst),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),
      U32.xor(Chess.get_turn(b),1),U64.zero(),
      Chess.attack(3,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),
        U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  from_certificate(c,b,src,dst,white,Provenance.member(O.pack(c),b,Chess.Ply{src,dst,0,0},good,here),
    good,nonempty,canonical,bypass,old_r,old_d,new_r,new_d,unchecked)

# A small arbitrary table satisfies exactly the required query contracts. It
# contains an actual knight candidate and a nonempty opponent rook plane.
def zero_cells(n: Nat) -> O.Cells:
  match n:
    case 0n: O.Cell{U64.zero()}
    case 1n+p:
      +child = zero_cells(p)
      O.Branch{child,child}

def fixture_board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.bit(19n),U64.zero(),U64.bit(37n),U64.zero(),U64.bit(0n),
    U64.or(U64.bit(0n),U64.bit(19n)),U64.bit(37n),1,0,64}

def fixture_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{zero_cells(4n),O.Branch{O.Branch{O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.bit(34n)}}},zero_cells(2n)},O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},U64.zero(),False{})},O.Cell{Tables.slider(0,True{},U64.zero(),False{})}}}}}},zero_cells(5n)},zero_cells(6n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(31)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}

def fixture_valid() -> {B.valid(fixture_board()) == True{} : Bool}: {==}
def fixture_nonempty() -> {U64.is_zero(K.plane(fixture_board())) == False{} : Bool}: {==}
def fixture_member() -> E.member(Chess.Ply{19,34,0,0},E.moves(C.candidates(O.pack(fixture_cells()),fixture_board()))): Inl{{==}}
def fixture_bypass() ->
  {Chess.filter_requires(Sensitive.sensitive(fixture_board(),Prepared.rays(fixture_cells(),fixture_board())),Chess.Ply{19,34,0,0}) == False{} : Bool}: {==}
def fixture_no_check() ->
  {Chess.in_check(O.pack(fixture_cells()),fixture_board(),1) == (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}: {==}
def fixture_old_rook() -> Q.rook(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def fixture_old_bishop() -> Q.bishop(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def fixture_new_rook() -> Q.rook(fixture_cells(),0n,0,Chess.occupied(F.moved(fixture_board(),19,34))): {==}
def fixture_new_bishop() -> Q.bishop(fixture_cells(),0n,0,Chess.occupied(F.moved(fixture_board(),19,34))): {==}
def fixture_safe() ->
  {Chess.attacked_rook(F.moved(fixture_board(),19,34),0,0,U64.zero(),
      Chess.attack(3,O.pack(fixture_cells()),0,0,Chess.occupied(F.moved(fixture_board(),19,34)))) ==
    (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}:
  use_slider(fixture_cells(),fixture_board(),19,34,True{},fixture_member(),fixture_valid(),fixture_nonempty(),{==},fixture_bypass(),
    fixture_old_rook(),fixture_old_bishop(),fixture_new_rook(),fixture_new_bishop(),fixture_no_check())

def bad_post_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(34n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},O.Branch{O.Branch{O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.bit(34n)}}},zero_cells(2n)},O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},U64.zero(),False{})},O.Cell{U64.bit(37n)}}}}}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{Tables.slider(0,True{},U64.zero(),False{})},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)}},zero_cells(6n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(32)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}

def bad_post_member() -> E.member(Chess.Ply{19,34,0,0},E.moves(C.candidates(O.pack(bad_post_cells()),fixture_board()))): Inl{{==}}
def bad_post_old_rook() -> Q.rook(bad_post_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def bad_post_old_bishop() -> Q.bishop(bad_post_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def bad_post_new_bishop() -> Q.bishop(bad_post_cells(),0n,0,Chess.occupied(F.moved(fixture_board(),19,34))): {==}
def bad_post_bypass() ->
  {Chess.filter_requires(Sensitive.sensitive(fixture_board(),Prepared.rays(bad_post_cells(),fixture_board())),Chess.Ply{19,34,0,0}) == False{} : Bool}: {==}
def bad_post_initial() ->
  {Chess.in_check(O.pack(bad_post_cells()),fixture_board(),1) == (O.pack(bad_post_cells()),False{}) : Array<U64> & Bool}: {==}
def bad_post_attacks() ->
  {Chess.attacked_rook(F.moved(fixture_board(),19,34),0,0,U64.zero(),
      Chess.attack(3,O.pack(bad_post_cells()),0,0,Chess.occupied(F.moved(fixture_board(),19,34)))) ==
    (O.pack(bad_post_cells()),True{}) : Array<U64> & Bool}: {==}

def blocker_board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.bit(8n),U64.zero(),U64.bit(16n),U64.zero(),U64.bit(0n),
    U64.or(U64.bit(0n),U64.bit(8n)),U64.bit(16n),1,0,64}

def blocker_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(8n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(18n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)}},O.Branch{zero_cells(3n),O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},Chess.occupied(F.moved(blocker_board(),8,18)),False{})},O.Cell{Tables.slider(0,False{},Chess.occupied(blocker_board()),False{})}}}}}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{Tables.slider(0,True{},Chess.occupied(blocker_board()),False{})},O.Cell{Tables.slider(0,True{},Chess.occupied(F.moved(blocker_board(),8,18)),False{})}},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(18n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(32)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}

def blocker_valid() -> {B.valid(blocker_board()) == True{} : Bool}: {==}
def blocker_member() -> E.member(Chess.Ply{8,18,0,0},E.moves(C.candidates(O.pack(blocker_cells()),blocker_board()))): Inr{Inl{{==}}}
def blocker_old_rook() -> Q.rook(blocker_cells(),0n,1,Chess.occupied(blocker_board())): {==}
def blocker_old_bishop() -> Q.bishop(blocker_cells(),0n,1,Chess.occupied(blocker_board())): {==}
def blocker_new_rook() -> Q.rook(blocker_cells(),0n,0,Chess.occupied(F.moved(blocker_board(),8,18))): {==}
def blocker_new_bishop() -> Q.bishop(blocker_cells(),0n,0,Chess.occupied(F.moved(blocker_board(),8,18))): {==}
def blocker_required() ->
  {Chess.filter_requires(Sensitive.sensitive(blocker_board(),Prepared.rays(blocker_cells(),blocker_board())),Chess.Ply{8,18,0,0}) == True{} : Bool}: {==}
def blocker_initial() ->
  {Chess.in_check(O.pack(blocker_cells()),blocker_board(),1) == (O.pack(blocker_cells()),False{}) : Array<U64> & Bool}: {==}
def blocker_attacks() ->
  {Chess.attacked_rook(F.moved(blocker_board(),8,18),0,0,U64.zero(),
      Chess.attack(3,O.pack(blocker_cells()),0,0,Chess.occupied(F.moved(blocker_board(),8,18)))) ==
    (O.pack(blocker_cells()),True{}) : Array<U64> & Bool}: {==}

def noncanonical_board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.bit(19n),U64.zero(),U64.bit(0n),
    U64.zero(),U64.or(U64.bit(0n),U64.bit(19n)),2,0,64}

def noncanonical_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(3n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},O.Branch{zero_cells(3n),O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},Chess.occupied(noncanonical_board()),False{})},O.Cell{Tables.slider(0,False{},Chess.occupied(F.moved(noncanonical_board(),19,3)),False{})}}}}}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{Tables.slider(0,True{},Chess.occupied(noncanonical_board()),False{})},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},O.Branch{O.Branch{O.Branch{zero_cells(1n),O.Branch{O.Cell{U64.bit(3n)},zero_cells(0n)}},zero_cells(2n)},zero_cells(3n)}}},zero_cells(6n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},O.Branch{O.Branch{O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.from_u32(50)}}},zero_cells(2n)},zero_cells(3n)}},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(32)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}

def noncanonical_valid() -> {B.valid(noncanonical_board()) == True{} : Bool}: {==}
def noncanonical_member() -> E.member(Chess.Ply{19,3,0,0},E.moves(C.candidates(O.pack(noncanonical_cells()),noncanonical_board()))): Inl{{==}}
def noncanonical_old_rook() -> Q.rook(noncanonical_cells(),0n,2,Chess.occupied(noncanonical_board())): {==}
def noncanonical_old_bishop() -> Q.bishop(noncanonical_cells(),0n,2,Chess.occupied(noncanonical_board())): {==}
def noncanonical_new_rook() -> Q.rook(noncanonical_cells(),0n,3,Chess.occupied(F.moved(noncanonical_board(),19,3))): {==}
def noncanonical_new_bishop() -> Q.bishop(noncanonical_cells(),0n,3,Chess.occupied(F.moved(noncanonical_board(),19,3))): {==}
def noncanonical_bypass() ->
  {Chess.filter_requires(Sensitive.sensitive(noncanonical_board(),Prepared.rays(noncanonical_cells(),noncanonical_board())),Chess.Ply{19,3,0,0}) == False{} : Bool}: {==}
def noncanonical_initial() ->
  {Chess.in_check(O.pack(noncanonical_cells()),noncanonical_board(),2) == (O.pack(noncanonical_cells()),False{}) : Array<U64> & Bool}: {==}
def noncanonical_attacks() ->
  {Chess.attacked_rook(F.moved(noncanonical_board(),19,3),0,3,U64.zero(),
      Chess.attack(3,O.pack(noncanonical_cells()),0,3,Chess.occupied(F.moved(noncanonical_board(),19,3)))) ==
    (O.pack(noncanonical_cells()),True{}) : Array<U64> & Bool}: {==}

# Bypass does not imply promotion=0: an actual back-rank pawn promotion bypasses.
def promotion_board() -> Chess.Board:
  Chess.Board{U64.bit(51n),U64.zero(),U64.zero(),U64.bit(37n),U64.zero(),U64.bit(0n),
    U64.or(U64.bit(0n),U64.bit(51n)),U64.bit(37n),1,0,64}
def promotion_valid() -> {B.valid(promotion_board()) == True{} : Bool}: {==}
def promotion_member() -> E.member(Chess.Ply{51,59,1,0},E.moves(C.candidates(O.pack(fixture_cells()),promotion_board()))): Inl{{==}}
def promotion_bypass() ->
  {Chess.filter_requires(Sensitive.sensitive(promotion_board(),Prepared.rays(fixture_cells(),promotion_board())),Chess.Ply{51,59,1,0}) == False{} : Bool}: {==}
def fixture_output() ->
  {Chess.attacked_rook(F.moved(fixture_board(),19,34),0,0,U64.zero(),
      Chess.attack(3,O.pack(fixture_cells()),0,0,Chess.occupied(F.moved(fixture_board(),19,34)))) ==
    (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}: {==}
