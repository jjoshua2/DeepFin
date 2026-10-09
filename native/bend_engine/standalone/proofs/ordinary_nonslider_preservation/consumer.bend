import Base
import ./Prefix.bend as Prefix
import ./Compose.bend as Compose
import ../ordinary_slider_preservation/consumer.bend as Slider
import ../ordinary_slider_preservation/Check.bend as Check
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

# Actual ordinary full-Ply candidate, same table and original side. The old/new
# slider contracts are explicit; fixed non-slider query equality is derived.
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
  +new_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(F.moved(b,src,dst)),False{})) : Array<U64> & U64},
  +new_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(F.moved(b,src,dst)),False{})) : Array<U64> & U64},
  +unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.in_check(O.pack(c),F.moved(b,src,dst),Chess.get_turn(b)) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  (source_cert,raw_target_clear) = certified
  (raw_bound,raw_owned) = source_cert
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool} = raw_bound
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool} = raw_owned
  +target_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64} = raw_target_clear
  +q = K.index(b)
  +side = Chess.get_turn(b)
  +by = U32.xor(side,1)
  +post = F.moved(b,src,dst)
  +sq = U64.ctz(F.plane(post,side))
  +chosen: {Q.king(b) == U32.from_nat(q) : U32} = K.chosen(b,nonempty)
  +selected: {sq == U32.from_nat(q) : U32} = Equal.trans(U32,sq,Q.king(b),U32.from_nat(q),
    Equal.cong(U64,U32,U64.ctz,F.plane(post,side),F.plane(b,side),
      King.from_certificate(b,Prepared.rays(c,b),src,dst,((bound,owned),target_clear),good,bypass)),chosen)
  +initial: {Check.hits(c,b,q,by) == U64.zero() : U64} = Check.initial_clear(c,b,q,chosen,old_r,old_d,unchecked)
  +non_slider: {Prefix.hits(c,post,q,by) == U64.zero() : U64} = Prefix.clear(c,b,q,src,dst,white,canonical,initial)
  +zero_selected = Slider.from_certificate(c,b,src,dst,white,((bound,owned),target_clear),good,nonempty,canonical,bypass,old_r,old_d,new_r,new_d,unchecked)
  +zero_q = Equal.trans(Array<U64> & Bool,
    Chess.attacked_rook(post,U32.from_nat(q),by,U64.zero(),Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(post))),
    Chess.attacked_rook(post,sq,by,U64.zero(),Chess.attack(3,O.pack(c),sq,by,Chess.occupied(post))),(O.pack(c),False{}),
    Equal.cong(U32,Array<U64> & Bool,x => Chess.attacked_rook(post,x,by,U64.zero(),Chess.attack(3,O.pack(c),x,by,Chess.occupied(post))),
      U32.from_nat(q),sq,Equal.sym(U32,sq,U32.from_nat(q),selected)),zero_selected)
  Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),post,side),
    Chess.attacked(O.pack(c),post,U32.from_nat(q),by),(O.pack(c),False{}),
    Equal.cong(U32,Array<U64> & Bool,x => Chess.attacked(O.pack(c),post,x,by),sq,U32.from_nat(q),selected),
    Compose.attacked_clear(c,post,q,by,new_r,new_d,non_slider,zero_q))

def use_no_check(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(O.pack(c),b))),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +new_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(F.moved(b,src,dst)),False{})) : Array<U64> & U64},
  +new_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(F.moved(b,src,dst)),False{})) : Array<U64> & U64},
  +unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.in_check(O.pack(c),F.moved(b,src,dst),Chess.get_turn(b)) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  from_certificate(c,b,src,dst,white,Provenance.member(O.pack(c),b,Chess.Ply{src,dst,0,0},good,here),
    good,nonempty,canonical,bypass,old_r,old_d,new_r,new_d,unchecked)

# Arbitrary 512-cell table: friendly raw hits remain nonzero after the move.
# The opponent has every piece kind; no builder or target-geometry claim.
def zero_cells(n: Nat) -> O.Cells:
  match n:
    case 0n: O.Cell{U64.zero()}
    case 1n+p:
      +child = zero_cells(p)
      O.Branch{child,child}

def fixture_board() -> Chess.Board:
  Chess.Board{U64.or(U64.bit(51n),U64.bit(37n)),U64.or(U64.bit(19n),U64.bit(38n)),
    U64.bit(42n),U64.bit(41n),U64.bit(43n),U64.or(U64.bit(0n),U64.bit(39n)),
    U64.or(U64.or(U64.bit(0n),U64.bit(19n)),U64.bit(51n)),
    U64.or(U64.or(U64.or(U64.bit(37n),U64.bit(38n)),U64.or(U64.bit(39n),U64.bit(41n))),U64.or(U64.bit(42n),U64.bit(43n))),1,0,64}

def fixture_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{zero_cells(4n),O.Branch{zero_cells(3n),O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},U64.zero(),False{})},O.Cell{Tables.slider(0,True{},U64.zero(),False{})}}}}}},zero_cells(5n)},zero_cells(6n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(31)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(19n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},O.Branch{O.Branch{O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.bit(34n)}}},zero_cells(2n)},zero_cells(3n)}},O.Branch{O.Branch{O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.bit(34n)}}}},zero_cells(3n)},zero_cells(4n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(0n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(51n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(37n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}}

def fixture_valid() -> {B.valid(fixture_board()) == True{} : Bool}: {==}
def fixture_nonempty() -> {U64.is_zero(K.plane(fixture_board())) == False{} : Bool}: {==}
def fixture_member() -> E.member(Chess.Ply{19,34,0,0},E.moves(C.candidates(O.pack(fixture_cells()),fixture_board()))): Inl{{==}}
def fixture_bypass() ->
  {Chess.filter_requires(Sensitive.sensitive(fixture_board(),Prepared.rays(fixture_cells(),fixture_board())),Chess.Ply{19,34,0,0}) == False{} : Bool}: {==}
def fixture_initial() ->
  {Chess.in_check(O.pack(fixture_cells()),fixture_board(),1) == (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}: {==}
def fixture_old_rook() -> Q.rook(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def fixture_old_bishop() -> Q.bishop(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def fixture_new_rook() -> Q.rook(fixture_cells(),0n,0,Chess.occupied(F.moved(fixture_board(),19,34))): {==}
def fixture_new_bishop() -> Q.bishop(fixture_cells(),0n,0,Chess.occupied(F.moved(fixture_board(),19,34))): {==}
def fixture_raw_prefix_nonempty() ->
  {U64.is_zero(Check.prefix(fixture_cells(),F.moved(fixture_board(),19,34),0,0)) == False{} : Bool}: {==}
def fixture_masked_prefix_empty() ->
  {Prefix.hits(fixture_cells(),F.moved(fixture_board(),19,34),0n,0) == U64.zero() : U64}: {==}
def fixture_safe() ->
  {Chess.in_check(O.pack(fixture_cells()),F.moved(fixture_board(),19,34),1) ==
    (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}:
  use_no_check(fixture_cells(),fixture_board(),19,34,True{},fixture_member(),fixture_valid(),fixture_nonempty(),{==},fixture_bypass(),
    fixture_old_rook(),fixture_old_bishop(),fixture_new_rook(),fixture_new_bishop(),fixture_initial())

# Turn flips to black, whose king is now attacked by the moved white knight.
def wrong_side_check() ->
  {Chess.in_check(O.pack(fixture_cells()),F.moved(fixture_board(),19,34),Chess.get_turn(F.moved(fixture_board(),19,34))) ==
    (O.pack(fixture_cells()),True{}) : Array<U64> & Bool}: {==}

# Pawn lookup must reverse the attacking side. The wrong orientation hits black.
def wrong_pawn_orientation() ->
  {Chess.attacked_pawn(F.moved(fixture_board(),19,34),0,0,
      Chess.attack(0,O.pack(fixture_cells()),0,0,Chess.occupied(F.moved(fixture_board(),19,34)))) ==
    (O.pack(fixture_cells()),True{}) : Array<U64> & Bool}: {==}
