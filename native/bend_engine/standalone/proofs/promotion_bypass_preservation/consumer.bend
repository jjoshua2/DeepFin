import Base
import ./Raw.bend as Raw
import ./Enemy.bend as Enemy
import ./Compare.bend as Compare
import ../ordinary_slider_preservation/Check.bend as Check
import ../ordinary_nonslider_preservation/Prefix.bend as Prefix
import ../destination_factorization/Spec.bend as Counts
import ../promotion/Spec.bend as P
import ../ordinary_nonslider_preservation/consumer.bend as Ordinary
import ../candidate_king_destination/Provenance.bend as Provenance
import ../candidate_king_destination/Emission.bend as Destination
import ../candidate_source_provenance/Source.bend as Source
import ../ordinary_king_frame/Source.bend as SourceFacts
import ../filter_ray_exclusion/Bits.bend as Bits
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

# Actual typed promotion occurrence supplies these certificates. The ordinary
# shadow is a proof comparison Board, with no ordinary-shadow membership premise.
def from_certificate(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  certified: Source.source(b,P.ply(src,dst,choice)) & Destination.clear(b,P.ply(src,dst,choice)),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),P.ply(src,dst,choice)) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +new_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(Raw.moved(b,src,dst,choice)),False{})) : Array<U64> & U64},
  +new_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(Raw.moved(b,src,dst,choice)),False{})) : Array<U64> & U64},
  +unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.in_check(O.pack(c),Raw.moved(b,src,dst,choice),Chess.get_turn(b)) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  (source_cert,raw_target_clear)=certified
  (raw_bound,raw_owned)=source_cert
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool}=raw_bound
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool}=raw_owned
  +target_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64}=raw_target_clear
  +q=K.index(b)
  +side=Chess.get_turn(b)
  +by=U32.xor(side,1)
  +plain=F.moved(b,src,dst)
  +post=Raw.moved(b,src,dst,choice)
  +bypass_plain: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool}=
    Equal.trans(Bool,Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}),
      Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),P.ply(src,dst,choice)),False{},
      Equal.sym(Bool,Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),P.ply(src,dst,choice)),
        Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}),Raw.bypass(b,Prepared.rays(c,b),src,dst,choice)),bypass)
  +clear_bit: {U64.test_bit(Chess.get_kings(b),U32.to_nat(src)) == False{} : Bool}=
    SourceFacts.king_clear(b,Prepared.rays(c,b),src,dst,bound,owned,bypass_plain)
  +source_clear: {U64.and(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) == U64.zero() : U64}=
    Bits.bit_zero(Chess.get_kings(b),U32.to_nat(src),bound,clear_bit)
  +not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}=
    SourceFacts.nonking(b,src,bound,good,owned,clear_bit)
  +selected_plain: {U64.ctz(F.plane(plain,side)) == U32.from_nat(q) : U32}=
    Equal.trans(U32,U64.ctz(F.plane(plain,side)),U64.ctz(F.plane(b,side)),U32.from_nat(q),
      F.selected_keep(b,src,dst,source_clear,target_clear,not_king),K.chosen(b,nonempty))
  +selected_post: {U64.ctz(F.plane(post,side)) == U32.from_nat(q) : U32}=
    Equal.trans(U32,U64.ctz(F.plane(post,side)),U64.ctz(F.plane(plain,side)),U32.from_nat(q),
      Equal.cong(U64,U32,U64.ctz,F.plane(post,side),F.plane(plain,side),Raw.plane_equal(b,src,dst,choice,side,not_king)),selected_plain)
  +r_plain: {Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(plain)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(plain),False{})) : Array<U64> & U64}=
    Compare.lookup_shadow(c,b,src,dst,choice,q,by,3,False{},new_r)
  +d_plain: {Chess.attack(2,O.pack(c),U32.from_nat(q),by,Chess.occupied(plain)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(plain),False{})) : Array<U64> & U64}=
    Compare.lookup_shadow(c,b,src,dst,choice,q,by,2,True{},new_d)
  +safe_plain: {Chess.in_check(O.pack(c),plain,side) == (O.pack(c),False{}) : Array<U64> & Bool}=
    Ordinary.from_certificate(c,b,src,dst,white,((bound,owned),target_clear),good,nonempty,canonical,bypass_plain,old_r,old_d,r_plain,d_plain,unchecked)
  Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),post,side),Chess.in_check(O.pack(c),plain,side),(O.pack(c),False{}),
    Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),post,side),Chess.attacked(O.pack(c),post,U32.from_nat(q),by),
      Chess.in_check(O.pack(c),plain,side),
      Equal.cong(U32,Array<U64> & Bool,sq => Chess.attacked(O.pack(c),post,sq,by),U64.ctz(F.plane(post,side)),U32.from_nat(q),selected_post),
      Equal.trans(Array<U64> & Bool,Chess.attacked(O.pack(c),post,U32.from_nat(q),by),Chess.attacked(O.pack(c),plain,U32.from_nat(q),by),
        Chess.in_check(O.pack(c),plain,side),Compare.attacked(c,b,q,src,dst,choice,white,canonical,new_r,new_d,r_plain,d_plain),
        Equal.sym(Array<U64> & Bool,Chess.in_check(O.pack(c),plain,side),Chess.attacked(O.pack(c),plain,U32.from_nat(q),by),
          Equal.cong(U32,Array<U64> & Bool,sq => Chess.attacked(O.pack(c),plain,sq,by),U64.ctz(F.plane(plain,side)),U32.from_nat(q),selected_plain)))),safe_plain)

def use_no_check(+c: O.Cells,+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  here: E.member(P.ply(src,dst,choice),E.moves(C.candidates(O.pack(c),b))),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),P.ply(src,dst,choice)) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +new_r: {Chess.attack(3,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),False{},Chess.occupied(Raw.moved(b,src,dst,choice)),False{})) : Array<U64> & U64},
  +new_d: {Chess.attack(2,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(Raw.moved(b,src,dst,choice))) ==
    (O.pack(c),Tables.slider(U32.from_nat(K.index(b)),True{},Chess.occupied(Raw.moved(b,src,dst,choice)),False{})) : Array<U64> & U64},
  +unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.in_check(O.pack(c),Raw.moved(b,src,dst,choice),Chess.get_turn(b)) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  from_certificate(c,b,src,dst,choice,white,Provenance.member(O.pack(c),b,P.ply(src,dst,choice),good,here),
    good,nonempty,canonical,bypass,old_r,old_d,new_r,new_d,unchecked)

# Actual scan emits only four promotions; the ordinary shadow is absent.
# The table is arbitrary, and contains deliberately different friendly raw hits.
def zero_cells(n: Nat) -> O.Cells:
  match n:
    case 0n: O.Cell{U64.zero()}
    case 1n+p:
      +child=zero_cells(p)
      O.Branch{child,child}

def fixture_board() -> Chess.Board:
  Chess.Board{U64.or(U64.bit(49n),U64.bit(37n)),U64.bit(38n),U64.bit(42n),U64.bit(41n),U64.bit(43n),
    U64.or(U64.bit(0n),U64.bit(39n)),U64.or(U64.bit(0n),U64.bit(49n)),
    U64.or(U64.or(U64.or(U64.bit(37n),U64.bit(38n)),U64.or(U64.bit(39n),U64.bit(41n))),U64.or(U64.bit(42n),U64.bit(43n))),1,0,64}

def fixture_cells() -> O.Cells:
  O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{zero_cells(4n),O.Branch{zero_cells(3n),O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{O.Cell{Tables.slider(0,False{},U64.zero(),False{})},O.Cell{Tables.slider(0,True{},U64.zero(),False{})}}}}}},zero_cells(5n)},zero_cells(6n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(30)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.from_u32(31)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(57n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},O.Branch{O.Branch{O.Branch{zero_cells(2n),O.Branch{zero_cells(1n),O.Branch{zero_cells(0n),O.Cell{U64.bit(57n)}}}},zero_cells(3n)},zero_cells(4n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(0n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(49n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)},O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Branch{O.Cell{U64.bit(37n)},zero_cells(0n)},zero_cells(1n)},zero_cells(2n)},zero_cells(3n)},zero_cells(4n)},zero_cells(5n)}}}}

def fixture_valid() -> {B.valid(fixture_board()) == True{} : Bool}: {==}
def fixture_nonempty() -> {U64.is_zero(K.plane(fixture_board())) == False{} : Bool}: {==}

def fixture_candidates() ->
  {C.candidates(O.pack(fixture_cells()),fixture_board()) == (O.pack(fixture_cells()),P.choices(49,57)) : Array<U64> & List<&2,Chess.Ply>}: {==}

def no_shadow_candidate() ->
  {Counts.count(E.moves(C.candidates(O.pack(fixture_cells()),fixture_board())),Chess.Ply{49,57,0,0}) == 0n : Nat}: {==}

def fixture_member(+choice: P.Choice) ->
  E.member(P.ply(49,57,choice),E.moves(C.candidates(O.pack(fixture_cells()),fixture_board()))):
  match choice:
    case P.Knight{}: Inl{{==}}
    case P.Bishop{}: Inr{Inl{{==}}}
    case P.Rook{}: Inr{Inr{Inl{{==}}}}
    case P.Queen{}: Inr{Inr{Inr{Inl{{==}}}}}

def fixture_bypass(+choice: P.Choice) ->
  {Chess.filter_requires(Sensitive.sensitive(fixture_board(),Prepared.rays(fixture_cells(),fixture_board())),P.ply(49,57,choice)) == False{} : Bool}: {==}

def fixture_initial() ->
  {Chess.in_check(O.pack(fixture_cells()),fixture_board(),1) == (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}: {==}

def fixture_old_rook() -> Q.rook(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}
def fixture_old_bishop() -> Q.bishop(fixture_cells(),0n,1,Chess.occupied(fixture_board())): {==}

def fixture_new_rook(+choice: P.Choice) -> Q.rook(fixture_cells(),0n,0,Chess.occupied(Raw.moved(fixture_board(),49,57,choice))):
  match choice:
    case P.Knight{}: {==}
    case P.Bishop{}: {==}
    case P.Rook{}: {==}
    case P.Queen{}: {==}

def fixture_new_bishop(+choice: P.Choice) -> Q.bishop(fixture_cells(),0n,0,Chess.occupied(Raw.moved(fixture_board(),49,57,choice))):
  match choice:
    case P.Knight{}: {==}
    case P.Bishop{}: {==}
    case P.Rook{}: {==}
    case P.Queen{}: {==}

def fixture_kind(+choice: P.Choice) ->
  {Chess.piece(Raw.moved(fixture_board(),49,57,choice),57) == P.tag(choice) : U32}:
  match choice:
    case P.Knight{}: {==}
    case P.Bishop{}: {==}
    case P.Rook{}: {==}
    case P.Queen{}: {==}

def raw_prefix_differs() ->
  {U64.is_eq(Check.prefix(fixture_cells(),Raw.moved(fixture_board(),49,57,P.Knight{}),0,0),
      Check.prefix(fixture_cells(),F.moved(fixture_board(),49,57),0,0)) == False{} : Bool}: {==}

def masked_prefix_same() ->
  {Prefix.hits(fixture_cells(),Raw.moved(fixture_board(),49,57,P.Knight{}),0n,0) ==
    Prefix.hits(fixture_cells(),F.moved(fixture_board(),49,57),0n,0) : U64}: {==}

def fixture_safe(+choice: P.Choice) ->
  {Chess.in_check(O.pack(fixture_cells()),Raw.moved(fixture_board(),49,57,choice),1) ==
    (O.pack(fixture_cells()),False{}) : Array<U64> & Bool}:
  use_no_check(fixture_cells(),fixture_board(),49,57,choice,True{},fixture_member(choice),fixture_valid(),fixture_nonempty(),{==},
    fixture_bypass(choice),fixture_old_rook(),fixture_old_bishop(),fixture_new_rook(choice),fixture_new_bishop(choice),fixture_initial())

def wrong_side_check() ->
  {Chess.in_check(O.pack(fixture_cells()),Raw.moved(fixture_board(),49,57,P.Knight{}),
      Chess.get_turn(Raw.moved(fixture_board(),49,57,P.Knight{}))) == (O.pack(fixture_cells()),True{}) : Array<U64> & Bool}: {==}

def ep_is_forced() -> {Chess.filter_requires(U64.zero(),Chess.Ply{49,57,0,1}) == True{} : Bool}: {==}

def raw_castle_can_bypass() -> {Chess.filter_requires(U64.zero(),Chess.Ply{49,57,0,2}) == False{} : Bool}: {==}
