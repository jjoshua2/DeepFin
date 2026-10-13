import Base
import ./Emission.bend as Destination
import ./Provenance.bend as Provenance
import ../candidate_source_provenance/Source.bend as Source
import ../ordinary_king_frame/Source.bend as OrdinarySource
import ./Targets.bend as Targets
import ../ordinary_king_frame/Frame.bend as F
import ../ordinary_king_frame/consumer.bend as Old
import ../board/Spec.bend as B
import ../castle_safety/Spec.bend as Sensitive
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../prepared_king_selection/Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../prepared_ray_coverage/Coverage.bend as G
import ../storage/Observe.bend as O
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as Chess

def from_certificate(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  certified: Source.source(b,Chess.Ply{src,dst,0,0}) & Destination.clear(b,Chess.Ply{src,dst,0,0}),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {F.plane(F.moved(b,src,dst),Chess.get_turn(b)) == F.plane(b,Chess.get_turn(b)) : U64}:
  (source_cert,target_clear) = certified
  Old.from_facts(b,src,dst,target_clear,OrdinarySource.facts_known(b,rays,src,dst,source_cert,good,bypass))

# The target certificate is derived from production generation, not supplied by the caller.
def use_frame(generation_table: Array<U64>,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  +good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {F.plane(F.moved(b,src,dst),Chess.get_turn(b)) == F.plane(b,Chess.get_turn(b)) : U64}:
  from_certificate(b,rays,src,dst,Provenance.member(generation_table,b,Chess.Ply{src,dst,0,0},good,here),good,bypass)

def use_selected(generation_table: Array<U64>,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))) == Q.king(b) : U32}:
  Equal.cong(U64,U32,U64.ctz,F.plane(F.moved(b,src,dst),Chess.get_turn(b)),F.plane(b,Chess.get_turn(b)),
    use_frame(generation_table,b,rays,src,dst,here,good,bypass))

# Whole actual in_check pair: original side is passed after turn metadata flips.

def use_check_square(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {Chess.in_check(O.pack(c),F.moved(b,src,dst),Chess.get_turn(b)) ==
    Chess.attacked(O.pack(c),F.moved(b,src,dst),Q.king(b),U32.xor(Chess.get_turn(b),1)) : Array<U64> & Bool}:
  Equal.cong(U32,Array<U64> & Bool,square => Chess.attacked(O.pack(c),F.moved(b,src,dst),square,U32.xor(Chess.get_turn(b),1)),
    U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),Q.king(b),
    use_selected(generation_table,b,rays,src,dst,here,good,bypass))


def post_queen(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ->
  {Chess.attack(4,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),G.actual(K.index(b),Chess.occupied(F.moved(b,src,dst)))) : Array<U64> & U64}:
  +chosen = Equal.trans(U32,U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),Q.king(b),U32.from_nat(K.index(b)),
    use_selected(generation_table,b,rays,src,dst,here,good,bypass),K.chosen(b,nonempty))
  Equal.trans(Array<U64> & U64,
    Chess.attack(4,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
    Chess.attack(4,O.pack(c),U32.from_nat(K.index(b)),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
    (O.pack(c),G.actual(K.index(b),Chess.occupied(F.moved(b,src,dst)))),
    Equal.cong(U32,Array<U64> & U64,square => Chess.attack(4,O.pack(c),square,U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
      U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.from_nat(K.index(b)),chosen),
    Q.queen(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)),r,d))

# Independent old/new occupancy contracts are composed at the preserved square.

def use_queries(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  old_r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  old_d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  new_r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  new_d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ->
  {Chess.attack(4,O.pack(c),Q.king(b),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),G.actual(K.index(b),Chess.occupied(b))) : Array<U64> & U64} &
  {Chess.attack(4,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),G.actual(K.index(b),Chess.occupied(F.moved(b,src,dst)))) : Array<U64> & U64}:
  (Old.old_queen(c,b,nonempty,old_r,old_d),post_queen(generation_table,c,b,rays,src,dst,nonempty,here,good,bypass,new_r,new_d))


def board() -> Chess.Board:
  Old.board()
def board_valid() -> {B.valid(board()) == True{} : Bool}: {==}
def actual_member() -> E.member(Chess.Ply{2,10,0,0},E.moves(C.candidates(ALeaf{U64.zero()},board()))): Inl{{==}}
def actual_destination() -> D.disjoint(U64.bit(10n),Chess.get_kings(board())):
  Destination.member(ALeaf{U64.zero()},board(),Chess.Ply{2,10,0,0},board_valid(),actual_member())
def actual_frame() -> {F.plane(F.moved(board(),2,10),1) == U64.bit(8n) : U64}:
  use_frame(ALeaf{U64.zero()},board(),U64.zero(),2,10,actual_member(),board_valid(),{==})
def actual_selected() -> {U64.ctz(F.plane(F.moved(board(),2,10),1)) == 8 : U32}:
  use_selected(ALeaf{U64.zero()},board(),U64.zero(),2,10,actual_member(),board_valid(),{==})

# An uncolored king is not occupied. Actual pawn generation reaches it without Board.valid.
def phantom() -> Chess.Board:
  Chess.Board{U64.bit(2n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(8n),U64.bit(10n)),
    U64.or(U64.bit(2n),U64.bit(8n)),U64.zero(),1,0,64}
def phantom_invalid() -> {B.valid(phantom()) == False{} : Bool}: {==}
def phantom_member() -> E.member(Chess.Ply{2,10,0,0},E.moves(C.candidates(ALeaf{U64.zero()},phantom()))): Inl{{==}}
def phantom_destination_set() -> {U64.test_bit(Chess.get_kings(phantom()),10n) == True{} : Bool}: {==}

# This uses the actual pawn target guard, including an occupied EP square.
def occupied_ep() -> Chess.Board:
  Chess.Board{U64.bit(32n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(40n),U64.bit(40n),U64.bit(32n),1,0,40}
def occupied_ep_valid() -> {B.valid(occupied_ep()) == True{} : Bool}: {==}
def occupied_ep_rejected() -> {Chess.ep_target(occupied_ep()) == U64.zero() : U64}: {==}

def enemy_king() -> Chess.Board:
  Chess.Board{U64.bit(9n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(18n),U64.bit(9n),U64.bit(18n),1,0,64}
def enemy_king_valid() -> {B.valid(enemy_king()) == True{} : Bool}: {==}
def pawn_capture_king_excluded() -> {U64.test_bit(Targets.mask(Chess.pawn_targets(ALeaf{U64.from_u32(262144)},enemy_king(),9)),18n) == False{} : Bool}: {==}

def duplicate_tail() -> List<&2,Chess.Ply>:
  Chess.Ply{2,10,0,0} <> Chess.Ply{2,10,0,0} <> Nil{}
def multi_bit_duplicate_after() ->
  {Chess.scan_after(2,False{},64,duplicate_tail(),(ALeaf{U64.bit(63n)},U64.or(U64.bit(10n),U64.bit(18n)))) ==
    (ALeaf{U64.bit(63n)},Chess.Ply{2,18,0,0} <> Chess.Ply{2,10,0,0} <> duplicate_tail()) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
def promotion_overrides_ep() ->
  {Chess.scan_after(9,True{},56,duplicate_tail(),(ALeaf{U64.zero()},U64.bit(56n))) ==
    (ALeaf{U64.zero()},Chess.Ply{9,56,1,0} <> Chess.Ply{9,56,2,0} <> Chess.Ply{9,56,3,0} <> Chess.Ply{9,56,4,0} <> duplicate_tail()) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
def ordinary_ep_flag() ->
  {Chess.scan_after(2,True{},10,duplicate_tail(),(ALeaf{U64.zero()},U64.bit(10n))) ==
    (ALeaf{U64.zero()},Chess.Ply{2,10,0,1} <> duplicate_tail()) : Array<U64> & List<&2,Chess.Ply>}:
  {==}
