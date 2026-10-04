import Base
import ./Frame.bend as F
import ./Source.bend as Source
import ./Enemy.bend as Enemy
import ../board/Spec.bend as B
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../castle_safety/Spec.bend as Sensitive
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../prepared_king_selection/Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../prepared_ray_coverage/Coverage.bend as G
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def from_facts(+b: Chess.Board,+src: U32,+dst: U32,
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  facts: D.disjoint(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) &
    {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {F.plane(F.moved(b,src,dst),Chess.get_turn(b)) == F.plane(b,Chess.get_turn(b)) : U64}:
  (source_clear,not_king) = facts
  F.plane_keep(b,src,dst,source_clear,target_clear,not_king)

# Membership proves source ownership/range; destination king exclusion is separate.
def use_frame(generation_table: Array<U64>,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {F.plane(F.moved(b,src,dst),Chess.get_turn(b)) == F.plane(b,Chess.get_turn(b)) : U64}:
  from_facts(b,src,dst,target_clear,Source.facts(generation_table,b,rays,src,dst,here,good,bypass))

def use_selected(generation_table: Array<U64>,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))) == Q.king(b) : U32}:
  Equal.cong(U64,U32,U64.ctz,F.plane(F.moved(b,src,dst),Chess.get_turn(b)),F.plane(b,Chess.get_turn(b)),
    use_frame(generation_table,b,rays,src,dst,target_clear,here,good,bypass))

# Whole actual in_check pair: original side is passed after turn metadata flips.
def use_check_square(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {Chess.in_check(O.pack(c),F.moved(b,src,dst),Chess.get_turn(b)) ==
    Chess.attacked(O.pack(c),F.moved(b,src,dst),Q.king(b),U32.xor(Chess.get_turn(b),1)) : Array<U64> & Bool}:
  Equal.cong(U32,Array<U64> & Bool,square => Chess.attacked(O.pack(c),F.moved(b,src,dst),square,U32.xor(Chess.get_turn(b),1)),
    U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),Q.king(b),
    use_selected(generation_table,b,rays,src,dst,target_clear,here,good,bypass))

def old_queen(+c: O.Cells,+b: Chess.Board,+nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b))) ->
  {Chess.attack(4,O.pack(c),Q.king(b),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),G.actual(K.index(b),Chess.occupied(b))) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Chess.attack(4,O.pack(c),Q.king(b),Chess.get_turn(b),Chess.occupied(b)),
    Chess.attack(4,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)),
    (O.pack(c),G.actual(K.index(b),Chess.occupied(b))),
    Equal.cong(U32,Array<U64> & U64,square => Chess.attack(4,O.pack(c),square,Chess.get_turn(b),Chess.occupied(b)),
      Q.king(b),U32.from_nat(K.index(b)),K.chosen(b,nonempty)),
    Q.queen(c,K.index(b),Chess.get_turn(b),Chess.occupied(b),r,d))

def post_queen(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  r: Q.rook(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))),
  d: Q.bishop(c,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst)))) ->
  {Chess.attack(4,O.pack(c),U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),U32.xor(Chess.get_turn(b),1),Chess.occupied(F.moved(b,src,dst))) ==
    (O.pack(c),G.actual(K.index(b),Chess.occupied(F.moved(b,src,dst)))) : Array<U64> & U64}:
  +chosen = Equal.trans(U32,U64.ctz(F.plane(F.moved(b,src,dst),Chess.get_turn(b))),Q.king(b),U32.from_nat(K.index(b)),
    use_selected(generation_table,b,rays,src,dst,target_clear,here,good,bypass),K.chosen(b,nonempty))
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
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
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
  (old_queen(c,b,nonempty,old_r,old_d),post_queen(generation_table,c,b,rays,src,dst,nonempty,target_clear,here,good,bypass,new_r,new_d))

def use_enemy(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  L.subset(Enemy.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),Enemy.enemy_kind(b,Chess.get_turn(b),piece)):
  Enemy.kind_subset(b,src,dst,piece,white,canonical)

def board() -> Chess.Board:
  Chess.Board{U64.bit(2n),U64.zero(),U64.zero(),U64.bit(63n),U64.zero(),U64.or(U64.bit(8n),U64.bit(56n)),
    U64.or(U64.bit(2n),U64.bit(8n)),U64.or(U64.bit(63n),U64.bit(56n)),1,0,64}
def board_valid() -> {B.valid(board()) == True{} : Bool}: {==}
def actual_member() -> E.member(Chess.Ply{2,10,0,0},E.moves(C.candidates(ALeaf{U64.zero()},board()))): Inl{{==}}
def actual_frame() -> {F.plane(F.moved(board(),2,10),1) == U64.bit(8n) : U64}:
  use_frame(ALeaf{U64.zero()},board(),U64.zero(),2,10,{==},actual_member(),board_valid(),{==})
def actual_selected() -> {U64.ctz(F.plane(F.moved(board(),2,10),1)) == 8 : U32}:
  use_selected(ALeaf{U64.zero()},board(),U64.zero(),2,10,{==},actual_member(),board_valid(),{==})
def metadata_turn_flips() -> {Chess.get_turn(F.moved(board(),2,10)) == 0 : U32}: {==}
def next_side_selects_other() -> {Q.king(F.moved(board(),2,10)) == 56 : U32}: {==}

# Total ordinary-update witnesses below assert representation behavior, not legality.
def capture_erases_rook() -> {Enemy.enemy_kind(F.moved(board(),2,63),1,B.Rook{}) == U64.zero() : U64}:
  Enemy.kind_after(board(),2,63,B.Rook{},True{},{==})
def capture_subset() -> L.subset(Enemy.enemy_kind(F.moved(board(),2,63),1,B.Rook{}),Enemy.enemy_kind(board(),1,B.Rook{})):
  use_enemy(board(),2,63,B.Rook{},True{},{==})
def king_destination_clears_plane() -> {U64.ctz(F.plane(F.moved(board(),2,8),1)) == 64 : U32}: {==}
def phantom() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(8n),U64.or(U64.bit(2n),U64.bit(8n)),U64.zero(),1,0,64}
def phantom_invalid() -> {B.valid(phantom()) == False{} : Bool}: {==}
def phantom_decoder() -> {Chess.piece(phantom(),2) == 5 : U32}: {==}
def phantom_adds_king() -> {Chess.get_kings(F.moved(phantom(),2,10)) == U64.or(U64.bit(8n),U64.bit(10n)) : U64}: {==}

def noncanonical() -> Chess.Board:
  Chess.Board{U64.bit(2n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(U64.bit(8n),U64.bit(56n)),
    U64.bit(8n),U64.or(U64.bit(2n),U64.bit(56n)),2,0,64}
def noncanonical_valid() -> {B.valid(noncanonical()) == True{} : Bool}: {==}
def noncanonical_enemy_adds_target() -> {U64.test_bit(Enemy.enemy(F.moved(noncanonical(),2,10),2),10n) == True{} : Bool}: {==}
def noncanonical_old_target_clear() -> {U64.test_bit(Enemy.enemy(noncanonical(),2),10n) == False{} : Bool}: {==}
# The intersection is empty; the adjacent new/old bit-10 witnesses show failed subset.
def noncanonical_pawn_intersection_empty() -> {U64.and(Enemy.enemy_kind(noncanonical(),2,B.Pawn{}),Enemy.enemy_kind(F.moved(noncanonical(),2,10),2,B.Pawn{})) == U64.zero() : U64}: {==}
