import Base
import ./Selection.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../prepared_ray_coverage/Coverage.bend as G
import ../prepared_ray_coverage/consumer.bend as Prior
import ../storage/Observe.bend as O
import ../accepted_castle/Spec.bend as Prepared
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../castle_safety/Spec.bend as Sensitive
import ../board/Spec.bend as Board
import ../occupancy_insertion/Lattice.bend as L
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables

def use_selection(+b: Chess.Board,+nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) ->
  {Nat.is_lt(K.index(b),64n) == True{} : Bool} &
  ({Q.king(b) == U32.from_nat(K.index(b)) : U32} &
  ({U64.test_bit(Chess.get_kings(b),K.index(b)) == True{} : Bool} &
  {U64.test_bit(Chess.color(b,Chess.get_turn(b)),K.index(b)) == True{} : Bool})):
  (K.bound(b,nonempty),(K.chosen(b,nonempty),(K.king_bit(b,nonempty),K.owned(b,nonempty))))

def use_kind(+b: Chess.Board,+nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  good: {Board.valid(b) == True{} : Bool}) ->
  {Chess.piece(b,Q.king(b)) == 5 : U32}:
  K.kind(b,nonempty,good)

def use_required(+b: Chess.Board,+rays: U64,+dst: U32,+promotion: U32,+flag: U32,
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) ->
  {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{Q.king(b),dst,promotion,flag}) == True{} : Bool}:
  K.required(b,rays,dst,promotion,flag,nonempty)

# Caller supplies no square, square bound, or selected-square equality.
def use_coverage(+c: O.Cells,+b: Chess.Board,+dir: Nat,
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  db: {Nat.is_lt(dir,8n) == True{} : Bool},
  r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b))) ->
  L.subset(G.path(K.index(b),dir,Chess.occupied(b)),Prepared.rays(c,b)):
  Q.coverage(c,b,K.index(b),dir,K.bound(b,nonempty),db,K.chosen(b,nonempty),r,d)

# The actual False dispatch and whole table/mask pair are preserved.
def use_filter_checked(+c: O.Cells,+b: Chess.Board,+moves: List<&2,Chess.Ply>,
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b))) ->
  {Chess.filter_checked(b,moves,(O.pack(c),False{})) ==
    Chess.filter_blockers(b,moves,(O.pack(c),G.actual(K.index(b),Chess.occupied(b)))) : Array<U64> & List<&2,Chess.Ply>}:
  +query = Equal.trans(Array<U64> & U64,
    Chess.attack(4,O.pack(c),Q.king(b),Chess.get_turn(b),Chess.occupied(b)),
    Chess.attack(4,O.pack(c),U32.from_nat(K.index(b)),Chess.get_turn(b),Chess.occupied(b)),
    (O.pack(c),G.actual(K.index(b),Chess.occupied(b))),
    Equal.cong(U32,Array<U64> & U64,sq => Chess.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b)),
      Q.king(b),U32.from_nat(K.index(b)),K.chosen(b,nonempty)),
    Q.queen(c,K.index(b),Chess.get_turn(b),Chess.occupied(b),r,d))
  Equal.cong(Array<U64> & U64,Array<U64> & List<&2,Chess.Ply>,
    pair => Chess.filter_blockers(b,moves,pair),
    Chess.attack(4,O.pack(c),Q.king(b),Chess.get_turn(b),Chess.occupied(b)),
    (O.pack(c),G.actual(K.index(b),Chess.occupied(b))),query)

def use_ray(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+dir: Nat,+src: U32,+dst: U32,+acc: U64,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  db: {Nat.is_lt(dir,8n) == True{} : Bool},
  r: Q.rook(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,K.index(b),Chess.get_turn(b),Chess.occupied(b)),
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  L.subset(Tables.ray(7n,False{},U32.from_nat(K.index(b)),U32.from_nat(dir),Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(7n,False{},U32.from_nat(K.index(b)),U32.from_nat(dir),Chess.occupied(b),False{},acc)):
  Prior.use_ray(generation_table,c,b,K.index(b),dir,src,dst,acc,here,
    K.bound(b,nonempty),db,K.chosen(b,nonempty),r,d,bypass)

# Multiple kings and noncanonical turns follow actual color/ctz semantics.
def kings(+white: U64,+black: U64,turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.or(white,black),
    white,black,turn,0,64}
def multi() -> Chess.Board:
  kings(U64.or(U64.bit(32n),U64.bit(63n)),U64.bit(0n),1)
def multi_valid() -> {Board.valid(multi()) == True{} : Bool}: {==}
def multi_selects_lowest() -> {Q.king(multi()) == 32 : U32}: {==}
def multi_use_range() -> {Nat.is_lt(K.index(multi()),64n) == True{} : Bool}:
  K.bound(multi(),{==})
def multi_use_kind() -> {Chess.piece(multi(),Q.king(multi())) == 5 : U32}:
  K.kind(multi(),{==},multi_valid())
def black_high() -> {Q.king(kings(U64.bit(0n),U64.bit(63n),0)) == 63 : U32}: {==}
def noncanonical_turn() -> {Q.king(kings(U64.bit(0n),U64.bit(63n),2)) == 63 : U32}: {==}
def selected_requires_check() ->
  {Chess.filter_requires(Sensitive.sensitive(multi(),U64.zero()),Chess.Ply{Q.king(multi()),33,0,0}) == True{} : Bool}:
  K.required(multi(),U64.zero(),33,0,0,{==})

# Board.valid alone admits no moving king: actual ctz(zero) is 64.
def empty() -> Chess.Board:
  kings(U64.zero(),U64.zero(),1)
def empty_valid() -> {Board.valid(empty()) == True{} : Bool}: {==}
def empty_selection() -> {Q.king(empty()) == 64 : U32}: {==}
def empty_outside() -> {Nat.is_lt(K.index(empty()),64n) == False{} : Bool}: {==}
def opposite_only() -> {Q.king(kings(U64.zero(),U64.bit(7n),1)) == 64 : U32}: {==}

# Presence alone does not rule out an earlier decoder kind at the same square.
def collision() -> Chess.Board:
  Chess.Board{U64.bit(32n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(32n),
    U64.bit(32n),U64.zero(),1,0,64}
def collision_nonempty() -> {U64.is_zero(K.plane(collision())) == False{} : Bool}: {==}
def collision_invalid() -> {Board.valid(collision()) == False{} : Bool}: {==}
def collision_piece() -> {Chess.piece(collision(),Q.king(collision())) == 0 : U32}: {==}
