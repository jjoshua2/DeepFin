import Base
import ./Algebra.bend as A
import ../candidate_king_destination/Algebra.bend as K
import ../storage/Observe.bend as O
import ../storage/Read.bend as Read
import ../accepted_castle/Spec.bend as Mask
import ../prepared_ray_coverage/Query.bend as Q
import ../prepared_ray_coverage/Coverage.bend as G
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Algebra.bend as R
import ../ordinary_king_frame/Enemy.bend as Enemy
import ../board/Spec.bend as B
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

# Reassemble an actual lookup pair from its storage observation; no mask contract.
def keep_pair(+c: O.Cells,r: Array<U64> & U64,kept: {O.storage(r) == O.pack(c) : Array<U64>}) ->
  {r == (O.pack(c),Mask.mask(r)) : Array<U64> & U64}:
  (table,+value) = r
  Equal.cong(Array<U64>,Array<U64> & U64,t => (t,value),table,O.pack(c),kept)

def get_pair(+c: O.Cells,+i: U32) ->
  {Array.get(U64,O.pack(c),i) == (O.pack(c),Mask.mask(Array.get(U64,O.pack(c),i))) : Array<U64> & U64}:
  keep_pair(c,Array.get(U64,O.pack(c),i),Read.read(c,i))
def pawn_index(+sq: U32,by: U32) -> U32:
  U32.add(sq,Bool.pick(U32,U32.is_eq(U32.xor(by,1),1),384,448))
def pawn(+c: O.Cells,+sq: U32,by: U32) -> U64:
  Mask.mask(Array.get(U64,O.pack(c),pawn_index(sq,by)))
def knight(+c: O.Cells,sq: U32) -> U64:
  Mask.mask(Array.get(U64,O.pack(c),U32.add(256,sq)))
def king(+c: O.Cells,sq: U32) -> U64:
  Mask.mask(Array.get(U64,O.pack(c),U32.add(320,sq)))
def prefix(+c: O.Cells,+b: Chess.Board,+sq: U32,by: U32) -> U64:
  U64.or(U64.or(U64.and(pawn(c,sq,by),Chess.get_pawns(b)),U64.and(knight(c,sq),Chess.get_knights(b))),
    U64.and(king(c,sq),Chess.get_kings(b)))
def rook_raw(+b: Chess.Board,q: Nat) -> U64:
  U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),U64.or(Chess.get_rooks(b),Chess.get_queens(b)))
def bishop_raw(+b: Chess.Board,q: Nat) -> U64:
  U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),U64.or(Chess.get_bishops(b),Chess.get_queens(b)))
def raw(+c: O.Cells,+b: Chess.Board,+q: Nat,by: U32) -> U64:
  U64.or(U64.or(prefix(c,b,U32.from_nat(q),by),rook_raw(b,q)),bishop_raw(b,q))
def hits(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32) -> U64:
  U64.and(raw(c,b,q,by),Chess.color(b,by))

# Expand the production callbacks at one table and square. The three fixed
# lookups preserve the table for arbitrary Cells and need no geometry premise.
def shape(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  r: Q.rook(c,q,by,Chess.occupied(b)),d: Q.bishop(c,q,by,Chess.occupied(b))) ->
  {Chess.attacked(O.pack(c),b,U32.from_nat(q),by) ==
    (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & U64,Chess.attack(0,O.pack(c),U32.from_nat(q),U32.xor(by,1),Chess.occupied(b)),
    (O.pack(c),pawn(c,U32.from_nat(q),by)),get_pair(c,pawn_index(U32.from_nat(q),by))) :
    {Chess.attacked_pawn(b,U32.from_nat(q),by,_) == (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(1,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),knight(c,U32.from_nat(q))),get_pair(c,U32.add(256,U32.from_nat(q)))) :
    {Chess.attacked_knight(b,U32.from_nat(q),by,U64.and(pawn(c,U32.from_nat(q),by),Chess.get_pawns(b)),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(5,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),king(c,U32.from_nat(q))),get_pair(c,U32.add(320,U32.from_nat(q)))) :
    {Chess.attacked_king(b,U32.from_nat(q),by,
      U64.or(U64.and(pawn(c,U32.from_nat(q),by),Chess.get_pawns(b)),U64.and(knight(c,U32.from_nat(q)),Chess.get_knights(b))),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})),r) :
    {Chess.attacked_rook(b,U32.from_nat(q),by,prefix(c,b,U32.from_nat(q),by),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(2,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})),d) :
    {Chess.attacked_bishop(b,by,U64.or(prefix(c,b,U32.from_nat(q),by),rook_raw(b,q)),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))) : Array<U64> & Bool}
  {==}

def flag(r: Array<U64> & Bool) -> Bool:
  (table,v) = r
  v

def initial_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,
  chosen: {Q.king(b) == U32.from_nat(q) : U32},
  r: Q.rook(c,q,Chess.get_turn(b),Chess.occupied(b)),d: Q.bishop(c,q,Chess.get_turn(b),Chess.occupied(b)),
  unchecked: {Chess.in_check(O.pack(c),b,Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}:
  +by = U32.xor(Chess.get_turn(b),1)
  +expanded = Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b)),
    Chess.attacked(O.pack(c),b,U32.from_nat(q),by),(O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))),
    Equal.cong(U32,Array<U64> & Bool,sq => Chess.attacked(O.pack(c),b,sq,by),Q.king(b),U32.from_nat(q),chosen),
    shape(c,b,q,by,r,d))
  K.zero_sound(hits(c,b,q,by),K.not_false(U64.is_zero(hits(c,b,q,by)),
    Equal.cong(Array<U64> & Bool,Bool,flag,(O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))),(O.pack(c),False{}),
      Equal.trans(Array<U64> & Bool,(O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))),Chess.in_check(O.pack(c),b,Chess.get_turn(b)),
        (O.pack(c),False{}),Equal.sym(Array<U64> & Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b)),
          (O.pack(c),Bool.not(U64.is_zero(hits(c,b,q,by)))),expanded),unchecked))))

def gate(+mask: U64,+a: U64,+b: U64,+enemy: U64) ->
  {U64.and(U64.and(mask,U64.or(a,b)),enemy) == U64.and(mask,U64.or(U64.and(a,enemy),U64.and(b,enemy))) : U64}:
  Equal.trans(U64,U64.and(U64.and(mask,U64.or(a,b)),enemy),U64.and(mask,U64.and(U64.or(a,b),enemy)),
    U64.and(mask,U64.or(U64.and(a,enemy),U64.and(b,enemy))),R.and_assoc(mask,U64.or(a,b),enemy),
    Equal.cong(U64,U64,x => U64.and(mask,x),U64.and(U64.or(a,b),enemy),U64.or(U64.and(a,enemy),U64.and(b,enemy)),L.and_or(a,b,enemy)))

def enemies(+b: Chess.Board,+side: U32,piece: B.Piece) -> U64:
  U64.or(Enemy.enemy_kind(b,side,piece),Enemy.enemy_kind(b,side,B.Queen{}))

def rook_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,
  clear: {hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}) ->
  {U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Rook{})) == U64.zero() : U64}:
  +by = U32.xor(Chess.get_turn(b),1)
  Equal.trans(U64,U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Rook{})),
    U64.and(rook_raw(b,q),Chess.color(b,by)),U64.zero(),
    Equal.sym(U64,U64.and(rook_raw(b,q),Chess.color(b,by)),
      U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Rook{})),
      gate(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),Chess.get_rooks(b),Chess.get_queens(b),Chess.color(b,by))),
    A.zero_subset(U64.and(rook_raw(b,q),Chess.color(b,by)),hits(c,b,q,by),
      A.and_mono(rook_raw(b,q),raw(c,b,q,by),Chess.color(b,by),Chess.color(b,by),
        G.lift_left(rook_raw(b,q),U64.or(prefix(c,b,U32.from_nat(q),by),rook_raw(b,q)),bishop_raw(b,q),
          G.lift_right(rook_raw(b,q),prefix(c,b,U32.from_nat(q),by),rook_raw(b,q),L.reflexive(rook_raw(b,q)))),L.reflexive(Chess.color(b,by))),clear))

def bishop_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,
  clear: {hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}) ->
  {U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Bishop{})) == U64.zero() : U64}:
  +by = U32.xor(Chess.get_turn(b),1)
  Equal.trans(U64,U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Bishop{})),
    U64.and(bishop_raw(b,q),Chess.color(b,by)),U64.zero(),
    Equal.sym(U64,U64.and(bishop_raw(b,q),Chess.color(b,by)),
      U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),enemies(b,Chess.get_turn(b),B.Bishop{})),
      gate(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),Chess.get_bishops(b),Chess.get_queens(b),Chess.color(b,by))),
    A.zero_subset(U64.and(bishop_raw(b,q),Chess.color(b,by)),hits(c,b,q,by),
      A.and_mono(bishop_raw(b,q),raw(c,b,q,by),Chess.color(b,by),Chess.color(b,by),
        G.lift_right(bishop_raw(b,q),U64.or(prefix(c,b,U32.from_nat(q),by),rook_raw(b,q)),bishop_raw(b,q),L.reflexive(bishop_raw(b,q))),L.reflexive(Chess.color(b,by))),clear))
