import Base
import ./Algebra.bend as A
import ./Check.bend as C
import ../ordinary_king_frame/Enemy.bend as Enemy
import ../ordinary_king_frame/Frame.bend as F
import ../board/Spec.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../storage/Observe.bend as O
import ../prepared_ray_coverage/Query.bend as Q
import ../ray/Algebra.bend as R
import ../ray/Slider.bend as Slider
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

# Canonical turn is explicit: Chess.color treats every non-1 U32 as black.
def enemies_subset(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  L.subset(C.enemies(F.moved(b,src,dst),Chess.get_turn(b),piece),C.enemies(b,Chess.get_turn(b),piece)):
  A.or_mono(Enemy.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),Enemy.enemy_kind(b,Chess.get_turn(b),piece),
    Enemy.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),B.Queen{}),Enemy.enemy_kind(b,Chess.get_turn(b),B.Queen{}),
    Enemy.kind_subset(b,src,dst,piece,white,canonical),Enemy.kind_subset(b,src,dst,B.Queen{},white,canonical))

def raw(+b: Chess.Board,+q: Nat) -> U64:
  U64.or(C.rook_raw(b,q),C.bishop_raw(b,q))
def hits(+b: Chess.Board,+q: Nat,by: U32) -> U64:
  U64.and(raw(b,q),Chess.color(b,by))

# This is the actual rook/bishop suffix with a zero non-slider accumulator.
def suffix(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  r: Q.rook(c,q,by,Chess.occupied(b)),d: Q.bishop(c,q,by,Chess.occupied(b))) ->
  {Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))) ==
    (O.pack(c),Bool.not(U64.is_zero(hits(b,q,by)))) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & U64,Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})),r) :
    {Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(2,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})),d) :
    {Chess.attacked_bishop(b,by,U64.or(U64.zero(),C.rook_raw(b,q)),_) ==
      (O.pack(c),Bool.not(U64.is_zero(hits(b,q,by)))) : Array<U64> & Bool}
  Equal.cong(U64,Array<U64> & Bool,x => (O.pack(c),Bool.not(U64.is_zero(U64.and(U64.or(x,C.bishop_raw(b,q)),Chess.color(b,by))))),
    U64.or(U64.zero(),C.rook_raw(b,q)),C.rook_raw(b,q),R.left_zero(C.rook_raw(b,q)))

def clear(+b: Chess.Board,+q: Nat,+side: U32,
  r: {U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),C.enemies(b,side,B.Rook{})) == U64.zero() : U64},
  d: {U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),C.enemies(b,side,B.Bishop{})) == U64.zero() : U64}) ->
  {hits(b,q,U32.xor(side,1)) == U64.zero() : U64}:
  +enemy = Chess.color(b,U32.xor(side,1))
  Equal.trans(U64,hits(b,q,U32.xor(side,1)),
    U64.or(U64.and(C.rook_raw(b,q),enemy),U64.and(C.bishop_raw(b,q),enemy)),U64.zero(),
    L.and_or(C.rook_raw(b,q),C.bishop_raw(b,q),enemy),
    Equal.trans(U64,U64.or(U64.and(C.rook_raw(b,q),enemy),U64.and(C.bishop_raw(b,q),enemy)),
      U64.or(U64.zero(),U64.zero()),U64.zero(),
      Slider.combine(U64.and(C.rook_raw(b,q),enemy),U64.zero(),U64.and(C.bishop_raw(b,q),enemy),U64.zero(),
        Equal.trans(U64,U64.and(C.rook_raw(b,q),enemy),
          U64.and(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),C.enemies(b,side,B.Rook{})),U64.zero(),
          C.gate(Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{}),Chess.get_rooks(b),Chess.get_queens(b),enemy),r),
        Equal.trans(U64,U64.and(C.bishop_raw(b,q),enemy),
          U64.and(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),C.enemies(b,side,B.Bishop{})),U64.zero(),
          C.gate(Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{}),Chess.get_bishops(b),Chess.get_queens(b),enemy),d)),{==}))

def safe(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32,
  r: Q.rook(c,q,U32.xor(side,1),Chess.occupied(b)),d: Q.bishop(c,q,U32.xor(side,1),Chess.occupied(b)),
  empty: {hits(b,q,U32.xor(side,1)) == U64.zero() : U64}) ->
  {Chess.attacked_rook(b,U32.from_nat(q),U32.xor(side,1),U64.zero(),
      Chess.attack(3,O.pack(c),U32.from_nat(q),U32.xor(side,1),Chess.occupied(b))) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.attacked_rook(b,U32.from_nat(q),U32.xor(side,1),U64.zero(),Chess.attack(3,O.pack(c),U32.from_nat(q),U32.xor(side,1),Chess.occupied(b))),
    (O.pack(c),Bool.not(U64.is_zero(hits(b,q,U32.xor(side,1))))),(O.pack(c),False{}),
    suffix(c,b,q,U32.xor(side,1),r,d),
    Equal.cong(U64,Array<U64> & Bool,x => (O.pack(c),Bool.not(U64.is_zero(x))),hits(b,q,U32.xor(side,1)),U64.zero(),empty))
