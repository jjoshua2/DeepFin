import Base
import ./Fixed.bend as Fixed
import ./Prefix.bend as Prefix
import ../ordinary_slider_preservation/Check.bend as C
import ../ordinary_slider_preservation/Hits.bend as H
import ../candidate_king_destination/Algebra.bend as K
import ../prepared_ray_coverage/Query.bend as Q
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Algebra.bend as R
import ../ray/Slider.bend as Slider
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def suffix_shape(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  r: Q.rook(c,q,by,Chess.occupied(b)),d: Q.bishop(c,q,by,Chess.occupied(b))) ->
  {Chess.attacked_rook(b,U32.from_nat(q),by,C.prefix(c,b,U32.from_nat(q),by),
      Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))) ==
    (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & U64,Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})),r) :
    {Chess.attacked_rook(b,U32.from_nat(q),by,C.prefix(c,b,U32.from_nat(q),by),_) ==
      (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(2,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)),
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})),d) :
    {Chess.attacked_bishop(b,by,U64.or(C.prefix(c,b,U32.from_nat(q),by),C.rook_raw(b,q)),_) ==
      (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))) : Array<U64> & Bool}
  {==}

def shape(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  r: Q.rook(c,q,by,Chess.occupied(b)),d: Q.bishop(c,q,by,Chess.occupied(b))) ->
  {Chess.attacked(O.pack(c),b,U32.from_nat(q),by) ==
    (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,Chess.attacked(O.pack(c),b,U32.from_nat(q),by),
    Chess.attacked_rook(b,U32.from_nat(q),by,C.prefix(c,b,U32.from_nat(q),by),
      Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))),
    (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))),
    Fixed.callbacks(c,b,U32.from_nat(q),by),suffix_shape(c,b,q,by,r,d))

# Extract the empty enemy-masked slider suffix from PR1034's actual callback pair.
def slider_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  r: Q.rook(c,q,by,Chess.occupied(b)),d: Q.bishop(c,q,by,Chess.occupied(b)),
  safe: {Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),
      Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))) ==
    (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {H.hits(b,q,by) == U64.zero() : U64}:
  K.zero_sound(H.hits(b,q,by),K.not_false(U64.is_zero(H.hits(b,q,by)),
    Equal.cong(Array<U64> & Bool,Bool,C.flag,(O.pack(c),Bool.not(U64.is_zero(H.hits(b,q,by)))),(O.pack(c),False{}),
      Equal.trans(Array<U64> & Bool,(O.pack(c),Bool.not(U64.is_zero(H.hits(b,q,by)))),
        Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))),
        (O.pack(c),False{}),
        Equal.sym(Array<U64> & Bool,
          Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))),
          (O.pack(c),Bool.not(U64.is_zero(H.hits(b,q,by)))),H.suffix(c,b,q,by,r,d)),safe))))

def split(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32) ->
  {C.hits(c,b,q,by) == U64.or(Prefix.hits(c,b,q,by),H.hits(b,q,by)) : U64}:
  +prefix = C.prefix(c,b,U32.from_nat(q),by)
  +rook = C.rook_raw(b,q)
  +bishop = C.bishop_raw(b,q)
  +enemy = Chess.color(b,by)
  Equal.trans(U64,C.hits(c,b,q,by),U64.and(U64.or(prefix,U64.or(rook,bishop)),enemy),
    U64.or(Prefix.hits(c,b,q,by),H.hits(b,q,by)),
    Equal.cong(U64,U64,x => U64.and(x,enemy),U64.or(U64.or(prefix,rook),bishop),
      U64.or(prefix,U64.or(rook,bishop)),R.or_assoc(prefix,rook,bishop)),
    L.and_or(prefix,U64.or(rook,bishop),enemy))

def empty(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  prefix: {Prefix.hits(c,b,q,by) == U64.zero() : U64},
  slider: {H.hits(b,q,by) == U64.zero() : U64}) ->
  {C.hits(c,b,q,by) == U64.zero() : U64}:
  Equal.trans(U64,C.hits(c,b,q,by),U64.or(Prefix.hits(c,b,q,by),H.hits(b,q,by)),U64.zero(),split(c,b,q,by),
    Equal.trans(U64,U64.or(Prefix.hits(c,b,q,by),H.hits(b,q,by)),U64.or(U64.zero(),U64.zero()),U64.zero(),
      Slider.combine(Prefix.hits(c,b,q,by),U64.zero(),H.hits(b,q,by),U64.zero(),prefix,slider),{==}))

def attacked_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32,
  +r: {Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +d: {Chess.attack(2,O.pack(c),U32.from_nat(q),by,Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})) : Array<U64> & U64},
  prefix: {Prefix.hits(c,b,q,by) == U64.zero() : U64},
  safe: {Chess.attacked_rook(b,U32.from_nat(q),by,U64.zero(),
      Chess.attack(3,O.pack(c),U32.from_nat(q),by,Chess.occupied(b))) ==
    (O.pack(c),False{}) : Array<U64> & Bool}) ->
  {Chess.attacked(O.pack(c),b,U32.from_nat(q),by) == (O.pack(c),False{}) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,Chess.attacked(O.pack(c),b,U32.from_nat(q),by),
    (O.pack(c),Bool.not(U64.is_zero(C.hits(c,b,q,by)))),(O.pack(c),False{}),shape(c,b,q,by,r,d),
    Equal.cong(U64,Array<U64> & Bool,x => (O.pack(c),Bool.not(U64.is_zero(x))),C.hits(c,b,q,by),U64.zero(),
      empty(c,b,q,by,prefix,slider_clear(c,b,q,by,r,d,safe))))
