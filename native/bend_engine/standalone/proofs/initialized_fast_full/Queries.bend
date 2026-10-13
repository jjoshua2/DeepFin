# Discharge old and actual-child slider contracts from exact initialization.
import Base
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Masks.bend as M
import ../forward_attacks/Keys.bend as Keys
import ../ray/Spec.bend as Ray
import ../ray/Slider.bend as Slider
import ../prepared_ray_coverage/Query.bend as Q
import ../prepared_king_selection/Selection.bend as K
import ../generated_bypass_safety/Spec.bend as S
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables

def rook_mask(+q: Nat,+occ: U64,+bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Ray.rays(q,occ) == Tables.slider(U32.from_nat(q),False{},occ,False{}) : U64}:
  Equal.trans(U64,Ray.rays(q,occ),Ray.directional(q,0n,occ),
    Tables.slider(U32.from_nat(q),False{},occ,False{}),Keys.rook(q,occ,bound),
    Equal.sym(U64,Tables.slider(U32.from_nat(q),False{},occ,False{}),
      Ray.directional(q,0n,occ),Slider.square(q,False{},occ,bound)))

def bishop_mask(+q: Nat,+occ: U64,+bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Ray.rays(64n+q,occ) == Tables.slider(U32.from_nat(q),True{},occ,False{}) : U64}:
  Equal.trans(U64,Ray.rays(64n+q,occ),Ray.directional(q,4n,occ),
    Tables.slider(U32.from_nat(q),True{},occ,False{}),Keys.bishop(q,occ,bound),
    Equal.sym(U64,Tables.slider(U32.from_nat(q),True{},occ,False{}),
      Ray.directional(q,4n,occ),Slider.square(q,True{},occ,bound)))

def rook_pair(-a: Array<U64>,+q: Nat,side: U32,+occ: U64) -> Type:
  {Chess.attack(3,a,U32.from_nat(q),side,occ) ==
    (a,Tables.slider(U32.from_nat(q),False{},occ,False{})) : Array<U64> & U64}
def bishop_pair(-a: Array<U64>,+q: Nat,side: U32,+occ: U64) -> Type:
  {Chess.attack(2,a,U32.from_nat(q),side,occ) ==
    (a,Tables.slider(U32.from_nat(q),True{},occ,False{})) : Array<U64> & U64}

# Slider dispatch ignores side; the arbitrary U32 side here is intentional.
def rook(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+q: Nat,+side: U32,+occ: U64,
  same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  +bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> Q.rook(c,q,side,occ):
  %Equal.sym(Array<U64>,O.pack(c),Init.run(d,seed,n,extra),same) : rook_pair(_,q,side,occ)
  Equal.trans(Array<U64> & U64,
    Chess.attack(3,Init.run(d,seed,n,extra),U32.from_nat(q),side,occ),
    (Init.run(d,seed,n,extra),Ray.rays(q,occ)),
    (Init.run(d,seed,n,extra),Tables.slider(U32.from_nat(q),False{},occ,False{})),
    M.rook(d,seed,n,extra,q,False{},occ,depth,tables,full,bound),
    Equal.cong(U64,Array<U64> & U64,mask => (Init.run(d,seed,n,extra),mask),
      Ray.rays(q,occ),Tables.slider(U32.from_nat(q),False{},occ,False{}),rook_mask(q,occ,bound)))

def bishop(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+q: Nat,+side: U32,+occ: U64,
  same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  +bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> Q.bishop(c,q,side,occ):
  %Equal.sym(Array<U64>,O.pack(c),Init.run(d,seed,n,extra),same) : bishop_pair(_,q,side,occ)
  Equal.trans(Array<U64> & U64,
    Chess.attack(2,Init.run(d,seed,n,extra),U32.from_nat(q),side,occ),
    (Init.run(d,seed,n,extra),Ray.rays(64n+q,occ)),
    (Init.run(d,seed,n,extra),Tables.slider(U32.from_nat(q),True{},occ,False{})),
    M.bishop(d,seed,n,extra,q,False{},occ,depth,tables,full,bound),
    Equal.cong(U64,Array<U64> & U64,mask => (Init.run(d,seed,n,extra),mask),
      Ray.rays(64n+q,occ),Tables.slider(U32.from_nat(q),True{},occ,False{}),bishop_mask(q,occ,bound)))

def old(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,
  +same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) -> S.old(c,b):
  (rook(c,d,seed,n,extra,K.index(b),Chess.get_turn(b),Chess.occupied(b),
      same,depth,tables,full,K.bound(b,nonempty)),
   bishop(c,d,seed,n,extra,K.index(b),Chess.get_turn(b),Chess.occupied(b),
      same,depth,tables,full,K.bound(b,nonempty)))

def post(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  +same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) -> S.post(c,b,m):
  (rook(c,d,seed,n,extra,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(Chess.make_move(b,m)),
      same,depth,tables,full,K.bound(b,nonempty)),
   bishop(c,d,seed,n,extra,K.index(b),U32.xor(Chess.get_turn(b),1),Chess.occupied(Chess.make_move(b,m)),
      same,depth,tables,full,K.bound(b,nonempty)))

# Each occurrence receives a fresh affine callback; no callback is duplicated.
def lookup(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  +same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) -> S.lookup(c,b,m):
  here => bypass => post(c,d,seed,n,extra,b,m,same,depth,tables,full,nonempty)

def occurrences(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  +same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool}) -> S.lookups(c,b,xs):
  match xs:
    case Nil{}: Unit{}
    case Con{m,rest}:
      (lookup(c,d,seed,n,extra,b,m,same,depth,tables,full,nonempty),
       occurrences(c,d,seed,n,extra,b,rest,same,depth,tables,full,nonempty))
