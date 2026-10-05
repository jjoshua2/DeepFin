import Base
import ./Algebra.bend as A
import ../filter_ray_exclusion/Filter.bend as F
import ../prepared_ray_coverage/Query.bend as Q
import ../storage/Observe.bend as O
import ../castle_safety/Spec.bend as S
import ../accepted_castle/Spec.bend as Prepared
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Rays.bend as R
import ../ray/Trace.bend as Trace
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

# Production uses seven steps, both flags False, and the zero accumulator.
def mask(+b: Chess.Board,+q: Nat,dir: Nat) -> U64:
  Tables.ray(7n,False{},U32.from_nat(q),U32.from_nat(dir),Chess.occupied(b),False{},U64.zero())

def ray_known(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,
  +qb: {Nat.is_lt(q,64n) == True{} : Bool},
  +chosen: {Q.king(b) == U32.from_nat(q) : U32},
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  +bypass: {Chess.filter_requires(S.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})) : Array<U64> & U64},+dir: Nat,
  +db: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  L.subset(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,dir),mask(b,q,dir)):
  F.actual_ray(7n,U32.from_nat(q),U32.from_nat(dir),R.path(7n,q,dir),b,src,dst,Prepared.rays(c,b),U64.zero(),
    Trace.finite(q,dir,qb,db),bound,owned,bypass,Q.coverage(c,b,q,dir,qb,db,chosen,old_r,old_d))

# Combine the four actual rook rays.
def rook(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,
  +qb: {Nat.is_lt(q,64n) == True{} : Bool},
  +chosen: {Q.king(b) == U32.from_nat(q) : U32},
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  +bypass: {Chess.filter_requires(S.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})) : Array<U64> & U64}) ->
  L.subset(Tables.slider(U32.from_nat(q),False{},Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{}),
    Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})):
  A.or_mono(U64.or(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,0n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,1n)),U64.or(mask(b,q,0n),mask(b,q,1n)),
    U64.or(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,2n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,3n)),U64.or(mask(b,q,2n),mask(b,q,3n)),
    A.or_mono(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,0n),mask(b,q,0n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,1n),mask(b,q,1n),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,0n,{==}),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,1n,{==})),
    A.or_mono(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,2n),mask(b,q,2n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,3n),mask(b,q,3n),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,2n,{==}),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,3n,{==})))

# Combine the four actual bishop rays.
def bishop(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,
  +qb: {Nat.is_lt(q,64n) == True{} : Bool},
  +chosen: {Q.king(b) == U32.from_nat(q) : U32},
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  +bypass: {Chess.filter_requires(S.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  +old_r: {Chess.attack(3,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),False{},Chess.occupied(b),False{})) : Array<U64> & U64},
  +old_d: {Chess.attack(2,O.pack(c),U32.from_nat(q),Chess.get_turn(b),Chess.occupied(b)) ==
    (O.pack(c),Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})) : Array<U64> & U64}) ->
  L.subset(Tables.slider(U32.from_nat(q),True{},Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{}),
    Tables.slider(U32.from_nat(q),True{},Chess.occupied(b),False{})):
  A.or_mono(U64.or(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,4n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,5n)),U64.or(mask(b,q,4n),mask(b,q,5n)),
    U64.or(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,6n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,7n)),U64.or(mask(b,q,6n),mask(b,q,7n)),
    A.or_mono(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,4n),mask(b,q,4n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,5n),mask(b,q,5n),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,4n,{==}),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,5n,{==})),
    A.or_mono(mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,6n),mask(b,q,6n),mask(Chess.make_move(b,Chess.Ply{src,dst,0,0}),q,7n),mask(b,q,7n),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,6n,{==}),ray_known(c,b,q,src,dst,qb,chosen,bound,owned,bypass,old_r,old_d,7n,{==})))
