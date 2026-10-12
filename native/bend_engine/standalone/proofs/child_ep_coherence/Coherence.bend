# Actual next-side EP coherence; the parent table is the exact initialized array.
import Base
import ./Spec.bend as S
import ./Ranks.bend as Ranks
import ./Closed.bend as Closed
import ./Move.bend as Move
import ./Fields.bend as Fields
import ./Post.bend as Post
import ../candidate_bypass_classification/Fields.bend as Shape
import ../child_metadata/Range.bend as R
import ../initialized_pawn_double/Spec.bend as D
import ../pawn_ep_occurrence/Bridge.bend as Prior
import ../bitboard_membership/AllEmittedSound.bend as All
import ../castle_emission/Spec.bend as E
import ../attack_geometry/Initialized.bend as I
import ../u64/Words.bend as Old
import ../move_update/Core.bend as Core
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def valid(+b: C.Board) -> Bool: Position.valid_ep(C.get_ep(b),b)
def child(+b: C.Board,+m: C.Ply) -> Bool: valid(C.make_move(b,m))

def canonical_white(white: Bool) -> {U32.is_eq(Bool.to_u32(white),1) == white : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def white(+b: C.Board,+w: Bool,canonical: {C.get_turn(b) == Bool.to_u32(w) : U32}) -> {D.white(b) == w : Bool}:
  Equal.trans(Bool,D.white(b),U32.is_eq(Bool.to_u32(w),1),w,Equal.cong(U32,Bool,x => U32.is_eq(x,1),C.get_turn(b),Bool.to_u32(w),canonical),canonical_white(w))

def not_true(a: Bool,yes: {Bool.not(a) == True{} : Bool}) -> {a == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: Old.false_true({True{} == False{} : Bool},yes)

def first_empty(+b: C.Board,+s: U32,guard: {D.empty2(b,s) == True{} : Bool}) -> {U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,D.white(b)))) == False{} : Bool}:
  not_true(U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,D.white(b)))),
    All.bool_and_true_right(U32.is_lt(D.first(s,D.white(b)),64),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,D.white(b))))),
      All.bool_and_true_left(D.empty1(b,s),Bool.and(D.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.second(s,D.white(b)))))),guard)))

def start(+b: C.Board,+s: U32,guard: {D.empty2(b,s) == True{} : Bool}) -> {D.rank(b,s) == True{} : Bool}:
  All.bool_and_true_left(D.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.second(s,D.white(b))))),All.bool_and_true_right(D.empty1(b,s),Bool.and(D.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.second(s,D.white(b)))))),guard))

def finite(+b: C.Board,+s: U32,+ep: Bool,+w: Bool,below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},rank: {Ranks.start_rank(s,w) == True{} : Bool},empty: {U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))) == False{} : Bool}) -> {Position.valid_ep(Move.midpoint(s,D.second(s,w)),Move.next(b,s,D.second(s,w),ep,w)) == True{} : Bool}:
  Closed.finite(b,s,ep,w,below,rank,empty)

def raw(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+w: Bool,+canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},double: D.double(b,s,d)) -> {Position.valid_ep(Move.midpoint(s,d),Move.next(b,s,d,ep,w)) == True{} : Bool}:
  (+guard,same)=double
  +turn=white(b,w,canonical)
  +destination=Equal.trans(U32,d,D.second(s,D.white(b)),D.second(s,w),same,Equal.cong(Bool,U32,x => D.second(s,x),D.white(b),w,turn))
  Equal.trans(Bool,Position.valid_ep(Move.midpoint(s,d),Move.next(b,s,d,ep,w)),Position.valid_ep(Move.midpoint(s,D.second(s,w)),Move.next(b,s,D.second(s,w),ep,w)),True{},
    Equal.cong(U32,Bool,x => Position.valid_ep(Move.midpoint(s,x),Move.next(b,s,x,ep,w)),d,D.second(s,w),destination),
    finite(b,s,ep,w,below,
      Equal.trans(Bool,Ranks.start_rank(s,w),D.rank(b,s),True{},Equal.cong(Bool,Bool,x => Ranks.start_rank(s,x),w,D.white(b),Equal.sym(Bool,D.white(b),w,turn)),start(b,s,guard)),
      Equal.trans(Bool,U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,w))),U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,D.white(b)))),False{},Equal.cong(Bool,Bool,x => U64.test_bit(C.occupied(b),U32.to_nat(D.first(s,x))),w,D.white(b),Equal.sym(Bool,D.white(b),w,turn)),first_empty(b,s,guard))))

def next_ep(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+w: Bool) -> {C.get_ep(Move.next(b,s,d,ep,w)) == Move.midpoint(s,d) : U32}:
  match b:
    case C.Board{p,n,bi,r,q,k,wa,bl,t,rr,e}: {==}

def actual(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+w: Bool,+canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},double: D.double(b,s,d),trigger: {Prior.ep_trigger(b,C.Ply{s,d,0,Bool.to_u32(ep)}) == True{} : Bool}) -> {child(b,C.Ply{s,d,0,Bool.to_u32(ep)}) == True{} : Bool}:
  Equal.trans(Bool,child(b,C.Ply{s,d,0,Bool.to_u32(ep)}),valid(Move.next(b,s,d,ep,w)),True{},
    Equal.cong(C.Board,Bool,x => valid(x),C.make_move(b,C.Ply{s,d,0,Bool.to_u32(ep)}),Move.next(b,s,d,ep,w),Move.actual(b,s,d,ep,w,trigger,canonical)),
    Equal.trans(Bool,valid(Move.next(b,s,d,ep,w)),Position.valid_ep(Move.midpoint(s,d),Move.next(b,s,d,ep,w)),True{},
      Equal.cong(U32,Bool,x => Position.valid_ep(x,Move.next(b,s,d,ep,w)),C.get_ep(Move.next(b,s,d,ep,w)),Move.midpoint(s,d),next_ep(b,s,d,ep,w)),raw(b,s,d,ep,w,canonical,below,double)))

def goal(+b: C.Board,-m: C.Ply,+w: Bool) -> Type:
  R.bounded(m) -> D.double(b,R.src(m),S.dst(m)) -> {Prior.ep_trigger(b,m) == True{} : Bool} -> {child(b,m) == True{} : Bool}

def plain(+b: C.Board,+m: C.Ply,+w: Bool,+canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},shape: Shape.plain(m)) -> goal(b,m,w):
  (+s,(+d,same))=shape
  %Equal.sym(C.Ply,m,C.Ply{s,d,0,0},same) : goal(b,_,w)
  below => double => trigger => actual(b,s,d,False{},w,canonical,below,double,trigger)

def ep(+b: C.Board,+m: C.Ply,+w: Bool,+canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},shape: Shape.ep(m)) -> goal(b,m,w):
  (+s,(+d,same))=shape
  %Equal.sym(C.Ply,m,C.Ply{s,d,0,1},same) : goal(b,_,w)
  below => double => trigger => actual(b,s,d,True{},w,canonical,below,double,trigger)

def ordinary(+b: C.Board,+m: C.Ply,+w: Bool,+canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},shape: S.ordinary(m)) -> goal(b,m,w):
  match shape:
    case Inl{p}: plain(b,m,w,canonical,p)
    case Inr{e}: ep(b,m,w,canonical,e)

def qualified(+b: C.Board,+m: C.Ply,+w: Bool,canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},trigger: {Prior.ep_trigger(b,m) == True{} : Bool},cert: R.bounded(m) & (D.double(b,R.src(m),S.dst(m)) & S.ordinary(m))) -> {child(b,m) == True{} : Bool}:
  (bound,(double,shape))=cert
  ordinary(b,m,w,canonical,shape)(bound)(double)(trigger)

def initialized(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+w: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> {child(b,m) == True{} : Bool}:
  qualified(b,m,w,canonical,trigger,Fields.initialized(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,here))

def no_ep(+b: C.Board,m: C.Ply,no: {Prior.ep_trigger(b,m) == False{} : Bool}) -> {child(b,m) == True{} : Bool}:
  match m:
    case C.Ply{+s,+d,+p,+f}:
      %Equal.sym(U32,C.get_ep(C.make_move(b,C.Ply{s,d,p,f})),64,
        %Equal.sym(Bool,Prior.ep_trigger(b,C.Ply{s,d,p,f}),False{},no) : {Bool.pick(U32,_,Move.midpoint(s,d),64) == 64 : U32}
        {==}) : {Position.valid_ep(_,C.make_move(b,C.Ply{s,d,p,f})) == True{} : Bool}
      {==}

def branch(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+w: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},flag: Bool,same: {Prior.ep_trigger(b,m) == flag : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> {child(b,m) == True{} : Bool}:
  match flag:
    case False{}: no_ep(b,m,same)
    case True{}: initialized(depth,seed,n,key,at,extra,b,m,w,depth_ok,full,canonical,same,here)

def total(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,+w: Bool,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},canonical: {C.get_turn(b) == Bool.to_u32(w) : U32},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> {child(b,m) == True{} : Bool}:
  branch(depth,seed,n,key,at,extra,b,m,w,depth_ok,full,canonical,Prior.ep_trigger(b,m),{==},here)
