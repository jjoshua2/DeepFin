# Exact expansion copies only Data certificates, never affine proof functions.
import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../bitboard_membership/AllEmittedSound.bend as All
import ../destination_factorization/Spec.bend as F
import ../destination_factorization/Emission.bend as Old
import ../table_preservation/Core.bend as T
import ../initialized_pawn_double/Spec.bend as D
import ../../../legal_probe/Chess.bend as Chess

def append(-P: Chess.Ply -> Type,+xs: List<&2,Chess.Ply>,+ys: List<&2,Chess.Ply>,
  px: C.every(P,xs),py: C.every(P,ys)) -> C.every(P,List.append(&2,Chess.Ply,xs,ys)):
  match xs:
    case Nil{}: py
    case Con{h,t}:
      (head,tail)=px
      (head,append(P,t,ys,tail,py))

def choose(+c: O.Cells,+b: Chess.Board,+s: U32,+d: U32,flag: Bool,
  hit: {U64.test_bit(D.mask(Chess.pawn_targets(O.pack(c),b,s)),U32.to_nat(d)) == True{} : Bool}) -> S.gate(flag,O.pack(c),b,s,d):
  match flag:
    case False{}: Unit{}
    case True{}: hit

def target(+c: O.Cells,+b: Chess.Board,+s: U32,+d: U32,+bb: U64,actualpawn: Bool,
  samepawn: {S.pawn(b,s) == actualpawn : Bool},mask: S.dispatch(c,b,s,bb,actualpawn),
  hit: {U64.test_bit(bb,U32.to_nat(d)) == True{} : Bool}) -> S.target(O.pack(c),b,s,d):
  match actualpawn:
    case False{}:
      %Equal.sym(Bool,S.pawn(b,s),False{},samepawn) : S.gate(Bool.and(_,S.trigger(s,d)),O.pack(c),b,s,d)
      Unit{}
    case True{}:
      %Equal.sym(Bool,S.pawn(b,s),True{},samepawn) : S.gate(Bool.and(_,S.trigger(s,d)),O.pack(c),b,s,d)
      choose(c,b,s,d,S.trigger(s,d),
        Equal.trans(Bool,U64.test_bit(D.mask(Chess.pawn_targets(O.pack(c),b,s)),U32.to_nat(d)),U64.test_bit(bb,U32.to_nat(d)),True{},
          Equal.cong(U64,Bool,x => U64.test_bit(x,U32.to_nat(d)),D.mask(Chess.pawn_targets(O.pack(c),b,s)),bb,
            Equal.sym(U64,bb,D.mask(Chess.pawn_targets(O.pack(c),b,s)),mask)),hit))

def mark(+c: O.Cells,+b: Chess.Board,+s: U32,+d: U32,p: U32,f: U32,+bb: U64,+actualpawn: Bool,
  bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},samepawn: {S.pawn(b,s) == actualpawn : Bool},mask: S.dispatch(c,b,s,bb,actualpawn),
  hit: {U64.test_bit(bb,U32.to_nat(d)) == True{} : Bool}) -> S.cells(c,b,Chess.Ply{s,d,p,f}):
  (bound,target(c,b,s,d,bb,actualpawn,samepawn,mask,hit))

def block(+c: O.Cells,+b: Chess.Board,promote: Bool,+s: U32,+d: U32,tag: U32,+bb: U64,+actualpawn: Bool,
  +bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},+samepawn: {S.pawn(b,s) == actualpawn : Bool},+mask: S.dispatch(c,b,s,bb,actualpawn),
  +hit: {U64.test_bit(bb,U32.to_nat(d)) == True{} : Bool}) -> C.every(m => S.cells(c,b,m),F.block(promote,s,d,tag)):
  match promote:
    case False{}: (mark(c,b,s,d,0,tag,bb,actualpawn,bound,samepawn,mask,hit),Unit{})
    case True{}:
      (mark(c,b,s,d,1,0,bb,actualpawn,bound,samepawn,mask,hit),(mark(c,b,s,d,2,0,bb,actualpawn,bound,samepawn,mask,hit),
        (mark(c,b,s,d,3,0,bb,actualpawn,bound,samepawn,mask,hit),(mark(c,b,s,d,4,0,bb,actualpawn,bound,samepawn,mask,hit),Unit{}))))

def expand(+c: O.Cells,+b: Chess.Board,+keys: List<&2,U32>,+bb: U64,+s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  +set: {All.all_bits_set(keys,bb) == True{} : Bool},+bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  +samepawn: {S.pawn(b,s) == pawn : Bool},+mask: S.dispatch(c,b,s,bb,pawn),good: C.every(m => S.cells(c,b,m),tail)) -> C.every(m => S.cells(c,b,m),F.expand(keys,s,pawn,ep,tail)):
  match keys:
    case Nil{}: good
    case Con{+d,+rest}:
      +head=All.bool_and_true_left(U64.test_bit(bb,U32.to_nat(d)),All.all_bits_set(rest,bb),set)
      +later=All.bool_and_true_right(U64.test_bit(bb,U32.to_nat(d)),All.all_bits_set(rest,bb),set)
      append(m => S.cells(c,b,m),F.block(F.promotes(pawn,d),s,d,F.flag(pawn,d,ep)),F.expand(rest,s,pawn,ep,tail),
        block(c,b,F.promotes(pawn,d),s,d,F.flag(pawn,d,ep),bb,pawn,bound,samepawn,mask,head),
        expand(c,b,rest,bb,s,pawn,ep,tail,later,bound,samepawn,mask,good))

def destinations(+c: O.Cells,+b: Chess.Board,+bb: U64,+s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},samepawn: {S.pawn(b,s) == pawn : Bool},mask: S.dispatch(c,b,s,bb,pawn),good: C.every(m => S.cells(c,b,m),tail)) ->
  C.every(m => S.cells(c,b,m),Chess.destinations(64n,U64.is_zero(bb),bb,s,pawn,ep,tail)):
  %Equal.sym(List<&2,Chess.Ply>,Chess.destinations(64n,U64.is_zero(bb),bb,s,pawn,ep,tail),
    F.expand(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),s,pawn,ep,tail),
    Old.destinations(64n,U64.is_zero(bb),bb,s,pawn,ep,tail)) : C.every(m => S.cells(c,b,m),_)
  expand(c,b,Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),bb,s,pawn,ep,tail,All.actual_bit_squares_all_set(bb),bound,samepawn,mask,good)

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+c: O.Cells,+b: Chess.Board,
  +s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  same: {T.table(U64,r) == O.pack(c) : Array<U64>},bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  samepawn: {S.pawn(b,s) == pawn : Bool},mask: S.dispatch(c,b,s,D.mask(r),pawn),good: C.every(m => S.cells(c,b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> {T.table(List<&2,Chess.Ply>,r) == O.pack(c) : Array<U64>} -> C.every(m => S.cells(c,b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(s,pawn,ep,tail,r)):
  (table,+targets)=r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,s,pawn,ep,tail)),same,
    destinations(c,b,targets,s,pawn,ep,tail,bound,samepawn,mask,good))
