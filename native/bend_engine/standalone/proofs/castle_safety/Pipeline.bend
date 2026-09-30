# Complete actual-generator equality to the explicit force-castling-check audit.
# Preserve the original five accepted contracts; use actual-array continuations.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Scan.bend as Scan
import ../castle_chain/Sides.bend as Sides
import ../castle_chain/Lists.bend as Lists
import ./Spec.bend as S
import ./Mask.bend as Mask
import ./Step.bend as Step
import ./Forced.bend as Force

def choose(+b: Chess.Board,+m: Chess.Ply,+rays: U64,p: C.classified(b,m)) ->
  {Force.required(b,m,rays) == Chess.filter_requires(S.sensitive(b,rays),m) : Bool}:
  match p:
    case Inl{ordinary}:
      Equal.cong(Bool,Bool,x => Bool.or(x,Chess.filter_requires(S.sensitive(b,rays),m)),
        U32.is_eq(C.flag(m),2),False{},ordinary)
    case Inr{castle}:
      %Equal.sym(Bool,Chess.filter_requires(S.sensitive(b,rays),m),True{},Mask.certificate(b,m,rays,castle)) :
        {Bool.or(U32.is_eq(C.flag(m),2),_) == _ : Bool}
      Mask.or_right(U32.is_eq(C.flag(m),2),True{},{==})

def fast(+xs: List<&2,Chess.Ply>,+b: Chess.Board,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>,good: C.every(m => C.classified(b,m),xs)) ->
  {Force.fast(xs,b,rays,r) == Chess.filter_fast(xs,b,S.sensitive(b,rays),r) : Array<U64> & List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      (pm,prest) = good
      %Equal.sym(Bool,Force.required(b,m,rays),Chess.filter_requires(S.sensitive(b,rays),m),choose(b,m,rays,pm)) :
        {Force.fast(rest,b,rays,Chess.filter_fast_step(_,b,m,r)) == Chess.filter_fast(rest,b,S.sensitive(b,rays),Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r)) : Array<U64> & List<&2,Chess.Ply>}
      fast(rest,b,rays,Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r),prest)

def blockers(+b: Chess.Board,+moves: List<&2,Chess.Ply>,r: Array<U64> & U64,good: C.every(m => C.classified(b,m),moves)) ->
  {Force.blockers(b,moves,r) == Chess.filter_blockers(b,moves,r) : Array<U64> & List<&2,Chess.Ply>}:
  (table,rays) = r
  fast(moves,b,rays,(table,Nil{}),good)

def checked(+b: Chess.Board,+moves: List<&2,Chess.Ply>,r: Array<U64> & Bool,good: C.every(m => C.classified(b,m),moves)) ->
  {Force.checked(b,moves,r) == Chess.filter_checked(b,moves,r) : Array<U64> & List<&2,Chess.Ply>}:
  (table,check) = r
  match check:
    case True{}: {==}
    case False{}:
      sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
      blockers(b,moves,Chess.attack(4,table,sq,Chess.get_turn(b),Chess.occupied(b)),good)

def prepare(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: C.every(m => C.classified(b,m),E.moves(r))) ->
  {Force.prepare(b,r) == Chess.filter_prepare(b,r) : Array<U64> & List<&2,Chess.Ply>}:
  (table,+moves) = r
  checked(b,moves,Chess.in_check(table,b,Chess.get_turn(b)),good)

def classified_suffix(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,good: C.every(m => C.classified(b,m),E.moves(r))) ->
  {Force.suffix(b,r) == C.suffix(b,r) : Array<U64> & List<&2,Chess.Ply>}:
  Sides.side(m => C.classified(b,m),z => {Force.prepare(b,Chess.castle_side(b,False{},z)) == Chess.filter_prepare(b,Chess.castle_side(b,False{},z)) : Array<U64> & List<&2,Chess.Ply>},
    b,True{},r,good,g => Inr{Inl{(g,{==})}},
    z => pz => Sides.side(m => C.classified(b,m),z => {Force.prepare(b,z) == Chess.filter_prepare(b,z) : Array<U64> & List<&2,Chess.Ply>},
      b,False{},z,pz,g => Inr{Inr{(g,{==})}},z => pz => prepare(b,z,pz)))

def from_scan(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,plain: C.every(C.ordinary,E.moves(r))) ->
  {Force.suffix(b,r) == C.suffix(b,r) : Array<U64> & List<&2,Chess.Ply>}:
  (table,+xs) = r
  classified_suffix(b,(table,xs),Lists.lift(b,xs,plain))

def legal(table: Array<U64>,+b: Chess.Board) ->
  {Force.legal_moves(table,b) == Chess.legal_moves(table,b) : Array<U64> & List<&2,Chess.Ply>}:
  Scan.initial(z => {Force.suffix(b,z) == C.suffix(b,z) : Array<U64> & List<&2,Chess.Ply>},table,b,z => pz => from_scan(b,z,pz))
