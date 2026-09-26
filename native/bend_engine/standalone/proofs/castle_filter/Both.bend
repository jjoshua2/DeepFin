import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S
import ./Lists.bend as L
import ./Filter.bend as F

def finish_then(+m: Chess.Ply,+start_check: Bool,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: P(m),tail: S.all(P,acc),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.castle_finish(m,start_check,acc,r)):
  (table,transit) = r
  +check = Bool.or(start_check,transit)
  next((table,Chess.retain_move(check,m,acc)),F.retain(check,m,acc,P,head,tail))

def start_then(+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+acc: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: P(Chess.Ply{src,dst,0,2}),tail: S.all(P,acc),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.castle_start(b,src,dst,transit,acc,r)):
  (table,check) = r
  finish_then(Chess.Ply{src,dst,0,2},check,acc,
    Chess.in_check(table,Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b)),P,Q,head,tail,next)

def checked_then(+ok: Bool,table: Array<U64>,+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+acc: List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: {ok == True{} : Bool} -> P(Chess.Ply{src,dst,0,2}),tail: S.all(P,acc),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.castle_checked(ok,table,b,src,dst,transit,acc)):
  match ok:
    case False{}: next((table,acc),tail)
    case True{}: start_then(b,src,dst,transit,acc,Chess.in_check(table,b,Chess.get_turn(b)),P,Q,head({==}),tail,next)

def side_then(+b: Chess.Board,+ks: Bool,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  head: {E.guard(b,ks) == True{} : Bool} -> P(E.move(b,ks)),tail: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(Chess.castle_side(b,ks,r)):
  (table,acc) = r
  checked_then(E.guard(b,ks),table,b,E.src(b),E.dst(b,ks),E.transit(b,ks),acc,P,Q,head,tail,next)

def both_then(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  -P: @-m: Chess.Ply -> Type,-Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,
  king: {E.guard(b,True{}) == True{} : Bool} -> P(E.move(b,True{})),
  queen: {E.guard(b,False{}) == True{} : Bool} -> P(E.move(b,False{})),
  tail: S.all(P,E.moves(r)),next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(P,E.moves(r)) -> Q(r)) -> Q(S.both(b,r)):
  side_then(b,True{},r,P,out => Q(Chess.castle_side(b,False{},out)),king,tail,
    out => p => side_then(b,False{},out,P,Q,queen,p,next))

def origin(+b: Chess.Board,table: Array<U64>,+tail: List<&2,Chess.Ply>) -> S.all(m => S.origin(b,tail,m),E.moves(S.both(b,(table,tail)))):
  both_then(b,(table,tail),m => S.origin(b,tail,m),out => S.all(m => S.origin(b,tail,m),E.moves(out)),
    e => Inr{Inl{(e,{==})}},e => Inr{Inr{(e,{==})}},L.as_origin(tail,b,tail,L.reflexive(tail)),out => p => p)

def filtered_origin(+b: Chess.Board,table: Array<U64>,+tail: List<&2,Chess.Ply>) -> S.all(m => S.origin(b,tail,m),E.moves(S.filtered(b,(table,tail)))):
  both_then(b,(table,tail),m => S.origin(b,tail,m),out => S.all(m => S.origin(b,tail,m),E.moves(Chess.filter_prepare(b,out))),
    e => Inr{Inl{(e,{==})}},e => Inr{Inr{(e,{==})}},L.as_origin(tail,b,tail,L.reflexive(tail)),out => p => F.prepare(b,out,m => S.origin(b,tail,m),p))

def classified_then(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  -Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,p: S.all(S.not_castle,E.moves(r)),
  next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(m => S.classified(b,m),E.moves(r)) -> Q(r)) -> Q(S.filtered(b,r)):
  (table,+tail) = r
  both_then(b,(table,tail),m => S.classified(b,m),out => Q(Chess.filter_prepare(b,out)),
    e => Inr{Inl{(e,{==})}},e => Inr{Inr{(e,{==})}},L.as_classified(tail,b,p),
    out => ps => F.prepare_then(b,out,m => S.classified(b,m),Q,ps,next))

def filtered_classified(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,p: S.all(S.not_castle,E.moves(r))) -> S.all(m => S.classified(b,m),E.moves(S.filtered(b,r))):
  classified_then(b,r,out => S.all(m => S.classified(b,m),E.moves(out)),p,out => ps => ps)
