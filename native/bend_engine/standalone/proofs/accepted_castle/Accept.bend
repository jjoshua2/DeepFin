# Extract the check decision for retained candidates, not just list provenance.
import Base
import ./Spec.bend as S
import ./Filter.bend as Filter
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as L
import ../castle_safety/Mask.bend as Mask
import ../storage/Observe.bend as O
import ../u64/Words.bend as W
import ../../../legal_probe/Chess.bend as Chess

def retained(-P: Chess.Ply -> Type,reject: Bool,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  added: {reject == False{} : Bool} -> P(m),good: C.every(P,acc)) ->
  C.every(P,S.retain(reject,m,acc)):
  match reject:
    case True{}: good
    case False{}: (added({==}),good)

def full(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,+acc: List<&2,Chess.Ply>,
  good: C.every(m => S.safe(c,b,m),acc)) -> C.every(m => S.safe(c,b,m),S.full(xs,c,b,acc)):
  match xs:
    case Nil{}: good
    case Con{+m,rest}:
      full(rest,c,b,S.retain(S.checked(c,b,m),m,acc),
        retained(x => S.safe(c,b,x),S.checked(c,b,m),m,acc,e => e,good))

def fast_insert(required: Bool,+c: O.Cells,+b: Chess.Board,+sensitive: U64,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  same: {Chess.filter_requires(sensitive,m) == required : Bool},
  good: C.every(x => S.required_safe(c,b,sensitive,x),acc)) ->
  C.every(x => S.required_safe(c,b,sensitive,x),S.fast_step(required,c,b,m,acc)):
  match required:
    case True{}:
      retained(x => S.required_safe(c,b,sensitive,x),S.checked(c,b,m),m,acc,e => r => e,good)
    case False{}:
      (r => W.false_true(S.safe(c,b,m),Equal.trans(Bool,False{},Chess.filter_requires(sensitive,m),True{},
        Equal.sym(Bool,Chess.filter_requires(sensitive,m),False{},same),r)),good)

def fast(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,+sensitive: U64,+acc: List<&2,Chess.Ply>,
  good: C.every(m => S.required_safe(c,b,sensitive,m),acc)) ->
  C.every(m => S.required_safe(c,b,sensitive,m),S.fast(xs,c,b,sensitive,acc)):
  match xs:
    case Nil{}: good
    case Con{+m,rest}:
      fast(rest,c,b,sensitive,S.fast_step(Chess.filter_requires(sensitive,m),c,b,m,acc),
        fast_insert(Chess.filter_requires(sensitive,m),c,b,sensitive,m,acc,{==},good))

def chosen(check: Bool,+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+rays: U64,+m: Chess.Ply,
  castle: C.castle(b,m),here: E.member(m,S.choose(check,c,b,xs,rays))) -> S.safe(c,b,m):
  match check:
    case True{}:
      L.at(x => S.safe(c,b,x),S.full(xs,c,b,Nil{}),m,full(xs,c,b,Nil{},Unit{}),here)
    case False{}:
      +sensitive = U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b)))
      L.at(x => S.required_safe(c,b,sensitive,x),S.fast(xs,c,b,sensitive,Nil{}),m,
        fast(xs,c,b,sensitive,Nil{},Unit{}),here)(Mask.certificate(b,m,rays,castle))

def transport(-x: Array<U64> & List<&2,Chess.Ply>,-y: Array<U64> & List<&2,Chess.Ply>,-m: Chess.Ply,
  e: {x == y : Array<U64> & List<&2,Chess.Ply>},here: E.member(m,E.moves(x))) -> E.member(m,E.moves(y)):
  %e : E.member(m,E.moves(_))
  here

def prepared(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  castle: C.castle(b,m),here: E.member(m,E.moves(Chess.filter_prepare(b,(O.pack(c),xs))))) -> S.safe(c,b,m):
  chosen(S.initial(c,b),c,b,xs,S.rays(c,b),m,castle,
    transport(Chess.filter_prepare(b,(O.pack(c),xs)),(O.pack(c),S.prepared(c,b,xs)),m,
      Filter.prepare(c,b,xs),here))
