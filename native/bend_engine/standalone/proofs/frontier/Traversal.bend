# Induction follows the actual parser, with no supplied trace or fresh-square oracle.
import Base
import ./Spec.bend as S
import ./Step.bend as Step
import ./Logic.bend as L
import ./Facts.bend as Facts
import ./Fresh.bend as Fresh
import ../parser/Spec.bend as PS
import ../parser/Actual.bend as Parent
import ../parser/Typed.bend as Typed
import ../board/Spec.bend as B
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess

def all(+s: String,+l: P.Layout,good: {S.safe(l) == True{} : Bool}) ->
  {S.safe(P.layout(s,l)) == True{} : Bool}:
  match s:
    case SNil{}: good
    case SCon{Chr{+c},rest}: all(rest,P.layout_char(c,l),Step.one(c,l,good))

def initialized(+s: String) -> {S.safe(P.layout(s,S.initial())) == True{} : Bool}:
  all(s,S.initial(),{==})

def when_live(+l: P.Layout,good: {S.safe(l) == True{} : Bool},live: {PS.live(l) == True{} : Bool}) ->
  {S.ready(l) == True{} : Bool}:
  match l:
    case P.Layout{b,f,r,True{}}: good
    case P.Layout{b,f,r,False{}}: L.impossible({S.state(b,f,r) == True{} : Bool},live)

def accepted_prefix(+prefix: String,+suffix: String,
  accepted: {PS.accepted(P.layout_result(P.layout(String.append(prefix,suffix),S.initial()))) == True{} : Bool}) ->
  {S.ready(P.layout(prefix,S.initial())) == True{} : Bool}:
  when_live(P.layout(prefix,S.initial()),initialized(prefix),Parent.accepted_prefix(prefix,suffix,S.initial(),accepted))

def fresh_cert(+b: Chess.Board,+f: U32,+r: U32,
  good: {S.state(b,f,r) == True{} : Bool},cert: Facts.piece_statement(f,r)) ->
  {Bool.and(B.fresh(b,S.square(f,r)),Nat.is_lt(U32.to_nat(S.square(f,r)),64n)) == True{} : Bool}:
  (next,(sub,(current,(apart,addr)))) = cert
  L.both(B.fresh(b,S.square(f,r)),Nat.is_lt(U32.to_nat(S.square(f,r)),64n),
    Fresh.narrow(b,U64.bit(U32.to_nat(S.square(f,r))),S.pending(f,r),Step.empty_ahead(b,f,r,good),current),addr)

def typed_fresh(+l: P.Layout,+pc: B.Piece,+white: Bool,
  good: {S.safe(l) == True{} : Bool},
  live: {PS.live(P.layout(S.typed(pc,white),l)) == True{} : Bool}) ->
  {S.fresh_target(l) == True{} : Bool}:
  match l:
    case P.Layout{+b,+f,+r,+v}:
      +both = Equal.trans(Bool,Bool.and(v,U32.is_lt(f,8)),PS.live(P.layout(S.typed(pc,white),P.Layout{b,f,r,v})),True{},
        Equal.sym(Bool,PS.live(P.layout(S.typed(pc,white),P.Layout{b,f,r,v})),Bool.and(v,U32.is_lt(f,8)),
          Equal.cong(P.Layout,Bool,x => PS.live(x),P.layout(S.typed(pc,white),P.Layout{b,f,r,v}),
            P.Layout{P.put(B.tag(pc),white,S.square(f,r),b),U32.inc(f),r,Bool.and(v,U32.is_lt(f,8))},Typed.step(pc,white,b,f,r,v))),live)
      +ready = when_live(P.Layout{b,f,r,v},good,L.left(v,U32.is_lt(f,8),both))
      fresh_cert(b,f,r,ready,Facts.piece(f,r,Step.bounded(b,f,r,ready),L.right(v,U32.is_lt(f,8),both)))

def initialized_typed(+prefix: String,+pc: B.Piece,+white: Bool,
  live: {PS.live(P.layout(S.typed(pc,white),P.layout(prefix,S.initial()))) == True{} : Bool}) ->
  {S.fresh_target(P.layout(prefix,S.initial())) == True{} : Bool}:
  typed_fresh(P.layout(prefix,S.initial()),pc,white,initialized(prefix),live)

def finish(+ok: Bool,+b: Chess.Board,good: {B.valid(b) == True{} : Bool}) ->
  {S.result_valid(P.board_when(ok,b)) == True{} : Bool}:
  match ok:
    case False{}: {==}
    case True{}: good

def result(+l: P.Layout,good: {S.safe(l) == True{} : Bool}) ->
  {S.result_valid(P.layout_result(l)) == True{} : Bool}:
  match l:
    case P.Layout{b,f,r,False{}}: {==}
    case P.Layout{+b,+f,+r,True{}}:
      finish(Bool.and(U32.is_eq(f,8),U32.is_zero(r)),b,Step.consistent(b,f,r,good))

def initialized_result(+s: String) ->
  {S.result_valid(P.layout_result(P.layout(s,S.initial()))) == True{} : Bool}:
  result(P.layout(s,S.initial()),initialized(s))
