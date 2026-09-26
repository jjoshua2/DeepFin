import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Flow.bend as F

def transport(-m: Chess.Ply,-a: List<&2,Chess.Ply>,-b: List<&2,Chess.Ply>,e: {a == b : List<&2,Chess.Ply>},p: S.member(m,a)) -> S.member(m,b):
  %e : S.member(m,_)
  p

def cons_new(-m: Chess.Ply,-head: Chess.Ply,-tail: List<&2,Chess.Ply>,p: S.member(m,Con{head,tail}),absent: S.member(m,tail) -> Empty) ->
  {m == head : Chess.Ply}:
  match p:
    case Inl{same}: same
    case Inr{old}: Empty.absurd({m == head : Chess.Ply},absent(old))

def extract(-ok: Bool,-head: Chess.Ply,-tail: List<&2,Chess.Ply>,-out: List<&2,Chess.Ply>,-m: Chess.Ply,
  growth: {out == tail : List<&2,Chess.Ply>} | ({ok == True{} : Bool} & {out == Con{head,tail} : List<&2,Chess.Ply>}),
  here: S.member(m,out),absent: S.member(m,tail) -> Empty) ->
  {ok == True{} : Bool} & {m == head : Chess.Ply}:
  match growth:
    case Inl{same}:
      Empty.absurd({ok == True{} : Bool} & {m == head : Chess.Ply},absent(transport(m,out,tail,same,here)))
    case Inr{proof}:
      (good,same) = proof
      (good,cons_new(m,head,tail,transport(m,out,Con{head,tail},same,here),absent))

def result(-ok: Bool,-head: Chess.Ply,-tail: List<&2,Chess.Ply>,-out: Array<U64> & List<&2,Chess.Ply>,-m: Chess.Ply,
  growth: S.growth(ok,head,tail,out),here: S.member(m,S.moves(out)),absent: S.member(m,tail) -> Empty) ->
  {ok == True{} : Bool} & {m == head : Chess.Ply}:
  extract(ok,head,tail,S.moves(out),m,growth,here,absent)

def first(-ok: Bool,-m: Chess.Ply,-head: Chess.Ply,p: {ok == True{} : Bool} & {m == head : Chess.Ply}) -> {ok == True{} : Bool}:
  (good,same) = p
  good

def second(-ok: Bool,-m: Chess.Ply,-head: Chess.Ply,p: {ok == True{} : Bool} & {m == head : Chess.Ply}) -> {m == head : Chess.Ply}:
  (good,same) = p
  same

def guard(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {S.guard(b,ks) == True{} : Bool}:
  first(S.guard(b,ks),m,S.move(b,ks),result(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail)),m,F.side(b,ks,table,tail),here,absent))

def exact(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {m == S.move(b,ks) : Chess.Ply}:
  second(S.guard(b,ks),m,S.move(b,ks),result(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail)),m,F.side(b,ks,table,tail),here,absent))

def side(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {S.guard(b,ks) == True{} : Bool} & {m == S.move(b,ks) : Chess.Ply}:
  result(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail)),m,F.side(b,ks,table,tail),here,absent)
