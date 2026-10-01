# Internal certificate composition. Public proofs supply actual initialized checks.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S
import ./Route.bend as R

def retain(+reject: Bool,+m: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {Chess.retain_move(reject,m,tail) == S.keep(reject,m,tail) : List<&2,Chess.Ply>}:
  match reject:
    case True{}: {==}
    case False{}: {==}

def start(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+tail: List<&2,Chess.Ply>,+first: Bool,+middle: Bool,
  a: {Chess.in_check(table,b,Chess.get_turn(b)) == (table,first) : Array<U64> & Bool},
  t: {Chess.in_check(table,K.mid(b,white,ks),Chess.get_turn(b)) == (table,middle) : Array<U64> & Bool}) ->
  {Chess.castle_start(b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail,Chess.in_check(table,b,Chess.get_turn(b))) ==
    (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(table,b,Chess.get_turn(b)),(table,first),a) :
    {Chess.castle_start(b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail,_) ==
     (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}
  %Equal.sym(Array<U64> & Bool,Chess.in_check(table,K.mid(b,white,ks),Chess.get_turn(b)),(table,middle),t) :
    {Chess.castle_finish(S.move(white,ks),first,tail,_) ==
     (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}
  Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,xs => (table,xs),
    Chess.retain_move(Bool.or(first,middle),S.move(white,ks),tail),S.keep(Bool.or(first,middle),S.move(white,ks),tail),retain(Bool.or(first,middle),S.move(white,ks),tail))

def producer(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+tail: List<&2,Chess.Ply>,+first: Bool,+middle: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},guard: {E.guard(b,ks) == True{} : Bool},
  a: {Chess.in_check(table,b,Chess.get_turn(b)) == (table,first) : Array<U64> & Bool},
  t: {Chess.in_check(table,K.mid(b,white,ks),Chess.get_turn(b)) == (table,middle) : Array<U64> & Bool}) ->
  {Chess.castle_side(b,ks,(table,tail)) == (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & List<&2,Chess.Ply>,Chess.castle_side(b,ks,(table,tail)),
    Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail),R.selected(b,white,ks,table,tail,turn)) :
    {_ == (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}
  %Equal.sym(Bool,E.guard(b,ks),True{},guard) :
    {Chess.castle_checked(_,table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) == (table,S.keep(Bool.or(first,middle),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}
  start(b,white,ks,table,tail,first,middle,a,t)

def destination(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+tail: List<&2,Chess.Ply>,+last: Bool,
  f: {Chess.in_check(table,K.final(b,white,ks),Chess.get_turn(b)) == (table,last) : Array<U64> & Bool}) ->
  {Chess.filter_step(b,S.move(white,ks),(table,tail)) == (table,S.keep(last,S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(table,K.final(b,white,ks),Chess.get_turn(b)),(table,last),f) :
    {Chess.filter_after(S.move(white,ks),tail,_) == (table,S.keep(last,S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}
  Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,xs => (table,xs),
    Chess.retain_move(last,S.move(white,ks),tail),S.keep(last,S.move(white,ks),tail),retain(last,S.move(white,ks),tail))

def final_list(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+early: Bool,+last: Bool,
  f: {Chess.in_check(table,K.final(b,white,ks),Chess.get_turn(b)) == (table,last) : Array<U64> & Bool}) ->
  {Chess.filter_legal(S.keep(early,S.move(white,ks),Nil{}),b,(table,Nil{})) ==
   (table,S.keep(Bool.or(early,last),S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  match early:
    case True{}: {==}
    case False{}: destination(b,white,ks,table,Nil{},last,f)

def pipeline(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+early: Bool,+last: Bool,
  producer: {Chess.castle_side(b,ks,(table,Nil{})) == (table,S.keep(early,S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>},
  final: {Chess.in_check(table,K.final(b,white,ks),Chess.get_turn(b)) == (table,last) : Array<U64> & Bool}) ->
  {S.run(table,b,ks) == (table,S.keep(Bool.or(early,last),S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & List<&2,Chess.Ply>,Chess.castle_side(b,ks,(table,Nil{})),(table,S.keep(early,S.move(white,ks),Nil{})),producer) :
    {S.filter_result(b,_) == (table,S.keep(Bool.or(early,last),S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>}
  final_list(b,white,ks,table,early,last,final)
