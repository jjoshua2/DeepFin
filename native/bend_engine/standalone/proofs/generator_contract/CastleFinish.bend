# Fold both candidate wings through the actual destination check.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../castle_sequence/Spec.bend as S
import ../castle_sequence/Checks.bend as Checks
import ../initialized_attacks/Build.bend as Init
import ./CastleSpec.bend as P

def after(table: Array<U64>,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,check: Bool) ->
  {Chess.filter_after(m,acc,(table,check)) ==
    (table,S.select(check,m,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  match check:
    case False{}: {==}
    case True{}: {==}

def step(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,+check: Bool,
  final: {Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)) == (table,check) : Array<U64> & Bool}) ->
  {Chess.filter_step(b,m,(table,acc)) == (table,S.select(check,m,acc)) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & Bool,
    Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)),(table,check),final) :
    {Chess.filter_after(m,acc,_) == (table,S.select(check,m,acc)) : Array<U64> & List<&2,Chess.Ply>}
  after(table,m,acc,check)

def advance(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+tail: List<&2,Chess.Ply>,+acc: List<&2,Chess.Ply>,
  start: Bool,transit: Bool,+check: Bool,
  final: {Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)) == (table,check) : Array<U64> & Bool}) ->
  {Chess.filter_legal(S.select(Bool.or(start,transit),m,tail),b,(table,acc)) ==
    Chess.filter_legal(tail,b,(table,S.select(Bool.or(start,Bool.or(transit,check)),m,acc))) : Array<U64> & List<&2,Chess.Ply>}:
  match start transit:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}:
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
        r => Chess.filter_legal(tail,b,r),
        Chess.filter_step(b,m,(table,acc)),(table,S.select(check,m,acc)),
        step(table,b,m,acc,check,final))

def wing_case(ok: Bool,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},
  guard: {E.guard(b,ks) == ok : Bool},+tail: List<&2,Chess.Ply>,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_legal(P.before_guard(ok,b,white,ks,tail),b,(Init.run(d,seed,n,extra),acc)) ==
    Chess.filter_legal(tail,b,(Init.run(d,seed,n,extra),P.after_guard(ok,b,white,ks,acc))) : Array<U64> & List<&2,Chess.Ply>}:
  match ok:
    case False{}: {==}
    case True{}:
      advance(Init.run(d,seed,n,extra),b,S.move(white,ks),tail,acc,
        S.checked(b,white,ks,K.Initial{}),S.checked(b,white,ks,K.Transit{}),S.checked(b,white,ks,K.Completed{}),
        Checks.stage(d,seed,n,extra,b,white,ks,K.Completed{},depth,tables,full,good,turn,one,guard))

def wing(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+tail: List<&2,Chess.Ply>,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_legal(P.before(b,white,ks,tail),b,(Init.run(d,seed,n,extra),acc)) ==
    Chess.filter_legal(tail,b,(Init.run(d,seed,n,extra),P.after(b,white,ks,acc))) : Array<U64> & List<&2,Chess.Ply>}:
  wing_case(E.guard(b,ks),d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,{==},tail,acc)

def both(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {Chess.filter_legal(P.before(b,white,False{},P.before(b,white,True{},Nil{})),b,(Init.run(d,seed,n,extra),Nil{})) ==
    (Init.run(d,seed,n,extra),P.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.filter_legal(P.before(b,white,False{},P.before(b,white,True{},Nil{})),b,(Init.run(d,seed,n,extra),Nil{})),
    Chess.filter_legal(P.before(b,white,True{},Nil{}),b,(Init.run(d,seed,n,extra),P.after(b,white,False{},Nil{}))),
    (Init.run(d,seed,n,extra),P.expected(b,white)),
    wing(d,seed,n,extra,b,white,False{},depth,tables,full,good,turn,one,P.before(b,white,True{},Nil{}),Nil{}),
    wing(d,seed,n,extra,b,white,True{},depth,tables,full,good,turn,one,Nil{},P.after(b,white,False{},Nil{})))
