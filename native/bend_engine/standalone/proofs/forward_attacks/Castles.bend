# Immediate composition into the qualified actual full-generator castling pair.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../castle_kings/Check.bend as Check
import ../castle_emission/Spec.bend as E
import ../castle_sequence/Spec.bend as S
import ../initialized_attacks/Build.bend as Init
import ../generator_contract/CastleSpec.bend as P
import ../generator_contract/CastleComposition.bend as Prior
import ./Spec.bend as Forward
import ./Bridge.bend as Bridge

def check(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage) -> Bool:
  Forward.attacked(K.stage_board(b,white,ks,stage),
    U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white))

def after_guard(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match ok:
    case False{}: tail
    case True{}:
      S.select(Bool.or(check(b,white,ks,K.Initial{}),
        Bool.or(check(b,white,ks,K.Transit{}),check(b,white,ks,K.Completed{}))),
        S.move(white,ks),tail)

def after(+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  after_guard(E.guard(b,ks),b,white,ks,tail)

def expected(+b: Chess.Board,+white: Bool) -> List<&2,Chess.Ply>:
  after(b,white,True{},after(b,white,False{},Nil{}))

def check_equal(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage) ->
  {S.checked(b,white,ks,stage) == check(b,white,ks,stage) : Bool}:
  Bridge.geometry(K.stage_board(b,white,ks,stage),
    U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white),Check.bounded(white,ks,stage))

def after_case(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.after_guard(ok,b,white,ks,tail) == after_guard(ok,b,white,ks,tail) : List<&2,Chess.Ply>}:
  match ok:
    case False{}: {==}
    case True{}:
      %check_equal(b,white,ks,K.Initial{}) :
        {P.after_guard(True{},b,white,ks,tail) ==
          S.select(Bool.or(_,Bool.or(check(b,white,ks,K.Transit{}),check(b,white,ks,K.Completed{}))),
            S.move(white,ks),tail) : List<&2,Chess.Ply>}
      %check_equal(b,white,ks,K.Transit{}) :
        {P.after_guard(True{},b,white,ks,tail) ==
          S.select(Bool.or(S.checked(b,white,ks,K.Initial{}),Bool.or(_,check(b,white,ks,K.Completed{}))),
            S.move(white,ks),tail) : List<&2,Chess.Ply>}
      %check_equal(b,white,ks,K.Completed{}) :
        {P.after_guard(True{},b,white,ks,tail) ==
          S.select(Bool.or(S.checked(b,white,ks,K.Initial{}),Bool.or(S.checked(b,white,ks,K.Transit{}),_)),
            S.move(white,ks),tail) : List<&2,Chess.Ply>}
      {==}

def after_equal(+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>) ->
  {P.after(b,white,ks,tail) == after(b,white,ks,tail) : List<&2,Chess.Ply>}:
  after_case(E.guard(b,ks),b,white,ks,tail)

def expected_equal(+b: Chess.Board,+white: Bool) ->
  {P.expected(b,white) == expected(b,white) : List<&2,Chess.Ply>}:
  Equal.trans(List<&2,Chess.Ply>,P.expected(b,white),
    P.after(b,white,True{},after(b,white,False{},Nil{})),expected(b,white),
    Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,xs => P.after(b,white,True{},xs),
      P.after(b,white,False{},Nil{}),after(b,white,False{},Nil{}),
      after_equal(b,white,False{},Nil{})),
    after_equal(b,white,True{},after(b,white,False{},Nil{})))

def exact(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==
    (Init.run(d,seed,n,extra),expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    P.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)),
    (Init.run(d,seed,n,extra),P.expected(b,white)),
    (Init.run(d,seed,n,extra),expected(b,white)),
    Prior.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one),
    Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
      xs => (Init.run(d,seed,n,extra),xs),P.expected(b,white),expected(b,white),expected_equal(b,white)))
