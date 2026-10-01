import Base
import ./Spec.bend as S
import ./Stages.bend as Stages
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../attack_witness/King.bend as King
import ../../../legal_probe/Chess.bend as Chess

def color(+b: Chess.Board,+white: Bool) ->
  {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(white))) == S.kings(b,white) : U64}:
  match white:
    case False{}: {==}
    case True{}: {==}

def at(table: Array<U64>,+b: Chess.Board,+white: Bool,+sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  one: {S.kings(b,white) == U64.bit(sq) : U64}) ->
  {Chess.in_check(table,b,Bool.to_u32(white)) == Chess.attacked(table,b,U32.from_nat(sq),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}:
  %Equal.sym(U32,U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(white)))),U32.from_nat(sq),
    King.selected(b,Bool.to_u32(white),sq,bound,
      Equal.trans(U64,U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(white))),S.kings(b,white),U64.bit(sq),color(b,white),one))) :
    {Chess.attacked(table,b,_,U32.xor(Bool.to_u32(white),1)) == Chess.attacked(table,b,U32.from_nat(sq),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}
  Equal.cong(U32,Array<U64> & Bool,x => Chess.attacked(table,b,U32.from_nat(sq),x),U32.xor(Bool.to_u32(white),1),Bool.to_u32(Bool.not(white)),Stages.opposite(white))

def bounded(+white: Bool,+ks: Bool,+stage: S.Stage) ->
  {Nat.is_lt(U32.to_nat(S.stage_square(white,ks,stage)),64n) == True{} : Bool}:
  match white ks stage:
    case False{} False{} S.Initial{}: {==}
    case False{} False{} S.Transit{}: {==}
    case False{} False{} S.Completed{}: {==}
    case False{} True{} S.Initial{}: {==}
    case False{} True{} S.Transit{}: {==}
    case False{} True{} S.Completed{}: {==}
    case True{} False{} S.Initial{}: {==}
    case True{} False{} S.Transit{}: {==}
    case True{} False{} S.Completed{}: {==}
    case True{} True{} S.Initial{}: {==}
    case True{} True{} S.Transit{}: {==}
    case True{} True{} S.Completed{}: {==}

def index(+white: Bool,+ks: Bool,+stage: S.Stage) ->
  {U32.from_nat(U32.to_nat(S.stage_square(white,ks,stage))) == S.stage_square(white,ks,stage) : U32}:
  match white ks stage:
    case False{} False{} S.Initial{}: {==}
    case False{} False{} S.Transit{}: {==}
    case False{} False{} S.Completed{}: {==}
    case False{} True{} S.Initial{}: {==}
    case False{} True{} S.Transit{}: {==}
    case False{} True{} S.Completed{}: {==}
    case True{} False{} S.Initial{}: {==}
    case True{} False{} S.Transit{}: {==}
    case True{} False{} S.Completed{}: {==}
    case True{} True{} S.Initial{}: {==}
    case True{} True{} S.Transit{}: {==}
    case True{} True{} S.Completed{}: {==}

def single(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: S.Stage,
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {S.kings(S.stage_board(b,white,ks,stage),white) == U64.bit(U32.to_nat(S.stage_square(white,ks,stage))) : U64}:
  match stage:
    case S.Initial{}: one
    case S.Transit{}: Stages.mid(b,white,ks,good,turn,one)
    case S.Completed{}: Stages.final(b,white,ks,good,turn,one,guard)

def stage(table: Array<U64>,+b: Chess.Board,+white: Bool,+ks: Bool,+stage: S.Stage,
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.in_check(table,S.stage_board(b,white,ks,stage),Bool.to_u32(white)) ==
   Chess.attacked(table,S.stage_board(b,white,ks,stage),S.stage_square(white,ks,stage),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.in_check(table,S.stage_board(b,white,ks,stage),Bool.to_u32(white)),
    Chess.attacked(table,S.stage_board(b,white,ks,stage),U32.from_nat(U32.to_nat(S.stage_square(white,ks,stage))),Bool.to_u32(Bool.not(white))),
    Chess.attacked(table,S.stage_board(b,white,ks,stage),S.stage_square(white,ks,stage),Bool.to_u32(Bool.not(white))),
    at(table,S.stage_board(b,white,ks,stage),white,U32.to_nat(S.stage_square(white,ks,stage)),bounded(white,ks,stage),single(b,white,ks,stage,good,turn,one,guard)),
    Equal.cong(U32,Array<U64> & Bool,x => Chess.attacked(table,S.stage_board(b,white,ks,stage),x,Bool.to_u32(Bool.not(white))),
      U32.from_nat(U32.to_nat(S.stage_square(white,ks,stage))),S.stage_square(white,ks,stage),index(white,ks,stage)))
