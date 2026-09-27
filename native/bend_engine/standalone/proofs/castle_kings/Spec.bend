# Selected mover's king plane and the two actual castling-stage Boards.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castling/Route.bend as R
import ../castle_emission/Spec.bend as E

def color(+b: Chess.Board,white: Bool) -> U64:
  Bool.pick(U64,white,Chess.get_white(b),Chess.get_black(b))

def kings(+b: Chess.Board,white: Bool) -> U64:
  U64.and(Chess.get_kings(b),color(b,white))

def src(white: Bool) -> U32:
  U32.add(Bool.pick(U32,white,0,56),4)

def dst(white: Bool,ks: Bool) -> U32:
  U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,6,2))

def transit(white: Bool,ks: Bool) -> U32:
  U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,5,3))

def mid(b: Chess.Board,+white: Bool,ks: Bool) -> Chess.Board:
  Chess.make_move(b,Chess.Ply{src(white),transit(white,ks),0,0})

def final(b: Chess.Board,+white: Bool,ks: Bool) -> Chess.Board:
  Chess.make_move(b,Chess.Ply{src(white),dst(white,ks),0,2})

def effect(+k: U64,+c: U64,+m: U64,+t: U64,r: U64) -> U64:
  U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(c,m),U64.or(t,r)))

type Stage is Data:
  Initial{}
  Transit{}
  Completed{}

def stage_board(b: Chess.Board,white: Bool,ks: Bool,stage: Stage) -> Chess.Board:
  match stage:
    case Initial{}: b
    case Transit{}: mid(b,white,ks)
    case Completed{}: final(b,white,ks)

def stage_square(white: Bool,ks: Bool,stage: Stage) -> U32:
  match stage:
    case Initial{}: src(white)
    case Transit{}: transit(white,ks)
    case Completed{}: dst(white,ks)
