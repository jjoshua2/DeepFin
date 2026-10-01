# Independent decisions for one chosen castling side, not full move generation.
import Base
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def checked(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage) -> Bool:
  Geometry.attacked(K.stage_board(b,white,ks,stage),U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white))

def move(+white: Bool,ks: Bool) -> Chess.Ply:
  Chess.Ply{K.src(white),K.dst(white,ks),0,2}

# This selector is independent of Chess.retain_move, including preservation of tails.
def select(reject: Bool,m: Chess.Ply,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match reject:
    case False{}: Con{m,tail}
    case True{}: tail

def before(+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  select(Bool.or(checked(b,white,ks,K.Initial{}),checked(b,white,ks,K.Transit{})),move(white,ks),tail)

def after(+b: Chess.Board,+white: Bool,+ks: Bool) -> List<&2,Chess.Ply>:
  select(Bool.or(checked(b,white,ks,K.Initial{}),Bool.or(checked(b,white,ks,K.Transit{}),checked(b,white,ks,K.Completed{}))),move(white,ks),Nil{})
