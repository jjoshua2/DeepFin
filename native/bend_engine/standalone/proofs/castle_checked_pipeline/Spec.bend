# One-route actual producer followed by the full destination filter.
# The target-centred geometry is independently initialized; not FIDE reachability.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Spec.bend as G

def move(+white: Bool,ks: Bool) -> Chess.Ply:
  Chess.Ply{K.src(white),K.dst(white,ks),0,2}

def check(+b: Chess.Board,+white: Bool,+ks: Bool,+stage: K.Stage) -> Bool:
  G.attacked(K.stage_board(b,white,ks,stage),U32.to_nat(K.stage_square(white,ks,stage)),Bool.not(white))

def keep(reject: Bool,m: Chess.Ply,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match reject:
    case True{}: tail
    case False{}: Con{m,tail}

def rejected(+b: Chess.Board,+white: Bool,+ks: Bool) -> Bool:
  Bool.or(Bool.or(check(b,white,ks,K.Initial{}),check(b,white,ks,K.Transit{})),check(b,white,ks,K.Completed{}))

def filter_result(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,moves) = r
  Chess.filter_legal(moves,b,(table,Nil{}))

def run(table: Array<U64>,+b: Chess.Board,ks: Bool) -> Array<U64> & List<&2,Chess.Ply>:
  filter_result(b,Chess.castle_side(b,ks,(table,Nil{})))
