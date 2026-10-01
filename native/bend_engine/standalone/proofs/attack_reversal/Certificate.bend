import Base
import ./Spec.bend as S
import ./Knight.bend as N
import ./King.bend as K
import ./WhitePawn.bend as W
import ./BlackPawn.bend as B
import ../attack_geometry/Geometry.bend as G

def all(+kind: G.Leaper,+src: Nat,bound: {Nat.is_lt(src,64n) == True{} : Bool}) -> S.certificate(kind,src):
  match kind:
    case G.Knight{}: N.rows(src,bound)
    case G.King{}: K.rows(src,bound)
    case G.WhitePawn{}: W.rows(src,bound)
    case G.BlackPawn{}: B.rows(src,bound)
