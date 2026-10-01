# Exact castling projection of the real generator; no substitute generator.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../castle_sequence/Spec.bend as S
import ../castle_kings/Spec.bend as K

def proj_bit(tag: Bool,m: Chess.Ply,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match tag:
    case True{}: Con{m,tail}
    case False{}: tail

def project(xs: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match xs:
    case Nil{}: Nil{}
    case Con{+m,rest}: proj_bit(U32.is_eq(C.flag(m),2),m,project(rest))

def result(r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,xs) = r
  (table,project(xs))

def before_guard(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match ok:
    case False{}: tail
    case True{}: S.before(b,white,ks,tail)

def before(+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  before_guard(E.guard(b,ks),b,white,ks,tail)

def after_guard(ok: Bool,+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match ok:
    case False{}: tail
    case True{}:
      S.select(Bool.or(S.checked(b,white,ks,K.Initial{}),
        Bool.or(S.checked(b,white,ks,K.Transit{}),S.checked(b,white,ks,K.Completed{}))),
        S.move(white,ks),tail)

def after(+b: Chess.Board,+white: Bool,+ks: Bool,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  after_guard(E.guard(b,ks),b,white,ks,tail)

def expected(+b: Chess.Board,+white: Bool) -> List<&2,Chess.Ply>:
  after(b,white,True{},after(b,white,False{},Nil{}))

def prefix(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.castle_side(b,False{},Chess.castle_side(b,True{},C.candidates(table,b)))
