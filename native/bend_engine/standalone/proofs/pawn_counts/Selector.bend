import Base
import ../backrank_pawns/Mask.bend as M
import ../u64/Words.bend as Old
import ../color_counts/Actual.bend as A
import ../../../legal_probe/Chess.bend as Chess

# U32.is_zero is exactly U32.is_eq(_,0) in the pinned Base.
def choice(+b: Chess.Board,+s: U32,+p: U32,zero: Bool,
  same: {U32.is_zero(p) == zero : Bool},
  put: {U32.is_eq(Bool.pick(U32,zero,Chess.piece(b,s),p),0) == True{} : Bool}) ->
  {M.pawn(b,s) == True{} : Bool}:
  match zero:
    case True{}:
      Equal.trans(Bool,M.pawn(b,s),U32.is_eq(Chess.piece(b,s),0),True{},
        Equal.sym(Bool,U32.is_eq(Chess.piece(b,s),0),M.pawn(b,s),M.decoder(b,s)),put)
    case False{}:
      Old.false_true({M.pawn(b,s) == True{} : Bool},
        Equal.trans(Bool,False{},U32.is_zero(p),True{},Equal.sym(Bool,U32.is_zero(p),False{},same),put))

def source(+b: Chess.Board,m: Chess.Ply,put: {U32.is_eq(M.put(b,m),0) == True{} : Bool}) ->
  {M.pawn(b,A.source(m)) == True{} : Bool}:
  match m:
    case Chess.Ply{s,d,+p,f}: choice(b,s,p,U32.is_zero(p),{==},put)

def put(+b: Chess.Board,+m: Chess.Ply) -> Bool: U32.is_eq(M.put(b,m),0)
def certificate(+pawn: U64,+src: Nat,put: Bool) -> Type:
  match put:
    case False{}: Unit
    case True{}: {U64.test_bit(pawn,src) == True{} : Bool}
def tagged(+b: Chess.Board,+m: Chess.Ply,flag: Bool,same: {put(b,m) == flag : Bool}) ->
  certificate(Chess.get_pawns(b),U32.to_nat(A.source(m)),flag):
  match flag:
    case False{}: Unit{}
    case True{}: source(b,m,same)
