# Actual initialized parser state at every character boundary. No proof predicate runs.
import Base
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def fields(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def accepted(r: Maybe<&2,Chess.Board>) -> U32:
  match r:
    case None{}: 0
    case Some{b}: 1

def state(+l: P.Layout) -> String:
  P.Layout{b,file,rank,live} = l
  U32.show(file) ++ " " ++ U32.show(rank) ++ " " ++ U32.show(Bool.to_u32(live)) ++ " " ++ U32.show(accepted(P.layout_result(l))) ++ " " ++ fields(b)

def trace(s: String,+l: P.Layout) -> IO(Unit):
  match s:
    case SNil{}: IO.print(state(l))
    case SCon{Chr{c},rest}:
      do IO<Unit>:
        IO.print(state(l))
        trace(rest,P.layout_char(c,l))

def checked(ok: Bool,s: String) -> IO(Unit):
  match ok:
    case True{}: trace(s,P.Layout{P.empty(),0,7,True{}})
    case False{}: IO.die(Unit,2,"invalid frontier probe string")

def rows(fuel: Nat,xs: List<String>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p +s <> tail:
      do IO<Unit>:
        checked(Nat.is_le(String.length(s),256n),s)
        rows(p,tail)
    case _ _: IO.die(Unit,2,"invalid frontier probe batch")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    rows(32n,xs)
