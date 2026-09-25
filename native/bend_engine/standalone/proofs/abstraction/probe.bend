# Actual full-board observations; structural snapshot functions do not execute natively.
import Base
import ../../Position.bend as Position
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def fields(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def squares(fuel: Nat,+sq: U32,+b: Chess.Board) -> String:
  match fuel:
    case 0n: ""
    case 1n+p:
      " " ++ U32.show(Bool.pick(U32,U64.test_bit(Chess.occupied(b),U32.to_nat(sq)),Chess.piece(b,sq),6)) ++
        " " ++ U32.show(Bool.to_u32(U64.test_bit(Chess.get_white(b),U32.to_nat(sq)))) ++ squares(p,U32.inc(sq),b)

def emit(+b: Chess.Board) -> IO(Unit):
  IO.print(fields(b) ++ squares(64n,0,b))

def put_checked(ok: Bool,kind: U32,color: U32,sq: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(Position.put(kind,U32.is_eq(color,1),sq,b))
    case False{}: IO.die(Unit,2,"invalid snapshot insertion request")

def execute(op: U32,+kind: U32,+color: U32,+sq: U32,b: Chess.Board) -> IO(Unit):
  match op:
    case 0: put_checked(Bool.and(U32.is_lt(kind,6),Bool.and(U32.is_lt(color,2),U32.is_lt(sq,64))),kind,color,sq,b)
    case 1: emit(Position.metadata(b,kind,color,sq))
    case 2: emit(Position.empty())
    case 3: emit(Position.board(Position.start()))
    case 4: emit(b)
    case _: IO.die(Unit,2,"invalid snapshot operation")

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def rows(fuel: Nat,xs: List<&2,U32>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+remaining op <> kind <> color <> sq <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> tail:
      do IO<Unit>:
        execute(op,kind,color,sq,Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep})
        rows(remaining,tail)
    case _ _: IO.die(Unit,2,"invalid snapshot tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid snapshot number")
    case Some{ys}: rows(64n,ys)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
