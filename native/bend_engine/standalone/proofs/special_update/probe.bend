# Actual representation operations. The oracle never supplies candidate outputs.
import Base
import ../../Position.bend as Position
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def emit(b: Chess.Board) -> IO(Unit):
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  IO.print(show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep))

def checked(ok: Bool,src: U32,dst: U32,promo: U32,flag: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(Chess.make_move(b,Chess.Ply{src,dst,promo,flag}))
    case False{}: IO.die(Unit,2,"invalid special-update request")

def execute(op: U32,+src: U32,+dst: U32,+promo: U32,b: Chess.Board) -> IO(Unit):
  match op:
    case 0: checked(Bool.and(U32.is_lt(src,64),Bool.and(U32.is_lt(dst,64),Bool.and(U32.is_ge(promo,1),U32.is_le(promo,4)))),src,dst,promo,0,b)
    case 1: checked(Bool.and(U32.is_lt(src,64),Bool.and(U32.is_lt(dst,64),U32.is_zero(promo))),src,dst,0,1,b)
    case _: IO.die(Unit,2,"invalid special-update operation")

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
    case _ _: IO.die(Unit,2,"invalid special-update tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid special-update number")
    case Some{ys}: rows(64n,ys)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
