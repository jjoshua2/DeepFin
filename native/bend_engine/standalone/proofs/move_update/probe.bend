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

# This wrapper lifts the actual scalar update; no proof module is imported.
def clear_board(b: Chess.Board,+mask: U64) -> Chess.Board:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  Chess.Board{Chess.update_piece(p,mask,U64.zero(),False{}),Chess.update_piece(n,mask,U64.zero(),False{}),Chess.update_piece(bi,mask,U64.zero(),False{}),Chess.update_piece(r,mask,U64.zero(),False{}),Chess.update_piece(q,mask,U64.zero(),False{}),Chess.update_piece(k,mask,U64.zero(),False{}),Chess.update_piece(w,mask,U64.zero(),False{}),Chess.update_piece(bl,mask,U64.zero(),False{}),turn,rights,ep}

def scalar_board(b: Chess.Board,put: Bool) -> Chess.Board:
  Chess.Board{+p,+n,+bi,r,q,k,w,bl,turn,rights,ep} = b
  Chess.Board{Chess.update_piece(p,n,bi,put),n,bi,r,q,k,w,bl,turn,rights,ep}

def checked_move(ok: Bool,src: U32,dst: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(Chess.make_move(b,Chess.Ply{src,dst,0,0}))
    case False{}: IO.die(Unit,2,"invalid move-update request")

def checked_clear(ok: Bool,hi: U32,lo: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(clear_board(b,U64.from_parts(hi,lo)))
    case False{}: IO.die(Unit,2,"invalid move-update request")

def checked_scalar(ok: Bool,put: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(scalar_board(b,U32.is_eq(put,1)))
    case False{}: IO.die(Unit,2,"invalid move-update request")

def execute(op: U32,+kind: U32,+color: U32,+sq: U32,b: Chess.Board) -> IO(Unit):
  match op:
    case 0: checked_move(Bool.and(U32.is_lt(kind,64),Bool.and(U32.is_lt(color,64),U32.is_zero(sq))),kind,color,b)
    case 1: checked_clear(U32.is_zero(sq),kind,color,b)
    case 2: checked_scalar(Bool.and(U32.is_lt(kind,2),Bool.and(U32.is_zero(color),U32.is_zero(sq))),kind,b)
    case _: IO.die(Unit,2,"invalid move-update operation")

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
    case _ _: IO.die(Unit,2,"invalid move-update tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid move-update number")
    case Some{ys}: rows(64n,ys)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
