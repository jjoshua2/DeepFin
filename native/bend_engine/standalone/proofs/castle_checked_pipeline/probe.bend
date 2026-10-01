# Sequential actual producer and full filter. No proof model or expected decisions.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def board_text(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def tail(mode: U32) -> List<&2,Chess.Ply>:
  match mode:
    case 0: Nil{}
    case 1: Con{Chess.Ply{17,25,0,0},Nil{}}
    case 2: Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{17,25,0,0},Nil{}}}
    case _: Con{Chess.Ply{60,58,0,2},Con{Chess.Ply{4,6,0,2},Nil{}}}

def emit_moves(xs: List<&2,Chess.Ply>,+b: Chess.Board) -> IO(Unit):
  match xs:
    case Nil{}: IO.print("end")
    case Con{Chess.Ply{+src,+dst,+promo,+flag},rest}:
      do IO<Unit>:
        IO.print("move " ++ U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(promo) ++ " " ++ U32.show(flag) ++ " " ++ board_text(Chess.make_move(b,Chess.Ply{src,dst,promo,flag})))
        emit_moves(rest,b)

def emit(b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> IO(Array<U64>):
  (table,+ms) = r
  do IO<Array<U64>>:
    IO.print("begin " ++ U32.show(U32.from_nat(List.length(&2,Chess.Ply,ms))))
    emit_moves(ms,b)
    return table

def final_filter(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,moves) = r
  Chess.filter_legal(moves,b,(table,Nil{}))

def pipeline(+b: Chess.Board,+ks: U32,mode: U32,table: Array<U64>) -> IO(Array<U64>):
  do IO<Array<U64>>:
    table : Array<U64> <- emit(b,Chess.castle_side(b,U32.is_eq(ks,1),(table,tail(mode))))
    emit(b,final_filter(b,Chess.castle_side(b,U32.is_eq(ks,1),(table,Nil{}))))

def finish(r: Array<U64> & U64) -> IO(Unit):
  (table,marker) = r
  IO.print("state " ++ show(marker))

def checked(ok: Bool,ks: U32,mode: U32,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: pipeline(b,ks,mode,table)
    case False{}: IO.die(Array<U64>,2,"invalid emission request")

def execute(+ks: U32,+mode: U32,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  checked(Bool.and(U32.is_le(ks,1),U32.is_le(mode,3)),ks,mode,b,table)

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def rows(fuel: Nat,xs: List<&2,U32>,table: Array<U64>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: finish(Array.get(U64,table,131071))
    case 1n+p ks <> mode <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> rest:
      do IO<Unit>:
        table : Array<U64> <- execute(ks,mode,Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},table)
        rows(p,rest,table)
    case _ _: IO.die(Unit,2,"invalid emission tuple or budget")

def initialize(context: U32,ys: List<&2,U32>) -> IO(Unit):
  match context:
    case 0: rows(64n,ys,Tables.build())
    case 1: rows(64n,ys,Tables.extras(64n,0,Tables.tables(128n,0,512,Array.new(U64,17n,U64.from_parts(324508639,610839776)))))
    case _: IO.die(Unit,2,"invalid emission initialization")

def nonempty(xs: List<&2,U32>) -> IO(Unit):
  match xs:
    case Nil{}: IO.die(Unit,2,"invalid emission missing initialization")
    case Con{context,ys}: initialize(context,ys)

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid emission number")
    case Some{ys}: nonempty(ys)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
