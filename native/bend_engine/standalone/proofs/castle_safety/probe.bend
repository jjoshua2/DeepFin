# Actual producer/generator castles: compare filter_blockers with full filter on arbitrary rays.
# No imports from proof specifications or external reference decisions.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def board_text(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def select_castle(yes: Bool,m: Chess.Ply,xs: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match yes:
    case True{}: Con{m,xs}
    case False{}: xs

def castles(xs: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match xs:
    case Nil{}: Nil{}
    case Con{Chess.Ply{src,dst,p,+f},rest}:
      select_castle(U32.is_eq(f,2),Chess.Ply{src,dst,p,f},castles(rest))

def observations(xs: List<&2,Chess.Ply>,+b: Chess.Board,+rays: U64) -> IO(Unit):
  match xs:
    case Nil{}: IO.print("observed")
    case Con{Chess.Ply{+src,+dst,+p,+f},rest}:
      required = Chess.filter_requires(U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),Chess.Ply{src,dst,p,f})
      do IO<Unit>:
        IO.print("candidate " ++ U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(p) ++ " " ++ U32.show(f) ++ " " ++ U32.show(Bool.to_u32(required)))
        observations(rest,b,rays)

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

def both(+b: Chess.Board,table: Array<U64>) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.castle_side(b,False{},Chess.castle_side(b,True{},(table,Nil{})))

def finish(+b: Chess.Board,+rays: U64,dup: Bool,r: Array<U64> & List<&2,Chess.Ply>) -> IO(Array<U64>):
  (table,all) = r
  +xs = castles(all)
  +ms = Bool.pick(List<&2,Chess.Ply>,dup,List.append(&2,Chess.Ply,xs,xs),xs)
  do IO<Array<U64>>:
    IO.print("input " ++ U32.show(U32.from_nat(List.length(&2,Chess.Ply,ms))))
    observations(ms,b,rays)
    table : Array<U64> <- emit(b,Chess.filter_blockers(b,ms,(table,rays)))
    emit(b,Chess.filter_legal(ms,b,(table,Nil{})))

def select(op: U32,+b: Chess.Board,rays: U64,table: Array<U64>) -> IO(Array<U64>):
  match op:
    case 0: finish(b,rays,False{},both(b,table))
    case 1: finish(b,rays,False{},Chess.legal_moves(table,b))
    case _: finish(b,rays,True{},both(b,table))

def checked(ok: Bool,op: U32,b: Chess.Board,rays: U64,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: select(op,b,rays,table)
    case False{}: IO.die(Array<U64>,2,"invalid safety request")

def execute(+op: U32,b: Chess.Board,rays: U64,table: Array<U64>) -> IO(Array<U64>):
  checked(U32.is_le(op,2),op,b,rays,table)

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def rows(fuel: Nat,xs: List<&2,U32>,table: Array<U64>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p op <> rayh <> rayl <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> rest:
      do IO<Unit>:
        table : Array<U64> <- execute(op,Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},U64.from_parts(rayh,rayl),table)
        rows(p,rest,table)
    case _ _: IO.die(Unit,2,"invalid safety tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid safety number")
    case Some{ys}: rows(64n,ys,Tables.build())

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
