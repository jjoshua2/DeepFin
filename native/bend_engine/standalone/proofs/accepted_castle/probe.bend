# Observe actual generated castling and query its actual destination with returned table.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def initialize(context: U32) -> Array<U64>:
  match context:
    case 0: Tables.build()
    case _: Tables.extras(64n,0,Tables.tables(128n,0,512,Array.new(U64,17n,U64.from_parts(305419896,2596069104))))

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def board_text(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def keep(flag: U32,m: Chess.Ply,rest: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match flag:
    case 2: Con{m,rest}
    case _: rest

def castles(xs: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match xs:
    case Nil{}: Nil{}
    case Con{Chess.Ply{s,d,p,+f},rest}:
      keep(f,Chess.Ply{s,d,p,f},castles(rest))

def checked_parts(+b: Chess.Board,+src: U32,+dst: U32,+p: U32,+f: U32,r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,check) = r
  do IO<Array<U64>>:
    IO.print("castle " ++ U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(p) ++ " " ++ U32.show(f) ++ " " ++ U32.show(Bool.to_u32(check)) ++ " " ++ board_text(Chess.make_move(b,Chess.Ply{src,dst,p,f})))
    return table

def checked(b: Chess.Board,m: Chess.Ply,r: Array<U64> & Bool) -> IO(Array<U64>):
  Chess.Ply{src,dst,p,f} = m
  checked_parts(b,src,dst,p,f,r)

def emit(xs: List<&2,Chess.Ply>,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  match xs:
    case Nil{}:
      do IO<Array<U64>>:
        IO.print("end")
        return table
    case Con{+m,rest}:
      do IO<Array<U64>>:
        table : Array<U64> <- checked(b,m,Chess.in_check(table,Chess.make_move(b,m),Chess.get_turn(b)))
        emit(rest,b,table)

def begin(+b: Chess.Board,table: Array<U64>,+ms: List<&2,Chess.Ply>) -> IO(Array<U64>):
  do IO<Array<U64>>:
    IO.print("begin " ++ U32.show(U32.from_nat(List.length(&2,Chess.Ply,ms))))
    emit(ms,b,table)

def generated(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> IO(Array<U64>):
  (table,ms) = r
  begin(b,table,castles(ms))

def valid(ok: Bool,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: generated(b,Chess.legal_moves(table,b))
    case False{}: IO.die(Array<U64>,2,"invalid accepted-castle board")

def execute(+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  valid(Bool.and(U32.is_lt(Chess.get_turn(b),2),
    U32.is_lt(U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)))),64)),b,table)

def finish(r: Array<U64> & U64) -> IO(Unit):
  (table,value) = r
  IO.print("marker " ++ show(value))

def rows(fuel: Nat,xs: List<&2,U32>,table: Array<U64>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: finish(Array.get(U64,table,131071))
    case 1n+p ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> rest:
      do IO<Unit>:
        table : Array<U64> <- execute(Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},table)
        rows(p,rest,table)
    case _ _: IO.die(Unit,2,"invalid accepted-castle tuple or budget")

def context(ok: Bool,ctx: U32,xs: List<&2,U32>) -> IO(Unit):
  match ok:
    case True{}: rows(64n,xs,initialize(ctx))
    case False{}: IO.die(Unit,2,"invalid accepted-castle context")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case Some{+ctx <> rest}: context(U32.is_lt(ctx,2),ctx,rest)
    case _: IO.die(Unit,2,"invalid accepted-castle header or number")

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
