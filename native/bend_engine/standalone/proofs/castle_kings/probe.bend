# Native candidate executes only production tables, moves and attack/check queries.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def board_text(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def emit(text: String,checked: U32,r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,attacked) = r
  do IO<Array<U64>>:
    IO.print(text ++ " " ++ U32.show(checked) ++ " " ++ U32.show(Bool.to_u32(attacked)))
    return table

def after_check(+b: Chess.Board,sq: U32,side: U32,text: String,r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,checked) = r
  emit(text,Bool.to_u32(checked),Chess.attacked(table,b,sq,U32.xor(side,1)))

def checked(has_king: Bool,+b: Chess.Board,sq: U32,+side: U32,text: String,table: Array<U64>) -> IO(Array<U64>):
  match has_king:
    case True{}: after_check(b,sq,side,text,Chess.in_check(table,b,side))
    case False{}: emit(text,2,Chess.attacked(table,b,sq,U32.xor(side,1)))

def observe(+b: Chess.Board,sq: U32,+side: U32,table: Array<U64>) -> IO(Array<U64>):
  +k = U64.and(Chess.get_kings(b),Chess.color(b,side))
  +selected = U64.ctz(k)
  checked(U32.is_lt(selected,64),b,sq,side,board_text(b) ++ " " ++ show(k) ++ " " ++ U32.show(selected),table)

def execute(+b: Chess.Board,+side: U32,+wing: U32,table: Array<U64>) -> IO(Array<U64>):
  +home = Bool.pick(U32,U32.is_eq(side,1),0,56)
  +src = U32.add(home,4)
  +dst = U32.add(home,Bool.pick(U32,U32.is_eq(wing,1),6,2))
  +mid = U32.add(home,Bool.pick(U32,U32.is_eq(wing,1),5,3))
  do IO<Array<U64>>:
    table : Array<U64> <- observe(b,src,side,table)
    table : Array<U64> <- observe(Chess.make_move(b,Chess.Ply{src,mid,0,0}),mid,side,table)
    observe(Chess.make_move(b,Chess.Ply{src,dst,0,2}),dst,side,table)

def valid(ok: Bool,b: Chess.Board,side: U32,wing: U32,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: execute(b,side,wing,table)
    case False{}: IO.die(Array<U64>,2,"invalid castle king side/wing")

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
    case 1n+p +side <> +wing <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> rest:
      do IO<Unit>:
        table : Array<U64> <- valid(Bool.and(U32.is_lt(side,2),U32.is_lt(wing,2)),Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},side,wing,table)
        rows(p,rest,table)
    case _ _: IO.die(Unit,2,"invalid castle king tuple/budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid castle king number")
    case Some{ys}: rows(128n,ys,Tables.build())

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
