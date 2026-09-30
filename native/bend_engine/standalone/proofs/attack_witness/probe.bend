# Actual callbacks/tables only. No proof or reference implementation imported.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def check_result(text: String,ksq: U32,r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,check) = r
  do IO<Array<U64>>:
    IO.print(text ++ " " ++ U32.show(ksq) ++ " " ++ U32.show(Bool.to_u32(check)))
    return table

def king_check(ok: Bool,+b: Chess.Board,+by: U32,ksq: U32,text: String,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case True{}: check_result(text,ksq,Chess.in_check(table,b,U32.xor(by,1)))
    case False{}:
      do IO<Array<U64>>:
        IO.print(text ++ " " ++ U32.show(ksq) ++ " 2")
        return table

def attack_result(+b: Chess.Board,+by: U32,text: String,r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,hit) = r
  +ksq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,U32.xor(by,1))))
  king_check(U32.is_lt(ksq,64),b,by,ksq,text ++ " " ++ U32.show(Bool.to_u32(hit)),table)

def bishop(+b: Chess.Board,+sq: U32,+by: U32,text: String,r: Array<U64> & U64) -> IO(Array<U64>):
  (table,ba) = r
  attack_result(b,by,text ++ " " ++ show(ba),Chess.attacked(table,b,sq,by))

def rook(+b: Chess.Board,+sq: U32,+by: U32,text: String,r: Array<U64> & U64) -> IO(Array<U64>):
  (table,ra) = r
  bishop(b,sq,by,text ++ " " ++ show(ra),Chess.attack(2,table,sq,by,Chess.occupied(b)))

def king(+b: Chess.Board,+sq: U32,+by: U32,text: String,r: Array<U64> & U64) -> IO(Array<U64>):
  (table,ka) = r
  rook(b,sq,by,text ++ " " ++ show(ka),Chess.attack(3,table,sq,by,Chess.occupied(b)))

def knight(+b: Chess.Board,+sq: U32,+by: U32,text: String,r: Array<U64> & U64) -> IO(Array<U64>):
  (table,na) = r
  king(b,sq,by,text ++ " " ++ show(na),Chess.attack(5,table,sq,by,Chess.occupied(b)))

def pawn(+b: Chess.Board,+sq: U32,+by: U32,r: Array<U64> & U64) -> IO(Array<U64>):
  (table,pa) = r
  knight(b,sq,by,"case " ++ show(pa),Chess.attack(1,table,sq,by,Chess.occupied(b)))

def valid(ok: Bool,+sq: U32,+by: U32,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  match ok:
    case False{}: IO.die(Array<U64>,2,"invalid attack witness domain")
    case True{}: pawn(b,sq,by,Chess.attack(0,table,sq,U32.xor(by,1),Chess.occupied(b)))

def execute(+sq: U32,+by: U32,b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  valid(Bool.and(U32.is_lt(sq,64),U32.is_lt(by,2)),sq,by,b,table)

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
    case 1n+p sq <> by <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> rest:
      do IO<Unit>:
        table : Array<U64> <- execute(sq,by,Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},table)
        rows(p,rest,table)
    case _ _: IO.die(Unit,2,"invalid attack witness tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid attack witness number")
    case Some{ys}: rows(512n,ys,Tables.build())

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
