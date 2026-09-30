# Actual decoder only. Neither the abstract square model nor proof predicates execute.
import Base
import ../../Position.bend as Position
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def bit(v: U64,sq: U32) -> String:
  U32.show(Bool.to_u32(U64.test_bit(v,U32.to_nat(sq))))

def emit(+sq: U32,+b: Chess.Board) -> IO(Unit):
  IO.print(U32.show(sq) ++ " " ++ bit(Chess.occupied(b),sq) ++ " " ++ U32.show(Chess.piece(b,sq)) ++ " " ++
    bit(Chess.get_pawns(b),sq) ++ " " ++ bit(Chess.get_knights(b),sq) ++ " " ++ bit(Chess.get_bishops(b),sq) ++ " " ++
    bit(Chess.get_rooks(b),sq) ++ " " ++ bit(Chess.get_queens(b),sq) ++ " " ++ bit(Chess.get_kings(b),sq) ++ " " ++
    bit(Chess.get_white(b),sq) ++ " " ++ bit(Chess.get_black(b),sq))

def insert(ok: Bool,kind: U32,color: U32,+sq: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case True{}: emit(sq,Position.put(kind,U32.is_eq(color,1),sq,b))
    case False{}: IO.die(Unit,2,"invalid decoder insertion")

def query(ok: Bool,op: U32,+kind: U32,+color: U32,+sq: U32,b: Chess.Board) -> IO(Unit):
  match ok op:
    case False{} _: IO.die(Unit,2,"invalid decoder square")
    case True{} 0: emit(sq,b)
    case True{} 1: insert(Bool.and(U32.is_lt(kind,6),U32.is_lt(color,2)),kind,color,sq,b)
    case True{} 2: emit(sq,Position.board(Position.start()))
    case True{} _: IO.die(Unit,2,"invalid decoder operation")

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
    case 1n+remaining op <> kind <> color <> +sq <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> tail:
      do IO<Unit>:
        query(U32.is_lt(sq,64),op,kind,color,sq,Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep})
        rows(remaining,tail)
    case _ _: IO.die(Unit,2,"invalid decoder tuple or budget")

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid decoder number")
    case Some{ys}: rows(64n,ys)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    parsed(numbers(xs,Nil{}))
