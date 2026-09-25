# Bounded diagnostic of actual layout traversal, including raw rejected-state changes.
import Base
import ../../Position.bend as Position
import ../../Text.bend as Text
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

def state(+l: Position.Layout) -> String:
  Position.Layout{b,file,rank,live} = l
  U32.show(file) ++ " " ++ U32.show(rank) ++ " " ++ U32.show(Bool.to_u32(live)) ++ " " ++ U32.show(accepted(Position.layout_result(l))) ++ " " ++ fields(b)

def initial(mode: U32) -> Chess.Board:
  match mode:
    case 0: Position.empty()
    case 1: Position.board(Position.start())
    case _:
      Chess.Board{U64.from_parts(305419896,2864434397),U64.from_parts(2271560481,4275878552),
        U64.from_parts(1431655765,2863311530),U64.from_parts(252645135,4042322160),
        U64.from_parts(858993459,3435973836),U64.from_parts(16711935,4278255360),
        U64.from_parts(2147483648,1),U64.from_parts(4294967295,4294967295),1,15,64}

def emit(+a: String,+b: String,+file: U32,+rank: U32,+live: U32,turn: U32,rights: U32,ep: U32,mode: U32) -> IO(Unit):
  +l: Position.Layout = Position.Layout{Position.metadata(initial(mode),turn,rights,ep),file,rank,U32.is_eq(live,1)}
  +prefix: Position.Layout = Position.layout(a,l)
  IO.print(state(prefix) ++ " " ++ state(Position.layout(b,prefix)) ++ " " ++ state(Position.layout(String.append(a,b),l)))

def checked(ok: Bool,a: String,b: String,file: U32,rank: U32,live: U32,turn: U32,rights: U32,ep: U32,mode: U32) -> IO(Unit):
  match ok:
    case True{}: emit(a,b,file,rank,live,turn,rights,ep,mode)
    case False{}: IO.die(Unit,2,"invalid parser probe domain")

def parsed(+a: String,+b: String,ints: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match ints:
    case Some{+file <> +rank <> +live <> turn <> rights <> ep <> +mode <> Nil{}}:
      checked(Bool.and(U32.is_le(file,9),Bool.and(U32.is_le(rank,8),Bool.and(U32.is_lt(live,2),Bool.and(U32.is_lt(mode,3),Nat.is_le(Nat.add(String.length(a),String.length(b)),512n))))),a,b,file,rank,live,turn,rights,ep,mode)
    case _: IO.die(Unit,2,"invalid parser probe number")

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def rows(fuel: Nat,xs: List<String>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p a <> b <> file <> rank <> live <> turn <> rights <> ep <> mode <> tail:
      do IO<Unit>:
        parsed(a,b,numbers(file <> rank <> live <> turn <> rights <> ep <> mode <> Nil{},Nil{}))
        rows(p,tail)
    case _ _: IO.die(Unit,2,"invalid parser probe tuple or budget")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    rows(64n,xs)
