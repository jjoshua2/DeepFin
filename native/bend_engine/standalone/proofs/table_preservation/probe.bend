# Test-only actual operations; no proof predicates or expected decisions.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def indexed(n: Nat,+i: U32,a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      indexed(p,U32.inc(i),Array.set(U64,a,i,U64.from_parts(U32.xor(i,2779096485),U32.add(U32.mul(i,2654435761),12345))))

def capacity(context: U32) -> U32:
  match context:
    case 0: 64
    case 1: 1024
    case _: 131072

def initialize(context: U32) -> Array<U64>:
  match context:
    case 0: indexed(64n,0,Array.new(U64,6n,U64.zero()))
    case 1: indexed(1024n,0,Array.new(U64,10n,U64.zero()))
    case 2: Tables.build()
    case _: indexed(131072n,0,Array.new(U64,17n,U64.zero()))

def cell(r: Array<U64> & U64) -> IO(Array<U64>):
  (table,v) = r
  do IO<Array<U64>>:
    IO.print("cell " ++ show(v))
    return table

def values(n: Nat,+i: U32,table: Array<U64>) -> IO(Array<U64>):
  match n:
    case 0n: IO.pure(Array<U64>,table)
    case 1n+p:
      do IO<Array<U64>>:
        table : Array<U64> <- cell(Array.get(U64,table,i))
        values(p,U32.inc(i),table)

def moves(xs: List<&2,Chess.Ply>) -> IO(Unit):
  match xs:
    case Nil{}: IO.print("end")
    case Con{Chess.Ply{s,d,p,f},rest}:
      do IO<Unit>:
        IO.print("move " ++ U32.show(s) ++ " " ++ U32.show(d) ++ " " ++ U32.show(p) ++ " " ++ U32.show(f))
        moves(rest)

def emit(r: Array<U64> & List<&2,Chess.Ply>) -> IO(Array<U64>):
  (table,ms) = r
  do IO<Array<U64>>:
    moves(ms)
    return table

def word(r: Array<U64> & U64) -> IO(Array<U64>):
  (table,v) = r
  do IO<Array<U64>>:
    IO.print("word " ++ show(v))
    return table

def boolean(r: Array<U64> & Bool) -> IO(Array<U64>):
  (table,v) = r
  do IO<Array<U64>>:
    IO.print("bool " ++ U32.show(Bool.to_u32(v)))
    return table

def tail() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{60,58,0,2},Con{Chess.Ply{28,35,0,1},
    Con{Chess.Ply{48,56,4,0},Con{Chess.Ply{1,18,0,0},Nil{}}}}}}

def operate(op: U32,+b: Chess.Board,table: Array<U64>) -> IO(Array<U64>):
  match op:
    case 0: word(Chess.attack(Chess.get_rights(b),table,Chess.get_ep(b),Chess.get_turn(b),Chess.occupied(b)))
    case 1: boolean(Chess.in_check(table,b,Chess.get_turn(b)))
    case 2:
      +own = Chess.color(b,Chess.get_turn(b))
      emit(Chess.scan(Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,tail())))
    case 3: emit(Chess.castle_side(b,True{},(table,tail())))
    case 4: emit(Chess.castle_side(b,False{},(table,tail())))
    case 5: emit(Chess.filter_legal(tail(),b,(table,tail())))
    case 6: emit(Chess.filter_fast(tail(),b,U64.zero(),(table,tail())))
    case 7: emit(Chess.filter_prepare(b,(table,tail())))
    case _: emit(Chess.legal_moves(table,b))

def size(r: Array<U64> & U32) -> IO(Unit):
  (table,n) = r
  IO.print("done " ++ U32.show(n))

def execute(+context: U32,op: U32,b: Chess.Board) -> IO(Unit):
  do IO<Unit>:
    IO.print("before " ++ U32.show(capacity(context)))
    table : Array<U64> <- values(U32.to_nat(capacity(context)),0,initialize(context))
    table : Array<U64> <- operate(op,b,table)
    IO.print("after " ++ U32.show(capacity(context)))
    table : Array<U64> <- values(U32.to_nat(capacity(context)),0,table)
    size(Array.size(U64,table))

def checked(ok: Bool,context: U32,op: U32,b: Chess.Board) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"invalid table preservation range")
    case True{}: execute(context,op,b)

def parsed(xs: List<&2,U32>) -> IO(Unit):
  match xs:
    case +context <> +op <> ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> Nil{}:
      checked(Bool.and(U32.is_le(context,3),U32.is_le(op,8)),context,op,
        Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep})
    case _: IO.die(Unit,2,"invalid table preservation tuple")

def numbers(xs: List<String>,acc: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{List.reverse(&2,U32,acc)}
    case h <> t:
      do Maybe<&2,List<&2,U32>>:
        v : U32 <- Text.number(h)
        numbers(t,v <> acc)

def validated(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case None{}: IO.die(Unit,2,"invalid table preservation number")
    case Some{ys}: parsed(ys)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    validated(numbers(args,Nil{}))
