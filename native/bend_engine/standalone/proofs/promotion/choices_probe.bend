# Actual put_move producer, including the caller tail. No proof model.
import Base
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def show(m: Chess.Ply) -> String:
  Chess.Ply{src,dst,prom,flag} = m
  U32.show(src) ++ " " ++ U32.show(dst) ++ " " ++ U32.show(prom) ++ " " ++ U32.show(flag)

def emit(xs: List<&2,Chess.Ply>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case h <> tail:
      do IO<Unit>:
        IO.print(show(h))
        emit(tail)

def check(ok: Bool,src: U32,dst: U32,flag: U32) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"invalid promotion-choice request")
    case True{}: emit(Chess.put_move(True{},src,dst,flag,Chess.Ply{9,10,0,3} <> Nil{}))

def parsed(a: Maybe<&2,U32>,b: Maybe<&2,U32>,c: Maybe<&2,U32>) -> IO(Unit):
  match a b c:
    case Some{+src} Some{+dst} Some{flag}: check(Bool.and(U32.is_lt(src,64),U32.is_lt(dst,64)),src,dst,flag)
    case _ _ _: IO.die(Unit,2,"invalid promotion-choice number")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case a <> b <> c <> Nil{}: parsed(Text.number(a),Text.number(b),Text.number(c))
    case _: IO.die(Unit,2,"invalid promotion-choice tuple")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
