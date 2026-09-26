# Small source-only parser/selection probe; it never owns a model.
import Base
import ./Dispatch.bend as D

def walk(fuel: Nat,+n: U32,+p: D.Policy) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+rest:
      do IO<Unit>:
        IO.print(U32.show(n) ++ " " ++ U32.show(D.limit(p,n,4)))
        walk(rest,U32.inc(n),p)

def main() -> IO(Unit):
  do IO<Unit>:
    +p : D.Policy <- D.read()
    +p : D.Policy <- D.checked(D.valid(p,4),p)
    D.notice(p)
    walk(17n,0,p)
