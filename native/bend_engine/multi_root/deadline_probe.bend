# Explicit tests of actual timer and preadmission-compaction functions. No model.
import Base
import ./Deadlines.bend as D
import ./DeadlineWork.bend as DW
import ./Cohort.bend as C
import ../standalone/Position.bend as Pos
import ../session_probe/Search.bend as S

def require(ok: Bool) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit,Unit{})
    case False{}: IO.die(Unit,2,"deadline probe mismatch")

def mask(t: D.Timers) -> U32:
  D.Timers{ls,n} = t
  n

def timer_checks() -> IO(Unit):
  +base = Nat.add(4294967295n,2n)
  +a = D.install(1,base,D.Timers{Nil{},0})
  +b = D.install(16,Nat.add(base,2n),a)
  +c = D.advance(base,65538,b)
  +d = D.advance(Nat.mul(4294967295n,2n),65536,c)
  do IO<Unit>:
    require(D.allowed(1,Nat.sub(base,1n),a))
    require(Bool.not(D.allowed(1,Nat.add(base,1n),a)))
    require(U32.is_zero(mask(D.advance(Nat.sub(base,1n),65538,b))))
    require(U32.is_eq(mask(c),2))
    require(Bool.not(D.allowed(1,1n,c)))
    require(U32.is_eq(mask(d),65538))
    require(U32.is_eq(mask(D.advance(0n,0,d)),65538))
    require(U32.is_eq(mask(D.advance(Nat.add(base,2n),0,b)),0))
    require(U32.is_eq(D.stopped(4,c),6))
    require(U32.is_eq(mask(D.advance(0n,2,D.install(1,0n,D.Timers{Nil{},0}))),2))
    IO.print("timers 10")

def tasks(fuel: Nat,+row: U32) -> List<C.Task>:
  match fuel:
    case 0n: Nil{}
    case 1n+p:
      +g = Pos.start()
      +b = Pos.board(g)
      +ep = U32.inc(row)
      tree : S.Tree = S.Tree{Array.new(S.Node,0n,S.blank(b)),1,4096,0,0,ep,17,0,2}
      Con{C.Pending{C.Root{g,tree,0,0},g,S.Ticket{ep,17,0,b,Nil{},True{}},Nil{},row},tasks(p,U32.inc(row))}

def fill(fuel: Nat,+at: U32,a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: fill(p,U32.inc(at),Array.set(F32,a,at,F32.from_nat(U32.to_nat(at))))

def value_checked(src: U32,ok: Bool,r: Array<F32> & F32) -> Array<F32> & Bool:
  (a,x) = r
  (a,Bool.and(ok,F32.is_eq(x,F32.from_nat(U32.to_nat(src)))))

def cells(fuel: Nat,+src: U32,+dst: U32,r: Array<F32> & Bool) -> Array<F32> & Bool:
  match fuel:
    case 0n: r
    case 1n+p:
      (a,ok) = r
      cells(p,U32.inc(src),U32.inc(dst),value_checked(src,ok,Array.get(F32,a,dst)))

def checked_cells(r: Array<F32> & Bool) -> IO(Array<F32>):
  (a,ok) = r
  do IO<Array<F32>>:
    require(ok)
    return a

def task_rows(ts: List<C.Task>,+width: U32,+at: U32,a: Array<F32>) -> IO(Array<F32>):
  match ts:
    case Nil{}: IO.pure(Array<F32>,a)
    case Con{C.Local{C.Root{g,S.Tree{pool,used,cap,n,+pending,+ep,req,+stop,depth},neural,draws}},rest}:
      do IO<Array<F32>>:
        require(Bool.and(U32.is_eq(stop,2),U32.is_eq(pending,4096)))
        IO.print("drop " ++ U32.show(ep))
        task_rows(rest,width,at,a)
    case Con{C.Pending{C.Root{g,S.Tree{pool,used,cap,n,pending,+ep,req,stop,depth},neural,draws},game,S.Ticket{+id,+request,+node,b,m,needs},entries,+row},rest}:
      do IO<Array<F32>>:
        require(Bool.and(U32.is_eq(ep,id),Bool.and(U32.is_eq(request,17),Bool.and(U32.is_zero(node),U32.is_eq(row,at)))))
        IO.print("keep " ++ U32.show(ep) ++ " " ++ U32.show(row))
        a : Array<F32> <- checked_cells(cells(U32.to_nat(width),U32.mul(U32.sub(ep,1),width),U32.mul(row,width),(a,True{})))
        task_rows(rest,width,U32.inc(at),a)


def compacted(+m: U32,+width: U32,c: DW.Compact) -> IO(Unit):
  DW.Compact{ts,a,rows} = c
  do IO<Unit>:
    IO.print("case " ++ U32.show(m) ++ " " ++ U32.show(width) ++ " " ++ U32.show(rows))
    a : Array<F32> <- task_rows(ts,width,0,a)
    a : Array<F32> <- checked_cells(cells(1n,U32.mul(4,width),U32.mul(4,width),(a,True{})))
    IO.pure(Unit,Unit{})

def cases(fuel: Nat,+m: U32,+width: U32) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      a = fill(U32.to_nat(U32.inc(U32.mul(4,width))),0,Array.new(F32,16n,0.0))
      do IO<Unit>:
        compacted(m,width,DW.compact(U32.shln(m,1n),width,tasks(4n,0),DW.Compact{Nil{},a,0}))
        cases(p,U32.inc(m),width)

def main() -> IO(Unit):
  do IO<Unit>:
    timer_checks()
    cases(16n,0,9344)
    cases(16n,0,11200)
    IO.print("compaction 32")
