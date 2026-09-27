import Base
import ./Runtime.bend as Run
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../../Text.bend as Text

def initialize(context: U32) -> Array<U64>:
  match context:
    case 0: Tables.build()
    case _: Tables.extras(64n,0,Tables.tables(128n,0,512,Array.new(U64,17n,U64.from_parts(305419896,2596069104))))

def list_text(xs: List<&2,Chess.Ply>) -> String:
  match xs:
    case Nil{}: ""
    case Con{Chess.Ply{s,d,p,f},tail}: " " ++ U32.show(s) ++ ":" ++ U32.show(d) ++ ":" ++ U32.show(p) ++ ":" ++ U32.show(f) ++ list_text(tail)

def emit(r: Array<U64> & (List<&2,Chess.Ply> & Bool)) -> IO(Array<U64>):
  (a,(ms,v)) = r
  do IO<Array<U64>>:
    IO.print("following " ++ U32.show(Bool.to_u32(v)) ++ list_text(ms))
    return a

def baseline(+b: Chess.Board,q: Chess.Board,side: U32,r: Array<U64> & List<&2,Chess.Ply>) -> IO(Array<U64>):
  (a,ms) = r
  do IO<Array<U64>>:
    IO.print("baseline" ++ list_text(ms))
    emit(Run.run(a,b,q,side))

def valid(ok: Bool,+b: Chess.Board,q: Chess.Board,side: U32,a: Array<U64>) -> IO(Array<U64>):
  match ok:
    case False{}: IO.die(Array<U64>,2,"invalid followup side or king")
    case True{}: baseline(b,q,side,Chess.legal_moves(a,b))

def finish(r: Array<U64> & U64) -> IO(Unit):
  (a,+v) = r
  IO.print("marker " ++ U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))

def rows(fuel: Nat,xs: List<&2,U32>,a: Array<U64>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: finish(Array.get(U64,a,131071))
    case 1n+p ph <> pl <> nh <> nl <> bh <> bl <> rh <> rl <> qh <> ql <> kh <> kl <> wh <> wl <> blh <> bll <> turn <> rights <> ep <> p2h <> p2l <> n2h <> n2l <> b2h <> b2l <> r2h <> r2l <> q2h <> q2l <> +k2h <> +k2l <> +w2h <> +w2l <> +c2h <> +c2l <> t2 <> r2 <> e2 <> +side <> rest:
      do IO<Unit>:
        a : Array<U64> <- valid(Bool.and(U32.is_lt(side,2),U32.is_lt(U64.ctz(U64.and(U64.from_parts(k2h,k2l),Bool.pick(U64,U32.is_eq(side,1),U64.from_parts(w2h,w2l),U64.from_parts(c2h,c2l)))),64)),
          Chess.Board{U64.from_parts(ph,pl),U64.from_parts(nh,nl),U64.from_parts(bh,bl),U64.from_parts(rh,rl),U64.from_parts(qh,ql),U64.from_parts(kh,kl),U64.from_parts(wh,wl),U64.from_parts(blh,bll),turn,rights,ep},
          Chess.Board{U64.from_parts(p2h,p2l),U64.from_parts(n2h,n2l),U64.from_parts(b2h,b2l),U64.from_parts(r2h,r2l),U64.from_parts(q2h,q2l),U64.from_parts(k2h,k2l),U64.from_parts(w2h,w2l),U64.from_parts(c2h,c2l),t2,r2,e2},side,a)
        rows(p,rest,a)
    case _ _: IO.die(Unit,2,"invalid followup tuple or budget")

def context(ok: Bool,ctx: U32,xs: List<&2,U32>) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"invalid followup context")
    case True{}: rows(128n,xs,initialize(ctx))

def parsed(xs: Maybe<&2,List<&2,U32>>) -> IO(Unit):
  match xs:
    case Some{+ctx <> rest}: context(U32.is_lt(ctx,2),ctx,rest)
    case _: IO.die(Unit,2,"invalid followup number or header")

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
