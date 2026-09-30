# Opt-in actual table metadata + PEXT + address + public array read/write probe.
# No proof specification, host encoder or expected-value oracle is imported.
import Base
import ../../Tables.bend as Tables
import ../../Text.bend as Text
import ../../../bitboard_probe/Sliders.bend as Sliders

def show(+v: U64) -> String:
  U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v))

def emit(id: U32,pi: U32,ri: U32,ai: U32,pj: U32,rj: U32,aj: U32,
  old: U64,after: U64,written: U64,r: Array<U64> & U32) -> IO(Unit):
  (a,n) = r
  IO.print(U32.show(id) ++ " " ++ U32.show(pi) ++ " " ++ U32.show(ri) ++ " " ++ U32.show(ai)
    ++ " " ++ U32.show(pj) ++ " " ++ U32.show(rj) ++ " " ++ U32.show(aj)
    ++ " " ++ show(old) ++ " " ++ show(after) ++ " " ++ show(written) ++ " " ++ U32.show(n))

def written(id: U32,pi: U32,ri: U32,ai: U32,pj: U32,rj: U32,aj: U32,
  old: U64,after: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  emit(id,pi,ri,ai,pj,rj,aj,old,after,v,Array.size(U64,a))

def after(id: U32,pi: U32,ri: U32,+ai: U32,pj: U32,rj: U32,aj: U32,
  old: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,v) = r
  written(id,pi,ri,ai,pj,rj,aj,old,v,Array.get(U64,a,ai))

def before(id: U32,pi: U32,ri: U32,+ai: U32,pj: U32,rj: U32,+aj: U32,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,old) = r
  after(id,pi,ri,ai,pj,rj,aj,old,Array.get(U64,Array.set(U64,a,ai,v),aj))

def mask_j(id: U32,pi: U32,ri: U32,ai: U32,+pj: U32,occj: U64,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,m) = r
  +rj = Sliders.pext_index(occj,m)
  +aj = U32.add(pj,rj)
  before(id,pi,ri,ai,pj,rj,aj,v,Array.get(U64,a,aj))

def prefix_j(id: U32,pi: U32,ri: U32,ai: U32,kj: U32,occj: U64,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,pj) = r
  mask_j(id,pi,ri,ai,U64.low(pj),occj,v,Array.get(U64,a,kj))

def mask_i(id: U32,+pi: U32,+kj: U32,occi: U64,occj: U64,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,m) = r
  +ri = Sliders.pext_index(occi,m)
  +ai = U32.add(pi,ri)
  prefix_j(id,pi,ri,ai,kj,occj,v,Array.get(U64,a,U32.add(128,kj)))

def prefix_i(id: U32,ki: U32,kj: U32,occi: U64,occj: U64,v: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,pi) = r
  mask_i(id,U64.low(pi),kj,occi,occj,v,Array.get(U64,a,ki))

def run(+id: U32,+ki: U32,kj: U32,occi: U64,occj: U64,v: U64) -> IO(Unit):
  prefix_i(id,ki,kj,occi,occj,v,Array.get(U64,Tables.build(),U32.add(128,ki)))

def checked(ok: Bool,id: U32,ki: U32,kj: U32,oi: U64,oj: U64,v: U64) -> IO(Unit):
  match ok:
    case True{}: run(id,ki,kj,oi,oj,v)
    case False{}: IO.die(Unit,2,"invalid relative address fixture or budget")

def parse(+id: U32,ki: Maybe<&2,U32>,kj: Maybe<&2,U32>,ih: Maybe<&2,U32>,il: Maybe<&2,U32>,
  jh: Maybe<&2,U32>,jl: Maybe<&2,U32>,vh: Maybe<&2,U32>,vl: Maybe<&2,U32>) -> IO(Unit):
  match ki kj ih il jh jl vh vl:
    case Some{+i} Some{+j} Some{ih} Some{il} Some{jh} Some{jl} Some{vh} Some{vl}:
      checked(Bool.and(U32.is_lt(id,512),Bool.and(U32.is_lt(i,128),U32.is_lt(j,128))),
        id,i,j,U64.from_parts(ih,il),U64.from_parts(jh,jl),U64.from_parts(vh,vl))
    case _ _ _ _ _ _ _ _: IO.die(Unit,2,"invalid relative address integer")

def cases(xs: List<String>,+id: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case ki <> kj <> ih <> il <> jh <> jl <> vh <> vl <> rest:
      do IO<Unit>:
        parse(id,Text.number(ki),Text.number(kj),Text.number(ih),Text.number(il),Text.number(jh),Text.number(jl),Text.number(vh),Text.number(vl))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected eight relative address operands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    cases(xs,0)
