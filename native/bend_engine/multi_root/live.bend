# Explicit persistent entrypoint. The fixed-cohort and UCI executables are intact.
import Base
import ./ArenaConfig.bend as Arena
import ./Cohort.bend as C
import ./LiveRun.bend as L
import ../standalone/Tables.bend as Tables
import ../standalone/Text.bend as T
import ../batch_backend/Batch.bend as B
import ../batch_backend/Async.bend as A

def with_capacity(cfg: C.Config,+capacity: U32) -> IO(Unit):
  do IO<Unit>:
    Arena.notice(capacity)
    cap : B.Capabilities <- B.Batch.open()
    L.opened(cfg,capacity,Tables.build(),cap)

def enabled(ok: Bool,cfg: C.Config) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"live cohorts require DEEPFIN_COHORT_ASYNC=1")
    case True{}:
      do IO<Unit>:
        capacity : U32 <- Arena.read()
        with_capacity(cfg,capacity)

def configured(ok: Bool,cfg: C.Config) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit,2,"expected sims 1..256, depth 1..32, evals 0..256, diagnostics 0..1")
    case True{}:
      do IO<Unit>:
        ok : Bool <- A.BatchAsync.enabled()
        enabled(ok,cfg)

def numbers(s: Maybe<&2,U32>,d: Maybe<&2,U32>,e: Maybe<&2,U32>,t: Maybe<&2,U32>) -> IO(Unit):
  match s d e t:
    case Some{+s} Some{+d} Some{+e} Some{+t}:
      configured(Bool.and(U32.is_gt(s,0),Bool.and(U32.is_le(s,256),Bool.and(U32.is_gt(d,0),Bool.and(U32.is_le(d,32),Bool.and(U32.is_le(e,256),U32.is_le(t,1)))))),C.Config{s,d,e,U32.is_eq(t,1)})
    case _ _ _ _: IO.die(Unit,2,"invalid live cohort numeric argument")

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case Con{s,Con{d,Con{e,Con{t,Nil{}}}}}: numbers(T.number(s),T.number(d),T.number(e),T.number(t))
    case _: IO.die(Unit,2,"usage: SIMS DEPTH EVALS DIAGNOSTICS; submit roots with add commands")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
