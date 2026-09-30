# Explicit fixed-package gather recommendations. No waits, drops or model switch.
# The launcher derives the 16-entry table from validated service observations;
# native model opening separately requires its exact bound package SHA256.
import Base
import ../standalone/Text.bend as Text

type Policy is Data:
  Greedy{}
  Profile{batch: U32, caps: List<&2,U32>}

def batch_valid(+n: U32) -> Bool:
  Bool.or(U32.is_eq(n,1),Bool.or(U32.is_eq(n,2),Bool.or(U32.is_eq(n,4),Bool.or(U32.is_eq(n,8),U32.is_eq(n,16)))))

def caps_valid(xs: List<&2,U32>,+batch: U32,+at: U32) -> Bool:
  match xs:
    case Nil{}: U32.is_eq(at,17)
    case Con{+n,rest}:
      Bool.and(U32.is_le(at,16),Bool.and(U32.is_gt(n,0),Bool.and(U32.is_le(n,batch),Bool.and(U32.is_le(n,at),caps_valid(rest,batch,U32.inc(at))))))

def valid(p: Policy,+physical: U32) -> Bool:
  match p:
    case Greedy{}: True{}
    case Profile{+batch,caps}:
      Bool.and(batch_valid(batch),Bool.and(U32.is_eq(batch,physical),caps_valid(caps,batch,1)))

def at(xs: List<&2,U32>,index: U32) -> U32:
  match xs index:
    case Con{n,rest} 0: n
    case Con{n,rest} _: at(rest,U32.sub(index,1))
    case Nil{} _: 0

def limit(p: Policy,+candidates: U32,physical: U32) -> U32:
  match p:
    case Greedy{}: physical
    case Profile{batch,caps}:
      # Empty queues must still make a local sweep and finish normally.
      Bool.pick(U32,U32.is_zero(candidates),physical,at(caps,U32.sub(candidates,1)))

def checked(ok: Bool,p: Policy) -> IO(Policy):
  match ok:
    case False{}: IO.die(Policy,2,"dispatch needs physical batch plus 16 bounded gather limits")
    case True{}: IO.pure(Policy,p)

def prepend(x: Maybe<&2,U32>,xs: Maybe<&2,List<&2,U32>>) -> Maybe<&2,List<&2,U32>>:
  match x xs:
    case Some{n} Some{rest}: Some{Con{n,rest}}
    case _ _: None{}

def parsed_caps(xs: List<&2,String>) -> Maybe<&2,List<&2,U32>>:
  match xs:
    case Nil{}: Some{Nil{}}
    case Con{x,rest}: prepend(Text.number(x),parsed_caps(rest))

def parsed(x: Maybe<&2,U32>,xs: Maybe<&2,List<&2,U32>>) -> IO(Policy):
  match x xs:
    case Some{+batch} Some{+caps}:
      checked(Bool.and(batch_valid(batch),caps_valid(caps,batch,1)),Profile{batch,caps})
    case _ _: checked(False{},Greedy{})

def words(xs: List<&2,String>) -> IO(Policy):
  match xs:
    case Con{b,rest}: parsed(Text.number(b),parsed_caps(rest))
    case Nil{}: checked(False{},Greedy{})

def bounded(ok: Bool,text: String) -> IO(Policy):
  match ok:
    case False{}: checked(False{},Greedy{})
    case True{}: words(Text.words(text))

def environment(r: Result<&1,&1,U32 & String,String>) -> IO(Policy):
  match r:
    case Fail{error}: IO.pure(Policy,Greedy{})
    case Done{+text}: bounded(Nat.is_le(String.length(text),64n),text)

def read() -> IO(Policy):
  do IO<Policy>:
    r : Result<&1,&1,U32 & String,String> <- IO.get_env("DEEPFIN_COHORT_DISPATCH")
    environment(r)

def text(xs: List<&2,U32>) -> String:
  match xs:
    case Nil{}: ""
    case Con{n,rest}: " " ++ U32.show(n) ++ text(rest)

def notice(p: Policy) -> IO(Unit):
  match p:
    case Greedy{}: IO.pure(Unit,Unit{})
    case Profile{b,caps}: IO.print("info string live_dispatch fixed-package-v1 " ++ U32.show(b) ++ text(caps))

def reject_nonlive(p: Policy) -> IO(Unit):
  match p:
    case Greedy{}: IO.pure(Unit,Unit{})
    case _: IO.die(Unit,2,"dispatch policy is supported only by the live entrypoint")
