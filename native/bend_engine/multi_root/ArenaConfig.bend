# Explicit opt-in fixed arena sizing; never grow storage during a search.
import Base
import ../session_probe/Search.bend as S
import ../standalone/Text.bend as Text

def checked(value: U32, valid: Bool) -> IO(U32):
  match valid:
    case False{}: IO.die(U32, 2, "DEEPFIN_COHORT_ARENA_NODES must be a decimal integer in 1..65536")
    case True{}: IO.pure(U32, value)

def parsed(value: Maybe<&2, U32>) -> IO(U32):
  match value:
    case None{}: checked(0, False{})
    case Some{+n}: checked(n, S.capacity_valid(n))

def bounded(text: String, valid: Bool) -> IO(U32):
  match valid:
    case False{}: checked(0, False{})
    case True{}: parsed(Text.number(text))

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(U32):
  match result:
    # The pinned get_env effect returns ENOENT only for the absent fixed name.
    case Fail{error}: IO.pure(U32, 4096)
    case Done{+text}: bounded(text, Nat.is_le(String.length(text), 5n))

def read() -> IO(U32):
  do IO<U32>:
    value : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_COHORT_ARENA_NODES")
    environment(value)

def notice_if(changed: Bool, +capacity: U32) -> IO(Unit):
  match changed:
    case False{}: IO.pure(Unit, Unit{})
    case True{}:
      IO.print("info string cohort_arena " ++ U32.show(capacity) ++ " "
        ++ U32.show(U32.shln(1, S.storage_depth(capacity))))

def notice(+capacity: U32) -> IO(Unit):
  notice_if(Bool.not(U32.is_eq(capacity, 4096)), capacity)
