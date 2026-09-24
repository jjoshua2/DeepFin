# Probe-only text protocol: r interns, g only looks up. Never manipulates storage.
import Base
import ./IdRegistry.bend as R

def next_text(next_id: Maybe<&2, U32>) -> String:
  match next_id:
    case None{}: "none"
    case Some{id}: U32.show(id)

def info_text(prefix: String, info: R.Info) -> String:
  R.Info{size, next_id} = info
  prefix ++ " size " ++ U32.show(size) ++ " next " ++ next_text(next_id)

def reported(prefix: String, r: R.Registry & R.Info) -> IO(R.Registry):
  (registry, info) = r
  do IO<R.Registry>:
    IO.print(info_text(prefix, info))
    return registry

def interned(prefix: String, r: R.Registry & R.Outcome) -> IO(R.Registry):
  (registry, outcome) = r
  match outcome:
    case R.Assigned{id}: reported(prefix ++ " assigned " ++ U32.show(id), R.inspect(registry))
    case R.Known{id}: reported(prefix ++ " known " ++ U32.show(id), R.inspect(registry))
    case R.TableFull{}: reported(prefix ++ " full", R.inspect(registry))
    case R.IdsExhausted{}: reported(prefix ++ " exhausted", R.inspect(registry))

def found(prefix: String, r: R.Registry & Maybe<&2, U32>) -> IO(R.Registry):
  (registry, value) = r
  match value:
    case None{}: reported(prefix ++ " missing", R.inspect(registry))
    case Some{id}: reported(prefix ++ " value " ++ U32.show(id), R.inspect(registry))

def words(registry: R.Registry, h: Maybe<&2, U32>, l: Maybe<&2, U32>, add: Bool) -> IO(R.Registry):
  match h l:
    case Some{+hi} Some{+lo}:
      match add:
        case True{}: interned("r " ++ U32.show(hi) ++ " " ++ U32.show(lo), R.intern(registry, U64{lo, hi}))
        case False{}: found("g " ++ U32.show(hi) ++ " " ++ U32.show(lo), R.get(registry, U64{lo, hi}))
    case _ _: IO.die(R.Registry, 2, "invalid registry operands")

def op(registry: R.Registry, line: List<&2, String>) -> IO(R.Registry):
  match line:
    case Con{SCon{'r', SNil{}}, Con{hi, Con{lo, Nil{}}}}:
      words(registry, U32.read(hi), U32.read(lo), True{})
    case Con{SCon{'g', SNil{}}, Con{hi, Con{lo, Nil{}}}}:
      words(registry, U32.read(hi), U32.read(lo), False{})
    case _: IO.die(R.Registry, 2, "invalid registry operation")

def finish(r: R.Registry & R.Info) -> IO(Unit):
  (registry, info) = r
  IO.print(info_text("end", info))

def run(lines: List<&2, String>, registry: R.Registry) -> IO(Unit):
  match lines:
    case Nil{}: finish(R.inspect(registry))
    case Con{line, tail}:
      do IO<Unit>:
        registry : R.Registry <- op(registry, String.split(line, ' '))
        run(tail, registry)

def ready(bits: U32, first: U32, lines: List<&2, String>, r: Maybe<&1, R.Registry>) -> IO(Unit):
  match r:
    case None{}: IO.print("invalid " ++ U32.show(bits) ++ " " ++ U32.show(first))
    case Some{registry}:
      do IO<Unit>:
        IO.print("begin " ++ U32.show(bits) ++ " " ++ U32.show(first))
        run(lines, registry)

def numbers(bits: Maybe<&2, U32>, first: Maybe<&2, U32>, lines: List<&2, String>) -> IO(Unit):
  match bits first:
    case Some{+b} Some{+f}: ready(b, f, lines, R.new(b, f))
    case _ _: IO.die(Unit, 2, "invalid registry configuration")

def config(header: List<&2, String>, lines: List<&2, String>) -> IO(Unit):
  match header:
    case Con{bits, Con{first, Nil{}}}: numbers(U32.read(bits), U32.read(first), lines)
    case _: IO.die(Unit, 2, "invalid registry header")

def input(lines: List<&2, String>) -> IO(Unit):
  match lines:
    case Nil{}: IO.die(Unit, 2, "missing registry header")
    case Con{header, tail}: config(String.split(header, ' '), tail)

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{err}: IO.die(Unit, 2, "missing DEEPFIN_ID_REGISTRY")
    case Done{text}: input(String.split(text, ';'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_ID_REGISTRY")
    environment(text)
