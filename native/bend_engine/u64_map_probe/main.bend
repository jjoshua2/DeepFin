# Probe-only text protocol. Operations are prepared by the external Python oracle.
# The map itself stores numeric keys and values; there is no string-key conversion.
import Base
import ./U64Map.bend as M

def sized(prefix: String, r: M.Table & U32) -> IO(M.Table):
  (t, n) = r
  do IO<M.Table>:
    IO.print(prefix ++ " " ++ U32.show(n))
    return t

def put_result(prefix: String, r: M.Table & M.Put) -> IO(M.Table):
  (t, result) = r
  match result:
    case M.Added{}: sized(prefix ++ " added", M.length(t))
    case M.Full{}: sized(prefix ++ " full", M.length(t))
    case M.Replaced{old}: sized(prefix ++ " replaced " ++ U32.show(old), M.length(t))

def value_result(prefix: String, r: M.Table & Maybe<&2, U32>) -> IO(M.Table):
  (t, result) = r
  match result:
    case None{}: sized(prefix ++ " missing", M.length(t))
    case Some{old}: sized(prefix ++ " value " ++ U32.show(old), M.length(t))

def put_words(t: M.Table, h: Maybe<&2, U32>, l: Maybe<&2, U32>, v: Maybe<&2, U32>) -> IO(M.Table):
  match h l v:
    case Some{+hi} Some{+lo} Some{value}:
      put_result("p " ++ U32.show(hi) ++ " " ++ U32.show(lo), M.put(t, U64{lo, hi}, value))
    case _ _ _: IO.die(M.Table, 2, "invalid probe operands")

def get_words(t: M.Table, h: Maybe<&2, U32>, l: Maybe<&2, U32>, erase: Bool) -> IO(M.Table):
  match h l:
    case Some{+hi} Some{+lo}:
      match erase:
        case False{}: value_result("g " ++ U32.show(hi) ++ " " ++ U32.show(lo), M.get(t, U64{lo, hi}))
        case True{}: value_result("d " ++ U32.show(hi) ++ " " ++ U32.show(lo), M.remove(t, U64{lo, hi}))
    case _ _: IO.die(M.Table, 2, "invalid probe operands")

def op(t: M.Table, words: List<&2, String>) -> IO(M.Table):
  match words:
    case Con{SCon{'p', SNil{}}, Con{h, Con{l, Con{v, Nil{}}}}}:
      put_words(t, U32.read(h), U32.read(l), U32.read(v))
    case Con{SCon{'g', SNil{}}, Con{h, Con{l, Nil{}}}}:
      get_words(t, U32.read(h), U32.read(l), False{})
    case Con{SCon{'d', SNil{}}, Con{h, Con{l, Nil{}}}}:
      get_words(t, U32.read(h), U32.read(l), True{})
    case _: IO.die(M.Table, 2, "invalid probe operation")

def finish(r: M.Table & U32) -> IO(Unit):
  (t, n) = r
  IO.print("end " ++ U32.show(n))

def run(ops: List<&2, String>, t: M.Table) -> IO(Unit):
  match ops:
    case Nil{}: finish(M.length(t))
    case Con{line, tail}:
      do IO<Unit>:
        t : M.Table <- op(t, String.split(line, ' '))
        run(tail, t)

def ready(bits: U32, ops: List<&2, String>, t: Maybe<&1, M.Table>) -> IO(Unit):
  match t:
    case None{}: IO.print("invalid " ++ U32.show(bits))
    case Some{table}:
      do IO<Unit>:
        IO.print("begin " ++ U32.show(bits))
        run(ops, table)

def config(bits: Maybe<&2, U32>, ops: List<&2, String>) -> IO(Unit):
  match bits:
    case None{}: IO.die(Unit, 2, "invalid probe capacity")
    case Some{+n}: ready(n, ops, M.new(n))

def input(lines: List<&2, String>) -> IO(Unit):
  match lines:
    case Nil{}: IO.die(Unit, 2, "missing probe capacity")
    case Con{bits, ops}: config(U32.read(bits), ops)

def environment(r: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match r:
    case Fail{err}: IO.die(Unit, 2, "missing probe input")
    case Done{text}: input(String.split(text, ';'))

def main() -> IO(Unit):
  do IO<Unit>:
    data : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_U64_MAP_TRACE")
    environment(data)
