# Runtime input is parsed and the table populated BEFORE the operation timer.
# The build harness changes only the Impl import for the dense-scan control.
import Base
import ./U64Map.bend as M
import ./U64Map.bend as Impl

type Op is Data:
  Op{kind: U32, key: U64, value: U32}

def fold(sum: U32, tag: U32, value: U32) -> U32:
  U32.add(U32.mul(sum, 1664525), U32.add(U32.mul(value, 7), tag))

def put_result(sum: U32, r: M.Table & M.Put) -> M.Table & U32:
  (t, result) = r
  match result:
    case M.Added{}: (t, fold(sum, 1, 0))
    case M.Replaced{v}: (t, fold(sum, 2, v))
    case M.Full{}: (t, fold(sum, 3, 0))

def get_result(sum: U32, tag: U32, r: M.Table & Maybe<&2, U32>) -> M.Table & U32:
  (t, result) = r
  match result:
    case None{}: (t, fold(sum, U32.inc(tag), 0))
    case Some{v}: (t, fold(sum, tag, v))

def step_kind(kind: U32, key: U64, value: U32, t: M.Table, sum: U32) -> M.Table & U32:
  match kind:
    case 0: get_result(sum, 4, Impl.get(t, key))
    case 1: put_result(sum, Impl.put(t, key, value))
    case _: get_result(sum, 6, Impl.remove(t, key))

def step_op(op: Op, t: M.Table, sum: U32) -> M.Table & U32:
  Op{kind, key, value} = op
  step_kind(kind, key, value, t, sum)

def step(op: Op, state: M.Table & U32) -> M.Table & U32:
  (t, sum) = state
  step_op(op, t, sum)

def tape(ops: List<&2, Op>, state: M.Table & U32) -> M.Table & U32:
  match ops:
    case Nil{}: state
    case Con{op, tail}: tape(tail, step(op, state))

def repeat(n: Nat, +ops: List<&2, Op>, state: M.Table & U32) -> M.Table & U32:
  match n:
    case 0n: state
    case 1n+p: repeat(p, ops, tape(ops, state))

def slot_text(slot: M.Slot) -> IO(Unit):
  match slot:
    case M.Empty{}: IO.pure(Unit, Unit{})
    case M.Entry{+key, value}:
      IO.print("entry " ++ U32.show(U64.high(key)) ++ " " ++ U32.show(U64.low(key)) ++ " " ++ U32.show(value))

def dump_one(r: Array<M.Slot> & M.Slot) -> IO(Array<M.Slot>):
  (a, slot) = r
  do IO<Array<M.Slot>>:
    x : Unit <- slot_text(slot)
    return a

def dump(n: Nat, +i: U32, a: Array<M.Slot>) -> IO(Unit):
  match n:
    case 0n: IO.print("end")
    case 1n+p:
      do IO<Unit>:
        a : Array<M.Slot> <- dump_one(Array.get(M.Slot, a, i))
        dump(p, U32.inc(i), a)

def finish(rounds: U32, started: Nat, state: M.Table & U32) -> IO(Unit):
  (t, sum) = state
  M.Table{a, mask, size} = t
  do IO<Unit>:
    ended : Nat <- IO.now()
    IO.print("sample " ++ U32.show(rounds) ++ " " ++ Nat.show(Nat.sub(ended, started))
      ++ " " ++ U32.show(sum) ++ " " ++ U32.show(size))
    dump(U32.to_nat(U32.inc(mask)), 0, a)

def run(+rounds: U32, ops: List<&2, Op>, state: M.Table & U32) -> IO(Unit):
  (t, ignored) = state
  do IO<Unit>:
    started : Nat <- IO.now()
    finish(rounds, started, repeat(U32.to_nat(rounds), ops, (t, 0)))

def table(rounds: U32, init: List<&2, Op>, ops: List<&2, Op>, owner: Maybe<&1, M.Table>) -> IO(Unit):
  match owner:
    case None{}: IO.die(Unit, 2, "invalid benchmark capacity")
    case Some{t}: run(rounds, ops, tape(init, (t, 0)))

# Invalid input is an error, not a missing operation; parse through IO.
def valid_op(kind: U32, hi: U32, lo: U32, value: U32, valid: Bool) -> IO(Op):
  match valid:
    case True{}: IO.pure(Op, Op{kind, U64{lo, hi}, value})
    case False{}: IO.die(Op, 2, "invalid benchmark operation")

def numbers(k: Maybe<&2, U32>, h: Maybe<&2, U32>, l: Maybe<&2, U32>, v: Maybe<&2, U32>) -> IO(Op):
  match k h l v:
    case Some{+kind} Some{hi} Some{lo} Some{value}:
      valid_op(kind, hi, lo, value, U32.is_le(kind, 2))
    case _ _ _ _: IO.die(Op, 2, "invalid benchmark number")

def fields(parts: List<&2, String>) -> IO(Op):
  match parts:
    case Con{k, Con{h, Con{l, Con{v, Nil{}}}}}:
      numbers(U32.read(k), U32.read(h), U32.read(l), U32.read(v))
    case _: IO.die(Op, 2, "invalid benchmark fields")

def parse_ops(lines: List<&2, String>) -> IO(List<&2, Op>):
  match lines:
    case Nil{}: IO.pure(List<&2, Op>, Nil{})
    case Con{line, tail}:
      do IO<List<&2, Op>>:
        op : Op <- fields(String.split(line, ' '))
        rest : List<&2, Op> <- parse_ops(tail)
        return Con{op, rest}

def configured(bits: U32, rounds: U32, initial: String, operations: String) -> IO(Unit):
  do IO<Unit>:
    init : List<&2, Op> <- parse_ops(String.split(initial, ';'))
    ops : List<&2, Op> <- parse_ops(String.split(operations, ';'))
    table(rounds, init, ops, M.new(bits))

def valid_config(bits: U32, rounds: U32, initial: String, ops: String, valid: Bool) -> IO(Unit):
  match valid:
    case True{}: configured(bits, rounds, initial, ops)
    case False{}: IO.die(Unit, 2, "invalid benchmark bounds")

def checked(b: Maybe<&2, U32>, r: Maybe<&2, U32>, initial: String, ops: String) -> IO(Unit):
  match b r:
    case Some{+bits} Some{+rounds}:
      valid_config(bits, rounds, initial, ops,
        Bool.and(U32.is_le(bits, 16), U32.is_le(rounds, 1048576)))
    case _ _: IO.die(Unit, 2, "invalid benchmark config")

def config(parts: List<&2, String>) -> IO(Unit):
  match parts:
    case Con{b, Con{r, Con{init, Con{ops, Nil{}}}}}:
      checked(U32.read(b), U32.read(r), init, ops)
    case _: IO.die(Unit, 2, "invalid benchmark config")

def environment(r: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match r:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_MAP_BENCH")
    case Done{text}: config(String.split(text, '|'))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_MAP_BENCH")
    environment(text)
