# Opt-in batching probe, not the UCI engine or a Python-controlled search.
import Base
import ./Batch.bend as B
import ../standalone/Text.bend as Text

def value(+step: U32, i: U32) -> F32:
  F32.mul(0.0625, F32.from_nat(U32.to_nat(U32.mod(U32.add(i, U32.mul(step, 7)), 31))))

def fill(fuel: Nat, +i: U32, +step: U32, a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: fill(p, U32.inc(i), step, Array.set(F32, a, i, value(step, i)))

def poison(fuel: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: poison(p, U32.inc(i), Array.set(F32, a, i, F32.div(0.0, 0.0)))

def reset(r: Array<F32> & U32) -> Array<F32>:
  (a, count) = r
  poison(U32.to_nat(count), 0, a)

def rendered(text: String, r: Array<F32> & F32) -> Array<F32> & String:
  (a, v) = r
  (a, " " ++ U32.show(F32.bits(v)) ++ text)

def render(fuel: Nat, +i: U32, r: Array<F32> & String) -> Array<F32> & String:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, text) = r
      +left = p
      render(left, i, rendered(text, Array.get(F32, a, U32.add(i, U32.from_nat(left)))))

def print_values(label: String, r: Array<F32> & String) -> IO(Array<F32>):
  (a, text) = r
  do IO<Array<F32>>:
    IO.print(label ++ text)
    return a

def returned(+step: U32, result: B.Result) -> IO(B.Result):
  B.Result{input, output, +real, +physical, +count} = result
  do IO<B.Result>:
    IO.print("batch " ++ U32.show(step) ++ " " ++ U32.show(real) ++ " " ++ U32.show(physical) ++ " " ++ U32.show(count))
    output : Array<F32> <- print_values("values", render(U32.to_nat(count), 0, (output, "")))
    # The backend may publish ONLY logical rows; the next word remains poisoned.
    output : Array<F32> <- print_values("tail", render(1n, count, (output, "")))
    return B.Result{input, output, real, physical, count}

def rows(+step: U32, batch: U32) -> U32:
  match step:
    case 2: 1
    case 3: U32.max(1, U32.sub(batch, 1))
    case 5: 1
    case _: batch

def cycle(fuel: Nat, +step: U32, +cap: B.Capabilities, result: B.Result) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      B.Capabilities{profile, +batch, +channels, ilog, olog} = cap
      B.Result{input, output, real, physical, count} = result
      +n = rows(step, batch)
      do IO<Unit>:
        # Reset the entire reusable host buffers to NaN; only leading real input
        # rows are meaningful. The bridge must neither consume nor return tail.
        result : B.Result <- B.Batch.run(n,
          fill(U32.to_nat(U32.mul(n, U32.mul(channels, 64))), 0, Bool.pick(U32, U32.is_eq(step, 5), 1, step),
            reset(Array.size(F32, input))),
          reset(Array.size(F32, output)))
        result : B.Result <- returned(step, result)
        cycle(p, U32.inc(step), cap, result)

def invalid(+mode: U32, cap: B.Capabilities) -> IO(Unit):
  B.Capabilities{profile, +batch, channels, +ilog, +olog} = cap
  input_bits = Bool.pick(U32, U32.is_eq(mode, 2), U32.sub(ilog, 1), ilog)
  output_bits = Bool.pick(U32, U32.is_eq(mode, 3), U32.sub(olog, 1), olog)
  do IO<Unit>:
    result : B.Result <- B.Batch.run(Bool.pick(U32, U32.is_eq(mode, 1), 0,
      Bool.pick(U32, U32.is_eq(mode, 4), U32.inc(batch), 1)),
      Array.new(F32, U32.to_nat(input_bits), 0.0), Array.new(F32, U32.to_nat(output_bits), 0.0))
    IO.die(Unit, 3, "bad probe request unexpectedly accepted")

def run(+mode: U32, +cap: B.Capabilities) -> IO(Unit):
  match mode:
    case 0:
      B.Capabilities{profile, batch, channels, ilog, olog} = cap
      cycle(5n, 1, cap, B.Result{Array.new(F32, U32.to_nat(ilog), 0.0), Array.new(F32, U32.to_nat(olog), 0.0), 0, batch, 0})
    case _: invalid(mode, cap)

def parsed(cap: B.Capabilities, mode: Maybe<&2, U32>) -> IO(Unit):
  match mode:
    case None{}: IO.die(Unit, 2, "expected probe mode")
    case Some{mode}: run(mode, cap)

def args(cap: B.Capabilities, xs: List<String>) -> IO(Unit):
  match xs:
    case Nil{}: run(0, cap)
    case Con{x, Nil{}}: parsed(cap, Text.number(x))
    case _: IO.die(Unit, 2, "expected one probe mode")

def main() -> IO(Unit):
  do IO<Unit>:
    cap : B.Capabilities <- B.Batch.open()
    xs : List<String> <- IO.args()
    args(cap, xs)
