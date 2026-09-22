# Opt-in transport test. Link buffer_probe.c, never a deployed model substitute.
import Base
import ./ModelBuffers.bend as Buffers
import ./Text.bend as Text

def fill(fuel: Nat, +i: U32, +v: F32, input: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: input
    case 1n+p: fill(p, U32.inc(i), v, Array.set(F32, input, i, v))

def checked(step: U32, count: U32, input: Array<F32>, r: Array<F32> & F32) -> IO(Buffers.Buffers):
  (output, last) = r
  do IO<Buffers.Buffers>:
    IO.print(U32.show(step) ++ " " ++ U32.show(count) ++ " " ++ U32.show(F32.bits(last)))
    return Buffers.Buffers{input, output, 1861}

def returned(step: U32, count: U32, b: Buffers.Buffers) -> IO(Buffers.Buffers):
  Buffers.Buffers{input, output, size} = b
  match size:
    case 1861: checked(step, count, input, Array.get(F32, output, 1860))
    case _: IO.die(Buffers.Buffers, 2, "invalid returned output shape")

def cycle(fuel: Nat, +step: U32, +count: U32, b: Buffers.Buffers) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      Buffers.Buffers{input, output, size} = b
      do IO<Unit>:
        b : Buffers.Buffers <- Buffers.Native.run(count,
          fill(U32.to_nat(count), 0, F32.from_nat(U32.to_nat(step)), input), output)
        b : Buffers.Buffers <- returned(step, count, b)
        cycle(p, U32.inc(step), count, b)

def run(mode: U32) -> IO(Unit):
  match mode:
    case 1: cycle(1n, 1, 9344, Buffers.Buffers{Array.new(F32, 13n, 0.0), Array.new(F32, 11n, 0.0), 1861})
    case 2: cycle(1n, 1, 9344, Buffers.Buffers{Array.new(F32, 14n, 0.0), Array.new(F32, 10n, 0.0), 1861})
    case 3: cycle(1n, 1, 9343, Buffers.Buffers{Array.new(F32, 14n, 0.0), Array.new(F32, 11n, 0.0), 1861})
    case 4: cycle(1n, 1, 9344, Buffers.Buffers{Array.new(F32, 15n, 0.0), Array.new(F32, 11n, 0.0), 1861})
    case 5: cycle(1n, 1, 9344, Buffers.Buffers{Array.new(F32, 14n, 0.0), Array.new(F32, 12n, 0.0), 1861})
    case 6: cycle(4n, 1, 11200, Buffers.Buffers{Array.new(F32, 14n, 0.0), Array.new(F32, 11n, 0.0), 1861})
    case _: cycle(4n, 1, 9344, Buffers.Buffers{Array.new(F32, 14n, 0.0), Array.new(F32, 11n, 0.0), 1861})

def parsed(r: Maybe<&2, U32>) -> IO(Unit):
  match r:
    case None{}: IO.die(Unit, 2, "expected probe mode")
    case Some{mode}: run(mode)

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case Con{x, Nil{}}: parsed(Text.number(x))
    case _: run(0)

def diagnostic(enabled: Bool) -> IO(Unit):
  match enabled:
    case True{}: IO.print("diagnostics=1")
    case False{}: IO.print("diagnostics=0")

def main() -> IO(Unit):
  do IO<Unit>:
    profile : U32 <- Buffers.Native.open()
    IO.print("profile=" ++ U32.show(profile))
    enabled : Bool <- Buffers.Native.diagnostics()
    diagnostic(enabled)
    xs : List<String> <- IO.args()
    args(xs)
