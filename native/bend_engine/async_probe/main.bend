# Generated-C foreign-effect smoke. The deterministic test backend is never
# linked into the engine. This is not search or a UCI responsiveness claim.
import Base
import ../standalone/Async.bend as Async

type State is Type:
  State{phase: U32, input: Array<F32>, output: Array<F32>, token: U32}

def submitted(phase: U32, output: Array<F32>, s: Async.AsyncSubmitted) -> IO(State):
  Async.AsyncSubmitted{input, +token} = s
  do IO<State>:
    IO.print("submitted " ++ U32.show(token))
    return State{phase, input, output, token}

def second(output: Array<F32>, s: Async.AsyncSubmitted) -> IO(State):
  Async.AsyncSubmitted{input, +token} = s
  do IO<State>:
    IO.print("submitted " ++ U32.show(token))
    return State{4, Array.set(F32, input, 0, 99.0), output, token}

def drained(input: Array<F32>, output: Array<F32>, token: U32, status: U32) -> IO(State):
  match status:
    case 1: IO.pure(State, State{2, input, output, token})
    case 3:
      do IO<State>:
        IO.print("cancelled 3")
        return State{3, input, output, token}
    case _: IO.die(State, 2, "cancelled request must drain exactly once")

def checked(ok: Bool, phase: U32, input: Array<F32>, output: Array<F32>, token: U32) -> IO(State):
  match ok:
    case True{}: IO.pure(State, State{phase, input, output, token})
    case False{}: IO.die(State, 2, "Bend async input/output or duplicate contract failed")

def first_seen(input: Array<F32>, token: U32, r: Array<F32> & F32) -> IO(State):
  (output, value) = r
  checked(U32.is_eq(F32.bits(value), 1073741824), 5, input, output, token)

def polled(input: Array<F32>, token: U32, p: Async.AsyncPolled) -> IO(State):
  Async.AsyncPolled{output, status} = p
  match status:
    case 1: IO.pure(State, State{4, input, output, token})
    case 2:
      do IO<State>:
        IO.print("complete 2")
        first_seen(input, token, Array.get(F32, output, 0))
    case _: IO.die(State, 2, "live poll failed")

def duplicated(input: Array<F32>, token: U32, p: Async.AsyncPolled) -> IO(State):
  Async.AsyncPolled{output, +status} = p
  do IO<State>:
    IO.print("duplicate " ++ U32.show(status))
    checked(U32.is_zero(status), 6, input, output, token)

def phase(code: U32, input: Array<F32>, output: Array<F32>, +token: U32) -> IO(State):
  match code:
    case 0:
      do IO<State>:
        s : Async.AsyncSubmitted <- Async.Async.submit(1, 1, 1, 9344, input)
        submitted(1, output, s)
    case 1:
      do IO<State>:
        Async.Async.cancel(token, 1, 1, 1)
        return State{2, input, output, token}
    case 2:
      do IO<State>:
        status : U32 <- Async.Async.drain(token, 1, 1, 1)
        drained(input, output, token, status)
    case 3:
      do IO<State>:
        s : Async.AsyncSubmitted <- Async.Async.submit(1, 1, 1, 9344, input)
        second(output, s)
    case 4:
      do IO<State>:
        p : Async.AsyncPolled <- Async.Async.poll(token, 1, 1, 1, output)
        polled(input, token, p)
    case 5:
      do IO<State>:
        p : Async.AsyncPolled <- Async.Async.poll(token, 1, 1, 1, output)
        duplicated(input, token, p)
    case _: IO.pure(State, State{code, input, output, token})

def step(s: State) -> IO(State):
  State{code, input, output, token} = s
  phase(code, input, output, token)

def finished_code(code: U32) -> IO(Unit):
  match code:
    case 6:
      do IO<Unit>:
        Async.Async.shutdown()
        IO.print("passed")
    case _: IO.die(Unit, 2, "async probe did not finish within its bounded polling loop")

def finished(s: State) -> IO(Unit):
  State{code, input, output, token} = s
  finished_code(code)

def cycle(fuel: Nat, state: State) -> IO(Unit):
  match fuel:
    case 0n: finished(state)
    case 1n+n:
      do IO<Unit>:
        s : State <- step(state)
        IO.sleep(1)
        cycle(n, s)

def selected(enabled: Bool) -> IO(Unit):
  match enabled:
    case False{}: IO.die(Unit, 2, "async probe requires DEEPFIN_BEND_ASYNC=1")
    case True{}: cycle(1000n, State{0, Array.new(F32, 14n, 2.0), Array.new(F32, 11n, 0.0), 0})

def main() -> IO(Unit):
  do IO<Unit>:
    enabled : Bool <- Async.Async.enabled()
    selected(enabled)
