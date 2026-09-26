# Test-only caller: carry an owned array through explicit failure, abort and retry.
# No allocator/OOM injection, external node publication, or engine integration.
import Base
import ./IdRegistry.bend as R

type Stage is Type:
  Stage{reservation: R.Reservation, payload: Array<U64>}

def key_a() -> U64:
  U64{0, 2147483648}

def key_b() -> U64:
  U64{4294967295, 4294967295}

def next_text(next_id: Maybe<&2, U32>) -> String:
  match next_id:
    case None{}: "none"
    case Some{id}: U32.show(id)

def state(prefix: String, result: R.Registry & R.Info) -> IO(R.Registry):
  (registry, info) = result
  R.Info{size, next_id} = info
  do IO<R.Registry>:
    IO.print(prefix ++ " " ++ U32.show(size) ++ " " ++ next_text(next_id))
    return registry

def lookup(prefix: String, result: R.Registry & Maybe<&2, U32>) -> IO(R.Registry):
  (registry, value) = result
  match value:
    case None{}:
      do IO<R.Registry>:
        IO.print(prefix ++ " missing")
        return registry
    case Some{id}:
      do IO<R.Registry>:
        IO.print(prefix ++ " value " ++ U32.show(id))
        return registry

def word(prefix: String, result: Array<U64> & U64) -> IO(Array<U64>):
  (payload, +value) = result
  do IO<Array<U64>>:
    IO.print(prefix ++ " " ++ U32.show(U64.high(value)) ++ " " ++ U32.show(U64.low(value)))
    return payload

def known(payload: Array<U64>, result: R.Preparation) -> IO(Unit):
  match result:
    case R.Prepared{reservation}: IO.die(Unit, 2, "published key lost")
    case R.Settled{registry, outcome}:
      match outcome:
        case R.Known{id}:
          do IO<Unit>:
            IO.print("known " ++ U32.show(id))
            registry : R.Registry <- state("final", R.inspect(registry))
            payload : Array<U64> <- word("payload-first", Array.get(U64, payload, 0))
            payload : Array<U64> <- word("payload-last", Array.get(U64, payload, 3))
            IO.print("end")
        case _: IO.die(Unit, 2, "published key did not settle as known")

def committed(payload: Array<U64>, result: R.Registry & R.Outcome) -> IO(Unit):
  (registry, outcome) = result
  match outcome:
    case R.Assigned{id}:
      do IO<Unit>:
        IO.print("assigned " ++ U32.show(id))
        registry : R.Registry <- state("committed", R.inspect(registry))
        registry : R.Registry <- lookup("old-key", R.get(registry, key_a()))
        registry : R.Registry <- lookup("new-key", R.get(registry, key_b()))
        known(payload, R.reserve(registry, key_b()))
    case _: IO.die(Unit, 2, "valid reservation did not commit")

def publish(stage: Stage) -> IO(Unit):
  Stage{reservation, payload} = stage
  committed(payload, R.commit(reservation))

def retry(payload: Array<U64>, result: R.Preparation) -> IO(Unit):
  match result:
    case R.Settled{registry, outcome}: IO.die(Unit, 2, "aborted ID unavailable for retry")
    case R.Prepared{reservation}: publish(Stage{reservation, payload})

def abort_stage(stage: Stage) -> R.Registry & Array<U64>:
  Stage{reservation, payload} = stage
  (R.abort(reservation), payload)

def aborted(result: R.Registry & Array<U64>) -> IO(Unit):
  (registry, payload) = result
  do IO<Unit>:
    registry : R.Registry <- state("aborted", R.inspect(registry))
    registry : R.Registry <- lookup("unpublished", R.get(registry, key_a()))
    # Explicit caller failure returned the actual payload owner. Reuse that
    # allocation for another key, with an observable private payload update.
    retry(Array.set(U64, payload, 3, U64{305419896, 4294967295}), R.reserve(registry, key_b()))

def prepared(result: R.Reservation & U32) -> IO(Unit):
  (reservation, +id) = result
  do IO<Unit>:
    IO.print("reserved " ++ U32.show(id))
    aborted(abort_stage(Stage{reservation, Array.new(U64, 2n, U64{id, U32.xor(id, 2779096485)})}))

def reservation(result: R.Preparation) -> IO(Unit):
  match result:
    case R.Settled{registry, outcome}: IO.die(Unit, 2, "fresh registry did not reserve")
    case R.Prepared{owner}: prepared(R.candidate(owner))

def created(result: Maybe<&1, R.Registry>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "valid registry construction failed")
    case Some{registry}: reservation(R.reserve(registry, key_a()))

def cases(ids: List<&2, U32>) -> IO(Unit):
  match ids:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{+id, tail}:
      do IO<Unit>:
        IO.print("case " ++ U32.show(id))
        done : Unit <- created(R.new(2, id))
        cases(tail)

def main() -> IO(Unit):
  cases([0, 7, 4294967294, 4294967295])
