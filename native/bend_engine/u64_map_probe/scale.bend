# Bounded full-capacity regression. Generate operations natively, outside env-size limits.
# Calls public map/registry APIs; never writes their storage representation.
import Base
import ./U64Map.bend as M
import ./InsertOnce.bend as I
import ./IdRegistry.bend as R

def key(+i: U32) -> U64:
  U64{U32.xor(i, 2779096485), U32.mul(i, 2654435761)}

def value(i: U32, salt: U32) -> U32:
  U32.xor(U32.mul(i, 2246822519), salt)

def size_line(prefix: String, result: M.Table & U32) -> IO(M.Table):
  (t, size) = result
  do IO<M.Table>:
    IO.print(prefix ++ " " ++ U32.show(size))
    return t

def put_line(prefix: String, result: M.Table & M.Put) -> IO(M.Table):
  (t, outcome) = result
  match outcome:
    case M.Added{}: size_line(prefix ++ " added", M.length(t))
    case M.Replaced{old}: size_line(prefix ++ " replaced " ++ U32.show(old), M.length(t))
    case M.Full{}: size_line(prefix ++ " full", M.length(t))

def get_line(prefix: String, result: M.Table & Maybe<&2, U32>) -> IO(M.Table):
  (t, found) = result
  match found:
    case None{}: size_line(prefix ++ " missing", M.length(t))
    case Some{v}: size_line(prefix ++ " value " ++ U32.show(v), M.length(t))

def once_line(prefix: String, result: M.Table & I.Outcome) -> IO(M.Table):
  (t, outcome) = result
  match outcome:
    case I.Inserted{v}: size_line(prefix ++ " inserted " ++ U32.show(v), M.length(t))
    case I.Existing{v}: size_line(prefix ++ " existing " ++ U32.show(v), M.length(t))
    case I.Saturated{}: size_line(prefix ++ " full", M.length(t))

def map_op(kind: U32, +i: U32, salt: U32, t: M.Table) -> IO(M.Table):
  match kind:
    case 0: put_line("p " ++ U32.show(i), M.put(t, key(i), value(i, salt)))
    case 1: get_line("g " ++ U32.show(i), M.get(t, key(i)))
    case 2: get_line("d " ++ U32.show(i), M.remove(t, key(i)))
    case _: once_line("e " ++ U32.show(i), I.get_or_insert(t, key(i), value(i, salt)))

def map_loop(n: Nat, +kind: U32, +i: U32, +stride: U32, +salt: U32, t: M.Table) -> IO(M.Table):
  match n:
    case 0n: IO.pure(M.Table, t)
    case 1n+p:
      do IO<M.Table>:
        t : M.Table <- map_op(kind, i, salt, t)
        map_loop(p, kind, U32.add(i, stride), stride, salt, t)

def map_run(+n: U32, t: M.Table) -> IO(Unit):
  do IO<Unit>:
    t : M.Table <- map_loop(U32.to_nat(n), 0, 0, 1, 0, t)
    t : M.Table <- map_op(0, n, 0, t)
    t : M.Table <- map_op(1, n, 0, t)
    t : M.Table <- map_loop(U32.to_nat(n), 0, 0, 1, 4294967295, t)
    t : M.Table <- map_loop(U32.to_nat(n), 3, 0, 1, 0, t)
    t : M.Table <- map_op(3, n, 0, t)
    t : M.Table <- map_loop(U32.to_nat(n), 1, 0, 1, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.shr(n)), 2, 0, 2, 0, t)
    t : M.Table <- map_loop(U32.to_nat(n), 1, 0, 1, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.shr(n)), 0, n, 1, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.add(n, U32.shr(n))), 1, 0, 1, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.shr(n)), 2, 1, 2, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.shr(n)), 2, n, 1, 0, t)
    t : M.Table <- map_loop(U32.to_nat(U32.add(n, U32.shr(n))), 1, 0, 1, 0, t)
    t : M.Table <- map_op(0, 0, 7, t)
    t : M.Table <- map_op(3, 0, 0, t)
    t : M.Table <- map_op(1, 0, 0, t)
    t : M.Table <- map_op(2, 0, 0, t)
    t : M.Table <- size_line("map-end", M.length(t))
    IO.pure(Unit, Unit{})

def map_start(+bits: U32, +n: U32, result: Maybe<&1, M.Table>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "scale map constructor failed")
    case Some{t}:
      do IO<Unit>:
        IO.print("map " ++ U32.show(bits) ++ " " ++ U32.show(n))
        map_run(n, t)

def next_text(next_id: Maybe<&2, U32>) -> String:
  match next_id:
    case None{}: "none"
    case Some{id}: U32.show(id)

def registry_line(prefix: String, result: R.Registry & R.Info) -> IO(R.Registry):
  (r, info) = result
  R.Info{size, next_id} = info
  do IO<R.Registry>:
    IO.print(prefix ++ " " ++ U32.show(size) ++ " " ++ next_text(next_id))
    return r

def registry_outcome(prefix: String, result: R.Registry & R.Outcome) -> IO(R.Registry):
  (r, outcome) = result
  match outcome:
    case R.Assigned{id}: registry_line(prefix ++ " assigned " ++ U32.show(id), R.inspect(r))
    case R.Known{id}: registry_line(prefix ++ " known " ++ U32.show(id), R.inspect(r))
    case R.TableFull{}: registry_line(prefix ++ " full", R.inspect(r))
    case R.IdsExhausted{}: registry_line(prefix ++ " exhausted", R.inspect(r))

def registry_get(prefix: String, result: R.Registry & Maybe<&2, U32>) -> IO(R.Registry):
  (r, found) = result
  match found:
    case None{}: registry_line(prefix ++ " missing", R.inspect(r))
    case Some{id}: registry_line(prefix ++ " value " ++ U32.show(id), R.inspect(r))

def reserve_resolve(label: String, accept: Bool, reservation: R.Reservation) -> IO(R.Registry):
  match accept:
    case True{}: registry_outcome(label, R.commit(reservation))
    case False{}: registry_line(label ++ " aborted", R.inspect(R.abort(reservation)))

def reserve_done(prefix: String, accept: Bool, result: R.Reservation & U32) -> IO(R.Registry):
  (reservation, id) = result
  reserve_resolve(prefix ++ " reserved " ++ U32.show(id), accept, reservation)

def reserve_line(prefix: String, accept: Bool, preparation: R.Preparation) -> IO(R.Registry):
  match preparation:
    case R.Settled{r, outcome}: registry_outcome(prefix, (r, outcome))
    case R.Prepared{reservation}: reserve_done(prefix, accept, R.candidate(reservation))

def registry_op(kind: U32, +i: U32, r: R.Registry) -> IO(R.Registry):
  match kind:
    case 0: registry_outcome("r " ++ U32.show(i), R.intern(r, key(i)))
    case 1: registry_get("g " ++ U32.show(i), R.get(r, key(i)))
    case 2: reserve_line("a " ++ U32.show(i), False{}, R.reserve(r, key(i)))
    case _: reserve_line("c " ++ U32.show(i), True{}, R.reserve(r, key(i)))

def registry_loop(n: Nat, +kind: U32, +i: U32, r: R.Registry) -> IO(R.Registry):
  match n:
    case 0n: IO.pure(R.Registry, r)
    case 1n+p:
      do IO<R.Registry>:
        r : R.Registry <- registry_op(kind, i, r)
        registry_loop(p, kind, U32.inc(i), r)

def registry_run(+n: U32, r: R.Registry) -> IO(Unit):
  do IO<Unit>:
    r : R.Registry <- registry_loop(U32.to_nat(U32.sub(n, 1)), 0, 0, r)
    r : R.Registry <- registry_op(2, U32.sub(n, 1), r)
    r : R.Registry <- registry_op(1, U32.sub(n, 1), r)
    r : R.Registry <- registry_op(3, n, r)
    r : R.Registry <- registry_op(1, U32.sub(n, 1), r)
    r : R.Registry <- registry_op(0, U32.sub(n, 1), r)
    r : R.Registry <- registry_loop(U32.to_nat(U32.sub(n, 1)), 3, 0, r)
    r : R.Registry <- registry_loop(U32.to_nat(U32.sub(n, 1)), 1, 0, r)
    r : R.Registry <- registry_op(0, n, r)
    r : R.Registry <- registry_op(1, n, r)
    r : R.Registry <- registry_op(3, U32.inc(n), r)
    r : R.Registry <- registry_line("registry-end", R.inspect(r))
    IO.pure(Unit, Unit{})

def registry_start(first: U32, n: U32, result: Maybe<&1, R.Registry>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "scale registry constructor failed")
    case Some{r}:
      do IO<Unit>:
        IO.print("registry " ++ U32.show(first))
        registry_run(n, r)

def run(+bits: U32, +n: U32) -> IO(Unit):
  do IO<Unit>:
    done : Unit <- map_start(bits, n, M.new(bits))
    done : Unit <- registry_start(0, n, R.new(bits, 0))
    registry_start(U32.sub(4294967295, U32.sub(n, 1)), n,
      R.new(bits, U32.sub(4294967295, U32.sub(n, 1))))

def valid(+bits: U32, ok: Bool) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit, 2, "invalid scale exponent")
    case True{}: run(bits, U32.shln(1, U32.to_nat(U32.sub(bits, 1))))

def config(result: Maybe<&2, U32>) -> IO(Unit):
  match result:
    case None{}: IO.die(Unit, 2, "invalid scale exponent")
    case Some{+bits}: valid(bits, Bool.and(U32.is_ge(bits, 2), U32.is_le(bits, 16)))

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match result:
    case Fail{e}: IO.die(Unit, 2, "missing DEEPFIN_MAP_SCALE_BITS")
    case Done{text}: config(U32.read(text))

def main() -> IO(Unit):
  do IO<Unit>:
    text : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_MAP_SCALE_BITS")
    environment(text)
