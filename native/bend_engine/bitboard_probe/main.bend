import Base
import ./Sliders.bend as Sliders

# Full bitboards enter once per square. No IO occurs in the lookup loop.
type Fixture is Type:
  Fixture{mask: U64, magic: U64, shift: Nat, entries: U32, count: U32,
    data: Array<U64>}

def Fixture.load(bishop: U32, square: U32) -> IO(Fixture):
  import "./fixture_call.c"

type Params is Data:
  Params{mask: U64, magic: U64, shift: Nat, entries: U32}

type Checked is Data:
  Checked{number: U32, failures: U32, first: U32, checksum: U64}

# The small continuations destructure read results, as required by Bend's
# parameter/field-only matches. They do not add foreign-effect boundaries.
def after_magic(
  state: Checked,
  +expected: U64, +pa: U64, bounds: Bool, r: Array<U64> & U64
) -> Array<U64> & Checked:
  Checked{+number, +failures, first, sum} = state
  (data, +ma) = r
  +ok = Bool.and(bounds, Bool.and(U64.is_eq(pa, expected), U64.is_eq(ma, expected)))
  first = Bool.pick(U32, Bool.and(U32.is_zero(failures), Bool.not(ok)), number, first)
  (data, Checked{U32.inc(number), U32.add(failures, Bool.to_u32(Bool.not(ok))), first,
      U64.add(sum, U64.xor(pa, U64.shl(ma)))})

def after_pext(
  +params: Params, state: Checked,
  expected: U64, mi: U32, bounds: Bool, r: Array<U64> & U64
) -> Array<U64> & Checked:
  Params{mask, magic, shift, entries} = params
  (data, pa) = r
  after_magic(state, expected, pa, bounds,
    Sliders.lookup(data, U32.add(8, entries), mi))

def after_expected(
  +params: Params, state: Checked,
  +occupied: U64, r: Array<U64> & U64
) -> Array<U64> & Checked:
  Params{mask, magic, shift, +entries} = params
  (data, expected) = r
  +pi = Sliders.pext_index(occupied, mask)
  +mi = Sliders.magic_index(occupied, mask, magic, shift)
  bounds = Bool.and(U32.is_lt(pi, entries), U32.is_lt(mi, entries))
  after_pext(params, state, expected, mi, bounds, Sliders.lookup(data, 8, pi))

def after_occupied(
  params: Params, at: U32, state: Checked, r: Array<U64> & U64
) -> Array<U64> & Checked:
  (data, occupied) = r
  after_expected(params, state, occupied, Array.get(U64, data, U32.inc(at)))

def step(params: Params, +at: U32, r: Array<U64> & Checked) -> Array<U64> & Checked:
  (data, state) = r
  after_occupied(params, at, state, Array.get(U64, data, at))

def cases(
  n: Nat, +params: Params, +at: U32, r: Array<U64> & Checked
) -> Array<U64> & Checked:
  match n:
    case 0n:
      r
    case 1n+p:
      cases(p, params, U32.add(at, 2), step(params, at, r))

def print_result(ok: Bool, text: String) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(text)
    case False{}:
      IO.die(Unit, 1, "U64 slider mismatch: " ++ text)

def report(bishop: U32, square: U32, r: Array<U64> & Checked) -> IO(Unit):
  (data, state) = r
  match state:
    case Checked{count, +failures, first, +sum}:
      print_result(U32.is_zero(failures), "bishop=" ++ U32.show(bishop) ++ " square=" ++ U32.show(square)
        ++ " cases=" ++ U32.show(count) ++ " failures=" ++ U32.show(failures)
        ++ " first=" ++ U32.show(first) ++ " sum_hi=" ++ U32.show(U64.high(sum))
        ++ " sum_lo=" ++ U32.show(U64.low(sum)))

def check_fixture(bishop: U32, square: U32, f: Fixture) -> IO(Unit):
  Fixture{mask, magic, shift, +entries, count, data} = f
  report(bishop, square,
    cases(U32.to_nat(count), Params{mask, magic, shift, entries},
      U32.add(8, U32.mul(2, entries)), (data, Checked{0, 0, 0, U64.zero()})))

def check_square(+bishop: U32, +square: U32) -> IO(Unit):
  do IO<Unit>:
    f : Fixture <- Fixture.load(bishop, square)
    check_fixture(bishop, square, f)

def squares(n: Nat, +square: U32) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        rook : Unit <- check_square(0, square)
        bishop : Unit <- check_square(1, square)
        squares(p, U32.inc(square))

def main() -> IO(Unit):
  squares(64n, 0)
