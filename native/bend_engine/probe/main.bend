import Base

# A deliberately tiny Bend-owned search slice over DeepFin's native CBoard.
#
# The C side owns opaque board handles and all U64 chess mechanics. Bend sees
# only U32 handles/action ids and performs the search scoring itself. This is
# intentionally not a production engine path; it is the first architecture
# probe for a future shared Bend search core.

type Best is Data:
  Best{action: U32, score: F32}

type Scan is Data:
  Scan{best: U32, checksum: U32}

# op codes are private to this probe:
#   0 new_fixture(fixture)
#   1 legal_count(handle)
#   2 legal_at(handle, sorted_slot)
#   3 push(handle, action) -> cloned child handle
#   4 in_check(handle)
#   5 zobrist_hash_low32(handle)
def Chess.call(op: U32, a: U32, b: U32) -> IO(U32):
  import "./chess_call.c"

def Chess.new(fixture: U32) -> IO(U32):
  Chess.call(0, fixture, 0)

def Chess.legal_count(handle: U32) -> IO(U32):
  Chess.call(1, handle, 0)

def Chess.legal_at(handle: U32, slot: U32) -> IO(U32):
  Chess.call(2, handle, slot)

def Chess.push(handle: U32, action: U32) -> IO(U32):
  Chess.call(3, handle, action)

def Chess.in_check(handle: U32) -> IO(U32):
  Chess.call(4, handle, 0)

def Chess.hash32(handle: U32) -> IO(U32):
  Chess.call(5, handle, 0)

# Deterministic synthetic statistics. They make the selector exercise the same
# shape as PUCT without involving a network in PR 1. Every operation is F32 so
# the Python parity oracle can reproduce the rounding exactly.
def puct_score(+action: U32) -> F32:
  prior = F32.div(U32.to_f32(U32.add(U32.mod(action, 97), 1)), 100.0)
  q = F32.div(U32.to_f32(U32.mod(action, 37)), 100.0)
  visits = U32.mod(action, 11)
  explore = F32.div(
    F32.mul(1.5, F32.mul(prior, F32.sqrt(128.0))),
    U32.to_f32(U32.add(visits, 1))
  )
  F32.add(q, explore)

def checksum(acc: U32, action: U32) -> U32:
  U32.xor(U32.mul(acc, 2654435761), U32.add(action, 1))

def better_case(
  gt: Bool,
  action: U32,
  score: F32,
  best_action: U32,
  best_score: F32
) -> Best:
  match gt:
    case True{}:
      Best{action, score}
    case False{}:
      Best{best_action, best_score}

def better(
  action: U32,
  +score: F32,
  best_action: U32,
  +best_score: F32
) -> Best:
  better_case(
    F32.is_gt(score, best_score),
    action,
    score,
    best_action,
    best_score
  )

# Bend 2.0.4 does not permit a reusable (+) pattern directly on an IO bind.
# Bind each effect result affinely, then pass it once into a helper whose
# top-level parameter is reusable. That keeps effect values linear while still
# allowing the pure search step to use an action for score + checksum.
def scan_after_best(
  handle: U32,
  n: Nat,
  slot: U32,
  action: U32,
  sum: U32,
  candidate: Best
) -> IO(Scan):
  match candidate:
    case Best{next_action, next_score}:
      scan_moves(
        handle,
        n,
        U32.add(slot, 1),
        next_action,
        next_score,
        checksum(sum, action)
      )

def scan_after_action(
  handle: U32,
  n: Nat,
  slot: U32,
  +action: U32,
  best_action: U32,
  best_score: F32,
  sum: U32
) -> IO(Scan):
  scan_after_best(
    handle,
    n,
    slot,
    action,
    sum,
    better(
      action,
      puct_score(action),
      best_action,
      best_score
    )
  )

def scan_moves(
  +handle: U32,
  n: Nat,
  +slot: U32,
  best_action: U32,
  best_score: F32,
  sum: U32
) -> IO(Scan):
  match n:
    case 0n:
      IO.pure(Scan, Scan{best_action, sum})
    case 1n+p:
      do IO<Scan>:
        action : U32 <- Chess.legal_at(handle, slot)
        scan_after_action(
          handle,
          p,
          slot,
          action,
          best_action,
          best_score,
          sum
        )

def choose_after_first(handle: U32, n: Nat, +first: U32) -> IO(Scan):
  scan_moves(
    handle,
    n,
    1,
    first,
    puct_score(first),
    checksum(0, first)
  )

def choose_n(+handle: U32, n: Nat) -> IO(Scan):
  match n:
    case 0n:
      IO.pure(Scan, Scan{4294967295, 0})
    case 1n+p:
      do IO<Scan>:
        first : U32 <- Chess.legal_at(handle, 0)
        choose_after_first(handle, p, first)

def choose(handle: U32, count: U32) -> IO(Scan):
  choose_n(handle, U32.to_nat(count))

def probe_after_hash(
  fixture: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32,
  child_sum: U32,
  check: U32,
  hash: U32
) -> IO(Unit):
  IO.print(
    "fixture=" ++ U32.show(fixture)
    ++ " legal=" ++ U32.show(count)
    ++ " best=" ++ U32.show(best)
    ++ " checksum=" ++ U32.show(sum)
    ++ " child_legal=" ++ U32.show(child_count)
    ++ " child_checksum=" ++ U32.show(child_sum)
    ++ " child_check=" ++ U32.show(check)
    ++ " child_hash32=" ++ U32.show(hash)
  )

def probe_after_check(
  fixture: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32,
  child_sum: U32,
  check: U32
) -> IO(Unit):
  do IO<Unit>:
    hash : U32 <- Chess.hash32(child)
    probe_after_hash(
      fixture, count, best, sum, child_count, child_sum, check, hash
    )

def probe_after_child_scan(
  fixture: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32,
  child_selected: Scan
) -> IO(Unit):
  Scan{_, child_sum} = child_selected
  do IO<Unit>:
    check : U32 <- Chess.in_check(child)
    probe_after_check(
      fixture, child, count, best, sum, child_count, child_sum, check
    )

def probe_after_child_count(
  fixture: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32,
  +child_count: U32
) -> IO(Unit):
  do IO<Unit>:
    child_selected : Scan <- choose(child, child_count)
    probe_after_child_scan(
      fixture, child, count, best, sum, child_count, child_selected
    )

def probe_after_push(
  fixture: U32,
  child: U32,
  count: U32,
  best: U32,
  sum: U32
) -> IO(Unit):
  do IO<Unit>:
    child_count : U32 <- Chess.legal_count(child)
    probe_after_child_count(fixture, child, count, best, sum, child_count)

def probe_after_selected(
  fixture: U32,
  +handle: U32,
  count: U32,
  selected: Scan
) -> IO(Unit):
  Scan{best, sum} = selected
  do IO<Unit>:
    child : U32 <- Chess.push(handle, best)
    probe_after_push(fixture, child, count, best, sum)

def probe_after_count(
  fixture: U32,
  +handle: U32,
  +count: U32
) -> IO(Unit):
  do IO<Unit>:
    selected : Scan <- choose(handle, count)
    probe_after_selected(fixture, handle, count, selected)

def probe_after_handle(fixture: U32, handle: U32) -> IO(Unit):
  do IO<Unit>:
    count : U32 <- Chess.legal_count(handle)
    probe_after_count(fixture, handle, count)

def probe_fixture(fixture: U32) -> IO(Unit):
  do IO<Unit>:
    handle : U32 <- Chess.new(fixture)
    probe_after_handle(fixture, handle)

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- probe_fixture(0)
    b : Unit <- probe_fixture(1)
    c : Unit <- probe_fixture(2)
    d : Unit <- probe_fixture(3)
    e : Unit <- probe_fixture(4)
    return Unit{}
