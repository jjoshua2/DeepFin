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
        +action : U32 <- Chess.legal_at(handle, slot)
        Best{next_action, next_score} = better(
          action,
          puct_score(action),
          best_action,
          best_score
        )
        scan_moves(
          handle,
          p,
          U32.add(slot, 1),
          next_action,
          next_score,
          checksum(sum, action)
        )

def choose_n(+handle: U32, n: Nat) -> IO(Scan):
  match n:
    case 0n:
      IO.pure(Scan, Scan{4294967295, 0})
    case 1n+p:
      do IO<Scan>:
        +first : U32 <- Chess.legal_at(handle, 0)
        scan_moves(
          handle,
          p,
          1,
          first,
          puct_score(first),
          checksum(0, first)
        )

def choose(handle: U32, count: U32) -> IO(Scan):
  choose_n(handle, U32.to_nat(count))

def probe_fixture(+fixture: U32) -> IO(Unit):
  do IO<Unit>:
    +handle : U32 <- Chess.new(fixture)
    +count : U32 <- Chess.legal_count(handle)
    selected : Scan <- choose(handle, count)
    Scan{+best, sum} = selected
    +child : U32 <- Chess.push(handle, best)
    +child_count : U32 <- Chess.legal_count(child)
    child_selected : Scan <- choose(child, child_count)
    Scan{_, child_sum} = child_selected
    check : U32 <- Chess.in_check(child)
    hash : U32 <- Chess.hash32(child)
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

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- probe_fixture(0)
    b : Unit <- probe_fixture(1)
    c : Unit <- probe_fixture(2)
    d : Unit <- probe_fixture(3)
    e : Unit <- probe_fixture(4)
    return Unit{}
