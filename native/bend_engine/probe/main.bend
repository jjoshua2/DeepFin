import Base

# A deliberately tiny Bend-owned search slice over DeepFin's native CBoard.
#
# The C side owns opaque board handles and all U64 chess mechanics. Bend sees
# only U32 handles/action ids and performs the search scoring itself. This is
# intentionally not a production engine path; it is the first architecture
# probe for a future shared Bend search core.

type Scan is Data:
  Scan{best: U32, checksum: U32}

# Scalar op codes private to this probe:
#   0 new_fixture(fixture)
#   1 legal_count(handle)
#   3 push(handle, action) -> cloned child handle
#   4 in_check(handle)
#   5 zobrist_hash_low32(handle)
#   6 feature_action(fixture) -> castle/ep action, or UINT32_MAX
def Chess.call(op: U32, a: U32, b: U32) -> IO(U32):
  import "./chess_call.c"

# The whole sorted move list crosses the effect boundary at once. This is a
# better approximation of the intended engine boundary than one IO call per
# child, and leaves the recursive scan entirely pure Bend.
def Chess.legal_moves(handle: U32) -> IO(List<U32>):
  import "./chess_call.c"

def Chess.new(fixture: U32) -> IO(U32):
  Chess.call(0, fixture, 0)

def Chess.legal_count(handle: U32) -> IO(U32):
  Chess.call(1, handle, 0)

def Chess.push(handle: U32, action: U32) -> IO(U32):
  Chess.call(3, handle, action)

def Chess.in_check(handle: U32) -> IO(U32):
  Chess.call(4, handle, 0)

def Chess.hash32(handle: U32) -> IO(U32):
  Chess.call(5, handle, 0)

def Chess.feature_action(fixture: U32) -> IO(U32):
  Chess.call(6, fixture, 0)

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

# Bend only matches parameters/fields, not computed expressions. Keep the
# branch in tiny helpers: the computed comparison is passed as their parameter.
def pick_action(gt: Bool, action: U32, best: U32) -> U32:
  match gt:
    case True{}:
      action
    case False{}:
      best

def pick_score(gt: Bool, score: F32, best: F32) -> F32:
  match gt:
    case True{}:
      score
    case False{}:
      best

def scan_moves(
  xs: List<U32>,
  best_action: U32,
  +best_score: F32,
  sum: U32
) -> Scan:
  match xs:
    case Nil{}:
      Scan{best_action, sum}
    case Con{+action, rest}:
      next_action = pick_action(
        F32.is_gt(puct_score(action), best_score),
        action,
        best_action
      )
      next_score = pick_score(
        F32.is_gt(puct_score(action), best_score),
        puct_score(action),
        best_score
      )
      scan_moves(
        rest,
        next_action,
        next_score,
        checksum(sum, action)
      )

def choose(xs: List<U32>) -> Scan:
  match xs:
    case Nil{}:
      Scan{4294967295, 0}
    case Con{+first, rest}:
      scan_moves(
        rest,
        first,
        puct_score(first),
        checksum(0, first)
      )

def checksum_moves(xs: List<U32>, acc: U32) -> U32:
  match xs:
    case Nil{}:
      acc
    case Con{action, rest}:
      checksum_moves(rest, checksum(acc, action))

def probe_feature_after_hash(
  fixture: U32,
  action: U32,
  check: U32,
  hash: U32
) -> IO(Unit):
  IO.print(
    "feature_fixture=" ++ U32.show(fixture)
    ++ " feature_action=" ++ U32.show(action)
    ++ " feature_check=" ++ U32.show(check)
    ++ " feature_hash32=" ++ U32.show(hash)
  )

def probe_feature_after_check(
  fixture: U32,
  action: U32,
  +child: U32,
  check: U32
) -> IO(Unit):
  do IO<Unit>:
    hash : U32 <- Chess.hash32(child)
    probe_feature_after_hash(fixture, action, check, hash)

def probe_feature_apply(
  fixture: U32,
  action: U32,
  +child: U32
) -> IO(Unit):
  do IO<Unit>:
    check : U32 <- Chess.in_check(child)
    probe_feature_after_check(fixture, action, child, check)

def probe_feature(
  fixture: U32,
  +handle: U32,
  +action: U32
) -> IO(Unit):
  do IO<Unit>:
    child : U32 <- Chess.push(handle, action)
    probe_feature_apply(fixture, action, child)

def probe_feature_if(
  has: Bool,
  fixture: U32,
  +handle: U32,
  +action: U32
) -> IO(Unit):
  match has:
    case True{}:
      probe_feature(fixture, handle, action)
    case False{}:
      do IO<Unit>:
        return Unit{}

def probe_after_hash(
  +fixture: U32,
  +handle: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32,
  child_sum: U32,
  check: U32,
  hash: U32
) -> IO(Unit):
  do IO<Unit>:
    printed : Unit <- IO.print(
      "fixture=" ++ U32.show(fixture)
      ++ " legal=" ++ U32.show(count)
      ++ " best=" ++ U32.show(best)
      ++ " checksum=" ++ U32.show(sum)
      ++ " child_legal=" ++ U32.show(child_count)
      ++ " child_checksum=" ++ U32.show(child_sum)
      ++ " child_check=" ++ U32.show(check)
      ++ " child_hash32=" ++ U32.show(hash)
    )
    +feature : U32 <- Chess.feature_action(fixture)
    probe_feature_if(
      U32.is_ne(feature, 4294967295),
      fixture,
      handle,
      feature
    )

def probe_after_check(
  fixture: U32,
  +handle: U32,
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
      fixture, handle, count, best, sum, child_count, child_sum, check, hash
    )

def probe_after_child_moves(
  fixture: U32,
  +handle: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32,
  child_moves: List<U32>
) -> IO(Unit):
  do IO<Unit>:
    check : U32 <- Chess.in_check(child)
    probe_after_check(
      fixture,
      handle,
      child,
      count,
      best,
      sum,
      child_count,
      checksum_moves(child_moves, 0),
      check
    )

def probe_after_child_count(
  fixture: U32,
  +handle: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32,
  child_count: U32
) -> IO(Unit):
  do IO<Unit>:
    child_moves : List<U32> <- Chess.legal_moves(child)
    probe_after_child_moves(
      fixture, handle, child, count, best, sum, child_count, child_moves
    )

def probe_after_push(
  fixture: U32,
  +handle: U32,
  +child: U32,
  count: U32,
  best: U32,
  sum: U32
) -> IO(Unit):
  do IO<Unit>:
    child_count : U32 <- Chess.legal_count(child)
    probe_after_child_count(
      fixture, handle, child, count, best, sum, child_count
    )

def probe_after_selected(
  fixture: U32,
  +handle: U32,
  count: U32,
  selected: Scan
) -> IO(Unit):
  match selected:
    case Scan{+best, sum}:
      do IO<Unit>:
        child : U32 <- Chess.push(handle, best)
        probe_after_push(fixture, handle, child, count, best, sum)

def probe_after_moves(
  fixture: U32,
  +handle: U32,
  count: U32,
  moves: List<U32>
) -> IO(Unit):
  probe_after_selected(fixture, handle, count, choose(moves))

def probe_after_count(
  fixture: U32,
  +handle: U32,
  count: U32
) -> IO(Unit):
  do IO<Unit>:
    moves : List<U32> <- Chess.legal_moves(handle)
    probe_after_moves(fixture, handle, count, moves)

def probe_after_handle(fixture: U32, +handle: U32) -> IO(Unit):
  do IO<Unit>:
    count : U32 <- Chess.legal_count(handle)
    probe_after_count(fixture, handle, count)

def probe_fixture(+fixture: U32) -> IO(Unit):
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
