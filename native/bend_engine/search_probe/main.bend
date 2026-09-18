import Base

# PR 4: Bend-owned root Gumbel sequential-halving probe.
#
# CBoard supplies the real legal policy actions. Everything after that is Bend:
# deterministic prior/Gumbel/fake-value generation, top-k candidate sampling,
# visit-budget allocation, completed-by-mix-value scoring, halving, and final
# survivor/visit accounting.

type Cand is Data:
  Cand{action: U32, visits: U32, active: Bool}

type Ranked is Data:
  Ranked{action: U32, visits: U32, score: F32}

type Seeded is Data:
  Seeded{cands: List<Cand>, sampled: U32, seed_xor: U32}

type Counted is Data:
  Counted{cands: List<Cand>, active: U32}

type Agg is Data:
  Agg{
    cands: List<Cand>,
    total_visits: U32,
    sum_prior: F32,
    weighted_q: F32,
    max_visit: U32
  }

type QRange is Data:
  QRange{cands: List<Cand>, min_q: F32, max_q: F32}

type RoundOut is Data:
  RoundOut{cands: List<Cand>, budget: U32}

type TailStats is Data:
  TailStats{total_visits: U32, visit_xor: U32, active: U32}

type SearchResult is Data:
  SearchResult{
    winner: U32,
    active: U32,
    budget: U32,
    total_visits: U32,
    visit_xor: U32
  }

# Reuse the PR-1 CBoard effect surface. The linked bridge owns U64 board state.
def Chess.call(op: U32, a: U32, b: U32) -> IO(U32):
  import "../probe/chess_call.c"

def Chess.legal_moves(handle: U32) -> IO(List<U32>):
  import "../probe/chess_call.c"

def Chess.new(fixture: U32) -> IO(U32):
  Chess.call(0, fixture, 0)

def Chess.legal_count(handle: U32) -> IO(U32):
  Chess.call(1, handle, 0)

# Deterministic fake evaluator. The values are intentionally functions only of
# the real DeepFin policy action, so Python can reproduce them exactly.
def fake_prior(+action: U32) -> F32:
  F32.div(U32.to_f32(U32.add(U32.mod(action, 29), 1)), 30.0)

def fake_gumbel(+action: U32) -> F32:
  base = F32.div(
    F32.sub(
      U32.to_f32(U32.mod(U32.mul(action, 17), 101)),
      50.0
    ),
    25.0
  )
  # Tiny deterministic tie-breaker keeps the sort total without changing the
  # shape of the synthetic Gumbel term.
  F32.add(base, F32.div(U32.to_f32(action), 100000.0))

def fake_q(+action: U32) -> F32:
  F32.div(
    F32.sub(
      U32.to_f32(U32.mod(U32.mul(action, 13), 61)),
      30.0
    ),
    30.0
  )

def initial_score(+action: U32) -> F32:
  F32.add(fake_gumbel(action), F32.log(fake_prior(action)))

def visit_word(+action: U32, visits: U32) -> U32:
  U32.xor(U32.mul(action, 2654435761), visits)

def rank_initial(xs: List<U32>) -> List<Ranked>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+action, rest}:
      Con{
        Ranked{action, 0, initial_score(action)},
        rank_initial(rest)
      }

def insert_ranked(item: Ranked, xs: List<Ranked>) -> List<Ranked>:
  match item:
    case Ranked{action, visits, +score}:
      match xs:
        case Nil{}:
          Con{Ranked{action, visits, score}, Nil{}}
        case Con{head, rest}:
          match head:
            case Ranked{h_action, h_visits, +h_score}:
              gt = F32.is_gt(score, h_score)
              match gt:
                case True{}:
                  Con{
                    Ranked{action, visits, score},
                    Con{Ranked{h_action, h_visits, h_score}, rest}
                  }
                case False{}:
                  Con{
                    Ranked{h_action, h_visits, h_score},
                    insert_ranked(Ranked{action, visits, score}, rest)
                  }

def sort_ranked(xs: List<Ranked>) -> List<Ranked>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{item, rest}:
      sorted = sort_ranked(rest)
      insert_ranked(item, sorted)

def mark_inactive(xs: List<Ranked>) -> List<Cand>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{item, rest}:
      match item:
        case Ranked{action, visits, score}:
          Con{Cand{action, visits, False{}}, mark_inactive(rest)}

def mark_keep(xs: List<Ranked>, keep: Nat) -> List<Cand>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{item, rest}:
      match item:
        case Ranked{action, visits, score}:
          match keep:
            case 0n:
              Con{Cand{action, visits, False{}}, mark_inactive(rest)}
            case 1n+p:
              Con{Cand{action, visits, True{}}, mark_keep(rest, p)}

def seed_cons(cand: Cand, tail: Seeded) -> Seeded:
  match cand:
    case Cand{+action, visits, +active}:
      match tail:
        case Seeded{cands, sampled, seed_xor}:
          add = Bool.pick(U32, active, 1, 0)
          word = Bool.pick(U32, active, U32.mul(action, 2246822519), 0)
          Seeded{
            Con{Cand{action, visits, active}, cands},
            U32.add(sampled, add),
            U32.xor(seed_xor, word)
          }

def summarize_seed(xs: List<Cand>) -> Seeded:
  match xs:
    case Nil{}:
      Seeded{Nil{}, 0, 0}
    case Con{cand, rest}:
      tail = summarize_seed(rest)
      seed_cons(cand, tail)

def seed_candidates(moves: List<U32>) -> Seeded:
  ranked = sort_ranked(rank_initial(moves))
  summarize_seed(mark_keep(ranked, 8n))

def count_cons(cand: Cand, tail: Counted) -> Counted:
  match cand:
    case Cand{action, visits, +active}:
      match tail:
        case Counted{cands, count}:
          Counted{
            Con{Cand{action, visits, active}, cands},
            U32.add(count, Bool.pick(U32, active, 1, 0))
          }

def count_active(xs: List<Cand>) -> Counted:
  match xs:
    case Nil{}:
      Counted{Nil{}, 0}
    case Con{cand, rest}:
      tail = count_active(rest)
      count_cons(cand, tail)

def visit_all(xs: List<Cand>, +vpa: U32) -> List<Cand>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{cand, rest}:
      match cand:
        case Cand{action, visits, +active}:
          add = Bool.pick(U32, active, vpa, 0)
          Con{
            Cand{action, U32.add(visits, add), active},
            visit_all(rest, vpa)
          }

def agg_cons(cand: Cand, tail: Agg) -> Agg:
  match cand:
    case Cand{+action, +visits, active}:
      match tail:
        case Agg{cands, total, sum_prior, weighted, max_visit}:
          prior = fake_prior(action)
          q = fake_q(action)
          visited = U32.is_gt(visits, 0)
          p_add = Bool.pick(F32, visited, prior, 0.0)
          w_add = Bool.pick(F32, visited, F32.mul(prior, q), 0.0)
          Agg{
            Con{Cand{action, visits, active}, cands},
            U32.add(total, visits),
            F32.add(sum_prior, p_add),
            F32.add(weighted, w_add),
            U32.max(max_visit, visits)
          }

def aggregate(xs: List<Cand>) -> Agg:
  match xs:
    case Nil{}:
      Agg{Nil{}, 0, 0.0, 0.0, 0}
    case Con{cand, rest}:
      tail = aggregate(rest)
      agg_cons(cand, tail)

def range_cons(cand: Cand, tail: QRange, +mixed: F32) -> QRange:
  match cand:
    case Cand{+action, +visits, active}:
      match tail:
        case QRange{cands, lo, hi}:
          q = Bool.pick(F32, U32.is_gt(visits, 0), fake_q(action), mixed)
          QRange{
            Con{Cand{action, visits, active}, cands},
            F32.min(lo, q),
            F32.max(hi, q)
          }

def completed_range(xs: List<Cand>, +mixed: F32) -> QRange:
  match xs:
    case Nil{}:
      QRange{Nil{}, 1.0e30, -1.0e30}
    case Con{cand, rest}:
      tail = completed_range(rest, mixed)
      range_cons(cand, tail, mixed)

def rank_round(
  xs: List<Cand>,
  +mixed: F32,
  +min_q: F32,
  +denom: F32,
  +q_scale: F32
) -> List<Ranked>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{cand, rest}:
      match cand:
        case Cand{+action, +visits, active}:
          q = Bool.pick(F32, U32.is_gt(visits, 0), fake_q(action), mixed)
          q_bar = F32.div(F32.sub(q, min_q), denom)
          live_score = F32.add(
            F32.add(fake_gumbel(action), F32.log(fake_prior(action))),
            F32.mul(q_scale, q_bar)
          )
          score = Bool.pick(F32, active, live_score, -1.0e30)
          Con{
            Ranked{action, visits, score},
            rank_round(rest, mixed, min_q, denom, q_scale)
          }

def rounds_for_count(+n: U32) -> U32:
  Bool.pick(
    U32,
    U32.is_gt(n, 4),
    3,
    Bool.pick(U32, U32.is_gt(n, 2), 2, 1)
  )

def round_after_range(
  qr: QRange,
  active: U32,
  budget: U32,
  vpa: U32,
  max_visit: U32,
  mixed: F32
) -> RoundOut:
  match qr:
    case QRange{cands, min_q, max_q}:
      denom = F32.max(F32.sub(max_q, min_q), 0.00000001)
      q_scale = F32.mul(0.1, F32.add(50.0, U32.to_f32(max_visit)))
      ranked = sort_ranked(rank_round(cands, mixed, min_q, denom, q_scale))
      keep = U32.div(U32.add(active, 1), 2)
      used = U32.mul(vpa, active)
      remaining = Bool.pick(
        U32,
        U32.is_gt(used, budget),
        0,
        U32.sub(budget, used)
      )
      RoundOut{mark_keep(ranked, U32.to_nat(keep)), remaining}

def round_after_agg(
  agg: Agg,
  active: U32,
  budget: U32,
  vpa: U32
) -> RoundOut:
  match agg:
    case Agg{cands, total, sum_prior, weighted, max_visit}:
      weighted_q = F32.div(weighted, sum_prior)
      mixed = F32.div(
        F32.add(0.125, F32.mul(U32.to_f32(total), weighted_q)),
        U32.to_f32(U32.add(total, 1))
      )
      qr = completed_range(cands, mixed)
      round_after_range(qr, active, budget, vpa, max_visit, mixed)

def round_work(cands: List<Cand>, +active: U32, +budget: U32) -> RoundOut:
  rounds = rounds_for_count(active)
  denom = U32.mul(active, rounds)
  vpa = U32.max(1, U32.div(budget, denom))
  visited = visit_all(cands, vpa)
  agg = aggregate(visited)
  round_after_agg(agg, active, budget, vpa)

def round_from_count(counted: Counted, budget: U32) -> RoundOut:
  match counted:
    case Counted{cands, active}:
      go = U32.is_gt(active, 1)
      match go:
        case True{}:
          round_work(cands, active, budget)
        case False{}:
          RoundOut{cands, budget}

def round_once(cands: List<Cand>, budget: U32) -> RoundOut:
  counted = count_active(cands)
  round_from_count(counted, budget)

def tail_stats(
  xs: List<Cand>,
  total: U32,
  xor: U32,
  active_count: U32
) -> TailStats:
  match xs:
    case Nil{}:
      TailStats{total, xor, active_count}
    case Con{cand, rest}:
      match cand:
        case Cand{+action, +visits, active}:
          tail_stats(
            rest,
            U32.add(total, visits),
            U32.xor(xor, visit_word(action, visits)),
            U32.add(active_count, Bool.pick(U32, active, 1, 0))
          )

def finish(cands: List<Cand>, budget: U32) -> SearchResult:
  match cands:
    case Nil{}:
      SearchResult{4294967295, 0, budget, 0, 0}
    case Con{head, rest}:
      match head:
        case Cand{+action, +visits, active}:
          stats = tail_stats(
            rest,
            visits,
            visit_word(action, visits),
            Bool.pick(U32, active, 1, 0)
          )
          match stats:
            case TailStats{total, xor, active_count}:
              SearchResult{action, active_count, budget, total, xor}

def search_three(cands: List<Cand>) -> SearchResult:
  r1 = round_once(cands, 64)
  match r1:
    case RoundOut{c1, b1}:
      r2 = round_once(c1, b1)
      match r2:
        case RoundOut{c2, b2}:
          r3 = round_once(c2, b2)
          match r3:
            case RoundOut{c3, b3}:
              finish(c3, b3)

def print_result(
  fixture: U32,
  legal: U32,
  sampled: U32,
  seed_xor: U32,
  result: SearchResult
) -> IO(Unit):
  match result:
    case SearchResult{winner, active, budget, total_visits, visit_xor}:
      IO.print(
        "fixture=" ++ U32.show(fixture)
        ++ " legal=" ++ U32.show(legal)
        ++ " sampled=" ++ U32.show(sampled)
        ++ " seed_xor=" ++ U32.show(seed_xor)
        ++ " winner=" ++ U32.show(winner)
        ++ " active=" ++ U32.show(active)
        ++ " budget=" ++ U32.show(budget)
        ++ " total_visits=" ++ U32.show(total_visits)
        ++ " visit_xor=" ++ U32.show(visit_xor)
      )

def probe_after_moves(
  fixture: U32,
  legal: U32,
  moves: List<U32>
) -> IO(Unit):
  seeded = seed_candidates(moves)
  match seeded:
    case Seeded{cands, sampled, seed_xor}:
      result = search_three(cands)
      print_result(fixture, legal, sampled, seed_xor, result)

def probe_after_count(
  fixture: U32,
  +handle: U32,
  legal: U32
) -> IO(Unit):
  do IO<Unit>:
    moves : List<U32> <- Chess.legal_moves(handle)
    probe_after_moves(fixture, legal, moves)

def probe_after_handle(fixture: U32, +handle: U32) -> IO(Unit):
  do IO<Unit>:
    legal : U32 <- Chess.legal_count(handle)
    probe_after_count(fixture, handle, legal)

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
