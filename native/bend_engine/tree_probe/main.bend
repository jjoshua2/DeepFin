import Base

# PR 5: two-ply Bend-owned MCTS semantics probe.
#
# CBoard supplies real legal chess topology. Bend owns the tree state, PUCT/FPU
# descent, first-visit expansion state, deterministic leaf values, and
# side-to-move alternating backup. The Python oracle drives DeepFin's production
# C MCTSTree with the same topology/priors/values.

type Kid is Data:
  Kid{action: U32, prior: F32, n: U32, w: F32}

type RootChild is Data:
  RootChild{
    action: U32,
    prior: F32,
    n: U32,
    w: F32,
    expanded: Bool,
    kids: List<&2, Kid>
  }

type Tree is Data:
  Tree{root_n: U32, root_w: F32, children: List<&2, RootChild>}

type Best is Data:
  Best{action: U32, score: F32}

type ChildStep is Data:
  ChildStep{child: RootChild, root_delta: F32, path_word: U32}

type ChildrenStep is Data:
  ChildrenStep{
    children: List<&2, RootChild>,
    root_delta: F32,
    path_word: U32
  }

type SimStep is Data:
  SimStep{tree: Tree, path_word: U32}

type Run is Data:
  Run{tree: Tree, path_mix: U32}

type VisitStats is Data:
  VisitStats{
    total: U32,
    xor: U32,
    best_action: U32,
    best_visits: U32,
    expanded: U32
  }

# Existing native chess substrate from PR 1.
def Chess.call(op: U32, a: U32, b: U32) -> IO(U32):
  import "../probe/chess_call.c"

def Chess.legal_moves(handle: U32) -> IO(List<&2, U32>):
  import "../probe/chess_call.c"

def Chess.new(fixture: U32) -> IO(U32):
  Chess.call(0, fixture, 0)

def Chess.push(handle: U32, action: U32) -> IO(U32):
  Chess.call(3, handle, action)

# Exact binary fractions keep C-double and Bend-F32 search decisions stable.
def fake_prior(+action: U32) -> F32:
  F32.div(U32.to_f32(U32.add(U32.mod(action, 4), 1)), 8.0)

# Value from the ROOT CHILD's side-to-move perspective.
def fake_child_value(+action: U32) -> F32:
  F32.div(
    F32.sub(
      U32.to_f32(U32.mod(U32.mul(action, 5), 7)),
      3.0
    ),
    4.0
  )

# Value from the GRANDCHILD's side-to-move perspective.
def fake_grand_value(+root_action: U32, +kid_action: U32) -> F32:
  F32.div(
    F32.sub(
      U32.to_f32(
        U32.mod(
          U32.add(U32.mul(root_action, 3), kid_action),
          7
        )
      ),
      3.0
    ),
    4.0
  )

def take_n(xs: List<&2, U32>, n: Nat) -> List<&2, U32>:
  match xs n:
    case Nil{} k:
      Nil{}
    case Con{item, rest} 0n:
      Nil{}
    case Con{item, rest} 1n+p:
      Con{item, take_n(rest, p)}

def list_len(xs: List<&2, U32>, acc: U32) -> U32:
  match xs:
    case Nil{}:
      acc
    case Con{item, rest}:
      list_len(rest, U32.add(acc, 1))

def kids_from_moves(xs: List<&2, U32>) -> List<&2, Kid>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+action, rest}:
      Con{
        Kid{action, fake_prior(action), 0, 0.0},
        kids_from_moves(rest)
      }

def build_children(
  +root: U32,
  moves: List<&2, U32>,
  n: Nat
) -> IO(List<&2, RootChild>):
  match moves n:
    case Nil{} k:
      IO.pure(List<&2, RootChild>, Nil{})
    case Con{action, rest} 0n:
      IO.pure(List<&2, RootChild>, Nil{})
    case Con{+action, rest} 1n+p:
      do IO<List<&2, RootChild>>:
        child : U32 <- Chess.push(root, action)
        legal : List<&2, U32> <- Chess.legal_moves(child)
        tail : List<&2, RootChild> <- build_children(root, rest, p)
        return Con{
          RootChild{
            action,
            fake_prior(action),
            0,
            0.0,
            False{},
            kids_from_moves(take_n(legal, 4n))
          },
          tail
        }

def node_q_case(visited: Bool, w: F32, n: U32) -> F32:
  match visited:
    case True{}:
      F32.div(w, U32.to_f32(n))
    case False{}:
      0.0

def node_q(+n: U32, w: F32) -> F32:
  node_q_case(U32.is_gt(n, 0), w, n)

def root_visited_cons(child: RootChild, tail: F32) -> F32:
  match child:
    case RootChild{action, prior, n, w, expanded, kids}:
      F32.add(
        Bool.pick(F32, U32.is_gt(n, 0), prior, 0.0),
        tail
      )

def root_visited_policy(xs: List<&2, RootChild>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{child, rest}:
      root_visited_cons(child, root_visited_policy(rest))

def kid_visited_cons(kid: Kid, tail: F32) -> F32:
  match kid:
    case Kid{action, prior, n, w}:
      F32.add(
        Bool.pick(F32, U32.is_gt(n, 0), prior, 0.0),
        tail
      )

def kid_visited_policy(xs: List<&2, Kid>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{kid, rest}:
      kid_visited_cons(kid, kid_visited_policy(rest))

def edge_score_case(
  visited: Bool,
  +parent_q: F32,
  +c_sqrt_n: F32,
  +visited_policy: F32,
  +fpu: F32,
  +n: U32,
  w: F32,
  prior: F32
) -> F32:
  match visited:
    case True{}:
      F32.add(
        F32.div(F32.sub(0.0, w), U32.to_f32(n)),
        F32.div(
          F32.mul(c_sqrt_n, prior),
          F32.add(1.0, U32.to_f32(n))
        )
      )
    case False{}:
      F32.add(
        F32.sub(
          parent_q,
          F32.mul(fpu, F32.sqrt(visited_policy))
        ),
        F32.mul(c_sqrt_n, prior)
      )

def edge_score(
  +parent_n: U32,
  parent_w: F32,
  visited_policy: F32,
  fpu: F32,
  +n: U32,
  w: F32,
  prior: F32
) -> F32:
  edge_score_case(
    U32.is_gt(n, 0),
    node_q(parent_n, parent_w),
    F32.mul(
      1.5,
      F32.sqrt(F32.max(U32.to_f32(parent_n), 1.0))
    ),
    visited_policy,
    fpu,
    n,
    w,
    prior
  )

def RootChild.action(child: RootChild) -> U32:
  match child:
    case RootChild{action, prior, n, w, expanded, kids}:
      action

def RootChild.score(
  child: RootChild,
  parent_n: U32,
  parent_w: F32,
  visited_policy: F32,
  fpu: F32
) -> F32:
  match child:
    case RootChild{action, prior, n, w, expanded, kids}:
      edge_score(parent_n, parent_w, visited_policy, fpu, n, w, prior)

def Kid.action(kid: Kid) -> U32:
  match kid:
    case Kid{action, prior, n, w}:
      action

def Kid.score(
  kid: Kid,
  parent_n: U32,
  parent_w: F32,
  visited_policy: F32,
  fpu: F32
) -> F32:
  match kid:
    case Kid{action, prior, n, w}:
      edge_score(parent_n, parent_w, visited_policy, fpu, n, w, prior)

def Best.score(best: Best) -> F32:
  match best:
    case Best{action, score}:
      score

def Best.action(best: Best) -> U32:
  match best:
    case Best{action, score}:
      action

def choose_better(
  gt: Bool,
  action: U32,
  score: F32,
  best: Best
) -> Best:
  match gt:
    case True{}:
      Best{action, score}
    case False{}:
      best

def root_best_fold(
  xs: List<&2, RootChild>,
  +parent_n: U32,
  +parent_w: F32,
  +visited_policy: F32,
  +fpu: F32,
  +best: Best
) -> Best:
  match xs:
    case Nil{}:
      best
    case Con{+child, rest}:
      root_best_fold(
        rest,
        parent_n,
        parent_w,
        visited_policy,
        fpu,
        choose_better(
          F32.is_gt(
            RootChild.score(child, parent_n, parent_w, visited_policy, fpu),
            Best.score(best)
          ),
          RootChild.action(child),
          RootChild.score(child, parent_n, parent_w, visited_policy, fpu),
          best
        )
      )

def choose_root_with_vp(
  xs: List<&2, RootChild>,
  +parent_n: U32,
  +parent_w: F32,
  +vp: F32
) -> U32:
  match xs:
    case Nil{}:
      4294967295
    case Con{+first, rest}:
      Best.action(
        root_best_fold(
          rest,
          parent_n,
          parent_w,
          vp,
          0.25,
          Best{
            RootChild.action(first),
            RootChild.score(first, parent_n, parent_w, vp, 0.25)
          }
        )
      )

def choose_root(
  +xs: List<&2, RootChild>,
  parent_n: U32,
  parent_w: F32
) -> U32:
  choose_root_with_vp(
    xs,
    parent_n,
    parent_w,
    root_visited_policy(xs)
  )

def kid_best_fold(
  xs: List<&2, Kid>,
  +parent_n: U32,
  +parent_w: F32,
  +visited_policy: F32,
  +fpu: F32,
  +best: Best
) -> Best:
  match xs:
    case Nil{}:
      best
    case Con{+kid, rest}:
      kid_best_fold(
        rest,
        parent_n,
        parent_w,
        visited_policy,
        fpu,
        choose_better(
          F32.is_gt(
            Kid.score(kid, parent_n, parent_w, visited_policy, fpu),
            Best.score(best)
          ),
          Kid.action(kid),
          Kid.score(kid, parent_n, parent_w, visited_policy, fpu),
          best
        )
      )

def choose_kid_with_vp(
  xs: List<&2, Kid>,
  +parent_n: U32,
  +parent_w: F32,
  +vp: F32
) -> U32:
  match xs:
    case Nil{}:
      4294967295
    case Con{+first, rest}:
      Best.action(
        kid_best_fold(
          rest,
          parent_n,
          parent_w,
          vp,
          0.15,
          Best{
            Kid.action(first),
            Kid.score(first, parent_n, parent_w, vp, 0.15)
          }
        )
      )

def choose_kid(
  +xs: List<&2, Kid>,
  parent_n: U32,
  parent_w: F32
) -> U32:
  choose_kid_with_vp(
    xs,
    parent_n,
    parent_w,
    kid_visited_policy(xs)
  )

def path_word_one(+action: U32) -> U32:
  U32.xor(U32.mul(action, 2654435761), 1)

def path_word_two(+root_action: U32, kid_action: U32) -> U32:
  U32.xor(
    U32.xor(U32.mul(root_action, 2654435761), U32.mul(kid_action, 2246822519)),
    2
  )

def update_kid_case(eq: Bool, kid: Kid, value: F32) -> Kid:
  match eq:
    case True{}:
      match kid:
        case Kid{action, prior, n, w}:
          Kid{action, prior, U32.add(n, 1), F32.add(w, value)}
    case False{}:
      kid

def update_kids(
  xs: List<&2, Kid>,
  +selected: U32,
  +value: F32
) -> List<&2, Kid>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+kid, rest}:
      Con{
        update_kid_case(
          U32.is_eq(Kid.action(kid), selected),
          kid,
          value
        ),
        update_kids(rest, selected, value)
      }

def first_visit_value(
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  kids: List<&2, Kid>,
  +value: F32
) -> ChildStep:
  ChildStep{
    RootChild{
      action,
      prior,
      U32.add(n, 1),
      F32.add(w, value),
      True{},
      kids
    },
    F32.sub(0.0, value),
    path_word_one(action)
  }

def first_visit(
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  kids: List<&2, Kid>
) -> ChildStep:
  first_visit_value(
    action,
    prior,
    n,
    w,
    kids,
    fake_child_value(action)
  )

def grand_visit_value(
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  +kids: List<&2, Kid>,
  +kid_action: U32,
  +value: F32
) -> ChildStep:
  ChildStep{
    RootChild{
      action,
      prior,
      U32.add(n, 1),
      F32.sub(w, value),
      True{},
      update_kids(kids, kid_action, value)
    },
    value,
    path_word_two(action, kid_action)
  }

def grand_visit(
  +action: U32,
  prior: F32,
  +n: U32,
  +w: F32,
  +kids: List<&2, Kid>
) -> ChildStep:
  +kid_action = choose_kid(kids, n, w)
  grand_visit_value(
    action,
    prior,
    n,
    w,
    kids,
    kid_action,
    fake_grand_value(action, kid_action)
  )

def expanded_visit_case(
  has_kids: Bool,
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  kids: List<&2, Kid>
) -> ChildStep:
  match has_kids:
    case True{}:
      grand_visit(action, prior, n, w, kids)
    case False{}:
      first_visit(action, prior, n, w, kids)

def expanded_visit(
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  +kids: List<&2, Kid>
) -> ChildStep:
  match kids:
    case Nil{}:
      first_visit(action, prior, n, w, Nil{})
    case Con{kid, rest}:
      grand_visit(action, prior, n, w, Con{kid, rest})

def simulate_child_case(
  expanded: Bool,
  +action: U32,
  prior: F32,
  n: U32,
  w: F32,
  kids: List<&2, Kid>
) -> ChildStep:
  match expanded:
    case True{}:
      expanded_visit(action, prior, n, w, kids)
    case False{}:
      first_visit(action, prior, n, w, kids)

def simulate_child(child: RootChild) -> ChildStep:
  match child:
    case RootChild{+action, prior, n, w, expanded, kids}:
      simulate_child_case(expanded, action, prior, n, w, kids)

def prepend_selected(
  rest: List<&2, RootChild>,
  step: ChildStep
) -> ChildrenStep:
  match step:
    case ChildStep{child, delta, path_word}:
      ChildrenStep{Con{child, rest}, delta, path_word}

def prepend_unselected(
  child: RootChild,
  step: ChildrenStep
) -> ChildrenStep:
  match step:
    case ChildrenStep{children, delta, path_word}:
      ChildrenStep{Con{child, children}, delta, path_word}

def update_children_decide(
  eq: Bool,
  child: RootChild,
  rest: List<&2, RootChild>,
  rec: ChildrenStep
) -> ChildrenStep:
  match eq:
    case True{}:
      prepend_selected(rest, simulate_child(child))
    case False{}:
      prepend_unselected(child, rec)

def update_children(
  xs: List<&2, RootChild>,
  +selected: U32
) -> ChildrenStep:
  match xs:
    case Nil{}:
      ChildrenStep{Nil{}, 0.0, 0}
    case Con{+child, +rest}:
      update_children_decide(
        U32.is_eq(RootChild.action(child), selected),
        child,
        rest,
        update_children(rest, selected)
      )

def simulate_after_children(
  +root_n: U32,
  root_w: F32,
  step: ChildrenStep
) -> SimStep:
  match step:
    case ChildrenStep{children, delta, path_word}:
      SimStep{
        Tree{U32.add(root_n, 1), F32.add(root_w, delta), children},
        path_word
      }

def simulate_selected(
  +root_n: U32,
  root_w: F32,
  children: List<&2, RootChild>,
  selected: U32
) -> SimStep:
  simulate_after_children(
    root_n,
    root_w,
    update_children(children, selected)
  )

def simulate_tree(tree: Tree) -> SimStep:
  match tree:
    case Tree{+root_n, +root_w, +children}:
      simulate_selected(
        root_n,
        root_w,
        children,
        choose_root(children, root_n, root_w)
      )

def apply_sim_step(
  +path_mix: U32,
  step: SimStep
) -> Run:
  match step:
    case SimStep{tree, path_word}:
      Run{
        tree,
        U32.xor(U32.mul(path_mix, 16777619), path_word)
      }

def apply_sim(run: Run) -> Run:
  match run:
    case Run{tree, path_mix}:
      apply_sim_step(path_mix, simulate_tree(tree))

def run_sims(sims: Nat, run: Run) -> Run:
  match sims run:
    case 0n Run{tree, path_mix}:
      Run{tree, path_mix}
    case 1n+p Run{tree, path_mix}:
      run_sims(p, apply_sim(Run{tree, path_mix}))

def stats_cons_decide(
  gt: Bool,
  +action: U32,
  +n: U32,
  expanded: Bool,
  total: U32,
  xor: U32,
  best_action: U32,
  best_visits: U32,
  expanded_count: U32
) -> VisitStats:
  match gt:
    case True{}:
      VisitStats{
        U32.add(total, n),
        U32.xor(
          xor,
          U32.xor(U32.mul(action, 2654435761), n)
        ),
        action,
        n,
        U32.add(expanded_count, Bool.pick(U32, expanded, 1, 0))
      }
    case False{}:
      VisitStats{
        U32.add(total, n),
        U32.xor(
          xor,
          U32.xor(U32.mul(action, 2654435761), n)
        ),
        best_action,
        best_visits,
        U32.add(expanded_count, Bool.pick(U32, expanded, 1, 0))
      }

def stats_cons(child: RootChild, tail: VisitStats) -> VisitStats:
  match child tail:
    case RootChild{+action, prior, +n, w, expanded, kids}
         VisitStats{total, xor, best_action, best_visits, expanded_count}:
      stats_cons_decide(
        U32.is_gt(n, best_visits),
        action,
        n,
        expanded,
        total,
        xor,
        best_action,
        best_visits,
        expanded_count
      )

def visit_stats(xs: List<&2, RootChild>) -> VisitStats:
  match xs:
    case Nil{}:
      VisitStats{0, 0, 4294967295, 0, 0}
    case Con{child, rest}:
      stats_cons(child, visit_stats(rest))

def print_stats(
  fixture: U32,
  root_moves: U32,
  root_n: U32,
  root_w: F32,
  path_mix: U32,
  stats: VisitStats
) -> IO(Unit):
  match stats:
    case VisitStats{total, xor, best_action, best_visits, expanded}:
      IO.print(
        "fixture=" ++ U32.show(fixture)
        ++ " root_moves=" ++ U32.show(root_moves)
        ++ " root_n=" ++ U32.show(root_n)
        ++ " root_w_q4=" ++ U32.show(
          F32.to_u32(
            F32.mul(F32.add(root_w, 64.0), 4.0)
          )
        )
        ++ " best=" ++ U32.show(best_action)
        ++ " expanded=" ++ U32.show(expanded)
        ++ " visit_xor=" ++ U32.show(xor)
        ++ " path_mix=" ++ U32.show(path_mix)
      )

def print_run(
  fixture: U32,
  root_moves: U32,
  run: Run
) -> IO(Unit):
  match run:
    case Run{tree, path_mix}:
      match tree:
        case Tree{root_n, root_w, children}:
          print_stats(
            fixture,
            root_moves,
            root_n,
            root_w,
            path_mix,
            visit_stats(children)
          )

def run_tree(
  fixture: U32,
  root_moves: U32,
  children: List<&2, RootChild>
) -> IO(Unit):
  print_run(
    fixture,
    root_moves,
    run_sims(24n, Run{Tree{0, 0.0, children}, 0})
  )

def after_root_moves(
  +fixture: U32,
  +root: U32,
  +moves: List<&2, U32>
) -> IO(Unit):
  capped = take_n(moves, 4n)
  do IO<Unit>:
    children : List<&2, RootChild> <- build_children(root, capped, 4n)
    run_tree(fixture, list_len(capped, 0), children)

def after_root(+fixture: U32, +root: U32) -> IO(Unit):
  do IO<Unit>:
    moves : List<&2, U32> <- Chess.legal_moves(root)
    after_root_moves(fixture, root, moves)

def probe_fixture(+fixture: U32) -> IO(Unit):
  do IO<Unit>:
    root : U32 <- Chess.new(fixture)
    after_root(fixture, root)

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- probe_fixture(0)
    b : Unit <- probe_fixture(1)
    c : Unit <- probe_fixture(2)
    d : Unit <- probe_fixture(3)
    e : Unit <- probe_fixture(4)
    return Unit{}
