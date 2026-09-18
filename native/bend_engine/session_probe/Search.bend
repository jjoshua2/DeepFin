# Bounded, resumable PUCT diagnostic. No CBoard search or evaluator inside.
# Node W/value are from that node's side-to-move perspective.
import Base
import ../legal_probe/Chess.bend as Chess

type Node is Data:
  Node{board: Chess.Board, move: Chess.Ply, parent: U32, first: U32, count: U32,
    n: U32, w: F32, prior: F32, value: F32, status: U32, depth: U32}

# Node status: 0=unevaluated, 1=expanded, 2=terminal, 3=depth-cutoff.
# Tree stop: 0=running/budget done, 1=capacity, 2=cancelled,
# 3=backend failure, 4=invalid reply. No partial expansion on errors.
type Tree is Type:
  Tree{nodes: Array<Node>, used: U32, cap: U32, completed: U32, pending: U32,
    epoch: U32, request: U32, stop: U32, depth_limit: U32}

type Ticket is Data:
  Ticket{epoch: U32, request: U32, node: U32, board: Chess.Board,
    moves: List<&2, Chess.Ply>, needs_reply: Bool}

type Prepared is Type:
  Prepared{table: Array<U64>, tree: Tree, ticket: Ticket}

type Reply is Type:
  Reply{epoch: U32, request: U32, node: U32, status: U32,
    win: F32, draw: F32, loss: F32, count: U32, policy: Array<F32>}

def node_board(a: Node) -> Chess.Board:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  board

def node_move(a: Node) -> Chess.Ply:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  move

def node_parent(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  parent

def node_first(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  first

def node_count(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  count

def node_n(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  n

def node_w(a: Node) -> F32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  w

def node_prior(a: Node) -> F32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  prior

def node_value(a: Node) -> F32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  value

def node_status(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  status

def node_depth(a: Node) -> U32:
  Node{board, move, parent, first, count, n, w, prior, value, status, depth} = a
  depth

def blank(b: Chess.Board) -> Node:
  Node{b, Chess.Ply{0, 0, 0, 0}, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0, 0}

def start(b: Chess.Board, cap: U32, epoch: U32, depth: U32) -> Tree:
  Tree{Array.new(Node, 12n, blank(b)), 1, cap, 0, 4096, epoch, 1, 0, depth}

def key(m: Chess.Ply) -> U32:
  Chess.Ply{src, dst, promotion, flag} = m
  U32.or(src, U32.or(U32.shln(dst, 6n), U32.or(U32.shln(promotion, 12n), U32.shln(flag, 15n))))

def mean(+a: Node) -> F32:
  F32.div(node_w(a), F32.from_nat(U32.to_nat(U32.max(1, node_n(a)))))

# Explicit step/result helpers keep affine arrays single-owned across loops.
def mass_add(acc: F32, r: Array<Node> & Node) -> Array<Node> & F32:
  (pool, +a) = r
  add = Bool.pick(F32, U32.is_zero(node_n(a)), 0.0, node_prior(a))
  (pool, F32.add(acc, add))

def mass_step(i: U32, r: Array<Node> & F32) -> Array<Node> & F32:
  (pool, acc) = r
  mass_add(acc, Array.get(Node, pool, i))

def mass(n: Nat, +i: U32, r: Array<Node> & F32) -> Array<Node> & F32:
  match n:
    case 0n: r
    case 1n+p: mass(p, U32.inc(i), mass_step(i, r))

type Scan is Type:
  Scan{pool: Array<Node>, best: U32, score: F32, best_key: U32}

def choose_seen(+i: U32, fpu: F32, scale: F32, best: U32, +score: F32, +best_key: U32,
  r: Array<Node> & Node) -> Scan:
  (pool, +a) = r
  q = Bool.pick(F32, U32.is_zero(node_n(a)), fpu, F32.neg(mean(a)))
  u = F32.div(F32.mul(scale, node_prior(a)), F32.from_nat(U32.to_nat(U32.inc(node_n(a)))))
  +s = F32.add(q, u)
  +k = key(node_move(a))
  +better = Bool.or(F32.is_gt(s, score), Bool.and(F32.is_eq(s, score), U32.is_lt(k, best_key)))
  Scan{pool, Bool.pick(U32, better, i, best), Bool.pick(F32, better, s, score), Bool.pick(U32, better, k, best_key)}

def choose_step(+i: U32, fpu: F32, scale: F32, s: Scan) -> Scan:
  Scan{pool, best, score, best_key} = s
  choose_seen(i, fpu, scale, best, score, best_key, Array.get(Node, pool, i))

def choose(n: Nat, +i: U32, +fpu: F32, +scale: F32, s: Scan) -> Scan:
  match n:
    case 0n: s
    case 1n+p: choose(p, U32.inc(i), fpu, scale, choose_step(i, fpu, scale, s))

def chosen(s: Scan) -> Array<Node> & U32:
  Scan{pool, best, score, best_key} = s
  (pool, best)

def pick_mass(+a: Node, r: Array<Node> & F32) -> Array<Node> & U32:
  (pool, visited_mass) = r
  reduction = Bool.pick(F32, U32.is_zero(node_depth(a)), 0.25, 0.15)
  fpu = F32.sub(mean(a), F32.mul(reduction, F32.sqrt(visited_mass)))
  scale = F32.mul(1.5, F32.sqrt(F32.from_nat(U32.to_nat(U32.max(1, node_n(a))))))
  chosen(choose(U32.to_nat(node_count(a)), node_first(a), fpu, scale,
    Scan{pool, node_first(a), F32.neg(1000.0), 4294967295}))

def pick_child(+a: Node, pool: Array<Node>) -> Array<Node> & U32:
  pick_mass(a, mass(U32.to_nat(node_count(a)), node_first(a), (pool, 0.0)))

def descent(expanded: Bool, id: U32, pool: Array<Node>, a: Node) -> Array<Node> & U32:
  match expanded:
    case True{}: pick_child(a, pool)
    case False{}: (pool, id)

def select_seen(id: U32, r: Array<Node> & Node) -> Array<Node> & U32:
  (pool, +a) = r
  descent(U32.is_eq(node_status(a), 1), id, pool, a)

def select_step(r: Array<Node> & U32) -> Array<Node> & U32:
  (pool, +id) = r
  select_seen(id, Array.get(Node, pool, id))

def select(fuel: Nat, r: Array<Node> & U32) -> Array<Node> & U32:
  match fuel:
    case 0n: r
    case 1n+p: select(p, select_step(r))

# After reaching a leaf, bounded selection only rereads it; no further descent.
def bump(a: Node, value: F32) -> Node:
  Node{b, m, parent, first, count, n, w, prior, cached, status, depth} = a
  Node{b, m, parent, first, count, U32.inc(n), F32.add(w, value), prior, cached, status, depth}

type Backing is Type:
  Backing{pool: Array<Node>, active: Bool, id: U32, value: F32}

def backup_seen(+id: U32, +value: F32, r: Array<Node> & Node) -> Backing:
  (pool, +a) = r
  Backing{Array.set(Node, pool, id, bump(a, value)), Bool.not(U32.is_zero(id)), node_parent(a), F32.neg(value)}

def backup_step(b: Backing) -> Backing:
  Backing{pool, active, +id, value} = b
  match active:
    case False{}: Backing{pool, False{}, id, value}
    case True{}: backup_seen(id, value, Array.get(Node, pool, id))

def backup(fuel: Nat, b: Backing) -> Array<Node>:
  match fuel:
    case 0n:
      Backing{pool, active, id, value} = b
      pool
    case 1n+p: backup(p, backup_step(b))

def complete(id: U32, value: F32, tree: Tree) -> Tree:
  Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
  Tree{backup(33n, Backing{pool, True{}, id, value}), used, cap, U32.inc(n), 4096,
    epoch, request, stop, limit}

def mark(a: Node, first: U32, count: U32, status: U32, value: F32) -> Node:
  Node{b, m, parent, oldfirst, oldcount, n, w, prior, cached, oldstatus, depth} = a
  Node{b, m, parent, first, count, n, w, prior, value, status, depth}

def halt(tree: Tree, code: U32) -> Tree:
  Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
  Tree{pool, used, cap, n, 4096, epoch, request, code, limit}

def terminal_finish(+id: U32, +a: Node, tree: Tree, r: Array<U64> & Bool) -> Prepared:
  Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
  (table, check) = r
  +value = Bool.pick(F32, check, F32.neg(1.0), 0.0)
  pool = Array.set(Node, pool, id, mark(a, 0, 0, 2, value))
  tree = complete(id, value, Tree{pool, used, cap, n, pending, epoch, request, stop, limit})
  Prepared{table, tree, Ticket{0, 0, 0, node_board(a), Nil{}, False{}}}

def ticket_ready(room: Bool, table: Array<U64>, tree: Tree, a: Node,
  +id: U32, ms: List<&2, Chess.Ply>) -> Prepared:
  match room:
    case False{}:
      Prepared{table, halt(tree, 1), Ticket{0, 0, 0, node_board(a), Nil{}, False{}}}
    case True{}:
      Tree{pool, used, cap, n, pending, +epoch, +request, stop, limit} = tree
      Prepared{table, Tree{pool, used, cap, n, id, epoch, request, stop, limit},
        Ticket{epoch, request, id, node_board(a), ms, True{}}}

def prepare_moves(+id: U32, +a: Node, tree: Tree, r: Array<U64> & List<&2, Chess.Ply>) -> Prepared:
  Tree{pool, +used, +cap, n, pending, epoch, request, stop, +limit} = tree
  (table, ms) = r
  match ms:
    case Nil{}:
      terminal_finish(id, a, Tree{pool, used, cap, n, pending, epoch, request, stop, limit},
        Chess.in_check(table, node_board(a), Chess.get_turn(node_board(a))))
    case Con{m, tail}:
      +ms = {Con{m, tail} : List<&2, Chess.Ply>}
      room = Bool.or(U32.is_ge(node_depth(a), limit),
        U32.is_le(U32.add(used, U32.from_nat(List.length(&2, Chess.Ply, ms))), cap))
      ticket_ready(room, table, Tree{pool, used, cap, n, pending, epoch, request, stop, limit}, a, id, ms)

def prepare_kind(cached: Bool, id: U32, +a: Node, table: Array<U64>, tree: Tree) -> Prepared:
  match cached:
    case True{}:
      Prepared{table, complete(id, node_value(a), tree), Ticket{0, 0, 0, node_board(a), Nil{}, False{}}}
    case False{}:
      prepare_moves(id, a, tree, Chess.legal_moves(table, node_board(a)))

def prepare_seen(id: U32, used: U32, cap: U32, n: U32, pending: U32, epoch: U32,
  request: U32, stop: U32, limit: U32, table: Array<U64>, r: Array<Node> & Node) -> Prepared:
  (pool, +a) = r
  prepare_kind(U32.is_ge(node_status(a), 2), id, a, table,
    Tree{pool, used, cap, n, pending, epoch, request, stop, limit})

def prepare_selected(used: U32, cap: U32, n: U32, pending: U32, epoch: U32, request: U32,
  stop: U32, limit: U32, table: Array<U64>, r: Array<Node> & U32) -> Prepared:
  (pool, +id) = r
  prepare_seen(id, used, cap, n, pending, epoch, request, stop, limit, table, Array.get(Node, pool, id))

def prepare(table: Array<U64>, tree: Tree) -> Prepared:
  Tree{pool, used, cap, n, pending, epoch, request, stop, +limit} = tree
  prepare_selected(used, cap, n, pending, epoch, request, stop, limit, table,
    select(U32.to_nat(U32.inc(limit)), (pool, 0)))

# Ordered comparisons reject NaN/infinity before committing an expansion.
def probability(+x: F32) -> Bool:
  Bool.and(F32.is_ge(x, 0.0), F32.is_le(x, 1.0))

def policy_add(acc: F32, valid: Bool, r: Array<F32> & F32) -> Array<F32> & (F32 & Bool):
  (policy, +v) = r
  (policy, (F32.add(acc, v), Bool.and(valid, probability(v))))

def policy_step(i: U32, r: Array<F32> & (F32 & Bool)) -> Array<F32> & (F32 & Bool):
  (policy, (acc, valid)) = r
  policy_add(acc, valid, Array.get(F32, policy, i))

def policy_sum(n: Nat, +i: U32, r: Array<F32> & (F32 & Bool)) -> Array<F32> & (F32 & Bool):
  match n:
    case 0n: r
    case 1n+p: policy_sum(p, U32.inc(i), policy_step(i, r))

type Expanded is Type:
  Expanded{nodes: Array<Node>, policy: Array<F32>}

def append_weight(+m: Chess.Ply, b: Chess.Board, parent: U32, depth: U32, i: U32,
  sum: F32, nodes: Array<Node>, r: Array<F32> & F32) -> Expanded:
  (policy, weight) = r
  child = {Node{Chess.make_move(b, m), m, parent, 0, 0, 0, 0.0, F32.div(weight, sum), 0.0, 0, depth} : Node}
  Expanded{Array.set(Node, nodes, i, child), policy}

def append_one(m: Chess.Ply, b: Chess.Board, parent: U32, depth: U32, i: U32,
  pi: U32, sum: F32, state: Expanded) -> Expanded:
  Expanded{nodes, policy} = state
  append_weight(m, b, parent, depth, i, sum, nodes, Array.get(F32, policy, pi))

def append(ms: List<&2, Chess.Ply>, +b: Chess.Board, +parent: U32, +depth: U32,
  +i: U32, +pi: U32, +sum: F32, state: Expanded) -> Expanded:
  match ms:
    case Nil{}: state
    case Con{m, rest}:
      append(rest, b, parent, depth, U32.inc(i), U32.inc(pi), sum,
        append_one(m, b, parent, depth, i, pi, sum, state))

def expand_done(+id: U32, a: Node, +used: U32, +count: U32, +value: F32,
  cap: U32, n: U32, pending: U32, epoch: U32, request: U32, stop: U32, limit: U32, r: Expanded) -> Tree:
  Expanded{pool, policy} = r
  pool = Array.set(Node, pool, id, mark(a, used, count, 1, value))
  complete(id, value, Tree{pool, U32.add(used, count), cap, n, pending, epoch, U32.inc(request), stop, limit})

def expand_kind(cut: Bool, +id: U32, +a: Node, ms: List<&2, Chess.Ply>, count: U32,
  +value: F32, sum: F32, policy: Array<F32>, tree: Tree) -> Tree:
  match cut:
    case True{}:
      Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
      pool = Array.set(Node, pool, id, mark(a, 0, 0, 3, value))
      complete(id, value, Tree{pool, used, cap, n, pending, epoch, U32.inc(request), stop, limit})
    case False{}:
      Tree{pool, +used, cap, n, pending, epoch, request, stop, limit} = tree
      expand_done(id, a, used, count, value, cap, n, pending, epoch, request, stop, limit,
        append(ms, node_board(a), id, U32.inc(node_depth(a)), used, 0, sum, Expanded{pool, policy}))

def accepted_seen(id: U32, ms: List<&2, Chess.Ply>, count: U32, value: F32, sum: F32,
  policy: Array<F32>, used: U32, cap: U32, n: U32, pending: U32, epoch: U32,
  request: U32, stop: U32, +limit: U32, r: Array<Node> & Node) -> Tree:
  (pool, +a) = r
  expand_kind(U32.is_ge(node_depth(a), limit), id, a, ms, count, value, sum, policy,
    Tree{pool, used, cap, n, pending, epoch, request, stop, limit})

def accepted(valid: Bool, +id: U32, ms: List<&2, Chess.Ply>, count: U32,
  value: F32, sum: F32, policy: Array<F32>, tree: Tree) -> Tree:
  match valid:
    case False{}: halt(tree, 4)
    case True{}:
      Tree{pool, used, cap, n, pending, epoch, request, stop, limit} = tree
      accepted_seen(id, ms, count, value, sum, policy, used, cap, n, pending, epoch,
        request, stop, limit, Array.get(Node, pool, id))

def reply_summed(id: U32, ms: List<&2, Chess.Ply>, count: U32, value: F32,
  tree: Tree, r: Array<F32> & (F32 & Bool)) -> Tree:
  (policy, (+sum, valid)) = r
  accepted(Bool.and(valid, F32.is_gt(sum, 0.0)), id, ms, count, value, sum, policy, tree)

def reply_values(id: U32, ms: List<&2, Chess.Ply>, tree: Tree, reply: Reply) -> Tree:
  Reply{epoch, request, node, status, +win, +draw, +loss, +count, policy} = reply
  +total = F32.add(F32.add(win, draw), loss)
  wdl_ok = Bool.and(probability(win), Bool.and(probability(draw), Bool.and(probability(loss),
    F32.is_le(F32.abs(F32.sub(total, 1.0)), 0.00001))))
  reply_summed(id, ms, count, F32.div(F32.sub(win, loss), F32.max(total, 0.00001)), tree,
    policy_sum(U32.to_nat(count), 0, (policy, (0.0, wdl_ok))))

def reply_status(status: U32, id: U32, ms: List<&2, Chess.Ply>, tree: Tree, reply: Reply) -> Tree:
  match status:
    case 0: reply_values(id, ms, tree, reply)
    case 1: halt(tree, 3)
    case 2: halt(tree, 2)
    case _: halt(tree, 4)

def reply_checked(valid: Bool, id: U32, ms: List<&2, Chess.Ply>, tree: Tree, reply: Reply) -> Tree:
  match valid:
    case False{}: halt(tree, 4)
    case True{}:
      Reply{epoch, request, node, +status, win, draw, loss, count, policy} = reply
      reply_status(status, id, ms, tree, Reply{epoch, request, node, status, win, draw, loss, count, policy})

def resume(tree: Tree, ticket: Ticket, reply: Reply) -> Tree:
  Tree{pool, used, cap, n, +pending, +epoch, +request, +stop, limit} = tree
  Ticket{te, tr, +id, board, +ms, needs_reply} = ticket
  Reply{+re, +rr, +rn, +status, win, draw, loss, +count, policy} = reply
  identity = Bool.and(needs_reply, Bool.and(U32.is_eq(te, epoch), Bool.and(U32.is_eq(tr, request),
    Bool.and(U32.is_zero(stop), Bool.and(U32.is_eq(pending, id),
      Bool.and(U32.is_eq(epoch, re), Bool.and(U32.is_eq(request, rr), U32.is_eq(id, rn))))))))
  shape = Bool.or(Bool.not(U32.is_zero(status)), U32.is_eq(count, U32.from_nat(List.length(&2, Chess.Ply, ms))))
  reply_checked(Bool.and(identity, Bool.and(shape, U32.is_le(count, 256))), id, ms,
    Tree{pool, used, cap, n, pending, epoch, request, stop, limit},
    Reply{re, rr, rn, status, win, draw, loss, count, policy})
