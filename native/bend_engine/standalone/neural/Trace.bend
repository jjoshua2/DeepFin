# Optional external-verifier evidence. Disabled normally; no reference evaluator.
import Base
import ../../session_probe/Search.bend as Search
import ../../legal_probe/Chess.bend as Chess
import ../main.bend as App
import ../Position.bend as Pos
import ../Policy.bend as Policy

# Read while returning the owner, so the SAME tensor goes to the real forward.
def row_read(text: String, r: Array<F32> & F32) -> Array<F32> & String:
  (a, x) = r
  (a, text ++ " " ++ U32.show(F32.bits(x)))

def row(fuel: Nat, +i: U32, r: Array<F32> & String) -> Array<F32> & String:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, text) = r
      row(p, U32.inc(i), row_read(text, Array.get(F32, a, i)))

def row_print(i: U32, r: Array<F32> & String) -> IO(Array<F32>):
  (a, text) = r
  do IO<Array<F32>>:
    IO.print("info string neural_input " ++ U32.show(i) ++ text)
    return a

def input_rows(fuel: Nat, +i: U32, a: Array<F32>) -> IO(Array<F32>):
  match fuel:
    case 0n: IO.pure(Array<F32>, a)
    case 1n+p:
      do IO<Array<F32>>:
        a : Array<F32> <- row_print(i, row(64n, U32.mul(i, 64), (a, "")))
        input_rows(p, U32.inc(i), a)

def actions(entries: List<&2, Policy.Entry>) -> IO(Unit):
  match entries:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{Policy.Entry{m, full, compact}, rest}:
      do IO<Unit>:
        IO.print("info string neural_action " ++ U32.show(Search.key(m)) ++ " " ++ U32.show(full) ++ " " ++ U32.show(compact))
        actions(rest)

def input(enabled: Bool, seq: U32, ticket: Search.Ticket, game: Pos.Game,
  entries: List<&2, Policy.Entry>, +channels: U32, a: Array<F32>) -> IO(Array<F32>):
  match enabled:
    case False{}: IO.pure(Array<F32>, a)
    case True{}:
      Search.Ticket{epoch, request, node, b, ms, needs} = ticket
      Pos.Game{gb, half, full, hist} = game
      do IO<Array<F32>>:
        IO.print("info string neural_leaf " ++ U32.show(seq) ++ " " ++ U32.show(epoch)
          ++ " " ++ U32.show(request) ++ " " ++ U32.show(node) ++ " " ++ U32.show(channels))
        IO.print("info string neural_board " ++ App.board_text(b) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full))
        actions(entries)
        input_rows(U32.to_nat(channels), 0, a)

def logits_rows(xs: List<&2, F32>, +i: U32) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{x, rest}:
      do IO<Unit>:
        IO.print("info string neural_logit " ++ U32.show(i) ++ " " ++ U32.show(F32.bits(x)))
        logits_rows(rest, U32.inc(i))

def logits(enabled: Bool, xs: List<&2, F32>, +i: U32) -> IO(Unit):
  match enabled:
    case False{}: IO.pure(Unit, Unit{})
    case True{}: logits_rows(xs, i)

def prior_read(i: U32, r: Array<F32> & F32) -> IO(Array<F32>):
  (a, x) = r
  do IO<Array<F32>>:
    IO.print("info string neural_prior " ++ U32.show(i) ++ " " ++ U32.show(F32.bits(x)))
    return a

def priors(fuel: Nat, +i: U32, a: Array<F32>) -> IO(Array<F32>):
  match fuel:
    case 0n: IO.pure(Array<F32>, a)
    case 1n+p:
      do IO<Array<F32>>:
        a : Array<F32> <- prior_read(i, Array.get(F32, a, i))
        priors(p, U32.inc(i), a)

def reply(enabled: Bool, r: Search.Reply) -> IO(Search.Reply):
  match enabled:
    case False{}: IO.pure(Search.Reply, r)
    case True{}:
      Search.Reply{epoch, request, node, +status, +win, +draw, +loss, +count, policy} = r
      do IO<Search.Reply>:
        IO.print("info string neural_prob " ++ U32.show(status) ++ " " ++ U32.show(F32.bits(win))
          ++ " " ++ U32.show(F32.bits(draw)) ++ " " ++ U32.show(F32.bits(loss)) ++ " " ++ U32.show(count))
        policy : Array<F32> <- priors(U32.to_nat(count), 0, policy)
        IO.print("info string neural_end")
        return Search.Reply{epoch, request, node, status, win, draw, loss, count, policy}

def node_text(a: Search.Node) -> String:
  Search.Node{b, m, parent, first, count, n, w, prior, value, status, depth} = a
  App.board_text(b) ++ " " ++ U32.show(Search.key(m)) ++ " " ++ U32.show(parent)
    ++ " " ++ U32.show(first) ++ " " ++ U32.show(count) ++ " " ++ U32.show(n)
    ++ " " ++ U32.show(F32.bits(w)) ++ " " ++ U32.show(F32.bits(prior))
    ++ " " ++ U32.show(F32.bits(value)) ++ " " ++ U32.show(status) ++ " " ++ U32.show(depth)

def node_read(i: U32, r: Array<Search.Node> & Search.Node) -> IO(Array<Search.Node>):
  (a, node) = r
  do IO<Array<Search.Node>>:
    IO.print("info string neural_node " ++ U32.show(i) ++ " " ++ node_text(node))
    return a

def nodes(fuel: Nat, +i: U32, a: Array<Search.Node>) -> IO(Array<Search.Node>):
  match fuel:
    case 0n: IO.pure(Array<Search.Node>, a)
    case 1n+p:
      do IO<Array<Search.Node>>:
        a : Array<Search.Node> <- node_read(i, Array.get(Search.Node, a, i))
        nodes(p, U32.inc(i), a)

def tree(enabled: Bool, t: Search.Tree) -> IO(Search.Tree):
  match enabled:
    case False{}: IO.pure(Search.Tree, t)
    case True{}:
      Search.Tree{pool, +used, cap, +n, +pending, +epoch, +request, +stop, limit} = t
      do IO<Search.Tree>:
        IO.print("info string neural_tree " ++ U32.show(epoch) ++ " " ++ U32.show(n)
          ++ " " ++ U32.show(used) ++ " " ++ U32.show(stop) ++ " " ++ U32.show(pending) ++ " " ++ U32.show(request))
        pool : Array<Search.Node> <- nodes(U32.to_nat(used), 0, pool)
        IO.print("info string neural_tree_end")
        return Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit}
