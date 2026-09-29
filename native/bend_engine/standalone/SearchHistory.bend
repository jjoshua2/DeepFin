# Reconstruct a requested leaf's history inside Bend, keeping arrays single-owned.
import Base
import ../legal_probe/Chess.bend as Chess
import ../session_probe/Search.bend as Search
import ./Position.bend as Pos
import ./Rules.bend as Rules

type Trace is Type:
  Trace{pool: Array<Search.Node>, path: List<&2, Chess.Ply>, id: U32, valid: Bool}

def read(+id: U32, path: List<&2, Chess.Ply>, r: Array<Search.Node> & Search.Node) -> Trace:
  (pool, +node) = r
  +parent = Search.node_parent(node)
  Trace{pool, Con{Search.node_move(node), path}, parent,
    Bool.and(U32.is_lt(parent, id), Bool.not(Search.is_claim(node)))}

def step(stopped: Bool, t: Trace) -> Trace:
  match stopped:
    case True{}: t
    case False{}:
      Trace{pool, path, +id, valid} = t
      read(id, path, Array.get(Search.Node, pool, id))

def dispatch(used: U32, t: Trace) -> Trace:
  Trace{pool, path, +id, valid} = t
  +ok = Bool.and(valid, U32.is_lt(id, used))
  step(Bool.or(U32.is_zero(id), Bool.not(ok)), Trace{pool, path, id, ok})

def trace(fuel: Nat, +used: U32, t: Trace) -> Trace:
  match fuel:
    case 0n: t
    case 1n+p: trace(p, used, dispatch(used, t))

def replay(path: List<&2, Chess.Ply>, g: Pos.Game) -> Pos.Game:
  match path:
    case Nil{}: g
    case Con{m, rest}: replay(rest, Pos.apply(g, m))

type Rebuilt is Type:
  Rebuilt{tree: Search.Tree, game: Pos.Game, valid: Bool}

def finish(+game: Pos.Game, expected: Chess.Board, valid: Bool, tree: Search.Tree) -> Rebuilt:
  Rebuilt{tree, game, Bool.and(valid, Rules.equal(Pos.board(game), expected))}

def traced(g: Pos.Game, expected: Chess.Board, used: U32, cap: U32, n: U32,
  pending: U32, epoch: U32, request: U32, stop: U32, limit: U32, t: Trace) -> Rebuilt:
  Trace{pool, path, id, valid} = t
  finish(replay(path, g), expected, Bool.and(valid, U32.is_zero(id)),
    Search.Tree{pool, used, cap, n, pending, epoch, request, stop, limit})

def rebuild(g: Pos.Game, id: U32, expected: Chess.Board, tree: Search.Tree) -> Rebuilt:
  Search.Tree{pool, +used, cap, n, pending, epoch, request, stop, limit} = tree
  traced(g, expected, used, cap, n, pending, epoch, request, stop, limit,
    trace(32n, used, Trace{pool, Nil{}, id, True{}}))
