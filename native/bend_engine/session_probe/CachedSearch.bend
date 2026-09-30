# Connection-owned legal moves only. Search state, replies and history stay separate.
import Base
import ./Search.bend as S
import ../legal_probe/Chess.bend as C
import ../u64_map_probe/MoveCache.bend as M
import ../standalone/Text.bend as Text

type State is Type:
  Disabled{}
  Enabled{bits: U32, cache: M.Cache, hits: U32, fills: U32, bypasses: U32}

type Ready is Type:
  Ready{state: State, prepared: S.Prepared}

def created(bits: U32, result: Maybe<&1, M.Cache>) -> IO(State):
  match result:
    case None{}: IO.die(State, 2, "invalid DEEPFIN_SESSION_MOVE_CACHE_BITS")
    case Some{cache}: IO.pure(State, Enabled{bits, cache, 0, 0, 0})

def new(+bits: U32) -> IO(State):
  match bits:
    case 0: IO.pure(State, Disabled{})
    case _: created(bits, M.new(bits))

# One or two ASCII decimal digits; reject whitespace, signs, overflow and junk.
def setting(text: String) -> Maybe<&2, U32>:
  match text:
    case SCon{a, SNil{}}: Text.number(SCon{a, SNil{}})
    case SCon{a, SCon{b, SNil{}}}: Text.number(SCon{a, SCon{b, SNil{}}})
    case _: None{}

def configured(result: Maybe<&2, U32>) -> IO(State):
  match result:
    case None{}: IO.die(State, 2, "invalid DEEPFIN_SESSION_MOVE_CACHE_BITS")
    case Some{bits}: new(bits)

def environment(result: Result<&1, &1, U32 & String, String>) -> IO(State):
  match result:
    case Fail{err}: IO.pure(State, Disabled{})
    case Done{text}: configured(setting(text))

def load() -> IO(State):
  do IO<State>:
    result : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_SESSION_MOVE_CACHE_BITS")
    environment(result)

def clear(state: State) -> IO(State):
  match state:
    case Disabled{}: IO.pure(State, Disabled{})
    case Enabled{bits, cache, hits, fills, bypasses}: new(bits)

def report(state: State) -> IO(State):
  match state:
    case Disabled{}:
      do IO<State>:
        IO.print("cache 0 0 0 0")
        return Disabled{}
    case Enabled{+bits, cache, +hits, +fills, +bypasses}:
      do IO<State>:
        IO.print("cache " ++ U32.show(bits) ++ " " ++ U32.show(hits) ++ " "
          ++ U32.show(fills) ++ " " ++ U32.show(bypasses))
        return Enabled{bits, cache, hits, fills, bypasses}

def counted(bits: U32, cache: M.Cache, hits: U32, fills: U32, bypasses: U32, access: M.Access) -> State:
  match access:
    case M.Hit{}: Enabled{bits, cache, U32.inc(hits), fills, bypasses}
    case M.Filled{}: Enabled{bits, cache, hits, U32.inc(fills), bypasses}
    case M.Bypassed{}: Enabled{bits, cache, hits, fills, U32.inc(bypasses)}

def generated(bits: U32, hits: U32, fills: U32, bypasses: U32, id: U32, node: S.Node,
  tree: S.Tree, result: M.Result) -> Ready:
  M.Result{table, cache, moves, access} = result
  Ready{counted(bits, cache, hits, fills, bypasses, access), S.prepare_moves(id, node, tree, (table, moves))}

def kind(cached_value: Bool, bits: U32, cache: M.Cache, hits: U32, fills: U32, bypasses: U32,
  id: U32, +node: S.Node, table: Array<U64>, tree: S.Tree) -> Ready:
  match cached_value:
    case True{}:
      Ready{Enabled{bits, cache, hits, fills, bypasses}, S.prepare_kind(True{}, id, node, table, tree)}
    case False{}:
      generated(bits, hits, fills, bypasses, id, node, tree, M.legal_moves(table, cache, S.node_board(node)))

def seen(bits: U32, cache: M.Cache, hits: U32, fills: U32, bypasses: U32, id: U32,
  used: U32, cap: U32, n: U32, pending: U32, epoch: U32, request: U32, stop: U32,
  limit: U32, table: Array<U64>, result: Array<S.Node> & S.Node) -> Ready:
  (pool, +node) = result
  kind(U32.is_ge(S.node_status(node), 2), bits, cache, hits, fills, bypasses, id, node, table,
    S.Tree{pool, used, cap, n, pending, epoch, request, stop, limit})

def selected(bits: U32, cache: M.Cache, hits: U32, fills: U32, bypasses: U32,
  used: U32, cap: U32, n: U32, pending: U32, epoch: U32, request: U32, stop: U32,
  limit: U32, table: Array<U64>, result: Array<S.Node> & U32) -> Ready:
  (pool, +id) = result
  seen(bits, cache, hits, fills, bypasses, id, used, cap, n, pending, epoch, request,
    stop, limit, table, Array.get(S.Node, pool, id))

# Use the existing selector and ticket/terminal/capacity machinery. Only the
# unevaluated leaf's legal-move source differs; completed nodes still use S's value.
def prepare(state: State, table: Array<U64>, tree: S.Tree) -> Ready:
  match state:
    case Disabled{}: Ready{Disabled{}, S.prepare(table, tree)}
    case Enabled{bits, cache, hits, fills, bypasses}:
      S.Tree{pool, used, cap, n, pending, epoch, request, stop, +limit} = tree
      selected(bits, cache, hits, fills, bypasses, used, cap, n, pending, epoch,
        request, stop, limit, table, S.select(U32.to_nat(U32.inc(limit)), (pool, 0)))
