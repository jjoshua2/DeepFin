# Boundary probes for actual Search.select / backup, not chess legality tests.
import Base
import ../session_probe/Search.bend as S
import ../legal_probe/Chess.bend as C

def board() -> C.Board:
  C.Board{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), 0, 0, 64}

def fill(n: Nat, +i: U32, +depth: U32, +leaf: U32, pool: Array<S.Node>) -> Array<S.Node>:
  match n:
    case 0n: pool
    case 1n+p:
      +expanded = U32.is_lt(i, depth)
      node = {S.Node{board(), C.Ply{0, 0, 0, 0}, U32.sub(i, Bool.to_u32(U32.is_gt(i, 0))), U32.inc(i), Bool.to_u32(expanded),
        2, 0.25, 1.0, 0.0, Bool.pick(U32, expanded, 1, leaf), i} : S.Node}
      fill(p, U32.inc(i), depth, leaf, Array.set(S.Node, pool, i, node))

def pool(depth: U32, leaf: U32) -> Array<S.Node>:
  fill(8n, 0, depth, leaf, Array.new(S.Node, 3n, S.blank(board())))

def selected(depth: U32, leaf: U32, fuel: U32, r: Array<S.Node> & U32) -> IO(Unit):
  (a, id) = r
  IO.print("select " ++ U32.show(depth) ++ " " ++ U32.show(leaf) ++ " " ++ U32.show(fuel) ++ " " ++ U32.show(id))

def selection(+depth: U32, +leaf: U32, +fuel: U32) -> IO(Unit):
  selected(depth, leaf, fuel, S.select(U32.to_nat(fuel), (pool(depth, leaf), 0)))

def float_bits(x: F32) -> U32:
  F32{bits} = x
  U32{bits}

def field(prefix: String, r: Array<S.Node> & S.Node) -> IO(Array<S.Node>):
  (a, +node) = r
  do IO<Array<S.Node>>:
    IO.print(prefix ++ " " ++ U32.show(S.node_n(node)) ++ " " ++ U32.show(float_bits(S.node_w(node))))
    return a

def dump(n: Nat, +i: U32, +prefix: String, a: Array<S.Node>) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Array<S.Node> <- field(prefix ++ " " ++ U32.show(i), Array.get(S.Node, a, i))
        dump(p, U32.inc(i), prefix, a)

def backing(+depth: U32, +fuel: U32, +active: Bool) -> IO(Unit):
  id = Bool.pick(U32, active, depth, 4294967295)
  prefix = "backup " ++ U32.show(depth) ++ " " ++ U32.show(fuel) ++ " " ++ U32.show(Bool.to_u32(active))
  dump(8n, 0, prefix, S.backup(U32.to_nat(fuel), S.Backing{pool(depth, 0), active, id, 0.5}))

def cases(n: Nat, +depth: U32) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Unit <- selection(depth, 0, 0)
        b : Unit <- selection(depth, 0, 1)
        c : Unit <- selection(depth, 0, 3)
        d : Unit <- selection(depth, 0, 33)
        e : Unit <- selection(depth, 2, 33)
        f : Unit <- selection(depth, 3, 33)
        g : Unit <- backing(depth, 0, True{})
        h : Unit <- backing(depth, 1, True{})
        i : Unit <- backing(depth, 3, True{})
        j : Unit <- backing(depth, 33, True{})
        k : Unit <- backing(depth, 33, False{})
        cases(p, U32.inc(depth))

def main() -> IO(Unit):
  cases(8n, 0)
