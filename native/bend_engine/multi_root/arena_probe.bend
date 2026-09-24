# Storage and transaction boundaries using the real Node/Tree representation.
# Synthetic metadata here is not a chess-rules or whole-scheduler proof.
import Base
import ../session_probe/Search.bend as S
import ../legal_probe/Chess.bend as C
import ./ArenaConfig.bend as A
import ./Cohort.bend as Coh
import ./AsyncRun.bend as Run
import ./DeadlineWork.bend as DW
import ../standalone/Position.bend as Pos

def board() -> C.Board:
  C.Board{U64{305419896, 4294967295}, U64.zero(), U64.zero(), U64.zero(),
    U64.zero(), U64.zero(), U64.zero(), U64.zero(), 0, 0, 64}

def node(tag: U32) -> S.Node:
  S.Node{board(), C.Ply{0,0,0,0}, 0, 0, 0, tag, 0.0, 1.0, 0.0, 0, 0}

def seen(label: String, at: U32, r: Array<S.Node> & S.Node) -> IO(Array<S.Node>):
  (pool, a) = r
  S.Node{C.Board{+p,n,b,r,q,k,w,bl,turn,rights,ep},m,parent,first,count,visits,value,prior,cached,status,depth} = a
  do IO<Array<S.Node>>:
    IO.print(label ++ " " ++ U32.show(at) ++ " " ++ U32.show(visits)
      ++ " " ++ U32.show(U64.high(p)) ++ " " ++ U32.show(U64.low(p)))
    return pool

def read_at(label: String, +at: U32, pool: Array<S.Node>) -> IO(Array<S.Node>):
  seen(label, at, Array.get(S.Node, pool, at))

def cross(enabled: Bool, pool: Array<S.Node>) -> IO(Array<S.Node>):
  match enabled:
    case False{}: IO.pure(Array<S.Node>, pool)
    case True{}: read_at("cross", 4096, pool)

def write_cross(enabled: Bool, pool: Array<S.Node>) -> Array<S.Node>:
  match enabled:
    case False{}: pool
    case True{}: Array.set(S.Node, pool, 4096, node(777))

def storage(valid: Bool, +cap: U32, pool: Array<S.Node>) -> IO(Unit):
  match valid:
    case False{}: IO.pure(Unit, Unit{})
    case True{}:
      pool = Array.set(S.Node, pool, 0, node(101))
      pool = write_cross(U32.is_gt(cap,4096), pool)
      pool = Array.set(S.Node, pool, U32.sub(cap,1), node(U32.add(cap,23)))
      do IO<Unit>:
        pool : Array<S.Node> <- read_at("first", 0, pool)
        pool : Array<S.Node> <- cross(U32.is_gt(cap,4096), pool)
        pool : Array<S.Node> <- read_at("last", U32.sub(cap,1), pool)
        return Unit{}

def sized(requested: U32, +cap: U32, used: U32, n: U32, pending: U32, +stop: U32,
  r: Array<S.Node> & U32) -> IO(Unit):
  (pool, size) = r
  do IO<Unit>:
    IO.print("arena " ++ U32.show(requested) ++ " " ++ U32.show(cap) ++ " " ++ U32.show(size)
      ++ " " ++ U32.show(used) ++ " " ++ U32.show(n) ++ " " ++ U32.show(pending) ++ " " ++ U32.show(stop))
    storage(U32.is_zero(stop), cap, pool)

def start_case(requested: U32, tree: S.Tree) -> IO(Unit):
  S.Tree{pool,used,cap,n,pending,epoch,request,stop,depth} = tree
  sized(requested,cap,used,n,pending,stop,Array.size(S.Node,pool))

def starts(caps: List<&2,U32>) -> IO(Unit):
  match caps:
    case Nil{}: IO.pure(Unit,Unit{})
    case Con{+cap,rest}:
      do IO<Unit>:
        done : Unit <- start_case(cap,S.start(board(),cap,10,4))
        starts(rest)

def reported(label: String, +id: U32, tree: S.Tree) -> IO(Unit):
  S.Tree{pool,+used,+cap,n,pending,ep,req,stop,depth} = tree
  do IO<Unit>:
    IO.print(label ++ " " ++ U32.show(id) ++ " " ++ U32.show(used) ++ " " ++ U32.show(cap)
      ++ " " ++ U32.show(n) ++ " " ++ U32.show(pending) ++ " " ++ U32.show(req) ++ " " ++ U32.show(stop))
    pool : Array<S.Node> <- read_at("root",0,pool)
    pool : Array<S.Node> <- read_at("target",U32.min(id,U32.sub(cap,1)),pool)
    return Unit{}

def resume_case(label: String,+id: U32,used: U32,pending: U32,tree: S.Tree) -> IO(Unit):
  S.Tree{pool,old_used,cap,n,old_pending,ep,req,stop,depth} = tree
  reported(label,id,S.resume(S.Tree{pool,used,cap,0,pending,10,20,0,depth},
    S.Ticket{10,20,id,board(),Nil{},True{}},
    S.Reply{10,20,id,3,0.0,1.0,0.0,0,Array.new(F32,8n,0.0)}))

def stopped_root(label: String, root: Coh.Root) -> IO(Unit):
  Coh.Root{game,tree,neural,draws} = root
  reported(label,0,tree)

def main() -> IO(Unit):
  do IO<Unit>:
    configured : U32 <- A.read()
    IO.print("config " ++ U32.show(configured))
    done : Unit <- starts([0,1,20,4095,4096,4097,8191,8192,8193,16384,16385,32768,32769,65536,65537,4294967295])
    done : Unit <- resume_case("idle",4096,1,4096,S.start(board(),4096,10,4))
    done : Unit <- resume_case("unallocated",4096,4096,4096,S.start(board(),8192,10,4))
    done : Unit <- resume_case("outside",8192,8193,8192,S.start(board(),8192,10,4))
    done : Unit <- resume_case("active",4096,4097,4096,S.start(board(),8192,10,4))
    done : Unit <- resume_case("active_last",65535,65536,65535,S.start(board(),65536,10,4))
    done : Unit <- reported("halt",0,S.halt(S.start(board(),8192,10,4),2))
    done : Unit <- stopped_root("cancel",Run.halt_root(Coh.Root{Pos.start(),S.start(board(),65536,10,4),0,0}))
    stopped_root("deadline",DW.halt(Coh.Root{Pos.start(),S.start(board(),65536,10,4),0,0}))
