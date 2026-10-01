# Reporting only. Neural rows and physical forward calls have different units.
import Base
import ./Cohort.bend as C
import ../session_probe/Search.bend as S
import ../legal_probe/Chess.bend as Chess
import ../standalone/Position.bend as Pos
import ../standalone/Text.bend as Text
import ../standalone/NeuralWork.bend as W

type Totals is Data:
  Totals{completed: U32, neural: U32, draws: U32, roots: U32}

def diagnostic(enabled: Bool, text: String) -> IO(Unit):
  match enabled:
    case False{}: IO.pure(Unit, Unit{})
    case True{}: IO.print(text)

def word(+v: U64) -> String:
  U32.show(U64.high(v)) ++ "," ++ U32.show(U64.low(v))

def board(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  word(p) ++ "," ++ word(n) ++ "," ++ word(bi) ++ "," ++ word(r) ++ "," ++ word(q) ++ "," ++ word(k)
    ++ "," ++ word(w) ++ "," ++ word(bl) ++ "," ++ U32.show(turn) ++ "," ++ U32.show(rights) ++ "," ++ U32.show(ep)

def node(ep: U32, at: U32, a: S.Node) -> String:
  S.Node{b,m,parent,first,count,n,w,prior,value,status,depth} = a
  "info string cohort_node " ++ U32.show(ep) ++ " " ++ U32.show(at) ++ " [" ++ board(b)
    ++ "," ++ U32.show(S.key(m)) ++ "," ++ U32.show(parent) ++ "," ++ U32.show(first) ++ "," ++ U32.show(count)
    ++ "," ++ U32.show(n) ++ "," ++ U32.show(F32.bits(w)) ++ "," ++ U32.show(F32.bits(prior))
    ++ "," ++ U32.show(F32.bits(value)) ++ "," ++ U32.show(status) ++ "," ++ U32.show(depth) ++ "]"

type Pick is Type:
  Pick{pool: Array<S.Node>, best: U32, visits: U32}

def picked(ep: U32, +at: U32, diagnostics: Bool, +best: U32, +visits: U32, r: Array<S.Node> & S.Node) -> IO(Pick):
  (pool, +a) = r
  +n = S.node_n(a)
  +key = S.key(S.node_move(a))
  valid = Bool.and(U32.is_gt(at, 0), U32.is_zero(S.node_parent(a)))
  +better = Bool.and(valid, Bool.or(U32.is_gt(n, visits), Bool.and(U32.is_eq(n, visits), U32.is_lt(key, best))))
  do IO<Pick>:
    diagnostic(diagnostics, node(ep, at, a))
    return Pick{pool, Bool.pick(U32, better, key, best), Bool.pick(U32, better, n, visits)}

def pick(fuel: Nat, +at: U32, +ep: U32, +diagnostics: Bool, p: Pick) -> IO(Pick):
  match fuel:
    case 0n: IO.pure(Pick, p)
    case 1n+left:
      Pick{pool,best,visits} = p
      do IO<Pick>:
        p : Pick <- picked(ep, at, diagnostics, best, visits, Array.get(S.Node, pool, at))
        pick(left, U32.inc(at), ep, diagnostics, p)

def move(+key: U32) -> String:
  match key:
    case 4294967295: "0000"
    case _: Pos.move_text(Chess.Ply{U32.and(key,63),U32.and(U32.shrn(key,6n),63),U32.and(U32.shrn(key,12n),7),0})

def boolean(v: Bool) -> String:
  match v:
    case False{}: "false"
    case True{}: "true"

def root_picked(cfg: C.Config, +ep: U32, +used: U32, +n: U32, +stop: U32, +neural: U32, +draws: U32, p: Pick) -> IO(Totals):
  C.Config{sims,depth,+budget,diagnostics} = cfg
  Pick{pool,+key,visits} = p
  do IO<Totals>:
    IO.print("info string cohort_root {\"root\":" ++ U32.show(ep) ++ ",\"completed_simulations\":" ++ U32.show(n)
      ++ ",\"executed_real_rows\":" ++ U32.show(neural) ++ ",\"accepted_neural_rows\":" ++ U32.show(neural)
      ++ ",\"rule_draw_replies\":" ++ U32.show(draws) ++ ",\"used_nodes\":" ++ U32.show(used) ++ ",\"stop_code\":" ++ U32.show(stop)
      ++ ",\"simulation_budget\":" ++ U32.show(sims) ++ ",\"neural_budget\":" ++ U32.show(budget)
      ++ ",\"neural_budget_met\":" ++ Bool.pick(String, U32.is_zero(budget), "null", boolean(U32.is_eq(neural,budget)))
      ++ ",\"searched_move\":" ++ boolean(Bool.not(U32.is_eq(key,4294967295))) ++ ",\"bestmove\":\"" ++ move(key) ++ "\"}")
    return Totals{n,neural,draws,1}

def root(+cfg: C.Config, r: C.Root) -> IO(Totals):
  C.Config{sims,depth,budget,diagnostics} = cfg
  C.Root{game,S.Tree{pool,+used,cap,+n,pending,+ep,request,+stop,lim},+neural,+draws} = r
  do IO<Totals>:
    p : Pick <- pick(U32.to_nat(used), 0, ep, diagnostics, Pick{pool,4294967295,0})
    root_picked(cfg,ep,used,n,stop,neural,draws,p)

def add(a: Totals,b: Totals) -> Totals:
  Totals{n,e,d,r} = a
  Totals{nn,ee,dd,rr} = b
  Totals{U32.add(n,nn),U32.add(e,ee),U32.add(d,dd),U32.add(r,rr)}

def roots(+cfg: C.Config, rs: List<C.Root>, total: Totals) -> IO(Totals):
  match rs:
    case Nil{}: IO.pure(Totals,total)
    case Con{r,rest}:
      do IO<Totals>:
        t : Totals <- root(cfg,r)
        roots(cfg,rest,add(total,t))

def deadline_text(began: Nat,due: Maybe<&2,Nat>) -> String:
  match due:
    case None{}: "null"
    case Some{stamp}: Nat.show(Nat.sub(stamp,began))

# Async final rows separate delivered/accepted work from cancelled physical work.
def async_picked(ok: Bool, cfg: C.Config,+ep: U32,+used: U32,+n: U32,+stop: U32,+neural: U32,+draws: U32,
  +sent: U32,+wasted: U32,cancelled: Bool,due: Maybe<&2,Nat>,expired: Bool,began: Nat,p: Pick) -> IO(Totals):
  match ok:
    case False{}: IO.die(Totals,2,"async root admission does not reconcile")
    case True{}:
      C.Config{sims,depth,+budget,diagnostics} = cfg
      Pick{pool,+key,visits} = p
      do IO<Totals>:
        IO.print("info string cohort_root {\"root\":" ++ U32.show(ep) ++ ",\"completed_simulations\":" ++ U32.show(n)
          ++ ",\"dispatched_real_rows\":" ++ U32.show(sent) ++ ",\"executed_real_rows\":" ++ U32.show(sent)
          ++ ",\"accepted_neural_rows\":" ++ U32.show(neural) ++ ",\"cancelled_rows\":" ++ U32.show(wasted)
          ++ ",\"deadline_offset_ms\":" ++ deadline_text(began,due) ++ ",\"deadline_expired\":" ++ boolean(expired)
          ++ ",\"cancel_requested\":" ++ boolean(cancelled) ++ ",\"rule_draw_replies\":" ++ U32.show(draws)
          ++ ",\"used_nodes\":" ++ U32.show(used) ++ ",\"stop_code\":" ++ U32.show(stop)
          ++ ",\"simulation_budget\":" ++ U32.show(sims) ++ ",\"neural_budget\":" ++ U32.show(budget)
          ++ ",\"neural_budget_met\":" ++ Bool.pick(String,U32.is_zero(budget),"null",boolean(U32.is_eq(neural,budget)))
          ++ ",\"searched_move\":" ++ boolean(Bool.not(U32.is_eq(key,4294967295))) ++ ",\"bestmove\":\"" ++ move(key) ++ "\"}")
        return Totals{n,neural,draws,1}

def async_root(+cfg: C.Config,r: C.Root,+sent: U32,+wasted: U32,cancelled: Bool,due: Maybe<&2,Nat>,expired: Bool,began: Nat) -> IO(Totals):
  C.Config{sims,depth,budget,diagnostics} = cfg
  C.Root{game,S.Tree{pool,+used,cap,+n,pending,+ep,request,+stop,lim},+neural,+draws} = r
  do IO<Totals>:
    p : Pick <- pick(U32.to_nat(used),0,ep,diagnostics,Pick{pool,4294967295,0})
    async_picked(U32.is_eq(sent,U32.add(neural,wasted)),cfg,ep,used,n,stop,neural,draws,sent,wasted,cancelled,due,expired,began,p)
