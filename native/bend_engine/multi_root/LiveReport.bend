# Report a settled generation exactly once without copying/dropping its tree.
import Base
import ./Cohort.bend as C
import ./Report.bend as R
import ../standalone/Position.bend as Pos
import ../session_probe/Search.bend as S

type Result is Type:
  Result{root: C.Root, total: R.Totals}

def picked(+cfg: C.Config,+game: Pos.Game,+slot: U32,+gen: U32,+used: U32,+cap: U32,+n: U32,+pending: U32,+req: U32,+stop: U32,+lim: U32,+neural: U32,+draws: U32,
  +sent: U32,+wasted: U32,+cancelled: Bool,+expired: Bool,due: Maybe<&2,Nat>,+began: Nat,p: R.Pick) -> IO(Result):
  C.Config{sims,depth,+budget,diagnostics} = cfg
  R.Pick{pool,+key,visits} = p
  do IO<Result>:
    IO.print("info string live_root {\"schema\":\"deepfin.live-root.v1\",\"slot\":" ++ U32.show(slot) ++ ",\"generation\":" ++ U32.show(gen)
      ++ ",\"completed_simulations\":" ++ U32.show(n) ++ ",\"dispatched_real_rows\":" ++ U32.show(sent) ++ ",\"executed_real_rows\":" ++ U32.show(sent)
      ++ ",\"accepted_neural_rows\":" ++ U32.show(neural) ++ ",\"cancelled_rows\":" ++ U32.show(wasted)
      ++ ",\"cancel_requested\":" ++ R.boolean(cancelled) ++ ",\"deadline_expired\":" ++ R.boolean(expired) ++ ",\"deadline_offset_ms\":" ++ R.deadline_text(began,due)
      ++ ",\"rule_draw_replies\":" ++ U32.show(draws) ++ ",\"used_nodes\":" ++ U32.show(used) ++ ",\"stop_code\":" ++ U32.show(stop)
      ++ ",\"arena_capacity\":" ++ U32.show(cap) ++ ",\"arena_physical_slots\":" ++ U32.show(U32.shln(1,S.storage_depth(cap)))
      ++ ",\"simulation_budget\":" ++ U32.show(sims) ++ ",\"neural_budget\":" ++ U32.show(budget)
      ++ ",\"neural_budget_met\":" ++ Bool.pick(String,U32.is_zero(budget),"null",R.boolean(U32.is_eq(neural,budget)))
      ++ ",\"searched_move\":" ++ R.boolean(Bool.not(U32.is_eq(key,4294967295))) ++ ",\"bestmove\":\"" ++ R.move(key) ++ "\"}")
    return Result{C.Root{game,S.Tree{pool,used,cap,n,pending,slot,req,stop,lim},neural,draws},R.Totals{n,neural,draws,1}}

def checked(+ok: Bool,+cfg: C.Config,root: C.Root,+gen: U32,+sent: U32,+wasted: U32,+cancelled: Bool,+expired: Bool,due: Maybe<&2,Nat>,+began: Nat) -> IO(Result):
  match ok:
    case False{}: IO.die(Result,2,"live generation work does not reconcile")
    case True{}:
      C.Config{sims,depth,budget,diagnostics} = cfg
      C.Root{game,S.Tree{pool,+used,cap,n,pending,+slot,req,stop,lim},neural,draws} = root
      do IO<Result>:
        IO.print("info string live_result_begin " ++ U32.show(slot) ++ " " ++ U32.show(gen))
        p : R.Pick <- R.pick(U32.to_nat(used),0,slot,diagnostics,R.Pick{pool,4294967295,0})
        picked(cfg,game,slot,gen,used,cap,n,pending,req,stop,lim,neural,draws,sent,wasted,cancelled,expired,due,began,p)

def emit(+cfg: C.Config,root: C.Root,+gen: U32,+sent: U32,+wasted: U32,+cancelled: Bool,+expired: Bool,due: Maybe<&2,Nat>,+began: Nat) -> IO(Result):
  C.Root{game,tree,+neural,draws} = root
  checked(U32.is_eq(sent,U32.add(neural,wasted)),cfg,C.Root{game,tree,neural,draws},gen,sent,wasted,cancelled,expired,due,began)
