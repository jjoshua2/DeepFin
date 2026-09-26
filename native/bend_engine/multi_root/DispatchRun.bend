# Runtime application of an immutable gather table; the owning FIFO is preserved.
import Base
import ./Dispatch.bend as D
import ./AsyncRun.bend as A
import ./Cohort.bend as C
import ./DeadlineWork.bend as DW
import ./Control.bend as Ctrl
import ./Report.bend as R
import ../batch_backend/Batch.bend as B
import ../collections_probe/Queue.bend as Q

def population(fuel: Nat,+mask: U32) -> U32:
  match fuel:
    case 0n: 0
    case 1n+p: U32.add(U32.and(mask,1),population(p,U32.shr(mask)))

def selected(ok: Bool,+cfg: C.Config,+cap: U32,+n: U32,+batch: U32,roots: Q.Queue<&1,C.Root>,store: C.Store,flags: Ctrl.Flags,counts: A.Counts,began: Nat,rounds: U32) -> IO(A.State):
  match ok:
    case False{}: IO.die(A.State,2,"dispatch gather limit outside physical capacity")
    case True{}:
      C.Config{sims,depth,budget,diagnostics} = cfg
      do IO<A.State>:
        R.diagnostic(Bool.and(diagnostics,U32.is_gt(n,0)),"info string live_dispatch_step " ++ U32.show(n) ++ " " ++ U32.show(cap) ++ " " ++ U32.show(batch))
        A.round_limited(cap,cfg,roots,store,flags,counts,began,rounds)

def counted(+policy: D.Policy,+cfg: C.Config,store: C.Store,+flags: Ctrl.Flags,counts: A.Counts,began: Nat,rounds: U32,r: Q.Queue<&1,C.Root> & U32) -> IO(A.State):
  C.Store{table,maps,scratch,logits,input,output,+caps} = store
  B.Capabilities{profile,+batch,channels,ilog,olog} = caps
  (roots,active) = r
  +n = population(17n,U32.and(active,U32.not(Ctrl.stopped(flags))))
  +chosen = D.limit(policy,n,batch)
  selected(Bool.and(U32.is_le(n,16),Bool.and(U32.is_gt(chosen,0),U32.is_le(chosen,batch))),cfg,chosen,n,batch,roots,
    C.Store{table,maps,scratch,logits,input,output,caps},flags,counts,began,rounds)

def planned(policy: D.Policy,+cfg: C.Config,s: A.State) -> IO(A.State):
  A.State{mode,store,flags,counts,began,+rounds} = s
  match mode rounds:
    case A.Ready{roots} 4098: IO.die(A.State,2,"async cohort progress bound")
    case A.Ready{roots} _: counted(policy,cfg,store,flags,counts,began,rounds,DW.scan_queue(cfg,roots,0))
    case _ _: A.progress(cfg,A.State{mode,store,flags,counts,began,rounds})

def progress(p: D.Policy,cfg: C.Config,s: A.State) -> IO(A.State):
  match p:
    # Default path avoids even the extra ready-root scan.
    case D.Greedy{}: A.progress(cfg,s)
    case _: planned(p,cfg,s)
