import Base
import ./LAWS.bend as Laws
import ./Runtime.bend as Run
import ./Transport.bend as T
import ../initialized_attacks/PROOF.bend as InitProof
import ../initialized_attacks/LAWS.bend as InitLaws
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def Laws.generation_then_check_matches_direct_check(a,b,q,side):
  T.actual(a,b,q,side)

def Laws.initialized_generation_then_check_matches_geometry(d,seed,n,extra,b,q,sq,side,depth,tables,full,bound,one):
  Equal.trans(Array<U64> & (List<&2,Chess.Ply> & Bool),
    Run.run(Init.run(d,seed,n,extra),b,q,Bool.to_u32(side)),
    Run.attach(Run.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),Chess.in_check(Init.run(d,seed,n,extra),q,Bool.to_u32(side))),
    (Init.run(d,seed,n,extra),(Run.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),Geometry.attacked(q,sq,Bool.not(side)))),
    T.actual(Init.run(d,seed,n,extra),b,q,Bool.to_u32(side)),
    Equal.cong(Array<U64> & Bool,Array<U64> & (List<&2,Chess.Ply> & Bool),
      r => Run.attach(Run.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),r),
      Chess.in_check(Init.run(d,seed,n,extra),q,Bool.to_u32(side)),
      (Init.run(d,seed,n,extra),Geometry.attacked(q,sq,Bool.not(side))),
      InitLaws.initialized_single_king_check_matches_geometry(d,seed,n,extra,q,sq,side,depth,tables,full,bound,one)))
