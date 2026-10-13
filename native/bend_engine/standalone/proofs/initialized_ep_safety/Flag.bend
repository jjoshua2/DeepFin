# Relate the two actual full-Ply observations explicitly.
import Base
import ../ep_occurrence_tag/Spec.bend as EP
import ../castle_chain/Spec.bend as C
import ../../../legal_probe/Chess.bend as Chess

def same(+m: Chess.Ply) -> {EP.flag(m) == C.flag(m) : U32}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}

def required(+m: Chess.Ply,tag: {C.flag(m) == 1 : U32}) -> {EP.flag(m) == 1 : U32}:
  Equal.trans(U32,EP.flag(m),C.flag(m),1,same(m),tag)
