import Base
import ../pawn_ep_occurrence/Spec.bend as PriorSpec
import ../candidate_bypass_classification/Fields.bend as Fields
import ../castle_chain/Spec.bend as Chain
import ../../../legal_probe/Chess.bend as C

def backrank(d: U32) -> Bool:
  +rank=U32.div(d,8)
  Bool.or(U32.is_zero(rank),U32.is_eq(rank,7))

def dst(m: C.Ply) -> U32:
  PriorSpec.dst(m)

def promoted(-m: C.Ply) -> Type:
  {backrank(dst(m)) == True{} : Bool} & Fields.promoted(m)

def scanned(-m: C.Ply) -> Type:
  Fields.plain(m) | (Fields.ep(m) | promoted(m))

def generated(-b: C.Board,-m: C.Ply) -> Type:
  scanned(m) | Chain.castle(b,m)

def rank_gate(promote: Bool,-d: U32) -> Data:
  match promote:
    case False{}: Unit
    case True{}: {backrank(d) == True{} : Bool}

def correlate_rank(pawn: Bool,rank: Bool,-d: U32,same: {backrank(d) == rank : Bool}) -> rank_gate(Bool.and(pawn,rank),d):
  match pawn:
    case False{}: Unit{}
    case True{}:
      match rank:
        case False{}: Unit{}
        case True{}: same

def correlate(pawn: Bool,+d: U32) -> rank_gate(Bool.and(pawn,backrank(d)),d):
  correlate_rank(pawn,backrank(d),d,{==})

def ordinary(-m: C.Ply) -> Type:
  Fields.plain(m) | Fields.ep(m)
