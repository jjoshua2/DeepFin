import Base
import ./Spec.bend as S
import ./Lists.bend as L
import ./Certificate.bend as C
import ../attack_geometry/Geometry.bend as G

def forward(+kind: G.Leaper,+src: Nat,+dst: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},here: S.member(dst,G.targets(kind,src))) -> S.member(src,G.targets(G.reverse(kind),dst)):
  L.at(t => S.neighbor(kind,src,t),G.targets(kind,src),dst,C.all(kind,src,source_bound),here)(target_bound)

def backward(+kind: G.Leaper,+src: Nat,+dst: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},here: S.member(src,G.targets(G.reverse(kind),dst))) -> S.member(dst,G.targets(kind,src)):
  match kind:
    case G.Knight{}: forward(G.Knight{},dst,src,target_bound,source_bound,here)
    case G.King{}: forward(G.King{},dst,src,target_bound,source_bound,here)
    case G.WhitePawn{}: forward(G.BlackPawn{},dst,src,target_bound,source_bound,here)
    case G.BlackPawn{}: forward(G.WhitePawn{},dst,src,target_bound,source_bound,here)

def bools(+kind: G.Leaper,+src: Nat,+dst: Nat,+source_bound: {Nat.is_lt(src,64n) == True{} : Bool},+target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}) -> {S.contains(dst,G.targets(kind,src)) == S.contains(src,G.targets(G.reverse(kind),dst)) : Bool}:
  L.equivalent(S.contains(dst,G.targets(kind,src)),S.contains(src,G.targets(G.reverse(kind),dst)),
    e => L.to_true(src,G.targets(G.reverse(kind),dst),forward(kind,src,dst,source_bound,target_bound,L.from_true(dst,G.targets(kind,src),e))),
    e => L.to_true(dst,G.targets(kind,src),backward(kind,src,dst,source_bound,target_bound,L.from_true(src,G.targets(G.reverse(kind),dst),e))))
