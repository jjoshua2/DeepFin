# Independent target-centred geometry equals the complete forward source scan.
import Base
import ./Spec.bend as S
import ./Masks.bend as M
import ./Packing.bend as Pack
import ../initialized_attacks/Spec.bend as I
import ../attack_witness/Spec.bend as W
import ../../../legal_probe/Chess.bend as Chess

def geometry(+b: Chess.Board,+target: Nat,+by: Bool,+bound: {Nat.is_lt(target,64n) == True{} : Bool}) ->
  {I.attacked(b,target,by) == S.attacked(b,target,by) : Bool}:
  Equal.trans(Bool,I.attacked(b,target,by),W.present(b,Bool.to_u32(by),Pack.masks(target,by,Chess.occupied(b))),S.attacked(b,target,by),
    Equal.cong(W.Masks,Bool,m => W.present(b,Bool.to_u32(by),m),I.masks(target,by,Chess.occupied(b)),Pack.masks(target,by,Chess.occupied(b)),M.masks(target,by,Chess.occupied(b),bound)),
    Pack.present(b,target,by))
