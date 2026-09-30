# Query-state composition from explicit internal read certificates.
# Public Composition supplies these from checked initialization producers.
import Base
import ./Build.bend as B
import ./Spec.bend as S
import ../board/Spec.bend as P
import ../attack_witness/Queries.bend as Q
import ../../../legal_probe/Chess.bend as Chess

def queries(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+by: Bool,
  pa: {Chess.attack(P.tag(P.Pawn{}),B.run(d,seed,n,extra),U32.from_nat(q),U32.xor(Bool.to_u32(by),1),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))) : Array<U64> & U64},
  na: {Chess.attack(P.tag(P.Knight{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.Knight{},q,by,Chess.occupied(b))) : Array<U64> & U64},
  ka: {Chess.attack(P.tag(P.King{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.King{},q,by,Chess.occupied(b))) : Array<U64> & U64},
  ra: {Chess.attack(P.tag(P.Rook{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.Rook{},q,by,Chess.occupied(b))) : Array<U64> & U64},
  ba: {Chess.attack(P.tag(P.Bishop{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.Bishop{},q,by,Chess.occupied(b))) : Array<U64> & U64}) ->
  {Q.run(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & U64,Chess.attack(P.tag(P.Pawn{}),B.run(d,seed,n,extra),U32.from_nat(q),U32.xor(Bool.to_u32(by),1),Chess.occupied(b)),(B.run(d,seed,n,extra),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))),pa) :
    {Q.pawn(b,U32.from_nat(q),Bool.to_u32(by),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(P.tag(P.Knight{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)),(B.run(d,seed,n,extra),S.mask(P.Knight{},q,by,Chess.occupied(b))),na) :
    {Q.knight(b,U32.from_nat(q),Bool.to_u32(by),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b)),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(P.tag(P.King{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)),(B.run(d,seed,n,extra),S.mask(P.King{},q,by,Chess.occupied(b))),ka) :
    {Q.king(b,U32.from_nat(q),Bool.to_u32(by),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b)),S.mask(P.Knight{},q,by,Chess.occupied(b)),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(P.tag(P.Rook{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)),(B.run(d,seed,n,extra),S.mask(P.Rook{},q,by,Chess.occupied(b))),ra) :
    {Q.rook(b,U32.from_nat(q),Bool.to_u32(by),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b)),S.mask(P.Knight{},q,by,Chess.occupied(b)),S.mask(P.King{},q,by,Chess.occupied(b)),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(P.tag(P.Bishop{}),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),Chess.occupied(b)),(B.run(d,seed,n,extra),S.mask(P.Bishop{},q,by,Chess.occupied(b))),ba) :
    {Q.bishop(b,Bool.to_u32(by),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b)),S.mask(P.Knight{},q,by,Chess.occupied(b)),S.mask(P.King{},q,by,Chess.occupied(b)),S.mask(P.Rook{},q,by,Chess.occupied(b)),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}
  {==}
