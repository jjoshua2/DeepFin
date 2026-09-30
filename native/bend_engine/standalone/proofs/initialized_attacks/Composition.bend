# State-thread all five initialized masks into the independent witness scan.
import Base
import ./Build.bend as B
import ./Spec.bend as S
import ./Facts.bend as F
import ./Masks.bend as M
import ./Wire.bend as Wire
import ../board/Spec.bend as P
import ../attack_witness/Queries.bend as Q
import ../attack_witness/Flow.bend as Flow
import ../attack_witness/King.bend as King
import ../../../legal_probe/Chess.bend as Chess

def reverse_pawn(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+by: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(0,B.run(d,seed,n,extra),U32.from_nat(q),U32.xor(Bool.to_u32(by),1),Chess.occupied(b)) ==
    (B.run(d,seed,n,extra),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))) : Array<U64> & U64}:
  %Equal.sym(U32,U32.xor(Bool.to_u32(by),1),Bool.to_u32(Bool.not(by)),F.flip(by)) :
    {Chess.attack(0,B.run(d,seed,n,extra),U32.from_nat(q),_,Chess.occupied(b)) ==
      (B.run(d,seed,n,extra),S.mask(P.Pawn{},q,Bool.not(by),Chess.occupied(b))) : Array<U64> & U64}
  M.actual(d,seed,n,extra,P.Pawn{},q,Bool.not(by),Chess.occupied(b),depth,tables,full,bound)

def queries(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+by: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},+bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Q.run(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}:
  Wire.queries(d,seed,n,extra,b,q,by,
    reverse_pawn(d,seed,n,extra,b,q,by,depth,tables,full,bound),
    M.actual(d,seed,n,extra,P.Knight{},q,by,Chess.occupied(b),depth,tables,full,bound),
    M.actual(d,seed,n,extra,P.King{},q,by,Chess.occupied(b),depth,tables,full,bound),
    M.actual(d,seed,n,extra,P.Rook{},q,by,Chess.occupied(b),depth,tables,full,bound),
    M.actual(d,seed,n,extra,P.Bishop{},q,by,Chess.occupied(b),depth,tables,full,bound))

def attacked(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+by: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attacked(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,
    Chess.attacked(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)),
    Q.run(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)),
    (B.run(d,seed,n,extra),S.attacked(b,q,by)),
    Flow.attacked(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)),
    queries(d,seed,n,extra,b,q,by,depth,tables,full,bound))

def checked(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},+bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(side))) == U64.bit(q) : U64}) ->
  {Chess.in_check(B.run(d,seed,n,extra),b,Bool.to_u32(side)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,Bool.not(side))) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & Bool,Chess.in_check(B.run(d,seed,n,extra),b,Bool.to_u32(side)),Q.run(B.run(d,seed,n,extra),b,U32.from_nat(q),U32.xor(Bool.to_u32(side),1)),King.checked(B.run(d,seed,n,extra),b,Bool.to_u32(side),q,bound,one)) :
    {_ == (B.run(d,seed,n,extra),S.attacked(b,q,Bool.not(side))) : Array<U64> & Bool}
  %Equal.sym(U32,U32.xor(Bool.to_u32(side),1),Bool.to_u32(Bool.not(side)),F.flip(side)) :
    {Q.run(B.run(d,seed,n,extra),b,U32.from_nat(q),_) ==
      (B.run(d,seed,n,extra),S.attacked(b,q,Bool.not(side))) : Array<U64> & Bool}
  queries(d,seed,n,extra,b,q,Bool.not(side),depth,tables,full,bound)
