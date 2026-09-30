import Base
import ./Spec.bend as S
import ./Build.bend as B
import ./Facts.bend as F
import ../board/Spec.bend as P
import ../attack_geometry/Geometry.bend as G
import ../attack_geometry/Query.bend as LQ
import ../attack_geometry/LAWS.bend as Leaper
import ../attack_geometry/PROOF.bend as LeaperProof
import ../builder/LAWS.bend as Slider
import ../builder/PROOF.bend as SliderProof
import ../../../legal_probe/Chess.bend as Chess

def rook(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+q: Nat,+side: Bool,+occ: U64,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(3,B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(side),occ) ==
    (B.run(d,seed,n,extra),S.mask(P.Rook{},q,side,occ)) : Array<U64> & U64}:
  Slider.full_table_lookup(d,seed,n,q,extra,0n,occ,F.rook(q,bound),depth,tables,B.extras_bound(extra,full))

def bishop(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+q: Nat,+side: Bool,+occ: U64,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},+bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(2,B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(side),occ) ==
    (B.run(d,seed,n,extra),S.mask(P.Bishop{},q,side,occ)) : Array<U64> & U64}:
  %Equal.sym(U32,U32.add(64,U32.from_nat(q)),U32.from_nat(Nat.add(64n,q)),F.key(q,bound)) :
    {Chess.slide(B.run(d,seed,n,extra),_,occ) == (B.run(d,seed,n,extra),S.mask(P.Bishop{},q,side,occ)) : Array<U64> & U64}
  Slider.full_table_lookup(d,seed,n,Nat.add(64n,q),extra,0n,occ,F.bishop(q,bound),depth,tables,B.extras_bound(extra,full))

def leaper(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+q: Nat,+kind: G.Leaper,+occ: U64,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(LQ.tag(kind),B.run(d,seed,n,extra),U32.from_nat(q),LQ.side(kind),occ) ==
    (B.run(d,seed,n,extra),G.mask(kind,q)) : Array<U64> & U64}:
  Leaper.initialized_leaper_query_matches_coordinates(d,seed,n,0,512,extra,kind,q,occ,depth,full,bound)

def actual(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+kind: P.Piece,+q: Nat,+side: Bool,+occ: U64,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},+bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(P.tag(kind),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(side),occ) ==
    (B.run(d,seed,n,extra),S.mask(kind,q,side,occ)) : Array<U64> & U64}:
  match kind:
    case P.Pawn{}:
      match side:
        case True{}: leaper(d,seed,n,extra,q,G.WhitePawn{},occ,depth,full,bound)
        case False{}: leaper(d,seed,n,extra,q,G.BlackPawn{},occ,depth,full,bound)
    case P.Knight{}: leaper(d,seed,n,extra,q,G.Knight{},occ,depth,full,bound)
    case P.King{}: leaper(d,seed,n,extra,q,G.King{},occ,depth,full,bound)
    case P.Rook{}: rook(d,seed,n,extra,q,side,occ,depth,tables,full,bound)
    case P.Bishop{}: bishop(d,seed,n,extra,q,side,occ,depth,tables,full,bound)
    case P.Queen{}:
      %Equal.sym(Array<U64> & U64,Chess.attack(3,B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(side),occ),(B.run(d,seed,n,extra),S.mask(P.Rook{},q,side,occ)),rook(d,seed,n,extra,q,side,occ,depth,tables,full,bound)) :
        {Chess.queen_after(U32.from_nat(q),occ,_) ==
          (B.run(d,seed,n,extra),S.mask(P.Queen{},q,side,occ)) : Array<U64> & U64}
      %Equal.sym(Array<U64> & U64,Chess.attack(2,B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(side),occ),(B.run(d,seed,n,extra),S.mask(P.Bishop{},q,side,occ)),bishop(d,seed,n,extra,q,side,occ,depth,tables,full,bound)) :
        {Chess.or_result(S.mask(P.Rook{},q,side,occ),_) ==
          (B.run(d,seed,n,extra),S.mask(P.Queen{},q,side,occ)) : Array<U64> & U64}
      {==}
