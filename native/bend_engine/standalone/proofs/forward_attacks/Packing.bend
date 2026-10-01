# Forward masks are a proof intermediate, then eliminated into the direct scan.
import Base
import ./Spec.bend as S
import ./Rows.bend as Rows
import ../attack_witness/Spec.bend as W
import ../attack_witness/Reduction.bend as R
import ../../../legal_probe/Chess.bend as Chess

def mask(+kind: S.AttackKind,+dst: Nat,+by: Bool,+occ: U64) -> U64:
  U64{U32{S.word(32n,0n,kind,dst,by,occ)},U32{S.word(32n,32n,kind,dst,by,occ)}}

def masks(+dst: Nat,+by: Bool,+occ: U64) -> W.Masks:
  W.Masks{mask(S.Pawn{},dst,by,occ),mask(S.Knight{},dst,by,occ),mask(S.King{},dst,by,occ),mask(S.Rook{},dst,by,occ),mask(S.Bishop{},dst,by,occ)}

def scan(+n: Nat,+src: Nat,+own: Word(n),+p: Word(n),+kn: Word(n),+b: Word(n),+r: Word(n),+q: Word(n),+k: Word(n),+dst: Nat,+by: Bool,+occ: U64) ->
  {W.scan(n,own,p,kn,b,r,q,k,S.word(n,src,S.Pawn{},dst,by,occ),S.word(n,src,S.Knight{},dst,by,occ),S.word(n,src,S.King{},dst,by,occ),S.word(n,src,S.Rook{},dst,by,occ),S.word(n,src,S.Bishop{},dst,by,occ)) == S.scan(n,src,own,p,kn,b,r,q,k,dst,by,occ) : Bool}:
  match n:
    case 0n: {==}
    case 1n+t:
      match own p kn b r q k:
        case WCon{+oh,+ot} WCon{+ph,+pt} WCon{+nh,+nt} WCon{+bh,+bt} WCon{+rh,+rt} WCon{+qh,+qt} WCon{+kh,+kt}:
          R.both(
            W.row(oh,ph,nh,bh,rh,qh,kh,S.edge(S.Pawn{},src,dst,by,occ),S.edge(S.Knight{},src,dst,by,occ),S.edge(S.King{},src,dst,by,occ),S.edge(S.Rook{},src,dst,by,occ),S.edge(S.Bishop{},src,dst,by,occ)),
            W.scan(t,ot,pt,nt,bt,rt,qt,kt,S.word(t,1n+src,S.Pawn{},dst,by,occ),S.word(t,1n+src,S.Knight{},dst,by,occ),S.word(t,1n+src,S.King{},dst,by,occ),S.word(t,1n+src,S.Rook{},dst,by,occ),S.word(t,1n+src,S.Bishop{},dst,by,occ)),
            S.row(oh,ph,nh,bh,rh,qh,kh,S.edge(S.Pawn{},src,dst,by,occ),S.edge(S.Knight{},src,dst,by,occ),S.edge(S.King{},src,dst,by,occ),S.edge(S.Rook{},src,dst,by,occ),S.edge(S.Bishop{},src,dst,by,occ)),
            S.scan(t,1n+src,ot,pt,nt,bt,rt,qt,kt,dst,by,occ),
            Rows.same(oh,ph,nh,bh,rh,qh,kh,S.edge(S.Pawn{},src,dst,by,occ),S.edge(S.Knight{},src,dst,by,occ),S.edge(S.King{},src,dst,by,occ),S.edge(S.Rook{},src,dst,by,occ),S.edge(S.Bishop{},src,dst,by,occ)),
            scan(t,1n+src,ot,pt,nt,bt,rt,qt,kt,dst,by,occ))

def observe(+b: Chess.Board,+own: U64,+dst: Nat,+by: Bool,+occ: U64) ->
  {W.observe(b,own,masks(dst,by,occ)) == S.observe(b,own,dst,by,occ) : Bool}:
  match b own:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},white,black,turn,rights,ep} U64{U32{+ol},U32{+oh}}:
      R.both(
        W.scan(32n,ol,pl,nl,bl,rl,ql,kl,S.word(32n,0n,S.Pawn{},dst,by,occ),S.word(32n,0n,S.Knight{},dst,by,occ),S.word(32n,0n,S.King{},dst,by,occ),S.word(32n,0n,S.Rook{},dst,by,occ),S.word(32n,0n,S.Bishop{},dst,by,occ)),
        W.scan(32n,oh,ph,nh,bh,rh,qh,kh,S.word(32n,32n,S.Pawn{},dst,by,occ),S.word(32n,32n,S.Knight{},dst,by,occ),S.word(32n,32n,S.King{},dst,by,occ),S.word(32n,32n,S.Rook{},dst,by,occ),S.word(32n,32n,S.Bishop{},dst,by,occ)),
        S.scan(32n,0n,ol,pl,nl,bl,rl,ql,kl,dst,by,occ),S.scan(32n,32n,oh,ph,nh,bh,rh,qh,kh,dst,by,occ),
        scan(32n,0n,ol,pl,nl,bl,rl,ql,kl,dst,by,occ),scan(32n,32n,oh,ph,nh,bh,rh,qh,kh,dst,by,occ))

def present(+b: Chess.Board,+dst: Nat,+by: Bool) ->
  {W.present(b,Bool.to_u32(by),masks(dst,by,Chess.occupied(b))) == S.attacked(b,dst,by) : Bool}:
  match by:
    case False{}: observe(b,W.select(b,False{}),dst,False{},Chess.occupied(b))
    case True{}: observe(b,W.select(b,True{}),dst,True{},Chess.occupied(b))
