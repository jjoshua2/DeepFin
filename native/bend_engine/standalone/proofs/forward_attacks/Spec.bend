# Independent source-centred geometry and owned-piece witness scan.
# No production attack query, reverse mask, table read, or accumulator defines the answer.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../attack_geometry/Geometry.bend as G
import ../attack_reversal/Spec.bend as Member
import ../slider_reversal/Spec.bend as Slider
import ../attack_witness/Spec.bend as W

type AttackKind is Data:
  Pawn{}
  Knight{}
  King{}
  Rook{}
  Bishop{}

def edge(kind: AttackKind,+src: Nat,+dst: Nat,by: Bool,occ: U64) -> Bool:
  match kind:
    case Pawn{}: Member.contains(dst,G.targets(Bool.pick(G.Leaper,by,G.WhitePawn{},G.BlackPawn{}),src))
    case Knight{}: Member.contains(dst,G.targets(G.Knight{},src))
    case King{}: Member.contains(dst,G.targets(G.King{},src))
    case Rook{}: Slider.answer(Slider.Rook{},Slider.paths(src,dst),occ)
    case Bishop{}: Slider.answer(Slider.Bishop{},Slider.paths(src,dst),occ)

# A proof-only tabulation of the forward predicate. attacked uses scan below.
def word(n: Nat,+src: Nat,+kind: AttackKind,+dst: Nat,+by: Bool,+occ: U64) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p: WCon{edge(kind,src,dst,by,occ),word(p,1n+src,kind,dst,by,occ)}

def row(+own: Bool,+p: Bool,+n: Bool,+b: Bool,+r: Bool,+q: Bool,+k: Bool,+pa: Bool,+na: Bool,+ka: Bool,+ra: Bool,+ba: Bool) -> Bool:
  Bool.and(own,Bool.or(Bool.or(Bool.or(Bool.or(Bool.and(p,pa),Bool.and(n,na)),Bool.and(k,ka)),Bool.and(Bool.or(r,q),ra)),Bool.and(Bool.or(b,q),ba)))

def scan(+n: Nat,+src: Nat,own: Word(n),p: Word(n),kn: Word(n),b: Word(n),r: Word(n),q: Word(n),k: Word(n),+dst: Nat,+by: Bool,+occ: U64) -> Bool:
  match n:
    case 0n: False{}
    case 1n+t:
      match own p kn b r q k:
        case WCon{oh,ot} WCon{ph,pt} WCon{nh,nt} WCon{bh,bt} WCon{rh,rt} WCon{qh,qt} WCon{kh,kt}:
          Bool.or(row(oh,ph,nh,bh,rh,qh,kh,edge(Pawn{},src,dst,by,occ),edge(Knight{},src,dst,by,occ),edge(King{},src,dst,by,occ),edge(Rook{},src,dst,by,occ),edge(Bishop{},src,dst,by,occ)),scan(t,1n+src,ot,pt,nt,bt,rt,qt,kt,dst,by,occ))

def observe(b: Chess.Board,own: U64,+dst: Nat,+by: Bool,+occ: U64) -> Bool:
  match b own:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},white,black,turn,rights,ep} U64{U32{ol},U32{oh}}:
      Bool.or(scan(32n,0n,ol,pl,nl,bl,rl,ql,kl,dst,by,occ),scan(32n,32n,oh,ph,nh,bh,rh,qh,kh,dst,by,occ))

def attacked(+b: Chess.Board,target: Nat,+by: Bool) -> Bool:
  observe(b,W.select(b,by),target,by,Chess.occupied(b))
