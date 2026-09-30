import Base
import ./Spec.bend as S
import ./Roundtrip.bend as R
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../decoder/LAWS.bend as DL
import ../decoder/PROOF.bend as DP
import ../../../legal_probe/Chess.bend as Chess

def length(+width: Nat,+cols: S.Planes<width>) -> {S.length(S.cells(width,cols)) == width : Nat}:
  match width:
    case 0n: {==}
    case 1n+t:
      match cols:
        case S.Planes{WCon{p,+pt},WCon{n,+nt},WCon{b,+bt},WCon{r,+rt},WCon{q,+qt},WCon{k,+kt},WCon{w,+wt},WCon{bl,+blt}}:
          Equal.cong(Nat,Nat,n => 1n+n,S.length(S.cells(t,S.Planes{pt,nt,bt,rt,qt,kt,wt,blt})),t,
            length(t,S.Planes{pt,nt,bt,rt,qt,kt,wt,blt}))

def lengths(+board: Chess.Board) -> {S.lengths(S.observe(board)) == (32n,32n) : Nat & Nat}:
  match board:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep}:
      +lo: S.Planes<32n> = S.Planes{pl,nl,bl,rl,ql,kl,wl,bll}
      +hi: S.Planes<32n> = S.Planes{ph,nh,bh,rh,qh,kh,wh,blh}
      Equal.trans(Nat & Nat,(S.length(S.cells(32n,lo)),S.length(S.cells(32n,hi))),
        (32n,S.length(S.cells(32n,hi))),(32n,32n),
        Equal.cong(Nat,Nat & Nat,n => (n,S.length(S.cells(32n,hi))),S.length(S.cells(32n,lo)),32n,length(32n,lo)),
        Equal.cong(Nat,Nat & Nat,n => (32n,n),S.length(S.cells(32n,hi)),32n,length(32n,hi)))

def injective(+a: Chess.Board,+b: Chess.Board,ga: {B.valid(a) == True{} : Bool},gb: {B.valid(b) == True{} : Bool},
  same: {S.observe(a) == S.observe(b) : S.Snapshot}) -> {a == b : Chess.Board}:
  Equal.trans(Chess.Board,a,S.restore(S.observe(a)),b,
    Equal.sym(Chess.Board,S.restore(S.observe(a)),a,R.board(a,ga)),
    Equal.trans(Chess.Board,S.restore(S.observe(a)),S.restore(S.observe(b)),b,
      Equal.cong(S.Snapshot,Chess.Board,s => S.restore(s),S.observe(a),S.observe(b),same),R.board(b,gb)))

# Reconcile the saved local empty-square contract with the published decoder suite.
def empty(+b: Chess.Board,+sq: U32,+good: {B.valid(b) == True{} : Bool},
  +bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  absent: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == False{} : Bool}) ->
  {D.row(b,sq) == D.encode(D.Empty{}) : B.Row}:
  Equal.trans(B.Row,D.row(b,sq),D.encode(D.square(b,sq)),D.encode(D.Empty{}),
    Equal.sym(B.Row,D.encode(D.square(b,sq)),D.row(b,sq),DL.abstract_square_roundtrip(b,sq,good,bound)),
    Equal.cong(D.Cell,B.Row,c => D.encode(c),D.square(b,sq),D.Empty{},
      Equal.trans(D.Cell,D.square(b,sq),D.decoded(b,sq),D.Empty{},
        Equal.sym(D.Cell,D.decoded(b,sq),D.square(b,sq),DL.guarded_decoder_matches_square(b,sq,good,bound)),
        Equal.cong(Bool,D.Cell,flag => Bool.pick(D.Cell,flag,D.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},D.Empty{}),
          U64.test_bit(Chess.occupied(b),U32.to_nat(sq)),False{},absent))))
