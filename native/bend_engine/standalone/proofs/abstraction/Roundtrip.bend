import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../decoder/Rows.bend as Rows
import ../ray/Frame.bend as F
import ../../../legal_probe/Chess.bend as Chess

def planes(+width: Nat,+cols: S.Planes<width>,+good: {S.valid(width,cols) == True{} : Bool}) ->
  {S.columns(width,S.cells(width,cols)) == cols : S.Planes<width>}:
  match width:
    case 0n:
      match cols:
        case S.Planes{WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{}}: {==}
    case 1n+t:
      match cols:
        case S.Planes{WCon{+p,+pt},WCon{+n,+nt},WCon{+b,+bt},WCon{+r,+rt},WCon{+q,+qt},WCon{+k,+kt},WCon{+w,+wt},WCon{+bl,+blt}}:
          +row: B.Row = B.Row{p,n,b,r,q,k,w,bl}
          +tail: S.Planes<t> = S.Planes{pt,nt,bt,rt,qt,kt,wt,blt}
          Equal.trans(S.Planes<1n+t>,S.cons(t,D.encode(D.classify(row)),S.columns(t,S.cells(t,tail))),
            S.cons(t,row,S.columns(t,S.cells(t,tail))),S.cons(t,row,tail),
            Equal.cong(B.Row,S.Planes<1n+t>,a => S.cons(t,a,S.columns(t,S.cells(t,tail))),D.encode(D.classify(row)),row,
              Rows.roundtrip(row,F.left(B.row_valid(row),S.valid(t,tail),good))),
            Equal.cong(S.Planes<t>,S.Planes<1n+t>,a => S.cons(t,row,a),S.columns(t,S.cells(t,tail)),tail,
              planes(t,tail,F.right(B.row_valid(row),S.valid(t,tail),good))))

def board(+board: Chess.Board,+good: {B.valid(board) == True{} : Bool}) ->
  {S.restore(S.observe(board)) == board : Chess.Board}:
  match board:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},+turn,+rights,+ep}:
      +lo: S.Planes<32n> = S.Planes{pl,nl,bl,rl,ql,kl,wl,bll}
      +hi: S.Planes<32n> = S.Planes{ph,nh,bh,rh,qh,kh,wh,blh}
      Equal.trans(Chess.Board,S.join(S.columns(32n,S.cells(32n,lo)),S.columns(32n,S.cells(32n,hi)),turn,rights,ep),
        S.join(lo,S.columns(32n,S.cells(32n,hi)),turn,rights,ep),S.join(lo,hi,turn,rights,ep),
        Equal.cong(S.Planes<32n>,Chess.Board,a => S.join(a,S.columns(32n,S.cells(32n,hi)),turn,rights,ep),S.columns(32n,S.cells(32n,lo)),lo,
          planes(32n,lo,F.left(S.valid(32n,lo),S.valid(32n,hi),good))),
        Equal.cong(S.Planes<32n>,Chess.Board,a => S.join(lo,a,turn,rights,ep),S.columns(32n,S.cells(32n,hi)),hi,
          planes(32n,hi,F.right(S.valid(32n,lo),S.valid(32n,hi),good))))
