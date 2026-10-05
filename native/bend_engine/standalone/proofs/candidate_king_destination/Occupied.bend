import Base
import ../board/Spec.bend as B
import ../ray/Frame.bend as F
import ../ordinary_king_frame/Algebra.bend as Parts
import ../u64/Words.bend as Old
import ../occupancy_insertion/Lattice.bend as L
import ../../../legal_probe/Chess.bend as Chess

def row_king(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  k
def row_occupied(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  Bool.or(w,bl)

def row_cover(+row: B.Row,good: {B.row_valid(row) == True{} : Bool}) ->
  {Bool.and(row_occupied(row),row_king(row)) == row_king(row) : Bool}:
  match row:
    case B.Row{p,n,b,r,q,False{},True{},bl}: {==}
    case B.Row{p,n,b,r,q,False{},False{},False{}}: {==}
    case B.Row{p,n,b,r,q,False{},False{},True{}}: {==}
    case B.Row{p,n,b,r,q,True{},True{},bl}: {==}
    case B.Row{p,n,b,r,q,True{},False{},True{}}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},False{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},False{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},False{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},True{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},True{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},True{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},False{},True{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},False{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},False{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},False{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},False{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},True{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},True{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},True{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{False{},True{},True{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},False{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},False{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},True{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},True{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},False{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},False{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},True{},False{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},True{},False{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},False{},False{}}: Old.false_true({False{} == True{} : Bool},good)

def words_cover(+width: Nat,+p: Word(width),+n: Word(width),+b: Word(width),+r: Word(width),+q: Word(width),+k: Word(width),+w: Word(width),+bl: Word(width),
  +good: {B.words(width,p,n,b,r,q,k,w,bl) == True{} : Bool}) ->
  {Word.and(width,Word.or(width,w,bl),k) == k : Word(width)}:
  match width:
    case 0n:
      match k:
        case WNil{}: {==}
    case 1n+t:
      match p n b r q k w bl:
        case WCon{+ph,+pt} WCon{+nh,+nt} WCon{+bh,+bt} WCon{+rh,+rt} WCon{+qh,+qt} WCon{+kh,+kt} WCon{+wh,+wt} WCon{+blh,+blt}:
          Equal.trans(Word(1n+t),WCon{Bool.and(Bool.or(wh,blh),kh),Word.and(t,Word.or(t,wt,blt),kt)},
            WCon{kh,Word.and(t,Word.or(t,wt,blt),kt)},WCon{kh,kt},
            Equal.cong(Bool,Word(1n+t),h => WCon{h,Word.and(t,Word.or(t,wt,blt),kt)},Bool.and(Bool.or(wh,blh),kh),kh,
              row_cover(B.Row{ph,nh,bh,rh,qh,kh,wh,blh},F.left(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good))),
            Equal.cong(Word(t),Word(1n+t),tail => WCon{kh,tail},Word.and(t,Word.or(t,wt,blt),kt),kt,
              words_cover(t,pt,nt,bt,rt,qt,kt,wt,blt,F.right(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good))))

def kings_occupied(+board: Chess.Board,+good: {B.valid(board) == True{} : Bool}) -> L.subset(Chess.get_kings(board),Chess.occupied(board)):
  match board:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,wl,bll),kl)},U32{Word.and(32n,Word.or(32n,wh,blh),kh)}},
        U64{U32{kl},U32{Word.and(32n,Word.or(32n,wh,blh),kh)}},U64{U32{kl},U32{kh}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,wh,blh),kh)}},Word.and(32n,Word.or(32n,wl,bll),kl),kl,
          words_cover(32n,pl,nl,bl,rl,ql,kl,wl,bll,F.left(B.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good))),
        Equal.cong(Word(32n),U64,h => U64{U32{kl},U32{h}},Word.and(32n,Word.or(32n,wh,blh),kh),kh,
          words_cover(32n,ph,nh,bh,rh,qh,kh,wh,blh,F.right(B.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good))))
