import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../decoder/Projection.bend as Prior
import ../../../legal_probe/Chess.bend as Chess

def row(+width: Nat,cols: S.Planes<width>,i: Nat) -> B.Row:
  S.Planes{p,n,b,r,q,k,w,bl} = cols
  D.word_row(width,p,n,b,r,q,k,w,bl,i)

def cells(+width: Nat,+cols: S.Planes<width>,+i: Nat) ->
  {S.at(S.cells(width,cols),i) == D.classify(row(width,cols,i)) : D.Cell}:
  match width:
    case 0n:
      match cols:
        case S.Planes{WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{}}: {==}
    case 1n+t:
      match cols i:
        case S.Planes{WCon{p,pt},WCon{n,nt},WCon{b,bt},WCon{r,rt},WCon{q,qt},WCon{k,kt},WCon{w,wt},WCon{bl,blt}} 0n: {==}
        case S.Planes{WCon{p,pt},WCon{n,nt},WCon{b,bt},WCon{r,rt},WCon{q,qt},WCon{k,kt},WCon{w,wt},WCon{bl,blt}} 1n+j:
          cells(t,S.Planes{pt,nt,bt,rt,qt,kt,wt,blt},j)

def mix(+choice: Bool,+a: B.Row,+b: B.Row) ->
  {Bool.pick(D.Cell,choice,D.classify(a),D.classify(b)) == D.classify(D.mix(choice,a,b)) : D.Cell}:
  match choice a b:
    case True{} B.Row{ap,an,ab,ar,aq,ak,aw,abl} B.Row{bp,bn,bb,br,bq,bk,bw,bbl}: {==}
    case False{} B.Row{ap,an,ab,ar,aq,ak,aw,abl} B.Row{bp,bn,bb,br,bq,bk,bw,bbl}: {==}

def combine(+lo: S.Planes<32n>,+hi: S.Planes<32n>,+sq: U32) ->
  {Bool.pick(D.Cell,Nat.is_lt(U32.to_nat(sq),32n),S.at(S.cells(32n,lo),U32.to_nat(sq)),S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n))) ==
    D.classify(D.mix(Nat.is_lt(U32.to_nat(sq),32n),row(32n,lo,U32.to_nat(sq)),row(32n,hi,Nat.sub(U32.to_nat(sq),32n)))) : D.Cell}:
  +pick = Nat.is_lt(U32.to_nat(sq),32n)
  +low: B.Row = row(32n,lo,U32.to_nat(sq))
  +high: B.Row = row(32n,hi,Nat.sub(U32.to_nat(sq),32n))
  Equal.trans(D.Cell,Bool.pick(D.Cell,pick,S.at(S.cells(32n,lo),U32.to_nat(sq)),S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n))),
    Bool.pick(D.Cell,pick,D.classify(low),D.classify(high)),D.classify(D.mix(pick,low,high)),
    Equal.trans(D.Cell,Bool.pick(D.Cell,pick,S.at(S.cells(32n,lo),U32.to_nat(sq)),S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n))),
      Bool.pick(D.Cell,pick,D.classify(low),S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n))),Bool.pick(D.Cell,pick,D.classify(low),D.classify(high)),
      Equal.cong(D.Cell,D.Cell,c => Bool.pick(D.Cell,pick,c,S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n))),
        S.at(S.cells(32n,lo),U32.to_nat(sq)),D.classify(low),cells(32n,lo,U32.to_nat(sq))),
      Equal.cong(D.Cell,D.Cell,c => Bool.pick(D.Cell,pick,D.classify(low),c),
        S.at(S.cells(32n,hi),Nat.sub(U32.to_nat(sq),32n)),D.classify(high),cells(32n,hi,Nat.sub(U32.to_nat(sq),32n)))),mix(pick,low,high))

def structural(+board: Chess.Board,+sq: U32) ->
  {S.square(S.observe(board),sq) == D.classify(D.structural(board,sq)) : D.Cell}:
  match board:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},U64{U32{wl},U32{wh}},U64{U32{bll},U32{blh}},turn,rights,ep}:
      combine(S.Planes{pl,nl,bl,rl,ql,kl,wl,bll},S.Planes{ph,nh,bh,rh,qh,kh,wh,blh},sq)

def actual(+board: Chess.Board,+sq: U32,bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.square(S.observe(board),sq) == D.square(board,sq) : D.Cell}:
  Equal.trans(D.Cell,S.square(S.observe(board),sq),D.classify(D.structural(board,sq)),D.square(board,sq),
    structural(board,sq),Equal.cong(B.Row,D.Cell,r => D.classify(r),D.structural(board,sq),D.row(board,sq),
      Equal.sym(B.Row,D.row(board,sq),D.structural(board,sq),Prior.observations(board,sq,bound))))
