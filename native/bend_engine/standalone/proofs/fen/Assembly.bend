import Base
import ../../Position.bend as P
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

def assemble(+mb: Maybe<&2,Chess.Board>,+mt: Maybe<&2,U32>,+mr: Maybe<&2,U32>,+me: Maybe<&2,U32>,+mh: Maybe<&2,U32>,+mf: Maybe<&2,U32>) -> Maybe<&2,P.Game>:
  Maybe.bind(&2,Chess.Board,P.Game,mb,+b => Maybe.bind(&2,U32,P.Game,mt,+t => Maybe.bind(&2,U32,P.Game,mr,+r => Maybe.bind(&2,U32,P.Game,me,+e => Maybe.bind(&2,U32,P.Game,mh,+h => Maybe.bind(&2,U32,P.Game,Text.bounded(h),+bh => Maybe.bind(&2,U32,P.Game,mf,+f => Maybe.bind(&2,U32,P.Game,Text.positive(f),+pf => Maybe.bind(&2,U32,P.Game,Text.bounded(pf),+bf => Maybe.pure(&2,P.Game,P.Game{P.metadata(b,t,r,e),bh,bf,Nil{}}))))))))))

def actual(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  {P.fen(pieces,turn,castle,target,half,full) ==
    assemble(S.placement(pieces),P.side(turn),P.rights(castle),P.ep(target),Text.number(half),Text.number(full)) : Maybe<&2,P.Game>}:
  {==}

def rejected(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String,
  no: {S.placement(pieces) == None{} : Maybe<&2,Chess.Board>}) ->
  {P.fen(pieces,turn,castle,target,half,full) == None{} : Maybe<&2,P.Game>}:
  Equal.trans(Maybe<&2,P.Game>,P.fen(pieces,turn,castle,target,half,full),
    assemble(S.placement(pieces),P.side(turn),P.rights(castle),P.ep(target),Text.number(half),Text.number(full)),None{},
    actual(pieces,turn,castle,target,half,full),
    Equal.cong(Maybe<&2,Chess.Board>,Maybe<&2,P.Game>,m => assemble(m,P.side(turn),P.rights(castle),P.ep(target),Text.number(half),Text.number(full)),
      S.placement(pieces),None{},no))

def bounded(+n: U32,ok: {U32.is_le(n,1000000) == True{} : Bool}) ->
  {Text.bounded(n) == Some{n} : Maybe<&2,U32>}:
  Equal.cong(Bool,Maybe<&2,U32>,b => Text.only(b,n),U32.is_le(n,1000000),True{},ok)

def positive(+n: U32,ok: {U32.is_gt(n,0) == True{} : Bool}) ->
  {Text.positive(n) == Some{n} : Maybe<&2,U32>}:
  Equal.cong(Bool,Maybe<&2,U32>,b => Text.only(b,n),U32.is_gt(n,0),True{},ok)

def certified(+mb: Maybe<&2,Chess.Board>,+mt: Maybe<&2,U32>,+mr: Maybe<&2,U32>,+me: Maybe<&2,U32>,+mh: Maybe<&2,U32>,+mf: Maybe<&2,U32>,
  +b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,+f: U32,
  pmb: {mb == Some{b} : Maybe<&2,Chess.Board>},
  pmt: {mt == Some{t} : Maybe<&2,U32>},
  pmr: {mr == Some{r} : Maybe<&2,U32>},
  pme: {me == Some{e} : Maybe<&2,U32>},
  pmh: {mh == Some{h} : Maybe<&2,U32>},
  pmf: {mf == Some{f} : Maybe<&2,U32>},
  hb: {U32.is_le(h,1000000) == True{} : Bool},fp: {U32.is_gt(f,0) == True{} : Bool},fb: {U32.is_le(f,1000000) == True{} : Bool}) ->
  {assemble(mb,mt,mr,me,mh,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}:
  %Equal.sym(Maybe<&2,Chess.Board>,mb,Some{b},pmb) : {assemble(_,mt,mr,me,mh,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,mt,Some{t},pmt) : {assemble(Some{b},_,mr,me,mh,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,mr,Some{r},pmr) : {assemble(Some{b},Some{t},_,me,mh,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,me,Some{e},pme) : {assemble(Some{b},Some{t},Some{r},_,mh,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,mh,Some{h},pmh) : {assemble(Some{b},Some{t},Some{r},Some{e},_,mf) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,mf,Some{f},pmf) : {assemble(Some{b},Some{t},Some{r},Some{e},Some{h},_) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,Text.bounded(h),Some{h},bounded(h,hb)) :
    {Maybe.bind(&2,U32,P.Game,_,+bh => Maybe.bind(&2,U32,P.Game,Text.positive(f),+pf => Maybe.bind(&2,U32,P.Game,Text.bounded(pf),+bf => Maybe.pure(&2,P.Game,P.Game{P.metadata(b,t,r,e),bh,bf,Nil{}})))) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,Text.positive(f),Some{f},positive(f,fp)) :
    {Maybe.bind(&2,U32,P.Game,_,+pf => Maybe.bind(&2,U32,P.Game,Text.bounded(pf),+bf => Maybe.pure(&2,P.Game,P.Game{P.metadata(b,t,r,e),h,bf,Nil{}}))) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  %Equal.sym(Maybe<&2,U32>,Text.bounded(f),Some{f},bounded(f,fb)) :
    {Maybe.bind(&2,U32,P.Game,_,+bf => Maybe.pure(&2,P.Game,P.Game{P.metadata(b,t,r,e),h,bf,Nil{}})) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
  {==}

def fields(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String,
  +b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,+f: U32,
  pb: {S.placement(pieces) == Some{b} : Maybe<&2,Chess.Board>},
  pt: {P.side(turn) == Some{t} : Maybe<&2,U32>},
  pr: {P.rights(castle) == Some{r} : Maybe<&2,U32>},
  pe: {P.ep(target) == Some{e} : Maybe<&2,U32>},
  ph: {Text.number(half) == Some{h} : Maybe<&2,U32>},
  pf: {Text.number(full) == Some{f} : Maybe<&2,U32>},
  hb: {U32.is_le(h,1000000) == True{} : Bool},fp: {U32.is_gt(f,0) == True{} : Bool},fb: {U32.is_le(f,1000000) == True{} : Bool}) ->
  {P.fen(pieces,turn,castle,target,half,full) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}:
  Equal.trans(Maybe<&2,P.Game>,P.fen(pieces,turn,castle,target,half,full),
    assemble(S.placement(pieces),P.side(turn),P.rights(castle),P.ep(target),Text.number(half),Text.number(full)),Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}},
    actual(pieces,turn,castle,target,half,full),
    certified(S.placement(pieces),P.side(turn),P.rights(castle),P.ep(target),Text.number(half),Text.number(full),b,t,r,e,h,f,pb,pt,pr,pe,ph,pf,hb,fp,fb))
