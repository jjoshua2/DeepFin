# Actual accepted transitions preserve a bounded, consistent, empty-ahead frontier.
import Base
import ./Spec.bend as S
import ./Logic.bend as L
import ./Finite.bend as Finite
import ./Facts.bend as Facts
import ./Fresh.bend as Fresh
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../parser/Spec.bend as PS
import ../parser/Actual.bend as Parent
import ../parser/Dispatch.bend as Dispatch
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess
import ../layout/Domain.bend as Domain

def bounded(+b: Chess.Board,+f: U32,+r: U32,e: {S.state(b,f,r) == True{} : Bool}) -> {S.bounds(f,r) == True{} : Bool}:
  L.left(S.bounds(f,r),Bool.and(B.valid(b),B.fresh_mask(b,S.pending(f,r))),e)

def consistent(+b: Chess.Board,+f: U32,+r: U32,e: {S.state(b,f,r) == True{} : Bool}) -> {B.valid(b) == True{} : Bool}:
  L.left(B.valid(b),B.fresh_mask(b,S.pending(f,r)),L.right(S.bounds(f,r),Bool.and(B.valid(b),B.fresh_mask(b,S.pending(f,r))),e))

def empty_ahead(+b: Chess.Board,+f: U32,+r: U32,e: {S.state(b,f,r) == True{} : Bool}) -> {B.fresh_mask(b,S.pending(f,r)) == True{} : Bool}:
  L.right(B.valid(b),B.fresh_mask(b,S.pending(f,r)),L.right(S.bounds(f,r),Bool.and(B.valid(b),B.fresh_mask(b,S.pending(f,r))),e))

def make(+b: Chess.Board,+f: U32,+r: U32,
  bounds: {S.bounds(f,r) == True{} : Bool},valid: {B.valid(b) == True{} : Bool},
  fresh: {B.fresh_mask(b,S.pending(f,r)) == True{} : Bool}) -> {S.state(b,f,r) == True{} : Bool}:
  L.both(S.bounds(f,r),Bool.and(B.valid(b),B.fresh_mask(b,S.pending(f,r))),bounds,
    L.both(B.valid(b),B.fresh_mask(b,S.pending(f,r)),valid,fresh))

def piece_cert(+b: Chess.Board,+f: U32,+r: U32,+pc: B.Piece,+white: Bool,
  +good: {S.state(b,f,r) == True{} : Bool},cert: Facts.piece_statement(f,r)) ->
  {S.state(P.put(B.tag(pc),white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool}:
  (nb,(sub,(current,(apart,addr)))) = cert
  make(P.put(B.tag(pc),white,S.square(f,r),b),U32.inc(f),r,nb,
    BA.insert(b,S.square(f,r),pc,white,consistent(b,f,r,good),
      Fresh.narrow(b,U64.bit(U32.to_nat(S.square(f,r))),S.pending(f,r),empty_ahead(b,f,r,good),current)),
    Fresh.after_mask(b,S.square(f,r),pc,white,S.pending(U32.inc(f),r),
      Fresh.narrow(b,S.pending(U32.inc(f),r),S.pending(f,r),empty_ahead(b,f,r,good),sub),apart))

def piece(+b: Chess.Board,+f: U32,+r: U32,+pc: B.Piece,+white: Bool,
  +good: {S.state(b,f,r) == True{} : Bool},fit: {U32.is_lt(f,8) == True{} : Bool}) ->
  {S.state(P.put(B.tag(pc),white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool}:
  piece_cert(b,f,r,pc,white,good,Facts.piece(f,r,bounded(b,f,r,good),fit))

def raw_nat(+kind: Nat,+b: Chess.Board,+f: U32,+r: U32,+white: Bool,
  bound: {Nat.is_lt(kind,6n) == True{} : Bool},good: {S.state(b,f,r) == True{} : Bool},fit: {U32.is_lt(f,8) == True{} : Bool}) ->
  {S.state(P.put(U32.from_nat(kind),white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool}:
  match kind:
    case 0n: piece(b,f,r,B.Pawn{},white,good,fit)
    case 1n: piece(b,f,r,B.Knight{},white,good,fit)
    case 2n: piece(b,f,r,B.Bishop{},white,good,fit)
    case 3n: piece(b,f,r,B.Rook{},white,good,fit)
    case 4n: piece(b,f,r,B.Queen{},white,good,fit)
    case 5n: piece(b,f,r,B.King{},white,good,fit)
    case 6n+p: Domain.impossible({S.state(P.put(U32.from_nat(6n+p),white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool},p,bound)

def raw_piece(+kind: U32,+b: Chess.Board,+f: U32,+r: U32,+white: Bool,
  bound: {U32.is_lt(kind,6) == True{} : Bool},good: {S.state(b,f,r) == True{} : Bool},fit: {U32.is_lt(f,8) == True{} : Bool}) ->
  {S.state(P.put(kind,white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool}:
  Finite.u6(kind,Finite.lt_nat(kind,6,bound),
    k => {S.state(P.put(k,white,S.square(f,r),b),U32.inc(f),r) == True{} : Bool},
    n => kb => raw_nat(n,b,f,r,white,kb,good,fit))

def digit_cert(+b: Chess.Board,+f: U32,+r: U32,+c: U32,
  +good: {S.state(b,f,r) == True{} : Bool},cert: Facts.digit_statement(f,r,c)) ->
  {S.state(b,U32.add(f,U32.sub(c,48)),r) == True{} : Bool}:
  (bound,sub) = cert
  make(b,U32.add(f,U32.sub(c,48)),r,bound,consistent(b,f,r,good),
    Fresh.narrow(b,S.pending(U32.add(f,U32.sub(c,48)),r),S.pending(f,r),empty_ahead(b,f,r,good),sub))

def slash_cert(+b: Chess.Board,+f: U32,+r: U32,
  +good: {S.state(b,f,r) == True{} : Bool},cert: Facts.slash_statement(f,r)) ->
  {S.state(b,0,U32.sub(r,1)) == True{} : Bool}:
  (bound,sub) = cert
  make(b,0,U32.sub(r,1),bound,consistent(b,f,r,good),
    Fresh.narrow(b,S.pending(0,U32.sub(r,1)),S.pending(f,r),empty_ahead(b,f,r,good),sub))

def guarded(+b: Chess.Board,+f: U32,+r: U32,+live: Bool,
  yes: {live == True{} : Bool} -> {S.state(b,f,r) == True{} : Bool}) ->
  {S.safe(P.Layout{b,f,r,live}) == True{} : Bool}:
  match live:
    case False{}: {==}
    case True{}: yes({==})

def slash(+b: Chess.Board,+f: U32,+r: U32,+good: {S.state(b,f,r) == True{} : Bool}) ->
  {S.safe(Dispatch.slash(P.Layout{b,f,r,True{}})) == True{} : Bool}:
  guarded(b,0,U32.sub(r,1),Bool.and(U32.is_eq(f,8),U32.is_gt(r,0)),
    fit => slash_cert(b,f,r,good,Facts.slash(f,r,bounded(b,f,r,good),fit)))

def digit(+d: Bool,+c: U32,+b: Chess.Board,+f: U32,+r: U32,
  +same: {d == S.digit(c) : Bool},+good: {S.state(b,f,r) == True{} : Bool}) ->
  {S.safe(P.digit(d,c,P.Layout{b,f,r,True{}})) == True{} : Bool}:
  match d:
    case True{}:
      guarded(b,U32.add(f,U32.sub(c,48)),r,U32.is_le(U32.add(f,U32.sub(c,48)),8),
        fit => digit_cert(b,f,r,c,good,Facts.digit(c,f,r,bounded(b,f,r,good),Equal.sym(Bool,True{},S.digit(c),same),fit)))
    case False{}:
      guarded(P.put(P.kind(U32.or(c,32)),U32.is_lt(c,97),S.square(f,r),b),U32.inc(f),r,
        Bool.and(U32.is_lt(P.kind(U32.or(c,32)),6),U32.is_lt(f,8)),
        +fit => raw_piece(P.kind(U32.or(c,32)),b,f,r,U32.is_lt(c,97),
          L.left(U32.is_lt(P.kind(U32.or(c,32)),6),U32.is_lt(f,8),fit),good,
          L.right(U32.is_lt(P.kind(U32.or(c,32)),6),U32.is_lt(f,8),fit)))

def dead(+l: P.Layout,e: {PS.live(l) == False{} : Bool}) -> {S.safe(l) == True{} : Bool}:
  match l:
    case P.Layout{b,f,r,+live}:
      Equal.cong(Bool,Bool,v => Bool.pick(Bool,v,S.state(b,f,r),True{}),live,False{},e)

def one(+c: U32,+l: P.Layout,+good: {S.safe(l) == True{} : Bool}) ->
  {S.safe(P.layout_char(c,l)) == True{} : Bool}:
  match l:
    case P.Layout{+b,+f,+r,True{}}:
      Dispatch.cases(c,P.Layout{b,f,r,True{}},x => {S.safe(x) == True{} : Bool},
        slash(b,f,r,good),digit(S.digit(c),c,b,f,r,{==},good))
    case P.Layout{b,f,r,False{}}:
      dead(P.layout_char(c,P.Layout{b,f,r,False{}}),Parent.char_invalid(c,P.Layout{b,f,r,False{}},{==}))
