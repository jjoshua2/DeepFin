# Actual full-generator membership implies independent destination geometry is clear.
import Base
import ./Spec.bend as S
import ./Generated.bend as Generated
import ./Destination.bend as D
import ./Finish.bend as Finish
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_kings/Spec.bend as K
import ../castle_stage_geometry/PROOF.bend as StageProof
import ../castle_stage_geometry/LAWS.bend as Stage
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess

def side(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,+m: Chess.Ply,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},part: C.side(b,ks,m),
  no: {S.answer(Chess.in_check(Init.run(d,seed,n,extra),Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}) ->
  {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool}:
  (guard,e) = part
  +same: {m == D.move(white,ks) : Chess.Ply} =
    Equal.trans(Chess.Ply,m,E.move(b,ks),D.move(white,ks),e,D.route(b,white,ks,turn))
  %Equal.sym(Chess.Ply,m,D.move(white,ks),same) :
    {G.attacked(Chess.make_move(b,_),U32.to_nat(D.square(_)),Bool.not(white)) == False{} : Bool}
  Finish.stage(Init.run(d,seed,n,extra),b,white,ks,
    Stage.initialized_castling_stage_check_matches_geometry(d,seed,n,extra,b,white,ks,K.Completed{},
      depth,tables,full,good,turn,one,guard),
    D.decide(Init.run(d,seed,n,extra),b,white,ks,m,same,turn,no))

def selected(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+m: Chess.Ply,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},part: C.castle(b,m),
  no: {S.answer(Chess.in_check(Init.run(d,seed,n,extra),Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}) ->
  {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool}:
  match part:
    case Inl{k}: side(d,seed,n,extra,b,white,True{},m,depth,tables,full,good,turn,one,k,no)
    case Inr{q}: side(d,seed,n,extra,b,white,False{},m,depth,tables,full,good,turn,one,q,no)

def actual(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+m: Chess.Ply,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),+tag: {C.flag(m) == 2 : U32}) ->
  {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool}:
  Generated.split_member(
    {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool},
    E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),m,here,origin => accepted =>
      selected(d,seed,n,extra,b,white,m,depth,tables,full,good,turn,one,
        Chain.legal_member(Init.run(d,seed,n,extra),b,m,origin,tag),
        Generated.actual(Init.run(d,seed,n,extra),b,m,accepted,tag)))
