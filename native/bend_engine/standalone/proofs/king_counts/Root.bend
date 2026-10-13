import Base
import ./Actual.bend as K
import ./Spec.bend as S
import ./consumer.bend as Counts
import ./Initialized.bend as Complete
import ../validated_root_kings/Kings.bend as Prior
import ../backrank_pawns/Root.bend as Backrank
import ../color_counts/Root.bend as Colors
import ../pawn_counts/Root.bend as Pawns
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../board/Spec.bend as Board
import ../ray/Frame.bend as F
import ../fen_validated_full/Root.bend as Value
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def one(-b: Chess.Board) -> Type:
  {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))) == 1 : U32} &
    {U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))) == 1 : U32}
def output(maybe: Maybe<&2,Pos.Game>) -> Type:
  match maybe:
    case None{}: Unit
    case Some{g}: one(Pos.board(g))
def coherent(+b: Chess.Board,good: {Pos.valid(b) == True{} : Bool}) -> {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool}:
  F.left(Pos.valid_ep(Chess.get_ep(b),b),Bool.and(Pos.valid_right(b,1,7,4,1),Bool.and(Pos.valid_right(b,2,0,4,1),Bool.and(Pos.valid_right(b,4,63,60,0),Pos.valid_right(b,8,56,60,0)))),
    F.right(Pos.structural(b),Prior.metadata_valid(b),good))
def counts_one(+w: Nat,+bl: Nat,maybe: Maybe<&2,Pos.Game>,pw: {U32.from_nat(w) == 1 : U32},pb: {U32.from_nat(bl) == 1 : U32},
  good: S.state(w,bl,maybe)) -> output(maybe):
  match maybe:
    case None{}: Unit{}
    case Some{g}:
      (metadata,(ep,(zero,(cw,cb))))=good
      (Equal.trans(U32,U64.popcount(U64.and(Chess.get_kings(Pos.board(g)),Chess.get_white(Pos.board(g)))),U32.from_nat(w),1,
         Equal.cong(Nat,U32,U32.from_nat,K.white(Pos.board(g)),w,cw),pw),
       Equal.trans(U32,U64.popcount(U64.and(Chess.get_kings(Pos.board(g)),Chess.get_black(Pos.board(g)))),U32.from_nat(bl),1,
         Equal.cong(Nat,U32,U32.from_nat,K.black(Pos.board(g)),bl,cb),pb))
def result_one(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+w: Nat,+bl: Nat,r: Array<U64> & Maybe<&2,Pos.Game>,
  initial: {U32.from_nat(w) == 1 : U32} & {U32.from_nat(bl) == 1 : U32},checked: S.result(a,d,seed,n,extra,w,bl,r)) -> output(Value.value(r)):
  (pw,pb)=initial
  (same,(known,(depth,(tables,(full,good)))))=checked
  counts_one(w,bl,Value.value(r),pw,pb,good)
def complete(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,-g: Pos.Game,-text: String,-tail: List<&2,String>) -> Type:
  Complete.complete(a,d,seed,n,extra,Some{g},text,tail) & output(Value.value(Protocol.moves(Con{text,tail},(a,Some{g}))))
def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+g: Pos.Game,+text: String,+tail: List<&2,String>,
  +rows: {Board.valid(Pos.board(g)) == True{} : Bool},+canonical: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(U32.is_eq(Chess.get_turn(Pos.board(g)),1)) : U32},
  +valid: {Pos.valid(Pos.board(g)) == True{} : Bool},+zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},+depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> complete(a,d,seed,n,extra,g,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : complete(_,d,seed,n,extra,g,text,tail)
  (Complete.use_replay(O.pack(c),d,seed,n,extra,Some{g},text,tail,(rows,canonical),coherent(Pos.board(g),valid),zero,
     Backrank.valid(Pos.board(g),valid),Colors.valid(Pos.board(g),valid),Pawns.valid(Pos.board(g),valid),known,depth,tables,full),
   result_one(O.pack(c),d,seed,n,extra,K.white(Pos.board(g)),K.black(Pos.board(g)),Protocol.moves(Con{text,tail},(O.pack(c),Some{g})),
     Prior.king_counts(Pos.board(g),Prior.structural(Pos.board(g),valid)),
     Counts.use_replay(O.pack(c),d,seed,n,extra,Some{g},Con{text,tail},(rows,canonical),coherent(Pos.board(g),valid),zero,known,depth,tables,full)))
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+g: Pos.Game,+text: String,+tail: List<&2,String>,
  rows: {Board.valid(Pos.board(g)) == True{} : Bool},canonical: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(U32.is_eq(Chess.get_turn(Pos.board(g)),1)) : U32},
  valid: {Pos.valid(Pos.board(g)) == True{} : Bool},zero: {Chess.get_rights(Pos.board(g)) == 0 : U32},
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) -> complete(a,d,seed,n,extra,g,text,tail):
  lift(a,d,seed,n,extra,g,text,tail,rows,canonical,valid,zero,built,depth,tables,full,R.reify(a))
