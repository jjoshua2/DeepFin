# EP-setting actual occurrences cannot take the emitted promotion/castle branches.
import Base
import ./Spec.bend as S
import ./Ranks.bend as Ranks
import ./Source.bend as Source
import ../pawn_ep_occurrence/Bridge.bend as Prior
import ../pawn_ep_occurrence/Pipeline.bend as Pipe
import ../pawn_ep_occurrence/Spec.bend as OldSpec
import ../child_metadata/Range.bend as R
import ../initialized_pawn_double/Spec.bend as D
import ../bitboard_membership/AllEmittedSound.bend as All
import ../castle_chain/Spec.bend as Chain
import ../castle_emission/Spec.bend as E
import ../attack_geometry/Initialized.bend as I
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as C

def rank(+b: C.Board,+m: C.Ply,below: R.bounded(m),double: D.double(b,R.src(m),S.dst(m))) -> {S.backrank(S.dst(m)) == False{} : Bool}:
  match m:
    case C.Ply{+s,+d,p,f}:
      (guard,same)=double
      +start=All.bool_and_true_left(D.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.second(s,D.white(b))))),
        All.bool_and_true_right(D.empty1(b,s),Bool.and(D.rank(b,s),Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(D.second(s,D.white(b)))))),guard))
      Equal.trans(Bool,S.backrank(d),S.backrank(D.second(s,D.white(b))),False{},
        Equal.cong(U32,Bool,x => S.backrank(x),d,D.second(s,D.white(b)),same),Ranks.second(s,D.white(b),below,start))

def scanned(+m: C.Ply,no_rank: {S.backrank(S.dst(m)) == False{} : Bool},fields: S.scanned(m)) -> S.ordinary(m):
  match fields:
    case Inl{plain}: Inl{plain}
    case Inr{later}:
      match later:
        case Inl{ep}: Inr{ep}
        case Inr{promoted}:
          (high,shape)=promoted
          Old.false_true(S.ordinary(m),Equal.trans(Bool,False{},S.backrank(S.dst(m)),True{},
            Equal.sym(Bool,S.backrank(S.dst(m)),False{},no_rank),high))

def castle(+b: C.Board,+m: C.Ply,ks: Bool,+xor16: {OldSpec.trigger(R.src(m),S.dst(m)) == True{} : Bool},fields: Chain.side(b,ks,m)) -> S.ordinary(m):
  (guard,same)=fields
  Old.false_true(S.ordinary(m),Equal.trans(Bool,False{},OldSpec.trigger(R.src(m),S.dst(m)),True{},
    Equal.trans(Bool,False{},OldSpec.trigger(R.src(E.move(b,ks)),S.dst(E.move(b,ks))),OldSpec.trigger(R.src(m),S.dst(m)),
      Equal.sym(Bool,OldSpec.trigger(R.src(E.move(b,ks)),S.dst(E.move(b,ks))),False{},Pipe.castle_xor(U32.is_eq(C.get_turn(b),1),ks)),
      Equal.cong(C.Ply,Bool,x => OldSpec.trigger(R.src(x),S.dst(x)),E.move(b,ks),m,Equal.sym(C.Ply,m,E.move(b,ks),same))),xor16))

def castle_choice(+b: C.Board,+m: C.Ply,+xor16: {OldSpec.trigger(R.src(m),S.dst(m)) == True{} : Bool},shape: Chain.castle(b,m)) -> S.ordinary(m):
  match shape:
    case Inl{ks}: castle(b,m,True{},xor16,ks)
    case Inr{qs}: castle(b,m,False{},xor16,qs)

def xor(+b: C.Board,m: C.Ply,trigger: {Prior.ep_trigger(b,m) == True{} : Bool}) -> {OldSpec.trigger(R.src(m),S.dst(m)) == True{} : Bool}:
  match m:
    case C.Ply{+s,d,p,f}: All.bool_and_true_right(U32.is_zero(C.piece(b,s)),OldSpec.trigger(s,d),trigger)

def generated(+b: C.Board,+m: C.Ply,no_rank: {S.backrank(S.dst(m)) == False{} : Bool},+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},fields: S.generated(b,m)) -> S.ordinary(m):
  match fields:
    case Inl{plain}: scanned(m,no_rank,plain)
    case Inr{castle_shape}:
      castle_choice(b,m,xor(b,m,trigger),castle_shape)

def compose_fields(+b: C.Board,+m: C.Ply,+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},
  +guard_eq: {D.empty2(b,R.src(m)) == True{} : Bool},+destination_eq: {S.dst(m) == D.second(R.src(m),D.white(b)) : U32},
  fields: Source.generated(b,m)) -> R.bounded(m) & (D.double(b,R.src(m),S.dst(m)) & S.ordinary(m)):
  (bound,shape)=fields
  +source_bound: {Nat.is_lt(U32.to_nat(R.src(m)),64n) == True{} : Bool}=bound
  (source_bound,((guard_eq,destination_eq),generated(b,m,rank(b,m,source_bound,(guard_eq,destination_eq)),trigger,shape)))

def compose(+b: C.Board,+m: C.Ply,+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},double: D.double(b,R.src(m),S.dst(m)),fields: Source.generated(b,m)) -> R.bounded(m) & (D.double(b,R.src(m),S.dst(m)) & S.ordinary(m)):
  (guard,same)=double
  compose_fields(b,m,trigger,guard,same,fields)

def from_parts(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},
  parts: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b))) & E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> R.bounded(m) & (D.double(b,R.src(m),S.dst(m)) & S.ordinary(m)):
  (for_double,for_fields)=parts
  compose(b,m,trigger,Prior.initialized(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,for_double),
    Source.member(I.run(depth,seed,n,key,at,extra),b,m,for_fields))

def initialized(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},+trigger: {Prior.ep_trigger(b,m) == True{} : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> R.bounded(m) & (D.double(b,R.src(m),S.dst(m)) & S.ordinary(m)):
  from_parts(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,
    Prior.duplicate(E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)),m,here))
