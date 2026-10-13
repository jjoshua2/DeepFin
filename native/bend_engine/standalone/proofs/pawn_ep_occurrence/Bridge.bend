# Legal-list membership supplies the PR1044 target-mask premise and source bound.
import Base
import ./Spec.bend as S
import ./Pipeline.bend as Pipe
import ../child_metadata/Range.bend as R
import ../initialized_pawn_double/Spec.bend as D
import ../initialized_pawn_double/Producer.bend as Double
import ../attack_geometry/Initialized.bend as I
import ../bitboard_membership/AllEmittedSound.bend as All
import ../castle_emission/Spec.bend as E
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as C

def kind(pawn: Bool,knight: Bool,bishop: Bool,rook: Bool,queen: Bool) -> U32:
  Bool.pick(U32,pawn,0,Bool.pick(U32,knight,1,Bool.pick(U32,bishop,2,Bool.pick(U32,rook,3,Bool.pick(U32,queen,4,5)))))

def kind_zero(pawn: Bool,knight: Bool,bishop: Bool,rook: Bool,queen: Bool) ->
  {U32.is_zero(kind(pawn,knight,bishop,rook,queen)) == pawn : Bool}:
  match pawn:
    case True{}: {==}
    case False{}:
      match knight:
        case True{}: {==}
        case False{}:
          match bishop:
            case True{}: {==}
            case False{}:
              match rook:
                case True{}: {==}
                case False{}:
                  match queen:
                    case True{}: {==}
                    case False{}: {==}

def actual_pawn(+b: C.Board,+s: U32,zero: {U32.is_zero(C.piece(b,s)) == True{} : Bool}) -> {S.pawn(b,s) == True{} : Bool}:
  Equal.trans(Bool,S.pawn(b,s),U32.is_zero(C.piece(b,s)),True{},
    Equal.sym(Bool,U32.is_zero(C.piece(b,s)),S.pawn(b,s),
      kind_zero(S.pawn(b,s),U64.test_bit(C.get_knights(b),U32.to_nat(s)),U64.test_bit(C.get_bishops(b),U32.to_nat(s)),
        U64.test_bit(C.get_rooks(b),U32.to_nat(s)),U64.test_bit(C.get_queens(b),U32.to_nat(s)))),zero)

def ep_trigger(+b: C.Board,m: C.Ply) -> Bool:
  C.Ply{+s,d,p,f}=m
  Bool.and(U32.is_zero(C.piece(b,s)),S.trigger(s,d))

def required(-table: Array<U64>,+b: C.Board,+s: U32,+d: U32,flag: Bool,
  yes: {flag == True{} : Bool},hit: S.gate(flag,table,b,s,d)) ->
  {U64.test_bit(D.mask(C.pawn_targets(table,b,s)),U32.to_nat(d)) == True{} : Bool}:
  match flag:
    case True{}: hit
    case False{}: Old.false_true({U64.test_bit(D.mask(C.pawn_targets(table,b,s)),U32.to_nat(d)) == True{} : Bool},yes)

def fields(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},+trigger: {ep_trigger(b,m) == True{} : Bool},
  origin: S.occurrence(I.run(depth,seed,n,key,at,extra),b,m)) -> D.double(b,R.src(m),S.dst(m)):
  match m:
    case C.Ply{+s,+d,p,f}:
      (bound,hit)=origin
      +kind0=All.bool_and_true_left(U32.is_zero(C.piece(b,s)),S.trigger(s,d),trigger)
      +xor16=All.bool_and_true_right(U32.is_zero(C.piece(b,s)),S.trigger(s,d),trigger)
      +pawn=actual_pawn(b,s,kind0)
      Double.initialized(depth,seed,n,key,at,extra,b,s,d,depth_ok,full,bound,xor16,
        required(I.run(depth,seed,n,key,at,extra),b,s,d,Bool.and(S.pawn(b,s),S.trigger(s,d)),
          All.bool_and_true(S.pawn(b,s),S.trigger(s,d),pawn,xor16),hit))

def initialized(+depth: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+b: C.Board,+m: C.Ply,
  depth_ok: {depth == 17n : Nat},full: {extra == 64n : Nat},trigger: {ep_trigger(b,m) == True{} : Bool},
  here: E.member(m,E.moves(C.legal_moves(I.run(depth,seed,n,key,at,extra),b)))) -> D.double(b,R.src(m),S.dst(m)):
  fields(depth,seed,n,key,at,extra,b,m,depth_ok,full,trigger,
    Pipe.member(I.run(depth,seed,n,key,at,extra),b,m,here))

def midpoint(+m: C.Ply) -> U32:
  U32.div(U32.add(R.src(m),S.dst(m)),2)

def tag(+b: C.Board,+m: C.Ply,trigger: {ep_trigger(b,m) == True{} : Bool}) ->
  {C.get_ep(C.make_move(b,m)) == midpoint(m) : U32}:
  match m:
    case C.Ply{+s,+d,p,f}:
      %Equal.sym(Bool,ep_trigger(b,C.Ply{s,d,p,f}),True{},trigger) :
        {Bool.pick(U32,_,U32.div(U32.add(s,d),2),64) == U32.div(U32.add(s,d),2) : U32}
      {==}

def duplicate_tail(+h: C.Ply,+t: List<&2,C.Ply>,+m: C.Ply,parts: E.member(m,t) & E.member(m,t)) ->
  E.member(m,Con{h,t}) & E.member(m,Con{h,t}):
  (left,right)=parts
  (Inr{left},Inr{right})

# Copy the membership certificate along its original full-Ply occurrence path.
def duplicate(+xs: List<&2,C.Ply>,+m: C.Ply,here: E.member(m,xs)) -> E.member(m,xs) & E.member(m,xs):
  match xs:
    case Nil{}: Empty.absurd(E.member(m,Nil{}) & E.member(m,Nil{}),here)
    case Con{+h,+t}:
      match here:
        case Inl{e}:
          +same: {m == h : C.Ply}=e
          (Inl{same},Inl{same})
        case Inr{later}:
          duplicate_tail(h,t,m,duplicate(t,m,later))
