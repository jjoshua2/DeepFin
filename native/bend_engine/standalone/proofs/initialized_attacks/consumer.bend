import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Build.bend as B
import ./Spec.bend as S
import ../board/Spec.bend as P
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def all_pieces(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+kind: P.Piece,+q: Nat,+by: Bool,+occ: U64,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(P.tag(kind),B.run(d,seed,n,extra),U32.from_nat(q),Bool.to_u32(by),occ) ==
    (B.run(d,seed,n,extra),S.mask(kind,q,by,occ)) : Array<U64> & U64}:
  Laws.initialized_piece_attack_matches_geometry(d,seed,n,extra,kind,q,by,occ,depth,tables,full,bound)

def all_boards(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+by: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attacked(B.run(d,seed,n,extra),b,U32.from_nat(q),Bool.to_u32(by)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,by)) : Array<U64> & Bool}:
  Laws.initialized_attacked_matches_geometry(d,seed,n,extra,b,q,by,depth,tables,full,bound)

def singleton_check(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(side))) == U64.bit(q) : U64}) ->
  {Chess.in_check(B.run(d,seed,n,extra),b,Bool.to_u32(side)) ==
    (B.run(d,seed,n,extra),S.attacked(b,q,Bool.not(side))) : Array<U64> & Bool}:
  Laws.initialized_single_king_check_matches_geometry(d,seed,n,extra,b,q,side,depth,tables,full,bound,one)

# The declared numeric domain is inhabited without expanding the huge literal table.
def configuration() -> {17n == 17n : Nat} & ({128n == 128n : Nat} & {64n == 64n : Nat}):
  ({==},({==},{==}))

# Independent geometry answers remain sensitive to color, blockers and direction.
def black_rook() -> Chess.Board:
  Position.put(3,False{},0,Position.put(5,True{},7,Position.empty()))

def rook_seen() -> {S.attacked(black_rook(),7n,False{}) == True{} : Bool}:
  {==}

def wrong_color_absent() -> {S.attacked(black_rook(),7n,True{}) == False{} : Bool}:
  {==}

def intermediate_blocker() -> {S.attacked(Position.put(1,True{},3,black_rook()),7n,False{}) == False{} : Bool}:
  {==}

def diagonal_pawn() -> {S.attacked(Position.put(0,False{},35,Position.empty()),28n,False{}) == True{} : Bool}:
  {==}

def no_king_is_not_singleton() -> {U64.is_zero(U64.and(Chess.get_kings(Position.empty()),Chess.color(Position.empty(),1))) == True{} : Bool}:
  {==}
