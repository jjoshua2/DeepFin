import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as B
import ./Domain.bend as D
import ../ray/Spec.bend as S
import ../../../legal_probe/Chess.bend as Chess

# Uses the engine's U32 key directly; neither the table nor its recipe depends on key.
def same_table_for_every_key(+d: Nat,+seed: U64,+n: Nat,+key: U32,+extra: Nat,+start: Nat,+occ: U64,
  valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},depth: {d == 17n : Nat},
  full: {n == 128n : Nat},eb: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(B.run(d,seed,n,extra,start),key,occ) ==
    (B.run(d,seed,n,extra,start),S.rays(U32.to_nat(key),occ)) : Array<U64> & U64}:
  Laws.u32_full_table_lookup(d,seed,n,key,extra,start,occ,valid,depth,full,eb)

# Full public defaults satisfy the small scalar premises, without expanding a table.
def public_depth() -> {17n == 17n : Nat}: {==}
def public_count() -> {128n == 128n : Nat}: {==}
def public_extras() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: D.full_extras()
def first_key() -> {Nat.is_lt(U32.to_nat(0),128n) == True{} : Bool}: {==}
def last_key() -> {Nat.is_lt(U32.to_nat(127),128n) == True{} : Bool}: {==}
def excluded_key() -> {Nat.is_lt(U32.to_nat(128),128n) == False{} : Bool}: {==}
