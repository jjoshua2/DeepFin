import Base
import ./LAWS.bend as Laws
import ./Spec.bend as Spec
import ./Clear.bend as Clear
import ./Frame.bend as Frame
import ./Certified.bend as Certified
import ./Nats.bend as N
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../layout/Spec.bend as Layout
import ../layout/Low.bend as Low
import ../prefix/Nats.bend as P
import ../relative/Spec.bend as Relative
import ../relative/LAWS.bend as RelativeLaws
import ../relative/PROOF.bend as RelativeProof
import ../relative/Bounds.bend as Bounds
import ../relative/Order.bend as Order

# No caller-supplied clear-path or expected-value equality.
def Laws.interval_clear(t,n,at,end,q,shape,eb,qb,room,outside):
  Clear.certificate(t,n,at,end,q,shape,eb,qb,room,outside)

def Laws.interval_fill_preserves_query(a,n,at,end,sq,bishop,mask,subset,q,shape,eb,qb,room,outside):
  Frame.preserve(a,n,at,end,sq,bishop,mask,subset,q,shape,eb,qb,room,outside)

def Laws.chess_fill_clear(t,k,q,shape,valid,qb,outside):
  Clear.certificate(t,U32.to_nat(Layout.size(U32.from_nat(k))),Prefix.prefix(k),Prefix.prefix(1n+k),q,
    shape,Certified.endpoint(k,valid),qb,Certified.room(k,valid),outside)

def Laws.chess_fill_preserves_query(a,k,q,shape,valid,qb,outside):
  Frame.preserve(a,U32.to_nat(Layout.size(U32.from_nat(k))),Prefix.prefix(k),Prefix.prefix(1n+k),
    U32.mod(U32.from_nat(k),64),U32.is_ge(U32.from_nat(k),64),Layout.mask(U32.from_nat(k)),U64.zero(),q,
    shape,Certified.endpoint(k,valid),qb,Certified.room(k,valid),outside)

def from_natural_le(+a: U32,+b: U32,e: {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}) ->
  {U32.is_le(a,b) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,U32.is_le(U32{x},U32{y}),Nat.is_le(Word.to_nat(32n,x),Word.to_nat(32n,y)),True{},
        Equal.cong(Cmp,Bool,c=>Cmp.is_le(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y)),e)

def later(+i: Nat,+j: Nat,+occ: U64,+before: {Nat.is_lt(i,j) == True{} : Bool},
  +valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {Spec.outside(Relative.address(j,Relative.index(j,occ)),Prefix.prefix(i),Prefix.prefix(1n+i)) == True{} : Bool}:
  +iv = Low.le_lt(i,j,128n,P.lt_to_le(i,j,before),valid)
  range = RelativeLaws.lookup_address_bounds(j,occ,valid)
  +end = Bounds.endpoint(Prefix.prefix(i),Layout.size(U32.from_nat(i)),Prefix.prefix(1n+i),PrefixLaws.step_no_overflow(i,iv))
  +ordered = Equal.trans(Bool,U32.is_le(Prefix.prefix(1n+i),Prefix.prefix(j)),
    U32.is_le(U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(j)),True{},
    Equal.cong(U32,Bool,x=>U32.is_le(x,Prefix.prefix(j)),Prefix.prefix(1n+i),U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),
      Equal.sym(U32,U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(1n+i),end)),PrefixLaws.blocks_ordered(i,j,before,valid))
  Clear.or_right(U32.is_lt(Relative.address(j,Relative.index(j,occ)),Prefix.prefix(i)),
    U32.is_le(Prefix.prefix(1n+i),Relative.address(j,Relative.index(j,occ))),
    from_natural_le(Prefix.prefix(1n+i),Relative.address(j,Relative.index(j,occ)),
      N.le_trans(U32.to_nat(Prefix.prefix(1n+i)),U32.to_nat(Prefix.prefix(j)),U32.to_nat(Relative.address(j,Relative.index(j,occ))),
        N.to_le(Prefix.prefix(1n+i),Prefix.prefix(j),ordered),
        N.to_le(Prefix.prefix(j),Relative.address(j,Relative.index(j,occ)),Bounds.lower(j,Relative.index(j,occ),range)))))

def Laws.fill_preserves_later_lookup(a,i,j,occ,shape,before,valid):
  Laws.chess_fill_preserves_query(a,i,Relative.address(j,Relative.index(j,occ)),shape,
    Low.le_lt(i,j,128n,P.lt_to_le(i,j,before),valid),
    Bounds.allocation(j,Relative.index(j,occ),RelativeLaws.lookup_address_bounds(j,occ,valid)),later(i,j,occ,before,valid))

def prefix_lower(+k: Nat,p: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool} & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  (a,b)=p
  a

def Laws.fill_preserves_reserved_query(a,k,q,shape,valid,reserved):
  Laws.chess_fill_preserves_query(a,k,q,shape,valid,Order.lt_lt(q,512,131072,reserved,{==}),
    Clear.or_left(U32.is_lt(q,Prefix.prefix(k)),U32.is_le(Prefix.prefix(1n+k),q),
      Order.lt_le(q,512,Prefix.prefix(k),reserved,prefix_lower(k,PrefixLaws.prefix_in_allocation(k,P.weaken(k,128n,valid))))))
