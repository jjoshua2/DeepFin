import Base
import ../ray/Algebra.bend as R
import ../candidate_king_destination/Algebra.bend as K
import ../occupancy_insertion/Lattice.bend as L
import ../ray_clear_disjoint/Algebra.bend as D

def or_comm_bool(+a: Bool,+b: Bool) -> {Bool.or(a,b) == Bool.or(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def or_comm_word(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {Word.or(n,a,b) == Word.or(n,b,a) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          Equal.trans(Word(1n+p),WCon{Bool.or(ah,bh),Word.or(p,at,bt)},
            WCon{Bool.or(bh,ah),Word.or(p,at,bt)},WCon{Bool.or(bh,ah),Word.or(p,bt,at)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,at,bt)},Bool.or(ah,bh),Bool.or(bh,ah),or_comm_bool(ah,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(bh,ah),t},Word.or(p,at,bt),Word.or(p,bt,at),or_comm_word(p,at,bt)))
def or_comm(+a: U64,+b: U64) -> {U64.or(a,b) == U64.or(b,a) : U64}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      Equal.trans(U64,U64{U32{Word.or(32n,al,bl)},U32{Word.or(32n,ah,bh)}},
        U64{U32{Word.or(32n,bl,al)},U32{Word.or(32n,ah,bh)}},U64{U32{Word.or(32n,bl,al)},U32{Word.or(32n,bh,ah)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,ah,bh)}},Word.or(32n,al,bl),Word.or(32n,bl,al),or_comm_word(32n,al,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,bl,al)},U32{h}},Word.or(32n,ah,bh),Word.or(32n,bh,ah),or_comm_word(32n,ah,bh)))

def trans(+a: U64,+b: U64,+c: U64,ab: L.subset(a,b),bc: L.subset(b,c)) -> L.subset(a,c):
  +same: {U64.and(b,a) == a : U64} = ab
  Equal.trans(U64,U64.and(c,a),U64.and(c,U64.and(b,a)),a,
    Equal.cong(U64,U64,x => U64.and(c,x),a,U64.and(b,a),Equal.sym(U64,U64.and(b,a),a,same)),
    Equal.trans(U64,U64.and(c,U64.and(b,a)),U64.and(U64.and(c,b),a),a,
      Equal.sym(U64,U64.and(U64.and(c,b),a),U64.and(c,U64.and(b,a)),R.and_assoc(c,b,a)),
      Equal.trans(U64,U64.and(U64.and(c,b),a),U64.and(b,a),a,
        Equal.cong(U64,U64,x => U64.and(x,a),U64.and(c,b),b,bc),same)))

def or_mono(+a: U64,+b: U64,+c: U64,+d: U64,ab: L.subset(a,b),cd: L.subset(c,d)) ->
  L.subset(U64.or(a,c),U64.or(b,d)):
  trans(U64.or(a,c),U64.or(b,c),U64.or(b,d),
    L.rewrite(U64.or(c,a),U64.or(c,b),U64.or(a,c),U64.or(b,c),or_comm(a,c),or_comm(b,c),L.or_subset(c,a,b,ab)),
    L.or_subset(b,c,d,cd))

def shuffle(+a: U64,+b: U64,+c: U64,+d: U64) ->
  {U64.and(U64.and(a,b),U64.and(c,d)) == U64.and(U64.and(a,c),U64.and(b,d)) : U64}:
  Equal.trans(U64,U64.and(U64.and(a,b),U64.and(c,d)),U64.and(a,U64.and(b,U64.and(c,d))),U64.and(U64.and(a,c),U64.and(b,d)),
    R.and_assoc(a,b,U64.and(c,d)),
    Equal.trans(U64,U64.and(a,U64.and(b,U64.and(c,d))),U64.and(a,U64.and(U64.and(b,c),d)),U64.and(U64.and(a,c),U64.and(b,d)),
      Equal.cong(U64,U64,x => U64.and(a,x),U64.and(b,U64.and(c,d)),U64.and(U64.and(b,c),d),
        Equal.sym(U64,U64.and(U64.and(b,c),d),U64.and(b,U64.and(c,d)),R.and_assoc(b,c,d))),
      Equal.trans(U64,U64.and(a,U64.and(U64.and(b,c),d)),U64.and(a,U64.and(U64.and(c,b),d)),U64.and(U64.and(a,c),U64.and(b,d)),
        Equal.cong(U64,U64,x => U64.and(a,U64.and(x,d)),U64.and(b,c),U64.and(c,b),K.and_commute(b,c)),
        Equal.trans(U64,U64.and(a,U64.and(U64.and(c,b),d)),U64.and(a,U64.and(c,U64.and(b,d))),U64.and(U64.and(a,c),U64.and(b,d)),
          Equal.cong(U64,U64,x => U64.and(a,x),U64.and(U64.and(c,b),d),U64.and(c,U64.and(b,d)),R.and_assoc(c,b,d)),
          Equal.sym(U64,U64.and(U64.and(a,c),U64.and(b,d)),U64.and(a,U64.and(c,U64.and(b,d))),R.and_assoc(a,c,U64.and(b,d)))))))

def and_mono(+a: U64,+b: U64,+c: U64,+d: U64,ab: L.subset(a,b),cd: L.subset(c,d)) ->
  L.subset(U64.and(a,c),U64.and(b,d)):
  Equal.trans(U64,U64.and(U64.and(b,d),U64.and(a,c)),U64.and(U64.and(b,a),U64.and(d,c)),U64.and(a,c),
    shuffle(b,d,a,c),
    Equal.trans(U64,U64.and(U64.and(b,a),U64.and(d,c)),U64.and(a,U64.and(d,c)),U64.and(a,c),
      Equal.cong(U64,U64,x => U64.and(x,U64.and(d,c)),U64.and(b,a),a,ab),
      Equal.cong(U64,U64,x => U64.and(a,x),U64.and(d,c),c,cd)))

def zero_subset(+small: U64,+big: U64,included: L.subset(small,big),empty: {big == U64.zero() : U64}) ->
  {small == U64.zero() : U64}:
  Equal.trans(U64,small,U64.and(big,small),U64.zero(),Equal.sym(U64,U64.and(big,small),small,included),
    Equal.trans(U64,U64.and(big,small),U64.and(U64.zero(),small),U64.zero(),
      Equal.cong(U64,U64,x => U64.and(x,small),big,U64.zero(),empty),D.left_zero(small)))

def transfer(+old_mask: U64,+new_mask: U64,+old_enemy: U64,+new_enemy: U64,
  mask: L.subset(new_mask,old_mask),enemy: L.subset(new_enemy,old_enemy),
  clear: {U64.and(old_mask,old_enemy) == U64.zero() : U64}) ->
  {U64.and(new_mask,new_enemy) == U64.zero() : U64}:
  zero_subset(U64.and(new_mask,new_enemy),U64.and(old_mask,old_enemy),and_mono(new_mask,old_mask,new_enemy,old_enemy,mask,enemy),clear)
