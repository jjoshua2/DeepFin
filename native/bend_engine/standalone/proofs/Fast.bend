# Proof-only reflection evaluator. Its agreement with public U64.pext is proved
# below; neither this implementation nor its proofs enter the engine runtime.
import Base
import ./u64/Words.bend as U64Words

def selected(n: Nat, a: Word(n), mask: Word(n)) -> List<&2, Bool>:
  match n:
    case 0n: Nil{}
    case 1n+p:
      match a mask:
        case WCon{b, t} WCon{False{}, m}: selected(p, t, m)
        case WCon{b, t} WCon{True{}, m}: Con{b, selected(p, t, m)}

def padded(n: Nat, bs: List<&2, Bool>) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match bs:
        case Nil{}: WCon{False{}, padded(p, Nil{})}
        case Con{b, t}: WCon{b, padded(p, t)}

def fits(n: Nat, bs: List<&2, Bool>) -> Bool:
  match n:
    case 0n:
      match bs:
        case Nil{}: True{}
        case Con{b, t}: False{}
    case 1n+p:
      match bs:
        case Nil{}: True{}
        case Con{b, t}: fits(p, t)

def fits_nil(n: Nat) -> {fits(n, Nil{}) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: {==}

def fits_succ(+n: Nat, +bs: List<&2, Bool>, e: {fits(n, bs) == True{} : Bool}) ->
  {fits(1n+n, bs) == True{} : Bool}:
  match n:
    case 0n:
      match bs:
        case Nil{}: {==}
        case Con{b, t}: U64Words.false_true({fits(1n, Con{b, t}) == True{} : Bool}, e)
    case 1n+p:
      match bs:
        case Nil{}: {==}
        case Con{b, t}: fits_succ(p, t, e)

def selected_fits(+n: Nat, +a: Word(n), +mask: Word(n)) ->
  {fits(n, selected(n, a, mask)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask:
        case WCon{b, t} WCon{False{}, m}:
          fits_succ(p, selected(p, t, m), selected_fits(p, t, m))
        case WCon{b, t} WCon{True{}, m}: selected_fits(p, t, m)

def padded_empty(n: Nat) -> {padded(n, Nil{}) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      %padded_empty(p) : {WCon{False{}, padded(p, Nil{})} == WCon{False{}, _} : Word(1n+p)}
      {==}

def padded_succ(+n: Nat, +bs: List<&2, Bool>, e: {fits(n, bs) == True{} : Bool}) ->
  {Word.shr.pad(n, padded(n, bs)) == padded(1n+n, bs) : Word(1n+n)}:
  match n:
    case 0n:
      match bs:
        case Nil{}: {==}
        case Con{b, t}:
          U64Words.false_true({Word.shr.pad(0n, padded(0n, Con{b, t})) == padded(1n, Con{b, t}) : Word(1n)}, e)
    case 1n+p:
      match bs:
        case Nil{}:
          %padded_succ(p, Nil{}, fits_nil(p)) : {WCon{False{}, Word.shr.pad(p, padded(p, Nil{}))} == WCon{False{}, _} : Word(2n+p)}
          {==}
        case Con{b, t}:
          %padded_succ(p, t, e) : {WCon{b, Word.shr.pad(p, padded(p, t))} == WCon{b, _} : Word(2n+p)}
          {==}

def word_agrees(+n: Nat, +a: Word(n), +mask: Word(n)) ->
  {Word.pext(n, a, mask) == padded(n, selected(n, a, mask)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask:
        case WCon{b, t} WCon{False{}, m}:
          %Equal.sym(Word(p), Word.pext(p, t, m), padded(p, selected(p, t, m)), word_agrees(p, t, m)) :
            {Word.shr.pad(p, _) == padded(1n+p, selected(p, t, m)) : Word(1n+p)}
          padded_succ(p, selected(p, t, m), selected_fits(p, t, m))
        case WCon{b, t} WCon{True{}, m}:
          %word_agrees(p, t, m) : {WCon{b, Word.pext(p, t, m)} == WCon{b, _} : Word(1n+p)}
          {==}

def pext(a: U64, mask: U64) -> U64:
  U64.from_word(padded(64n, selected(64n, U64.to_word(a), U64.to_word(mask))))

def u64_agrees(+a: U64, +mask: U64) -> {U64.pext(a, mask) == pext(a, mask) : U64}:
  %word_agrees(64n, U64.to_word(a), U64.to_word(mask)) : {U64.pext(a, mask) == U64.from_word(_) : U64}
  {==}
