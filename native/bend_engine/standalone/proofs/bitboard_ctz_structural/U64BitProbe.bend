import Base
import ../layout/Domain.bend as Domain

def word_singleton(+n: Nat,+i: Nat) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match i:
        case 0n: WCon{True{},Word.zero(p)}
        case 1n+q: WCon{False{},word_singleton(p,q)}

def bit_word_exact(+i: Nat,e: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.to_word(U64.bit(i)) == word_singleton(64n,i) : Word(64n)}:
  match i:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: Domain.impossible({U64.to_word(U64.bit(64n+p)) == word_singleton(64n,64n+p) : Word(64n)},p,e)
