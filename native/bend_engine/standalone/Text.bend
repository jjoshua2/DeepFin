import Base

def push_word(token: String, acc: List<&2, String>) -> List<&2, String>:
  match token:
    case SNil{}: acc
    case SCon{h, t}: Con{String.reverse(SCon{h, t}), acc}

def split_step(space: Bool, c: Char, token: String, acc: List<&2, String>) -> String & List<&2, String>:
  match space:
    case True{}: ("", push_word(token, acc))
    case False{}: (SCon{c, token}, acc)

def words_go(s: String, r: String & List<&2, String>) -> List<&2, String>:
  match s:
    case SNil{}:
      (token, acc) = r
      List.reverse(&2, String, push_word(token, acc))
    case SCon{+h, t}:
      (token, acc) = r
      words_go(t, split_step(Char.is_space(h), h, token, acc))

def words(s: String) -> List<&2, String>:
  words_go(s, ("", Nil{}))

def number_go(s: String, +n: U32, ok: Bool) -> Maybe<&2, U32>:
  match s:
    case SNil{}:
      match ok:
        case True{}: Some{n}
        case False{}: None{}
    case SCon{Chr{+c}, rest}:
      next_ok = Bool.and(ok, Bool.and(U32.is_ge(c, 48), Bool.and(U32.is_le(c, 57),
        Bool.or(U32.is_lt(n, 429496729), Bool.and(U32.is_eq(n, 429496729), U32.is_le(c, 53))))))
      number_go(rest, U32.add(U32.mul(n, 10), U32.sub(c, 48)), next_ok)

def number(+s: String) -> Maybe<&2, U32>:
  number_go(s, 0, Bool.not(String.is_empty(s)))

def only(valid: Bool, value: U32) -> Maybe<&2, U32>:
  match valid:
    case True{}: Some{value}
    case False{}: None{}

def square(s: String) -> Maybe<&2, U32>:
  match s:
    case SCon{Chr{+file}, SCon{Chr{+rank}, SNil{}}}:
      only(Bool.and(U32.is_ge(file, 97), Bool.and(U32.is_le(file, 104),
        Bool.and(U32.is_ge(rank, 49), U32.is_le(rank, 56)))),
        U32.add(U32.sub(file, 97), U32.mul(U32.sub(rank, 49), 8)))
    case _: None{}

def square_text(+sq: U32) -> String:
  SCon{Chr{U32.add(97, U32.mod(sq, 8))}, SCon{Chr{U32.add(49, U32.div(sq, 8))}, SNil{}}}

def positive(+n: U32) -> Maybe<&2, U32>:
  only(U32.is_gt(n, 0), n)

def bounded(+n: U32) -> Maybe<&2, U32>:
  only(U32.is_le(n, 1000000), n)
