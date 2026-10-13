import Base

def bool_comm(+a: Bool,+b: Bool) -> {Bool.and(a,b) == Bool.and(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def bool_assoc(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.and(a,b),c) == Bool.and(a,Bool.and(b,c)) : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}
def bool_absorb(+a: Bool,+b: Bool) -> {Bool.and(Bool.and(a,b),a) == Bool.and(a,b) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def bool_disjoint(+a: Bool,+b: Bool) ->
  {Bool.and(Bool.and(a,Bool.not(b)),b) == False{} : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def bool_union(+a: Bool,+b: Bool) -> {Bool.and(Bool.or(a,b),a) == a : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def word_comm(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,a,b) == Word.and(n,b,a) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          %bool_comm(ah,bh) :
            {WCon{Bool.and(ah,bh),Word.and(p,at,bt)} == WCon{_,Word.and(p,bt,at)} : Word(1n+p)}
          %word_comm(p,at,bt) :
            {WCon{Bool.and(ah,bh),Word.and(p,at,bt)} == WCon{Bool.and(ah,bh),_} : Word(1n+p)}
          {==}

def word_assoc(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) -> {Word.and(n,Word.and(n,a,b),c) == Word.and(n,a,Word.and(n,b,c)) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          %bool_assoc(ah,bh,ch) :
            {WCon{Bool.and(Bool.and(ah,bh),ch),Word.and(p,Word.and(p,at,bt),ct)} == WCon{_,Word.and(p,at,Word.and(p,bt,ct))} : Word(1n+p)}
          %word_assoc(p,at,bt,ct) :
            {WCon{Bool.and(Bool.and(ah,bh),ch),Word.and(p,Word.and(p,at,bt),ct)} == WCon{Bool.and(Bool.and(ah,bh),ch),_} : Word(1n+p)}
          {==}

def word_absorb(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,Word.and(n,a,b),a) == Word.and(n,a,b) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          %bool_absorb(ah,bh) :
            {WCon{Bool.and(Bool.and(ah,bh),ah),Word.and(p,Word.and(p,at,bt),at)} == WCon{_,Word.and(p,at,bt)} : Word(1n+p)}
          %word_absorb(p,at,bt) :
            {WCon{Bool.and(Bool.and(ah,bh),ah),Word.and(p,Word.and(p,at,bt),at)} == WCon{Bool.and(Bool.and(ah,bh),ah),_} : Word(1n+p)}
          {==}

def word_disjoint(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,Word.and(n,a,Word.not(n,b)),b) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          %bool_disjoint(ah,bh) :
            {WCon{Bool.and(Bool.and(ah,Bool.not(bh)),bh),Word.and(p,Word.and(p,at,Word.not(p,bt)),bt)} == WCon{_,Word.zero(p)} : Word(1n+p)}
          %word_disjoint(p,at,bt) :
            {WCon{Bool.and(Bool.and(ah,Bool.not(bh)),bh),Word.and(p,Word.and(p,at,Word.not(p,bt)),bt)} == WCon{Bool.and(Bool.and(ah,Bool.not(bh)),bh),_} : Word(1n+p)}
          {==}

def word_union(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,Word.or(n,a,b),a) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          %bool_union(ah,bh) :
            {WCon{Bool.and(Bool.or(ah,bh),ah),Word.and(p,Word.or(p,at,bt),at)} == WCon{_,at} : Word(1n+p)}
          %word_union(p,at,bt) :
            {WCon{Bool.and(Bool.or(ah,bh),ah),Word.and(p,Word.or(p,at,bt),at)} == WCon{Bool.and(Bool.or(ah,bh),ah),_} : Word(1n+p)}
          {==}

def word_zero(+n: Nat,+a: Word(n)) -> {Word.and(n,Word.zero(n),a) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a:
        case WCon{h,t}:
          %word_zero(p,t) : {WCon{False{},Word.and(p,Word.zero(p),t)} == WCon{False{},_} : Word(1n+p)}
          {==}

def comm(+a: U32,+b: U32) -> {U32.and(a,b) == U32.and(b,a) : U32}:
  match a b:
    case U32{aw} U32{bw}:
      %word_comm(32n,aw,bw) : {U32{Word.and(32n,aw,bw)} == U32{_} : U32}
      {==}

def assoc(+a: U32,+b: U32,+c: U32) -> {U32.and(U32.and(a,b),c) == U32.and(a,U32.and(b,c)) : U32}:
  match a b c:
    case U32{aw} U32{bw} U32{cw}:
      %word_assoc(32n,aw,bw,cw) : {U32{Word.and(32n,Word.and(32n,aw,bw),cw)} == U32{_} : U32}
      {==}

def absorb(+a: U32,+b: U32) -> {U32.and(U32.and(a,b),a) == U32.and(a,b) : U32}:
  match a b:
    case U32{aw} U32{bw}:
      %word_absorb(32n,aw,bw) : {U32{Word.and(32n,Word.and(32n,aw,bw),aw)} == U32{_} : U32}
      {==}

def disjoint(+a: U32,+b: U32) -> {U32.and(U32.and(a,U32.not(b)),b) == 0 : U32}:
  match a b:
    case U32{aw} U32{bw}:
      %word_disjoint(32n,aw,bw) : {U32{Word.and(32n,Word.and(32n,aw,Word.not(32n,bw)),bw)} == U32{_} : U32}
      {==}

def union(+a: U32,+b: U32) -> {U32.and(U32.or(a,b),a) == a : U32}:
  match a b:
    case U32{aw} U32{bw}:
      %word_union(32n,aw,bw) : {U32{Word.and(32n,Word.or(32n,aw,bw),aw)} == U32{_} : U32}
      {==}

def zero(+a: U32) -> {U32.and(0,a) == 0 : U32}:
  match a:
    case U32{aw}:
      %word_zero(32n,aw) : {U32{Word.and(32n,Word.zero(32n),aw)} == U32{_} : U32}
      {==}

def word_self(+n: Nat,+a: Word(n)) -> {Word.and(n,a,a) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a:
        case WCon{False{},t}:
          %word_self(p,t) : {WCon{False{},Word.and(p,t,t)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t}:
          %word_self(p,t) : {WCon{True{},Word.and(p,t,t)} == WCon{True{},_} : Word(1n+p)}
          {==}
def self(+a: U32) -> {U32.and(a,a) == a : U32}:
  match a:
    case U32{w}:
      %word_self(32n,w) : {U32{Word.and(32n,w,w)} == U32{_} : U32}
      {==}
def bool_or_comm(+a: Bool,+b: Bool) -> {Bool.or(a,b) == Bool.or(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}
def word_or_comm(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.or(n,a,b) == Word.or(n,b,a) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          %bool_or_comm(ah,bh) : {WCon{Bool.or(ah,bh),Word.or(p,at,bt)} == WCon{_,Word.or(p,bt,at)} : Word(1n+p)}
          %word_or_comm(p,at,bt) : {WCon{Bool.or(ah,bh),Word.or(p,at,bt)} == WCon{Bool.or(ah,bh),_} : Word(1n+p)}
          {==}
def or_comm(+a: U32,+b: U32) -> {U32.or(a,b) == U32.or(b,a) : U32}:
  match a b:
    case U32{aw} U32{bw}:
      %word_or_comm(32n,aw,bw) : {U32{Word.or(32n,aw,bw)} == U32{_} : U32}
      {==}
def subset(-a: U32,-b: U32) -> Type:
  {U32.and(b,a) == b : U32}
def union_left(+a: U32,+b: U32) -> subset(U32.or(a,b),a):
  %comm(U32.or(a,b),a) : {_ == a : U32}
  union(a,b)
def union_right(+a: U32,+b: U32) -> subset(U32.or(a,b),b):
  %or_comm(b,a) : {U32.and(b,_) == b : U32}
  union_left(b,a)
def subset_zero(+a: U32,good: subset(0,a)) -> {a == 0 : U32}:
  Equal.trans(U32,a,U32.and(a,0),0,
    Equal.sym(U32,U32.and(a,0),a,good),
    Equal.trans(U32,U32.and(a,0),U32.and(0,a),0,comm(a,0),zero(a)))
def trans(+a: U32,+b: U32,+c: U32,ab: subset(a,b),+bc: {U32.and(c,b) == c : U32}) -> subset(a,c):
  %bc : {U32.and(_,a) == c : U32}
  %Equal.sym(U32,U32.and(U32.and(c,b),a),U32.and(c,U32.and(b,a)),assoc(c,b,a)) : {_ == c : U32}
  %Equal.sym(U32,U32.and(b,a),b,ab) : {U32.and(c,_) == c : U32}
  bc
def disjoint_subset(+a: U32,+lost: U32,+bit: U32,included: subset(lost,bit)) ->
  {U32.and(U32.and(a,U32.not(lost)),bit) == 0 : U32}:
  %included :
    {U32.and(U32.and(a,U32.not(lost)),_) == 0 : U32}
  %assoc(U32.and(a,U32.not(lost)),bit,lost) :
    {_ == 0 : U32}
  %comm(lost,U32.and(U32.and(a,U32.not(lost)),bit)) : {_ == 0 : U32}
  %assoc(lost,U32.and(a,U32.not(lost)),bit) : {_ == 0 : U32}
  %comm(U32.and(a,U32.not(lost)),lost) : {U32.and(_,bit) == 0 : U32}
  %Equal.sym(U32,U32.and(U32.and(a,U32.not(lost)),lost),0,disjoint(a,lost)) : {U32.and(_,bit) == 0 : U32}
  zero(bit)
