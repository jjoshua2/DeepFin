import Base

# PR 2 architecture probe:
#
# Bend owns the control flow and consumes only compact scalar results. The
# native C++ side owns tensor construction and AOTInductor/LibTorch execution.
# This keeps large tensor buffers out of Bend's term representation, matching
# the intended engine architecture.

type Summary is Data:
  Summary{count: U32, checksum: U32}

def NN.eval(seed: U32) -> IO(List<U32>):
  import "./aoti_call.c"

def summarize(
  xs: List<U32>,
  count: U32,
  checksum: U32
) -> Summary:
  match xs:
    case Nil{}:
      Summary{count, checksum}
    case Con{word, rest}:
      summarize(
        rest,
        U32.add(count, 1),
        U32.xor(U32.mul(checksum, 2654435761), word)
      )

def print_summary(seed: U32, summary: Summary) -> IO(Unit):
  match summary:
    case Summary{count, checksum}:
      IO.print(
        "seed=" ++ U32.show(seed)
        ++ " count=" ++ U32.show(count)
        ++ " checksum=" ++ U32.show(checksum)
      )

def report(seed: U32, words: List<U32>) -> IO(Unit):
  print_summary(seed, summarize(words, 0, 0))

def probe(+seed: U32) -> IO(Unit):
  do IO<Unit>:
    words : List<U32> <- NN.eval(seed)
    report(seed, words)

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- probe(0)
    b : Unit <- probe(1)
    c : Unit <- probe(7)
    return Unit{}
