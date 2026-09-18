import Base

# GPU parity probe.
#
# Bulk board tensors and neural outputs stay native. Bend receives only compact
# scalar summaries of the dense search policy + WDL result.

def NN.run() -> IO(List<U32>):
  import "./cuda_call.c"

def report(xs: List<U32>) -> IO(Unit):
  match xs:
    case Con{batch,
      Con{raw_width,
      Con{policy_count,
      Con{policy_xor,
      Con{policy_mix,
      Con{wdl_count,
      Con{wdl_xor,
      Con{wdl_mix, Nil{}}}}}}}}}:
      IO.print(
        "batch=" ++ U32.show(batch)
        ++ " raw_policy_width=" ++ U32.show(raw_width)
        ++ " policy_count=" ++ U32.show(policy_count)
        ++ " policy_xor=" ++ U32.show(policy_xor)
        ++ " policy_mix=" ++ U32.show(policy_mix)
        ++ " wdl_count=" ++ U32.show(wdl_count)
        ++ " wdl_xor=" ++ U32.show(wdl_xor)
        ++ " wdl_mix=" ++ U32.show(wdl_mix)
      )
    case other:
      IO.print("cuda_parity_error=bad_native_summary")

def main() -> IO(Unit):
  do IO<Unit>:
    summary : List<U32> <- NN.run()
    report(summary)
