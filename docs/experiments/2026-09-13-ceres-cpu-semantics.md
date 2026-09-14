# Ceres CPU encoding and value-getter semantics

Status: the CPU encoding/getter check completed after a comparator-only recovery. All 144 history encodings matched byte-for-byte. The small getter differences from the conventional mathematical blend are reported below; no neural or backend parity is established.

The question is whether the current Python input encoding and conventional value blend match the actual upstream CPU entrypoints. This can explain implementation differences before another Ceres mixture is considered; it does not establish neural or backend parity, or playing strength.

The harness targets exact upstream Ceres `64558176ad4933d3cd85b5604133d6576de921d2`. It reconstructs full FEN/UCI histories before upstream repetition calculation and truncation, then emits all 8,768 square bytes with Q-blunder inputs 0.03. Sixteen fixed fixtures cover sides, castling, promotion, en passant and repetition. A transport adds the existing 128 saved matched rows, verifies identities and exact FP16 raw logits, and feeds those bits through the public value-batch constructor. Twelve W/D/L/V getter outputs are exported as FP16 bits. Temperatures 0.55/1.5 and blend 0.4 are an explicitly named parameterless-options profile, not an assertion about effective deployed settings. Comparisons retain byte disagreements and value deltas without silently repairing them.

The authorized budget is **one hour inclusive**, CPU **10,11**, two numeric/.NET threads and no GPU visibility. Setup/restore and build each have a 1,200-second ceiling; transport/oracle/readout has 900 seconds, all under one shared deadline with cleanup margin. RAM headroom is 48 GiB at startup and 32 GiB while running. Disk requirements are 162 GiB at startup and 150 GiB while running, with an 8 GiB aggregate allocated-byte allowance; startup includes another 4 GiB for other writers. CoreCLR has no address-space cap. SDK 10.0.401, source, restore caches, temporary files and outputs stay under a fresh task directory. No global installation, evaluator construction, inference fallback or automatic extension is authorized.

The initial source allocation used CPU 2,3. Before launch, the parent selected CPU 10,11 to avoid the arena allocation; the affinity-only amendment and exact command are retained as implementation v2. Original v1 evidence remains intact. The executing checkout is preserved, and the README command link is corrected only in this separate publication checkout.

Static preparation passed Python AST, project XML, all sixteen fixture histories, Ruff and shell syntax. Independent review passed on the original source; parent review passed the literal affinity amendment. These checks did not compile the C# project or read the saved bank. At launch, dependency closure, build compatibility and runtime initialization remained actual-run questions; the completed observations follow below.

[Compact launch evidence](evidence/ceres-cpu-semantics-launched-20260913.json) binds the plan, command, reviews and immutable snapshots of the small actual start receipts. The implementation is in [tools/ceres_cpu_oracle](../../tools/ceres_cpu_oracle/README.md).


## Completed CPU readout and comparator recovery

SDK/source restore and the C# build completed successfully in 34.1051 and 16.046 seconds respectively. Export and the native CPU oracle also produced their saved outputs. The original host attempt exited 1 after 51.2149 seconds because the Python comparator incorrectly demanded float64 equality from JSON values emitted as float32. For example, getter bits 13653 decode exactly to 0.333251953125, while the C# JSON representation is 0.33325195; both are exactly equal at the producing float32 precision.

The narrow correction checks exact float32 equality, with no numerical tolerance, and derives downstream deltas from the authoritative FP16 bits. Four regression cases include the actual decimal example, rejection of a one-float32-ULP error, and wrong bits/shape. The original runtime, failure receipts and native outputs are preserved. The parent reran only comparison, which exited 0 in **0.283844 seconds**, before the original one-hour deadline. Setup, build, export and the oracle were not repeated.

The completed CPU evidence covers **144 histories** (16 explicit fixtures plus 128 saved rows) and **131 value cases** (three synthetic plus 128 saved raw-logit pairs). There were **zero byte mismatches** across 272 comparisons: 144 against the current Python encoder and 128 against stored bank bytes. Maximum absolute differences from the conventional float64 mathematical mixture were:

| Cases | WDL | Actual FP16 GetV | Decoded W minus L |
| --- | --- | --- | --- |
| Saved bank, 128 | 0.0012483831 | 0.0012965670 | 0.0008723146 |
| Synthetic, 3 | 0.0007860234 | 0.0007973128 | 0.0005226546 |

The FP16 GetV getter and subtraction of decoded W/L are distinct quantities. These compare actual upstream CPU getters under the explicitly named parameterless profile with the mathematical blend; they do not prove equivalent neural inference, all-history universal parity, or effective deployed configuration.

[Completed CPU evidence](evidence/ceres-cpu-semantics-completed-20260913.json) retains the failed original attempt, exact recovery and independent review. This resolves the tested encoder and getter question at the saved inputs. No teacher replacement, corpus rewrite, or training was performed in this check. It makes no playing-strength claim.
